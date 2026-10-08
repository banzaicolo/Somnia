#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=============================================================================
Full calibration pipeline demo — see how a complete calibration chain cleans the dirty data
=============================================================================

[What does this script do?]

The Zero Calibration Demo only showed you one move (subtraction). This file chains
**all the moves together**: crosstalk correction + temp drift compensation + zero
calibration + gain calibration, step by step washing the dirty data back to the truth.

It also adds a "plot": calibration was done at 25 degrees, but at runtime the
temperature soars to 40 degrees and the sensor drifts. You'll see how much
difference "not compensating temperature" vs "compensating temperature" makes.

[How to run]

    python3 src/calibration_demo.py

It prints the error numbers at each step and generates one comparison figure in outputs/:
  calibration_full.png

[What to focus on]

After running, look at the "error shrank by X%" in the terminal, and the error map in
the bottom-right of the last figure — before calibration it's all red (false reports
everywhere), after full calibration it's nearly all white (close to the truth).
=============================================================================
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["PingFang SC", "Arial Unicode MS", "Heiti SC"]
plt.rcParams["axes.unicode_minus"] = False

from sensor_config import (
    GRID_W, GRID_H, BODY_PARTS, OUT_DIR, SEED,
    make_ideal_pressure,
)
import calibration as cal
from verdict import describe_improvement


# ---- parameters of the calibration scenario (all "real-world" numbers) ----
T_REF = 25.0      # temperature during calibration (Celsius)
T_RUN = 40.0      # temperature at runtime (drifted by 15 degrees)
CROSSTALK = 0.10  # crosstalk coefficient
NOISE_STD = 3.0   # noise strength
P0 = 50.0         # "standard pressure" used for calibration (equivalent to a known-mass weight)


def build_sensor_world(rng):
    """
    Construct a "virtual sensor" world: each point's sensitivity, zero offset, and
    temperature coefficients are random but fixed (these are a real sensor's
    "factory parameters").

    Returns a dict holding all these parameters.
    """
    # per-point sensitivity (ground truth, floats ±15% around 1)
    gain_ref = rng.normal(1.0, 0.15, size=(GRID_H, GRID_W))

    # per-point baseline zero offset (caused by aging + manufacturing, 0~8)
    offset_ref = rng.uniform(0, 8, size=(GRID_H, GRID_W))

    # temperature coefficients: per +1 degree, how much the zero drifts (TCO) and how much the sensitivity ratio changes (TCS)
    tco_map = rng.uniform(0.3, 0.8, size=(GRID_H, GRID_W))   # drifts 0.3~0.8 per degree
    tcs_map = rng.uniform(0.001, 0.004, size=(GRID_H, GRID_W))  # changes 0.1%~0.4% per degree

    return {
        "gain_ref": gain_ref,
        "offset_ref": offset_ref,
        "tco_map": tco_map,
        "tcs_map": tcs_map,
    }


def offset_at(world, temp):
    """Zero offset at a given temperature = baseline offset + temp drift."""
    return world["offset_ref"] + world["tco_map"] * (temp - T_REF)


def gain_at(world, temp):
    """Sensitivity at a given temperature = baseline sensitivity × (1 + temp drift gain)."""
    return world["gain_ref"] * (1.0 + world["tcs_map"] * (temp - T_REF))


def main():
    print("=" * 64)
    print(" Full sensor calibration pipeline demo (crosstalk + temp drift + zero + gain, four moves in a row)")
    print("=" * 64)
    print()

    rng = np.random.default_rng(SEED)
    world = build_sensor_world(rng)

    # ---- 1. true body pressure (ground truth) ----
    ideal = make_ideal_pressure(GRID_W, GRID_H, BODY_PARTS)
    print("[1/5] ground truth generated (what a person actually pressing looks like)")

    # ---- 2. dirty data the sensor reads at runtime (temperature 40 degrees) ----
    offset_run = offset_at(world, T_RUN)
    gain_run = gain_at(world, T_RUN)
    noise = rng.normal(0, NOISE_STD, size=(GRID_H, GRID_W))
    raw = cal.forward(ideal, gain_run, offset_run, CROSSTALK, noise)
    print(f"[2/5] runtime temperature {T_RUN:.0f}°C, dirty data generated (with crosstalk / temp drift / noise)")
    err_raw = cal.mean_abs_error(raw, ideal)
    print(f"      dirty data mean error = {err_raw:.1f}")

    # ---- 3. simulate the "calibration actions" (measure parameters at 25 degrees) ----
    # Real calibration takes many frames and averages them to push noise nearly to zero.
    # So here the calibration data uses a "no-noise" version (the ideal result after
    # enough frame averaging), to clearly show the calibration chain's extreme power
    # against "systematic error".
    # 3a. measure the baseline at no-load
    baseline = cal.estimate_baseline(
        offset_at(world, T_REF), c=CROSSTALK, noise_std=0.0, rng=rng)
    # 3b. measure gain with the standard pressure P0
    ideal_ref = np.full((GRID_H, GRID_W), P0)   # a uniform slab of standard pressure
    raw_ref = cal.forward(ideal_ref, gain_at(world, T_REF),
                          offset_at(world, T_REF), CROSSTALK, 0.0)
    gain_est = cal.estimate_gain(ideal_ref, raw_ref, baseline, c=CROSSTALK)
    print("[3/5] calibration actions done: no-load measured the baseline, standard pressure measured each point's sensitivity")

    # ---- 4. run the full calibration pipeline ----
    cleaned = cal.calibrate(
        raw,
        baseline=baseline,
        gain_map=gain_est,
        c=CROSSTALK,
        temp=T_RUN, t_ref=T_REF,
        tco_map=world["tco_map"], tcs_map=world["tcs_map"],
    )
    err_clean = cal.mean_abs_error(cleaned, ideal)
    print(f"[4/5] after full calibration, mean error = {err_clean:.1f}")
    print(f"      error shrank by {cal.improvement(err_raw, err_clean):.0f}%")

    # ---- 4b. comparison: what if we "skip temperature compensation" ----
    cleaned_no_temp = cal.calibrate(
        raw, baseline=baseline, gain_map=gain_est, c=CROSSTALK)
    err_no_temp = cal.mean_abs_error(cleaned_no_temp, ideal)
    print(f"      (control) without temperature compensation, error remains {err_no_temp:.1f}, "
          f"only shrank {cal.improvement(err_raw, err_no_temp):.0f}%")

    # ---- 4c. key point: distinguish "systematic error" from "random noise" ----
    # The error remaining after the calibration above is mostly random noise (old TV
    # static); noise differs every time, so calibration can't remove it — that needs
    # "filtering". Turn the noise off and look again, and you'll see the calibration
    # chain's real power against "systematic error" (gain / offset / crosstalk / temp drift):
    raw_sys = cal.forward(ideal, gain_run, offset_run, CROSSTALK, 0.0)
    cleaned_sys = cal.calibrate(
        raw_sys, baseline=baseline, gain_map=gain_est, c=CROSSTALK,
        temp=T_RUN, t_ref=T_REF,
        tco_map=world["tco_map"], tcs_map=world["tcs_map"])
    err_raw_sys = cal.mean_abs_error(raw_sys, ideal)
    err_clean_sys = cal.mean_abs_error(cleaned_sys, ideal)
    print(f"      (no-noise control) pure systematic error {err_raw_sys:.1f} -> {err_clean_sys:.1f}, "
          f"shrank {cal.improvement(err_raw_sys, err_clean_sys):.0f}%")

    # ---- 5. plot ----
    plot(ideal, raw, cleaned_no_temp, cleaned, err_raw, err_no_temp, err_clean)
    print("[5/5] comparison figure generated; open calibration_full.png in outputs/")
    print()
    # The ending no longer hard-codes "all red ... nearly all white"; it's decided dynamically by the error numbers.
    improve_full = cal.improvement(err_raw, err_clean)
    print(f" ✅ Done. Mean error after full calibration {err_clean:.1f}, "
          f"error shrank {improve_full:.0f}%: {describe_improvement(improve_full)}")
    print("    (bottom-right error map: red = reads high, blue = low, white = accurate; compare yourself)")


def plot(ideal, raw, no_temp, cleaned, err_raw, err_no_temp, err_clean):
    """Draw a 2×2 comparison figure with a unified colorbar; bottom-right is the error map."""
    os.makedirs(OUT_DIR, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(16, 13))

    vmax = max(ideal.max(), raw.max())

    # top-left: ground truth
    im0 = axes[0, 0].imshow(ideal, cmap="hot", vmin=0, vmax=vmax, interpolation="bicubic")
    axes[0, 0].set_title(f"Ground truth", fontsize=12)
    fig.colorbar(im0, ax=axes[0, 0], fraction=0.046)

    # top-right: dirty data
    im1 = axes[0, 1].imshow(raw, cmap="hot", vmin=0, vmax=vmax, interpolation="bicubic")
    axes[0, 1].set_title(f"Dirty data (err {err_raw:.0f})", fontsize=12)
    fig.colorbar(im1, ax=axes[0, 1], fraction=0.046)

    # bottom-left: calibration without temperature compensation
    im2 = axes[1, 0].imshow(no_temp, cmap="hot", vmin=0, vmax=vmax, interpolation="bicubic")
    axes[1, 0].set_title(f"No temp compensation (err {err_no_temp:.0f})", fontsize=12)
    fig.colorbar(im2, ax=axes[1, 0], fraction=0.046)

    # bottom-right: error map (cleaned - truth), red=high blue=low white=correct
    err_map = cleaned - ideal
    lim = max(abs(err_map.min()), abs(err_map.max()), 1.0)
    im3 = axes[1, 1].imshow(err_map, cmap="seismic", vmin=-lim, vmax=lim, interpolation="bicubic")
    axes[1, 1].set_title(f"Error after full calibration (err {err_clean:.1f})", fontsize=12)
    fig.colorbar(im3, ax=axes[1, 1], fraction=0.046)

    for ax in axes.flat:
        ax.set_xticks([])
        ax.set_yticks([])

    fig.suptitle("Full calibration chain: four fixes that recover the truth", fontsize=15)
    plt.tight_layout()
    path = os.path.join(OUT_DIR, "calibration_full.png")
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return path


if __name__ == "__main__":
    main()
