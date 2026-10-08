#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=============================================================================
Zero calibration demo (teaching version, runnable directly)
=============================================================================

[What does this script prove?]

It answers your last question: "How do you know how to fix it?"

The first move for fixing a sensor is called [zero calibration], and it's really
just one sentence:

    "At the moment the person gets off the bed, record the sensor reading,
     then subtract it from every reading afterwards."

The principle is dead simple: when nobody presses the sensor, the reading should
be 0. But because of temp drift, zero drift and the like, it reads a fake 15~30.
Those fake numbers are "added on top of" the true signal — so just subtract them
away, and it's clean.

This script simulates the whole process so you can see with your own eyes how
different "before" and "after" the fix are.

[The full flow (exactly like a real product)]

  Moment ① no-load (the person got off the bed)
        -> sensor reading = 0 + zero offset + noise
        -> we record this frame and call it the "baseline"

  Moment ② person on (someone lies down)
        -> sensor reading = true body pressure × sensitivity + zero offset + noise
        -> this is the "dirty data", the raw input for the AI

  Calibration action (a single subtraction)
        -> calibrated = dirty data - baseline

  Key point: the zero offset is [exactly the same both times] (it's the sensor's
  own fault, doesn't change with the person), so subtracting cancels it. All
  that's left is noise and sensitivity error.

[How to run]

    python3 src/zero_calibration.py

After running it generates "zero_calibration.png" and prints the "before/after" error numbers.
=============================================================================
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")   # don't pop up a window; save figures directly to files
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["PingFang SC", "Arial Unicode MS", "Heiti SC"]
plt.rcParams["axes.unicode_minus"] = False


# ============================================================================
# Parameter section (taken from sensor_config.py; this script only overrides "temp drift")
# ============================================================================
from sensor_config import (
    GRID_W, GRID_H, BODY_PARTS,
    SENS_GAIN_STD, ZERO_DRIFT_MAX, NOISE_STD,
    OUT_DIR, SEED, FILE_ZERO_CAL,
    make_ideal_pressure,
)
from verdict import describe_improvement

# Turn temp drift up separately: make it "dirtier" so the difference is obvious at a glance.
# (A real product isn't this extreme; this is just for teaching. Changing it doesn't affect other scripts.)
TEMP_DRIFT = 30


# ============================================================================
# Core functions
# ============================================================================
# (make_ideal_pressure moved to sensor_config.py; see the import above)


def simulate_calibration(seed=SEED):
    """
    Run a full "zero calibration" simulation and return the computed results.

    [Why factor this out of main()?]
    Because the tests need to verify "does calibration work". If the test re-wrote
    the same computation itself, the test and the script would be two copies of the
    code; change one and forget the other, and the test lies. Factoring it into a
    function that both main() and the test call guarantees the test exercises the
    exact logic that actually runs.

    Returns a dict with these keys and meanings:
      ideal        ground truth (what a person actually pressing should look like)
      raw          dirty data (before calibration, what the sensor directly reads)
      calibrated   after calibration (baseline subtracted)
      baseline     the no-load baseline (that "free fake number")
      err_before   mean error vs ground truth before calibration
      err_after    mean error vs ground truth after calibration
    """
    rng = np.random.default_rng(seed)   # fixed random seed so results are reproducible

    # ---- 1. the sensor's "own faults" (don't change with the person, fixed) ----
    sensitivity = rng.normal(1.0, SENS_GAIN_STD, size=(GRID_H, GRID_W))  # non-uniform sensitivity
    temp_map = TEMP_DRIFT * rng.uniform(0.7, 1.3, size=(GRID_H, GRID_W))  # temp drift
    zero_map = rng.uniform(0, ZERO_DRIFT_MAX, size=(GRID_H, GRID_W))      # zero drift
    offset = temp_map + zero_map                                          # together called "zero offset"

    # ---- 2. moment ① no-load: baseline = offset + noise ----
    baseline = offset + rng.normal(0, NOISE_STD, size=(GRID_H, GRID_W))

    # ---- 3. moment ② person on: dirty data = true pressure × sensitivity + offset + noise ----
    ideal = make_ideal_pressure(GRID_W, GRID_H, BODY_PARTS)
    raw = ideal * sensitivity + offset + rng.normal(0, NOISE_STD, size=(GRID_H, GRID_W))

    # ---- 4. calibration action: a single subtraction ----
    calibrated = raw - baseline

    # ---- 5. mean absolute error (how far each point is from the correct answer, averaged) ----
    err_before = np.abs(raw - ideal).mean()
    err_after = np.abs(calibrated - ideal).mean()

    return {
        "ideal": ideal,
        "raw": raw,
        "calibrated": calibrated,
        "baseline": baseline,
        "err_before": err_before,
        "err_after": err_after,
    }


def main():
    print("=" * 62)
    print(" Zero calibration demo: does zeroing when the person gets off clean up the dirty data?")
    print("=" * 62)
    print()

    # call the factored-out core function (the test calls the same one, so it exercises the real logic)
    r = simulate_calibration(seed=42)
    ideal = r["ideal"]
    raw = r["raw"]
    calibrated = r["calibrated"]
    baseline = r["baseline"]
    err_before = r["err_before"]
    err_after = r["err_after"]
    improve = (err_before - err_after) / err_before * 100

    print(f"[no-load] nobody lying down, yet the reading reaches {baseline.max():.1f} (should be 0)")
    print(f"         this {baseline.max():.1f} is the sensor's 'free fake number'; record it as the baseline")
    print()
    print(f"[person on] dirty data generated, mean error vs ground truth = {err_before:.1f}")
    print(f"[calibrate] after subtracting baseline, mean error vs ground truth = {err_after:.1f}")
    print(f"       ✅ error shrank by {improve:.0f}%")
    print()

    # ---- 6. draw the comparison ----
    # Key lesson: earlier the three maps each scaled their own colorbar, so the dirty
    # data was lifted overall but the difference was invisible. Now it uses a [unified
    # colorbar] + [error map], so the difference is obvious at a glance.
    out_dir = OUT_DIR
    os.makedirs(out_dir, exist_ok=True)

    # main maps share a 0~200 colorbar, so "lifted overall" directly shows as "brighter"
    VMAX = 200

    # colorbar range for the error maps (symmetric, red = too high, blue = too low)
    EMAX = max(np.abs(raw - ideal).max(), np.abs(calibrated - ideal).max())

    fig, axes = plt.subplots(2, 3, figsize=(20, 13))

    # ---- top row: three main maps (unified colorbar) ----
    for ax, data, title in zip(
        axes[0],
        [ideal, raw, calibrated],
        ["Ground truth\n(what the person actually presses)",
         "Dirty data\n(before calibration)",
         "Calibrated\n(baseline subtracted)"],
    ):
        im = ax.imshow(data, cmap="hot", interpolation="bicubic",
                       vmin=0, vmax=VMAX)
        ax.set_title(title, fontsize=15)
        ax.set_xticks([])
        ax.set_yticks([])
        fig.colorbar(im, ax=ax, fraction=0.046, label="pressure")

    # ---- bottom row: two error maps (how far from the correct answer — this is the point) ----
    err_dirty = raw - ideal
    err_cal = calibrated - ideal

    im1 = axes[1, 0].imshow(err_dirty, cmap="RdBu_r", interpolation="bicubic",
                            vmin=-EMAX, vmax=EMAX)
    axes[1, 0].set_title("Error of dirty data\n(red = reads high)", fontsize=15)
    axes[1, 0].set_xticks([])
    axes[1, 0].set_yticks([])
    fig.colorbar(im1, ax=axes[1, 0], fraction=0.046, label="error")

    im2 = axes[1, 2].imshow(err_cal, cmap="RdBu_r", interpolation="bicubic",
                            vmin=-EMAX, vmax=EMAX)
    # The title no longer hard-codes a subjective phrase like "almost all white = fixed";
    # it just shows the number. White or not, clean or not — judge by the numbers and
    # the figure with your own eyes; the program doesn't conclude for you.
    axes[1, 2].set_title(f"Error after calibration\n(mean error {err_after:.1f})", fontsize=15)
    axes[1, 2].set_xticks([])
    axes[1, 2].set_yticks([])
    fig.colorbar(im2, ax=axes[1, 2], fraction=0.046, label="error")

    # write the explanation in the middle cell
    axes[1, 1].axis("off")
    axes[1, 1].text(
        0.5, 0.5,
        "How to read the error map:\n\n"
        "white = 0 = exactly correct\n"
        "red   = reads high\n"
        "blue  = reads low\n\n"
        f"mean error before calibration {err_before:.1f}\n"
        f"mean error after calibration {err_after:.1f}\n"
        f"shrank by {improve:.0f}%",
        ha="center", va="center", fontsize=16,
    )

    fig.suptitle("Zero calibration: under a unified colorbar the dirty data clearly 'turns red and washed-out', and calibration makes it clean again",
                 fontsize=17)
    plt.tight_layout()
    path = os.path.join(out_dir, FILE_ZERO_CAL)
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    print(f" Output figure: {path}")
    print()
    # The ending no longer hard-codes "doesn't it look a lot like the left one";
    # it's decided dynamically by the improve number, and the program reports honestly.
    print(f" ✅ Done. Error shrank by {improve:.0f}%: {describe_improvement(improve)}")
    print("    (figure: far left is the truth, middle is dirty data, far right is calibrated — compare yourself)")


if __name__ == "__main__":
    main()
