#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=============================================================================
Simulated pressure sensor data generator (teaching demo, runnable directly)
=============================================================================

[What is this thing?]

The core of your AI medical bed is the row of flexible pressure sensors under
the mattress. You don't have real hardware yet, but "no hardware" doesn't stop
you from developing the algorithm — because a sensor's output follows
[physical laws]. When a person lies down, the places under heavy pressure
(shoulders, hips) read higher; we can "act out" these laws entirely with math.

So this script does one thing:

    Use math to "act out" a body-pressure map of "a person lying on the
    mattress", then "add in" all the faults a real sensor has, so you can see
    the difference between the ideal world and the world a sensor actually reads.

[Why is this step so important?]

I told you before: the easiest place for this project to fail is not the AI
model — it's [sensor calibration], because sensors lie. This script lets you
[see with your own eyes] how they lie. Only when you see clearly what the enemy
looks like can you write the right calibration algorithm.

[What does this file produce?]

After running, it generates these in the "outputs" folder:

  1. imperfections.png    -- 4 comparison plots, showing what each fault adds
  2. hysteresis_curve.png -- a "load-unload" loop, showing the sensor "holding a grudge"
  3. ideal_pressure.csv   -- what a person actually presses (the "ground truth")
  4. sensor_readings.csv  -- what the sensor actually reads (the "dirty data")

[How to run]

Open "Terminal" on Mac, paste and press Enter:

    python3 src/pressure_simulator.py

(requires numpy and matplotlib; the "dependencies" section below tells you whether they're installed and how to install them)
=============================================================================
"""

import os
import numpy as np
import matplotlib
# Don't pop up a window; save figures directly to files (important when running in the Mac terminal)
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Chinese font support (font available on Mac)
plt.rcParams["font.sans-serif"] = ["PingFang SC", "Arial Unicode MS", "Heiti SC"]
plt.rcParams["axes.unicode_minus"] = False   # make the minus sign display correctly


# ============================================================================
# Part 1: parameter section (all "knobs" live in src/sensor_config.py)
# ============================================================================
# To change grid size, body parts, or sensor fault strength, edit the single file
# sensor_config.py. Importing it here means this script and others share the same
# parameters instead of each editing its own copy.
from sensor_config import (
    GRID_W, GRID_H, BODY_PARTS,
    SENS_GAIN_STD, TEMP_DRIFT, ZERO_DRIFT_MAX, NOISE_STD, CROSSTALK, HYSTERESIS,
    OUT_DIR, SEED,
    FILE_IMPERFECTIONS, FILE_HYSTERESIS, FILE_IDEAL_CSV, FILE_RAW_CSV,
    make_ideal_pressure,
)


# ============================================================================
# Part 2: core functions
# ============================================================================
# (make_ideal_pressure moved to sensor_config.py; see the import above)


def add_imperfections(ideal):
    """
    Turn the "ideal body-pressure map" into "what a sensor actually reads", by adding faults layer by layer.

    This function returns a dict that also holds "what each step looks like after
    adding", so we can draw comparison plots and see what each fault layer did.

    Args: ideal -- the ideal body-pressure map
    Returns: a dict whose keys are step names and values are the maps at each step
    """
    steps = {"Ideal pressure": ideal}   # step 0: ground truth

    # ---- fault ① batch mismatch ----
    # Multiply each point by a "sensitivity coefficient" that randomly floats around 1.
    # A fixed random seed keeps every run identical so you can compare.
    rng = np.random.default_rng(SEED)
    sensitivity = rng.normal(1.0, SENS_GAIN_STD, size=ideal.shape)
    after_gain = ideal * sensitivity
    steps["+ batch mismatch (sensitivity ±15%)"] = after_gain

    # ---- fault ② temp drift ----
    # Lift the whole baseline. In the real world "higher temperature lifts more, lower lifts less";
    # here we simplify: lift everything by TEMP_DRIFT, plus a little spatial non-uniformity.
    temp_map = TEMP_DRIFT * rng.uniform(0.7, 1.3, size=ideal.shape)
    after_temp = after_gain + temp_map
    steps["+ temp drift (baseline lifted)"] = after_temp

    # ---- fault ③ zero drift ----
    # Add a random 0~ZERO_DRIFT_MAX "long-term offset" to each point.
    # It's temperature-independent, caused by "aging", and differs per point.
    zero_map = rng.uniform(0, ZERO_DRIFT_MAX, size=ideal.shape)
    after_zero = after_temp + zero_map
    steps["+ zero drift (per-point random offset)"] = after_zero

    # ---- fault ④ crosstalk ----
    # Each point "leaks" some signal to its four neighbors (up/down/left/right).
    # Recipe: new reading = (1-c) × self  +  c × average of the four neighbors.
    # c is CROSSTALK; the larger it is, the worse the leakage.
    if CROSSTALK > 0:
        c = CROSSTALK
        # np.pad "pads a ring" around the map so edge points can also read neighbors
        padded = np.pad(after_zero, 1, mode="edge")
        # average of the four neighbors (up/down/left/right)
        neighbors = (padded[:-2, 1:-1] + padded[2:, 1:-1] +
                     padded[1:-1, :-2] + padded[1:-1, 2:]) / 4.0
        after_cross = (1 - c) * after_zero + c * neighbors
    else:
        after_cross = after_zero
    steps["+ crosstalk (neighbor leakage 10%)"] = after_cross

    # ---- fault ⑤ noise ----
    # Finally add a layer of random noise.
    after_noise = after_cross + rng.normal(0, NOISE_STD, size=ideal.shape)
    steps["final reading (+ noise)"] = after_noise

    return steps


def hysteresis_curve():
    """
    Generate a "hysteresis loop" (for figure 2).

    Hysteresis means: if you press the sensor from 0 to 100 then release from 100
    back to 0, the "loading" and "unloading" paths don't match — plotting them
    gives a loop, not a single line. The fatter the loop, the more the sensor
    "holds a grudge".

    Here we use a simplified model (constant width equal to HYSTERESIS):
      - loading:   reading = true pressure + HYSTERESIS/2
      - unloading: reading = true pressure - HYSTERESIS/2
    Real hysteresis is far more complex, but this loop is enough to develop the calibration algorithm.
    """
    p = np.linspace(0, 100, 50)   # true pressure from 0 to 100, 50 points

    # loading (0 to 100): reads high by half the width
    load = p + HYSTERESIS / 2.0

    # unloading (100 to 0): reads low by half the width
    unload = p - HYSTERESIS / 2.0

    return p, load, unload


def simulate_load_unload(ideal, n_steps=60, h=HYSTERESIS, noise=0.0, seed=None):
    """
    Simulate the full process of "a person slowly lying down, then slowly getting up", with hysteresis on top.

    Why a separate function? Because hysteresis is "dynamic" — it depends on whether
    the pressure is rising or falling, so it can't be added to a single static
    pressure map (add_imperfections handles a single static map). So we simulate a
    process:

        lie down (loading: pressure rises from 0 to max) -> get up (unloading: pressure falls from max to 0)

        Each frame adds the hysteresis offset according to "loading or unloading right now":

        loading segment:   reading = p + h/2    (high)
        unloading segment: reading = p - h/2    (low)

    Args:
      ideal    -- ideal body-pressure map (ground truth with the person fully lying down), (H, W)
      n_steps  -- total frame count (loading and unloading roughly half each)
      h        -- hysteresis width (same number as HYSTERESIS)
      noise    -- random noise added per frame (default 0, so you see pure hysteresis first)
      seed     -- random seed; None uses SEED from sensor_config

    Returns:
      true_frames -- (n_frames, H, W) true pressure per frame (ground truth)
      raw_frames  -- (n_frames, H, W) sensor reading per frame (with hysteresis)
      weights     -- (n_frames,) "pressure coefficient" 0~1 per frame, rising then falling
    """
    rng = np.random.default_rng(seed if seed is not None else SEED)

    # loading 0→1, unloading 1→0 (count the peak only once; don't include w=1 in both segments)
    n_up = n_steps // 2
    n_down = n_steps - n_up
    up = np.linspace(0.0, 1.0, n_up)
    down = np.linspace(1.0, 0.0, n_down + 1)[1:]   # drop the duplicated peak
    weights = np.concatenate([up, down])

    true_frames = np.array([ideal * w for w in weights])
    raw_frames = np.empty_like(true_frames)

    for i, w in enumerate(weights):
        p = true_frames[i]
        if i < n_up:
            raw = p + h / 2.0     # loading: reads high
        else:
            raw = p - h / 2.0     # unloading: reads low
        if noise > 0:
            raw = raw + rng.normal(0, noise, size=p.shape)
        raw_frames[i] = raw

    return true_frames, raw_frames, weights


# ============================================================================
# Part 3: plot + save
# ============================================================================


def plot_and_save(steps):
    """
    Plot each step and stitch them into one 4-panel comparison figure, saved as png.
    """
    out_dir = OUT_DIR
    os.makedirs(out_dir, exist_ok=True)

    # pick 4 key steps to show: ideal, batch mismatch, temp drift + zero drift, final
    keys = list(steps.keys())
    # pick 4 representative ones (1st, 2nd, 4th, last)
    chosen = [keys[0], keys[1], keys[3], keys[-1]]

    fig, axes = plt.subplots(1, 4, figsize=(18, 5))

    for ax, k in zip(axes, chosen):
        im = ax.imshow(steps[k], cmap="hot", interpolation="bicubic")
        ax.set_title(k, fontsize=11)
        ax.set_xticks([])   # hide tick marks for a cleaner plot
        ax.set_yticks([])
        fig.colorbar(im, ax=ax, fraction=0.046)

    fig.suptitle("Ideal pressure -> raw sensor reading (brighter = higher pressure)", fontsize=14)
    plt.tight_layout()
    path1 = os.path.join(out_dir, FILE_IMPERFECTIONS)
    plt.savefig(path1, dpi=300, bbox_inches="tight")
    plt.close(fig)

    return path1


def plot_hysteresis():
    """Plot the hysteresis loop (figure 2)."""
    out_dir = OUT_DIR
    os.makedirs(out_dir, exist_ok=True)

    p, load, unload = hysteresis_curve()

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(p, load,   color="tab:red",  linewidth=2, label="increasing")
    ax.plot(p, unload, color="tab:blue", linewidth=2, label="decreasing")

    # ideal 45-degree line: an honest sensor would read exactly the pressure
    ax.plot([0, 100], [0, 100], "--", color="gray", label="ideal")

    ax.set_xlabel("true pressure")
    ax.set_ylabel("sensor reading")
    ax.set_title("Hysteresis: same pressure, different reading on the way up vs down")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path2 = os.path.join(out_dir, FILE_HYSTERESIS)
    plt.savefig(path2, dpi=300, bbox_inches="tight")
    plt.close(fig)

    return path2


def save_csv(steps):
    """Save the ideal map and final reading as CSV, ready to feed to the algorithm later."""
    out_dir = OUT_DIR
    os.makedirs(out_dir, exist_ok=True)

    keys = list(steps.keys())
    ideal = steps[keys[0]]
    final = steps[keys[-1]]

    np.savetxt(os.path.join(out_dir, FILE_IDEAL_CSV), ideal,
               fmt="%.2f", delimiter=",")
    np.savetxt(os.path.join(out_dir, FILE_RAW_CSV), final,
               fmt="%.2f", delimiter=",")


# ============================================================================
# Part 4: main program
# ============================================================================


def main():
    print("=" * 62)
    print(" Simulated pressure sensor data generator")
    print(" Purpose: with no real hardware, 'act out' sensor data to develop the calibration algorithm")
    print("=" * 62)
    print()

    # ---- 1. generate the ideal body-pressure map ----
    ideal = make_ideal_pressure(GRID_W, GRID_H, BODY_PARTS)
    print(f"[1/4] Generated ideal body-pressure map ({GRID_W}×{GRID_H} = {GRID_W*GRID_H} sensing points)")

    # ---- 2. add faults layer by layer ----
    steps = add_imperfections(ideal)
    print("[2/4] Added 5 sensor faults (batch mismatch / temp drift / zero drift / crosstalk / noise)")

    # ---- 3. plot + save ----
    path1 = plot_and_save(steps)
    path2 = plot_hysteresis()
    save_csv(steps)
    print("[3/4] Generated 2 figures + 2 CSV files")

    # ---- 4. print the key "ground truth vs dirty data" numbers for an intuitive feel ----
    final = steps[list(steps.keys())[-1]]
    print()
    print("[4/4] Key comparison (same point, ideal vs sensor reading):")
    print(f"      ideal peak        : {ideal.max():>6.1f}")
    print(f"      sensor peak       : {final.max():>6.1f}")
    print(f"      difference        : {final.max() - ideal.max():>6.1f}")
    print()
    print(" Output files are all in the 'outputs' folder:")
    print(f"   · {path1}")
    print(f"   · {path2}")
    print(f"   · {os.path.join(OUT_DIR, FILE_IDEAL_CSV)} (ground truth)")
    print(f"   · {os.path.join(OUT_DIR, FILE_RAW_CSV)} (dirty data, input to the calibration algorithm)")
    print()
    print(" ✅ Done. Open those two PNG figures and you'll see why the sensor 'lies'.")


# standard Python idiom: run only when executed directly, not when imported
if __name__ == "__main__":
    main()
