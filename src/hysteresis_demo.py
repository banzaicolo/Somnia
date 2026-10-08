#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=============================================================================
Hysteresis compensation demo — automatically fix the "sensor holding a grudge" fault
=============================================================================

[What does this script demonstrate?]

Hysteresis is the most "sneaky" sensor fault: at the same pressure, pressing up
(loading) reads high, releasing (unloading) reads low. Plotted it's a loop, not a
line (you saw that loop in the Pressure Simulator's hysteresis_curve.png).

The hard part of fixing it isn't the formula (it's simple) — it's that **the program
must first know "am I loading or unloading right now"**, and the sensor won't tell
you the direction.

So this script wires up a full closed loop demonstrating "auto-detect direction + compensate":

    ① build an ideal body-pressure map (what a lying person looks like = ground truth)
    ② simulate "person slowly lying down (loading) -> slowly getting up (unloading)", dozens of frames,
       each frame adds hysteresis according to the direction at that moment, producing the "dirty readings"
    ③ the program compares adjacent frames, judges the direction, then applies the compensation formula
    ④ compare: how far off before vs after compensation

The conclusion (like the other demos) isn't hard-coded; it's computed from the
numbers and generated dynamically via the verdict module.

[How to run]

    python3 src/hysteresis_demo.py

After running it generates hysteresis_compensation.png in "outputs".
=============================================================================
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")   # don't pop up a window; save figures directly
import matplotlib.pyplot as plt

# Chinese font support
plt.rcParams["font.sans-serif"] = ["PingFang SC", "Arial Unicode MS", "Heiti SC"]
plt.rcParams["axes.unicode_minus"] = False

from sensor_config import (
    GRID_W, GRID_H, BODY_PARTS, HYSTERESIS, OUT_DIR,
    FILE_HYSTERESIS_COMP,
    make_ideal_pressure,
)
from pressure_simulator import simulate_load_unload
import calibration as cal
from verdict import describe_improvement


def main():
    print("=" * 64)
    print(" Hysteresis compensation demo: auto-detect loading/unloading direction and fix the 'grudge'")
    print("=" * 64)
    print()

    # ---- ① ground truth: ideal body-pressure map of a lying person ----
    ideal = make_ideal_pressure(GRID_W, GRID_H, BODY_PARTS)
    print(f"[1/5] ideal body-pressure map (ground truth): {GRID_W}×{GRID_H}, peak {ideal.max():.1f}")

    # ---- ② simulate "lie down -> get up", adding hysteresis ----
    n_steps = 60
    true_frames, raw_frames, weights = simulate_load_unload(
        ideal, n_steps=n_steps, h=HYSTERESIS, noise=0.0)
    print(f"[2/5] simulated lie-down->get-up over {len(weights)} frames (loading first, unloading after), each frame adds hysteresis {HYSTERESIS}")

    # ---- ③ auto-detect direction + compensate ----
    corrected = cal.correct_hysteresis_sequence(raw_frames, HYSTERESIS, deadband=0.0)
    print("[3/5] auto-detected each frame's direction and finished hysteresis compensation")

    # ---- ④ compute error: before vs after compensation ----
    # mean absolute error over the whole run: all frames, all points, averaged "distance from ground truth"
    err_before = float(np.abs(raw_frames - true_frames).mean())
    err_after = float(np.abs(corrected - true_frames).mean())
    print(f"[4/5] mean error over the whole run: before {err_before:.2f} -> after {err_after:.2f}")

    # ---- ⑤ dynamic conclusion (not hard-coded; computed from the numbers) ----
    improve = cal.improvement(err_before, err_after)
    print()
    print(" Conclusion of this demo (auto-generated from the numbers above, not hard-coded):")
    print(f"   mean error caused by hysteresis {err_before:.2f}, down to {err_after:.2f} after compensation,")
    print(f"   error shrank {improve:.1f}% -> {describe_improvement(improve)}.")
    print()

    # ---- plot ----
    path = plot_result(true_frames, raw_frames, corrected, weights, ideal)
    print(f" Output figure: {path}")
    print()
    print(" ✅ Done. Open that figure: the blue line (compensated) will nearly hug the gray dashed line (truth),")
    print("    while the red line (dirty reading) clearly deviates — that's what fixed hysteresis looks like.")


def plot_result(true_frames, raw_frames, corrected, weights, ideal):
    """Draw two figures: left = pressure over time at the peak point; right = per-frame mean error (before/after compensation)."""
    os.makedirs(OUT_DIR, exist_ok=True)

    # find the peak point (where the hips are heaviest); its "pressure-time" curve is the most intuitive
    peak_idx = np.unravel_index(np.argmax(ideal), ideal.shape)
    t = np.arange(len(weights))

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # ---- left: peak point over time ----
    ax = axes[0]
    ax.plot(t, true_frames[:, peak_idx[0], peak_idx[1]], "--",
            color="gray", linewidth=2, label="true pressure (ground truth)")
    ax.plot(t, raw_frames[:, peak_idx[0], peak_idx[1]], "-",
            color="tab:red", linewidth=2, label="sensor reading (with hysteresis)")
    ax.plot(t, corrected[:, peak_idx[0], peak_idx[1]], "-",
            color="tab:blue", linewidth=2, label="compensated (auto-detected direction)")
    ax.set_xlabel("frame (time -> lie down first, then get up)")
    ax.set_ylabel("pressure")
    ax.set_title("Peak point: hysteresis shifts the reading, compensation pulls it back to truth")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # ---- right: per-frame mean error ----
    ax = axes[1]
    err_before = np.abs(raw_frames - true_frames).reshape(len(weights), -1).mean(axis=1)
    err_after = np.abs(corrected - true_frames).reshape(len(weights), -1).mean(axis=1)
    ax.plot(t, err_before, "-", color="tab:red", linewidth=2, label="error before compensation")
    ax.plot(t, err_after, "-", color="tab:blue", linewidth=2, label="error after compensation")
    ax.set_xlabel("frame (time -> lie down first, then get up)")
    ax.set_ylabel("mean absolute error")
    ax.set_title("Per-frame error: nearly zero after compensation")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    fig.suptitle("Hysteresis auto-compensation: the program judges the loading/unloading direction itself", fontsize=14)
    plt.tight_layout()
    path = os.path.join(OUT_DIR, FILE_HYSTERESIS_COMP)
    plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return path


if __name__ == "__main__":
    main()
