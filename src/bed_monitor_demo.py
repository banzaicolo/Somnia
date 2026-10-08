#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=============================================================================
Bed-exit monitoring demo — act out a "whole night of an elderly person" and
show when the system records and when it alarms
=============================================================================

What does this script demonstrate?

It applies the bed-exit monitoring from bed_monitor.py to a complete night-long
story, so you can see how the system goes from pressure maps, one frame at a
time, to deciding "where the person is and whether to alarm".

Story (280 frames, assuming 1 frame per second, i.e. 4 min 40 s; a real night
is 8 hours — time is compressed here just to tell the story clearly):

  frames  0~ 39   fall asleep at night, lying flat
  frames 40~ 49   turn over (total pressure dips slightly, but still in bed, no trigger)
  frames 50~ 89   keep sleeping
  frames 90~ 99   sit up to drink water (sitting state, no alarm)
  frames 100~129  lie back down and keep sleeping
  frames 130~169  get up at night to the toilet, return on their own after 40 s (normal, no alarm)
  frames 170~199  back in bed, keep sleeping
  frames 200~279  get out of bed again, this time never coming back → alarm only after the time limit

What the system does:
  1. Compute each frame's total pressure (one number)
  2. Judge each frame as "in bed / sitting up / out of bed" by ratio
  3. Record each "bed-exit event" (when they left, when they returned, how long) — no alarm
  4. Only "out of bed and not returning" triggers the alarm, cleared the moment they return

How to run:

    python3 src/bed_monitor_demo.py

After running, it writes bed_monitor.png into "outputs" — a state timeline chart.
=============================================================================
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")   # don't open a window, just save the figure
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["PingFang SC", "Arial Unicode MS", "Heiti SC"]
plt.rcParams["axes.unicode_minus"] = False

from sensor_config import (
    GRID_W, GRID_H, BODY_PARTS, NOISE_STD, OUT_DIR, SEED,
    FILE_BED_MONITOR, make_ideal_pressure,
)
import bed_monitor as bm


# The demo "timeout" duration. Real deployment is 30 minutes (1800 frames), which
# won't fit on the chart; compressed to 50 frames here so the "timeout alarm"
# moment is visible in the figure.
DEMO_AWAY_TIMEOUT = 50


def build_story(ideal, seed=SEED):
    """
    Build the whole-night frame sequence (see the file header for the story).

    Each frame = ideal body pressure map × a "pressure coefficient", plus a bit
    of noise (simulating real readings). The pressure coefficient w is "how
    solidly the person is pressing on the bed":
        w=1.0 lying flat fully pressed; w=0.7 turned over (center of mass shifted a bit but still in bed);
        w=0.4 sitting up; w=0 out of bed.
    """
    rng = np.random.default_rng(seed)

    weights = np.concatenate([
        np.full(40, 1.0),   # 0~39   fall asleep
        np.full(10, 0.7),   # 40~49  turn over (still in bed)
        np.full(40, 1.0),   # 50~89  keep sleeping
        np.full(10, 0.4),   # 90~99  sit up to drink water
        np.full(30, 1.0),   # 100~129 lie back down
        np.full(40, 0.0),   # 130~169 get up to the toilet (back in 40 s)
        np.full(30, 1.0),   # 170~199 back in bed, keep sleeping
        np.full(80, 0.0),   # 200~279 out of bed, never returning → timeout alarm
    ])

    frames = []
    for w in weights:
        frame = ideal * w
        frame = frame + rng.normal(0, NOISE_STD, size=frame.shape)
        frames.append(frame)
    return np.array(frames), weights


def main():
    print("=" * 64)
    print(" Bed-exit monitoring demo: one night of an elderly person (turning / drinking / night exit / timeout)")
    print("=" * 64)
    print()

    # ---- 1. build the ideal pressure map + story ----
    ideal = make_ideal_pressure(GRID_W, GRID_H, BODY_PARTS)
    frames, weights = build_story(ideal)
    print("[1/4] Built 280 frames of the whole-night story (see the file header)")

    # ---- 2. state machine: judge in bed or not per frame ----
    states = bm.detect_states(frames)

    # ---- 3. bed-exit events: record only, no alarm ----
    exits = bm.detect_exits(states)
    # ---- 4. alarm: only sounds when gone past the time limit ----
    alarms = bm.detect_alarms(states, away_timeout=DEMO_AWAY_TIMEOUT)

    # ---- print bed-exit events (the routine table) ----
    print()
    print("[2/4] Bed-exit events recorded (getting out of bed is normal; record only, no alarm):")
    if exits:
        for start, end in exits:
            print(f"      frame {start:>3} out of bed → frame {end:>3} back, away {end - start} frames")
    else:
        print("      (no valid exit the whole night)")

    # ---- print the alarm ----
    print()
    alarm_idx = [i for i, a in enumerate(alarms) if a]
    if alarm_idx:
        print(f"[3/4] Alarm: sounds at frame {alarm_idx[0]} (out of bed {DEMO_AWAY_TIMEOUT} frames with no return)")
        print(f"       cleared at frame {alarm_idx[-1] + 1} (the person returned)")
    else:
        print("[3/4] Peaceful night, no alarm triggered")

    # ---- plot ----
    path = plot_timeline(frames, states, alarms)
    print()
    print(f" output figure: {path}")
    print()
    print(" ✅ Done. Open the figure: green=in bed, yellow=sitting up, red=out of bed.")
    print("    Note: the night exit (frames 130~169) out-of-bed stretch is recorded only, no alarm;")
    print(f"    only the last out-of-bed stretch past {DEMO_AWAY_TIMEOUT} frames with no return sounds the alarm.")


def plot_timeline(frames, states, alarms):
    """Plot total pressure over time, colored by state, with alarm intervals marked."""
    os.makedirs(OUT_DIR, exist_ok=True)

    totals = frames.reshape(frames.shape[0], -1).sum(axis=1)
    peak = totals.max()
    t = np.arange(len(totals))

    fig, ax = plt.subplots(figsize=(14, 5.5))

    # background colored by state (light colors, so they don't drown out the curve)
    color_map = {bm.IN_BED: "#c6e8c6", bm.SITTING: "#ffe6b0", bm.OUT_OF_BED: "#f5c6c6"}
    i = 0
    while i < len(states):
        s = states[i]
        j = i
        while j < len(states) and states[j] == s:
            j += 1
        ax.axvspan(i, j, color=color_map[s], alpha=0.5, zorder=0)
        i = j

    # total pressure curve
    ax.plot(t, totals, color="black", linewidth=1.8, zorder=2, label="Total pressure")

    # two threshold lines: 60% in-bed line, 15% out-of-bed line
    ax.axhline(peak * 0.60, color="green", linestyle="--", linewidth=1.2,
               label="In-bed line (60%)")
    ax.axhline(peak * 0.15, color="red", linestyle="--", linewidth=1.2,
               label="Out-of-bed line (15%)")

    # alarm intervals: marked with red dots (only the timeout stretch alarms)
    if any(alarms):
        alarm_idx = [i for i, a in enumerate(alarms) if a]
        ax.scatter(alarm_idx, [totals[i] for i in alarm_idx],
                   color="red", s=45, zorder=3, label="Timeout alarm")

    ax.set_xlabel("Frame (1 frame per second = seconds)")
    ax.set_ylabel("Total pressure (sum of all sensor points)")
    ax.set_title("Bed-exit monitoring: total pressure + state + timeout-alarm timeline")
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(OUT_DIR, FILE_BED_MONITOR)
    plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return path


if __name__ == "__main__":
    main()
