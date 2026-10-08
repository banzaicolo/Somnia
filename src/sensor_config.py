#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=============================================================================
Central configuration module — all "adjustable knobs" and shared functions live here
=============================================================================

[Why does this file exist?]

Originally three scripts (pressure simulator / zero calibration / body-shape proof)
each copied the same "grid size", "body parts", and "ideal pressure formula".

That had a big problem: if you want to change a body part (e.g. widen the shoulders),
you'd have to edit three files, and miss one and everything falls apart — the three
"ground truths" wouldn't match.

Now they're pulled into this file, and all three scripts read from here. **Change
once, takes effect in three places.** This is what software engineering calls a
"Single Source of Truth".

[How to use it?]

In other scripts write:

    import sensor_config as cfg
    ideal = cfg.make_ideal_pressure(cfg.GRID_W, cfg.GRID_H, cfg.BODY_PARTS)

To temporarily override a parameter (e.g. a script wants a more exaggerated temp
drift), just reassign it in that script; it won't affect the others:

    cfg.TEMP_DRIFT = 30   # takes effect only in this script

=============================================================================
"""

import numpy as np


# ============================================================================
# 1. Sensor array size
# ============================================================================
GRID_W = 12   # how many sensing points horizontally (columns)
GRID_H = 8    # how many sensing points vertically (rows)
              # 12×8 = 96 points, matching the 96 points in the edge control backbone


# ============================================================================
# 2. Pressure parameters of each body part
# ============================================================================
# When a person lies supine, the body doesn't press evenly — a few "protruding parts"
# press hardest: head, shoulder blades, hips, heels. Each part is modeled as a
# "Gaussian ellipse" — a bump that's high in the middle and tapers off around the
# edges, very close to the real pressure distribution.
# Parameter order: (name, center col x, center row y, width sx, height sy, peak pressure, rotation angle)
# Note y goes top-down: smaller y is closer to the head of the bed (head), larger y closer to the foot (heels).
BODY_PARTS = [
    ("Head",      5.5, 1.0, 1.8, 1.0, 35, 0),   # topmost, least pressure
    ("Shoulders", 5.5, 3.0, 2.6, 1.6, 80, 0),   # shoulders, medium pressure
    ("Hips",      5.5, 5.3, 2.8, 1.8, 110, 0),  # hips, heaviest of all
    ("Heels",     5.5, 7.2, 1.4, 0.7, 30, 0),   # bottommost, least pressure
]


# ============================================================================
# 2.5. Three sleeping postures (data source for "posture classification")
# ============================================================================
# The three most typical postures, each a "list of body parts" in the same format
# as BODY_PARTS:
#   (name, center col x, center row y, width sx, height sy, peak pressure, rotation angle)
#
# Why distinguish postures? Look at the features of their pressure maps:
#   supine: all parts on the center axis (left-right symmetric), hips heaviest
#   side:   all parts shifted to one side (x=9), shoulder and hip become "point contacts", pressure is actually highest
#   prone:  lying face down, chest/abdomen touch the bed over a large area, pressure is spread thin (lower peak but larger area)
POSES = {
    "supine": [   # supine (face up)
        ("Head",      5.5, 1.0, 1.8, 1.0, 35, 0),
        ("Shoulders", 5.5, 3.0, 2.6, 1.6, 80, 0),
        ("Hips",      5.5, 5.3, 2.8, 1.8, 110, 0),
        ("Heels",     5.5, 7.2, 1.4, 0.7, 30, 0),
    ],
    "side": [     # side (lying on the right side, all parts shifted to the x=9 side)
        ("Head",     9.0, 1.0, 1.6, 1.0, 45, 0),
        ("Shoulder", 9.0, 3.0, 2.0, 1.4, 120, 0),   # point contact -> high pressure
        ("Hip",      9.0, 5.5, 2.2, 1.5, 130, 0),   # point contact -> heaviest of all
        ("Legs",     9.0, 7.3, 1.8, 1.0, 60, 0),
    ],
    "prone": [    # prone (lying face down, head turned to one side)
        ("Head",   4.5, 0.8, 1.6, 1.0, 40, 0),
        ("Chest",  5.5, 3.5, 3.2, 2.2, 70, 0),    # large contact area -> pressure spread thin
        ("Thighs", 5.5, 5.5, 2.6, 1.4, 60, 0),
        ("Feet",   5.5, 7.5, 2.0, 0.8, 25, 0),
    ],
}

# label list for posture classification (fixed order; training and prediction both map indices with this order)
POSE_NAMES = ["supine", "side", "prone"]
POSE_LABELS_CN = {"supine": "Supine", "side": "Side", "prone": "Prone"}


# ============================================================================
# 3. Strength of each sensor "fault" (each can be tuned alone to see what it alone causes)
# ============================================================================

# fault ① batch mismatch: within one batch of sensors each sensitivity differs (15% = 0.15)
SENS_GAIN_STD = 0.15

# fault ② temp drift: when temperature changes, the whole baseline shifts. Bigger = lifted more
TEMP_DRIFT = 12

# fault ③ zero drift: over time each point slowly "drifts on its own", simulated as random 0~8
ZERO_DRIFT_MAX = 8

# fault ④ noise: small random jitter, like old TV static
NOISE_STD = 3

# fault ⑤ crosstalk: adjacent points "leak" into each other; 0.1 means 10% leaks to neighbors
CROSSTALK = 0.10

# hysteresis loop width: how much loading and unloading can differ (for figure 2; also the shared forward/reverse parameter of hysteresis compensation)
HYSTERESIS = 8


# ============================================================================
# 4. Output settings (no longer hard-coded; change only here)
# ============================================================================
OUT_DIR = "outputs"   # which folder to save figures and csv to
SEED = 42             # random seed: fixed so every run gives the same result, easy to compare

# output filenames (English, so cross-platform and GitHub won't get garbled)
FILE_IMPERFECTIONS = "imperfections.png"       # figure 1: four-panel fault stacking
FILE_HYSTERESIS = "hysteresis_curve.png"       # figure 2: hysteresis loop
FILE_IDEAL_CSV = "ideal_pressure.csv"          # ground truth
FILE_RAW_CSV = "sensor_readings.csv"           # dirty data (input to the calibration algorithm)
FILE_ZERO_CAL = "zero_calibration.png"         # zero calibration comparison
FILE_BODY_PROOF = "body_shape_proof.png"       # body-shape proof figure
FILE_HYSTERESIS_COMP = "hysteresis_compensation.png"   # hysteresis auto-compensation comparison figure
FILE_BED_MONITOR = "bed_monitor.png"                   # bed-exit detection status timeline
FILE_BCG = "bcg_monitor.png"                           # BCG respiration/heartbeat monitoring figure
FILE_SLEEP_PIPELINE = "sleep_pipeline.png"             # signal -> feature -> staging end-to-end demo figure
FILE_PUPU_REAL = "pupu_real_poses.png"                 # PoPu real data visualization (before/after calibration)
FILE_PUPU_TRAINING = "pupu_training.png"               # real data training (loss curve + confusion matrix)


# ============================================================================
# Shared functions
# ============================================================================

def make_ideal_pressure(w, h, parts):
    """
    Generate the "ideal body-pressure map" — the ground truth: what a person actually pressing should look like.

    Recipe: treat each body part as a Gaussian bump and stack them all together.

    Args:
      w, h  -- array width and height
      parts -- the BODY_PARTS list
    Returns: a 2D array of h rows by w columns (i.e. an "image"); larger values mean higher pressure.
    This function [has no random numbers], so the body shape is fixed in place, not randomly generated.
    """
    # np.mgrid generates the coordinate grid. yy is each point's "row index", xx is "column index".
    yy, xx = np.mgrid[0:h, 0:w]
    img = np.zeros((h, w))   # start with an all-zero image

    for name, cx, cy, sx, sy, amp, ang in parts:
        # Gaussian ellipse formula: first translate coordinates to "origin at the bump center"
        dx = xx - cx
        dy = yy - cy

        # If a rotation angle is set, rotate the coordinates (default 0 degrees here,
        # i.e. no rotation, but this line is kept so you can simulate side/reclined lying later)
        a = np.cos(np.radians(ang))
        b = np.sin(np.radians(ang))
        xr = dx * a + dy * b
        yr = -dx * b + dy * a

        # 2D Gaussian: center = amp, decaying outward on the sx/sy scales
        gauss = amp * np.exp(-(xr ** 2 / (2 * sx ** 2) + yr ** 2 / (2 * sy ** 2)))
        img += gauss   # stack it on

    return img
