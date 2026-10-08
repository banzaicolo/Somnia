#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=============================================================================
Sleep-pose dataset generator — synthesize training data when there is no real data
=============================================================================

What does it do?

Training a "sleep-pose classifier" needs lots of samples: many people, many
postures, many heights and weights. With no real hardware and no real data,
we "fabricate" it with a simulator:

    Starting from the three sleep postures (supine/side/prone), each sample is
    randomly jittered — position shifted a bit, pressure changed a bit, body
    weight varied — like 1000 different people lying down. This is exactly
    what machine learning calls "data augmentation".

What does it produce?

generate_dataset(n_per_class=200) returns:
    X — a (n_samples, 96) 2-D array. Each sample is an 8×12 pressure map
        flattened into 96 numbers
    y — a (n_samples,) array of class indices (0=supine, 1=side, 2=prone)

=============================================================================
"""

import numpy as np

from sensor_config import (
    GRID_W, GRID_H, POSES, POSE_NAMES,
    make_ideal_pressure,
)


# Normalization baseline: pressure maps range roughly 0~150; dividing by 200
# squeezes them into 0~0.75. Neural networks prefer "small and uniform" inputs;
# overly large numbers make training unstable.
NORMALIZE = 200.0


def make_pose_pressure(pose, rng, jitter=True):
    """
    Generate one pressure map for a given sleep posture.

    Args:
      pose   — posture name: "supine" / "side" / "prone"
      rng    — random number generator (np.random.default_rng(seed))
      jitter — whether to add random jitter. True = simulate "different people";
               False = canonical posture (for drawing diagrams)

    Returns: a (GRID_H, GRID_W) pressure map.
    """
    parts = POSES[pose]          # the "body-part table" for this posture

    if jitter:
        # Jitter each body part in two ways, simulating "different people lying down":
        #   position ±0.5 cell — some lie higher, some lower, some slightly skewed
        #   peak ±20%        — some heavier, some lighter, softer/harder mattresses
        parts = [(name,
                  cx + rng.uniform(-0.5, 0.5),          # jitter position
                  cy + rng.uniform(-0.5, 0.5),
                  sx, sy,
                  amp * rng.uniform(0.8, 1.2),          # jitter peak
                  ang)
                 for name, cx, cy, sx, sy, amp, ang in parts]

    # Use the shared "ideal pressure" function to overlay the parts into one map
    return make_ideal_pressure(GRID_W, GRID_H, parts)


def generate_dataset(n_per_class, seed=0, shuffle=True):
    """
    Generate a complete synthetic dataset.

    Args:
      n_per_class — how many samples per posture (e.g. 200, for 600 total)
      seed        — random seed. Fixing it makes the data reproducible across runs
      shuffle     — whether to shuffle the order. Training sets are usually
                    shuffled (otherwise the network learns all the supine
                    samples first, then the side ones, and becomes "biased")

    Returns:
      X — (N, 96) 2-D array, each row a flattened pressure map (normalized to ~0~1)
      y — (N,) array of class indices
      classes — list of class names ["supine", "side", "prone"]
    """
    rng = np.random.default_rng(seed)

    X_list, y_list = [], []
    for label, pose in enumerate(POSE_NAMES):   # label: 0/1/2
        for _ in range(n_per_class):
            img = make_pose_pressure(pose, rng, jitter=True)
            X_list.append(img.ravel())          # flatten 8×12 into 96
            y_list.append(label)

    X = np.array(X_list) / NORMALIZE            # normalize
    y = np.array(y_list)

    if shuffle:
        idx = rng.permutation(len(X))           # generate shuffled indices
        X, y = X[idx], y[idx]

    return X, y, list(POSE_NAMES)


def split_train_test(X, y, test_ratio=0.2, seed=0):
    """
    Split the data into a "training set" and an "exam paper".

    Why split? Testing on the training data is like copying the original
    questions — the score is inflated. Only a held-out set the model has never
    seen measures real skill.
    """
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(X))
    n_test = int(len(X) * test_ratio)

    test_idx, train_idx = idx[:n_test], idx[n_test:]
    return X[train_idx], y[train_idx], X[test_idx], y[test_idx]
