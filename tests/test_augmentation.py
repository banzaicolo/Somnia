# -*- coding: utf-8 -*-
"""Left/right mirror augmentation must flip the map and swap the label together."""

import numpy as np

import pupu_loader as pl


def test_mirror_flips_map_and_swaps_left_right():
    X = np.arange(72, dtype=float).reshape(1, 72)
    left = pl.POSE_ORDER.index("left")
    right = pl.POSE_ORDER.index("right")

    Xa, ya = pl.mirror_augment(X, np.array([left]), list(pl.POSE_ORDER), shuffle=False)

    assert len(Xa) == 2
    assert ya[0] == left
    assert ya[1] == right
    np.testing.assert_array_equal(Xa[1], X.reshape(1, 12, 6)[:, :, ::-1].ravel())


def test_mirror_keeps_supine_and_prone_labels():
    X = np.zeros((2, 72))
    y = np.array([pl.POSE_ORDER.index("supine"), pl.POSE_ORDER.index("prone")])

    _, ya = pl.mirror_augment(X, y, list(pl.POSE_ORDER), shuffle=False)

    np.testing.assert_array_equal(ya, [y[0], y[1], y[0], y[1]])


def test_no_shuffle_keeps_metadata_alignment():
    """Regression guard: shuffling X/y without shuffling meta breaks by-volunteer splits."""
    X = np.arange(6 * 72, dtype=float).reshape(6, 72)
    y = np.array([0, 1, 2, 3, 0, 1])

    Xa, ya = pl.mirror_augment(X, y, list(pl.POSE_ORDER), shuffle=False)

    np.testing.assert_array_equal(Xa[:6], X)
    np.testing.assert_array_equal(ya[:6], y)
