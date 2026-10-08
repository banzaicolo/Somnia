#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tests for the sleep-pose dataset generator."""

import numpy as np

from pose_dataset import (
    make_pose_pressure, generate_dataset, split_train_test, NORMALIZE,
)
from sensor_config import GRID_W, GRID_H, POSES, POSE_NAMES


class TestMakePosePressure:
    def test_shape(self):
        """All three postures produce an H×W image."""
        rng = np.random.default_rng(0)
        for pose in POSE_NAMES:
            img = make_pose_pressure(pose, rng)
            assert img.shape == (GRID_H, GRID_W)

    def test_not_random_garbage(self):
        """Generated maps must have real structure: max pressure clearly above min pressure."""
        rng = np.random.default_rng(0)
        for pose in POSE_NAMES:
            img = make_pose_pressure(pose, rng)
            assert img.max() > 20          # somewhere is pressed above 20
            assert img.min() < img.max() / 3   # and somewhere is almost unpressed

    def test_side_pose_asymmetric(self):
        """Side posture signature: pressure center shifts to one side. Left/right sums should differ a lot."""
        rng = np.random.default_rng(0)
        img = make_pose_pressure("side", rng, jitter=False)
        left_sum = img[:, : GRID_W // 2].sum()
        right_sum = img[:, GRID_W // 2:].sum()
        assert right_sum > left_sum * 1.5   # side lies to the right (x=9), right half clearly heavier

    def test_supine_symmetric(self):
        """Supine posture signature: roughly left-right symmetric (similar pressure sums on both sides of the midline)."""
        rng = np.random.default_rng(0)
        img = make_pose_pressure("supine", rng, jitter=False)
        left_sum = img[:, : GRID_W // 2].sum()
        right_sum = img[:, GRID_W // 2:].sum()
        # Two 40+ pressure bumps overlap; left/right should differ by no more than 10%
        assert abs(left_sum - right_sum) / max(left_sum, right_sum) < 0.1


class TestGenerateDataset:
    def test_shapes_and_labels(self):
        X, y, classes = generate_dataset(n_per_class=10, seed=0)
        assert X.shape == (30, GRID_W * GRID_H)   # 3 classes × 10 = 30 samples
        assert y.shape == (30,)
        assert classes == POSE_NAMES
        assert set(y) == {0, 1, 2}

    def test_normalized(self):
        """After normalization, values should stay in a reasonable small range (not exploding)."""
        X, _, _ = generate_dataset(n_per_class=5, seed=0)
        assert X.max() < 2.0   # after dividing by 200, there should be no very large values

    def test_seed_reproducible(self):
        """Fixed seed → two runs produce identical data (reproducible)."""
        X1, y1, _ = generate_dataset(n_per_class=5, seed=7)
        X2, y2, _ = generate_dataset(n_per_class=5, seed=7)
        np.testing.assert_allclose(X1, X2)
        np.testing.assert_array_equal(y1, y2)


class TestSplit:
    def test_ratio_and_no_overlap(self):
        X, y, _ = generate_dataset(n_per_class=50, seed=0)
        Xtr, ytr, Xte, yte = split_train_test(X, y, test_ratio=0.2, seed=0)
        assert len(Xte) == 30        # 150 × 0.2 = 30
        assert len(Xtr) == 120
        # Exam samples must never appear in the training set (no leaking answers)
        assert set(map(tuple, Xte)).isdisjoint(set(map(tuple, Xtr)))
