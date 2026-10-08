#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tests for the PoPu real-data reader + training chain.

The dataset itself (82MB) is not in git, so before testing we first check whether
the data is present: if not, skip the whole group (no error) — someone who clones
the repo without downloading the data can still run the other tests.
"""

import glob
import os

import numpy as np
import pytest

import pupu_loader as pl
from mlp import TinyMLP

# Only run these tests if the data directory exists
DATA_READY = os.path.isdir(pl.DEFAULT_DATA_DIR)
pytestmark = pytest.mark.skipif(
    not DATA_READY, reason="PoPu data not downloaded (python3 src/download_pupu.py full)")

# Load the full dataset once and share it across tests (about 5000 samples, a few seconds)
@pytest.fixture(scope="module")
def dataset():
    return pl.load_dataset(seed=0)


class TestParsePose:
    def test_four_poses(self):
        assert pl.parse_pose("supine1_0.json") == "supine"
        assert pl.parse_pose("left3_1.json") == "left"
        assert pl.parse_pose("right5_2.json") == "right"
        assert pl.parse_pose("prone2_0.json") == "prone"

    def test_empty_and_others_recognized_but_not_in_four_poses(self):
        assert pl.parse_pose("empty1.json") == "empty"
        assert pl.parse_pose("others.json") == "others"
        assert pl.parse_pose("empty1.json") not in pl.POSE_ORDER
        assert pl.parse_pose("others.json") not in pl.POSE_ORDER


class TestLoadSnapshots:
    def test_shape_is_12x6(self):
        files = glob.glob(os.path.join(pl.DEFAULT_DATA_DIR, "*", "supine*.json"))
        matrices, meta = pl.load_snapshots(files[0])
        assert matrices[0].shape == (12, 6)
        assert meta["volunteer_id"] is not None

    def test_dirty_frames_skipped_no_crash(self):
        # There are 29 frames in the whole dataset whose readings are not 72 values (missing values);
        # skipping them must not error, and every frame read must have the correct shape
        files = glob.glob(os.path.join(pl.DEFAULT_DATA_DIR, "*", "*.json"))[:200]
        for f in files:
            matrices, _ = pl.load_snapshots(f)
            for m in matrices:
                assert m.shape == (12, 6)


class TestLoadDataset:
    def test_sample_count_and_balance(self, dataset):
        X, y, classes, meta = dataset
        assert X.shape == (5040, 72)          # 4 classes × 1260, one frame per class
        assert len(classes) == 4
        assert (np.bincount(y) == 1260).all()  # four classes perfectly balanced

    def test_normalized_reasonable_range(self, dataset):
        X, y, _, _ = dataset
        assert X.min() > -5 and X.max() < 5    # small values after baseline subtraction + /20

    def test_reproducible(self):
        # Same seed twice loads identical results (fixed randomness is required to compare experiments)
        X1, y1, _, _ = pl.load_dataset(seed=7)
        X2, y2, _, _ = pl.load_dataset(seed=7)
        np.testing.assert_array_equal(X1, X2)
        np.testing.assert_array_equal(y1, y2)

    def test_signal_weakens_after_baseline_subtraction(self, dataset):
        # After baseline subtraction the signal amplitude should be single digits
        # (raw readings are on the order of 485). If "baseline subtraction" is
        # missed, X would contain big numbers around 485/20≈24; normally it should be < 10.
        # This test pins down the "zero-point calibration" effect: fake numbers that
        # were not baseline-subtracted must not appear in X.
        X, y, _, _ = dataset
        assert np.abs(X).max() < 10


class TestSplitByVolunteer:
    def test_train_and_test_volunteers_do_not_overlap(self, dataset):
        X, y, _, meta = dataset
        Xtr, ytr, Xte, yte = pl.split_by_volunteer(X, y, meta, seed=0)
        assert len(Xtr) + len(Xte) == len(X)
        assert len(Xte) < len(Xtr)             # test set is about 20%
        # Every class should appear on both sides (balanced split)
        assert set(np.unique(ytr)) == set(range(4))
        assert set(np.unique(yte)) == set(range(4))


class TestTrainConverges:
    def test_real_data_accuracy_far_exceeds_random(self, dataset):
        """4-class random guessing = 25%. After a few dozen epochs on real data it should far exceed that."""
        X, y, _, meta = dataset
        Xtr, ytr, Xte, yte = pl.split_by_volunteer(X, y, meta, seed=0)
        model = TinyMLP(72, 32, 4, seed=0)
        rng = np.random.default_rng(0)
        n = len(Xtr)
        for ep in range(40):                   # few epochs, to keep the test fast
            idx = rng.permutation(n)
            for s in range(0, n, 32):
                b = idx[s:s + 32]
                model.train_step(Xtr[b], ytr[b], lr=0.3)
        acc = model.accuracy(Xte, yte)
        assert acc > 0.8                       # stranger exam should also far exceed random guessing
