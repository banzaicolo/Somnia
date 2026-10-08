#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=============================================================================
test_sleep_staging.py — tests for the sleep staging module
=============================================================================

Tests these things:
  1. a synthesized full night: correct shape, reproducible, all four classes present
  2. feature values in a reasonable range (non-negative movement, bounded heart/breathing rate)
  3. evaluation metrics: confusion matrix computed correctly, Cohen's Kappa at two extremes (perfect = 1, random ≈ 0)
  4. the classifier really can learn to stage (accuracy far above random guessing's 25%)
  5. standardized features have mean 0 and std 1
"""

import numpy as np

import sleep_staging as ss


# ============================================================================
# 1. Synthesize a full night
# ============================================================================

def test_synthesize_night_shape():
    """Correct output shapes: X is (N,5), y is (N,), names is the four class names."""
    X, y, names = ss.synthesize_night(100, seed=0)
    assert X.shape == (100, 5)
    assert y.shape == (100,)
    assert names == ss.STAGES


def test_synthesize_night_reproducible():
    """Two syntheses with the same seed produce identical results (easy to compare/debug)."""
    X1, y1, _ = ss.synthesize_night(100, seed=7)
    X2, y2, _ = ss.synthesize_night(100, seed=7)
    np.testing.assert_array_equal(X1, X2)
    np.testing.assert_array_equal(y1, y2)


def test_synthesize_night_different_seed_differs():
    """Different seeds should produce different nights (otherwise the synthesis is fake)."""
    _, y1, _ = ss.synthesize_night(200, seed=1)
    _, y2, _ = ss.synthesize_night(200, seed=2)
    assert not np.array_equal(y1, y2)


def test_sleep_structure_has_four_stages():
    """In a long enough night, all four stages (wake/light/deep/REM) should appear."""
    _, y, _ = ss.synthesize_night(500, seed=3)
    assert set(y.tolist()) == {0, 1, 2, 3}


def test_feature_ranges_sane():
    """Feature values don't go out of bounds: non-negative movement, heart rate ≥30, breathing rate ≥4 (clamped lower bounds)."""
    X, _, _ = ss.synthesize_night(300, seed=4)
    assert (X[:, 0] >= 0).all()      # movement amplitude
    assert (X[:, 2] >= 30).all()     # heart rate
    assert (X[:, 1] >= 4).all()      # breathing rate


# ============================================================================
# 2. Evaluation metrics
# ============================================================================

def test_confusion_matrix():
    """The confusion matrix accumulates correctly along "true → predicted"."""
    y_true = np.array([0, 0, 1, 1, 2, 2, 3, 3])
    y_pred = np.array([0, 1, 1, 1, 2, 3, 3, 3])
    cm = ss.confusion_matrix(y_true, y_pred, 4)
    assert cm.shape == (4, 4)
    assert cm[0, 0] == 1   # true 0 predicted 0: 1 case
    assert cm[0, 1] == 1   # true 0 predicted 1: 1 case
    assert cm[1, 1] == 2   # true 1 predicted 1: 2 cases
    assert cm[2, 2] == 1
    assert cm[2, 3] == 1
    assert cm[3, 3] == 2


def test_kappa_perfect_agreement():
    """Prediction identical to truth → κ must be 1.0 (perfect score)."""
    y = np.array([0, 1, 2, 3] * 10)
    assert ss.cohens_kappa(y, y, 4) == 1.0


def test_kappa_random_near_zero():
    """Pure random guessing → κ should be near 0 (no skill, subtracting the luck)."""
    rng = np.random.default_rng(0)
    y_true = np.array([0, 1, 2, 3] * 50)
    y_pred = rng.integers(0, 4, size=len(y_true))
    k = ss.cohens_kappa(y_true, y_pred, 4)
    assert abs(k) < 0.15


# ============================================================================
# 3. The classifier really can learn (end to end)
# ============================================================================

def test_classifier_accuracy_above_random():
    """With a quick train on little data, accuracy should far exceed random guessing's 25%."""
    from sleep_staging import generate_dataset, standardize
    from mlp import TinyMLP

    X, y = generate_dataset(n_nights=3, n_epochs=120, seed=42)
    X_scaled, _, _ = standardize(X)

    rng = np.random.default_rng(42)
    idx = rng.permutation(len(X))
    n_test = int(len(X) * 0.2)
    X_train, y_train = X_scaled[idx[n_test:]], y[idx[n_test:]]
    X_test, y_test = X_scaled[idx[:n_test]], y[idx[:n_test]]

    model = TinyMLP(n_input=5, n_hidden=16, n_output=4, seed=42)
    n = len(X_train)
    for _ in range(80):
        p = rng.permutation(n)
        for start in range(0, n, 32):
            model.train_step(X_train[p[start:start + 32]],
                             y_train[p[start:start + 32]], lr=0.3)

    acc = model.accuracy(X_test, y_test)
    assert acc > 0.5   # four-class random guess = 25%; reaching 50%+ shows real learning


# ============================================================================
# 4. Feature standardization
# ============================================================================

def test_standardize():
    """After standardization, each column has mean ≈0 and std ≈1 (standard ML preprocessing)."""
    X = np.array([
        [0.0, 10, 60],
        [1.0, 20, 80],
        [2.0, 30, 70],
        [1.5, 15, 55],
    ], dtype=float)
    Xs, mean, std = ss.standardize(X)
    np.testing.assert_allclose(Xs.mean(axis=0), 0.0, atol=1e-9)
    np.testing.assert_allclose(Xs.std(axis=0), 1.0, atol=1e-9)
    assert mean.shape == (3,) and std.shape == (3,)


def test_standardize_constant_column_no_div_by_zero():
    """When a column is all the same (std 0), it must not divide by zero; it should pass safely."""
    X = np.array([
        [0.0, 10, 60],
        [1.0, 10, 80],
        [2.0, 10, 70],
    ], dtype=float)
    Xs, mean, std = ss.standardize(X)
    assert np.isfinite(Xs).all()   # no NaN or inf
