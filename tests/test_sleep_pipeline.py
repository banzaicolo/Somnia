#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=============================================================================
test_sleep_pipeline.py — tests for the sleep staging "wiring" module
=============================================================================

Tests these things:
  1. synthesized full-night signal: correct shape, reproducible
  2. digging features from the waveform: correct shape, finite values, movement direction correct (wake > deep)
  3. feature-extraction accuracy: dug-out breathing/heart rate close to the true value (the core evidence of wiring)
  4. end to end: the dug-out features really train a classifier stronger than random
  5. end-to-end full night: staging a full night, κ above random level
"""

import numpy as np

import sleep_pipeline as sp
import bcg_monitor as bcg
from sleep_staging import standardize, cohens_kappa, STAGE_CENTERS


# ============================================================================
# 1. Synthesize a full-night signal
# ============================================================================

def test_simulate_signal_shape():
    """Signal length = segments × 30 s × sampling rate, label length = segments."""
    n_epochs = 50
    signal, y, names = sp.simulate_night_signal(n_epochs=n_epochs, seed=0)
    assert len(signal) == n_epochs * int(sp.EPOCH_SEC * sp.FS)
    assert y.shape == (n_epochs,)
    assert len(names) == n_epochs


def test_simulate_reproducible():
    """Two syntheses with the same seed produce identical signals."""
    s1, y1, _ = sp.simulate_night_signal(40, seed=7)
    s2, y2, _ = sp.simulate_night_signal(40, seed=7)
    np.testing.assert_array_equal(s1, s2)
    np.testing.assert_array_equal(y1, y2)


def test_simulate_different_seed_differs():
    """Different seeds produce different nights."""
    _, y1, _ = sp.simulate_night_signal(100, seed=1)
    _, y2, _ = sp.simulate_night_signal(100, seed=2)
    assert not np.array_equal(y1, y2)


# ============================================================================
# 2. Dig features out of the waveform
# ============================================================================

def test_extract_feature_shape():
    """The extracted feature matrix has the correct shape (segments, 5), all finite values."""
    signal, _, _ = sp.simulate_night_signal(30, seed=3)
    X = sp.extract_epoch_features(signal)
    assert X.shape == (30, 5)
    assert np.isfinite(X).all()


def test_extract_movement_direction():
    """Wake-segment movement energy should be clearly larger than deep-sleep (physiological movement difference)."""
    fs = sp.FS
    rng = np.random.default_rng(0)
    wake = sp._synthesize_epoch(STAGE_CENTERS["wake"][1], STAGE_CENTERS["wake"][2],
                                STAGE_CENTERS["wake"][3], STAGE_CENTERS["wake"][4],
                                STAGE_CENTERS["wake"][0] * sp.MOV_AMP_SCALE, fs, rng)
    deep = sp._synthesize_epoch(STAGE_CENTERS["deep"][1], STAGE_CENTERS["deep"][2],
                                STAGE_CENTERS["deep"][3], STAGE_CENTERS["deep"][4],
                                STAGE_CENTERS["deep"][0] * sp.MOV_AMP_SCALE, fs, rng)
    wake_mov = bcg.band_energy(wake - wake.mean(), fs, sp.MOV_LOW, sp.MOV_HIGH)
    deep_mov = bcg.band_energy(deep - deep.mean(), fs, sp.MOV_LOW, sp.MOV_HIGH)
    assert wake_mov > deep_mov


# ============================================================================
# 3. Feature-extraction accuracy (the core evidence of wiring)
# ============================================================================

def test_extract_breathing_and_heart_rate_accurate():
    """The breathing/heart rate dug from a fixed signal should be close to the true value (30 s window resolution ≈ 2/min)."""
    fs = 50
    t = np.arange(int(30 * fs)) / fs
    # breathing 15 breaths/min + heartbeat 72 beats/min, no noise
    sig = 100 + 5 * np.sin(2 * np.pi * (15 / 60) * t) \
             + 0.5 * np.sin(2 * np.pi * (72 / 60) * t)
    X = sp.extract_epoch_features(sig, fs=fs)
    assert X.shape == (1, 5)
    assert abs(X[0, 1] - 15.0) < 2.5   # breathing rate
    assert abs(X[0, 2] - 72.0) < 3.0   # heart rate


# ============================================================================
# 4. End to end: the dug-out features train a useful classifier
# ============================================================================

def test_end_to_end_staging_accuracy_above_random():
    """Dig features from the waveform → train → exam; accuracy should far exceed four-class random guessing's 25%."""
    from mlp import TinyMLP

    X, y = sp.extract_dataset(n_nights=4, n_epochs=120, seed=42)
    X_scaled, _, _ = standardize(X)

    rng = np.random.default_rng(42)
    idx = rng.permutation(len(X))
    n_test = int(len(X) * 0.2)
    X_train, y_train = X_scaled[idx[n_test:]], y[idx[n_test:]]
    X_test, y_test = X_scaled[idx[:n_test]], y[idx[:n_test]]

    model = TinyMLP(n_input=5, n_hidden=16, n_output=4, seed=42)
    n = len(X_train)
    for _ in range(100):
        p = rng.permutation(n)
        for start in range(0, n, 32):
            model.train_step(X_train[p[start:start + 32]],
                             y_train[p[start:start + 32]], lr=0.3)

    acc = model.accuracy(X_test, y_test)
    assert acc > 0.6   # random guess 25%; reaching 60%+ shows the "wiring" really works


def test_full_night_staging_kappa_above_random():
    """Staging a full night end to end, κ should far exceed random (near 0)."""
    from mlp import TinyMLP

    # train a small model
    X, y = sp.extract_dataset(n_nights=4, n_epochs=120, seed=42)
    X_scaled, mean, std = standardize(X)
    rng = np.random.default_rng(42)
    model = TinyMLP(5, 16, 4, seed=42)
    n = len(X_scaled)
    for _ in range(100):
        p = rng.permutation(n)
        for start in range(0, n, 32):
            model.train_step(X_scaled[p[start:start + 32]],
                             y[p[start:start + 32]], lr=0.3)

    # stage a full night
    signal, y_night, _ = sp.simulate_night_signal(240, seed=999)
    preds = sp.stage_from_signal(signal, model, mean, std)
    kappa = cohens_kappa(y_night, preds, 4)
    assert kappa > 0.5
