#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=============================================================================
sleep_staging.py — sleep staging: split a full night into "wake / light / deep / REM"
=============================================================================

What problem does this solve?

People don't sleep through the night in one stretch — they cycle between several
"sleep stages", rising and falling like a roller coaster. Sleep staging labels
the whole night segment by segment: awake, light sleep, deep sleep, or dreaming.

Why stage at all? Because "sleeping 8 hours" is not the same as "sleeping well".
What matters is whether deep sleep and REM are sufficient. Deep sleep handles
physical recovery; REM consolidates memory. When a doctor looks at someone's
sleep, they don't look at total duration — they look at this "sleep staging chart".

How is it staged? How many classes?

The medical gold standard (PSG, polysomnography) uses brain waves and can
distinguish 5 classes. Our mattress has no EEG, so it can only guess indirectly
from "movement + heart rate + breathing", and uses 4 classes:

    0 wake  — awake
    1 light — just fell asleep, not deeply (N1 + N2 in medicine)
    2 deep  — the deepest, most restorative sleep (N3 in medicine, a.k.a. slow-wave sleep)
    3 rem   — rapid eye movement sleep (REM), dreaming and memory consolidation

Core principle: each stage's "movement, heart rate, breathing" looks different

This is the physiological basis that lets sleep staging "guess indirectly"
(no EEG, so it relies on these clues):

    wake:  lots of movement, fast and erratic heart rate, unstable breathing
    light: little movement, heart rate eases down, breathing stabilizes
    deep:  movement near zero, slowest and steadiest heart rate, slow steady breathing
    rem:   slight movement, large heart-rate swings (like awake), irregular breathing

Every 30 seconds (one "staging unit", an epoch), the program computes 5 numbers:

    1. movement amplitude   — how much the body moved in these 30 s (tossing vs. still)
    2. mean breathing rate  — breaths per minute
    3. mean heart rate      — beats per minute
    4. breathing variability — how steady breathing is (stable vs. speeding up/slowing down)
    5. heart-rate variability — how steady the heartbeat is (a simplified HRV, stable vs. erratic)

Feed these 5 numbers to a classifier and it decides which stage that segment is.

Honest note: why "synthetic features" instead of "extracted from the signal"?

The method for "digging" breathing rate and heart rate out of the raw mattress
signal was already fully demonstrated in the previous module, bcg_monitor.py
(splitting by frequency with FFT).

This module focuses on the new topic of "sleep staging", so it synthesizes the
"5 features per 30 s" mathematically — equivalent to "the sensor has already
computed movement, heart rate, and breathing and hands them straight to the
stager". In a real mattress system, those 5 numbers are obtained by running the
signal through processing like bcg_monitor first. Splitting into two steps keeps
each one focused on a single idea.

Usage:

    import sleep_staging as ss
    X, y, names = ss.synthesize_night(480)   # synthesize one night (480 epochs)
    # X: (480, 5) features; y: (480,) stage labels; names: list of stage names

=============================================================================
"""

import numpy as np


# ============================================================================
# 1. Four sleep stages + the "typical feature values" for each stage
# ============================================================================

# Stage name list: fixed order, indices 0/1/2/3 correspond to this list,
# used in both training and prediction.
STAGES = ["wake", "light", "deep", "rem"]

# Human-readable stage labels (used in plots and console output)
STAGE_LABELS = {
    "wake": "Wake",
    "light": "Light",
    "deep": "Deep",
    "rem": "REM",
}

# The "typical feature values" (feature centers) for each stage, fixed order:
#   [movement amplitude, mean breathing rate, mean heart rate, breathing variability, HR variability]
#
# These numbers are reasonable approximations based on sleep physiology (not made up):
#   heart rate: fastest when awake (72), slowest in deep sleep (52), mid-high in REM (64) with large swings
#   breathing: slowest and steadiest in deep sleep (13), most erratic in REM (variability 2.8)
#   movement: highest when awake (0.70), near zero in deep sleep (0.03), slight in REM (0.08)
STAGE_CENTERS = {
    "wake":  [0.70, 16.0, 72.0, 2.5, 8.0],
    "light": [0.15, 15.0, 60.0, 1.5, 4.0],
    "deep":  [0.03, 13.0, 52.0, 0.8, 2.5],
    "rem":   [0.08, 16.5, 64.0, 2.8, 7.0],
}

# Per-feature "between-subject noise" — simulating different people, different nights.
# Order matches above. Value is how much random fluctuation (std) to add to that feature.
# E.g. heart rate gets ±4 bpm to simulate "slept extra deeply tonight / a different person".
STAGE_NOISE = [0.10, 1.5, 4.0, 0.8, 2.0]

# One staging unit every 30 s. 30 s is the standard staging duration in medicine (AASM standard).
EPOCH_SEC = 30


# ============================================================================
# 2. Synthesize one full night's "sleep structure" (stage sequence)
# ============================================================================

def _build_sleep_structure(n_epochs, rng):
    """
    Generate one full night's "stage sequence" — a long list where each element
    is one of "wake"/"light"/"deep"/"rem", with length = n_epochs.

    This simulates the real "sleep cycle": after falling asleep, the body cycles
    through "light → deep → light → REM", about 90 minutes per cycle, 4–5 cycles
    per night. There are two hard rules:

      1. deep sleep concentrates in the first half of the night (deepest in the
         first few hours), thinning out toward morning;
      2. REM concentrates in the second half (most dreams near dawn), getting
         longer as the night goes on.

    These two rules are common knowledge in sleep medicine, reflected here by
    "deep-sleep length decreases cycle by cycle, REM length increases cycle by cycle".

    Parameters:
      n_epochs — total number of 30 s units (8 hours = 960)
      rng      — random number generator

    Returns:
      stages — list[str], length n_epochs
    """
    stages = []

    # Before falling asleep: a brief awake stretch first
    # (lying in bed but not yet asleep, 2–5 units)
    stages += ["wake"] * int(rng.integers(2, 6))

    cycle = 0   # which sleep cycle we're on
    while len(stages) < n_epochs:
        # ---- Structure of each cycle: light → deep → light → REM ----
        # Stage durations are set so the whole-night proportions match real sleep:
        #   light ~half (45–55%), deep ~20%, REM ~20–25%, wake <5%.
        #
        # deep-sleep length: decreases with each cycle (less deep sleep later in the night)
        deep_len = int(rng.integers(18, 35) * max(0.35, 1.0 - 0.15 * cycle))
        # REM length: increases with each cycle (dreams get longer later in the night)
        rem_len = int(rng.integers(8, 18) * (1.0 + 0.15 * cycle))
        # light-sleep length: the two stretches together are the bulk; the first stretch is longer
        light_len = int(rng.integers(18, 32))
        light2_len = int(rng.integers(12, 24))

        stages += ["light"] * light_len
        stages += ["deep"] * max(3, deep_len)
        stages += ["light"] * light2_len
        stages += ["rem"] * max(2, rem_len)

        # One cycle done: with ~50% probability, briefly wake (toss, half-awake)
        if rng.random() < 0.5:
            stages += ["wake"] * int(rng.integers(1, 4))

        cycle += 1

    # Truncate to exactly n_epochs
    return stages[:n_epochs]


def synthesize_night(n_epochs=480, seed=42):
    """
    Synthesize "one full night" of sleep data: stage sequence + per-unit feature vector.

    This is the no-real-data fallback (same idea as the stress simulator and
    the BCG synthesized signal): use math to "act out" a ground truth, used to
    develop and validate the sleep-staging algorithm.

    Parameters:
      n_epochs — how many 30 s units this night has (default 480 = 4 hours,
                 enough for a demo; a real 8-hour sleep = 960)
      seed     — random seed; fixing it makes results reproducible

    Returns:
      X      — (n_epochs, 5) feature matrix. Each row is those 5 feature numbers
      y      — (n_epochs,) stage index (0=wake 1=light 2=deep 3=rem)
      names  — list of stage names, i.e. STAGES (so callers know which index means what)
    """
    rng = np.random.default_rng(seed)

    stages = _build_sleep_structure(n_epochs, rng)

    X = np.zeros((len(stages), len(STAGE_NOISE)))
    for i, stage in enumerate(stages):
        center = np.array(STAGE_CENTERS[stage], dtype=float)
        noise = np.array(STAGE_NOISE, dtype=float)
        # center + Gaussian noise → this unit's 5 features
        X[i] = center + rng.normal(0, 1, size=len(center)) * noise
        # movement amplitude physically can't be negative, and breathing/heart rate
        # have lower bounds too, so clamp them
        X[i, 0] = max(0.0, X[i, 0])
        X[i, 1] = max(4.0, X[i, 1])   # breathing rate floor 4 breaths/min
        X[i, 2] = max(30.0, X[i, 2])  # heart rate floor 30 bpm

    y = np.array([STAGES.index(s) for s in stages], dtype=int)
    return X, y, list(STAGES)


# ============================================================================
# 3. Build a "multi-night" dataset (training a classifier needs lots of samples)
# ============================================================================

def generate_dataset(n_nights=10, n_epochs=240, seed=42):
    """
    Synthesize multiple nights and combine them into one large dataset for training.

    Parameters:
      n_nights — how many nights to synthesize (default 10)
      n_epochs — how many units per night (default 240 = 2 hours; enough and fast for training)
      seed     — random seed

    Returns:
      X — (total units, 5) feature matrix
      y — (total units,) stage index
    """
    X_list, y_list = [], []
    for i in range(n_nights):
        # a different seed per night, simulating "different nights"
        X, y, _ = synthesize_night(n_epochs, seed=seed + i)
        X_list.append(X)
        y_list.append(y)
    return np.vstack(X_list), np.concatenate(y_list)


# ============================================================================
# 4. Feature standardization (standard ML practice)
# ============================================================================

def standardize(X, mean=None, std=None):
    """
    Standardize features: for each column "subtract mean, divide by std",
    yielding mean 0 and std 1.

    Why is this required? The 5 features have wildly different units: heart rate
    is a big number around 50–70, while movement amplitude is a small 0–1 value.
    Without standardization the network would be "biased" — looking only at the
    big heart-rate number and ignoring the small movement value. Standardization
    puts all 5 features "on the same starting line".

    Parameters:
      X    — (N, 5) feature matrix
      mean — per-column means. Pass None during training (computed here and returned);
             pass the training-set mean during prediction, so you "measure with the same ruler"
      std  — per-column std, same as above

    Returns:
      X_scaled — standardized matrix
      mean, std — the computed mean and std (reuse them at prediction time)
    """
    if mean is None:
        mean = X.mean(axis=0)
    if std is None:
        std = X.std(axis=0)
    std_safe = np.where(std == 0, 1.0, std)   # avoid division by zero if a column is constant
    return (X - mean) / std_safe, mean, std


# ============================================================================
# 5. Evaluation metrics: confusion matrix + Cohen's Kappa
# ============================================================================

def confusion_matrix(y_true, y_pred, n_classes):
    """
    Confusion matrix: an n×n table, rows = true stage, columns = predicted stage.
    The bigger the diagonal (correct guesses) the better; off-diagonal shows
    "what got mistaken for what".

    Medical AI evaluation uses this rather than a blanket accuracy, because you
    need to know where the errors are. For example, mistaking deep sleep for
    light sleep is far less harmful than mistaking it for wake.

    Parameters:
      y_true, y_pred — true/predicted stage-index arrays
      n_classes      — number of classes (4)

    Returns:
      cm — (n_classes, n_classes) integer matrix, cm[i][j] = count of true i predicted as j
    """
    cm = np.zeros((n_classes, n_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    return cm


def cohens_kappa(y_true, y_pred, n_classes):
    """
    Cohen's Kappa (κ) — measures "how much better than random guessing the staging is".

    Accuracy alone has a trap: if one class is very frequent (say light sleep is
    50%), a model that mindlessly guesses "all light" still gets 50% accuracy, but
    that's not skill. Kappa corrects for this: it subtracts the part you'd get
    right by blind luck and only counts "real skill".

    Formula: κ = (observed agreement - expected agreement) / (1 - expected agreement)
      - observed agreement: the proportion actually guessed correctly (i.e. accuracy)
      - expected agreement: the proportion a blind guesser would get right from class priors alone

    Interpreting κ (consensus in the sleep-staging field):
      κ = 1     perfect agreement
      κ = 0     no better than guessing
      κ ≥ 0.81  meets the medical-grade sleep-staging bar (EEG/PSG level)
      κ = 0.3~0.6  typical for consumer-grade non-EEG (watch/mattress)

    Parameters:
      y_true, y_pred — true/predicted stage-index arrays
      n_classes      — number of classes

    Returns:
      kappa — float, -1 ~ 1
    """
    cm = confusion_matrix(y_true, y_pred, n_classes)
    n = cm.sum()
    if n == 0:
        return 0.0

    observed = np.trace(cm) / n          # observed agreement = accuracy
    # expected agreement: sum over classes of "true fraction × predicted fraction"
    expected = 0.0
    for i in range(n_classes):
        row_ratio = cm[i].sum() / n      # true fraction of class i
        col_ratio = cm[:, i].sum() / n   # predicted fraction of class i
        expected += row_ratio * col_ratio

    if expected >= 1.0:
        return 1.0
    return float((observed - expected) / (1.0 - expected))
