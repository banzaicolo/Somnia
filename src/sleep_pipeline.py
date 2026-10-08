#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=============================================================================
sleep_pipeline.py — wire up the full loop "signal → features → staging"
=============================================================================

What problem does this solve?

The previous two modules each do one thing, but they aren't connected:

  · bcg_monitor.py — digs breathing rate and heart rate out of the mattress
                     pressure waveform (splitting by frequency with FFT)
  · sleep_staging.py — sleep staging, but its 5 features (movement / breathing rate /
                       heart rate / breathing variability / HR variability) are
                       synthesized "out of thin air", equivalent to pretending the
                       sensor already computed the numbers and handed them straight
                       to the stager.

Reality is not that cheap — a real mattress only gives you a "pressure waveform",
and you must dig those 5 numbers out of the waveform yourself. This module wires
up that chain:

    raw pressure waveform ──FFT──▶ 5 features ──neural net──▶ sleep stage
    (a full night)        (movement/breathing/HR/variability) (wake/light/deep/REM)

Once this is done, the whole project is a real runnable pipeline, not a handful
of isolated parts.

How is it wired up? Three steps.

  Step 1 · synthesize the full-night signal (simulate_night_signal)
      — first let the sleep-staging module "lay out" a full night's stage order
        (light → deep → REM ...), then synthesize the corresponding pressure
        waveform mathematically according to each stage's physiology (fast
        breathing when awake, slow in deep sleep, erratic heart rate in REM).
        This way the waveform and stage labels are "paired up".

  Step 2 · dig features out of the waveform (extract_epoch_features)
      — slice the full-night waveform into 30 s segments, then per segment use
        FFT (Fast Fourier Transform):
          movement   → energy of the lowest band (tossing/repositioning, slower than breathing)
          breathing rate → the dominant frequency in the breathing band
          heart rate → the dominant frequency in the heartbeat band
          breathing/HR variability → a sliding window within the segment, to see
        how steady the dominant frequency is. This yields an (N, 5) feature matrix,
        column-aligned with the sleep-staging module's 5 columns.

  Step 3 · train + stage (extract_dataset + stage_from_signal)
      — use the dug-out features to train the same hand-written neural network,
        then stage a full night.

Honest note: the two "variability" features are engineering approximations.

Breathing variability and HR variability (a simplified HRV) are, in a real system,
measured "beat by beat" — precise to every heartbeat / breath interval, which
requires the more complex peak-detection pipeline. To keep complexity in check,
this module approximates them with "the fluctuation of the dominant frequency
across sliding windows within a 30 s segment". The direction is right (deep sleep
is steadiest, REM is most erratic), but precision is limited. This is stated
honestly here, without overselling.

Usage:

    import sleep_pipeline as sp
    signal, y, stages = sp.simulate_night_signal(480)   # synthesize a full-night waveform
    X = sp.extract_epoch_features(signal)               # dig 5 features out of the waveform
    # X: (480, 5), column order same as sleep_staging: movement/breathing rate/heart rate/breathing variability/HR variability

=============================================================================
"""

import numpy as np

# Reuse the BCG module's signal processing (breathing/heart rate extraction, band energy, dominant peak)
import bcg_monitor as bcg
# Reuse the sleep-staging module's stage definitions, physiological parameters, feature standardization
from sleep_staging import (
    STAGES, STAGE_CENTERS, EPOCH_SEC, standardize, _build_sleep_structure,
)


# ============================================================================
# 1. Signal synthesis parameters (all adjustable knobs gathered here)
# ============================================================================

FS = bcg.FS                    # sampling rate: 50 points per second (consistent with the BCG module)
RESP_AMP = 5.0                 # breathing waveform amplitude (large chest rise, big amplitude)
HR_AMP = 0.5                   # heartbeat waveform amplitude (micro-vibration, about 1/10 of breathing)
BASELINE = 100.0               # static body pressure (DC), representing the person's weight

# Movement: tossing and repositioning are low-frequency swings "slower than breathing"
MOV_FREQ = 0.05                # movement frequency (Hz) = one swing every 20 s
MOV_AMP_SCALE = 10.0           # scale the "normalized movement amplitude (0~0.7)" up to waveform scale
                               # wake movement 0.70→7.0 (obvious tossing), deep sleep 0.03→0.3 (almost still)

# The "movement band" (Hz) used for movement extraction — lower than the breathing band (0.1~0.6), avoiding breathing
MOV_LOW = 0.02
MOV_HIGH = 0.09

# Sliding window (s) for variability extraction: within a 30 s segment, 15 s window, 5 s hop
VARIABILITY_WIN_SEC = 15.0
VARIABILITY_HOP_SEC = 5.0


# ============================================================================
# 2. Synthesize a "full night" of pressure waveform (with stage labels)
# ============================================================================

def _synthesize_epoch(resp_rate, hr_rate, resp_var, hr_var, mov_amp, fs, rng):
    """
    Synthesize the pressure waveform of "one 30 s segment".

    This segment = static body pressure + breathing + heartbeat + movement + noise,
    the same model as the BCG module, with two extra "sleep staging" twists:

      1. breathing rate and heart rate "drift slowly" — resp_var / hr_var control the drift size.
         Deep sleep barely drifts (steady), REM/wake drift a lot (erratic). This is what
         "variability" really means.
      2. movement strength is controlled by mov_amp — big amplitude when tossing awake,
         almost still in deep sleep.

    The drift is done by "phase integration" (not hard segmenting): the instantaneous
    frequency changes smoothly over time, so the phase accumulates smoothly and the
    waveform doesn't jump at segment boundaries.

    Parameters:
      resp_rate, hr_rate — breathing/heart rate centers (per minute)
      resp_var, hr_var   — breathing/heart rate drift size (per minute); larger = less steady
      mov_amp            — movement amplitude (waveform scale)
      fs                 — sampling rate
      rng                — random number generator (for reproducibility)

    Returns: a length-(n,) pressure waveform
    """
    n = int(EPOCH_SEC * fs)
    t = np.arange(n) / fs

    # instantaneous breathing rate: center + slow drift
    # (0.04 Hz modulation = about one swing every 25 s)
    resp_inst = resp_rate + resp_var * np.sin(2 * np.pi * 0.04 * t
                                              + rng.uniform(0, 2 * np.pi))
    # instantaneous heart rate: center + slow drift (0.06 Hz)
    hr_inst = hr_rate + hr_var * np.sin(2 * np.pi * 0.06 * t
                                        + rng.uniform(0, 2 * np.pi))

    # phase integration: frequency (per min) ÷ 60 = Hz, cumulative sum then ×2π gives continuous phase
    resp_phase = 2 * np.pi * np.cumsum(resp_inst / 60.0) / fs
    hr_phase = 2 * np.pi * np.cumsum(hr_inst / 60.0) / fs

    respiration = RESP_AMP * np.sin(resp_phase)
    heartbeat = HR_AMP * np.sin(hr_phase)
    movement = mov_amp * np.sin(2 * np.pi * MOV_FREQ * t
                                + rng.uniform(0, 2 * np.pi))
    noise = rng.normal(0, 0.05, n)

    return BASELINE + respiration + heartbeat + movement + noise


def simulate_night_signal(n_epochs=480, fs=FS, seed=42):
    """
    Synthesize a "full night" of raw pressure waveform, with each 30 s segment's true stage label.

    This is the key first step of wiring up: first let the sleep-staging module lay out
    the stage order, then synthesize the waveform per stage, so "waveform" and "stage
    label" are naturally paired — exactly what training and supervision need.

    Parameters:
      n_epochs — how many 30 s segments this night has (8 hours = 960)
      fs       — sampling rate
      seed     — random seed; fixing it makes results reproducible

    Returns:
      signal — (n_epochs × 30 × fs,) one-dimensional pressure waveform
      y      — (n_epochs,) each segment's true stage index (0=wake/1=light/2=deep/3=rem)
      stages — (n_epochs,) list of stage names (for easy debugging)
    """
    rng = np.random.default_rng(seed)

    # 1) lay out the full-night stage order first
    #    (reuse the sleep-staging module: more deep sleep in the first half, more REM in the second)
    stage_names = _build_sleep_structure(n_epochs, rng)

    # 2) synthesize the waveform segment by segment, per stage
    n_per = int(EPOCH_SEC * fs)
    signal = np.zeros(n_epochs * n_per)
    for i, stage in enumerate(stage_names):
        c = STAGE_CENTERS[stage]          # [movement, breathing rate, heart rate, breathing variability, HR variability]
        mov_amp = c[0] * MOV_AMP_SCALE    # normalized movement → waveform amplitude
        seg = _synthesize_epoch(c[1], c[2], c[3], c[4], mov_amp, fs, rng)
        signal[i * n_per:(i + 1) * n_per] = seg

    y = np.array([STAGES.index(s) for s in stage_names], dtype=int)
    return signal, y, stage_names


# ============================================================================
# 3. Dig 5 features out of the waveform (the key second step of wiring up)
# ============================================================================

def _rate_variability(seg, fs, low, high):
    """
    Estimate "the fluctuation of a band's dominant frequency", as an engineering
    approximation of breathing variability / HR variability.

    How: slice the segment into overlapping windows (15 s window, 5 s hop), find
    the dominant frequency in each window with FFT, then take the standard deviation
    of those dominant peaks. In deep sleep the peak stays put (small std); in
    REM/wake the peak speeds up and slows down (large std).

    Honest note: this is an engineering approximation (see the module header), not
    the beat-by-beat true HRV.

    Parameters:
      seg       — a signal segment (one 30 s segment)
      fs        — sampling rate
      low, high — the target band (breathing band or heartbeat band)

    Returns:
      variability value (per minute). Returns 0 if there are fewer than 2 windows.
    """
    win = int(VARIABILITY_WIN_SEC * fs)
    hop = int(VARIABILITY_HOP_SEC * fs)
    if win <= 0 or win > len(seg):
        return 0.0

    rates = []
    for start in range(0, len(seg) - win + 1, hop):
        window = seg[start:start + win]
        peak_hz = bcg.dominant_freq(window, fs, low, high)
        rates.append(peak_hz * 60.0)     # Hz → per minute

    if len(rates) < 2:
        return 0.0
    return float(np.std(rates))


def extract_epoch_features(signal, fs=FS, epoch_sec=EPOCH_SEC):
    """
    Slice a full-night raw waveform into 30 s segments and dig 5 features out of each.

    The 5 columns are exactly aligned with the sleep-staging module
    (so they can plug directly into its classifier):
      [0] movement amplitude — energy of the movement band (lower frequency than breathing)
      [1] breathing rate — dominant frequency of the breathing band × 60
      [2] heart rate — dominant frequency of the heartbeat band × 60
      [3] breathing variability — fluctuation of the dominant frequency across windows
      [4] HR variability — fluctuation of the dominant frequency across windows

    Key detail: remove DC (subtract the mean) before extracting features. The movement
    band is very close to 0 Hz; if the DC component (body weight) isn't removed, it
    leaks in and contaminates the movement measurement.

    Parameters:
      signal    — a full-night waveform (one-dimensional)
      fs        — sampling rate
      epoch_sec — segment length (default 30 s, the AASM standard staging duration)

    Returns:
      X — (n_epochs, 5) feature matrix
    """
    n_per = int(epoch_sec * fs)
    n_epochs = len(signal) // n_per

    X = np.zeros((n_epochs, 5))
    for i in range(n_epochs):
        seg = signal[i * n_per:(i + 1) * n_per]
        seg = seg - seg.mean()          # remove DC, so body weight doesn't contaminate low-frequency measurement

        X[i, 0] = bcg.band_energy(seg, fs, MOV_LOW, MOV_HIGH)     # movement
        X[i, 1] = bcg.estimate_breathing_rate(seg, fs)            # breathing rate
        X[i, 2] = bcg.estimate_heart_rate(seg, fs)                # heart rate
        X[i, 3] = _rate_variability(seg, fs, bcg.RESP_LOW, bcg.RESP_HIGH)
        X[i, 4] = _rate_variability(seg, fs, bcg.HR_LOW, bcg.HR_HIGH)

    return X


# ============================================================================
# 4. Build multi-night training set + end-to-end staging (step 3)
# ============================================================================

def extract_dataset(n_nights=6, n_epochs=240, fs=FS, seed=42):
    """
    Synthesize multiple nights, run "waveform → features" per night, and combine
    them into a large dataset for training a classifier.

    Parameters:
      n_nights — how many nights to synthesize
      n_epochs — how many segments per night
      fs       — sampling rate
      seed     — random seed (seed+i per night, simulating "different nights")

    Returns:
      X — (total segments, 5) feature matrix
      y — (total segments,) stage index
    """
    X_list, y_list = [], []
    for i in range(n_nights):
        signal, y, _ = simulate_night_signal(n_epochs=n_epochs, fs=fs,
                                             seed=seed + i)
        X = extract_epoch_features(signal, fs=fs)
        X_list.append(X)
        y_list.append(y)
    return np.vstack(X_list), np.concatenate(y_list)


def stage_from_signal(signal, model, mean, std, fs=FS):
    """
    End-to-end staging: raw waveform → features → standardization → model prediction.

    Parameters:
      signal — a full-night waveform
      model  — a trained classifier (TinyMLP)
      mean, std — standardization parameters from the training set (same ruler)
      fs     — sampling rate

    Returns:
      preds — (n_epochs,) predicted stage indices
    """
    X = extract_epoch_features(signal, fs=fs)
    Xs, _, _ = standardize(X, mean, std)
    return model.predict(Xs)
