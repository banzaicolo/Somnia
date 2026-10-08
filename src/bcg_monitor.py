#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=============================================================================
bcg_monitor.py — breathing/heartbeat monitoring (BCG, ballistocardiography)
=============================================================================

What problem does this solve?

When a person lies in bed, every heartbeat and every breath produces extremely
tiny vibrations that travel through the mattress to the pressure sensor underneath.
BCG (ballistocardiography) is "using these micro-vibrations to measure breathing
and heartbeat without any contact".

Unlike a watch: a watch must touch the skin and use light to measure blood flow;
BCG requires wearing nothing — just lie down, and the sensor under the mattress
"hears" your breathing and heartbeat.

What can and can't it measure? (the honest boundary)

  ~ breathing rate — can be stably extracted on synthesized signals, but real-world
              performance has NOT been verified. Only pure mathematically-generated
              sine signals have been tested so far, which is "grading your own
              homework" and must not be treated as evidence of real performance.
  ~ heart rate — can be extracted on synthesized signals, but in real signals the
              "echo" of breathing (harmonics) falls into the heartbeat band and
              biases the heart rate. Fixing this requires harmonic removal first,
              or counting heartbeat peaks directly in the time domain (neither done yet).
  ✅ apnea — judged by "relative drop in breathing-band energy"; the logic holds,
              and it has passed verification on synthesized signals.
  ❌ blood oxygen (SpO2) — physically impossible to measure. Blood oxygen requires
             light shining through the skin (PPG, photoplethysmography); the mattress
             only has vibration signals and carries no such information.
             Do not promise this feature.

This module's verification status (stated clearly, no ambiguity):

  · Verified: the "split by frequency" logic is correct on synthesized signals
  · Not verified: the error range on real mattress signals (missing synchronized ECG / breathing-belt reference data)
  · Known failure scenario: when the breathing waveform has harmonics, the heart rate
    can lock onto the 3rd harmonic of breathing

Core principle: one signal contains three mixed "rhythms"

The pressure signal the mattress reads is a superposition of several things:

  1. static body pressure (DC) — the "base weight" of the person lying down, a constant; just remove it.
  2. breathing fluctuation (low frequency) — the chest rising and falling makes the pressure slowly oscillate.
     Low frequency: adults at rest about 12~20 breaths/min = 0.2~0.33 Hz (hertz, cycles per second).
  3. heartbeat fluctuation (high frequency) — the tiny vibration of the heart pumping blood.
     High frequency: about 60~100 beats/min = 1~1.67 Hz.
  4. noise — the sensor's own noise floor, like TV static.

To "dig" breathing rate and heart rate out, the trick is: **split the signal by frequency**.
Breathing is in the low band, heartbeat in the high band — find each in its own band
without interfering. This "split by frequency" tool is called FFT (Fast Fourier
Transform) — it turns a waveform into a table of "how much energy each frequency has".

Usage:

    import bcg_monitor as bcg
    t, signal = bcg.simulate_bcg()               # synthesize a mattress signal segment
    resp = bcg.estimate_breathing_rate(signal)   # breathing rate (breaths/min)
    hr   = bcg.estimate_heart_rate(signal)       # heart rate (beats/min)

=============================================================================
"""

import numpy as np


# ============================================================================
# Monitoring parameters (business parameters; to tune, change only here)
# ============================================================================

FS = 50        # sampling rate: 50 points per second. Real BCG systems commonly use 50~1000 Hz.
               # Higher is finer but more compute; 50 is enough for demo and tests.

# Breathing and heartbeat each occupy a "band" (in Hz, hertz = cycles per second).
# The band ranges must cover all normal cases without the two bands overlapping:
RESP_LOW = 0.1    # breathing band lower bound: 0.1 Hz = 6 breaths/min (slow breathing)
RESP_HIGH = 0.6   # breathing band upper bound: 0.6 Hz = 36 breaths/min (fast breathing, infant)
HR_LOW = 0.7      # heartbeat band lower bound: 0.7 Hz = 42 beats/min (athlete resting heart rate)
HR_HIGH = 3.0     # heartbeat band upper bound: 3.0 Hz = 180 beats/min (infant, exercise)


# ============================================================================
# 1. Synthesize a signal: mathematically "act out" a mattress pressure signal (develop without hardware)
# ============================================================================

def simulate_bcg(duration_sec=60.0, fs=FS, resp_rate=15.0, heart_rate=72.0,
                 noise_std=0.05, apnea=None, seed=42):
    """
    Synthesize a mattress pressure signal: static body pressure + breathing + heartbeat + noise.

    A real BCG signal is exactly these components superimposed. Here we "act them out"
    mathematically as a ground truth, to develop and validate the extraction algorithm
    (same idea as the calibration approach: you need a ground truth first before you can
    judge whether the algorithm digs accurately).

    Parameters:
      duration_sec — signal duration (seconds)
      fs           — sampling rate (points per second)
      resp_rate    — true breathing rate (breaths/min), the "ground truth"
      heart_rate   — true heart rate (beats/min), also the "ground truth"
      noise_std    — noise strength (larger = dirtier)
      apnea        — optional (start_sec, end_sec), marking a "breathing pause" interval
                     (breathing amplitude drops to 0, but the heartbeat keeps going —
                     apnea ≠ cardiac arrest). Default None = normal breathing throughout.
      seed         — random seed; fix it and results are the same every time, for easy comparison

    Returns:
      t      — time axis (seconds)
      signal — the synthesized pressure signal (one-dimensional array)
    """
    rng = np.random.default_rng(seed)
    n = int(duration_sec * fs)
    t = np.arange(n) / fs

    resp_freq = resp_rate / 60.0   # breathing: breaths/min → Hz (cycles per second)
    hr_freq = heart_rate / 60.0    # heartbeat: beats/min → Hz

    baseline = 100.0               # static body pressure (DC), representing the person's weight
    # breathing has a large amplitude (obvious chest rise), heartbeat a small one (about 1/10 of breathing)
    respiration = 5.0 * np.sin(2 * np.pi * resp_freq * t)
    heartbeat = 0.5 * np.sin(2 * np.pi * hr_freq * t)

    # apnea: squeeze the breathing amplitude to 0 over [start, end)
    if apnea is not None:
        start, end = apnea
        mask = (t >= start) & (t < end)
        respiration = np.where(mask, 0.0, respiration)

    noise = rng.normal(0, noise_std, size=n)

    signal = baseline + respiration + heartbeat + noise
    return t, signal


# ============================================================================
# 2. Internal utility: use FFT to find "the strongest frequency in a band"
# ============================================================================

def _dominant_freq(signal, fs, low, high):
    """
    Use FFT (Fast Fourier Transform) to split the signal into "how much energy each
    frequency has", then find the strongest frequency within the [low, high] band.

    This frequency × 60 is the "count per minute" for that band.
    E.g. if the strongest frequency in the heartbeat band is 1.2 Hz, the heart rate
    is 1.2 × 60 = 72 beats/min.

    Parameters:
      signal — one-dimensional signal
      fs     — sampling rate
      low, high — target band (Hz)

    Returns:
      the dominant frequency (Hz). Returns 0 if the band has no points at all.
    """
    n = len(signal)
    spectrum = np.abs(np.fft.rfft(signal))          # energy at each frequency
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)          # each frequency in Hz
    mask = (freqs >= low) & (freqs <= high)
    if not mask.any():
        return 0.0
    band_spectrum = spectrum[mask]
    band_freqs = freqs[mask]
    # The band has no energy at all (e.g. an all-zero signal) → no signal, report 0,
    # rather than letting argmax blindly return the first point in the band
    if band_spectrum.max() <= 0:
        return 0.0
    # the frequency with the most energy in the band is the "dominant peak" we want
    return float(band_freqs[np.argmax(band_spectrum)])


def _band_energy(signal, fs, low, high):
    """Compute the total energy of the [low, high] band = the sum of all FFT magnitudes in that band."""
    n = len(signal)
    spectrum = np.abs(np.fft.rfft(signal))
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)
    mask = (freqs >= low) & (freqs <= high)
    return float(spectrum[mask].sum())


def dominant_freq(signal, fs, low, high):
    """Public "find band dominant peak", for other modules (e.g. sleep staging) to reuse.
    Internally it's _dominant_freq; it's just a public entry point without the underscore."""
    return _dominant_freq(signal, fs, low, high)


def band_energy(signal, fs, low, high):
    """Public "compute band energy", for other modules (e.g. sleep staging) to reuse.
    Internally it's _band_energy; same reason as above."""
    return _band_energy(signal, fs, low, high)


# ============================================================================
# 3. Extract breathing rate and heart rate: each finds its dominant peak in its own band
# ============================================================================

def estimate_breathing_rate(signal, fs=FS):
    """
    Estimate the breathing rate (breaths/min).

    How: breathing is low-frequency vibration, so find the strongest frequency in the
    "breathing band" (0.1~0.6 Hz), then ×60 to convert to "counts per minute".

    Parameters:
      signal — mattress pressure signal
      fs     — sampling rate

    Returns:
      breathing rate (breaths/min, float)
    """
    peak_hz = _dominant_freq(signal, fs, RESP_LOW, RESP_HIGH)
    return peak_hz * 60.0


def estimate_heart_rate(signal, fs=FS):
    """
    Estimate the heart rate (beats/min).

    How: heartbeat is high-frequency micro-vibration, so find the strongest frequency
    in the "heartbeat band" (0.7~3.0 Hz), then ×60 to convert to "counts per minute".

    Parameters:
      signal — mattress pressure signal
      fs     — sampling rate

    Returns:
      heart rate (beats/min, float)
    """
    peak_hz = _dominant_freq(signal, fs, HR_LOW, HR_HIGH)
    return peak_hz * 60.0


# ============================================================================
# 4. Apnea detection: the chest stopped moving for a stretch
# ============================================================================

def detect_apnea(signal, fs=FS, window_sec=10.0, ratio=0.25):
    """
    Detect breathing pauses (apnea).

    Principle: during normal breathing the breathing band (0.1~0.6 Hz) has ample
    energy; when breathing pauses, the chest stops moving and that band's energy
    collapses to near 0.

    How (sliding window):
      1. slice the whole signal into non-overlapping windows of window_sec;
      2. compute each window's "breathing-band energy";
      3. if energy < ratio × global max energy → flag "suspected apnea".

    Why a ratio instead of a fixed number? Because different people have very
    different breathing amplitudes; body size and sleep posture both change it.
    Using a ratio of the "relative global max" adapts naturally.

    Parameters:
      signal     — mattress pressure signal
      fs         — sampling rate
      window_sec — how long each window is (seconds). Too short = noisy, too long = insensitive; 10 s is reasonable
      ratio      — energy below this fraction of the global peak counts as a pause (default 0.25)

    Returns:
      apnea_flags — list[bool], whether each window is a suspected pause
      window_t    — list[float], each window's start time (seconds)
    """
    win = int(window_sec * fs)
    n_windows = len(signal) // win
    if n_windows == 0:
        return [], []

    energies = []
    for i in range(n_windows):
        seg = signal[i * win:(i + 1) * win]
        energies.append(_band_energy(seg, fs, RESP_LOW, RESP_HIGH))
    energies = np.array(energies)

    peak = energies.max()
    if peak <= 0:
        flags = [False] * n_windows
    else:
        flags = (energies < ratio * peak).tolist()

    window_t = [i * window_sec for i in range(n_windows)]
    return flags, window_t
