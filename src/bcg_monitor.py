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

  ~ breathing rate — stably extracted on synthesized signals (both pure sine and
              asymmetric, harmonic-rich breathing). Real-mattress accuracy is still
              unverified (no synchronized reference data yet).
  ~ heart rate — extracted with respiration-harmonic removal: locate the breathing rate,
              knock out its harmonics that fall into the heartbeat band, then pick the
              remaining peak. Verified on asymmetric (harmonic-rich) synthesized signals:
              it recovers 72 beats/min where the naive method reads 45. Real-mattress
              accuracy is still unverified (no synchronized ECG reference yet).
  ✅ apnea — judged by "relative drop in breathing-band energy"; the logic holds,
              and it has passed verification on synthesized signals.
  ❌ blood oxygen (SpO2) — physically impossible to measure. Blood oxygen requires
             light shining through the skin (PPG, photoplethysmography); the mattress
             only has vibration signals and carries no such information.
             Do not promise this feature.

This module's verification status (stated clearly, no ambiguity):

  · Verified: breathing-rate and heart-rate extraction on synthesized signals, including
    asymmetric breathing whose harmonics leak into the heartbeat band (harmonic removal
    recovers 72 beats/min where the naive method reads 45).
  · Not verified: the error range on real mattress signals (missing synchronized ECG /
    breathing-belt reference data).
  · Known limitation: when the heart rate happens to be an integer multiple of the
    breathing rate (e.g. 60 bpm = 4 × 15 breaths/min), the heartbeat lands exactly on a
    breathing harmonic and no frequency-domain method can separate the two.

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
                 noise_std=0.05, apnea=None, resp_asymmetry=0.0, seed=42):
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
      resp_asymmetry — how asymmetric the breathing waveform is, from 0.0 to 1.0.
                     0.0 = a pure sine wave (no harmonics — the "grading your own homework"
                     idealization). 1.0 = a sawtooth-like "sharp inhale, slow exhale" shape
                     whose harmonics leak into the heartbeat band. Real breathing is
                     asymmetric, so use 1.0 to test the harmonic-removal path honestly.
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
    theta = 2 * np.pi * resp_freq * t
    if resp_asymmetry <= 0.0:
        respiration = 5.0 * np.sin(theta)
    else:
        # A sawtooth-like shape: sharp inhale, slow exhale. Its Fourier series carries
        # harmonics at 2x, 3x, 4x... the breathing rate, which is exactly what real
        # breathing does (unlike a pure sine). Blend between pure sine (0.0) and full
        # sawtooth (1.0) so callers can dial in how "real" the test is.
        saw = _sawtooth(theta)
        respiration = 5.0 * ((1.0 - resp_asymmetry) * np.sin(theta) + resp_asymmetry * saw)
    heartbeat = 0.5 * np.sin(2 * np.pi * hr_freq * t)

    # apnea: squeeze the breathing amplitude to 0 over [start, end)
    if apnea is not None:
        start, end = apnea
        mask = (t >= start) & (t < end)
        respiration = np.where(mask, 0.0, respiration)

    noise = rng.normal(0, noise_std, size=n)

    signal = baseline + respiration + heartbeat + noise
    return t, signal


def _sawtooth(theta, n_harmonics=8):
    """
    A normalized sawtooth wave in [-1, 1], built from its Fourier series.

    A sawtooth is the canonical "asymmetric" waveform: it rises slowly (or sharply) and
    snaps back, so unlike a pure sine it contains harmonics at 2x, 3x, 4x... the base
    frequency. Real breathing is asymmetric ("sharp inhale, slow exhale"), so modeling it
    as a sawtooth (instead of a sine) injects the harmonics that a naive heart-rate
    extractor will mistakenly lock onto.

    The Fourier series of a sawtooth is  sum_k (-1)^(k-1) sin(k·theta) / k.
    We keep the first n_harmonics terms (8 is plenty — higher terms are tiny) and rescale
    by 2/pi so the peak amplitude lands near 1.0.

    Parameters:
      theta        — phase angle(s) in radians (array or scalar)
      n_harmonics  — how many Fourier terms to keep

    Returns:
      sawtooth values in roughly [-1, 1]
    """
    theta = np.asarray(theta, dtype=float)
    wave = np.zeros_like(theta)
    for k in range(1, n_harmonics + 1):
        wave += ((-1) ** (k - 1)) * np.sin(k * theta) / k
    return wave * (2.0 / np.pi)


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


def estimate_heart_rate_naive(signal, fs=FS):
    """
    Estimate the heart rate the "naive" way: find the strongest frequency in the
    heartbeat band (0.7~3.0 Hz) and ×60.

    This is kept as a separate function on purpose — it is the buggy baseline. When the
    breathing waveform is asymmetric, its harmonics fall into the heartbeat band and are
    often stronger than the real heartbeat, so this function locks onto a breathing echo
    (e.g. reports ~45 beats/min when the true rate is 72). Compare it against
    estimate_heart_rate(), which removes those echoes first.

    Parameters:
      signal — mattress pressure signal
      fs     — sampling rate

    Returns:
      heart rate (beats/min, float)
    """
    peak_hz = _dominant_freq(signal, fs, HR_LOW, HR_HIGH)
    return peak_hz * 60.0


def estimate_heart_rate(signal, fs=FS):
    """
    Estimate the heart rate (beats/min) with respiration-harmonic removal.

    Why the extra step: real breathing is asymmetric, so its spectrum carries "echoes"
    (harmonics) at 2x, 3x, 4x... the breathing rate. Those echoes land inside the
    heartbeat band, and because breathing is much stronger than the heartbeat, they can
    out-shout the true heartbeat peak. The naive method then locks onto a breathing echo
    instead of the heart.

    Fix: first locate the breathing rate reliably (its peak is large, so this is easy and
    trustworthy). Then check whether the naive heartbeat-band peak actually sits on a
    respiration harmonic (k × breathing rate, k ≥ 2). Only if it does do we knock out the
    in-band harmonics and pick the strongest survivor; if the naive peak is not on a
    harmonic, it is the real heartbeat and is returned directly.

    Parameters:
      signal — mattress pressure signal
      fs     — sampling rate

    Returns:
      heart rate (beats/min, float)
    """
    # 1. Locate the breathing rate. Its peak is large, so this estimate is trustworthy.
    f_resp = _dominant_freq(signal, fs, RESP_LOW, RESP_HIGH)
    if f_resp <= 0.0:
        # No breathing detected — nothing to remove, fall back to the naive lookup.
        return estimate_heart_rate_naive(signal, fs)

    n = len(signal)
    df = fs / n  # frequency resolution of this FFT
    f_naive = _dominant_freq(signal, fs, HR_LOW, HR_HIGH)

    # 2. Is the naive peak actually a respiration harmonic? If it does NOT sit on
    #    k * f_resp (k >= 2), it is the real heartbeat and there is nothing to remove —
    #    return it directly. Only when the naive peak sits exactly on a harmonic has the
    #    naive method locked onto a breathing echo, so we must remove the echoes and look
    #    again for the heartbeat underneath.
    k_nearest = int(round(f_naive / f_resp))
    if k_nearest < 2 or abs(f_naive - k_nearest * f_resp) >= 0.5 * df:
        return f_naive * 60.0

    # 3. Remove every in-band respiration harmonic, then find the strongest survivor.
    spectrum = np.abs(np.fft.rfft(signal))
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)
    cleaned = spectrum.copy()
    k = 2
    while k * f_resp <= HR_HIGH:
        f_harm = k * f_resp
        if f_harm >= HR_LOW:
            cleaned[np.abs(freqs - f_harm) < 1.0 * df] = 0.0
        k += 1

    band = (freqs >= HR_LOW) & (freqs <= HR_HIGH)
    if cleaned[band].max() <= 0.0:
        # We removed everything in the band — fall back rather than report 0.
        return estimate_heart_rate_naive(signal, fs)
    # Guard: if removing the harmonics collapsed the band's peak almost to nothing, the
    # true heartbeat was almost certainly sitting exactly on a respiration harmonic
    # (heart rate = an integer multiple of the breathing rate) and got removed with it.
    # In that degenerate case no frequency-domain method can separate the two; fall back
    # to the naive answer instead of reporting a stray spectral-leakage bin.
    if cleaned[band].max() < 0.05 * spectrum[band].max():
        return estimate_heart_rate_naive(signal, fs)
    return float(freqs[band][np.argmax(cleaned[band])]) * 60.0


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
