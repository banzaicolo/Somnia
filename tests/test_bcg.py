#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tests for the BCG breathing/heartbeat monitoring (bcg_monitor) — validating the "synthesize + extract + apnea detection" logic.

Focuses on five things:
  1. synthesized signal shape (duration × sampling rate)
  2. breathing-rate extraction: exact when aligned (integer breaths/min), within tolerance when not
  3. heart-rate extraction: exact when aligned, still accurate with noise
  4. apnea detection: no false positives on normal segments, pauses caught
  5. all-zero signal doesn't divide by zero
"""

import numpy as np

import bcg_monitor as bcg


# ============================================================================
# simulate_bcg: synthesized signal
# ============================================================================


def test_simulate_shape():
    """10 s × 50 Hz = 500 points; time axis and signal are the same length."""
    t, sig = bcg.simulate_bcg(duration_sec=10, fs=50)
    assert len(t) == 500
    assert len(sig) == 500


# ============================================================================
# estimate_breathing_rate / estimate_heart_rate: extraction
# ============================================================================


def test_extract_breathing_rate_aligned_exact():
    """No noise + integer breathing rate (15 breaths/min), extraction should be almost exact."""
    _, sig = bcg.simulate_bcg(duration_sec=60, resp_rate=15, heart_rate=72,
                              noise_std=0)
    est = bcg.estimate_breathing_rate(sig)
    assert abs(est - 15.0) < 0.5


def test_extract_heart_rate_aligned_exact():
    """No noise + integer heart rate (72 beats/min), extraction should be almost exact."""
    _, sig = bcg.simulate_bcg(duration_sec=60, resp_rate=15, heart_rate=72,
                              noise_std=0)
    est = bcg.estimate_heart_rate(sig)
    assert abs(est - 72.0) < 0.5


def test_extract_noisy_still_accurate():
    """With noise, breathing rate and heart rate can still be extracted (error < 1 /min)."""
    _, sig = bcg.simulate_bcg(duration_sec=60, resp_rate=15, heart_rate=72,
                              noise_std=0.1)
    assert abs(bcg.estimate_breathing_rate(sig) - 15.0) < 1.0
    assert abs(bcg.estimate_heart_rate(sig) - 72.0) < 1.0


def test_extract_non_integer_breathing_rate_within_tolerance():
    """A non-integer breathing rate (14.5 breaths/min) leaks spectrally, but the dominant peak stays near (error < 1.5)."""
    _, sig = bcg.simulate_bcg(duration_sec=60, resp_rate=14.5, heart_rate=72,
                              noise_std=0)
    est = bcg.estimate_breathing_rate(sig)
    assert abs(est - 14.5) < 1.5


def test_extract_different_heart_rate_accurate():
    """Try a different heart rate (68 beats/min) to verify it's not just lucky at 72."""
    _, sig = bcg.simulate_bcg(duration_sec=60, resp_rate=15, heart_rate=68,
                              noise_std=0)
    est = bcg.estimate_heart_rate(sig)
    assert abs(est - 68.0) < 0.5


# ============================================================================
# detect_apnea: breathing-pause detection
# ============================================================================


def test_apnea_detection_normal_no_false_positive():
    """Normal breathing throughout should not falsely report any pause."""
    _, sig = bcg.simulate_bcg(duration_sec=30, resp_rate=15, heart_rate=72,
                              noise_std=0.05)
    flags, _ = bcg.detect_apnea(sig, window_sec=10)
    assert flags == [False, False, False]


def test_apnea_detection_catches_pause():
    """A 10-second pause in the middle is caught exactly."""
    _, sig = bcg.simulate_bcg(duration_sec=30, resp_rate=15, heart_rate=72,
                              noise_std=0.05, apnea=(10, 20))
    flags, _ = bcg.detect_apnea(sig, window_sec=10)
    # 30 s → 3 windows: [0,10) normal / [10,20) pause / [20,30) normal
    assert flags == [False, True, False]


# ============================================================================
# Edge case: all-zero signal
# ============================================================================


def test_all_zero_signal_no_div_by_zero():
    """Nobody home (all-zero signal): extraction returns 0 and apnea detection doesn't crash."""
    sig = np.zeros(3000)   # 60 s × 50 Hz
    assert bcg.estimate_breathing_rate(sig) == 0.0
    assert bcg.estimate_heart_rate(sig) == 0.0
    flags, _ = bcg.detect_apnea(sig)
    assert flags == [False] * 6   # 6 windows of 10 s, all-zero energy, not flagged as pause
