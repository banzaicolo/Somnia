#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tests for hysteresis compensation — verify that the whole "auto-detect direction + fix it" action is correct.

Five things are tested:
  1. Direction detection: loading/unloading/unchanged, are the three cases judged correctly
  2. Deadband: a change below the threshold shouldn't trigger a "direction reversal"
  3. Lie-down/get-up simulation: correct shape, loading segment high, unloading segment low
  4. Auto compensation: with no noise it should exactly recover the true pressure
  5. Wrong direction: makes the data worse (proving "judging direction" is the soul of hysteresis compensation)
"""

import numpy as np
import pytest

import calibration as cal
import pressure_simulator as ps
from sensor_config import (
    GRID_W, GRID_H, BODY_PARTS, HYSTERESIS, make_ideal_pressure,
)


def _ideal():
    """Build an ideal body-pressure map (ground truth) for reuse across tests."""
    return make_ideal_pressure(GRID_W, GRID_H, BODY_PARTS)


# ============================================================================
# estimate_direction: direction detection
# ============================================================================


def test_estimate_direction_three_cases():
    """Rising -> loading, unchanged -> 0, falling -> unloading."""
    prev = np.array([[0.0, 5.0, 8.0]])
    curr = np.array([[10.0, 5.0, 3.0]])
    d = cal.estimate_direction(prev, curr, deadband=0.0)
    assert d[0, 0] == 1    # 0->10 clearly rising -> loading
    assert d[0, 1] == 0    # 5->5 unchanged -> 0
    assert d[0, 2] == -1   # 8->3 clearly falling -> unloading


def test_estimate_direction_deadband():
    """A change below the deadband threshold = not a turn; above it counts."""
    prev = np.array([[5.0]])
    curr = np.array([[5.4]])
    # change 0.4 < deadband 0.5 -> treated as unchanged
    assert cal.estimate_direction(prev, curr, deadband=0.5)[0, 0] == 0
    # change 0.4 > deadband 0.3 -> counted as rising
    assert cal.estimate_direction(prev, curr, deadband=0.3)[0, 0] == 1


# ============================================================================
# simulate_load_unload: lie-down/get-up simulation
# ============================================================================


def test_simulate_shape_and_segments():
    """Returns three items; frame count and shapes are correct."""
    ideal = _ideal()
    n = 60
    true_frames, raw_frames, weights = ps.simulate_load_unload(ideal, n_steps=n)
    assert true_frames.shape == (n, GRID_H, GRID_W)
    assert raw_frames.shape == (n, GRID_H, GRID_W)
    assert len(weights) == n


def test_simulate_loading_high_unloading_low():
    """Loading-segment reading >= truth, unloading-segment reading <= truth (hysteresis directionality)."""
    ideal = _ideal()
    n = 60
    true_frames, raw_frames, _ = ps.simulate_load_unload(ideal, n_steps=n)
    n_up = n // 2
    assert (raw_frames[:n_up] >= true_frames[:n_up]).all()   # lie down: high
    assert (raw_frames[n_up:] <= true_frames[n_up:]).all()   # get up: low


# ============================================================================
# correct_hysteresis_sequence: auto-detect direction + compensate
# ============================================================================


def test_compensation_exactly_recovers():
    """With no noise, auto-detect direction + compensate should exactly recover the true pressure."""
    ideal = _ideal()
    true_frames, raw_frames, _ = ps.simulate_load_unload(ideal, n_steps=60)
    corrected = cal.correct_hysteresis_sequence(raw_frames, HYSTERESIS, deadband=0.0)
    assert corrected.shape == raw_frames.shape
    np.testing.assert_allclose(corrected, true_frames, atol=1e-6)


def test_error_after_much_smaller_than_before():
    """Mean error over the whole run: after compensation should be near zero, before clearly larger."""
    ideal = _ideal()
    true_frames, raw_frames, _ = ps.simulate_load_unload(ideal, n_steps=60)
    corrected = cal.correct_hysteresis_sequence(raw_frames, HYSTERESIS, deadband=0.0)
    err_before = float(np.abs(raw_frames - true_frames).mean())
    err_after = float(np.abs(corrected - true_frames).mean())
    assert err_before > 1.0          # hysteresis width 8, mean error should clearly exceed 1
    assert err_after < 1e-3          # after compensation nearly zero


# ============================================================================
# Wrong direction: the soul of hysteresis compensation is "judging direction correctly"
# ============================================================================


def test_wrong_direction_makes_worse():
    """Compensating a loading as unloading doubles the error — proving the direction must not be misjudged."""
    ideal = _ideal()
    # pure loading scenario: true pressure = ideal, reading high by h/2
    raw = ideal + HYSTERESIS / 2.0
    # correct: compensating as load recovers it; here we only verify the consequence of "misjudging"
    wrong = cal.correct_hysteresis(raw, "unload", HYSTERESIS)
    err_before = float(np.abs(raw - ideal).mean())   # = h/2
    err_wrong = float(np.abs(wrong - ideal).mean())  # ≈ h
    assert err_wrong > err_before                    # the more you "fix" the worse it gets


def test_correct_direction_recovers():
    """Control: judging the direction correctly (load) exactly recovers, error ≈ 0."""
    ideal = _ideal()
    raw = ideal + HYSTERESIS / 2.0
    right = cal.correct_hysteresis(raw, "load", HYSTERESIS)
    np.testing.assert_allclose(right, ideal, atol=1e-9)
