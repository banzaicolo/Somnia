#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tests for the calibration toolchain: each move must be verifiable on its own as "really fixed it"."""

import numpy as np
import pytest

from calibration import (
    forward,
    correct_crosstalk, correct_zero, correct_gain,
    correct_temperature_offset, correct_temperature_gain,
    correct_hysteresis,
    estimate_baseline, estimate_gain,
    calibrate, mean_abs_error, improvement,
)


class TestForward:
    def test_no_imperfections_returns_ideal(self):
        """With no faults, the reading should equal the true pressure."""
        ideal = np.array([[1.0, 2.0], [3.0, 4.0]])
        raw = forward(ideal, gain=np.ones_like(ideal),
                      offset=np.zeros_like(ideal), c=0.0, noise=0.0)
        np.testing.assert_allclose(raw, ideal)

    def test_offset_shifts_baseline(self):
        """Zero offset lifts the whole reading."""
        ideal = np.zeros((2, 2))
        offset = np.full((2, 2), 5.0)
        raw = forward(ideal, np.ones((2, 2)), offset)
        np.testing.assert_allclose(raw, 5.0)

    def test_gain_scales(self):
        """Sensitivity amplifies the reading."""
        ideal = np.full((2, 2), 10.0)
        gain = np.full((2, 2), 1.5)
        raw = forward(ideal, gain, np.zeros((2, 2)))
        np.testing.assert_allclose(raw, 15.0)


class TestCorrectZero:
    def test_subtract(self):
        raw = np.array([[30.0, 20.0], [10.0, 0.0]])
        baseline = np.array([[5.0, 5.0], [5.0, 5.0]])
        out = correct_zero(raw, baseline)
        np.testing.assert_allclose(out, [[25, 15], [5, -5]])


class TestCorrectGain:
    def test_divide(self):
        raw = np.array([[20.0, 30.0]])
        gain = np.array([[2.0, 3.0]])
        out = correct_gain(raw, gain)
        np.testing.assert_allclose(out, [[10.0, 10.0]])

    def test_zero_gain_protected(self):
        """When gain is 0, must not divide by zero; skip safely."""
        raw = np.array([[10.0, 10.0]])
        gain = np.array([[0.0, 2.0]])
        out = correct_gain(raw, gain)
        assert np.all(np.isfinite(out))
        np.testing.assert_allclose(out, [[10.0, 5.0]])


class TestCorrectCrosstalk:
    def test_roundtrip(self):
        """After crosstalk then correction, the original map should be roughly recovered."""
        rng = np.random.default_rng(0)
        x = rng.uniform(0, 100, size=(8, 12))
        c = 0.1
        # forward crosstalk
        padded = np.pad(x, 1, mode="edge")
        neighbors = (padded[:-2, 1:-1] + padded[2:, 1:-1] +
                     padded[1:-1, :-2] + padded[1:-1, 2:]) / 4.0
        raw = (1 - c) * x + c * neighbors
        # reverse correction
        out = correct_crosstalk(raw, c)
        np.testing.assert_allclose(out, x, atol=1e-6)


class TestTemperature:
    def test_offset_compensation(self):
        raw = np.array([[40.0]])
        tco = np.array([[2.0]])
        # going from 40 degrees back to the 25-degree reference, drifted 15 degrees × 2 = 30, should be subtracted
        out = correct_temperature_offset(raw, temp=40.0, t_ref=25.0, tco_map=tco)
        np.testing.assert_allclose(out, [[10.0]])

    def test_gain_compensation(self):
        raw = np.array([[110.0]])
        tcs = np.array([[0.01]])
        # gain drifted 15 degrees × 1% = 15%; the reading 110 was amplified by 1.15, divide by it
        out = correct_temperature_gain(raw, temp=40.0, t_ref=25.0, tcs_map=tcs)
        np.testing.assert_allclose(out, [[110.0 / 1.15]])


class TestHysteresis:
    def test_load_roundtrip(self):
        p = np.linspace(0, 100, 50)
        h = 8.0
        load = p + h / 2.0
        out = correct_hysteresis(load, "load", h)
        np.testing.assert_allclose(out, p, atol=1e-9)

    def test_unload_roundtrip(self):
        p = np.linspace(0, 100, 50)
        h = 8.0
        unload = p - h / 2.0
        out = correct_hysteresis(unload, "unload", h)
        np.testing.assert_allclose(out, p, atol=1e-9)

    def test_bad_direction_raises(self):
        with pytest.raises(ValueError):
            correct_hysteresis(np.array([1.0]), "sideways", 8.0)


class TestEstimate:
    def test_estimate_gain_recovers_true_gain(self):
        """Gain calibration should approximately recover the true sensitivity."""
        rng = np.random.default_rng(1)
        gain_true = rng.normal(1.0, 0.15, size=(8, 12))
        offset = rng.uniform(0, 8, size=(8, 12))
        c = 0.1
        P0 = 50.0

        ideal_ref = np.full((8, 12), P0)
        baseline = estimate_baseline(offset, c=c, noise_std=0.0, rng=rng)
        raw_ref = forward(ideal_ref, gain_true, offset, c, 0.0)
        gain_est = estimate_gain(ideal_ref, raw_ref, baseline, c=c)

        np.testing.assert_allclose(gain_est, gain_true, atol=0.2)


class TestCalibratePipeline:
    def test_pipeline_reduces_error(self):
        """Full pipeline: after calibration the dirty data's error should drop significantly."""
        rng = np.random.default_rng(2)
        H, W = 8, 12
        c = 0.1
        # Keep noise small: calibration removes "systematic error" (gain/offset/crosstalk),
        # not random noise. With too much noise the improvement is "diluted" and hard to see.
        noise_std = 0.5

        # build a "human-shaped" ideal map (a few Gaussian bumps)
        yy, xx = np.mgrid[0:H, 0:W]
        ideal = (80 * np.exp(-((xx - 6) ** 2 / 8 + (yy - 5) ** 2 / 4)) +
                 40 * np.exp(-((xx - 6) ** 2 / 6 + (yy - 1) ** 2 / 2)))

        gain_true = rng.normal(1.0, 0.15, size=(H, W))
        offset = rng.uniform(0, 8, size=(H, W))

        raw = forward(ideal, gain_true, offset, c,
                      rng.normal(0, noise_std, size=(H, W)))

        # calibration actions
        baseline = estimate_baseline(offset, c=c, noise_std=noise_std, rng=rng)
        ideal_ref = np.full((H, W), 50.0)
        raw_ref = forward(ideal_ref, gain_true, offset, c,
                          rng.normal(0, noise_std, size=(H, W)))
        gain_est = estimate_gain(ideal_ref, raw_ref, baseline, c=c)

        cleaned = calibrate(raw, baseline=baseline, gain_map=gain_est, c=c)

        err_raw = mean_abs_error(raw, ideal)
        err_clean = mean_abs_error(cleaned, ideal)
        assert improvement(err_raw, err_clean) > 50  # shrink by at least half


class TestMetrics:
    def test_mean_abs_error_zero_when_equal(self):
        assert mean_abs_error(np.ones((3, 3)), np.ones((3, 3))) == 0

    def test_improvement(self):
        assert improvement(100, 20) == 80.0
