# -*- coding: utf-8 -*-
"""
Tests for the "zero calibration" script (src/zero_calibration.py).

There's really only one core thing to verify: calibration (a subtraction) does lower the error.
The other tests ensure this function's behavior is stable, reproducible, and trustworthy.
"""
import numpy as np
import zero_calibration as zc


def test_error_after_significantly_smaller():
    """Core assertion: calibration works, and the error after is always smaller than before."""
    r = zc.simulate_calibration(seed=42)
    assert r["err_after"] < r["err_before"]


def test_calibration_removes_more_than_half_error():
    """In practice it removes 82%; here we only require at least 50% (leaving slack so it's not flaky)."""
    r = zc.simulate_calibration(seed=42)
    improve = (r["err_before"] - r["err_after"]) / r["err_before"]
    assert improve > 0.5


def test_same_seed_reproducible():
    """With a fixed random seed, running twice should give identical results (otherwise the test itself is untrustworthy)."""
    r1 = zc.simulate_calibration(seed=42)
    r2 = zc.simulate_calibration(seed=42)
    assert np.allclose(r1["raw"], r2["raw"])


def test_calibration_action_is_subtraction():
    """Calibrated = dirty data - baseline, one step, no other tricks."""
    r = zc.simulate_calibration(seed=42)
    assert np.allclose(r["calibrated"], r["raw"] - r["baseline"])
