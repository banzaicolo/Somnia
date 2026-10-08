#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tests for the verdict module — verify the "number → conclusion" decision rules.

Two main things to test:
  1. Each function returns the right wording in every band (especially boundaries)
  2. The render_pose_verdict combined function returns the right number of lines and content

Why write tests for the "verdict generator"? Because it is a pile of if-conditions,
and boundary conditions are the easiest to get wrong. Also, if someone tweaks a
threshold (e.g. changes 80 to 90), the test fails immediately, reminding them
"you changed a rule — check whether that was intentional".
"""

from verdict import (
    describe_improvement,
    describe_robustness,
    describe_rescue,
    describe_recovery,
    render_pose_verdict,
)


# ============================================================================
# describe_improvement: improvement level
# ============================================================================


def test_improvement_negative():
    """Negative improvement = error rose."""
    assert "Error did not decrease but increased" in describe_improvement(-5.0)


def test_improvement_zero():
    """0% = almost no improvement."""
    assert "Almost no improvement" in describe_improvement(0.0)


def test_improvement_low_band():
    """Below 20% = almost no improvement."""
    assert "Almost no improvement" in describe_improvement(10.0)


def test_improvement_mid_band():
    """20%~50% = limited improvement."""
    assert "Limited improvement" in describe_improvement(30.0)


def test_improvement_high_band():
    """50%~80% = clear improvement."""
    assert "Clear improvement" in describe_improvement(60.0)


def test_improvement_clean():
    """Above 80% = essentially eliminated."""
    assert "Essentially eliminated" in describe_improvement(82.0)


def test_improvement_boundary_80():
    """Exactly 80% (boundary) also counts as 'essentially eliminated'."""
    assert "Essentially eliminated" in describe_improvement(80.0)


# ============================================================================
# describe_robustness: robustness
# ============================================================================


def test_robustness_very():
    """Drop < 5 = very robust."""
    assert "Very robust" in describe_robustness(2.0)


def test_robustness_fairly():
    """Drop 5~15 = fairly robust."""
    assert "Fairly robust" in describe_robustness(10.0)


def test_robustness_hurt():
    """Drop 15~30 = performance clearly degraded."""
    assert "Performance clearly degraded" in describe_robustness(20.0)


def test_robustness_collapse():
    """Drop >= 30 = performance severely degraded."""
    assert "Performance severely degraded" in describe_robustness(40.0)


def test_robustness_negative_treated_as_zero():
    """Negative drop (dirty data scoring higher) treated as 0 = very robust."""
    assert "Very robust" in describe_robustness(-3.0)


# ============================================================================
# describe_rescue: calibration rescue effect
# ============================================================================


def test_rescue_backfire():
    """Negative recovery = no positive effect."""
    assert "no positive effect" in describe_rescue(-2.0)


def test_rescue_none():
    """Recovery < 1 = almost no improvement."""
    assert "almost no improvement" in describe_rescue(0.3)


def test_rescue_little():
    """Recovery 1~5 = slight improvement."""
    assert "slight improvement" in describe_rescue(3.0)


def test_rescue_clear():
    """Recovery 5~15 = clear improvement."""
    assert "clear improvement" in describe_rescue(10.0)


def test_rescue_boundary_5():
    """Exactly 5.0 points (boundary): < 5 is slight, 5.0 falls into 'clear improvement'."""
    assert "clear improvement" in describe_rescue(5.0)
    assert "slight improvement" in describe_rescue(4.9)


def test_rescue_big():
    """Recovery >= 15 = substantial improvement."""
    assert "substantial improvement" in describe_rescue(20.0)


# ============================================================================
# describe_recovery: recovery level
# ============================================================================


def test_recovery_full():
    """Residual < 1 = essentially fully recovered."""
    assert "Essentially fully recovered" in describe_recovery(0.2)


def test_recovery_almost():
    """Residual 1~5 = mostly recovered, minor residual error remains."""
    assert "minor residual error remains" in describe_recovery(3.0)


def test_recovery_incomplete():
    """Residual >= 5 = not yet fully recovered, with the specific number."""
    out = describe_recovery(8.0)
    assert "Not yet fully recovered" in out
    assert "8.0" in out


# ============================================================================
# render_pose_verdict: combined verdict
# ============================================================================


def test_render_returns_three_lines():
    """The combined verdict should return 3 lines, each with its number."""
    lines = render_pose_verdict(0.99, 0.88, 0.99)
    assert isinstance(lines, list)
    assert len(lines) == 3
    # First line is robustness (drop 11.0)
    assert "11.0" in lines[0]
    # Second line is calibration effect (recovered 11.0)
    assert "11.0" in lines[1]
    # Third line is recovery level (residual 0.0)
    assert "0.0" in lines[2]


def test_render_full_recovery_scenario():
    """Clean 99%, dirty 88%, rescued 99%: expect robust + clear improvement + fully recovered."""
    lines = render_pose_verdict(0.99, 0.88, 0.99)
    assert "Fairly robust" in lines[0]        # dropped 11 points → 5~15 band
    assert "clear improvement" in lines[1]    # recovered 11 points → 5~15 band
    assert "Essentially fully recovered" in lines[2]     # residual 0 points


def test_render_degraded_scenario():
    """Clean 99%, dirty 50%, rescued 55%: severe degradation, 5 points recovered (clear improvement), not fully recovered."""
    lines = render_pose_verdict(0.99, 0.50, 0.55)
    assert "Performance severely degraded" in lines[0]     # dropped 49 points
    assert "clear improvement" in lines[1]         # recovered 5.0 points → falls into 'clear improvement' band
    assert "Not yet fully recovered" in lines[2]     # residual 44 points
