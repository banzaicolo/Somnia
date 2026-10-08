#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=============================================================================
verdict.py — verdict generator: let the program "speak from the numbers"
instead of hard-coding conclusions
=============================================================================

[What problem does this module solve?]

Earlier scripts had several "conclusions" hard-coded in the code. For example:

    print(" The model is robust to the problem")

No matter how the actual numbers changed, this sentence was always printed.
That has a fatal flaw: if parameters change and the numbers collapse (e.g.
accuracy drops to 60%), that "very robust" line still prints — the conclusion
becomes a lie, and the program itself does not even know it.

The right approach is "hard-code the rules, generate the conclusion dynamically":

    · Rules (thresholds) are hard-coded: a drop < 5 points is "robust",
      > 30 points is "severe performance degradation"
    · The conclusion is generated dynamically: the program applies this run's
      real numbers to the rules and derives what to say this time

As an analogy: in a health report, the rule "blood pressure > 140 = hypertension"
is hard-coded, but the conclusion "is your blood pressure high today" is derived
dynamically from that day's measured numbers. The standard is fixed; the result
is live.

This module does one thing: it gathers the "number → conclusion" decision rules
in one place for other scripts to call. When the numbers change, the conclusions
follow.

[Why are the thresholds hard-coded?]

Thresholds are not "conclusions"; they are "judgment criteria". Just as "60 points
is a pass" is a criterion, not a conclusion. Hard-coding criteria is fine; the
problem is hard-coding the conclusion alongside them. And these thresholds live in
the constants section below, easy to see and easy to adjust, not buried deep in code.

[How to use it?]

    from verdict import describe_improvement, render_pose_verdict

    # Example 1: error shrank by 82% — ask "is it fully cleaned up?"
    print(describe_improvement(82.0))   # → "Essentially eliminated"

    # Example 2: three accuracies — generate a full dynamic verdict
    for line in render_pose_verdict(0.99, 0.88, 0.99):
        print(line)
=============================================================================
"""

# ============================================================================
# Decision thresholds ("criteria"; hard-coding is reasonable — adjust here only)
# Units:
#   - drop / recovery / residual are in "percentage points", e.g. 2.3 = 2.3 pp
#   - improvement is in "percent", e.g. 82 = error shrank by 82%
# ============================================================================

# ---- Model robustness: how many points a bad sensor costs the model ----
ROBUST_DROP = 5.0      # drop below 5 points → very robust
DEGRADE_DROP = 30.0    # drop above 30 points → severe performance degradation
MID_DROP = 15.0        # one more band in between: 5~15 fairly robust, 15~30 clearly degraded


# ---- Calibration rescue effect: how many points recovered after re-calibration ----
RESCUE_TINY = 1.0      # recovered below 1 point → almost no improvement
RESCUE_BIG = 15.0      # recovered above 15 points → substantial improvement
RESCUE_MID = 5.0       # middle band: 1~5 slight improvement, 5~15 clear improvement

# ---- Recovery level: how far below "clean level" after re-calibration ----
RECOVER_GOOD = 1.0     # residual below 1 point → essentially fully recovered
RECOVER_BAD = 5.0      # residual above 5 points → not yet fully recovered

# ---- Improvement level (general): by what percent the error shrank ----
IMPROVE_BIG = 80.0     # above 80% → essentially eliminated
IMPROVE_MID = 50.0     # 50%~80% → clear improvement
IMPROVE_LOW = 20.0     # 20%~50% → limited improvement


# ============================================================================
# 1. General: improvement level (error reduction percent → conclusion)
# ============================================================================


def describe_improvement(improve_pct):
    """
    Return a one-sentence conclusion based on "by what percent the error shrank".

    Args:
      improve_pct —— the percentage by which the error shrank (can be negative).
                     Positive = got better; negative = made it worse (error rose).

    Returns:
      One English sentence directly describing the degree of improvement.

    Examples:
      describe_improvement(82)  → "Essentially eliminated"
      describe_improvement(30)  → "Limited improvement; considerable residual error remains"
      describe_improvement(-5)  → "Error did not decrease but increased; the calibration method is unsuitable for this scenario"
    """
    if improve_pct < 0:
        return "Error did not decrease but increased; the calibration method is unsuitable for this scenario"
    if improve_pct < IMPROVE_LOW:
        return "Almost no improvement; the problem lies outside the error sources removable by this step"
    if improve_pct < IMPROVE_MID:
        return "Limited improvement; considerable residual error remains"
    if improve_pct < IMPROVE_BIG:
        return "Clear improvement, but not yet fully eliminated"
    return "Essentially eliminated"


# ============================================================================
# 2. Pose classification add-on experiment: three accuracies → three conclusions
# ============================================================================


def describe_robustness(drop_points):
    """
    Return a robustness conclusion based on "how many points a bad sensor costs".

    Args:
      drop_points —— the drop (percentage points, >=0). 0 means no drop at all.

    Examples:
      describe_robustness(2)   → "Very robust; performance barely affected"
      describe_robustness(40)  → "Performance severely degraded; close to random guessing"
    """
    drop = max(0.0, drop_points)   # negative (dirty data scoring higher) treated as 0
    if drop < ROBUST_DROP:
        return "Very robust; performance barely affected"
    if drop < MID_DROP:
        return "Fairly robust; only slightly affected"
    if drop < DEGRADE_DROP:
        return "Performance clearly degraded; accuracy dropped noticeably"
    return "Performance severely degraded; close to random guessing"


def describe_rescue(recovered_points):
    """
    Return a calibration-effect conclusion based on "how many points recovered".

    Args:
      recovered_points —— points recovered (percentage points). Negative = calibration made it worse.

    Examples:
      describe_rescue(0.3)  → "Calibration brought almost no improvement"
      describe_rescue(20)   → "Calibration brought substantial improvement"
    """
    if recovered_points < 0:
        return "Calibration had no positive effect and made results worse (calibration parameters mismatch the actual error source)"
    if recovered_points < RESCUE_TINY:
        return "Calibration brought almost no improvement"
    if recovered_points < RESCUE_MID:
        return "Calibration brought slight improvement"
    if recovered_points < RESCUE_BIG:
        return "Calibration brought clear improvement"
    return "Calibration brought substantial improvement"


def describe_recovery(residual_points):
    """
    Return a recovery-level conclusion based on "how far below clean level after re-calibration".

    Args:
      residual_points —— residual points (percentage points). 0 = fully recovered.

    Examples:
      describe_recovery(0.2)  → "Essentially fully recovered"
      describe_recovery(8)    → "Not yet fully recovered; residual error of 8.0 points remains"
    """
    if residual_points < RECOVER_GOOD:
        return "Essentially fully recovered"
    if residual_points < RECOVER_BAD:
        return "Mostly recovered; minor residual error remains"
    return f"Not yet fully recovered; residual error of {residual_points:.1f} points remains"


def render_pose_verdict(test_acc, acc_dirty, acc_rescued):
    """
    Generate the full dynamic conclusion for this add-on experiment from three accuracies.

    Args (all decimals in 0~1, e.g. 0.99 means 99%):
      test_acc    —— clean-data test accuracy
      acc_dirty   —— bad-sensor test accuracy
      acc_rescued —— test accuracy after calibration recovery

    Returns:
      A list of strings, one sentence per line. Just iterate and print.

    Logic:
      drop      = clean - dirty    → describes robustness
      recovered = rescued - dirty  → describes calibration effect
      residual  = clean - rescued  → describes recovery level
    """
    drop = (test_acc - acc_dirty) * 100
    recovered = (acc_rescued - acc_dirty) * 100
    residual = (test_acc - acc_rescued) * 100

    return [
        f"Model robustness: sensor anomaly caused a {drop:.1f}-point accuracy drop → "
        f"{describe_robustness(drop)}.",
        f"Calibration effect: calibration recovered {recovered:.1f} points → "
        f"{describe_rescue(recovered)}.",
        f"Recovery level: after calibration still {residual:.1f} points below baseline → "
        f"{describe_recovery(residual)}.",
    ]


# ============================================================================
# 3. Sleep staging: Cohen's Kappa agreement + deep-sleep ratio
# ============================================================================

# ---- Cohen's Kappa band thresholds ----
KAPPA_POOR = 0.2     # below 0.2 → poor agreement
KAPPA_FAIR = 0.4     # 0.2~0.4 → fair agreement
KAPPA_MODERATE = 0.6 # 0.4~0.6 → moderate agreement (typical for consumer grade)
KAPPA_GOOD = 0.8     # 0.6~0.8 → good agreement; ≥0.81 is the medical threshold

# ---- Deep-sleep ratio (share of the night in deep sleep) normal range ----
DEEP_LOW = 10.0     # below 10% → deep sleep on the low side
DEEP_HIGH = 25.0    # above 25% → deep sleep on the high side (possible rebound after deprivation)


def describe_kappa(kappa):
    """
    Return an agreement conclusion based on the Cohen's Kappa value.

    Args:
      kappa —— the κ value (-1 ~ 1)

    Examples:
      describe_kappa(0.05) → "Poor agreement, close to random guessing"
      describe_kappa(0.45) → "Moderate agreement (typical for consumer-grade non-EEG staging)"
    """
    if kappa < KAPPA_POOR:
        return "Poor agreement, close to random guessing"
    if kappa < KAPPA_FAIR:
        return "Fair agreement"
    if kappa < KAPPA_MODERATE:
        return "Moderate agreement (typical for consumer-grade non-EEG staging)"
    if kappa < KAPPA_GOOD:
        return "Good agreement"
    return "High agreement, close to medical-grade sleep staging standards"


def describe_deep_ratio(deep_pct):
    """
    Return a conclusion on whether deep sleep is reasonable based on its share of the night.

    Args:
      deep_pct —— deep-sleep ratio (percent, e.g. 18 means 18%)

    Examples:
      describe_deep_ratio(18) → "Deep sleep ratio 18.0%, within the normal range"
      describe_deep_ratio(6)  → "Deep sleep ratio 6.0%, on the low side"
    """
    if deep_pct < DEEP_LOW:
        return f"Deep sleep ratio {deep_pct:.1f}%, on the low side"
    if deep_pct > DEEP_HIGH:
        return f"Deep sleep ratio {deep_pct:.1f}%, on the high side (possibly compensatory rebound after sleep deprivation)"
    return f"Deep sleep ratio {deep_pct:.1f}%, within the normal range"
