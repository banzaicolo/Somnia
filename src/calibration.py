#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=============================================================================
Sensor calibration toolchain — the complete set of moves to clean up "dirty data"
=============================================================================

[First, plain language: what is calibration?]

Sensors lie (you've already seen it in the Pressure Simulator: the four corners
read fake values of 15~40 even when nobody is pressing on them). "Calibration"
means: come up with a set of math steps that turn the dirty data a sensor spits
out back into clean data close to "what a person actually pressed".

[Why is this file the most important one?]

In this whole AI medical bed project, the easiest place to fail is not the AI
model — it's sensor calibration. Many teams build a strong model but die right
here. This file writes out all the calibration moves, and each one can be tested
on its own and chained into one complete pipeline.

[What are the calibration moves?]

The dirty data a sensor reads gets "dirtied" like this (left to right, one layer
of contamination at a time):

    true pressure -> ① multiply by sensitivity -> ② add temp drift -> ③ add zero drift -> ④ crosstalk leakage -> ⑤ add noise
    (ideal)                                                                                          (raw dirty data)

So "cleaning it up" is the reverse, working backwards step by step:

    dirty data -> denoise -> ④ remove crosstalk -> ③② remove additive offset -> ① remove gain -> clean data

This file implements that reverse process. One function per move, and finally
calibrate() chains them all together.

[How to use it?]

    import calibration as cal
    cleaned = cal.calibrate(raw, baseline=..., gain_map=..., c=0.1)

=============================================================================
"""

import numpy as np


# ============================================================================
# 0. Forward model (understand how the sensor "dirties" data before "cleaning" it)
# ============================================================================

def forward(ideal, gain, offset, c=0.0, noise=0.0):
    """
    Forward model: turn "true pressure" into a "sensor reading".

    This is the "bad example" of the whole calibration story — the contamination
    process a real sensor goes through.

    Args:
      ideal  -- true body-pressure map (ground truth), a 2D array of shape (H, W)
      gain   -- per-point sensitivity, (H, W). In the real world each point differs (batch mismatch)
      offset -- per-point zero offset, (H, W). The "fake baseline" caused by temp drift + aging
      c      -- crosstalk coefficient, 0~1. 0.1 means 10% of the signal leaks to neighbors
      noise  -- noise, (H, W) or scalar. Random jitter

    Returns: the dirty data (raw) the sensor reads, (H, W)
    """
    # ① multiply by sensitivity + ②③ add offset (multiplicative and additive both here)
    x = ideal * gain + offset

    # ④ crosstalk: each point leaks some signal to its four neighbors (up/down/left/right)
    if c > 0:
        padded = np.pad(x, 1, mode="edge")   # pad one ring so boundary points can also read neighbors
        neighbors = (padded[:-2, 1:-1] + padded[2:, 1:-1] +
                     padded[1:-1, :-2] + padded[1:-1, 2:]) / 4.0
        x = (1 - c) * x + c * neighbors

    # ⑤ add noise
    return x + noise


# ============================================================================
# 1. The four correction functions (reverse process, each fixing one kind of fault)
# ============================================================================

def correct_crosstalk(raw, c, n_iter=30):
    """
    Move ④: crosstalk correction (remove "neighbor leakage").

    The forward crosstalk is: raw = (1-c)·x + c·(average of the four neighbors).
    To recover x, solve this equation. Since c is usually small (0.1), iteration
    is the most intuitive approach:

        First guess x≈raw, then repeatedly apply  x_new = (raw - c·neighbors(x)) / (1-c)
        Each iteration gets closer to the true x.

    Args:
      raw    -- data contaminated by crosstalk
      c      -- crosstalk coefficient (same number as in the forward model)
      n_iter -- how many iterations. The smaller c is, the faster it converges; 30 is plenty
    """
    x = np.array(raw, dtype=float)
    for _ in range(n_iter):
        padded = np.pad(x, 1, mode="edge")
        neighbors = (padded[:-2, 1:-1] + padded[2:, 1:-1] +
                     padded[1:-1, :-2] + padded[1:-1, 2:]) / 4.0
        x = (raw - c * neighbors) / (1 - c)
    return x


def correct_zero(raw, baseline):
    """
    Move ②③: zero calibration (subtraction).

    When the person gets off the bed the reading should be 0, but the sensor reads
    a "fake baseline" (temp drift + zero drift). Record the no-load reading as the
    baseline, then subtract it from every later reading.

    This is the "one subtraction removes 82% of the error" you saw earlier in the
    Zero Calibration Demo.
    """
    return raw - baseline


def correct_gain(raw, gain_map):
    """
    Move ①: gain calibration (division).

    Each point has a different sensitivity (same pressure: point A reads 110,
    point B reads 90). During calibration you measure each point's gain with a
    "known standard pressure" (gain_map), then divide later readings by it to
    recover the true pressure.

    Note: if gain_map contains values near 0, dividing directly blows up (division
    by zero), so we add a guard: treat too-small values as 1 (no scaling).
    """
    safe_gain = np.where(np.abs(gain_map) < 1e-6, 1.0, gain_map)
    return raw / safe_gain


def correct_temperature_offset(raw, temp, t_ref, tco_map):
    """
    Move ② advanced: temperature zero-offset compensation (subtract a temperature-dependent term).

    The sensor's "fake baseline" changes with temperature: each +1 degree makes the
    baseline drift a little. This "how much per degree" is called TCO (temperature
    coefficient of offset) and differs per point.

    During calibration you measured the baseline at the reference temperature t_ref;
    now the operating temperature is temp, so the baseline drifted by an extra
    tco_map × (temp - t_ref) — subtract that amount.

    Args:
      raw     -- reading
      temp    -- current temperature (e.g. 40)
      t_ref   -- reference temperature during calibration (e.g. 25)
      tco_map -- per-point temperature-to-offset coefficient, (H, W)
    """
    return raw - tco_map * (temp - t_ref)


def correct_temperature_gain(raw, temp, t_ref, tcs_map):
    """
    Move ① advanced: temperature gain compensation (divide by a temperature-dependent term).

    Sensitivity also changes with temperature: each +1 degree changes the gain a
    little. This is called TCS (temperature coefficient of sensitivity). By the same
    logic, divide it back out.

    tcs_map is "how much ratio per degree", e.g. 0.002 means 0.2% per degree.
    """
    return raw / (1.0 + tcs_map * (temp - t_ref))


def correct_hysteresis(reading, direction, h):
    """
    Move (bonus): hysteresis compensation.

    Hysteresis is the sensor "holding a grudge": at the same pressure it reads high
    while loading and low while unloading. The math model (same as the loop in the
    Pressure Simulator, constant width h):

        loading:   reading = p + h/2   (high by half the width)
        unloading: reading = p - h/2   (low by half the width)

    To recover the true pressure p, just invert those two equations:

        loading:   p = reading - h/2
        unloading: p = reading + h/2

    Why use a "constant offset" instead of a complex model that "converges with
    pressure"? Because the complex model has to assume "what the full scale is"
    (e.g. 100); once the true pressure exceeds full scale (hip peak 110 > 100) the
    direction flips — loading would read low instead. The constant-offset model has
    no such hidden assumption, holds for any pressure range, and has the simplest
    formula.

    Args:
      reading   -- sensor reading (can be an array)
      direction -- "load" = loading (reads high) or "unload" = unloading (reads low)
      h         -- hysteresis width (difference between load and unload readings, same number as HYSTERESIS)
    """
    reading = np.asarray(reading, dtype=float)
    if direction == "load":
        return reading - h / 2.0
    elif direction == "unload":
        return reading + h / 2.0
    else:
        # wrong direction given: better to raise than guess
        raise ValueError("direction must be 'load' or 'unload'")


def estimate_direction(prev, curr, deadband=0.0):
    """
    Decide "which way the pressure is moving": loading / unloading / unchanged.

    correct_hysteresis needs you to tell it "is this loading or unloading". But in
    the real world the sensor won't tell you the direction, so you have to guess by
    comparing this frame with the previous one:

        this frame clearly larger than previous -> loading (load, reads high)
        this frame clearly smaller than previous -> unloading (unload, reads low)
        roughly the same                        -> unchanged, keep previous direction

    [Why the deadband?]

    Noise makes readings jitter meaninglessly. Say the hysteresis width is only 8
    and the noise is ±3: without a deadband, a +1 jitter would be treated as a
    "direction reversal", and the compensation would thrash wildly between loading
    and unloading, making things worse. Add a deadband so only a change above the
    threshold counts as a real turn.

    Args:
      prev, curr -- previous and current pressure maps, (H, W)
      deadband   -- deadband threshold. An absolute change <= deadband counts as "unchanged"

    Returns:
      direction -- integer array of the same shape as prev: +1=loading / -1=unloading / 0=unchanged
    """
    delta = np.asarray(curr, dtype=float) - np.asarray(prev, dtype=float)
    direction = np.zeros(np.shape(delta), dtype=int)
    direction[delta > deadband] = 1
    direction[delta < -deadband] = -1
    return direction


def correct_hysteresis_sequence(readings, h, deadband=0.0):
    """
    The "full version" of hysteresis compensation: auto-detect direction and fix the data frame by frame.

    The single-frame correct_hysteresis needs you to supply the direction by hand;
    this function handles the real scenario — the program receives a stream of
    consecutive frames (e.g. 10 per second) and must figure out the direction
    itself. Here "detect direction + compensate" are chained into one flow.

    Each frame does two things:
      1. compensate this frame using "the direction remembered since the previous frame";
      2. compare this frame with the previous one, and only update the direction if it changed clearly, otherwise keep it.

    The first frame has no "previous frame", so it defaults to "loading" — because
    getting onto the bed always starts with "pressing down (loading)", a reasonable
    prior.

    Args:
      readings -- sensor reading sequence of shape (n_frames, H, W)
      h        -- hysteresis width (same number as HYSTERESIS in sensor_config)
      deadband -- deadband threshold, see estimate_direction

    Returns:
      corrected -- compensated sequence of the same shape as readings
    """
    readings = np.asarray(readings, dtype=float)
    n = readings.shape[0]
    corrected = np.empty_like(readings)

    # direction memory: +1=loading, -1=unloading. First frame defaults to loading.
    direction = np.ones(readings.shape[1:], dtype=int)

    for i in range(n):
        cur = readings[i]
        load_mask = direction > 0

        # the vectorized version of correct_hysteresis: one formula for loading points, another for unloading points
        frame = np.empty_like(cur)
        frame[load_mask] = cur[load_mask] - h / 2.0
        frame[~load_mask] = cur[~load_mask] + h / 2.0
        corrected[i] = frame

        # update direction (skip the last frame); only "clearly changed" points turn
        if i < n - 1:
            step = estimate_direction(readings[i], readings[i + 1], deadband)
            direction[step != 0] = step[step != 0]

    return corrected


# ============================================================================
# 2. Calibration parameter estimation (simulate "real calibration steps": no-load for baseline, weights for gain)
# ============================================================================

def estimate_baseline(offset, c=0.0, noise_std=0.0, rng=None):
    """
    Simulate "no-load calibration": with the person off the bed, what you read is the fake baseline.

    Real hardware step: when the device powers on and before anyone lies down,
    take several readings and average them to get the baseline. Here we run the
    forward model once using the known offset (zero-offset map).

    Returns: an estimate of baseline (including crosstalk and noise, just as
    "imperfect" as the real scenario).
    """
    if rng is None:
        rng = np.random.default_rng(0)
    noise = rng.normal(0, noise_std, size=np.shape(offset))
    # no-load: ideal = 0, so forward(0, gain, offset) = offset passed through crosstalk and noise
    return forward(np.zeros_like(offset), np.ones_like(offset), offset, c, noise)


def estimate_gain(ideal_ref, raw_ref, baseline, c=0.0):
    """
    Simulate "gain calibration": press with a known standard pressure to back out each point's sensitivity.

    Real hardware step: press each point with a standard weight of known mass (or a
    known pressure pad); reading ÷ standard pressure = that point's sensitivity.

    Here:
      ideal_ref -- the "standard pressure map" used during calibration (known)
      raw_ref   -- what the sensor reads when the standard pressure is applied
      baseline  -- the baseline measured earlier at no-load
      c         -- crosstalk coefficient

    Steps: remove baseline -> remove crosstalk -> divide by standard pressure = per-point gain.
    """
    # remove additive offset (subtract baseline)
    x = raw_ref - baseline
    # remove crosstalk (solve the leakage)
    if c > 0:
        x = correct_crosstalk(x, c)
    # divide by standard pressure to get gain; skip points where standard pressure is 0 (avoid division by zero)
    safe = np.where(np.abs(ideal_ref) < 1e-6, 1.0, ideal_ref)
    return x / safe


# ============================================================================
# 3. Full pipeline: chain all the moves above together
# ============================================================================

def calibrate(raw, baseline=None, gain_map=None, c=0.0,
              temp=None, t_ref=25.0, tco_map=None, tcs_map=None,
              direction=None, h=0.0):
    """
    Full calibration pipeline: dirty data in, clean data close to "true pressure" out.

    Order (exactly the reverse of the contamination order):

        ① crosstalk correction (first solve the spatial "neighbor leakage")
        ② temperature zero-offset compensation (subtract the temperature-drifting baseline, optional)
        ③ zero calibration (subtract the no-load baseline)
        ④ temperature gain compensation (divide by the temperature-varying sensitivity, optional)
        ⑤ gain calibration (divide by each point's sensitivity)
        ⑥ hysteresis compensation (optional, when the direction is known)

    Every parameter can be omitted (None skips that step), so you can tune a single move on its own.
    """
    result = np.array(raw, dtype=float)

    if c > 0:
        result = correct_crosstalk(result, c)

    if tco_map is not None and temp is not None:
        result = correct_temperature_offset(result, temp, t_ref, tco_map)

    if baseline is not None:
        result = correct_zero(result, baseline)

    if tcs_map is not None and temp is not None:
        result = correct_temperature_gain(result, temp, t_ref, tcs_map)

    if gain_map is not None:
        result = correct_gain(result, gain_map)

    if direction is not None and h > 0:
        result = correct_hysteresis(result, direction, h)

    return result


# ============================================================================
# 4. Error metrics (how good the calibration is, in numbers)
# ============================================================================

def mean_abs_error(cleaned, ideal):
    """Mean absolute error: how far each point is from the correct answer, averaged. Smaller is better."""
    return float(np.abs(np.asarray(cleaned) - np.asarray(ideal)).mean())


def improvement(err_before, err_after):
    """How many percent the error shrank. Positive = improved."""
    if err_before == 0:
        return 0.0
    return float((err_before - err_after) / err_before * 100)
