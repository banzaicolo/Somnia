#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=============================================================================
bed_monitor.py — bed-exit monitoring: judge "is the person in bed or not",
and "should an alarm be raised"
=============================================================================

What problem does this solve?

The scariest thing in a nursing home or hospital ward: an elderly person gets
out of bed alone in the middle of the night, falls in the bathroom, or faints
in the hallway with no one noticing. This module is the bed's "night watchman"
— using pressure sensors under the mattress, it decides whether the person is
in bed, sitting up, or already out of bed.

Key point: getting out of bed does NOT by itself raise an alarm!

In real life it's perfectly normal for an elderly person to get up at night to
use the toilet, drink some water, or take a couple of steps. If every exit
triggered an alarm, the caregiver would mute the system within a couple of days
and the alarm would become useless.

What's truly dangerous is "getting out of bed and not coming back for a long
time" — that's when they may have fallen, fainted, or a confused resident may
have wandered out of the room and gotten lost.

So this module has exactly one alarm rule:

    "Out of bed" persisting past a time limit (default 30 minutes) with no
    return → only then raise an alarm.

The exit itself is only *recorded* (when they left, when they returned, how
long they were gone), so caregivers can grasp the resident's routine without
disturbing anyone.

How do others do it? (simplified from industry practice)

We checked patents and papers: the core idea of bed-exit monitoring is very
simple and does not rely on fancy deep learning:

  1. Look at just one number: total pressure (the sum of all sensor readings).
     Person in bed → this number is large; person out of bed → it drops to zero.
  2. Use three states instead of two ("in/out"):
       in bed (lying flat) → sitting up (weight drops by more than half) → out of bed (nearly zero)
  3. Use "ratios" for thresholds rather than fixed numbers: people of different
     weights differ a lot. "Below 15% of peak means out of bed" works for both
     a 50-jin and a 180-jin person.
  4. Two-stage debounce:
       short debounce (10 s) — filter out "false exits" like turning over or sitting up to drink;
       long limit (30 min) — after an exit is confirmed, alarm only if they don't return in time.

This module turns that logic into testable, reusable functions.

How to use it?

    import bed_monitor as bm
    states = bm.detect_states(frames)    # which state the person is in per frame
    exits  = bm.detect_exits(states)     # start/end of each exit (record only, no alarm)
    alarms = bm.detect_alarms(states)    # alarm only when gone past the time limit

=============================================================================
"""

import numpy as np


# ============================================================================
# Monitoring thresholds (business parameters; change only here if you want to tune)
# ============================================================================

# Sampling frame rate: how many frames per second. All "times" below are converted
# to "frame counts" via this value. A real system's frame rate may be higher
# (e.g. 10 per second); change this one number and the durations below follow.
FRAMES_PER_SECOND = 1

# Short debounce: how many consecutive seconds of "out of bed" count as a real exit.
# Turning over, sitting up to drink, or sitting at the edge briefly drops total
# pressure, but not enough to count as "a real exit". This debounce filters out
# those glitches.
EXIT_CONFIRM_SECONDS = 10
EXIT_CONFIRM_FRAMES = int(EXIT_CONFIRM_SECONDS * FRAMES_PER_SECOND)

# Timeout alarm: after an exit is confirmed, how many minutes without returning
# before the alarm sounds. Life logic: getting out of bed (toilet, water, a short
# walk) is normal and should not alarm; only "gone and not returning for a long
# time" is dangerous — possibly a fall in the bathroom, a faint in the hallway,
# or a confused resident wandering off.
AWAY_TIMEOUT_MINUTES = 30
AWAY_TIMEOUT_FRAMES = int(AWAY_TIMEOUT_MINUTES * 60 * FRAMES_PER_SECOND)

# total pressure = peak (this person's full-bed weight when lying flat) × ratio;
# whichever band it falls into determines the state
IN_BED_RATIO = 0.60      # total pressure >= 60% of peak  → in bed
OUT_OF_BED_RATIO = 0.15  # total pressure < 15% of peak    → out of bed (the middle band = sitting up)

# Three states (string constants, convenient for building reports; display labels in STATE_CN)
IN_BED = "in_bed"
SITTING = "sitting"
OUT_OF_BED = "out_of_bed"
STATE_CN = {IN_BED: "In bed", SITTING: "Sitting", OUT_OF_BED: "Out of bed"}


# ============================================================================
# 1. Total pressure: the single most important number for bed-exit monitoring
# ============================================================================

def total_pressure(frame):
    """
    Compute the "total pressure" of one pressure map = the sum of all sensor
    point readings.

    Why is this number so important? Because "is someone there" doesn't depend
    on any single point but on the bed's total "weight". A person lying down
    makes dozens of points rise together, so the sum is substantial; a person
    getting out makes every point drop to zero, so the sum is zero too. One
    number captures the essence of "in bed or not".

    Args:
      frame — one pressure map (an (H, W) array, or any shape; it will be
              flattened and summed)

    Returns:
      A float: the total pressure.
    """
    return float(np.asarray(frame, dtype=float).sum())


# ============================================================================
# 2. State machine: translate "total pressure over time" into "what state the person is in"
# ============================================================================

def detect_states(frames, in_ratio=IN_BED_RATIO, out_ratio=OUT_OF_BED_RATIO):
    """
    Input a sequence of pressure frames, output the per-frame state (in bed /
    sitting up / out of bed).

    Approach (three straightforward steps):
      1. Compute the "peak total pressure" of the whole sequence as this
         person's "full-bed weight" baseline;
      2. Divide each frame's total pressure by the peak to get a 0~1 ratio;
      3. Bucket by ratio:
            >= 60%  → in bed
            15%~60% → sitting up (the person sat up; weight dropped by more than half but still on the bed)
            < 15%   → out of bed (basically no one there)

    Why use a "ratio" instead of a fixed number? Because a 50-jin and a 180-jin
    person's full-bed weights differ by over 3×; a fixed number (e.g. "total
    pressure < 50 means out of bed") can't be universal. Ratio thresholds
    naturally adapt to body weight and hold for anyone.

    Args:
      frames    — (n_frames, H, W) sequence of pressure maps
      in_ratio  — lower bound ratio for "in bed" (default 0.60)
      out_ratio — upper bound ratio for "out of bed" (default 0.15)

    Returns:
      list[str], length = number of frames, each element is "in_bed" / "sitting" / "out_of_bed"
    """
    frames = np.asarray(frames, dtype=float)
    # total pressure per frame (flatten to (n_frames, all points) then sum per row)
    totals = frames.reshape(frames.shape[0], -1).sum(axis=1)
    peak = totals.max()
    if peak <= 0:
        peak = 1.0   # nobody ever pressed (all zeros); avoid division by zero, states will all be "out of bed"

    states = []
    for t in totals:
        ratio = t / peak
        if ratio >= in_ratio:
            states.append(IN_BED)
        elif ratio < out_ratio:
            states.append(OUT_OF_BED)
        else:
            states.append(SITTING)
    return states


# ============================================================================
# 3. Bed-exit events: record only, no alarm (getting out of bed is perfectly normal)
# ============================================================================

def detect_exits(states, confirm_frames=EXIT_CONFIRM_FRAMES):
    """
    Record every "bed-exit event": when the person got out, when they came back,
    and how long they were gone.

    Note: this function only *records*; it does not alarm — getting out of bed
    is perfectly normal. It gives the caregiver a "routine table": when the
    person got up at night, how long they were gone, when they returned, so the
    caregiver can grasp the resident's activity pattern. Truly abnormal patterns
    (e.g. frequent night exits, progressively longer exits) are things the
    caregiver can spot by looking at this table.

    Debounce: an out-of-bed stretch shorter than confirm_frames (e.g. a 1~2 frame
    glitch from turning over) does not count as a real "exit" and is ignored.

    Args:
      states         — output of detect_states, a sequence of state strings
      confirm_frames — how many frames out of bed at minimum counts as a valid event (default 10)

    Returns:
      list[tuple], each element (start, end) marking one exit's frame interval
      [start, end). Duration = end - start frames.
    """
    exits = []
    start = None
    for i, s in enumerate(states):
        if s == OUT_OF_BED:
            if start is None:
                start = i          # entered out-of-bed, record the start
        else:
            if start is not None:
                # only counts as a valid "exit" if gone for confirm_frames or more
                if i - start >= confirm_frames:
                    exits.append((start, i))
                start = None
    # still out of bed at the end of the sequence (never returned); close it out
    if start is not None and len(states) - start >= confirm_frames:
        exits.append((start, len(states)))
    return exits


# ============================================================================
# 4. Alarm: only sounds when gone past the time limit, and clears the moment the person returns
# ============================================================================

def detect_alarms(states, away_timeout=AWAY_TIMEOUT_FRAMES):
    """
    Bed-exit timeout alarm: getting out of bed itself does NOT alarm; only
    staying gone too long does.

    Real-life logic:
      - Getting up at night for the toilet, a drink, or a short stroll is all
        normal. Alarm on every exit and the caregiver mutes the system within
        days, rendering the alarm useless.
      - What's truly dangerous is "gone and not back for a long time" — possibly
        a fall in the bathroom, a faint in the hallway, or a confused resident
        wandering out of the room.

    So this function only cares about one thing: how long the person has been
    *continuously* out of bed (out_of_bed).
      - Fewer than away_timeout frames → safe, no alarm;
      - away_timeout frames or more → sound the "bed-exit timeout" alarm;
      - the moment the person is back in bed (in_bed lying down, or sitting back
        at the edge) → clear immediately and restart the timer.

    Args:
      states       — output of detect_states, a sequence of state strings
      away_timeout — how many consecutive out-of-bed frames trigger the alarm (default 1800 frames = 30 minutes)

    Returns:
      list[bool], length = number of frames, True means that frame is alarming
    """
    away_count = 0      # consecutive out-of-bed counter (unit: frames)
    alarming = False    # whether currently alarming
    alarms = []

    for s in states:
        if s == OUT_OF_BED:
            away_count += 1
            if away_count >= away_timeout:
                alarming = True    # out of bed long enough with no return, sound the alarm
        else:
            # person is back (lying down or sitting at the edge) — safe; clear the alarm and reset the timer
            away_count = 0
            alarming = False
        alarms.append(alarming)

    return alarms
