#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tests for bed-exit monitoring (bed_monitor) — verifying the "state machine +
bed-exit event recording + timeout alarm" logic.

Five things are tested in focus:
  1. Whether total pressure is computed correctly
  2. The threshold boundaries of the three-state machine (in bed / sitting up / out of bed)
  3. Bed-exit event recording: correct start/end frames, filtering turnover glitches, recording multiple exits
  4. Timeout alarm: a brief nighttime exit does not alarm; only going past the time limit does
  5. Returning to bed / sitting up both count as "returned", clearing the alarm and resetting the timer
"""

import numpy as np

import bed_monitor as bm
from sensor_config import (
    GRID_W, GRID_H, BODY_PARTS, make_ideal_pressure,
)


def _ideal():
    return make_ideal_pressure(GRID_W, GRID_H, BODY_PARTS)


# ============================================================================
# total_pressure: total pressure
# ============================================================================


def test_total_pressure_sum():
    """Total pressure = the sum of all points."""
    frame = np.array([[1.0, 2.0], [3.0, 4.0]])
    assert bm.total_pressure(frame) == 10.0


# ============================================================================
# detect_states: three-state state machine
# ============================================================================


def test_states_three_state_detection():
    """Lying flat → in bed, sitting up → sitting, empty bed → out of bed."""
    ideal = _ideal()
    frames = np.array([ideal * 1.0, ideal * 0.4, ideal * 0.0])
    states = bm.detect_states(frames)
    assert states == [bm.IN_BED, bm.SITTING, bm.OUT_OF_BED]


def test_states_threshold_boundaries():
    """Use single-point frames to test boundaries precisely: 60%=in bed, 15%=sitting, 14.9%=out of bed."""
    # peak = 100; each frame's total pressure (a single-point frame is just its own value)
    frames = np.array([[100.0], [60.0], [15.0], [14.9], [0.0]])
    states = bm.detect_states(frames)   # peak = 100
    assert states[0] == bm.IN_BED       # ratio 1.0
    assert states[1] == bm.IN_BED       # ratio 0.60, >= 0.60 counts as in bed
    assert states[2] == bm.SITTING      # ratio 0.15, not < 0.15, falls in the sitting band
    assert states[3] == bm.OUT_OF_BED   # ratio 0.149 < 0.15 counts as out of bed
    assert states[4] == bm.OUT_OF_BED   # ratio 0.0


def test_states_all_empty_no_division_by_zero():
    """Nobody ever pressed (all zeros); peak treated as 1, no division-by-zero, all states out of bed."""
    frames = np.zeros((5, GRID_H, GRID_W))
    states = bm.detect_states(frames)
    assert all(s == bm.OUT_OF_BED for s in states)


# ============================================================================
# detect_exits: bed-exit event recording (record only, no alarm)
# ============================================================================


def test_exits_records_start_and_end_frames():
    """One exit; start/end frames and duration recorded correctly."""
    # indices 0~2 in bed, 3~7 out of bed (5 frames), 8~11 back in bed
    states = [bm.IN_BED] * 3 + [bm.OUT_OF_BED] * 5 + [bm.IN_BED] * 4
    exits = bm.detect_exits(states, confirm_frames=1)
    assert exits == [(3, 8)]   # [3, 8), duration 5 frames


def test_exits_filters_turnover_glitches():
    """Out of bed only 1~2 frames (a turnover glitch); not a valid exit, filtered out."""
    states = [bm.IN_BED] * 3 + [bm.OUT_OF_BED] * 2 + [bm.IN_BED] * 5
    exits = bm.detect_exits(states, confirm_frames=3)
    assert exits == []   # 2 frames < 3 frames, ignored


def test_exits_records_multiple_exits():
    """Two exits (with a return to bed in between), each recorded separately."""
    states = (
        [bm.IN_BED] * 2 +        # indices 0~1
        [bm.OUT_OF_BED] * 5 +    # first exit: indices 2~6
        [bm.IN_BED] * 3 +        # indices 7~9 (back in bed)
        [bm.OUT_OF_BED] * 4 +    # second exit: indices 10~13
        [bm.IN_BED] * 2          # indices 14~15 (back in bed)
    )
    exits = bm.detect_exits(states, confirm_frames=1)
    assert exits == [(2, 7), (10, 14)]


def test_exits_closes_final_open_exit():
    """If the sequence ends while still out of bed, record that final stretch too."""
    states = [bm.IN_BED] * 3 + [bm.OUT_OF_BED] * 6
    exits = bm.detect_exits(states, confirm_frames=1)
    assert exits == [(3, 9)]   # [3, 9), duration 6 frames


# ============================================================================
# detect_alarms: alarm only when gone past the time limit
# ============================================================================


def test_alarm_nighttime_exit_no_alarm():
    """Out of bed 8 frames (< timeout 10) then back; a normal nighttime exit, no alarm."""
    states = [bm.IN_BED] * 3 + [bm.OUT_OF_BED] * 8 + [bm.IN_BED] * 5
    alarms = bm.detect_alarms(states, away_timeout=10)
    assert not any(alarms)


def test_alarm_alerts_only_after_timeout():
    """Out of bed 10 consecutive frames (= timeout) → alarm at frame 10; cleared on return."""
    # indices: 0~2 in bed, 3~12 out of bed (10 frames), 13~17 back in bed
    states = [bm.IN_BED] * 3 + [bm.OUT_OF_BED] * 10 + [bm.IN_BED] * 5
    alarms = bm.detect_alarms(states, away_timeout=10)
    assert alarms[11] is False   # 9th out-of-bed frame, not yet timed out, no alarm
    assert alarms[12] is True    # 10th out-of-bed frame, timeout sounds
    assert alarms[13] is False   # first frame back in bed, cleared


def test_alarm_sitting_up_counts_as_return():
    """After the alarm, sitting back at the edge (sitting) also counts as 'returned'; alarm cleared."""
    # indices: 0~1 in bed, 2~11 out of bed (10 frames), 12~14 sitting, 15~17 back in bed
    states = (
        [bm.IN_BED] * 2 + [bm.OUT_OF_BED] * 10 +
        [bm.SITTING] * 3 + [bm.IN_BED] * 3
    )
    alarms = bm.detect_alarms(states, away_timeout=10)
    assert alarms[11] is True    # 10th out-of-bed frame, alarm sounds
    assert alarms[12] is False   # sitting = returned, cleared
    assert alarms[15] is False   # back in bed, still cleared


def test_alarm_timer_resets_midway():
    """9 frames out of bed → 1 frame sitting → out again: timer resets, no accumulated false alarm."""
    # first 9 frames out of bed (less than 10), interrupted by sitting, then 5 more out of bed
    states = (
        [bm.IN_BED] * 2 + [bm.OUT_OF_BED] * 9 +
        [bm.SITTING] * 1 + [bm.OUT_OF_BED] * 5 + [bm.IN_BED] * 2
    )
    alarms = bm.detect_alarms(states, away_timeout=10)
    # sitting interrupted the timer; only 5 consecutive out-of-bed frames remain, fewer than 10, no alarm
    assert not any(alarms)
