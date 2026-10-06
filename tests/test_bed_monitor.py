#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
离床监测（bed_monitor）的测试 —— 验证「状态机 + 离床事件记录 + 超时报警」这套逻辑。

重点测五件事：
  1. 总压力算得对不对
  2. 状态机三档（在床/坐起/离床）的阈值边界
  3. 离床事件记录：起止帧对、过滤翻身毛刺、多段离床都记录
  4. 超时报警：起夜（短暂离床）不报警、超时未归才报警
  5. 回床 / 坐起都算"回来了"，报警解除并重新计时
"""

import numpy as np

import bed_monitor as bm
from sensor_config import (
    GRID_W, GRID_H, BODY_PARTS, make_ideal_pressure,
)


def _ideal():
    return make_ideal_pressure(GRID_W, GRID_H, BODY_PARTS)


# ============================================================================
# total_pressure：总压力
# ============================================================================


def test_total_pressure_求和():
    """总压力 = 所有点之和。"""
    frame = np.array([[1.0, 2.0], [3.0, 4.0]])
    assert bm.total_pressure(frame) == 10.0


# ============================================================================
# detect_states：三态状态机
# ============================================================================


def test_states_三态判断():
    """躺平→在床、坐起→坐起、空床→离床。"""
    ideal = _ideal()
    frames = np.array([ideal * 1.0, ideal * 0.4, ideal * 0.0])
    states = bm.detect_states(frames)
    assert states == [bm.IN_BED, bm.SITTING, bm.OUT_OF_BED]


def test_states_阈值边界():
    """用单点帧精确测边界：60%=在床、15%=坐起、14.9%=离床。"""
    # 峰值 = 100，各帧总压力（单点帧就是它自己的值）
    frames = np.array([[100.0], [60.0], [15.0], [14.9], [0.0]])
    states = bm.detect_states(frames)   # peak = 100
    assert states[0] == bm.IN_BED       # ratio 1.0
    assert states[1] == bm.IN_BED       # ratio 0.60，>= 0.60 算在床
    assert states[2] == bm.SITTING      # ratio 0.15，不 < 0.15，落在坐起档
    assert states[3] == bm.OUT_OF_BED   # ratio 0.149 < 0.15 算离床
    assert states[4] == bm.OUT_OF_BED   # ratio 0.0


def test_states_全空床不除零():
    """全程没人压（全是 0），峰值按 1 处理，不会除零报错，状态全判离床。"""
    frames = np.zeros((5, GRID_H, GRID_W))
    states = bm.detect_states(frames)
    assert all(s == bm.OUT_OF_BED for s in states)


# ============================================================================
# detect_exits：离床事件记录（只记录，不报警）
# ============================================================================


def test_exits_记录起止帧():
    """一段离床，正确记下起止帧和持续时间。"""
    # 索引 0~2 在床，3~7 离床（5 帧），8~11 回床
    states = [bm.IN_BED] * 3 + [bm.OUT_OF_BED] * 5 + [bm.IN_BED] * 4
    exits = bm.detect_exits(states, confirm_frames=1)
    assert exits == [(3, 8)]   # [3, 8)，持续 5 帧


def test_exits_过滤翻身毛刺():
    """只离床 1~2 帧（翻身毛刺），不算一次有效下床，被过滤。"""
    states = [bm.IN_BED] * 3 + [bm.OUT_OF_BED] * 2 + [bm.IN_BED] * 5
    exits = bm.detect_exits(states, confirm_frames=3)
    assert exits == []   # 2 帧 < 3 帧，忽略


def test_exits_多段离床都记录():
    """两段离床（中间回床），分别记录。"""
    states = (
        [bm.IN_BED] * 2 +        # 索引 0~1
        [bm.OUT_OF_BED] * 5 +    # 第一段离床：索引 2~6
        [bm.IN_BED] * 3 +        # 索引 7~9（回床）
        [bm.OUT_OF_BED] * 4 +    # 第二段离床：索引 10~13
        [bm.IN_BED] * 2          # 索引 14~15（回床）
    )
    exits = bm.detect_exits(states, confirm_frames=1)
    assert exits == [(2, 7), (10, 14)]


def test_exits_末尾还在离床也收尾():
    """序列结束时人还没回来，也要记下这最后一段。"""
    states = [bm.IN_BED] * 3 + [bm.OUT_OF_BED] * 6
    exits = bm.detect_exits(states, confirm_frames=1)
    assert exits == [(3, 9)]   # [3, 9)，持续 6 帧


# ============================================================================
# detect_alarms：超时未归才报警
# ============================================================================


def test_alarm_起夜不报警():
    """下床 8 帧（< 超时 10 帧）就回来，属于正常起夜，不报警。"""
    states = [bm.IN_BED] * 3 + [bm.OUT_OF_BED] * 8 + [bm.IN_BED] * 5
    alarms = bm.detect_alarms(states, away_timeout=10)
    assert not any(alarms)


def test_alarm_超时未归才报警():
    """下床连续 10 帧（= 超时）→ 第 10 帧拉响；回床立即解除。"""
    # 索引：0~2 在床，3~12 离床（10 帧），13~17 回床
    states = [bm.IN_BED] * 3 + [bm.OUT_OF_BED] * 10 + [bm.IN_BED] * 5
    alarms = bm.detect_alarms(states, away_timeout=10)
    assert alarms[11] is False   # 离床第 9 帧，还没到超时，不报警
    assert alarms[12] is True    # 离床第 10 帧，超时拉响
    assert alarms[13] is False   # 回床第一帧，解除


def test_alarm_坐起也算回来_解除报警():
    """报警后老人坐回床边（坐起），也算"回来了"，解除报警。"""
    # 索引：0~1 在床，2~11 离床（10 帧），12~14 坐起，15~17 回床
    states = (
        [bm.IN_BED] * 2 + [bm.OUT_OF_BED] * 10 +
        [bm.SITTING] * 3 + [bm.IN_BED] * 3
    )
    alarms = bm.detect_alarms(states, away_timeout=10)
    assert alarms[11] is True    # 离床第 10 帧，拉响
    assert alarms[12] is False   # 坐起 = 回来了，解除
    assert alarms[15] is False   # 回床，仍解除


def test_alarm_计时中途归零():
    """离床 9 帧 → 坐起 1 帧 → 再离床：计时清零，不会累加误报。"""
    # 先离床 9 帧（不够 10），坐起打断，再离床 5 帧
    states = (
        [bm.IN_BED] * 2 + [bm.OUT_OF_BED] * 9 +
        [bm.SITTING] * 1 + [bm.OUT_OF_BED] * 5 + [bm.IN_BED] * 2
    )
    alarms = bm.detect_alarms(states, away_timeout=10)
    # 坐起打断了计时，后面只有连续 5 帧离床，不够 10，全程不报警
    assert not any(alarms)
