#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
迟滞补偿的测试 —— 验证「自动判方向 + 修回来」这一整套动作正确。

重点测五件事：
  1. 方向判断：加压/减压/不变 三种情况判得对不对
  2. 死区：小于阈值的变化不该触发"方向反转"
  3. 躺下-起身模拟：形状对、加压段偏高、减压段偏低
  4. 自动补偿：无噪声时能精确还原真实压力
  5. 方向判反：会把数据越修越糟（证明"判方向"是迟滞补偿的灵魂）
"""

import numpy as np
import pytest

import calibration as cal
import pressure_simulator as ps
from sensor_config import (
    GRID_W, GRID_H, BODY_PARTS, HYSTERESIS, make_ideal_pressure,
)


def _ideal():
    """造一张理想体压图（标准答案），供多个测试复用。"""
    return make_ideal_pressure(GRID_W, GRID_H, BODY_PARTS)


# ============================================================================
# estimate_direction：方向判断
# ============================================================================


def test_estimate_direction_三种情况():
    """上升→加压、不变→0、下降→减压。"""
    prev = np.array([[0.0, 5.0, 8.0]])
    curr = np.array([[10.0, 5.0, 3.0]])
    d = cal.estimate_direction(prev, curr, deadband=0.0)
    assert d[0, 0] == 1    # 0→10 明显上升 → 加压
    assert d[0, 1] == 0    # 5→5 没变 → 0
    assert d[0, 2] == -1   # 8→3 明显下降 → 减压


def test_estimate_direction_死区():
    """变化小于死区阈值 = 不算转向；超过才算。"""
    prev = np.array([[5.0]])
    curr = np.array([[5.4]])
    # 变化 0.4 < 死区 0.5 → 当作没变
    assert cal.estimate_direction(prev, curr, deadband=0.5)[0, 0] == 0
    # 变化 0.4 > 死区 0.3 → 算上升
    assert cal.estimate_direction(prev, curr, deadband=0.3)[0, 0] == 1


# ============================================================================
# simulate_load_unload：躺下-起身模拟
# ============================================================================


def test_simulate_形状与段数():
    """返回三个量，帧数、形状都对。"""
    ideal = _ideal()
    n = 60
    true_frames, raw_frames, weights = ps.simulate_load_unload(ideal, n_steps=n)
    assert true_frames.shape == (n, GRID_H, GRID_W)
    assert raw_frames.shape == (n, GRID_H, GRID_W)
    assert len(weights) == n


def test_simulate_加压偏高减压偏低():
    """加压段读数 >= 真实，减压段读数 <= 真实（迟滞的方向性）。"""
    ideal = _ideal()
    n = 60
    true_frames, raw_frames, _ = ps.simulate_load_unload(ideal, n_steps=n)
    n_up = n // 2
    assert (raw_frames[:n_up] >= true_frames[:n_up]).all()   # 躺下：偏高
    assert (raw_frames[n_up:] <= true_frames[n_up:]).all()   # 起身：偏低


# ============================================================================
# correct_hysteresis_sequence：自动判方向 + 补偿
# ============================================================================


def test_补偿精确还原():
    """无噪声时，自动判方向 + 补偿应精确还原真实压力。"""
    ideal = _ideal()
    true_frames, raw_frames, _ = ps.simulate_load_unload(ideal, n_steps=60)
    corrected = cal.correct_hysteresis_sequence(raw_frames, HYSTERESIS, deadband=0.0)
    assert corrected.shape == raw_frames.shape
    np.testing.assert_allclose(corrected, true_frames, atol=1e-6)


def test_补偿后误差远小于补偿前():
    """全程平均误差：补偿后应该基本归零，补偿前明显偏大。"""
    ideal = _ideal()
    true_frames, raw_frames, _ = ps.simulate_load_unload(ideal, n_steps=60)
    corrected = cal.correct_hysteresis_sequence(raw_frames, HYSTERESIS, deadband=0.0)
    err_before = float(np.abs(raw_frames - true_frames).mean())
    err_after = float(np.abs(corrected - true_frames).mean())
    assert err_before > 1.0          # 迟滞宽度 8，平均误差应明显大于 1
    assert err_after < 1e-3          # 补偿后几乎归零


# ============================================================================
# 方向判反：迟滞补偿的灵魂是"判对方向"
# ============================================================================


def test_方向判反越修越糟():
    """把加压误判成减压来补，误差反而翻倍——证明方向不能判错。"""
    ideal = _ideal()
    # 纯加压场景：真实压力 = ideal，读数偏高 h/2
    raw = ideal + HYSTERESIS / 2.0
    # 正确：按 load 补，能还原；这里只验证"判反"的后果
    wrong = cal.correct_hysteresis(raw, "unload", HYSTERESIS)
    err_before = float(np.abs(raw - ideal).mean())   # = h/2
    err_wrong = float(np.abs(wrong - ideal).mean())  # ≈ h
    assert err_wrong > err_before                    # 越修越糟


def test_正确方向能还原():
    """对照：方向判对（load）就能精确还原，误差≈0。"""
    ideal = _ideal()
    raw = ideal + HYSTERESIS / 2.0
    right = cal.correct_hysteresis(raw, "load", HYSTERESIS)
    np.testing.assert_allclose(right, ideal, atol=1e-9)
