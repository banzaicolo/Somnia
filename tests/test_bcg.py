#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BCG 呼吸/心跳监测（bcg_monitor）的测试 —— 验证「合成信号 + 提取 + 暂停检测」这套逻辑。

重点测五件事：
  1. 合成信号的形状（时长 × 采样率）
  2. 提取呼吸率：对齐（整数次/分）时精确、非整数时在容差内
  3. 提取心率：对齐时精确、加噪声仍准确
  4. 呼吸暂停检测：正常段不误报、暂停段被揪出
  5. 全零信号不除零报错
"""

import numpy as np

import bcg_monitor as bcg


# ============================================================================
# simulate_bcg：合成信号
# ============================================================================


def test_simulate_形状():
    """10 秒 × 50 Hz = 500 个点，时间轴和信号等长。"""
    t, sig = bcg.simulate_bcg(duration_sec=10, fs=50)
    assert len(t) == 500
    assert len(sig) == 500


# ============================================================================
# estimate_breathing_rate / estimate_heart_rate：提取
# ============================================================================


def test_提取呼吸率_对齐精确():
    """无噪声 + 整数呼吸率（15 次/分），提取应几乎精确。"""
    _, sig = bcg.simulate_bcg(duration_sec=60, resp_rate=15, heart_rate=72,
                              noise_std=0)
    est = bcg.estimate_breathing_rate(sig)
    assert abs(est - 15.0) < 0.5


def test_提取心率_对齐精确():
    """无噪声 + 整数心率（72 次/分），提取应几乎精确。"""
    _, sig = bcg.simulate_bcg(duration_sec=60, resp_rate=15, heart_rate=72,
                              noise_std=0)
    est = bcg.estimate_heart_rate(sig)
    assert abs(est - 72.0) < 0.5


def test_提取_噪声下仍准确():
    """加了噪声，呼吸率和心率仍能提取（误差 < 1 次/分）。"""
    _, sig = bcg.simulate_bcg(duration_sec=60, resp_rate=15, heart_rate=72,
                              noise_std=0.1)
    assert abs(bcg.estimate_breathing_rate(sig) - 15.0) < 1.0
    assert abs(bcg.estimate_heart_rate(sig) - 72.0) < 1.0


def test_提取_非整数呼吸率在容差内():
    """非整数呼吸率（14.5 次/分）会频谱泄漏，但主峰仍在附近（误差 < 1.5）。"""
    _, sig = bcg.simulate_bcg(duration_sec=60, resp_rate=14.5, heart_rate=72,
                              noise_std=0)
    est = bcg.estimate_breathing_rate(sig)
    assert abs(est - 14.5) < 1.5


def test_提取_不同心率也准():
    """换一个心率（68 次/分），验证不是只对 72 这一个数碰巧。"""
    _, sig = bcg.simulate_bcg(duration_sec=60, resp_rate=15, heart_rate=68,
                              noise_std=0)
    est = bcg.estimate_heart_rate(sig)
    assert abs(est - 68.0) < 0.5


# ============================================================================
# detect_apnea：呼吸暂停检测
# ============================================================================


def test_暂停检测_正常段不误报():
    """全程正常呼吸，不该误报任何暂停。"""
    _, sig = bcg.simulate_bcg(duration_sec=30, resp_rate=15, heart_rate=72,
                              noise_std=0.05)
    flags, _ = bcg.detect_apnea(sig, window_sec=10)
    assert flags == [False, False, False]


def test_暂停检测_检出暂停():
    """中间 10 秒暂停，正好被揪出来。"""
    _, sig = bcg.simulate_bcg(duration_sec=30, resp_rate=15, heart_rate=72,
                              noise_std=0.05, apnea=(10, 20))
    flags, _ = bcg.detect_apnea(sig, window_sec=10)
    # 30 秒 → 3 个窗口：[0,10) 正常 / [10,20) 暂停 / [20,30) 正常
    assert flags == [False, True, False]


# ============================================================================
# 边界：全零信号
# ============================================================================


def test_全零信号不除零():
    """全程没人（信号全 0），提取返回 0、暂停检测不崩。"""
    sig = np.zeros(3000)   # 60 秒 × 50 Hz
    assert bcg.estimate_breathing_rate(sig) == 0.0
    assert bcg.estimate_heart_rate(sig) == 0.0
    flags, _ = bcg.detect_apnea(sig)
    assert flags == [False] * 6   # 6 个 10 秒窗口，能量全 0，不判暂停
