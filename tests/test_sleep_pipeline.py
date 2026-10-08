#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=============================================================================
test_sleep_pipeline.py —— 睡眠分期「打通」模块的测试
=============================================================================

测这些事：
  1. 合成整夜信号：形状对不对、能不能复现
  2. 从波形挖特征：形状正确、数值有限、体动方向对（清醒 > 深睡）
  3. 特征提取精度：呼吸率、心率挖出来离真实值很近（打通的核心证据）
  4. 端到端：挖出来的特征真能训练出比随机强的分期器
  5. 端到端整夜：对一整夜分期，κ 超过随机水平
"""

import numpy as np

import sleep_pipeline as sp
import bcg_monitor as bcg
from sleep_staging import standardize, cohens_kappa, STAGE_CENTERS


# ============================================================================
# 一、合成整夜信号
# ============================================================================

def test_simulate_信号形状():
    """信号长度 = 段数 × 30 秒 × 采样率，标签长度 = 段数。"""
    n_epochs = 50
    signal, y, names = sp.simulate_night_signal(n_epochs=n_epochs, seed=0)
    assert len(signal) == n_epochs * int(sp.EPOCH_SEC * sp.FS)
    assert y.shape == (n_epochs,)
    assert len(names) == n_epochs


def test_simulate_可复现():
    """同一 seed 两次合成，信号一模一样。"""
    s1, y1, _ = sp.simulate_night_signal(40, seed=7)
    s2, y2, _ = sp.simulate_night_signal(40, seed=7)
    np.testing.assert_array_equal(s1, s2)
    np.testing.assert_array_equal(y1, y2)


def test_simulate_不同seed不同():
    """不同 seed 生成不同的夜。"""
    _, y1, _ = sp.simulate_night_signal(100, seed=1)
    _, y2, _ = sp.simulate_night_signal(100, seed=2)
    assert not np.array_equal(y1, y2)


# ============================================================================
# 二、从波形挖特征
# ============================================================================

def test_extract_特征形状():
    """提取的特征矩阵形状正确：(段数, 5)，且都是有限数。"""
    signal, _, _ = sp.simulate_night_signal(30, seed=3)
    X = sp.extract_epoch_features(signal)
    assert X.shape == (30, 5)
    assert np.isfinite(X).all()


def test_extract_体动方向():
    """清醒段的体动能量应明显大于深睡段（体动的生理差异）。"""
    fs = sp.FS
    rng = np.random.default_rng(0)
    wake = sp._synthesize_epoch(STAGE_CENTERS["wake"][1], STAGE_CENTERS["wake"][2],
                                STAGE_CENTERS["wake"][3], STAGE_CENTERS["wake"][4],
                                STAGE_CENTERS["wake"][0] * sp.MOV_AMP_SCALE, fs, rng)
    deep = sp._synthesize_epoch(STAGE_CENTERS["deep"][1], STAGE_CENTERS["deep"][2],
                                STAGE_CENTERS["deep"][3], STAGE_CENTERS["deep"][4],
                                STAGE_CENTERS["deep"][0] * sp.MOV_AMP_SCALE, fs, rng)
    wake_mov = bcg.band_energy(wake - wake.mean(), fs, sp.MOV_LOW, sp.MOV_HIGH)
    deep_mov = bcg.band_energy(deep - deep.mean(), fs, sp.MOV_LOW, sp.MOV_HIGH)
    assert wake_mov > deep_mov


# ============================================================================
# 三、特征提取精度（打通的核心证据）
# ============================================================================

def test_提取呼吸率心率准确():
    """从固定信号挖出的呼吸率/心率，应接近真实值（30 秒窗分辨率约 2 次/分）。"""
    fs = 50
    t = np.arange(int(30 * fs)) / fs
    # 呼吸 15 次/分 + 心跳 72 次/分，无噪声
    sig = 100 + 5 * np.sin(2 * np.pi * (15 / 60) * t) \
             + 0.5 * np.sin(2 * np.pi * (72 / 60) * t)
    X = sp.extract_epoch_features(sig, fs=fs)
    assert X.shape == (1, 5)
    assert abs(X[0, 1] - 15.0) < 2.5   # 呼吸率
    assert abs(X[0, 2] - 72.0) < 3.0   # 心率


# ============================================================================
# 四、端到端：挖出的特征能训练出有用的分期器
# ============================================================================

def test_端到端分期准确率高于随机():
    """从波形挖特征 → 训练 → 考试，准确率应远超四类随机猜的 25%。"""
    from mlp import TinyMLP

    X, y = sp.extract_dataset(n_nights=4, n_epochs=120, seed=42)
    X_scaled, _, _ = standardize(X)

    rng = np.random.default_rng(42)
    idx = rng.permutation(len(X))
    n_test = int(len(X) * 0.2)
    X_train, y_train = X_scaled[idx[n_test:]], y[idx[n_test:]]
    X_test, y_test = X_scaled[idx[:n_test]], y[idx[:n_test]]

    model = TinyMLP(n_input=5, n_hidden=16, n_output=4, seed=42)
    n = len(X_train)
    for _ in range(100):
        p = rng.permutation(n)
        for start in range(0, n, 32):
            model.train_step(X_train[p[start:start + 32]],
                             y_train[p[start:start + 32]], lr=0.3)

    acc = model.accuracy(X_test, y_test)
    assert acc > 0.6   # 随机猜 25%，能到 60%+ 才说明「打通」真有用


def test_整夜分期kappa高于随机():
    """对一整夜端到端分期，κ 应远超随机（0 附近）。"""
    from mlp import TinyMLP

    # 训练一个小模型
    X, y = sp.extract_dataset(n_nights=4, n_epochs=120, seed=42)
    X_scaled, mean, std = standardize(X)
    rng = np.random.default_rng(42)
    model = TinyMLP(5, 16, 4, seed=42)
    n = len(X_scaled)
    for _ in range(100):
        p = rng.permutation(n)
        for start in range(0, n, 32):
            model.train_step(X_scaled[p[start:start + 32]],
                             y[p[start:start + 32]], lr=0.3)

    # 对一整夜分期
    signal, y_night, _ = sp.simulate_night_signal(240, seed=999)
    preds = sp.stage_from_signal(signal, model, mean, std)
    kappa = cohens_kappa(y_night, preds, 4)
    assert kappa > 0.5
