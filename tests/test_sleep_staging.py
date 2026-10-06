#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=============================================================================
test_sleep_staging.py —— 睡眠分期模块的测试
=============================================================================

测这些事：
  1. 合成的一整夜：形状对不对、能不能复现、四类齐不齐
  2. 特征数值在合理范围（体动非负、心率呼吸有下限）
  3. 评估指标：混淆矩阵算得对不对、Cohen's Kappa 两个极端（全对=1、乱猜≈0）
  4. 分类器真的能学会分期（准确率远超随机猜的 25%）
  5. 特征标准化后均值 0、标准差 1
"""

import numpy as np

import sleep_staging as ss


# ============================================================================
# 一、合成一整夜
# ============================================================================

def test_synthesize_night_形状():
    """输出形状正确：X 是 (N,5)，y 是 (N,)，names 是四类名。"""
    X, y, names = ss.synthesize_night(100, seed=0)
    assert X.shape == (100, 5)
    assert y.shape == (100,)
    assert names == ss.STAGES


def test_synthesize_night_可复现():
    """同一 seed 两次合成，结果一模一样（方便对比、排查）。"""
    X1, y1, _ = ss.synthesize_night(100, seed=7)
    X2, y2, _ = ss.synthesize_night(100, seed=7)
    np.testing.assert_array_equal(X1, X2)
    np.testing.assert_array_equal(y1, y2)


def test_synthesize_night_不同seed结果不同():
    """不同 seed 应生成不同的夜（否则合成是假的）。"""
    _, y1, _ = ss.synthesize_night(200, seed=1)
    _, y2, _ = ss.synthesize_night(200, seed=2)
    assert not np.array_equal(y1, y2)


def test_睡眠结构包含四类():
    """足够长的一夜，四个阶段（醒/浅/深/做梦）都应该出现。"""
    _, y, _ = ss.synthesize_night(500, seed=3)
    assert set(y.tolist()) == {0, 1, 2, 3}


def test_特征范围合理():
    """特征数值不越界：体动非负、心率≥30、呼吸率≥4（合成时夹过的下限）。"""
    X, _, _ = ss.synthesize_night(300, seed=4)
    assert (X[:, 0] >= 0).all()      # 体动幅度
    assert (X[:, 2] >= 30).all()     # 心率
    assert (X[:, 1] >= 4).all()      # 呼吸率


# ============================================================================
# 二、评估指标
# ============================================================================

def test_confusion_matrix():
    """混淆矩阵按「真实→预测」正确累加。"""
    y_true = np.array([0, 0, 1, 1, 2, 2, 3, 3])
    y_pred = np.array([0, 1, 1, 1, 2, 3, 3, 3])
    cm = ss.confusion_matrix(y_true, y_pred, 4)
    assert cm.shape == (4, 4)
    assert cm[0, 0] == 1   # 真实0 猜成0：1 个
    assert cm[0, 1] == 1   # 真实0 猜成1：1 个
    assert cm[1, 1] == 2   # 真实1 猜成1：2 个
    assert cm[2, 2] == 1
    assert cm[2, 3] == 1
    assert cm[3, 3] == 2


def test_kappa_完全一致():
    """预测和真实完全一样 → κ 必须是 1.0（满分）。"""
    y = np.array([0, 1, 2, 3] * 10)
    assert ss.cohens_kappa(y, y, 4) == 1.0


def test_kappa_随机接近零():
    """纯乱猜 → κ 应接近 0（没本事，扣掉蒙对的）。"""
    rng = np.random.default_rng(0)
    y_true = np.array([0, 1, 2, 3] * 50)
    y_pred = rng.integers(0, 4, size=len(y_true))
    k = ss.cohens_kappa(y_true, y_pred, 4)
    assert abs(k) < 0.15


# ============================================================================
# 三、分类器真能学会（端到端）
# ============================================================================

def test_分类器准确率高于随机():
    """用少量数据快训练，分类器准确率应远超随机猜的 25%。"""
    from sleep_staging import generate_dataset, standardize
    from mlp import TinyMLP

    X, y = generate_dataset(n_nights=3, n_epochs=120, seed=42)
    X_scaled, _, _ = standardize(X)

    rng = np.random.default_rng(42)
    idx = rng.permutation(len(X))
    n_test = int(len(X) * 0.2)
    X_train, y_train = X_scaled[idx[n_test:]], y[idx[n_test:]]
    X_test, y_test = X_scaled[idx[:n_test]], y[idx[:n_test]]

    model = TinyMLP(n_input=5, n_hidden=16, n_output=4, seed=42)
    n = len(X_train)
    for _ in range(80):
        p = rng.permutation(n)
        for start in range(0, n, 32):
            model.train_step(X_train[p[start:start + 32]],
                             y_train[p[start:start + 32]], lr=0.3)

    acc = model.accuracy(X_test, y_test)
    assert acc > 0.5   # 四类随机猜 25%，能学到 50%+ 才说明真学到了


# ============================================================================
# 四、特征标准化
# ============================================================================

def test_standardize():
    """标准化后每列均值≈0、标准差≈1（机器学习标准预处理）。"""
    X = np.array([
        [0.0, 10, 60],
        [1.0, 20, 80],
        [2.0, 30, 70],
        [1.5, 15, 55],
    ], dtype=float)
    Xs, mean, std = ss.standardize(X)
    np.testing.assert_allclose(Xs.mean(axis=0), 0.0, atol=1e-9)
    np.testing.assert_allclose(Xs.std(axis=0), 1.0, atol=1e-9)
    assert mean.shape == (3,) and std.shape == (3,)


def test_standardize_常量列不除零():
    """某列全一样（标准差 0）时，不能除以 0，应安全通过。"""
    X = np.array([
        [0.0, 10, 60],
        [1.0, 10, 80],
        [2.0, 10, 70],
    ], dtype=float)
    Xs, mean, std = ss.standardize(X)
    assert np.isfinite(Xs).all()   # 没有 NaN 或 inf
