#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""迷你神经网络的测试。"""

import numpy as np
import pytest

from mlp import relu, softmax, TinyMLP


class TestActivations:
    def test_relu(self):
        np.testing.assert_allclose(relu(np.array([-2, 0, 3])), [0, 0, 3])

    def test_softmax_sums_to_one(self):
        probs = softmax(np.array([[1.0, 2.0, 3.0], [0.0, 0.0, 0.0]]))
        np.testing.assert_allclose(probs.sum(axis=1), [1.0, 1.0])
        assert (probs > 0).all()

    def test_softmax_order_preserved(self):
        """分数大的，概率也要大（softmax 只变尺度不变排序）。"""
        probs = softmax(np.array([[1.0, 3.0, 2.0]]))
        assert probs[0, 1] > probs[0, 2] > probs[0, 0]


class TestTinyMLP:
    def test_xor_learnable(self):
        """异或问题：非线性的经典小考题，能学会说明反向传播没写错。"""
        X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
        y = np.array([0, 1, 1, 0])          # 异或：相同为 0，不同为 1
        model = TinyMLP(n_input=2, n_hidden=8, n_output=2, seed=0)
        for _ in range(500):
            model.train_step(X, y, lr=1.0)
        assert model.accuracy(X, y) == 1.0

    def test_loss_decreases(self):
        """训练应该让损失下降（网络在"学"，不是在"抽风"）。"""
        rng = np.random.default_rng(0)
        X = rng.normal(0, 1, size=(60, 10))
        y = (X[:, 0] > 0).astype(int)       # 简单规则：第一列正负决定类别
        model = TinyMLP(n_input=10, n_hidden=16, n_output=2, seed=0)
        first = model.train_step(X, y, lr=0.5)
        for _ in range(200):
            last = model.train_step(X, y, lr=0.5)
        assert last < first * 0.5           # 损失至少降到一半

    def test_predict_shape(self):
        model = TinyMLP(n_input=96, n_hidden=8, n_output=3, seed=0)
        preds = model.predict(np.zeros((5, 96)))
        assert preds.shape == (5,)
        assert set(preds).issubset({0, 1, 2})
