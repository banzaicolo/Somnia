#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""睡姿数据集生成器的测试。"""

import numpy as np

from pose_dataset import (
    make_pose_pressure, generate_dataset, split_train_test, NORMALIZE,
)
from sensor_config import GRID_W, GRID_H, POSES, POSE_NAMES


class TestMakePosePressure:
    def test_shape(self):
        """三种睡姿生成的图都是 H×W。"""
        rng = np.random.default_rng(0)
        for pose in POSE_NAMES:
            img = make_pose_pressure(pose, rng)
            assert img.shape == (GRID_H, GRID_W)

    def test_not_random_garbage(self):
        """生成的图要有真实结构：最大压力明显大于最小压力。"""
        rng = np.random.default_rng(0)
        for pose in POSE_NAMES:
            img = make_pose_pressure(pose, rng)
            assert img.max() > 20          # 有地方被压到 20 以上
            assert img.min() < img.max() / 3   # 也有几乎没人压的地方

    def test_side_pose_asymmetric(self):
        """侧卧的特征：压力中心偏向一侧。左右两半的压力和应该差很多。"""
        rng = np.random.default_rng(0)
        img = make_pose_pressure("side", rng, jitter=False)
        left_sum = img[:, : GRID_W // 2].sum()
        right_sum = img[:, GRID_W // 2:].sum()
        assert right_sum > left_sum * 1.5   # 侧卧偏右侧（x=9），右半明显更重

    def test_supine_symmetric(self):
        """仰卧的特征：左右基本对称（中轴线两侧压力和接近）。"""
        rng = np.random.default_rng(0)
        img = make_pose_pressure("supine", rng, jitter=False)
        left_sum = img[:, : GRID_W // 2].sum()
        right_sum = img[:, GRID_W // 2:].sum()
        # 两个 40+ 压力的鼓包叠起来，左右差不该超过 10%
        assert abs(left_sum - right_sum) / max(left_sum, right_sum) < 0.1


class TestGenerateDataset:
    def test_shapes_and_labels(self):
        X, y, classes = generate_dataset(n_per_class=10, seed=0)
        assert X.shape == (30, GRID_W * GRID_H)   # 3 类 × 10 = 30 个样本
        assert y.shape == (30,)
        assert classes == POSE_NAMES
        assert set(y) == {0, 1, 2}

    def test_normalized(self):
        """归一化后数值应该在合理小范围（不爆表）。"""
        X, _, _ = generate_dataset(n_per_class=5, seed=0)
        assert X.max() < 2.0   # 除以 200 之后，不应有特别大的数

    def test_seed_reproducible(self):
        """固定种子 → 两次生成的数据一模一样（可复现）。"""
        X1, y1, _ = generate_dataset(n_per_class=5, seed=7)
        X2, y2, _ = generate_dataset(n_per_class=5, seed=7)
        np.testing.assert_allclose(X1, X2)
        np.testing.assert_array_equal(y1, y2)


class TestSplit:
    def test_ratio_and_no_overlap(self):
        X, y, _ = generate_dataset(n_per_class=50, seed=0)
        Xtr, ytr, Xte, yte = split_train_test(X, y, test_ratio=0.2, seed=0)
        assert len(Xte) == 30        # 150 × 0.2 = 30
        assert len(Xtr) == 120
        # 考试卷的样本不能在训练集里出现过（不能泄题）
        assert set(map(tuple, Xte)).isdisjoint(set(map(tuple, Xtr)))
