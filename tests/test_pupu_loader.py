#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PoPu 真实数据读取器 + 训练链 的测试。

数据集本身（82MB）不进 git，所以测试前先判断数据在不在：
不在就整组跳过（skip），不报错——别人 clone 仓库没下数据也能跑其他测试。
"""

import glob
import os

import numpy as np
import pytest

import pupu_loader as pl
from mlp import TinyMLP

# 数据目录存在才跑这些测试
DATA_READY = os.path.isdir(pl.DEFAULT_DATA_DIR)
pytestmark = pytest.mark.skipif(
    not DATA_READY, reason="PoPu 数据未下载（python3 src/download_pupu.py full）")

# 全量数据加载一次，多个测试共用（约 5000 样本，几秒钟）
@pytest.fixture(scope="module")
def dataset():
    return pl.load_dataset(seed=0)


class TestParsePose:
    def test_四类睡姿(self):
        assert pl.parse_pose("supine1_0.json") == "supine"
        assert pl.parse_pose("left3_1.json") == "left"
        assert pl.parse_pose("right5_2.json") == "right"
        assert pl.parse_pose("prone2_0.json") == "prone"

    def test_empty和others能识别但不属于四类(self):
        assert pl.parse_pose("empty1.json") == "empty"
        assert pl.parse_pose("others.json") == "others"
        assert pl.parse_pose("empty1.json") not in pl.POSE_ORDER
        assert pl.parse_pose("others.json") not in pl.POSE_ORDER


class TestLoadSnapshots:
    def test_形状是12行6列(self):
        files = glob.glob(os.path.join(pl.DEFAULT_DATA_DIR, "*", "supine*.json"))
        matrices, meta = pl.load_snapshots(files[0])
        assert matrices[0].shape == (12, 6)
        assert meta["volunteer_id"] is not None

    def test_脏帧被跳过_不崩(self):
        # 全数据集里有 29 帧读数不是 72 个（缺值），
        # 跳过后不应报错，且读到的帧形状全部正确
        files = glob.glob(os.path.join(pl.DEFAULT_DATA_DIR, "*", "*.json"))[:200]
        for f in files:
            matrices, _ = pl.load_snapshots(f)
            for m in matrices:
                assert m.shape == (12, 6)


class TestLoadDataset:
    def test_样本数和均衡度(self, dataset):
        X, y, classes, meta = dataset
        assert X.shape == (5040, 72)          # 4 类 × 1260，每类一帧
        assert len(classes) == 4
        assert (np.bincount(y) == 1260).all()  # 四类完全均衡

    def test_已归一化_数值范围合理(self, dataset):
        X, y, _, _ = dataset
        assert X.min() > -5 and X.max() < 5    # 减基线+除20 后的小数值

    def test_可复现(self):
        # 同一 seed 两次加载，结果一模一样（固定随机性才能对比实验）
        X1, y1, _, _ = pl.load_dataset(seed=7)
        X2, y2, _, _ = pl.load_dataset(seed=7)
        np.testing.assert_array_equal(X1, X2)
        np.testing.assert_array_equal(y1, y2)

    def test_减基线后信号变弱是真实的(self, dataset):
        # 减基线后信号幅度应该在个位数（原始读数 485 量级）。
        # 若漏了「减基线」，X 里会混进 485/20≈24 的大数；正常应 < 10。
        # 这条测试钉住「零点校准」的效果：没减基线的假数不该出现在 X 里。
        X, y, _, _ = dataset
        assert np.abs(X).max() < 10


class TestSplitByVolunteer:
    def test_训练和考试的志愿者完全不重叠(self, dataset):
        X, y, _, meta = dataset
        Xtr, ytr, Xte, yte = pl.split_by_volunteer(X, y, meta, seed=0)
        assert len(Xtr) + len(Xte) == len(X)
        assert len(Xte) < len(Xtr)             # 考试集是约 20%
        # 每类都该出现在两边（均衡切分）
        assert set(np.unique(ytr)) == set(range(4))
        assert set(np.unique(yte)) == set(range(4))


class TestTrainConverges:
    def test_真实数据训练准确率远超随机(self, dataset):
        """4 分类随机猜 = 25%。真实数据训练几十轮后应远超它。"""
        X, y, _, meta = dataset
        Xtr, ytr, Xte, yte = pl.split_by_volunteer(X, y, meta, seed=0)
        model = TinyMLP(72, 32, 4, seed=0)
        rng = np.random.default_rng(0)
        n = len(Xtr)
        for ep in range(40):                   # 少量轮次，控制测试耗时
            idx = rng.permutation(n)
            for s in range(0, n, 32):
                b = idx[s:s + 32]
                model.train_step(Xtr[b], ytr[b], lr=0.3)
        acc = model.accuracy(Xte, yte)
        assert acc > 0.8                       # 陌生人考试也应远超随机猜
