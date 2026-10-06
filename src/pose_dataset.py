#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=============================================================================
睡姿数据集生成器 —— 没有真数据时，"造"出训练数据
=============================================================================

【干嘛的？】

训练"睡姿分类器"需要大量样本：很多人、各种睡姿、各种身高体重。
没有真硬件、没有真数据时，就用模拟器"造"：

    在三种睡姿（仰卧/侧卧/俯卧）的基础上，每个样本随机抖动
    —— 位置偏一点、压力变一点、胖瘦不同 ——
    就像 1000 个不同的人躺上去。机器学习的"数据增强"就是这么回事。

【产出什么？】

generate_dataset(n_per_class=200) 会返回：
    X —— (样本数, 96) 的二维数组。每个样本是一张 8×12 压力图摊平成 96 个数
    y —— (样本数,) 的类别编号（0=仰卧, 1=侧卧, 2=俯卧）

=============================================================================
"""

import numpy as np

from sensor_config import (
    GRID_W, GRID_H, POSES, POSE_NAMES,
    make_ideal_pressure,
)


# 归一化基准：压力图数值大约 0~150，除以 200 拉到 0~0.75 之间。
# 神经网络喜欢"小而均匀"的输入，太大的数会让训练不稳定。
NORMALIZE = 200.0


def make_pose_pressure(pose, rng, jitter=True):
    """
    生成一张指定睡姿的压力图。

    参数：
      pose   —— 睡姿名："supine" / "side" / "prone"
      rng    —— 随机数生成器（np.random.default_rng(种子)）
      jitter —— 是否加随机抖动。True = 模拟"不同的人"，
                False = 标准姿势（画示意图用）

    返回：(GRID_H, GRID_W) 的压力图。
    """
    parts = POSES[pose]          # 拿到这个睡姿的"身体部位表"

    if jitter:
        # 每个部位做两处随机抖动，模拟"不同的人躺上去"：
        #   位置抖 ±0.5 格 —— 有人睡得靠上、有人靠下、有人歪一点
        #   峰值抖 ±20%   —— 有人重有人轻、胖瘦不同、床垫软硬不同
        parts = [(name,
                  cx + rng.uniform(-0.5, 0.5),          # 位置抖
                  cy + rng.uniform(-0.5, 0.5),
                  sx, sy,
                  amp * rng.uniform(0.8, 1.2),          # 峰值抖
                  ang)
                 for name, cx, cy, sx, sy, amp, ang in parts]

    # 用公共的"理想体压"函数把部位叠成一张图
    return make_ideal_pressure(GRID_W, GRID_H, parts)


def generate_dataset(n_per_class, seed=0, shuffle=True):
    """
    生成完整的合成数据集。

    参数：
      n_per_class —— 每种睡姿生成多少个样本（比如 200，总共 600）
      seed        —— 随机种子。固定它，每次生成的数据一样，方便对比
      shuffle     —— 是否打乱顺序。训练集一般要打乱（不然网络先学完
                     全部仰卧再学侧卧，会"偏科"）

    返回：
      X —— (N, 96) 的二维数组，每行是一张摊平的压力图（已归一化到 0~1 左右）
      y —— (N,) 的类别编号数组
      classes —— 类别名列表 ["supine", "side", "prone"]
    """
    rng = np.random.default_rng(seed)

    X_list, y_list = [], []
    for label, pose in enumerate(POSE_NAMES):   # label: 0/1/2
        for _ in range(n_per_class):
            img = make_pose_pressure(pose, rng, jitter=True)
            X_list.append(img.ravel())          # 8×12 摊平成 96
            y_list.append(label)

    X = np.array(X_list) / NORMALIZE            # 归一化
    y = np.array(y_list)

    if shuffle:
        idx = rng.permutation(len(X))           # 生成打乱的下标
        X, y = X[idx], y[idx]

    return X, y, list(POSE_NAMES)


def split_train_test(X, y, test_ratio=0.2, seed=0):
    """
    把数据切成"训练集"和"考试卷"。

    为什么要分？——用训练数据考试等于"抄原题"，分数虚高。
    留一部分模型从没见过的数据来考，才叫真本事。
    """
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(X))
    n_test = int(len(X) * test_ratio)

    test_idx, train_idx = idx[:n_test], idx[n_test:]
    return X[train_idx], y[train_idx], X[test_idx], y[test_idx]
