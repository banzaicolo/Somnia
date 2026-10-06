# -*- coding: utf-8 -*-
"""
测试「人形证明图」脚本（src/body_shape_proof.py）。

重点验证：模拟的人形是"按人体结构摆好的"，不是随机滚出来的。
"""
import numpy as np
import body_shape_proof as bp


def test_标准答案形状正确():
    """理想图应该是 8 行 × 12 列。"""
    ideal = bp.make_ideal_pressure(bp.GRID_W, bp.GRID_H, bp.BODY_PARTS)
    assert ideal.shape == (bp.GRID_H, bp.GRID_W)


def test_四个部位从上到下排列():
    """行号（y 坐标）应该严格递增：头 → 肩胛 → 臀 → 脚跟。"""
    ys = [part[2] for part in bp.BODY_PARTS]
    assert ys == sorted(ys)


def test_臀部峰值最大():
    """四个部位里，峰值压力最大的应该是臀部（仰卧时最重的地方）。"""
    amps = [part[5] for part in bp.BODY_PARTS]
    heaviest = bp.BODY_PARTS[amps.index(max(amps))]
    assert heaviest[0] == "臀部"
