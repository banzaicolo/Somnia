# -*- coding: utf-8 -*-
"""
测试「模拟压力传感器」脚本（src/pressure_simulator.py）。

重点验证三件事：
1. 理想体压图是"摆好的人形"，不是随机数（形状对、臀部最重）
2. 加了 5 种传感器毛病后，读数确实被"污染"了（跟理想不一样、无人区被抬高）
3. 迟滞现象：同一压力，加压读数 > 减压读数
"""
import numpy as np
import pressure_simulator as ps


def test_理想体压图形状正确():
    """理想图应该是 8 行 × 12 列（=96 个点），且压力不会出现负数。"""
    ideal = ps.make_ideal_pressure(ps.GRID_W, ps.GRID_H, ps.BODY_PARTS)
    assert ideal.shape == (ps.GRID_H, ps.GRID_W)
    assert ideal.min() >= 0


def test_臀部是全场最重():
    """峰值应该出现在臀部附近（第 5~6 行，即索引 4~5）。"""
    ideal = ps.make_ideal_pressure(ps.GRID_W, ps.GRID_H, ps.BODY_PARTS)
    row, _col = np.unravel_index(np.argmax(ideal), ideal.shape)
    assert 4 <= row <= 5


def test_加了毛病之后读数跟理想不一样():
    """传感器毛病一定会让最终读数偏离理想图。"""
    ideal = ps.make_ideal_pressure(ps.GRID_W, ps.GRID_H, ps.BODY_PARTS)
    steps = ps.add_imperfections(ideal)
    final = list(steps.values())[-1]
    assert not np.allclose(final, ideal)


def test_没人压的角落被抬高了():
    """左上角没人压，理想值接近 0，加毛病后应明显大于 0（这就是"骗人"）。"""
    ideal = ps.make_ideal_pressure(ps.GRID_W, ps.GRID_H, ps.BODY_PARTS)
    steps = ps.add_imperfections(ideal)
    final = list(steps.values())[-1]
    assert ideal[0, 0] < 5          # 角上没人压，理想值很小（只有高斯鼓包尾巴的残留）
    assert final[0, 0] > ideal[0, 0]  # 加毛病后被抬高了


def test_迟滞_加压高于减压():
    """迟滞回线：同一压力，加压读数 >= 减压读数（两端相等，中间严格大于）。"""
    p, load, unload = ps.hysteresis_curve()
    assert p.shape == load.shape == unload.shape
    assert (load >= unload).all()            # 端点两者相等（迟滞在两端收拢为 0）
    assert (load[1:-1] > unload[1:-1]).all()  # 中间段加压严格大于减压


def test_身体部位正好四个():
    """人体是头、肩胛、臀、脚后跟四个部位拼出来的。"""
    assert len(ps.BODY_PARTS) == 4
