# -*- coding: utf-8 -*-
"""
测试「零点校准」脚本（src/zero_calibration.py）。

核心要验证的只有一条：校准（一个减法）确实把误差降下来了。
其余测试确保这个函数的行为是稳定、可复现、可信的。
"""
import numpy as np
import zero_calibration as zc


def test_校准后误差显著小于校准前():
    """核心断言：校准有效，校准后的误差一定比校准前小。"""
    r = zc.simulate_calibration(seed=42)
    assert r["err_after"] < r["err_before"]


def test_校准能消掉一大半误差():
    """实测能缩小 82%，这里只要求至少消掉 50%（留足余地，不 flaky）。"""
    r = zc.simulate_calibration(seed=42)
    improve = (r["err_before"] - r["err_after"]) / r["err_before"]
    assert improve > 0.5


def test_同一种子结果可复现():
    """固定随机种子后，跑两遍结果应该完全一致（否则测试本身不可信）。"""
    r1 = zc.simulate_calibration(seed=42)
    r2 = zc.simulate_calibration(seed=42)
    assert np.allclose(r1["raw"], r2["raw"])


def test_校准动作就是减法():
    """校准后 = 脏数据 - 基线，一步到位，没有别的花活。"""
    r = zc.simulate_calibration(seed=42)
    assert np.allclose(r["calibrated"], r["raw"] - r["baseline"])
