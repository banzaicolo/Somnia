#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
verdict 模块的测试 —— 验证「数字 → 结论」的判断规则正确。

重点测两件事：
  1. 每个函数在每个分档区间，返回的字样对不对（阈值边界尤其要测）
  2. render_pose_verdict 综合函数，返回的行数和内容对不对

为什么给「结论生成器」写测试？因为它是一堆 if 判断，
边界条件最容易写错。而且一旦有人调了阈值（比如把 80 改成 90），
测试会立刻报错提醒他「你改规则了，看看是不是有意为之」。
"""

from verdict import (
    describe_improvement,
    describe_robustness,
    describe_rescue,
    describe_recovery,
    render_pose_verdict,
)


# ============================================================================
# describe_improvement：改善程度
# ============================================================================


def test_improvement_negative():
    """改善为负 = 误差不降反升。"""
    assert "误差不降反升" in describe_improvement(-5.0)


def test_improvement_zero():
    """0% = 几乎无改善。"""
    assert "几乎无改善" in describe_improvement(0.0)


def test_improvement_low_band():
    """20% 以下 = 几乎无改善。"""
    assert "几乎无改善" in describe_improvement(10.0)


def test_improvement_mid_band():
    """20%~50% = 改善有限。"""
    assert "改善有限" in describe_improvement(30.0)


def test_improvement_high_band():
    """50%~80% = 明显改善。"""
    assert "明显改善" in describe_improvement(60.0)


def test_improvement_clean():
    """80% 以上 = 基本消除。"""
    assert "基本消除" in describe_improvement(82.0)


def test_improvement_boundary_80():
    """刚好 80%（边界）也算「基本消除」。"""
    assert "基本消除" in describe_improvement(80.0)


# ============================================================================
# describe_robustness：稳健性
# ============================================================================


def test_robustness_very():
    """掉分 < 5 = 非常稳健。"""
    assert "非常稳健" in describe_robustness(2.0)


def test_robustness_fairly():
    """掉分 5~15 = 较为稳健。"""
    assert "较为稳健" in describe_robustness(10.0)


def test_robustness_hurt():
    """掉分 15~30 = 性能明显下降。"""
    assert "性能明显下降" in describe_robustness(20.0)


def test_robustness_collapse():
    """掉分 >= 30 = 性能严重下降。"""
    assert "性能严重下降" in describe_robustness(40.0)


def test_robustness_negative_treated_as_zero():
    """掉分为负（脏数据反而更高分）按 0 处理 = 非常稳健。"""
    assert "非常稳健" in describe_robustness(-3.0)


# ============================================================================
# describe_rescue：标定救援效果
# ============================================================================


def test_rescue_backfire():
    """救回为负 = 未起正面作用。"""
    assert "未起正面作用" in describe_rescue(-2.0)


def test_rescue_none():
    """救回 < 1 = 几乎没有改善。"""
    assert "几乎没有改善" in describe_rescue(0.3)


def test_rescue_little():
    """救回 1~5 = 轻微改善。"""
    assert "轻微改善" in describe_rescue(3.0)


def test_rescue_clear():
    """救回 5~15 = 明显改善。"""
    assert "明显改善" in describe_rescue(10.0)


def test_rescue_boundary_5():
    """刚好 5.0 分（边界）：< 5 才算轻微改善，5.0 已进入「明显改善」。"""
    assert "明显改善" in describe_rescue(5.0)
    assert "轻微改善" in describe_rescue(4.9)


def test_rescue_big():
    """救回 >= 15 = 大幅改善。"""
    assert "大幅改善" in describe_rescue(20.0)


# ============================================================================
# describe_recovery：恢复程度
# ============================================================================


def test_recovery_full():
    """残留 < 1 = 基本完全恢复。"""
    assert "基本完全恢复" in describe_recovery(0.2)


def test_recovery_almost():
    """残留 1~5 = 基本恢复，仍有少量残余误差。"""
    assert "仍有少量残余" in describe_recovery(3.0)


def test_recovery_incomplete():
    """残留 >= 5 = 尚未完全恢复，且带具体分数。"""
    out = describe_recovery(8.0)
    assert "尚未完全恢复" in out
    assert "8.0" in out


# ============================================================================
# render_pose_verdict：综合结论
# ============================================================================


def test_render_returns_three_lines():
    """综合结论应返回 3 行，且每行都带上了对应的数字。"""
    lines = render_pose_verdict(0.99, 0.88, 0.99)
    assert isinstance(lines, list)
    assert len(lines) == 3
    # 第一行讲稳健性（掉分 11.0）
    assert "11.0" in lines[0]
    # 第二行讲标定效果（救回 11.0）
    assert "11.0" in lines[1]
    # 第三行讲恢复程度（残留 0.0）
    assert "0.0" in lines[2]


def test_render_full_recovery_scenario():
    """干净 99%、脏 88%、洗回 99%：应得到「稳健+明显改善+完全恢复」的组合。"""
    lines = render_pose_verdict(0.99, 0.88, 0.99)
    assert "较为稳健" in lines[0]        # 掉 11 分 → 5~15 档
    assert "明显改善" in lines[1]         # 救回 11 分 → 5~15 档
    assert "基本完全恢复" in lines[2]     # 残留 0 分


def test_render_degraded_scenario():
    """干净 99%、脏 50%、洗回 55%：性能严重下降、救回 5 分（明显改善）、尚未完全恢复。"""
    lines = render_pose_verdict(0.99, 0.50, 0.55)
    assert "性能严重下降" in lines[0]     # 掉 49 分
    assert "明显改善" in lines[1]         # 救回 5.0 分 → 落在「明显改善」档
    assert "尚未完全恢复" in lines[2]     # 残留 44 分
