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
    """改善为负 = 越修越糟。"""
    assert "越修越糟" in describe_improvement(-5.0)


def test_improvement_zero():
    """0% = 几乎没改善。"""
    assert "几乎没改善" in describe_improvement(0.0)


def test_improvement_low_band():
    """20% 以下 = 几乎没改善。"""
    assert "几乎没改善" in describe_improvement(10.0)


def test_improvement_mid_band():
    """20%~50% = 改善有限。"""
    assert "改善有限" in describe_improvement(30.0)


def test_improvement_high_band():
    """50%~80% = 明显改善。"""
    assert "明显改善" in describe_improvement(60.0)


def test_improvement_clean():
    """80% 以上 = 基本修干净。"""
    assert "基本修干净" in describe_improvement(82.0)


def test_improvement_boundary_80():
    """刚好 80%（边界）也算「基本修干净」。"""
    assert "基本修干净" in describe_improvement(80.0)


# ============================================================================
# describe_robustness：鲁棒性
# ============================================================================


def test_robustness_very():
    """掉分 < 5 = 非常鲁棒。"""
    assert "非常鲁棒" in describe_robustness(2.0)


def test_robustness_fairly():
    """掉分 5~15 = 比较鲁棒。"""
    assert "比较鲁棒" in describe_robustness(10.0)


def test_robustness_hurt():
    """掉分 15~30 = 明显受伤。"""
    assert "明显受伤" in describe_robustness(20.0)


def test_robustness_collapse():
    """掉分 >= 30 = 严重退化。"""
    assert "严重退化" in describe_robustness(40.0)


def test_robustness_negative_treated_as_zero():
    """掉分为负（脏数据反而更高分）按 0 处理 = 非常鲁棒。"""
    assert "非常鲁棒" in describe_robustness(-3.0)


# ============================================================================
# describe_rescue：标定救援效果
# ============================================================================


def test_rescue_backfire():
    """救回为负 = 帮倒忙。"""
    assert "帮了倒忙" in describe_rescue(-2.0)


def test_rescue_none():
    """救回 < 1 = 没帮上忙。"""
    assert "没帮上忙" in describe_rescue(0.3)


def test_rescue_little():
    """救回 1~5 = 帮了一点小忙。"""
    assert "帮了一点小忙" in describe_rescue(3.0)


def test_rescue_clear():
    """救回 5~15 = 有明显帮助。"""
    assert "有明显帮助" in describe_rescue(10.0)


def test_rescue_boundary_5():
    """刚好 5.0 分（边界）：< 5 才算帮小忙，5.0 已进入「明显帮助」。"""
    assert "有明显帮助" in describe_rescue(5.0)
    assert "帮了一点小忙" in describe_rescue(4.9)


def test_rescue_big():
    """救回 >= 15 = 大幅救回。"""
    assert "大幅救回" in describe_rescue(20.0)


# ============================================================================
# describe_recovery：恢复程度
# ============================================================================


def test_recovery_full():
    """残留 < 1 = 基本完全恢复。"""
    assert "基本完全恢复" in describe_recovery(0.2)


def test_recovery_almost():
    """残留 1~5 = 基本恢复，只剩一点尾巴。"""
    assert "只剩一点小尾巴" in describe_recovery(3.0)


def test_recovery_incomplete():
    """残留 >= 5 = 还没完全恢复，且带具体分数。"""
    out = describe_recovery(8.0)
    assert "还没完全恢复" in out
    assert "8.0" in out


# ============================================================================
# render_pose_verdict：综合结论
# ============================================================================


def test_render_returns_three_lines():
    """综合结论应返回 3 行，且每行都带上了对应的数字。"""
    lines = render_pose_verdict(0.99, 0.88, 0.99)
    assert isinstance(lines, list)
    assert len(lines) == 3
    # 第一行讲鲁棒性（掉分 11.0）
    assert "11.0" in lines[0]
    # 第二行讲标定效果（救回 11.0）
    assert "11.0" in lines[1]
    # 第三行讲恢复程度（残留 0.0）
    assert "0.0" in lines[2]


def test_render_full_recovery_scenario():
    """干净 99%、脏 88%、洗回 99%：应得到「鲁棒+大幅救回+完全恢复」的组合。"""
    lines = render_pose_verdict(0.99, 0.88, 0.99)
    assert "比较鲁棒" in lines[0]        # 掉 11 分 → 5~15 档
    assert "有明显帮助" in lines[1]       # 救回 11 分 → 5~15 档
    assert "基本完全恢复" in lines[2]     # 残留 0 分


def test_render_degraded_scenario():
    """干净 99%、脏 50%、洗回 55%：掉分严重、救回 5 分（有明显帮助）、仍未恢复。"""
    lines = render_pose_verdict(0.99, 0.50, 0.55)
    assert "严重退化" in lines[0]         # 掉 49 分
    assert "有明显帮助" in lines[1]        # 救回 5.0 分 → 落在「明显帮助」档
    assert "还没完全恢复" in lines[2]      # 残留 44 分
