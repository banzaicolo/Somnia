#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=============================================================================
verdict.py —— 结论生成器：让程序「根据数字说话」，而不是写死结论
=============================================================================

【这个模块解决什么问题？】

之前脚本里有几处「结论」是写死在代码里的。比如：

    print(" 模型对毛病很鲁棒")

不管实际数字怎么变，都打印这句话。这有个致命缺陷：万一哪天参数
变了、数字崩了（比如准确率掉到 60%），这句「很鲁棒」还照样打印——
结论就变成了睁眼说瞎话，而且程序自己还不知道。

正确的做法是「规则写死，结论动态」：

    · 规则（阈值）写死：掉分 < 5 分算「鲁棒」，> 30 分算「严重退化」
    · 结论动态生成：程序拿本次真实数字去套规则，得出这次该说什么

打个比方：体检报告里「血压 > 140 算高血压」这条【规则】是写死的，
但「你今天血压高不高」这条【结论】是根据当天实测数字动态得出的。
标准是死的，结果是活的。

本模块只做一件事：把「数字 → 结论」的判断规则集中放好，
供各个脚本调用。数字变，结论就跟着变。

【阈值为什么写死？】

阈值不是「结论」，是「判断标准」。就像「60 分及格」是标准、不是结论。
标准写死完全没问题，有问题的是「结论」跟着写死。而且这些阈值都放在
下面的常量区里，一眼能看到、想改也好改，不是藏在代码深处。

【怎么用？】

    from verdict import describe_improvement, render_pose_verdict

    # 例1：误差缩小了 82%，问一句「修干净了吗」
    print(describe_improvement(82.0))   # → "基本修干净了"

    # 例2：三个准确率，生成一段完整的动态结论
    for line in render_pose_verdict(0.99, 0.88, 0.99):
        print(line)
=============================================================================
"""

# ============================================================================
# 判断阈值（「标准」，写死是合理的；想调只改这里）
# 单位说明：
#   - 掉分/救回/残留 都用「百分点」（percentage point），比如 2.3 表示 2.3 个百分点
#   - 改善 用「百分比」，比如 82 表示误差缩小了 82%
# ============================================================================

# ---- 模型鲁棒性：坏传感器让模型掉多少分 ----
ROBUST_DROP = 5.0      # 掉分低于 5 个点 → 非常鲁棒
DEGRADE_DROP = 30.0    # 掉分超过 30 个点 → 严重退化
MID_DROP = 15.0        # 中间再分一档：5~15 比较鲁棒，15~30 明显受伤

# ---- 标定救援效果：洗回后救回多少分 ----
RESCUE_TINY = 1.0      # 救回低于 1 个点 → 没帮上忙
RESCUE_BIG = 15.0      # 救回超过 15 个点 → 大幅救回
RESCUE_MID = 5.0       # 中间档：1~5 帮小忙，5~15 明显帮助

# ---- 恢复程度：洗回后离「干净水平」还差多少分 ----
RECOVER_GOOD = 1.0     # 残留低于 1 个点 → 基本完全恢复
RECOVER_BAD = 5.0      # 残留超过 5 个点 → 还没完全恢复

# ---- 改善程度（通用）：误差缩小了百分之多少 ----
IMPROVE_BIG = 80.0     # 80% 以上 → 基本修干净
IMPROVE_MID = 50.0     # 50%~80% → 明显改善
IMPROVE_LOW = 20.0     # 20%~50% → 改善有限


# ============================================================================
# 一、通用：改善程度（误差缩小百分比 → 结论）
# ============================================================================


def describe_improvement(improve_pct):
    """
    根据「误差缩小了百分之多少」，返回一句话结论。

    参数：
      improve_pct —— 误差缩小的百分比（可正可负）。
                     正 = 变好了；负 = 越修越糟（误差不降反升）。

    返回：
      一句中文结论，直接描述改善程度。

    例子：
      describe_improvement(82)  → "基本修干净了"
      describe_improvement(30)  → "改善有限"
      describe_improvement(-5)  → "越修越糟，误差不降反升"
    """
    if improve_pct < 0:
        return "越修越糟，误差不降反升（说明这招用错了场景）"
    if improve_pct < IMPROVE_LOW:
        return "几乎没改善（问题不在这一步能消掉的地方）"
    if improve_pct < IMPROVE_MID:
        return "改善有限，还剩不少残留"
    if improve_pct < IMPROVE_BIG:
        return "明显改善，但还没到干净的程度"
    return "基本修干净了"


# ============================================================================
# 二、睡姿分类附加实验专用：三个准确率 → 三段动态结论
# ============================================================================


def describe_robustness(drop_points):
    """
    根据「坏传感器让模型掉多少分」，返回鲁棒性结论。

    参数：
      drop_points —— 掉分（百分点，>=0）。0 表示一点没掉。

    例子：
      describe_robustness(2)   → "非常鲁棒，毛病几乎没伤到它"
      describe_robustness(40)  → "严重退化，接近瞎猜"
    """
    drop = max(0.0, drop_points)   # 负数（脏数据反而更高分）按 0 处理
    if drop < ROBUST_DROP:
        return "非常鲁棒，毛病几乎没伤到它"
    if drop < MID_DROP:
        return "比较鲁棒，受了点影响但扛得住"
    if drop < DEGRADE_DROP:
        return "明显受伤，毛病显著拖低了准确率"
    return "严重退化，已经接近瞎猜了"


def describe_rescue(recovered_points):
    """
    根据「标定洗回救回多少分」，返回标定效果结论。

    参数：
      recovered_points —— 救回的分（百分点）。负 = 标定反而更糟。

    例子：
      describe_rescue(0.3)  → "标定几乎没帮上忙"
      describe_rescue(20)   → "标定大幅救回"
    """
    if recovered_points < 0:
        return "标定反而帮了倒忙（标定参数跟实际污染对不上）"
    if recovered_points < RESCUE_TINY:
        return "标定几乎没帮上忙"
    if recovered_points < RESCUE_MID:
        return "标定帮了一点小忙"
    if recovered_points < RESCUE_BIG:
        return "标定有明显帮助"
    return "标定大幅救回"


def describe_recovery(residual_points):
    """
    根据「洗回后离干净水平还差多少分」，返回恢复程度结论。

    参数：
      residual_points —— 残留的分（百分点）。0 = 完全恢复。

    例子：
      describe_recovery(0.2)  → "基本完全恢复"
      describe_recovery(8)    → "还没完全恢复，仍差 8.0 分"
    """
    if residual_points < RECOVER_GOOD:
        return "基本完全恢复"
    if residual_points < RECOVER_BAD:
        return "基本恢复，只剩一点小尾巴"
    return f"还没完全恢复，仍差 {residual_points:.1f} 分"


def render_pose_verdict(test_acc, acc_dirty, acc_rescued):
    """
    根据三个准确率，生成本次附加实验的完整动态结论。

    参数（都是 0~1 的小数，比如 0.99 表示 99%）：
      test_acc    —— 干净数据考试准确率
      acc_dirty   —— 坏传感器考试准确率
      acc_rescued —— 标定洗回后考试准确率

    返回：
      一个字符串列表，每行一句话。直接 for 循环打印即可。

    逻辑：
      掉分   = 干净 - 坏    → 描述鲁棒性
      救回   = 洗回 - 坏    → 描述标定效果
      残留   = 干净 - 洗回  → 描述恢复程度
    """
    drop = (test_acc - acc_dirty) * 100
    recovered = (acc_rescued - acc_dirty) * 100
    residual = (test_acc - acc_rescued) * 100

    return [
        f"模型鲁棒性：坏传感器让它掉了 {drop:.1f} 分 → "
        f"{describe_robustness(drop)}。",
        f"标定效果　：洗回后救回 {recovered:.1f} 分 → "
        f"{describe_rescue(recovered)}。",
        f"恢复程度　：洗回后仍比干净水平低 {residual:.1f} 分 → "
        f"{describe_recovery(residual)}。",
    ]
