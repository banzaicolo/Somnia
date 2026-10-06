#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=============================================================================
verdict.py —— 结论生成器：让程序「根据数字说话」，而不是写死结论
=============================================================================

【这个模块解决什么问题？】

之前脚本里有几处「结论」是写死在代码里的。比如：

    print(" 模型对毛病很稳健")

不管实际数字怎么变，都打印这句话。这有个致命缺陷：万一哪天参数
变了、数字崩了（比如准确率掉到 60%），这句「很稳健」还照样打印——
结论就变成了睁眼说瞎话，而且程序自己还不知道。

正确的做法是「规则写死，结论动态」：

    · 规则（阈值）写死：掉分 < 5 分算「稳健」，> 30 分算「性能严重下降」
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
    print(describe_improvement(82.0))   # → "基本消除"

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

# ---- 模型稳健性：坏传感器让模型掉多少分 ----
ROBUST_DROP = 5.0      # 掉分低于 5 个点 → 非常稳健
DEGRADE_DROP = 30.0    # 掉分超过 30 个点 → 性能严重下降
MID_DROP = 15.0        # 中间再分一档：5~15 较为稳健，15~30 明显下降


# ---- 标定救援效果：洗回后救回多少分 ----
RESCUE_TINY = 1.0      # 救回低于 1 个点 → 几乎没有改善
RESCUE_BIG = 15.0      # 救回超过 15 个点 → 大幅改善
RESCUE_MID = 5.0       # 中间档：1~5 轻微改善，5~15 明显改善

# ---- 恢复程度：洗回后离「干净水平」还差多少分 ----
RECOVER_GOOD = 1.0     # 残留低于 1 个点 → 基本完全恢复
RECOVER_BAD = 5.0      # 残留超过 5 个点 → 尚未完全恢复

# ---- 改善程度（通用）：误差缩小了百分之多少 ----
IMPROVE_BIG = 80.0     # 80% 以上 → 基本消除
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
      describe_improvement(82)  → "基本消除"
      describe_improvement(30)  → "改善有限，仍存在较多残余误差"
      describe_improvement(-5)  → "误差不降反升，校准方法不适用于当前场景"
    """
    if improve_pct < 0:
        return "误差不降反升，校准方法不适用于当前场景"
    if improve_pct < IMPROVE_LOW:
        return "几乎无改善，问题不在此步骤可消除的误差来源内"
    if improve_pct < IMPROVE_MID:
        return "改善有限，仍存在较多残余误差"
    if improve_pct < IMPROVE_BIG:
        return "明显改善，但尚未完全消除"
    return "基本消除"


# ============================================================================
# 二、睡姿分类附加实验专用：三个准确率 → 三段动态结论
# ============================================================================


def describe_robustness(drop_points):
    """
    根据「坏传感器让模型掉多少分」，返回稳健性结论。

    参数：
      drop_points —— 掉分（百分点，>=0）。0 表示一点没掉。

    例子：
      describe_robustness(2)   → "非常稳健，性能几乎不受影响"
      describe_robustness(40)  → "性能严重下降，已接近随机猜测水平"
    """
    drop = max(0.0, drop_points)   # 负数（脏数据反而更高分）按 0 处理
    if drop < ROBUST_DROP:
        return "非常稳健，性能几乎不受影响"
    if drop < MID_DROP:
        return "较为稳健，受影响较小"
    if drop < DEGRADE_DROP:
        return "性能明显下降，准确率显著降低"
    return "性能严重下降，已接近随机猜测水平"


def describe_rescue(recovered_points):
    """
    根据「标定洗回救回多少分」，返回标定效果结论。

    参数：
      recovered_points —— 救回的分（百分点）。负 = 标定反而更糟。

    例子：
      describe_rescue(0.3)  → "标定几乎没有改善"
      describe_rescue(20)   → "标定带来大幅改善"
    """
    if recovered_points < 0:
        return "标定未起正面作用，反而使结果更差（标定参数与实际误差来源不匹配）"
    if recovered_points < RESCUE_TINY:
        return "标定几乎没有改善"
    if recovered_points < RESCUE_MID:
        return "标定带来轻微改善"
    if recovered_points < RESCUE_BIG:
        return "标定带来明显改善"
    return "标定带来大幅改善"


def describe_recovery(residual_points):
    """
    根据「洗回后离干净水平还差多少分」，返回恢复程度结论。

    参数：
      residual_points —— 残留的分（百分点）。0 = 完全恢复。

    例子：
      describe_recovery(0.2)  → "基本完全恢复"
      describe_recovery(8)    → "尚未完全恢复，仍存在 8.0 分误差"
    """
    if residual_points < RECOVER_GOOD:
        return "基本完全恢复"
    if residual_points < RECOVER_BAD:
        return "基本恢复，仍有少量残余误差"
    return f"尚未完全恢复，仍存在 {residual_points:.1f} 分误差"


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
      掉分   = 干净 - 坏    → 描述稳健性
      救回   = 洗回 - 坏    → 描述标定效果
      残留   = 干净 - 洗回  → 描述恢复程度
    """
    drop = (test_acc - acc_dirty) * 100
    recovered = (acc_rescued - acc_dirty) * 100
    residual = (test_acc - acc_rescued) * 100

    return [
        f"模型稳健性：传感器异常导致准确率下降 {drop:.1f} 分 → "
        f"{describe_robustness(drop)}。",
        f"标定效果　：标定还原后挽回 {recovered:.1f} 分 → "
        f"{describe_rescue(recovered)}。",
        f"恢复程度　：标定还原后仍比基准水平低 {residual:.1f} 分 → "
        f"{describe_recovery(residual)}。",
    ]


# ============================================================================
# 三、睡眠分期专用：Kappa 一致性 + 深睡占比
# ============================================================================

# ---- Cohen's Kappa 分档阈值 ----
KAPPA_POOR = 0.2     # 低于 0.2 → 一致性差
KAPPA_FAIR = 0.4     # 0.2~0.4 → 一致性一般
KAPPA_MODERATE = 0.6 # 0.4~0.6 → 中等一致（消费级典型）
KAPPA_GOOD = 0.8     # 0.6~0.8 → 良好一致；≥0.81 医疗达标线

# ---- 深睡占比（一整夜深睡占多少比例）正常范围 ----
DEEP_LOW = 10.0     # 低于 10% → 深睡偏少
DEEP_HIGH = 25.0    # 高于 25% → 深睡偏多（可能睡眠剥夺后反弹）


def describe_kappa(kappa):
    """
    根据 Cohen's Kappa 值，返回一致性结论。

    参数：
      kappa —— κ 值（-1 ~ 1）

    例子：
      describe_kappa(0.05) → "一致性差，接近随机猜测"
      describe_kappa(0.45) → "中等一致（消费级非脑电分期的典型水平）"
    """
    if kappa < KAPPA_POOR:
        return "一致性差，接近随机猜测"
    if kappa < KAPPA_FAIR:
        return "一致性一般"
    if kappa < KAPPA_MODERATE:
        return "中等一致（消费级非脑电分期的典型水平）"
    if kappa < KAPPA_GOOD:
        return "良好一致"
    return "高度一致，接近医疗级睡眠分期的标准"


def describe_deep_ratio(deep_pct):
    """
    根据「深睡占整夜的比例」，返回深睡是否合理的结论。

    参数：
      deep_pct —— 深睡占比（百分比，比如 18 表示 18%）

    例子：
      describe_deep_ratio(18) → "深睡占比 18.0%，处于正常范围"
      describe_deep_ratio(6)  → "深睡占比 6.0%，偏少"
    """
    if deep_pct < DEEP_LOW:
        return f"深睡占比 {deep_pct:.1f}%，偏少"
    if deep_pct > DEEP_HIGH:
        return f"深睡占比 {deep_pct:.1f}%，偏高（可能为睡眠不足后的补偿性反弹）"
    return f"深睡占比 {deep_pct:.1f}%，处于正常范围"
