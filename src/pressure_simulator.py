#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=============================================================================
模拟压力传感器数据生成器（教学演示版，可直接运行）
=============================================================================

【这东西是干嘛的？】

你那张 AI 医疗床的核心，是床垫下的一排柔性压力传感器。
你现在没有真硬件。但"没硬件"挡不住你开发算法——因为传感器这个
东西，它输出的数据是【有物理规律】的。人躺上去，压力大的地方
（肩、屁股）读数就大；这些规律我们完全可以用数学"演"出来。

所以这个脚本干一件事：

    用数学"演"出一个"人躺在床垫上"的体压图，
    再往里"加"各种真传感器会有的毛病，
    让你看清：理想的世界 vs 传感器真实读到的世界，差多少。

【为什么这一步特别重要？】

我前面跟你说过：这个项目最容易翻车的地方，不是 AI 模型，
是【传感器标定】——传感器会骗人。这个脚本就是让你【亲眼看到】
它是怎么骗人的。看清楚敌人长什么样，你才写得对标定算法。

【这个文件会产出什么？】

跑完之后，会在「outputs」文件夹里生成：

  1. 图1_体压与传感器毛病.png  —— 4 张对比图，看每个毛病往里加了什么
  2. 图2_迟滞回线.png          —— 一个"加载-卸载"循环，看传感器"记仇"
  3. 理想体压.csv               —— 人真实压上去的样子（这是"标准答案"）
  4. 传感器读数.csv             —— 传感器实际读到的（这是"脏数据"）

【怎么运行】

Mac 打开"终端"，粘进去回车：

    python3 src/pressure_simulator.py

（需要 numpy 和 matplotlib，下面"依赖"那节会告诉你装没装、怎么装）
=============================================================================
"""

import os
import numpy as np
import matplotlib
# 不弹窗口，直接把图存成文件（在 Mac 终端里跑，这一行很重要）
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# 让中文能正常显示（Mac 上的中文字体）
plt.rcParams["font.sans-serif"] = ["PingFang SC", "Arial Unicode MS", "Heiti SC"]
plt.rcParams["axes.unicode_minus"] = False   # 让负号正常显示


# ============================================================================
# 第一部分：参数区（所有"旋钮"都在这，想改只改这里）
# ============================================================================

# ---- 传感器阵列的尺寸 ----
GRID_W = 12   # 横向有多少个感应点（列）
GRID_H = 8    # 纵向有多少个感应点（行）
              # 12×8 = 96 个点，跟你那个主控骨架里的 96 个点正好对上

# ---- 人体各部位的压力参数 ----
# 人仰卧时，不是全身均匀压下去，而是几个"凸起的部位"压得最重：
# 头、肩胛、屁股、脚后跟。每个部位用"一个高斯椭圆"来模拟——
# 高斯椭圆就是"中间高、四周慢慢变矮"的一座小鼓包，跟真实压力分布很像。
# 每个部位的参数：(名字, 中心x, 中心y, 宽sx, 高sy, 峰值, 旋转角度)
# x/y 的单位是"感应点编号"，比如 6 就是横向正中间那个点。
BODY_PARTS = [
    ("头部",   6.0, 1.0, 1.8, 1.0, 35, 0),
    ("肩胛",   6.0, 3.0, 2.6, 1.6, 80, 0),
    ("臀部",   6.0, 5.3, 2.8, 1.8, 110, 0),
    ("脚后跟", 6.0, 7.2, 1.4, 0.7, 30, 0),
]

# ---- 传感器"毛病"的强度（每个都能单独调，看它单独会造成什么）----

# 毛病① 批次不一致：同一批生产的传感器，每个灵敏度不一样。
#   真实世界里，两个"一模一样"的传感器，对同样的压力，读数可能差 15%。
#   这就是为什么要"逐个标定"。这里 0.15 就是 15% 的意思。
SENS_GAIN_STD = 0.15

# 毛病② 温漂：温度一变，传感器基线就整体偏移。
#   夏天和冬天，没人躺在上面，读数都不是 0。这个值越大，抬得越狠。
TEMP_DRIFT = 12

# 毛病③ 零点漂移：用久了（比如三个月），每个点慢慢"自己偏了"。
#   哪怕没人躺，读数也会从 0 慢慢飘走。这里模拟"每个点随机偏 0~8"。
ZERO_DRIFT_MAX = 8

# 毛病④ 噪声：信号里随机的小抖动，像老电视的雪花。
NOISE_STD = 3

# 毛病⑤ 串扰：相邻感应点会互相"漏电"，你压 A 点，旁边的 B 点也读到了。
#   0.1 表示 10% 的信号漏给邻居。这是阵列传感器最经典的问题。
CROSSTALK = 0.10

# ---- 迟滞回线的参数（图2 用）----
# 迟滞（Hysteresis）是柔性传感器最"阴"的毛病：同一个压力，
# "加压时"和"减压时"的读数不一样，像橡胶被压久了回弹不过来。
HYSTERESIS = 8     # 回线宽度：加压和减压之间能差多少


# ============================================================================
# 第二部分：核心函数
# ============================================================================


def make_ideal_pressure(w, h, parts):
    """
    生成"理想体压图"——这是标准答案：人真实压上去应该长这样。

    做法：把每个身体部位看成一座高斯小鼓包，全部叠在一起。

    参数：
      w, h  —— 阵列宽、高
      parts —— BODY_PARTS 那个列表
    返回：一个 h 行 w 列的二维数组（就是一张"图"），数值越大压力越大
    """
    # np.mgrid 生成坐标网格。yy 是每个点的"行号"，xx 是"列号"。
    yy, xx = np.mgrid[0:h, 0:w]
    img = np.zeros((h, w))   # 先造一张全 0 的图

    for name, cx, cy, sx, sy, amp, ang in parts:
        # 高斯椭圆的公式：先把坐标平移到"以鼓包中心为原点"
        dx = xx - cx
        dy = yy - cy

        # 如果要转角度，就把坐标旋转一下（这里默认 0 度，等于没转，
        # 但保留了这行，以后想模拟侧躺、斜躺就能用上）
        a = np.cos(np.radians(ang))
        b = np.sin(np.radians(ang))
        xr = dx * a + dy * b
        yr = -dx * b + dy * a

        # 二维高斯：中心 = amp，往四周按 sx/sy 的尺度衰减
        gauss = amp * np.exp(-(xr ** 2 / (2 * sx ** 2) + yr ** 2 / (2 * sy ** 2)))
        img += gauss   # 叠加上去

    return img


def add_imperfections(ideal):
    """
    把"理想体压图"加工成"传感器真实读到的图"，一层一层加毛病。

    这个函数会返回一个字典，里面同时有"每一步加完之后长什么样"，
    这样我们就能画对比图，看清每层毛病各自干了什么。

    参数：ideal —— 理想体压图
    返回：字典，key 是步骤名，value 是当时的图
    """
    steps = {"理想体压": ideal}   # 第 0 步：标准答案

    # ---- 毛病① 批次不一致 ----
    # 每个点乘以一个"灵敏度系数"，这个系数围绕 1 随机浮动。
    # 用固定随机种子，保证每次跑出来结果一样，方便你对比。
    rng = np.random.default_rng(42)
    sensitivity = rng.normal(1.0, SENS_GAIN_STD, size=ideal.shape)
    after_gain = ideal * sensitivity
    steps["+批次不一致(灵敏度±15%)"] = after_gain

    # ---- 毛病② 温漂 ----
    # 整体抬一个基线。真实世界是"温度高抬多少、低抬多少"，
    # 这里简化为：整体抬 TEMP_DRIFT，再叠加一点点空间上的不均匀。
    temp_map = TEMP_DRIFT * rng.uniform(0.7, 1.3, size=ideal.shape)
    after_temp = after_gain + temp_map
    steps["+温漂(基线整体抬高)"] = after_temp

    # ---- 毛病③ 零点漂移 ----
    # 每个点随机加一个 0~ZERO_DRIFT_MAX 的"长期偏移"。
    # 它跟温度无关，是"老化"造成的，且每个点不一样。
    zero_map = rng.uniform(0, ZERO_DRIFT_MAX, size=ideal.shape)
    after_zero = after_temp + zero_map
    steps["+零点漂移(每个点随机偏)"] = after_zero

    # ---- 毛病④ 串扰 ----
    # 每个点把一部分信号"漏"给上下左右的邻居。
    # 做法：每个点的新读数 = (1-c) × 自己  +  c × 上下左右四个邻居的平均。
    # c 就是 CROSSTALK，越大漏得越狠。
    if CROSSTALK > 0:
        c = CROSSTALK
        # np.pad 给图四周"垫一圈边"，这样边缘的点也能取到邻居
        padded = np.pad(after_zero, 1, mode="edge")
        # 上下左右四个邻居的平均值
        neighbors = (padded[:-2, 1:-1] + padded[2:, 1:-1] +
                     padded[1:-1, :-2] + padded[1:-1, 2:]) / 4.0
        after_cross = (1 - c) * after_zero + c * neighbors
    else:
        after_cross = after_zero
    steps["+串扰(邻点漏电10%)"] = after_cross

    # ---- 毛病⑤ 噪声 ----
    # 最后叠一层随机噪声。
    after_noise = after_cross + rng.normal(0, NOISE_STD, size=ideal.shape)
    steps["最终读数(+噪声)"] = after_noise

    return steps


def hysteresis_curve():
    """
    生成一条"迟滞回线"（图2 用）。

    迟滞的意思：你把传感器从 0 压到 100，再从 100 放回 0，
    这"加压"和"减压"两条路径的读数对不上，画出来是一个圈，
    而不是一条线。圈越胖，说明传感器越"记仇"。

    这里用简化模型模拟：
      - 加压时，读数 = 真实压力 + 一部分迟滞
      - 减压时，读数 = 真实压力 - 一部分迟滞
    真实迟滞比这复杂得多，但这条圈够用来开发标定算法了。
    """
    p = np.linspace(0, 100, 50)   # 真实压力从 0 到 100，取 50 个点

    # 加压（从 0 到 100）：读数偏高
    load = p + HYSTERESIS * (1 - p / 100.0)

    # 减压（从 100 到 0）：读数偏低
    unload = p - HYSTERESIS * (1 - p / 100.0)

    return p, load, unload


# ============================================================================
# 第三部分：画图 + 保存
# ============================================================================


def plot_and_save(steps):
    """
    把每一步画出来，拼成一张 4 宫格对比图，存成 png。
    """
    out_dir = "outputs"
    os.makedirs(out_dir, exist_ok=True)

    # 取 4 个关键步骤来展示：理想、批次不一致、温漂+零点、最终
    keys = list(steps.keys())
    # 挑 4 个有代表性的（第1个、第2个、第4个、最后1个）
    chosen = [keys[0], keys[1], keys[3], keys[-1]]

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))

    for ax, k in zip(axes, chosen):
        im = ax.imshow(steps[k], cmap="hot", interpolation="nearest")
        ax.set_title(k, fontsize=11)
        ax.set_xticks([])   # 隐藏坐标刻度，图更干净
        ax.set_yticks([])
        fig.colorbar(im, ax=ax, fraction=0.046)

    fig.suptitle("理想体压 → 传感器真实读数（颜色越亮 = 压力越大）", fontsize=14)
    plt.tight_layout()
    path1 = os.path.join(out_dir, "图1_体压与传感器毛病.png")
    plt.savefig(path1, dpi=120, bbox_inches="tight")
    plt.close(fig)

    return path1


def plot_hysteresis():
    """画迟滞回线（图2）。"""
    out_dir = "outputs"
    os.makedirs(out_dir, exist_ok=True)

    p, load, unload = hysteresis_curve()

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(p, load,   color="tab:red",  linewidth=2, label="加压（往上压）")
    ax.plot(p, unload, color="tab:blue", linewidth=2, label="减压（往回放）")

    # 画一条"理想"的 45 度虚线：如果传感器诚实，读数就该等于压力
    ax.plot([0, 100], [0, 100], "--", color="gray", label="理想（读数=压力）")

    ax.set_xlabel("真实压力")
    ax.set_ylabel("传感器读数")
    ax.set_title("迟滞回线：同一个压力，加压和减压读数不一样")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path2 = os.path.join(out_dir, "图2_迟滞回线.png")
    plt.savefig(path2, dpi=120, bbox_inches="tight")
    plt.close(fig)

    return path2


def save_csv(steps):
    """把理想体压图和最终读数存成 csv，方便以后喂给算法。"""
    out_dir = "outputs"
    os.makedirs(out_dir, exist_ok=True)

    keys = list(steps.keys())
    ideal = steps[keys[0]]
    final = steps[keys[-1]]

    np.savetxt(os.path.join(out_dir, "理想体压.csv"), ideal,
               fmt="%.2f", delimiter=",")
    np.savetxt(os.path.join(out_dir, "传感器读数.csv"), final,
               fmt="%.2f", delimiter=",")


# ============================================================================
# 第四部分：主程序
# ============================================================================


def main():
    print("=" * 62)
    print(" 模拟压力传感器数据生成器")
    print(" 用途：没真硬件时，先'演'出传感器数据，开发标定算法")
    print("=" * 62)
    print()

    # ---- 1. 生成理想体压图 ----
    ideal = make_ideal_pressure(GRID_W, GRID_H, BODY_PARTS)
    print(f"[1/4] 已生成理想体压图（{GRID_W}×{GRID_H} = {GRID_W*GRID_H} 个感应点）")

    # ---- 2. 一层层加毛病 ----
    steps = add_imperfections(ideal)
    print("[2/4] 已叠加 5 种传感器毛病（批次不一致/温漂/零点漂移/串扰/噪声）")

    # ---- 3. 画图 + 保存 ----
    path1 = plot_and_save(steps)
    path2 = plot_hysteresis()
    save_csv(steps)
    print("[3/4] 已生成 2 张图 + 2 个 csv 文件")

    # ---- 4. 打印"标准答案 vs 脏数据"的关键数字，让你有直观感受 ----
    final = steps[list(steps.keys())[-1]]
    print()
    print("[4/4] 关键对比（同一个点，理想 vs 传感器读数）：")
    print(f"      理想体压峰值  : {ideal.max():>6.1f}")
    print(f"      传感器读数峰值 : {final.max():>6.1f}")
    print(f"      差异           : {final.max() - ideal.max():>6.1f}")
    print()
    print(" 产出文件都在「outputs」文件夹里：")
    print(f"   · {path1}")
    print(f"   · {path2}")
    print("   · outputs/理想体压.csv（标准答案）")
    print("   · outputs/传感器读数.csv（脏数据，标定算法的输入）")
    print()
    print(" ✅ 完成。去打开那两张 png 图看看，你就明白传感器为什么会'骗人'了。")


# Python 固定套路：直接运行才执行，被 import 不执行
if __name__ == "__main__":
    main()
