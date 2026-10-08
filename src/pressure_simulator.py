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

  1. imperfections.png     —— 4 张对比图，看每个毛病往里加了什么
  2. hysteresis_curve.png  —— 一个"加载-卸载"循环，看传感器"记仇"
  3. ideal_pressure.csv    —— 人真实压上去的样子（这是"标准答案"）
  4. sensor_readings.csv   —— 传感器实际读到的（这是"脏数据"）

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
# 第一部分：参数区（所有"旋钮"统一放在 src/sensor_config.py 里）
# ============================================================================
# 想改网格尺寸、身体部位、传感器毛病强度，去改 sensor_config.py 那一个文件。
# 这里 import 进来，本脚本和其他脚本就拿到同一套参数，不会各改各的。
from sensor_config import (
    GRID_W, GRID_H, BODY_PARTS,
    SENS_GAIN_STD, TEMP_DRIFT, ZERO_DRIFT_MAX, NOISE_STD, CROSSTALK, HYSTERESIS,
    OUT_DIR, SEED,
    FILE_IMPERFECTIONS, FILE_HYSTERESIS, FILE_IDEAL_CSV, FILE_RAW_CSV,
    make_ideal_pressure,
)


# ============================================================================
# 第二部分：核心函数
# ============================================================================
# （make_ideal_pressure 已移到 sensor_config.py，见上面的 import）


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
    rng = np.random.default_rng(SEED)
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

    这里用简化模型模拟（宽度恒等于 HYSTERESIS）：
      - 加压时，读数 = 真实压力 + HYSTERESIS/2
      - 减压时，读数 = 真实压力 - HYSTERESIS/2
    真实迟滞比这复杂得多，但这条圈够用来开发标定算法了。
    """
    p = np.linspace(0, 100, 50)   # 真实压力从 0 到 100，取 50 个点

    # 加压（从 0 到 100）：读数偏高半个宽度
    load = p + HYSTERESIS / 2.0

    # 减压（从 100 到 0）：读数偏低半个宽度
    unload = p - HYSTERESIS / 2.0

    return p, load, unload


def simulate_load_unload(ideal, n_steps=60, h=HYSTERESIS, noise=0.0, seed=None):
    """
    模拟「人慢慢躺下，再慢慢起身」的完整过程，并叠加迟滞。

    为什么要单独写这个？因为迟滞是「动态」的——它跟"压力在升还是降"有关，
    没法加到一张静止的压力图上（add_imperfections 处理的是单张静态图）。
    所以要模拟一个过程：

        躺下（加压：压力从 0 升到最大）→ 起身（减压：压力从最大降到 0）

        每一帧都按"当时是加压还是减压"来加迟滞偏移：

        加压段：读数 = p + h/2    （偏高）
        减压段：读数 = p - h/2    （偏低）

    参数：
      ideal    —— 理想体压图（人完全躺平时的标准答案），(H, W)
      n_steps  —— 总帧数（加压段和减压段各约一半）
      h        —— 迟滞宽度（跟 HYSTERESIS 同一个数）
      noise    —— 每帧叠加的随机噪声（默认 0，先看纯粹的迟滞效果）
      seed     —— 随机种子，None 则用 sensor_config 的 SEED

    返回：
      true_frames —— (n_frames, H, W) 每帧的真实压力（标准答案）
      raw_frames  —— (n_frames, H, W) 每帧传感器读数（含迟滞）
      weights     —— (n_frames,) 每帧的「压力系数」0~1，先升后降
    """
    rng = np.random.default_rng(seed if seed is not None else SEED)

    # 加压段 0→1，减压段 1→0（顶点只算一次，别让两段都包含 w=1 那一帧）
    n_up = n_steps // 2
    n_down = n_steps - n_up
    up = np.linspace(0.0, 1.0, n_up)
    down = np.linspace(1.0, 0.0, n_down + 1)[1:]   # 去掉重复的顶点
    weights = np.concatenate([up, down])

    true_frames = np.array([ideal * w for w in weights])
    raw_frames = np.empty_like(true_frames)

    for i, w in enumerate(weights):
        p = true_frames[i]
        if i < n_up:
            raw = p + h / 2.0     # 加压：读数偏高
        else:
            raw = p - h / 2.0     # 减压：读数偏低
        if noise > 0:
            raw = raw + rng.normal(0, noise, size=p.shape)
        raw_frames[i] = raw

    return true_frames, raw_frames, weights


# ============================================================================
# 第三部分：画图 + 保存
# ============================================================================


def plot_and_save(steps):
    """
    把每一步画出来，拼成一张 4 宫格对比图，存成 png。
    """
    out_dir = OUT_DIR
    os.makedirs(out_dir, exist_ok=True)

    # 取 4 个关键步骤来展示：理想、批次不一致、温漂+零点、最终
    keys = list(steps.keys())
    # 挑 4 个有代表性的（第1个、第2个、第4个、最后1个）
    chosen = [keys[0], keys[1], keys[3], keys[-1]]

    fig, axes = plt.subplots(1, 4, figsize=(18, 5))

    for ax, k in zip(axes, chosen):
        im = ax.imshow(steps[k], cmap="hot", interpolation="bicubic")
        ax.set_title(k, fontsize=11)
        ax.set_xticks([])   # 隐藏坐标刻度，图更干净
        ax.set_yticks([])
        fig.colorbar(im, ax=ax, fraction=0.046)

    fig.suptitle("理想体压 → 传感器真实读数（颜色越亮 = 压力越大）", fontsize=14)
    plt.tight_layout()
    path1 = os.path.join(out_dir, FILE_IMPERFECTIONS)
    plt.savefig(path1, dpi=200, bbox_inches="tight")
    plt.close(fig)

    return path1


def plot_hysteresis():
    """画迟滞回线（图2）。"""
    out_dir = OUT_DIR
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
    path2 = os.path.join(out_dir, FILE_HYSTERESIS)
    plt.savefig(path2, dpi=120, bbox_inches="tight")
    plt.close(fig)

    return path2


def save_csv(steps):
    """把理想体压图和最终读数存成 csv，方便以后喂给算法。"""
    out_dir = OUT_DIR
    os.makedirs(out_dir, exist_ok=True)

    keys = list(steps.keys())
    ideal = steps[keys[0]]
    final = steps[keys[-1]]

    np.savetxt(os.path.join(out_dir, FILE_IDEAL_CSV), ideal,
               fmt="%.2f", delimiter=",")
    np.savetxt(os.path.join(out_dir, FILE_RAW_CSV), final,
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
    print(f"   · {os.path.join(OUT_DIR, FILE_IDEAL_CSV)}（标准答案）")
    print(f"   · {os.path.join(OUT_DIR, FILE_RAW_CSV)}（脏数据，标定算法的输入）")
    print()
    print(" ✅ 完成。去打开那两张 png 图看看，你就明白传感器为什么会'骗人'了。")


# Python 固定套路：直接运行才执行，被 import 不执行
if __name__ == "__main__":
    main()
