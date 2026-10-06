#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=============================================================================
迟滞补偿演示 —— 把「传感器记仇」这条毛病，自动修回来
=============================================================================

【这个脚本演示什么？】

迟滞（hysteresis）是传感器最"鸡贼"的一种毛病：同一个压力，你往上压
（加压）时它读得偏高，往回放（减压）时它读得偏低。画出来是一个圈，
不是一条线（你在《压力模拟器》的 hysteresis_curve.png 里见过那个圈）。

修它的难点不在公式（公式很简单），而在——**程序得先知道"现在是在加压
还是减压"**，而传感器不会告诉你方向。

所以这个脚本串起一个完整闭环，演示"自动判方向 + 补偿"：

    ① 造一张理想体压图（人躺平的样子 = 标准答案）
    ② 模拟"人慢慢躺下（加压）→ 慢慢起身（减压）"，共几十帧，
       每一帧按当时方向叠上迟滞，得到"脏读数"
    ③ 程序自己拿相邻帧比较、判断方向，再套补偿公式
    ④ 对比：补偿前差多少、补偿后差多少

结论（和别的演示一样）不是写死的，是拿数字算出来、用 verdict 模块
动态生成的。

【怎么运行】

    python3 src/hysteresis_demo.py

跑完会在「outputs」里生成 hysteresis_compensation.png。
=============================================================================
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")   # 不弹窗口，直接存图
import matplotlib.pyplot as plt

# 中文显示
plt.rcParams["font.sans-serif"] = ["PingFang SC", "Arial Unicode MS", "Heiti SC"]
plt.rcParams["axes.unicode_minus"] = False

from sensor_config import (
    GRID_W, GRID_H, BODY_PARTS, HYSTERESIS, OUT_DIR,
    FILE_HYSTERESIS_COMP,
    make_ideal_pressure,
)
from pressure_simulator import simulate_load_unload
import calibration as cal
from verdict import describe_improvement


def main():
    print("=" * 64)
    print(" 迟滞补偿演示：自动判断加压/减压方向，把「记仇」修回来")
    print("=" * 64)
    print()

    # ---- ① 标准答案：人躺平的理想体压图 ----
    ideal = make_ideal_pressure(GRID_W, GRID_H, BODY_PARTS)
    print(f"[1/5] 理想体压图（标准答案）：{GRID_W}×{GRID_H}，峰值 {ideal.max():.1f}")

    # ---- ② 模拟"躺下→起身"，叠迟滞 ----
    n_steps = 60
    true_frames, raw_frames, weights = simulate_load_unload(
        ideal, n_steps=n_steps, h=HYSTERESIS, noise=0.0)
    print(f"[2/5] 模拟躺下→起身共 {len(weights)} 帧（前加压、后减压），每帧叠迟滞 {HYSTERESIS}")

    # ---- ③ 自动判方向 + 补偿 ----
    corrected = cal.correct_hysteresis_sequence(raw_frames, HYSTERESIS, deadband=0.0)
    print("[3/5] 已自动判断每帧方向，并完成迟滞补偿")

    # ---- ④ 算误差：补偿前 vs 补偿后 ----
    # 全程平均绝对误差：所有帧、所有点，"跟标准答案差多少"的平均
    err_before = float(np.abs(raw_frames - true_frames).mean())
    err_after = float(np.abs(corrected - true_frames).mean())
    print(f"[4/5] 全程平均误差：补偿前 {err_before:.2f} → 补偿后 {err_after:.2f}")

    # ---- ⑤ 动态结论（不是写死，是拿数字算的） ----
    improve = cal.improvement(err_before, err_after)
    print()
    print(" 本次演示结论（程序根据上面的数字自动生成，非写死）：")
    print(f"   迟滞造成的平均误差 {err_before:.2f}，补偿后降到 {err_after:.2f}，")
    print(f"   误差缩小 {improve:.1f}% → {describe_improvement(improve)}。")
    print()

    # ---- 画图 ----
    path = plot_result(true_frames, raw_frames, corrected, weights, ideal)
    print(f" 产出图：{path}")
    print()
    print(" ✅ 完成。打开那张图：蓝线（补偿后）会几乎贴住灰虚线（真实），")
    print("    红线（脏读数）则明显偏离——这就是迟滞被修回来的样子。")


def plot_result(true_frames, raw_frames, corrected, weights, ideal):
    """画两张图：左=峰值点的压力随时间；右=每帧平均误差（补偿前后）。"""
    os.makedirs(OUT_DIR, exist_ok=True)

    # 找峰值点（屁股最重的地方），看它的"压力-时间"曲线最直观
    peak_idx = np.unravel_index(np.argmax(ideal), ideal.shape)
    t = np.arange(len(weights))

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # ---- 左图：峰值点随时间 ----
    ax = axes[0]
    ax.plot(t, true_frames[:, peak_idx[0], peak_idx[1]], "--",
            color="gray", linewidth=2, label="真实压力（标准答案）")
    ax.plot(t, raw_frames[:, peak_idx[0], peak_idx[1]], "-",
            color="tab:red", linewidth=2, label="传感器读数（含迟滞）")
    ax.plot(t, corrected[:, peak_idx[0], peak_idx[1]], "-",
            color="tab:blue", linewidth=2, label="补偿后（自动判方向）")
    ax.set_xlabel("帧（时间 → 先躺下、后起身）")
    ax.set_ylabel("压力")
    ax.set_title("峰值点：迟滞让读数偏移，补偿拉回真实")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # ---- 右图：每帧平均误差 ----
    ax = axes[1]
    err_before = np.abs(raw_frames - true_frames).reshape(len(weights), -1).mean(axis=1)
    err_after = np.abs(corrected - true_frames).reshape(len(weights), -1).mean(axis=1)
    ax.plot(t, err_before, "-", color="tab:red", linewidth=2, label="补偿前误差")
    ax.plot(t, err_after, "-", color="tab:blue", linewidth=2, label="补偿后误差")
    ax.set_xlabel("帧（时间 → 先躺下、后起身）")
    ax.set_ylabel("平均绝对误差")
    ax.set_title("每帧误差：补偿后几乎归零")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    fig.suptitle("迟滞自动补偿：加压/减压方向由程序自己判断", fontsize=14)
    plt.tight_layout()
    path = os.path.join(OUT_DIR, FILE_HYSTERESIS_COMP)
    plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return path


if __name__ == "__main__":
    main()
