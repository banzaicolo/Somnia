#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=============================================================================
标定全流程演示 —— 看一条完整的标定链把脏数据洗到多干净
=============================================================================

【这脚本干嘛的？】

《零点校准演示》只给你看了一招（减法）。但这个文件把**全部招式串起来**：
串扰校正 + 温漂补偿 + 零点校准 + 增益校准，一步一步把脏数据洗回真相。

还加了个"剧情"：标定是在 25 度做的，结果运行时温度飙到 40 度，
传感器漂了。你看"不补偿温度"和"补偿温度"差多少。

【怎么运行】

    python3 src/calibration_demo.py

会打印每一步的误差数字，并在 outputs/ 生成一张对比图：
  calibration_full.png

【你要重点看什么】

跑完看终端里那个"误差缩小了 X%"，以及最后那张图右下角的误差图——
校准前一片红（到处虚报），完整校准后几乎全白（接近真相）。
=============================================================================
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["PingFang SC", "Arial Unicode MS", "Heiti SC"]
plt.rcParams["axes.unicode_minus"] = False

from sensor_config import (
    GRID_W, GRID_H, BODY_PARTS, OUT_DIR, SEED,
    make_ideal_pressure,
)
import calibration as cal
from verdict import describe_improvement


# ---- 标定场景的参数（都是"真实世界会有"的数）----
T_REF = 25.0      # 标定时的温度（摄氏度）
T_RUN = 40.0      # 运行时的温度（温度漂了 15 度）
CROSSTALK = 0.10  # 串扰系数
NOISE_STD = 3.0   # 噪声强度
P0 = 50.0         # 标定用的"标准压力"（相当于一个已知重量的砝码）


def build_sensor_world(rng):
    """
    构造一个"假想传感器"的世界：每个点的灵敏度、零点偏移、
    温度系数都随机但固定（这就是真实传感器的"出厂参数"）。

    返回一个字典，装着所有这些参数。
    """
    # 每个点的灵敏度（真值，围绕 1 浮动 ±15%）
    gain_ref = rng.normal(1.0, 0.15, size=(GRID_H, GRID_W))

    # 每个点的基准零点偏移（老化 + 工艺造成，0~8）
    offset_ref = rng.uniform(0, 8, size=(GRID_H, GRID_W))

    # 温度系数：每升 1 度，零点飘多少（TCO）、灵敏度变多少比例（TCS）
    tco_map = rng.uniform(0.3, 0.8, size=(GRID_H, GRID_W))   # 每度飘 0.3~0.8
    tcs_map = rng.uniform(0.001, 0.004, size=(GRID_H, GRID_W))  # 每度变 0.1%~0.4%

    return {
        "gain_ref": gain_ref,
        "offset_ref": offset_ref,
        "tco_map": tco_map,
        "tcs_map": tcs_map,
    }


def offset_at(world, temp):
    """某温度下的零点偏移 = 基准偏移 + 温漂。"""
    return world["offset_ref"] + world["tco_map"] * (temp - T_REF)


def gain_at(world, temp):
    """某温度下的灵敏度 = 基准灵敏度 × (1 + 温漂增益)。"""
    return world["gain_ref"] * (1.0 + world["tcs_map"] * (temp - T_REF))


def main():
    print("=" * 64)
    print(" 传感器标定全流程演示（串扰+温漂+零点+增益 四招连发）")
    print("=" * 64)
    print()

    rng = np.random.default_rng(SEED)
    world = build_sensor_world(rng)

    # ---- 1. 真实体压（标准答案）----
    ideal = make_ideal_pressure(GRID_W, GRID_H, BODY_PARTS)
    print("[1/5] 标准答案已生成（人真实压上去的样子）")

    # ---- 2. 运行时（温度 40 度）传感器读到的脏数据 ----
    offset_run = offset_at(world, T_RUN)
    gain_run = gain_at(world, T_RUN)
    noise = rng.normal(0, NOISE_STD, size=(GRID_H, GRID_W))
    raw = cal.forward(ideal, gain_run, offset_run, CROSSTALK, noise)
    print(f"[2/5] 运行时温度 {T_RUN:.0f}℃，脏数据已生成（含串扰/温漂/噪声）")
    err_raw = cal.mean_abs_error(raw, ideal)
    print(f"      脏数据平均误差 = {err_raw:.1f}")

    # ---- 3. 模拟"标定动作"（在 25 度下测参数）----
    # 真实标定会采很多帧取平均，把噪声压到几乎为零。所以这里标定数据
    # 用"无噪声"版本（等于充分多帧平均后的理想结果），这样能看清
    # 标定链对"系统误差"的极限威力。
    # 3a. 空载测基线
    baseline = cal.estimate_baseline(
        offset_at(world, T_REF), c=CROSSTALK, noise_std=0.0, rng=rng)
    # 3b. 用标准压力 P0 测增益
    ideal_ref = np.full((GRID_H, GRID_W), P0)   # 一块均匀的标准压力
    raw_ref = cal.forward(ideal_ref, gain_at(world, T_REF),
                          offset_at(world, T_REF), CROSSTALK, 0.0)
    gain_est = cal.estimate_gain(ideal_ref, raw_ref, baseline, c=CROSSTALK)
    print("[3/5] 标定动作完成：空载测得基线、标准压力测得每个点灵敏度")

    # ---- 4. 跑完整校准流水线 ----
    cleaned = cal.calibrate(
        raw,
        baseline=baseline,
        gain_map=gain_est,
        c=CROSSTALK,
        temp=T_RUN, t_ref=T_REF,
        tco_map=world["tco_map"], tcs_map=world["tcs_map"],
    )
    err_clean = cal.mean_abs_error(cleaned, ideal)
    print(f"[4/5] 完整校准后，平均误差 = {err_clean:.1f}")
    print(f"      误差缩小了 {cal.improvement(err_raw, err_clean):.0f}%")

    # ---- 4b. 对比：如果"不做温度补偿"会怎样 ----
    cleaned_no_temp = cal.calibrate(
        raw, baseline=baseline, gain_map=gain_est, c=CROSSTALK)
    err_no_temp = cal.mean_abs_error(cleaned_no_temp, ideal)
    print(f"      （对照）不做温度补偿，误差还剩 {err_no_temp:.1f}，"
          f"只缩小 {cal.improvement(err_raw, err_no_temp):.0f}%")

    # ---- 4c. 关键：区分「系统误差」和「随机噪声」 ----
    # 上面校准后还剩 {err_clean:.1f} 误差，大头是随机噪声（老电视雪花），
    # 噪声每次都不一样，标定消不掉，要靠"滤波"。把噪声关掉再看，
    # 就能看清标定链对"系统误差"（增益/偏移/串扰/温漂）的真实威力：
    raw_sys = cal.forward(ideal, gain_run, offset_run, CROSSTALK, 0.0)
    cleaned_sys = cal.calibrate(
        raw_sys, baseline=baseline, gain_map=gain_est, c=CROSSTALK,
        temp=T_RUN, t_ref=T_REF,
        tco_map=world["tco_map"], tcs_map=world["tcs_map"])
    err_raw_sys = cal.mean_abs_error(raw_sys, ideal)
    err_clean_sys = cal.mean_abs_error(cleaned_sys, ideal)
    print(f"      （无噪声对照）纯系统误差 {err_raw_sys:.1f} → {err_clean_sys:.1f}，"
          f"缩小 {cal.improvement(err_raw_sys, err_clean_sys):.0f}%")

    # ---- 5. 画图 ----
    plot(ideal, raw, cleaned_no_temp, cleaned, err_raw, err_no_temp, err_clean)
    print("[5/5] 对比图已生成，去 outputs/ 打开 calibration_full.png")
    print()
    # 结尾不再写死「一片红…几乎全白」，改成根据误差数字动态判断。
    improve_full = cal.improvement(err_raw, err_clean)
    print(f" ✅ 完成。完整校准后平均误差 {err_clean:.1f}，"
          f"误差缩小 {improve_full:.0f}%：{describe_improvement(improve_full)}")
    print("    （右下角误差图：红=读数偏高，蓝=偏低，白=准确，可自行对照）")


def plot(ideal, raw, no_temp, cleaned, err_raw, err_no_temp, err_clean):
    """画 2×2 对比图，统一色标，右下是误差图。"""
    os.makedirs(OUT_DIR, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    vmax = max(ideal.max(), raw.max())

    # 左上：标准答案
    im0 = axes[0, 0].imshow(ideal, cmap="hot", vmin=0, vmax=vmax)
    axes[0, 0].set_title(f"标准答案（真实压力）", fontsize=12)
    fig.colorbar(im0, ax=axes[0, 0], fraction=0.046)

    # 右上：脏数据
    im1 = axes[0, 1].imshow(raw, cmap="hot", vmin=0, vmax=vmax)
    axes[0, 1].set_title(f"脏数据（误差 {err_raw:.0f}）", fontsize=12)
    fig.colorbar(im1, ax=axes[0, 1], fraction=0.046)

    # 左下：不做温度补偿的校准
    im2 = axes[1, 0].imshow(no_temp, cmap="hot", vmin=0, vmax=vmax)
    axes[1, 0].set_title(f"不补偿温度（误差 {err_no_temp:.0f}）", fontsize=12)
    fig.colorbar(im2, ax=axes[1, 0], fraction=0.046)

    # 右下：误差图（完整校准 - 标准答案），红=偏高 蓝=偏低 白=准确
    err_map = cleaned - ideal
    lim = max(abs(err_map.min()), abs(err_map.max()), 1.0)
    im3 = axes[1, 1].imshow(err_map, cmap="seismic", vmin=-lim, vmax=lim)
    axes[1, 1].set_title(f"完整校准后的误差图（误差 {err_clean:.1f}）", fontsize=12)
    fig.colorbar(im3, ax=axes[1, 1], fraction=0.046)

    for ax in axes.flat:
        ax.set_xticks([])
        ax.set_yticks([])

    fig.suptitle("完整标定链：四招连发，把脏数据洗回真相", fontsize=15)
    plt.tight_layout()
    path = os.path.join(OUT_DIR, "calibration_full.png")
    plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return path


if __name__ == "__main__":
    main()
