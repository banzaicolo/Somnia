#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=============================================================================
BCG 呼吸/心跳监测演示 —— 演一段床垫信号，看算法怎么把呼吸率和心率挖出来
=============================================================================

【这个脚本演示什么？】

把 bcg_monitor.py 的呼吸/心跳监测，套在一段合成的床垫信号上，让你看清：
  1. 床垫信号长啥样（呼吸 + 心跳 + 噪声混在一起）
  2. 算法怎么"按频率切"把呼吸和心跳分开
  3. 提取出的呼吸率、心率，跟真实值对不对得上
  4. 一段"呼吸暂停"，算法能不能发现（婴儿窒息、成人睡眠呼吸暂停都靠它）

剧情（60 秒）：
  第  0~20 秒  正常呼吸（15 次/分）+ 心跳（72 次/分）
  第 20~40 秒  呼吸暂停（胸廓不动了，但心跳还在）——模拟窒息
  第 40~60 秒  恢复呼吸

【怎么运行】

    python3 src/bcg_demo.py

跑完会在「outputs」里生成 bcg_monitor.png。
=============================================================================
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")   # 不弹窗口，直接存图
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["PingFang SC", "Arial Unicode MS", "Heiti SC"]
plt.rcParams["axes.unicode_minus"] = False

from sensor_config import OUT_DIR, FILE_BCG
import bcg_monitor as bcg


def main():
    print("=" * 64)
    print(" BCG 呼吸/心跳监测演示：正常 → 呼吸暂停 → 恢复")
    print("=" * 64)
    print()

    # ---- ① 合成一段床垫信号（标准答案：呼吸 15 次/分，心跳 72 次/分）----
    TRUE_RESP = 15.0
    TRUE_HR = 72.0
    t, signal = bcg.simulate_bcg(
        duration_sec=60, resp_rate=TRUE_RESP, heart_rate=TRUE_HR,
        apnea=(20, 40),   # 第 20~40 秒呼吸暂停
    )
    print("[1/4] 已合成 60 秒床垫信号（呼吸 15 次/分 + 心跳 72 次/分 + 噪声）")
    print(f"      第 20~40 秒人为设置了一段「呼吸暂停」")

    # ---- ② 提取呼吸率、心率 ----
    est_resp = bcg.estimate_breathing_rate(signal)
    est_hr = bcg.estimate_heart_rate(signal)
    print()
    print("[2/4] 提取结果 vs 真实值：")
    print(f"      呼吸率：真实 {TRUE_RESP:>4.1f} 次/分 → 提取 {est_resp:>4.1f} 次/分"
          f"（误差 {abs(est_resp - TRUE_RESP):.1f}）")
    print(f"      心率　：真实 {TRUE_HR:>4.1f} 次/分 → 提取 {est_hr:>4.1f} 次/分"
          f"（误差 {abs(est_hr - TRUE_HR):.1f}）")

    # ---- ③ 检测呼吸暂停 ----
    flags, window_t = bcg.detect_apnea(signal, window_sec=10)
    print()
    print("[3/4] 呼吸暂停检测（每 10 秒一个窗口）：")
    apnea_windows = []
    for i, (flag, wt) in enumerate(zip(flags, window_t)):
        mark = "⚠️ 疑似呼吸暂停" if flag else "正常"
        if flag:
            apnea_windows.append(wt)
        print(f"      第 {wt:>3.0f}~{wt + 10:>3.0f} 秒 → {mark}")
    if apnea_windows:
        print(f"      → 成功揪出 {len(apnea_windows)} 段暂停（正是第 20~40 秒）")
    else:
        print("      → 未检出暂停")

    # ---- ④ 画图 ----
    path = plot_signal(t, signal, flags, window_t)
    print()
    print(f" 产出图：{path}")
    print()
    print(" ✅ 完成。打开那张图：")
    print("    上图 = 原始信号（呼吸的大波浪 + 心跳的小锯齿，暂停段被红框标出）")
    print("    中图 = 放大前 5 秒，看清呼吸（慢波）和心跳（快波）的叠加")
    print("    下图 = FFT 频谱，两个峰就是呼吸（0.25 Hz）和心跳（1.2 Hz）")


def plot_signal(t, signal, apnea_flags, window_t):
    """画三张子图：全貌信号 / 放大细节 / FFT 频谱。"""
    os.makedirs(OUT_DIR, exist_ok=True)
    fs = bcg.FS

    fig, axes = plt.subplots(3, 1, figsize=(13, 11))

    # ---- 子图1：60 秒全貌，标出暂停区间 ----
    ax = axes[0]
    ax.plot(t, signal, color="#2c5f8a", linewidth=1.0, label="床垫压力信号")
    # 暂停区间画红框
    for i, flag in enumerate(apnea_flags):
        if flag:
            ax.axvspan(window_t[i], window_t[i] + 10, color="red", alpha=0.25)
    ax.set_title("① 整段信号（60 秒）：红色阴影 = 检测出的呼吸暂停")
    ax.set_xlabel("时间（秒）")
    ax.set_ylabel("压力读数")
    ax.set_ylim(bottom=signal.min() - 2, top=signal.max() + 2)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # ---- 子图2：放大前 5 秒，看清呼吸慢波 + 心跳快波 ----
    ax = axes[1]
    seg = t < 5
    ax.plot(t[seg], signal[seg], color="#2c5f8a", linewidth=1.4)
    ax.set_title("② 放大前 5 秒：慢的大波浪 = 呼吸，快的小锯齿 = 心跳")
    ax.set_xlabel("时间（秒）")
    ax.set_ylabel("压力读数")
    ax.grid(True, alpha=0.3)

    # ---- 子图3：FFT 频谱，标出两个峰 ----
    ax = axes[2]
    n = len(signal)
    spectrum = np.abs(np.fft.rfft(signal))
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)
    # 用 log 尺度，让大的呼吸峰和小心跳峰都看得清；+1 防止 log(0)
    ax.plot(freqs, np.log10(spectrum + 1), color="#7a4d8f", linewidth=1.4)
    # 标出呼吸峰和心跳峰
    resp_peak = bcg.estimate_breathing_rate(signal) / 60.0
    hr_peak = bcg.estimate_heart_rate(signal) / 60.0
    ax.axvline(resp_peak, color="green", linestyle="--", linewidth=1.4,
               label=f"呼吸峰 {resp_peak:.2f} Hz")
    ax.axvline(hr_peak, color="red", linestyle="--", linewidth=1.4,
               label=f"心跳峰 {hr_peak:.2f} Hz")
    ax.set_xlim(0, 2.0)   # 只看 0~2 Hz，够罩住呼吸和心跳了
    ax.set_title("③ FFT 频谱：两个峰 = 呼吸（低频）和心跳（高频）各占一段")
    ax.set_xlabel("频率（Hz）")
    ax.set_ylabel("能量（log 尺度）")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(OUT_DIR, FILE_BCG)
    plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return path


if __name__ == "__main__":
    main()
