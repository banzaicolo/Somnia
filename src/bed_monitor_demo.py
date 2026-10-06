#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=============================================================================
离床监测演示 —— 演一出「老人一整夜」的剧情，看清系统何时记录、何时报警
=============================================================================

【这个脚本演示什么？】

把 bed_monitor.py 的离床监测，套在一段"有头有尾"的整夜剧情上，让你
看清楚系统怎么从一张张压力图，判断出"人在哪、要不要报警"的。

剧情（280 帧，假设每秒 1 帧，就是 4 分 40 秒；真实一晚要 8 小时，
这里把时间压缩了，只为把故事讲清楚）：

  帧  0~ 39   夜里入睡，躺平睡觉
  帧 40~ 49   翻了个身（总压力微微掉，但人还在床上，不触发任何事）
  帧 50~ 89   继续睡
  帧 90~ 99   坐起来喝水（坐起状态，不报警）
  帧 100~129  又躺下接着睡
  帧 130~169  起夜下床去厕所，40 秒后自己回来了（正常起夜，不报警）
  帧 170~199  回床继续睡
  帧 200~279  又一次下床，这次迟迟没回来 → 超过时限才报警

【系统要做的事】
  1. 算出每帧的总压力（一个数）
  2. 按比例判断每帧是"在床 / 坐起 / 离床"
  3. 记录每次"离床事件"（几点下床、几点回来、去了多久）——不报警
  4. 只有"下床后迟迟不归"才拉警报，人一回来就解除

【怎么运行】

    python3 src/bed_monitor_demo.py

跑完会在「outputs」里生成 bed_monitor.png，一张状态时间线图。
=============================================================================
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")   # 不弹窗口，直接存图
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["PingFang SC", "Arial Unicode MS", "Heiti SC"]
plt.rcParams["axes.unicode_minus"] = False

from sensor_config import (
    GRID_W, GRID_H, BODY_PARTS, NOISE_STD, OUT_DIR, SEED,
    FILE_BED_MONITOR, make_ideal_pressure,
)
import bed_monitor as bm


# 演示用的"超时"时长。真实部署是 30 分钟（1800 帧），图上画不下，
# 这里压缩成 50 帧，只为让"超时报警"这一下能在图里看得到。
DEMO_AWAY_TIMEOUT = 50


def build_story(ideal, seed=SEED):
    """
    构造整夜剧情帧序列（见文件头说明）。

    每一帧 = 理想体压图 × 一个"压力系数"，再叠一点噪声（模拟真实读数）。
    压力系数 w 就是"人有多实在压在床上"：
        w=1.0 躺平全压着；w=0.7 翻身（重心略动但人还在）；w=0.4 坐起；w=0 下床。
    """
    rng = np.random.default_rng(seed)

    weights = np.concatenate([
        np.full(40, 1.0),   # 0~39   入睡
        np.full(10, 0.7),   # 40~49  翻身（还在床）
        np.full(40, 1.0),   # 50~89  继续睡
        np.full(10, 0.4),   # 90~99  坐起喝水
        np.full(30, 1.0),   # 100~129 又躺下
        np.full(40, 0.0),   # 130~169 起夜去厕所（40 秒就回来）
        np.full(30, 1.0),   # 170~199 回床继续睡
        np.full(80, 0.0),   # 200~279 下床，这次迟迟不归 → 超时报警
    ])

    frames = []
    for w in weights:
        frame = ideal * w
        frame = frame + rng.normal(0, NOISE_STD, size=frame.shape)
        frames.append(frame)
    return np.array(frames), weights


def main():
    print("=" * 64)
    print(" 离床监测演示：老人一整夜（翻身 / 喝水 / 起夜 / 超时未归）")
    print("=" * 64)
    print()

    # ---- ① 造理想体压图 + 剧情 ----
    ideal = make_ideal_pressure(GRID_W, GRID_H, BODY_PARTS)
    frames, weights = build_story(ideal)
    print("[1/4] 已造出 280 帧整夜剧情（详见文件头说明）")

    # ---- ② 状态机：判每帧在不在床 ----
    states = bm.detect_states(frames)

    # ---- ③ 离床事件：只记录，不报警 ----
    exits = bm.detect_exits(states)
    # ---- ④ 报警：超时未归才拉响 ----
    alarms = bm.detect_alarms(states, away_timeout=DEMO_AWAY_TIMEOUT)

    # ---- 打印离床事件（作息表）----
    print()
    print("[2/4] 离床事件记录（下床很正常，只记录、不报警）：")
    if exits:
        for start, end in exits:
            print(f"      第 {start:>3} 帧下床 → 第 {end:>3} 帧回来，离开 {end - start} 帧")
    else:
        print("      （整夜没有一次有效的下床）")

    # ---- 打印报警 ----
    print()
    alarm_idx = [i for i, a in enumerate(alarms) if a]
    if alarm_idx:
        print(f"[3/4] 报警：第 {alarm_idx[0]} 帧拉响（下床已满 {DEMO_AWAY_TIMEOUT} 帧仍未回）")
        print(f"       第 {alarm_idx[-1] + 1} 帧解除（老人回来了）")
    else:
        print("[3/4] 整夜平安，未触发任何报警")

    # ---- 画图 ----
    path = plot_timeline(frames, states, alarms)
    print()
    print(f" 产出图：{path}")
    print()
    print(" ✅ 完成。打开那张图：绿=在床、黄=坐起、红=离床。")
    print("    注意：起夜（第 130~169 帧）那一段离床，系统只记录、没报警；")
    print(f"    只有最后那一段离床超过 {DEMO_AWAY_TIMEOUT} 帧还没回来，才拉响警报。")


def plot_timeline(frames, states, alarms):
    """画总压力随时间的变化，按状态着色，标出报警区间。"""
    os.makedirs(OUT_DIR, exist_ok=True)

    totals = frames.reshape(frames.shape[0], -1).sum(axis=1)
    peak = totals.max()
    t = np.arange(len(totals))

    fig, ax = plt.subplots(figsize=(14, 5.5))

    # 背景按状态着色（颜色淡，不抢曲线）
    color_map = {bm.IN_BED: "#c6e8c6", bm.SITTING: "#ffe6b0", bm.OUT_OF_BED: "#f5c6c6"}
    i = 0
    while i < len(states):
        s = states[i]
        j = i
        while j < len(states) and states[j] == s:
            j += 1
        ax.axvspan(i, j, color=color_map[s], alpha=0.5, zorder=0)
        i = j

    # 总压力曲线
    ax.plot(t, totals, color="black", linewidth=1.8, zorder=2, label="总压力")

    # 两条阈值线：60% 在床线、15% 离床线
    ax.axhline(peak * 0.60, color="green", linestyle="--", linewidth=1.2,
               label="在床线（60%）")
    ax.axhline(peak * 0.15, color="red", linestyle="--", linewidth=1.2,
               label="离床线（15%）")

    # 报警区间：用红点标出（只有超时未归的那一段才报警）
    if any(alarms):
        alarm_idx = [i for i, a in enumerate(alarms) if a]
        ax.scatter(alarm_idx, [totals[i] for i in alarm_idx],
                   color="red", s=45, zorder=3, label="超时报警")

    ax.set_xlabel("帧（每秒 1 帧 → 就是秒）")
    ax.set_ylabel("总压力（所有传感点之和）")
    ax.set_title("离床监测：总压力 + 状态 + 超时报警时间线")
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(OUT_DIR, FILE_BED_MONITOR)
    plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return path


if __name__ == "__main__":
    main()
