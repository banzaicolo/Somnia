#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=============================================================================
PoPu 真实数据可视化 —— 把真人躺床垫的压力图，画出来给你看
=============================================================================

【干嘛的？】

前面项目全是「合成数据」，人形是咱家用高斯公式画的。这个脚本第一次
把**真人的真实压力数据**画出来，让你亲眼看到：
  - 真实人躺上去，压力图到底长啥样
  - 四种睡姿（仰卧/左侧卧/右侧卧/俯卧）的真实区别
  - 「零点校准」这个动作有多关键——不校准，人形全被 482 的假数淹没

【怎么运行】

    python3 src/pupu_demo.py

会生成 outputs/pupu_real_poses.png。

【图怎么看】

上下两排：
  上排 = 原始读数（校准前）：四张几乎一样，全是 ~485，根本看不出人形
  下排 = 减基线后（校准后）：人形清清楚楚——仰卧中间亮、侧卧偏一边、俯卧大片亮
=============================================================================
"""

import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["PingFang SC", "Arial Unicode MS", "Heiti SC"]
plt.rcParams["axes.unicode_minus"] = False

import pupu_loader as pl
from sensor_config import OUT_DIR, FILE_PUPU_REAL


# 输出文件名统一放 sensor_config（跟其他图一致）
OUTPUT_NAME = FILE_PUPU_REAL


def pick_sample_per_pose(data_dir, pose, seed=0):
    """
    从真实数据里，为某个睡姿挑一张「代表帧」来画。

    做法：找到该姿势的第一个文件，取它的第一帧（减基线前 + 减基线后都要）。
    返回：(原始读数矩阵 12×6, 减基线后矩阵 12×6, 该志愿者元数据)
    """
    import glob
    vol_dirs = sorted(
        [d for d in glob.glob(os.path.join(data_dir, "*")) if os.path.isdir(d)],
        key=lambda p: int(os.path.basename(p)) if os.path.basename(p).isdigit() else 999999,
    )
    for vol_dir in vol_dirs:
        baseline = pl.load_baseline(vol_dir)
        for path in sorted(glob.glob(os.path.join(vol_dir, pose + "*.json"))):
            matrices, meta = pl.load_snapshots(path)
            if not matrices:
                continue
            raw = matrices[0]
            return raw, raw - baseline, meta
    return None


def main():
    print("=" * 62)
    print(" PoPu 真实数据可视化：真人压力图 + 零点校准的效果")
    print("=" * 62)
    print()

    data_dir = pl.DEFAULT_DATA_DIR

    # 每类挑一张代表帧
    samples = {}
    for pose in pl.POSE_ORDER:
        r = pick_sample_per_pose(data_dir, pose)
        if r is None:
            print(f"  ⚠️ 没找到 {pose} 的样本")
            continue
        samples[pose] = r
        raw, cal, meta = r
        print(f"  {pl.POSE_LABELS_CN[pose]:>4}：志愿者 {meta['volunteer_id']}，"
              f"原始读数 {raw.min():.0f}~{raw.max():.0f}，"
              f"减基线后 {cal.min():.0f}~{cal.max():.0f}")

    print()
    print("  上面那句就是「零点校准」的证据：")
    print("  原始读数全卡在 ~485，四种姿势拉不开差距；")
    print("  减掉基线后，信号才浮出来，人形才可见。")

    # ---- 画图：上排原始，下排减基线后 ----
    os.makedirs(OUT_DIR, exist_ok=True)
    n = len(samples)
    fig, axes = plt.subplots(2, n, figsize=(4.2 * n, 8.5))

    # 统一色标：减基线后的最大压力值（忽略噪声负值）
    vmax = max(np.max(s[1]) for s in samples.values())
    vmax = max(vmax, 1.0)

    for col, pose in enumerate(pl.POSE_ORDER):
        if pose not in samples:
            continue
        raw, cal, meta = samples[pose]

        # 上排：原始读数（校准前）
        ax_raw = axes[0, col]
        im_raw = ax_raw.imshow(raw, cmap="hot", vmin=0, vmax=raw.max(),
                               interpolation="bicubic")
        ax_raw.set_title(f"{pl.POSE_LABELS_CN[pose]}\n原始读数（校准前）", fontsize=13)
        ax_raw.set_xticks([])
        ax_raw.set_yticks([])
        fig.colorbar(im_raw, ax=ax_raw, fraction=0.046)

        # 下排：减基线后（校准后）
        ax_cal = axes[1, col]
        im_cal = ax_cal.imshow(cal, cmap="hot", vmin=0, vmax=vmax,
                               interpolation="bicubic")
        ax_cal.set_title(f"{pl.POSE_LABELS_CN[pose]}\n减基线后（校准后）", fontsize=13)
        ax_cal.set_xticks([])
        ax_cal.set_yticks([])
        fig.colorbar(im_cal, ax=ax_cal, fraction=0.046)

    fig.suptitle(
        "真实人躺床垫的压力图（PoPu 数据集）\n"
        "上排：校准前看不出人形（全被 ~485 的零点偏移淹没）；下排：减基线后，"
        "仰卧居中、侧卧偏一侧、俯卧大片亮",
        fontsize=15,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.93])

    path = os.path.join(OUT_DIR, OUTPUT_NAME)
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print()
    print(f"  ✅ 图已生成：{path}")


if __name__ == "__main__":
    main()
