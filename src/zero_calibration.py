#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=============================================================================
零点校准演示（教学版，可直接运行）
=============================================================================

【这个脚本证明一件什么事？】

回答你上一个问题："你怎么知道怎么修？"

修传感器的第一招叫【零点校准】，它其实就是一句话：

    "人下床那一刻，把传感器读数记下来，之后每个读数都减掉它。"

原理简单到爆：传感器没人压的时候，读数本该是 0。但它因为
温漂、零点漂移这些毛病，读出了 15~30 的"假数"。这些假数是
"加在"真信号上面的 —— 那只要把它减掉，不就干净了？

这个脚本就模拟完整过程，让你亲眼对比"修之前"和"修之后"差多少。

【完整流程（跟真实产品一模一样）】

  时刻① 空载（人下床了）
        → 传感器读数 = 0 + 零点偏移 + 噪声
        → 我们把这帧记下来，叫"基线"

  时刻② 上人（人躺上去了）
        → 传感器读数 = 真实体压 × 灵敏度 + 零点偏移 + 噪声
        → 这就是"脏数据"，要拿去做 AI 的原始输入

  校准动作（一个减法）
        → 校准后 = 脏数据 - 基线

  关键点：零点偏移【两次一模一样】（它是传感器自己的毛病，不随人变），
  所以一减就消掉了。剩下的只有噪声和灵敏度偏差。

【怎么运行】

    python3 src/zero_calibration.py

跑完会生成「zero_calibration.png」，并打印"修之前/修之后"的误差数字。
=============================================================================
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")   # 不弹窗口，直接把图存成文件
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["PingFang SC", "Arial Unicode MS", "Heiti SC"]
plt.rcParams["axes.unicode_minus"] = False


# ============================================================================
# 参数区（统一从 sensor_config.py 拿，本脚本只覆盖"温漂"这一个值）
# ============================================================================
from sensor_config import (
    GRID_W, GRID_H, BODY_PARTS,
    SENS_GAIN_STD, ZERO_DRIFT_MAX, NOISE_STD,
    OUT_DIR, SEED, FILE_ZERO_CAL,
    make_ideal_pressure,
)
from verdict import describe_improvement

# 温漂单独调大：让"脏"得更明显、一眼看得出区别。
# （真实产品没这么夸张，这里是为了教学看效果。改这个不影响其他脚本。）
TEMP_DRIFT = 30


# ============================================================================
# 核心函数
# ============================================================================
# （make_ideal_pressure 已移到 sensor_config.py，见上面的 import）


def simulate_calibration(seed=SEED):
    """
    跑一遍完整的「零点校准」模拟，返回计算结果。

    【为什么要把这个从 main() 里抽出来？】
    因为测试要验证"校准有没有效"。如果测试里自己再写一遍同样的计算，
    那测试和脚本就成了两份代码，哪天改了一个忘了另一个，测试就骗人了。
    抽成一个函数，让 main() 和测试都调用它，保证测的就是真实跑的那套逻辑。

    返回一个字典，key 和含义：
      ideal        标准答案（人真实压上去该长什么样）
      raw          脏数据（校准前，传感器直接读到的）
      calibrated   校准后（减掉基线）
      baseline     空载时的基线（那个"白送的假数"）
      err_before   校准前跟标准答案的平均误差
      err_after    校准后跟标准答案的平均误差
    """
    rng = np.random.default_rng(seed)   # 固定随机种子，结果可复现

    # ---- 1. 传感器的"自身毛病"（不随人变，固定）----
    sensitivity = rng.normal(1.0, SENS_GAIN_STD, size=(GRID_H, GRID_W))  # 灵敏度不一
    temp_map = TEMP_DRIFT * rng.uniform(0.7, 1.3, size=(GRID_H, GRID_W))  # 温漂
    zero_map = rng.uniform(0, ZERO_DRIFT_MAX, size=(GRID_H, GRID_W))      # 零点漂移
    offset = temp_map + zero_map                                          # 合起来叫"零点偏移"

    # ---- 2. 时刻① 空载：基线 = 偏移 + 噪声 ----
    baseline = offset + rng.normal(0, NOISE_STD, size=(GRID_H, GRID_W))

    # ---- 3. 时刻② 上人：脏数据 = 真实体压 × 灵敏度 + 偏移 + 噪声 ----
    ideal = make_ideal_pressure(GRID_W, GRID_H, BODY_PARTS)
    raw = ideal * sensitivity + offset + rng.normal(0, NOISE_STD, size=(GRID_H, GRID_W))

    # ---- 4. 校准动作：一个减法 ----
    calibrated = raw - baseline

    # ---- 5. 平均绝对误差（每个点"跟正确答案差多少"再取平均）----
    err_before = np.abs(raw - ideal).mean()
    err_after = np.abs(calibrated - ideal).mean()

    return {
        "ideal": ideal,
        "raw": raw,
        "calibrated": calibrated,
        "baseline": baseline,
        "err_before": err_before,
        "err_after": err_after,
    }


def main():
    print("=" * 62)
    print(" 零点校准演示：人下床归零，是不是能把脏数据修干净？")
    print("=" * 62)
    print()

    # 调用抽出来的核心函数（测试也调同一个，保证测的就是真实逻辑）
    r = simulate_calibration(seed=42)
    ideal = r["ideal"]
    raw = r["raw"]
    calibrated = r["calibrated"]
    baseline = r["baseline"]
    err_before = r["err_before"]
    err_after = r["err_after"]
    improve = (err_before - err_after) / err_before * 100

    print(f"[空载] 没人躺，读数却高达 {baseline.max():.1f}（本该是 0）")
    print(f"       这 {baseline.max():.1f} 就是传感器'白送的假数'，记下来当基线")
    print()
    print(f"[上人] 脏数据已生成，跟标准答案平均误差 = {err_before:.1f}")
    print(f"[校准] 减掉基线后，跟标准答案平均误差 = {err_after:.1f}")
    print(f"       ✅ 误差缩小了 {improve:.0f}%")
    print()

    # ---- 6. 画图对比 ----
    # 关键教训：之前三张图各自独立缩放色标，导致脏数据整体抬高
    # 却看不出差别。现在改成【统一色标】+【误差图】，区别一眼可见。
    out_dir = OUT_DIR
    os.makedirs(out_dir, exist_ok=True)

    # 主体图统一用 0~200 的色标，这样"整体抬高了"会直接显示成"更亮"
    VMAX = 200

    # 误差图的色标范围（对称，红=偏大，蓝=偏小）
    EMAX = max(np.abs(raw - ideal).max(), np.abs(calibrated - ideal).max())

    fig, axes = plt.subplots(2, 3, figsize=(20, 13))

    # ---- 上排：三张主体图（统一色标）----
    for ax, data, title in zip(
        axes[0],
        [ideal, raw, calibrated],
        ["标准答案\n（人真实压上去）",
         "脏数据\n（校准前）",
         "校准后\n（减掉基线）"],
    ):
        im = ax.imshow(data, cmap="hot", interpolation="bicubic",
                       vmin=0, vmax=VMAX)
        ax.set_title(title, fontsize=15)
        ax.set_xticks([])
        ax.set_yticks([])
        fig.colorbar(im, ax=ax, fraction=0.046, label="压力")

    # ---- 下排：两张误差图（离正确答案差多少，这才是重点）----
    err_dirty = raw - ideal
    err_cal = calibrated - ideal

    im1 = axes[1, 0].imshow(err_dirty, cmap="RdBu_r", interpolation="bicubic",
                            vmin=-EMAX, vmax=EMAX)
    axes[1, 0].set_title("脏数据的误差\n（红=读数偏大）", fontsize=15)
    axes[1, 0].set_xticks([])
    axes[1, 0].set_yticks([])
    fig.colorbar(im1, ax=axes[1, 0], fraction=0.046, label="误差")

    im2 = axes[1, 2].imshow(err_cal, cmap="RdBu_r", interpolation="bicubic",
                            vmin=-EMAX, vmax=EMAX)
    # 标题不再写死「几乎全白=修干净了」这种主观词，改成直接显示数字。
    # 白不白、干不干净，你拿数字说话、拿眼睛看图判断，程序不替你下结论。
    axes[1, 2].set_title(f"校准后的误差\n（平均误差 {err_after:.1f}）", fontsize=15)
    axes[1, 2].set_xticks([])
    axes[1, 2].set_yticks([])
    fig.colorbar(im2, ax=axes[1, 2], fraction=0.046, label="误差")

    # 中间格子写文字说明
    axes[1, 1].axis("off")
    axes[1, 1].text(
        0.5, 0.5,
        "误差图怎么看：\n\n"
        "白色 = 0 = 完全正确\n"
        "红色 = 读数偏大\n"
        "蓝色 = 读数偏小\n\n"
        f"校准前平均误差 {err_before:.1f}\n"
        f"校准后平均误差 {err_after:.1f}\n"
        f"缩小了 {improve:.0f}%",
        ha="center", va="center", fontsize=16,
    )

    fig.suptitle("零点校准：统一色标下，脏数据明显'发红发虚'，校准后重新变干净",
                 fontsize=17)
    plt.tight_layout()
    path = os.path.join(out_dir, FILE_ZERO_CAL)
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    print(f" 产出图：{path}")
    print()
    # 结尾不再写死「是不是跟左边像多了」，改成根据 improve 数字动态判断。
    # 修没修干净，由数字分档决定，程序如实说。
    print(f" ✅ 完成。误差缩小了 {improve:.0f}%：{describe_improvement(improve)}")
    print("    （图：最左是真相，中间是脏数据，最右是校准后，可自行对照）")


if __name__ == "__main__":
    main()
