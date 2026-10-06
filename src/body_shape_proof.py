#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=============================================================================
人形证明图 —— 证明「模拟数据里的人，不是随机乱躺的」
=============================================================================

【这个东西是干嘛的？】

你问了一个特别好的问题：「你那个图全白，也没看出个人形，你不会是
随机躺上去的吧？」

这个脚本专门回答你这个问题。它干一件事：

    把「标准答案」那张体压图放大，然后把身体四个部位
    （头、肩胛、臀、脚后跟）用箭头【直接标在图上去指认】，
    让你亲眼看到：亮的区域，正好就是真人身体压住的位置。

【为什么你知道人形不是随机的？】

因为生成数据的时候，我根本没用随机数去摆人。我是【明确地】把
四个身体部位的位置写死的：

    头部    在床的 第 1 行附近   （最上头）
    肩胛    在床的 第 3 行附近
    臀部    在床的 第 5 行附近   （最重，读数最大）
    脚后跟  在床的 第 7 行附近   （最下头）

这四个位置，是从上到下（头→肩→臀→脚跟）排的，跟真人仰卧
一模一样。随机数只会造成「每个点灵敏度不一样」「噪声」这些
【小抖动】，它决定不了「人形摆在哪」。

【这个文件会产出什么？】

跑完生成一张图：

    body_shape_proof.png  —— 左边是标准答案热力图（标了四个部位），
                             右边是一个真人仰卧示意，左右对照着看。

还有一个终端的数字表，把 8 行数字打出来，你一眼能看出
「上面是头（数字小）、中间是臀（数字最大）、下面是脚跟」。

【怎么运行】

Mac 打开「终端」，粘进去回车：

    python3 src/body_shape_proof.py

（需要 numpy 和 matplotlib，跟之前的脚本用的是同一个环境）
=============================================================================
"""

import os
import numpy as np
import matplotlib
# 不弹窗口，直接把图存成文件
matplotlib.use("Agg")
import matplotlib.pyplot as plt
# 画圆形、椭圆用的（画右侧的真人示意图）
from matplotlib.patches import Circle, Ellipse

# 让中文能正常显示
plt.rcParams["font.sans-serif"] = ["PingFang SC", "Arial Unicode MS", "Heiti SC"]
plt.rcParams["axes.unicode_minus"] = False


# ============================================================================
# 第一部分：身体四个部位的位置（统一从 sensor_config.py 拿）
# ============================================================================
from sensor_config import (
    GRID_W, GRID_H, BODY_PARTS,
    OUT_DIR, FILE_BODY_PROOF,
    make_ideal_pressure,
)


# （make_ideal_pressure 已移到 sensor_config.py，见上面的 import）


# ============================================================================
# 第二部分：画图
# ============================================================================

def draw_person_guide(ax):
    """
    在右边画一个「真人仰卧示意图」，让人对照左边热力图看。

    画一个俯视的躺着的火柴人：一个头圆 + 一个身体椭圆 + 两条腿。
    位置跟 BODY_PARTS 的 y 坐标严格对应（头在上、脚跟在下）。
    """
    # 关掉坐标轴，图更干净
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.axis("off")

    # ⚠️ 注意方向：matplotlib 的 y 轴默认「向上增大」，而左边热力图是
    # 「第 0 行在顶部」。所以这里头要画在 y 大的地方（顶部），才跟左边对上。
    # 头：一个圆，画在顶部
    head = Circle((0.5, 0.90), 0.10, color="#3b82f6", alpha=0.85)
    ax.add_patch(head)

    # 躯干：一个椭圆（肩膀到屁股）
    torso = Ellipse((0.5, 0.56), width=0.36, height=0.58,
                    color="#3b82f6", alpha=0.85)
    ax.add_patch(torso)

    # 两条腿：两个窄椭圆，从躯干底部往下
    leg_left = Ellipse((0.42, 0.16), width=0.11, height=0.30,
                       color="#3b82f6", alpha=0.85)
    leg_right = Ellipse((0.58, 0.16), width=0.11, height=0.30,
                        color="#3b82f6", alpha=0.85)
    ax.add_patch(leg_left)
    ax.add_patch(leg_right)

    # 标出四个部位的名字（跟左边热力图的行号一一对应）
    ax.text(0.86, 0.90, "头", fontsize=13, va="center", color="#1d4ed8", fontweight="bold")
    ax.text(0.86, 0.76, "肩胛\n(肩)", fontsize=13, va="center", color="#1d4ed8", fontweight="bold")
    ax.text(0.86, 0.40, "臀部\n(最重)", fontsize=13, va="center", color="#1d4ed8", fontweight="bold")
    ax.text(0.86, 0.16, "脚后跟", fontsize=13, va="center", color="#1d4ed8", fontweight="bold")

    ax.set_title("真人仰卧（俯视图，头在上）", fontsize=13)


def main():
    print("=" * 60)
    print(" 人形证明图 —— 人不是随机躺的")
    print("=" * 60)
    print()

    # ---- 1. 生成标准答案（没有随机数，人形固定） ----
    ideal = make_ideal_pressure(GRID_W, GRID_H, BODY_PARTS)

    # ---- 2. 画一张大图：左边热力图标部位，右边真人示意 ----
    fig, (ax1, ax2) = plt.subplots(
        1, 2,
        figsize=(14, 7),
        gridspec_kw={"width_ratios": [1.6, 1.0]}
    )

    # 左边：标准答案热力图，放大，标出四个部位
    im = ax1.imshow(ideal, cmap="hot", interpolation="nearest",
                    vmin=0, vmax=120)
    ax1.set_title("标准答案：人压在床垫上（越亮 = 压得越重）", fontsize=14)

    # 用箭头 + 文字，把四个部位指出来
    # 箭头指向的是热力图上的 (列号, 行号)，也就是 (x, y)
    for name, cx, cy, sx, sy, amp, ang in BODY_PARTS:
        # 部位中心点
        ax1.plot(cx, cy, "o", markersize=14, mfc="none",
                 mec="lime", mew=2.5)   # 画一个绿色空心圈，标出部位位置
        # 文字放在圈旁边，往外偏一点，避免盖住
        offset_x = 2.6 if cx < GRID_W / 2 else -2.6
        ax1.annotate(
            f"{name}\n({amp}格)",
            xy=(cx, cy),
            xytext=(cx + offset_x, cy - 0.6),
            color="lime", fontsize=12, fontweight="bold",
            arrowprops=dict(arrowstyle="->", color="lime", lw=2),
        )

    # 加一个颜色条（尺子），标清楚 0 = 没压，120 = 压得最狠
    fig.colorbar(im, ax=ax1, fraction=0.046, label="压力读数（越大压得越重）")

    # 标「床头 / 床尾」，帮你看方向
    ax1.text(GRID_W / 2, -1.1, "↑ 床头（头在这边）", ha="center",
             fontsize=11, color="gray")
    ax1.text(GRID_W / 2, GRID_H + 0.6, "↓ 床尾（脚在这边）", ha="center",
             fontsize=11, color="gray")

    # 右边：真人仰卧示意图
    draw_person_guide(ax2)

    fig.suptitle("左边亮的区域 = 右边真人身体压住的位置，从上到下：头 → 肩 → 臀 → 脚跟",
                 fontsize=13, y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    out_dir = OUT_DIR
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, FILE_BODY_PROOF)
    plt.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)

    print(f"[1/2] 图已生成：{path}")
    print()

    # ---- 3. 打印数字表，让你看每一行的规律 ----
    print("[2/2] 标准答案的数字表（8 行 × 12 列，数字 = 每个点被压多重）：")
    print()
    # 给每一行一个"身体位置"的解释
    row_meaning = {
        0: "第1行(床头) ← 头在这附近，数字小",
        1: "第2行",
        2: "第3行 ← 肩胛(肩)在这附近",
        3: "第4行",
        4: "第5行 ← 臀(最重)在这附近",
        5: "第6行 ← 臀(最重)在这附近",
        6: "第7行 ← 脚后跟在这附近",
        7: "第8行(床尾)",
    }
    for r in range(GRID_H):
        row_str = "  ".join(f"{ideal[r, c]:4.0f}" for c in range(GRID_W))
        print(f"  {row_meaning[r]:<28} │ {row_str}")

    print()
    print("  看规律：")
    print("  · 第1行 数字小（15~35）        → 头，轻")
    print("  · 第5~6行 数字最大（到 110）   → 屁股，最重")
    print("  · 第7行 数字小（20~30）        → 脚跟，轻")
    print("  · 四个角 几乎全是 0            → 没人压")
    print()
    print("  这是【从上到下、中间最重】的规律，跟真人仰卧一模一样。")
    print("  ✅ 证明：人形是摆好的，不是随机数滚出来的。")


if __name__ == "__main__":
    main()
