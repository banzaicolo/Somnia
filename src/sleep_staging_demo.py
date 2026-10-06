#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=============================================================================
sleep_staging_demo.py —— 睡眠分期演示：训练分类器 + 画一张「睡眠分期图」
=============================================================================

【干嘛的？】

1. 用《睡眠分期模块》合成 10 个晚上的数据；
2. 喂给《迷你神经网络》，训练它学会分辨「清醒/浅睡/深睡/做梦」；
3. 合成一个「完整的一整夜」（8 小时），让模型从头到尾分期；
4. 画三张图：
     ① 睡眠分期图（hypnogram）—— 一整夜像一条彩色带子，颜色=阶段；
     ② 各阶段占比 —— 这一晚清醒/浅睡/深睡/做梦各占多少；
     ③ 混淆矩阵 —— 模型考试时把什么错认成什么。

【怎么运行】

    python3 src/sleep_staging_demo.py

=============================================================================
"""

import argparse
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

plt.rcParams["font.sans-serif"] = ["PingFang SC", "Arial Unicode MS", "Heiti SC"]
plt.rcParams["axes.unicode_minus"] = False

from sensor_config import OUT_DIR
from sleep_staging import (
    synthesize_night, generate_dataset, standardize,
    confusion_matrix, cohens_kappa, STAGES, STAGE_CN, EPOCH_SEC,
)
from mlp import TinyMLP
from verdict import describe_kappa, describe_deep_ratio


# ---- 训练超参数默认值（命令行可覆盖）----
N_NIGHTS = 10     # 训练用几个晚上
N_EPOCH = 200     # 训练轮数
N_HIDDEN = 16     # 隐层神经元
BATCH = 64        # 批大小
LR = 0.3          # 学习率
SEED = 42         # 随机种子

# 演示用的「一整夜」：960 个单元 × 30 秒 = 8 小时
NIGHT_EPOCHS = 960

# 四个阶段在「睡眠分期图」上的颜色（参考主流睡眠 App 的配色习惯）
# 清醒=浅红、浅睡=浅蓝、深睡=深蓝、做梦=紫色
STAGE_COLORS = {
    "wake": "#f4a7a3",
    "light": "#9ecae1",
    "deep": "#08519c",
    "rem": "#b39ddb",
}


def train(X_train, y_train, n_hidden, n_epoch, batch, lr, seed):
    """训练分类器。返回训练好的模型和每轮损失。"""
    model = TinyMLP(n_input=X_train.shape[1], n_hidden=n_hidden,
                    n_output=len(STAGES), seed=seed)
    rng = np.random.default_rng(seed)
    n = len(X_train)
    losses = []

    for epoch in range(n_epoch):
        idx = rng.permutation(n)
        epoch_loss = 0.0
        steps = 0
        for start in range(0, n, batch):
            b_idx = idx[start:start + batch]
            epoch_loss += model.train_step(X_train[b_idx], y_train[b_idx], lr=lr)
            steps += 1
        losses.append(epoch_loss / steps)

        if (epoch + 1) % 50 == 0:
            acc = model.accuracy(X_train, y_train)
            print(f"     第 {epoch + 1:>3} 轮：损失 {losses[-1]:.4f}，"
                  f"训练集准确率 {acc * 100:.1f}%")

    return model, losses


def plot_hypnogram(ax, y_true, y_pred, colors):
    """
    画「睡眠分期图」：一整夜像一条彩色带子，每格颜色 = 那个时刻的阶段。
    上面一行是「真实答案」，下面一行是「模型猜的」，方便一眼看出准不准。

    两条带子用同一个矩阵一次画成（两行叠一起），天然等高、无缝拼接。
    """
    n = len(y_pred)
    cmap = ListedColormap([colors[s] for s in STAGES])

    # 两行叠一次画：第一行（顶部）真实，第二行预测
    ax.imshow(np.vstack([y_true, y_pred]), aspect="auto", cmap=cmap,
              vmin=0, vmax=3)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["真实", "预测"])

    # x 轴按小时标（每 120 个单元 = 1 小时）
    hour_epochs = int(3600 / EPOCH_SEC)
    ticks = np.arange(0, n + 1, hour_epochs)
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{int(t // hour_epochs)}h" for t in ticks])
    ax.set_xlabel("睡眠时间")
    ax.set_title("睡眠分期图（一整夜）：上=真实，下=模型预测")

    # 图例放到条带右侧外面，不挡图
    handles = [plt.Rectangle((0, 0), 1, 1, color=colors[s]) for s in STAGES]
    ax.legend(handles, [STAGE_CN[s] for s in STAGES],
              loc="upper left", bbox_to_anchor=(1.005, 1.0),
              fontsize=9, frameon=False)


def plot_stage_ratio(ax, y, colors):
    """画各阶段占比柱状图。"""
    counts = np.bincount(y, minlength=len(STAGES))
    total = counts.sum()
    ratios = counts / total * 100
    bars = [colors[s] for s in STAGES]
    ax.bar(range(len(STAGES)), ratios, color=bars)
    ax.set_xticks(range(len(STAGES)))
    ax.set_xticklabels([STAGE_CN[s] for s in STAGES])
    ax.set_ylabel("占比（%）")
    ax.set_title("这一晚各睡眠阶段占比")
    for i, r in enumerate(ratios):
        ax.text(i, r + 0.5, f"{r:.1f}%", ha="center", fontsize=9)


def plot_confusion(ax, cm):
    """画混淆矩阵热图。"""
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(STAGES)))
    ax.set_yticks(range(len(STAGES)))
    ax.set_xticklabels([STAGE_CN[s] for s in STAGES])
    ax.set_yticklabels([STAGE_CN[s] for s in STAGES])
    ax.set_xlabel("模型猜的")
    ax.set_ylabel("真实阶段")
    ax.set_title("混淆矩阵（模型考试）")
    for i in range(len(STAGES)):
        for j in range(len(STAGES)):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    plt.colorbar(im, ax=ax, fraction=0.046)


def main(n_nights=N_NIGHTS, n_epoch=N_EPOCH, n_hidden=N_HIDDEN,
         batch=BATCH, lr=LR, seed=SEED):
    print("=" * 62)
    print(" 睡眠分期：把一整夜切成「清醒/浅睡/深睡/做梦」")
    print("=" * 62)
    print()

    # ---- 1. 造数据 + 标准化 ----
    print("[1/5] 合成训练数据……")
    X, y = generate_dataset(n_nights=n_nights, n_epochs=240, seed=seed)
    X_scaled, mean, std = standardize(X)
    print(f"      共 {len(X)} 个 30 秒单元（{n_nights} 个晚上）")

    # 切分训练集 / 考试卷
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(X))
    n_test = int(len(X) * 0.2)
    test_idx, train_idx = idx[:n_test], idx[n_test:]
    X_train, y_train = X_scaled[train_idx], y[train_idx]
    X_test, y_test = X_scaled[test_idx], y[test_idx]
    print(f"      训练 {len(train_idx)} 个 + 考试 {len(test_idx)} 个")

    # ---- 2. 训练 ----
    print("[2/5] 训练分类器……")
    model, losses = train(X_train, y_train, n_hidden, n_epoch, batch, lr, seed)

    # ---- 3. 考试 ----
    print("[3/5] 考试（模型从没见过的数据）……")
    preds = model.predict(X_test)
    test_acc = model.accuracy(X_test, y_test)
    kappa = cohens_kappa(y_test, preds, len(STAGES))
    cm = confusion_matrix(y_test, preds, len(STAGES))
    print(f"      准确率：{test_acc * 100:.1f}%")
    print(f"      Cohen's Kappa：{kappa:.3f}")

    # ---- 4. 合成一整夜，模型分期 ----
    print("[4/5] 合成一个 8 小时整夜，模型从头到尾分期……")
    night_X, night_y, _ = synthesize_night(n_epochs=NIGHT_EPOCHS, seed=seed + 999)
    night_X_scaled, _, _ = standardize(night_X, mean, std)
    night_pred = model.predict(night_X_scaled)

    # ---- 5. 画图 + 结论 ----
    print("[5/5] 画图 + 生成结论……")
    fig, axes = plt.subplots(3, 1, figsize=(13, 11))

    plot_hypnogram(axes[0], night_y, night_pred, STAGE_COLORS)
    plot_stage_ratio(axes[1], night_pred, STAGE_COLORS)
    plot_confusion(axes[2], cm)

    plt.tight_layout()
    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, "sleep_staging.png")
    plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"      图已保存：{path}")

    # ---- 动态结论（程序根据真实数字生成，非写死）----
    night_counts = np.bincount(night_pred, minlength=len(STAGES))
    deep_ratio = night_counts[STAGES.index("deep")] / night_counts.sum() * 100
    rem_ratio = night_counts[STAGES.index("rem")] / night_counts.sum() * 100

    print()
    print("=" * 62)
    print(" 本次睡眠报告（根据上面的数字自动生成，非写死）")
    print("=" * 62)
    print()
    print(f"  模型分期质量：准确率 {test_acc * 100:.1f}%，κ = {kappa:.3f}")
    print(f"               → {describe_kappa(kappa)}。")
    print()
    print("  这一晚的睡眠结构（模型预测）：")
    for s in STAGES:
        pct = night_counts[STAGES.index(s)] / night_counts.sum() * 100
        print(f"    {STAGE_CN[s]:<12} {pct:5.1f}%")
    print(f"    → {describe_deep_ratio(deep_ratio)}。")
    print(f"    → 快速眼动(REM)占比 {rem_ratio:.1f}%。")
    print()
    print(" 附 · 诚实说明（固定讲解，不是本次结论）：")
    print("   上述是「合成数据」上的理想成绩——因为合成时各类特征分得比较开，")
    print("   模型容易学好。真实世界没有这么干净：消费级非脑电分期的 κ 一般")
    print("   只有 0.3~0.6（分得清「睡没睡」，但细分浅睡/深睡/做梦误差较大），")
    print("   医疗级脑电 PSG 才能到 κ≥0.81。所以本功能定位是「睡眠趋势参考」，")
    print("   不是医疗诊断。")
    print()

    return test_acc, kappa


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="睡眠分期演示（训练+整夜分期图）")
    parser.add_argument("--nights", type=int, default=N_NIGHTS, help="训练用几个晚上")
    parser.add_argument("--epochs", type=int, default=N_EPOCH, help="训练轮数")
    parser.add_argument("--lr", type=float, default=LR, help="学习率")
    parser.add_argument("--seed", type=int, default=SEED, help="随机种子")
    args = parser.parse_args()
    main(n_nights=args.nights, n_epoch=args.epochs, lr=args.lr, seed=args.seed)
