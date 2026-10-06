#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=============================================================================
睡姿分类训练脚本 —— 训练出这个项目第一个"真模型"
=============================================================================

【干嘛的？】

把《睡姿数据集生成器》造的 600 张压力图，喂给《迷你神经网络》，
训练它学会区分三种睡姿（仰卧/侧卧/俯卧）。

训练完输出两个东西：
  1. 终端里的准确率数字（考试卷上考了多少分）
  2. outputs/pose_training.png（损失下降曲线 + 混淆矩阵）

【混淆矩阵是啥？】

一张 3×3 的表格，行是"真实睡姿"，列是"模型猜的睡姿"。
对角线（猜对）的数字越大越好；非对角线能看到"模型把什么错认成什么"。
医疗 AI 评价模型都用它，不用笼统的"准确率"——因为要知道错在哪。

【怎么运行】

    python3 src/train_pose_classifier.py

=============================================================================
"""

import argparse
import logging
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["PingFang SC", "Arial Unicode MS", "Heiti SC"]
plt.rcParams["axes.unicode_minus"] = False

from sensor_config import GRID_W, GRID_H, POSE_NAMES, POSE_LABELS_CN, OUT_DIR
from pose_dataset import generate_dataset, split_train_test
from mlp import TinyMLP
from verdict import render_pose_verdict


# ---- 训练超参数的「默认值」（想改不用改代码，命令行传参即可，见文件底部）----
N_PER_CLASS = 200     # 每种睡姿 200 个样本，共 600
N_HIDDEN = 32         # 隐层 32 个神经元
N_EPOCH = 300         # 全部数据过 300 遍
BATCH = 32            # 每次喂 32 个样本（一小口一小口学，比一口闷稳）
LR = 0.5              # 学习率：每次权重挪多大步
SEED = 0              # 固定随机种子，每次跑结果一样


def main(n_per_class=N_PER_CLASS, n_hidden=N_HIDDEN, n_epoch=N_EPOCH,
         batch=BATCH, lr=LR, seed=SEED):
    print("=" * 62)
    print(" 睡姿分类：训练这个项目的第一个真模型")
    print("=" * 62)
    print()

    # ---- 1. 造数据 + 切分 ----
    X, y, classes = generate_dataset(n_per_class, seed=seed)
    X_train, y_train, X_test, y_test = split_train_test(X, y, test_ratio=0.2, seed=seed)
    logging.info("数据集已生成：%d 个训练样本 + %d 个考试样本", len(X_train), len(X_test))
    print(f"      三种睡姿：仰卧(0) / 侧卧(1) / 俯卧(2)")

    # ---- 2. 建网络 ----
    model = TinyMLP(n_input=GRID_W * GRID_H, n_hidden=n_hidden,
                    n_output=len(classes), seed=seed)
    logging.info("网络已建：%d → %d（隐层）→ %d（三类睡姿）",
                 GRID_W * GRID_H, n_hidden, len(classes))

    # ---- 3. 训练 ----
    losses, train_accs = [], []
    rng = np.random.default_rng(seed)
    n = len(X_train)

    for epoch in range(n_epoch):
        # 打乱顺序再切成小批（避免网络"按顺序背答案"）
        idx = rng.permutation(n)
        epoch_loss = 0.0
        for start in range(0, n, batch):
            batch_idx = idx[start:start + batch]
            epoch_loss += model.train_step(X_train[batch_idx], y_train[batch_idx], lr=lr)
        epoch_loss /= (n // batch + 1)

        losses.append(epoch_loss)
        train_accs.append(model.accuracy(X_train, y_train))

        # 每 50 遍报一次进度，让你看到它"在学"（走 logging，带时间戳）
        if (epoch + 1) % 50 == 0:
            logging.info("第 %3d 遍：损失 %.4f，训练集准确率 %.1f%%",
                         epoch + 1, epoch_loss, train_accs[-1] * 100)

    # ---- 4. 考试（用模型从没见过的数据）----
    test_acc = model.accuracy(X_test, y_test)
    print(f"[4/4] 考试结果（{len(X_test)} 个从没见过的样本）：{test_acc*100:.1f}%")

    # 每类准确率 + 混淆矩阵
    preds = model.predict(X_test)
    print()
    print("      每类准确率：")
    cm = np.zeros((len(classes), len(classes)), dtype=int)   # 混淆矩阵
    for true, pred in zip(y_test, preds):
        cm[true, pred] += 1
    for i, cls in enumerate(classes):
        correct = cm[i, i]
        total = cm[i].sum()
        cn = POSE_LABELS_CN[cls]
        print(f"        {cn}({cls:>6})：{correct}/{total} = {correct/total*100:.0f}%")

    print()
    print("      混淆矩阵（行=真实，列=预测，对角线=猜对）：")
    header = "                " + "  ".join(f"{POSE_LABELS_CN[c]:>4}" for c in classes)
    print(header)
    for i, cls in enumerate(classes):
        row = "  ".join(f"{v:>4}" for v in cm[i])
        print(f"        真实{POSE_LABELS_CN[cls]:<4} {row}")

    # ---- 画图：损失曲线 + 混淆矩阵 ----
    plot(losses, train_accs, cm, classes, test_acc)

    # ==================================================================
    # 附加实验：传感器毛病到底伤不伤分类模型？（真实测试，结果反直觉）
    # ==================================================================
    # 把考试卷过一遍"坏传感器"：逐点灵敏度毛病 + 假基线 + 重度串扰，
    # 再用《标定工具链》洗回来，看准确率变化。
    print()
    print("=" * 62)
    print(" 附加实验：传感器毛病 vs 标定救援（实测）")
    print("=" * 62)

    from calibration import correct_crosstalk

    def apply_crosstalk(img, c):
        """给一张图加串扰（跟标定工具链里的正向模型一致）。"""
        padded = np.pad(img, 1, mode="edge")
        nb = (padded[:-2, 1:-1] + padded[2:, 1:-1] +
              padded[1:-1, :-2] + padded[1:-1, 2:]) / 4.0
        return (1 - c) * img + c * nb

    rng_imp = np.random.default_rng(seed + 99)
    pixel_gain = rng_imp.uniform(0.5, 1.5, size=(1, GRID_W * GRID_H))  # 逐点灵敏度
    pixel_offset = rng_imp.uniform(0.02, 0.15, size=(1, GRID_W * GRID_H))  # 假基线
    C_HEAVY = 0.3   # 重度串扰（真实产品 0.1 左右，这里放大伤害）

    def to_dirty(X):
        """干净数据 → 坏传感器读数。"""
        out = X * pixel_gain + pixel_offset
        return np.array([apply_crosstalk(x.reshape(GRID_H, GRID_W), C_HEAVY).ravel()
                         for x in out])

    def to_rescued(Xd):
        """坏读数 → 标定洗回（已知标定参数：减基线、除增益、解串扰）。"""
        out = (Xd - pixel_offset) / pixel_gain
        return np.array([correct_crosstalk(x.reshape(GRID_H, GRID_W), C_HEAVY).ravel()
                         for x in out])

    X_dirty = to_dirty(X_test)
    X_rescued = to_rescued(X_dirty)
    acc_dirty = model.accuracy(X_dirty, y_test)
    acc_rescued = model.accuracy(X_rescued, y_test)
    print(f"  干净数据考试　　　　　　　　：{test_acc*100:.1f}%")
    print(f"  坏传感器考试（增益+基线+串扰）：{acc_dirty*100:.1f}%")
    print(f"  标定洗回后考试　　　　　　　：{acc_rescued*100:.1f}%")

    # ---- 动态结论：程序根据上面三个数字自己说话，不写死 ----
    # 这里调用了「结论生成器」render_pose_verdict：它拿 test_acc /
    # acc_dirty / acc_rescued 三个数字去套判断规则，生成这次该说什么。
    # 数字变，结论就跟着变——不会再出现"数字崩了还在喊稳健"的瞎话。
    print()
    print(" 本次实验结论（程序根据上面的数字自动生成，非写死）：")
    print()
    for line in render_pose_verdict(test_acc, acc_dirty, acc_rescued):
        print("   " + line)
    print()
    # 下面这段是「背景知识」——固定的讲解，不是本次实验的结论。
    # 它解释的是"为什么会有上面那种现象"，跟数字无关，所以可以写死，
    # 但要明确标注：这是知识点，不是结论。
    print(" 附 · 背景知识（固定讲解，不是本次结论）：")
    print("   三种睡姿的「形状」差异很大（侧卧偏一侧 vs 仰卧居中），")
    print("   模型分类主要靠「哪里亮」，对「多亮」（幅度）不敏感——")
    print("   所以它对幅度类的毛病（增益、基线）天然免疫。")
    print("   但标定在「看绝对压力值」的任务里不可替代：压疮预警中，")
    print("   组织长期受压 >32 mmHg 会缺血损伤；增益漂 +30% 会把安全的")
    print("   25 mmHg 读成危险的 32.5 mmHg。分类可容忍毛病，量化任务")
    print("   一点都不能——这就是标定的价值线。")


def plot(losses, train_accs, cm, classes, test_acc):
    os.makedirs(OUT_DIR, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # 左：损失下降曲线（越低越好）
    ax = axes[0]
    ax.plot(losses, color="tab:blue", label="损失（错得多离谱）")
    ax.set_xlabel("训练轮数")
    ax.set_ylabel("损失")
    ax.set_title("训练过程：损失越来越低 = 学得越来越好")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 右：混淆矩阵热图
    ax = axes[1]
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(classes)))
    ax.set_yticks(range(len(classes)))
    ax.set_xticklabels([POSE_LABELS_CN[c] for c in classes])
    ax.set_yticklabels([POSE_LABELS_CN[c] for c in classes])
    ax.set_xlabel("模型猜的")
    ax.set_ylabel("真实睡姿")
    ax.set_title(f"混淆矩阵（测试集准确率 {test_acc*100:.1f}%）")
    # 在格子里填数字
    for i in range(len(classes)):
        for j in range(len(classes)):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")

    plt.tight_layout()
    path = os.path.join(OUT_DIR, "pose_training.png")
    plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"\n 图已保存：{path}")


if __name__ == "__main__":
    # ---- 命令行参数：不改代码就能调参 ----
    # 例如：python3 src/train_pose_classifier.py --epochs 100 --lr 0.3
    parser = argparse.ArgumentParser(
        description="训练睡姿分类器（仰卧/侧卧/俯卧）")
    parser.add_argument("--n-per-class", type=int, default=N_PER_CLASS,
                        help=f"每种睡姿的样本数（默认 {N_PER_CLASS}）")
    parser.add_argument("--epochs", type=int, default=N_EPOCH,
                        help=f"训练轮数（默认 {N_EPOCH}）")
    parser.add_argument("--batch", type=int, default=BATCH,
                        help=f"批大小（默认 {BATCH}）")
    parser.add_argument("--lr", type=float, default=LR,
                        help=f"学习率（默认 {LR}）")
    parser.add_argument("--seed", type=int, default=SEED,
                        help=f"随机种子（默认 {SEED}，固定可复现）")
    args = parser.parse_args()

    # ---- 日志：带时间戳的标准进度输出（工程实践示范）----
    logging.basicConfig(level=logging.INFO,
                        format="[%(asctime)s] %(message)s", datefmt="%H:%M:%S")
    main(n_per_class=args.n_per_class, n_epoch=args.epochs,
         batch=args.batch, lr=args.lr, seed=args.seed)
