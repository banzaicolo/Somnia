#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=============================================================================
真实数据训练链 —— 把 PoPu 真人数据喂给神经网络，得到「真实准确率」
=============================================================================

【这个脚本跟 train_pose_classifier.py 有什么不同？】

  train_pose_classifier.py  → 合成数据（数学造的），准确率 100%（理想值）
  本脚本                    → PoPu 真人数据（60 人），准确率是【真实数字】

真实数据比合成数据难得多：
  - 信号弱：减基线后只有 0~12 的起伏（合成数据是 0~150）
  - 个体差异：60 个人的身高体重床垫软硬都不同
  - 左右镜像：左侧卧和右侧卧的图形是镜像关系，最容易混

【两种考试，两种分数（这是本脚本最重要的设计）】

  考试① 随机切分：同一个人的数据可能一半训练一半考试
        → 模型「见过这个人」，分数偏高
  考试② 按人切分：训练用 48 人，考试用另外 12 个陌生人
        → 模型「没见过考试里的人」，这才是真实部署场景
          （新用户买回家，模型没见过他）的真正性能

  两个分数都报。对外的「真实准确率」以考试②为准——它不掺水。

【怎么运行】

    python3 src/train_real_pose.py

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

import pupu_loader as pl
from mlp import TinyMLP
from sleep_staging import cohens_kappa
from sensor_config import OUT_DIR, FILE_PUPU_TRAINING


# ---- 训练超参数（跟合成版保持同量级，公平对比）----
N_HIDDEN = 32
N_EPOCH = 150
BATCH = 32
LR = 0.3
SEED = 0


def train_model(X_train, y_train, n_classes, n_hidden, n_epoch, batch, lr, seed):
    """训练一个模型，返回 (模型, 损失曲线, 训练准确率曲线)。"""
    model = TinyMLP(n_input=X_train.shape[1], n_hidden=n_hidden,
                    n_output=n_classes, seed=seed)
    losses, train_accs = [], []
    rng = np.random.default_rng(seed)
    n = len(X_train)
    for epoch in range(n_epoch):
        idx = rng.permutation(n)
        epoch_loss = 0.0
        for start in range(0, n, batch):
            b = idx[start:start + batch]
            epoch_loss += model.train_step(X_train[b], y_train[b], lr=lr)
        epoch_loss /= (n // batch + 1)
        losses.append(epoch_loss)
        train_accs.append(model.accuracy(X_train, y_train))
    return model, losses, train_accs


def evaluate(model, X_test, y_test, classes):
    """考试：返回 (准确率, kappa, 混淆矩阵)。"""
    acc = model.accuracy(X_test, y_test)
    preds = model.predict(X_test)
    kappa = cohens_kappa(y_test, preds, len(classes))
    cm = np.zeros((len(classes), len(classes)), dtype=int)
    for t, p in zip(y_test, preds):
        cm[t, p] += 1
    return acc, kappa, cm


def main(n_hidden=N_HIDDEN, n_epoch=N_EPOCH, batch=BATCH, lr=LR, seed=SEED):
    print("=" * 62)
    print(" 真实数据训练：PoPu 60 人 × 4 种睡姿（仰卧/左侧卧/右侧卧/俯卧）")
    print("=" * 62)
    print()

    # ---- 1. 加载真实数据 ----
    X, y, classes, meta = pl.load_dataset(seed=seed)
    print(f"[1/4] 数据已加载：{len(X)} 个真实样本，每类 {np.bincount(y)[0]} 个")

    # ---- 2. 考试① 随机切分（模型可能「见过本人」）----
    Xtr1, ytr1, Xte1, yte1 = pl.split_train_test(X, y, seed=seed)
    m1, losses1, _ = train_model(Xtr1, ytr1, len(classes),
                                 n_hidden, n_epoch, batch, lr, seed)
    acc1, kappa1, _ = evaluate(m1, Xte1, yte1, classes)
    print(f"[2/4] 考试① 随机切分（模型见过本人）：准确率 {acc1*100:.1f}%，κ={kappa1:.3f}")

    # ---- 3. 考试② 按人切分（考试里全是陌生人）——主数字 ----
    Xtr2, ytr2, Xte2, yte2 = pl.split_by_volunteer(X, y, meta, seed=seed)
    n_train_v = len(set(m["volunteer_id"] for m in meta)) - 12   # 训练人数
    m2, losses2, _ = train_model(Xtr2, ytr2, len(classes),
                                 n_hidden, n_epoch, batch, lr, seed)
    acc2, kappa2, cm = evaluate(m2, Xte2, yte2, classes)
    print(f"[3/4] 考试② 按人切分（考试全是陌生人）：准确率 {acc2*100:.1f}%，κ={kappa2:.3f}")
    print(f"      （训练用 {n_train_v} 人，考试用 12 个没见过的陌生人，共 {len(Xte2)} 个样本）")

    # ---- 4. 混淆矩阵 + 每类准确率（以考试②为准）----
    print(f"[4/4] 陌生人考试的混淆矩阵（行=真实，列=预测）：")
    print()
    print("      每类准确率：")
    for i, cls in enumerate(classes):
        c, t = cm[i, i], cm[i].sum()
        print(f"        {pl.POSE_LABELS_CN[cls]:>4}({cls:>6})：{c}/{t} = {c/t*100:.0f}%")
    print()
    header = "              " + "  ".join(f"{pl.POSE_LABELS_CN[c]:>4}" for c in classes)
    print(header)
    for i, cls in enumerate(classes):
        row = "  ".join(f"{v:>4}" for v in cm[i])
        print(f"        真实{pl.POSE_LABELS_CN[cls]:<4} {row}")

    # ---- 诚实解读 ----
    print()
    print("      ── 诚实解读（本次实验的事实，非客套话）──")
    print(f"      · 合成数据（数学造的）：3 类 100% —— 只证明算法逻辑对")
    print(f"      · 真实·见过本人：{acc1*100:.1f}% —— 含「记住用户」的水分")
    print(f"      · 真实·陌生人：{acc2*100:.1f}% —— 新用户买回家就用，这才是真实性能")
    print(f"      · 两者的差距 = 模型「记住老用户」带来的水分，")
    print(f"        写进 README、对外宣传，一律用陌生人那个数。")

    plot(losses2, cm, classes, acc2, kappa2)
    return acc2, kappa2


def plot(losses, cm, classes, test_acc, kappa):
    os.makedirs(OUT_DIR, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    ax = axes[0]
    ax.plot(losses, color="tab:blue", label="training loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Training loss on real PoPu data")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(classes)))
    ax.set_yticks(range(len(classes)))
    ax.set_xticklabels([pl.POSE_LABELS_EN[c] for c in classes])
    ax.set_yticklabels([pl.POSE_LABELS_EN[c] for c in classes])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True pose")
    ax.set_title(f"Unseen volunteers (accuracy {test_acc*100:.1f}%, kappa {kappa:.3f})")
    for i in range(len(classes)):
        for j in range(len(classes)):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")

    plt.tight_layout()
    path = os.path.join(OUT_DIR, FILE_PUPU_TRAINING)
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"\n 图已保存：{path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="用 PoPu 真实数据训练睡姿分类器（4 类，含陌生人泛化考试）")
    parser.add_argument("--epochs", type=int, default=N_EPOCH)
    parser.add_argument("--batch", type=int, default=BATCH)
    parser.add_argument("--lr", type=float, default=LR)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="[%(asctime)s] %(message)s", datefmt="%H:%M:%S")
    main(n_epoch=args.epochs, batch=args.batch, lr=args.lr, seed=args.seed)
