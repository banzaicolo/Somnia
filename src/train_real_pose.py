#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=============================================================================
Real-data training chain — feed PoPu real-person data into the neural network
to get the "real accuracy"
=============================================================================

[How is this script different from train_pose_classifier.py?]

  train_pose_classifier.py  → synthetic data (math-generated), accuracy 100% (ideal)
  this script               → PoPu real-person data (60 people), accuracy is the [real number]

Real data is much harder than synthetic:
  - weak signal: only 0~12 of variation after baseline subtraction (synthetic is 0~150)
  - individual differences: 60 people with different height, weight, and mattress firmness
  - left-right mirroring: left-side and right-side maps are mirror images, easiest to confuse

[Two exams, two scores (this is the most important design of this script)]

  Exam 1 random split: one person's data may be half training half testing
        → the model "has seen this person", score is inflated
  Exam 2 split by person: train on 48 people, test on 12 other strangers
        → the model "has never seen the people in the test", which is the real
          deployment scenario (a new user buys it home and the model has never seen them)

  Both scores are reported. The public-facing "real accuracy" is Exam 2 — it has no water in it.

[How to run]

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


# ---- Training hyperparameters (same order of magnitude as the synthetic version, for a fair comparison) ----
N_HIDDEN = 32
N_EPOCH = 150
BATCH = 32
LR = 0.3
SEED = 0


def train_model(X_train, y_train, n_classes, n_hidden, n_epoch, batch, lr, seed):
    """Train a model and return (model, loss curve, training accuracy curve)."""
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
    """Exam: return (accuracy, kappa, confusion matrix)."""
    acc = model.accuracy(X_test, y_test)
    preds = model.predict(X_test)
    kappa = cohens_kappa(y_test, preds, len(classes))
    cm = np.zeros((len(classes), len(classes)), dtype=int)
    for t, p in zip(y_test, preds):
        cm[t, p] += 1
    return acc, kappa, cm


def main(n_hidden=N_HIDDEN, n_epoch=N_EPOCH, batch=BATCH, lr=LR, seed=SEED):
    print("=" * 62)
    print(" Real-data training: PoPu 60 people × 4 poses (supine/left side/right side/prone)")
    print("=" * 62)
    print()

    # ---- 1. Load the real data ----
    X, y, classes, meta = pl.load_dataset(seed=seed)
    print(f"[1/4] Data loaded: {len(X)} real samples, {np.bincount(y)[0]} per class")

    # ---- 2. Exam 1 random split (model may have "seen the person") ----
    Xtr1, ytr1, Xte1, yte1 = pl.split_train_test(X, y, seed=seed)
    m1, losses1, _ = train_model(Xtr1, ytr1, len(classes),
                                 n_hidden, n_epoch, batch, lr, seed)
    acc1, kappa1, _ = evaluate(m1, Xte1, yte1, classes)
    print(f"[2/4] Exam 1 random split (model has seen the person): accuracy {acc1*100:.1f}%, κ={kappa1:.3f}")

    # ---- 3. Exam 2 split by person (test set is all strangers) — the headline number ----
    Xtr2, ytr2, Xte2, yte2 = pl.split_by_volunteer(X, y, meta, seed=seed)
    n_train_v = len(set(m["volunteer_id"] for m in meta)) - 12   # number of training people
    m2, losses2, _ = train_model(Xtr2, ytr2, len(classes),
                                 n_hidden, n_epoch, batch, lr, seed)
    acc2, kappa2, cm = evaluate(m2, Xte2, yte2, classes)
    print(f"[3/4] Exam 2 split by person (test is all strangers): accuracy {acc2*100:.1f}%, κ={kappa2:.3f}")
    print(f"      (trained on {n_train_v} people, tested on 12 unseen strangers, {len(Xte2)} samples total)")

    # ---- 4. Confusion matrix + per-class accuracy (based on Exam 2) ----
    print(f"[4/4] Confusion matrix on the stranger exam (row = true, column = predicted):")
    print()
    print("      Per-class accuracy:")
    for i, cls in enumerate(classes):
        c, t = cm[i, i], cm[i].sum()
        print(f"        {pl.POSE_LABELS_CN[cls]:>4}({cls:>6}): {c}/{t} = {c/t*100:.0f}%")
    print()
    header = "              " + "  ".join(f"{pl.POSE_LABELS_CN[c]:>4}" for c in classes)
    print(header)
    for i, cls in enumerate(classes):
        row = "  ".join(f"{v:>4}" for v in cm[i])
        print(f"        true {pl.POSE_LABELS_CN[cls]:<4} {row}")

    # ---- Honest interpretation ----
    print()
    print("      ── Honest interpretation (facts of this run, not polite filler) ──")
    print(f"      · Synthetic data (math-generated): 3 classes 100% — only proves the algorithm logic is right")
    print(f"      · Real · seen the person: {acc1*100:.1f}% — contains the 'memorized the user' inflation")
    print(f"      · Real · stranger: {acc2*100:.1f}% — a new user buys it and uses it right away; this is the real performance")
    print(f"      · The gap between the two = the inflation from the model 'remembering old users';")
    print(f"        in the README and all public claims, always use the stranger number.")

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
    print(f"\n Figure saved: {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train a pose classifier on PoPu real data (4 classes, including stranger-generalization exam)")
    parser.add_argument("--epochs", type=int, default=N_EPOCH)
    parser.add_argument("--batch", type=int, default=BATCH)
    parser.add_argument("--lr", type=float, default=LR)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="[%(asctime)s] %(message)s", datefmt="%H:%M:%S")
    main(n_epoch=args.epochs, batch=args.batch, lr=args.lr, seed=args.seed)
