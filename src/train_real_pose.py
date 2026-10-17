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
N_EPOCH = 100
BATCH = 32
LR = 0.3
SEED = 0
FRAMES_PER_FILE = 10   # use every frame in each JSON file (10x the samples)
N_REPEAT = 5           # repeat the by-volunteer split with different seeds


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


def main(n_hidden=N_HIDDEN, n_epoch=N_EPOCH, batch=BATCH, lr=LR, seed=SEED,
         frames_per_file=FRAMES_PER_FILE, n_repeat=N_REPEAT):
    print("=" * 62)
    print(" Real-data training: PoPu 60 people × 4 poses (supine/left side/right side/prone)")
    print("=" * 62)
    print()

    # ---- 1. Load the real data (every frame, not just the first) ----
    X, y, classes, meta = pl.load_dataset(seed=seed, frames_per_file=frames_per_file)
    print(f"[1/5] Data loaded: {len(X)} real samples, {np.bincount(y)[0]} per class")

    # ---- 1b. Mirror augmentation: a left-side map flipped is a right-side map ----
    # shuffle=False is essential: meta is concatenated below, and if X/y were
    # permuted here the volunteer ids would no longer line up, silently breaking
    # the by-volunteer split (and inflating the score through leakage).
    X, y = pl.mirror_augment(X, y, classes, shuffle=False)
    meta = meta + meta                      # keep metadata aligned with the doubled set
    print(f"      after left/right mirror augmentation: {len(X)} samples")

    # ---- 2. Exam 1: random split (model may have "seen the person") ----
    Xtr1, ytr1, Xte1, yte1 = pl.split_train_test(X, y, seed=seed)
    m1, losses1, _ = train_model(Xtr1, ytr1, len(classes),
                                 n_hidden, n_epoch, batch, lr, seed)
    acc1, kappa1, _ = evaluate(m1, Xte1, yte1, classes)
    print(f"[2/5] Exam 1 random split (model has seen the person): accuracy {acc1*100:.1f}%, κ={kappa1:.3f}")

    # ---- 3. Exam 2: repeat the by-volunteer split with different seeds ----
    # One split can land anywhere between ~91% and ~98% by luck, so report the
    # mean and spread over several independent splits instead of a single point.
    accs, kappas = [], []
    acc_at_seed = kappa_at_seed = cm = losses2 = None
    for repeat in range(n_repeat):
        Xtr2, ytr2, Xte2, yte2 = pl.split_by_volunteer(X, y, meta, seed=repeat)
        m2, run_losses, _ = train_model(Xtr2, ytr2, len(classes),
                                        n_hidden, n_epoch, batch, lr, repeat)
        run_acc, run_kappa, run_cm = evaluate(m2, Xte2, yte2, classes)
        accs.append(run_acc)
        kappas.append(run_kappa)
        if repeat == seed or cm is None:
            acc_at_seed, kappa_at_seed = run_acc, run_kappa
            cm, losses2 = run_cm, run_losses

    accs = np.array(accs)
    kappas = np.array(kappas)
    n_train_v = len(set(m["volunteer_id"] for m in meta[:len(meta) // 2])) - 12
    print(f"[3/5] Exam 2 by person, {n_repeat} independent splits "
          f"(test is always unseen strangers): "
          f"accuracy {accs.mean()*100:.1f}% ± {accs.std()*100:.1f}, "
          f"κ={kappas.mean():.3f} ± {kappas.std():.3f}")
    print(f"      per-split accuracy: " + ", ".join(f"{a*100:.1f}%" for a in accs))
    print(f"      (trained on {n_train_v} people, tested on 12 unseen strangers per split)")

    # ---- 4. Confusion matrix + per-class accuracy (one representative split) ----
    print(f"[4/5] Confusion matrix on the stranger exam (row = true, column = predicted):")
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
    print(f"      · Real · stranger ({n_repeat} splits): {accs.mean()*100:.1f}% ± {accs.std()*100:.1f} "
          f"— a new user buys it and uses it right away; this is the real performance")
    print(f"      · The gap between the two = the inflation from the model 'remembering old users';")
    print(f"        in the README and all public claims, always use the stranger number.")

    plot(losses2, cm, classes, acc_at_seed, kappa_at_seed,
         mean_acc=float(accs.mean()), mean_std=float(accs.std()))
    return float(accs.mean()), float(kappas.mean())


def plot(losses, cm, classes, test_acc, kappa, mean_acc=None, mean_std=None):
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
    title = f"One split: accuracy {test_acc*100:.1f}%, kappa {kappa:.3f}"
    if mean_acc is not None and mean_std is not None:
        # Show both so the figure cannot be mistaken for the headline number.
        title += f"\n5-split mean: {mean_acc*100:.1f}% ± {mean_std*100:.1f}"
    ax.set_title(title, fontsize=11)
    for i in range(len(classes)):
        for j in range(len(classes)):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")

    plt.tight_layout()
    path = os.path.join(OUT_DIR, FILE_PUPU_TRAINING)
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"\n Figure saved: {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train a pose classifier on PoPu real data (4 classes, including stranger-generalization exam)")
    parser.add_argument("--epochs", type=int, default=N_EPOCH)
    parser.add_argument("--batch", type=int, default=BATCH)
    parser.add_argument("--lr", type=float, default=LR)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--frames", type=int, default=FRAMES_PER_FILE,
                        help="frames per JSON file (default: all 10)")
    parser.add_argument("--repeats", type=int, default=N_REPEAT,
                        help="independent by-volunteer splits to average over")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="[%(asctime)s] %(message)s", datefmt="%H:%M:%S")
    main(n_epoch=args.epochs, batch=args.batch, lr=args.lr, seed=args.seed,
         frames_per_file=args.frames, n_repeat=args.repeats)
