#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=============================================================================
sleep_staging_demo.py — sleep staging demo: train a classifier + draw a "sleep staging chart"
=============================================================================

What does this do?

1. synthesize 10 nights of data using the sleep staging module;
2. feed them to the mini neural network and train it to tell "wake/light/deep/REM" apart;
3. synthesize one "complete full night" (8 hours) and stage it end to end;
4. draw three plots:
     ① sleep staging chart (hypnogram) — the whole night as a colored band, color = stage;
     ② stage breakdown — how much of the night is wake/light/deep/REM;
     ③ confusion matrix — what the model mistakes for what on the exam.

How to run

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
    confusion_matrix, cohens_kappa, STAGES, STAGE_LABELS, EPOCH_SEC,
)
from mlp import TinyMLP
from verdict import describe_kappa, describe_deep_ratio


# ---- Training hyperparameter defaults (overridable via CLI) ----
N_NIGHTS = 10     # how many nights to train on
N_EPOCH = 200     # training epochs
N_HIDDEN = 16     # hidden units
BATCH = 64        # batch size
LR = 0.3          # learning rate
SEED = 42         # random seed

# The demo "full night": 960 units × 30 s = 8 hours
NIGHT_EPOCHS = 960

# Colors for the four stages on the "sleep staging chart"
# (following the color conventions of mainstream sleep apps)
# wake = light red, light = light blue, deep = dark blue, REM = purple
STAGE_COLORS = {
    "wake": "#f4a7a3",
    "light": "#9ecae1",
    "deep": "#08519c",
    "rem": "#b39ddb",
}


def train(X_train, y_train, n_hidden, n_epoch, batch, lr, seed):
    """Train the classifier. Return the trained model and per-epoch loss."""
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
            print(f"     epoch {epoch + 1:>3}: loss {losses[-1]:.4f}, "
                  f"train accuracy {acc * 100:.1f}%")

    return model, losses


def plot_hypnogram(ax, y_true, y_pred, colors):
    """
    Draw the "sleep staging chart": the whole night as a colored band, each cell's
    color = the stage at that moment. The top row is the "ground truth", the bottom
    row is "the model's guess", so you can see at a glance whether it's accurate.

    The two bands are drawn in one matrix call (two stacked rows), naturally
    equal height and seamlessly joined.
    """
    n = len(y_pred)
    cmap = ListedColormap([colors[s] for s in STAGES])

    # two rows drawn in one call: row 1 (top) truth, row 2 prediction
    ax.imshow(np.vstack([y_true, y_pred]), aspect="auto", cmap=cmap,
              vmin=0, vmax=3)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["Truth", "Predicted"])

    # x axis in hours (hour_epochs units = 1 hour)
    hour_epochs = int(3600 / EPOCH_SEC)
    ticks = np.arange(0, n + 1, hour_epochs)
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{int(t // hour_epochs)}h" for t in ticks])
    ax.set_xlabel("sleep time")
    ax.set_title("Hypnogram (one full night): top = truth, bottom = prediction")

    # legend outside the band so it doesn't cover the plot
    handles = [plt.Rectangle((0, 0), 1, 1, color=colors[s]) for s in STAGES]
    ax.legend(handles, [STAGE_LABELS[s] for s in STAGES],
              loc="upper left", bbox_to_anchor=(1.005, 1.0),
              fontsize=9, frameon=False)


def plot_stage_ratio(ax, y, colors):
    """Draw the stage-breakdown bar chart."""
    counts = np.bincount(y, minlength=len(STAGES))
    total = counts.sum()
    ratios = counts / total * 100
    bars = [colors[s] for s in STAGES]
    ax.bar(range(len(STAGES)), ratios, color=bars)
    ax.set_xticks(range(len(STAGES)))
    ax.set_xticklabels([STAGE_LABELS[s] for s in STAGES])
    ax.set_ylabel("% of night")
    ax.set_title("Stage breakdown for this night")
    for i, r in enumerate(ratios):
        ax.text(i, r + 0.5, f"{r:.1f}%", ha="center", fontsize=9)


def plot_confusion(ax, cm):
    """Draw the confusion-matrix heatmap."""
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(STAGES)))
    ax.set_yticks(range(len(STAGES)))
    ax.set_xticklabels([STAGE_LABELS[s] for s in STAGES])
    ax.set_yticklabels([STAGE_LABELS[s] for s in STAGES])
    ax.set_xlabel("predicted")
    ax.set_ylabel("true stage")
    ax.set_title("Confusion matrix (test set)")
    for i in range(len(STAGES)):
        for j in range(len(STAGES)):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    plt.colorbar(im, ax=ax, fraction=0.046)


def main(n_nights=N_NIGHTS, n_epoch=N_EPOCH, n_hidden=N_HIDDEN,
         batch=BATCH, lr=LR, seed=SEED):
    print("=" * 62)
    print(" Sleep staging: split a full night into wake/light/deep/REM")
    print("=" * 62)
    print()

    # ---- 1. build data + standardize ----
    print("[1/5] Synthesizing training data ...")
    X, y = generate_dataset(n_nights=n_nights, n_epochs=240, seed=seed)
    X_scaled, mean, std = standardize(X)
    print(f"      {len(X)} 30-second units in total ({n_nights} nights)")

    # split into train set / exam set
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(X))
    n_test = int(len(X) * 0.2)
    test_idx, train_idx = idx[:n_test], idx[n_test:]
    X_train, y_train = X_scaled[train_idx], y[train_idx]
    X_test, y_test = X_scaled[test_idx], y[test_idx]
    print(f"      train {len(train_idx)} units + exam {len(test_idx)} units")

    # ---- 2. train ----
    print("[2/5] Training classifier ...")
    model, losses = train(X_train, y_train, n_hidden, n_epoch, batch, lr, seed)

    # ---- 3. exam ----
    print("[3/5] Exam (data the model has never seen) ...")
    preds = model.predict(X_test)
    test_acc = model.accuracy(X_test, y_test)
    kappa = cohens_kappa(y_test, preds, len(STAGES))
    cm = confusion_matrix(y_test, preds, len(STAGES))
    print(f"      accuracy: {test_acc * 100:.1f}%")
    print(f"      Cohen's Kappa: {kappa:.3f}")

    # ---- 4. synthesize one full night, model stages it ----
    print("[4/5] Synthesizing an 8-hour night, model stages it end to end ...")
    night_X, night_y, _ = synthesize_night(n_epochs=NIGHT_EPOCHS, seed=seed + 999)
    night_X_scaled, _, _ = standardize(night_X, mean, std)
    night_pred = model.predict(night_X_scaled)

    # ---- 5. draw plots + conclusion ----
    print("[5/5] Drawing plots + generating conclusion ...")
    fig, axes = plt.subplots(3, 1, figsize=(13, 11))

    plot_hypnogram(axes[0], night_y, night_pred, STAGE_COLORS)
    plot_stage_ratio(axes[1], night_pred, STAGE_COLORS)
    plot_confusion(axes[2], cm)

    plt.tight_layout()
    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, "sleep_staging.png")
    plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"      plot saved to: {path}")

    # ---- dynamic conclusion (generated from real numbers, not hardcoded) ----
    night_counts = np.bincount(night_pred, minlength=len(STAGES))
    deep_ratio = night_counts[STAGES.index("deep")] / night_counts.sum() * 100
    rem_ratio = night_counts[STAGES.index("rem")] / night_counts.sum() * 100

    print()
    print("=" * 62)
    print(" Sleep report for this session (auto-generated from the numbers above, not hardcoded)")
    print("=" * 62)
    print()
    print(f"  staging quality: accuracy {test_acc * 100:.1f}%, κ = {kappa:.3f}")
    print(f"               → {describe_kappa(kappa)}.")
    print()
    print("  This night's sleep structure (model prediction):")
    for s in STAGES:
        pct = night_counts[STAGES.index(s)] / night_counts.sum() * 100
        print(f"    {STAGE_LABELS[s]:<12} {pct:5.1f}%")
    print(f"    → {describe_deep_ratio(deep_ratio)}.")
    print(f"    → REM ratio {rem_ratio:.1f}%.")
    print()
    print(" Appendix · honest note (fixed explanation, not this run's conclusion):")
    print("   The above is the ideal score on synthesized data — because the classes were")
    print("   deliberately made well-separated, the model learns easily. The real world is")
    print("   not this clean: consumer-grade non-EEG staging typically has κ of only 0.3~0.6")
    print("   (it can tell sleep from wake, but fine-grained light/deep/REM errors are large),")
    print("   while medical-grade EEG PSG reaches κ≥0.81. So this feature is positioned as a")
    print("   'sleep trend reference', not a medical diagnosis.")
    print()

    return test_acc, kappa


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sleep staging demo (train + full-night staging chart)")
    parser.add_argument("--nights", type=int, default=N_NIGHTS, help="how many nights to train on")
    parser.add_argument("--epochs", type=int, default=N_EPOCH, help="training epochs")
    parser.add_argument("--lr", type=float, default=LR, help="learning rate")
    parser.add_argument("--seed", type=int, default=SEED, help="random seed")
    args = parser.parse_args()
    main(n_nights=args.nights, n_epoch=args.epochs, lr=args.lr, seed=args.seed)
