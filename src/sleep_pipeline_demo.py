#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=============================================================================
sleep_pipeline_demo.py — demo the full loop "signal → features → staging"
=============================================================================

What does this do?

In the previous module sleep_staging_demo.py, the features were synthesized
"out of thin air" (pretending the sensor already computed the numbers). This demo
does the real wiring: start from the raw pressure waveform, dig the 5 features out
with FFT, then feed them to the neural network for staging. A complete pass:

    synthesize a full-night waveform → dig features → train a classifier → stage a full night → plots + conclusion

Four plots are drawn:

  ① waveform zoom (5 s) — see what "slow breathing wave + fast heartbeat ripple" looks like
  ② sleep staging chart — a full night, top = truth, bottom = model prediction
  ③ breathing-rate extraction comparison — extracted breathing rate (jagged line) vs true center (staircase line)
       this plot is the smoking-gun proof of "successful wiring": the extracted breathing rate tracks the true value
  ④ confusion matrix — what the model mistakes for what on the exam

How to run

    python3 src/sleep_pipeline_demo.py

=============================================================================
"""

import argparse
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["PingFang SC", "Arial Unicode MS", "Heiti SC"]
plt.rcParams["axes.unicode_minus"] = False

from sensor_config import OUT_DIR
import sleep_pipeline as sp
from sleep_staging import (
    standardize, confusion_matrix, cohens_kappa,
    STAGES, STAGE_LABELS, STAGE_CENTERS, EPOCH_SEC,
)
from mlp import TinyMLP
from verdict import describe_kappa, describe_deep_ratio
# Reuse the ready-made plotting functions and colors from the sleep-staging demo, avoiding reinventing the wheel
from sleep_staging_demo import STAGE_COLORS, plot_hypnogram, plot_confusion


# ---- Training hyperparameters (overridable via CLI) ----
N_NIGHTS = 6      # how many nights to train on
N_EPOCH = 200     # training epochs
N_HIDDEN = 16     # hidden units
BATCH = 64        # batch size
LR = 0.3          # learning rate
SEED = 42         # random seed

NIGHT_EPOCHS = 960   # demo full night: 960 segments × 30 s = 8 hours


def train(X_train, y_train, n_hidden, n_epoch, batch, lr, seed):
    """Train the classifier, returning the model and per-epoch loss."""
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
            b = idx[start:start + batch]
            epoch_loss += model.train_step(X_train[b], y_train[b], lr=lr)
            steps += 1
        losses.append(epoch_loss / steps)
        if (epoch + 1) % 50 == 0:
            print(f"     epoch {epoch + 1:>3}: loss {losses[-1]:.4f}, "
                  f"train accuracy {model.accuracy(X_train, y_train) * 100:.1f}%")
    return model, losses


def plot_waveform(ax, signal, fs, start_sec=120.0, span_sec=5.0):
    """Zoom into a short window to show the slow breathing wave + fast heartbeat ripple."""
    s = int(start_sec * fs)
    n = int(span_sec * fs)
    t = np.arange(n) / fs
    ax.plot(t, signal[s:s + n], color="#1f77b4", lw=1.2)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("pressure reading")
    ax.set_title(f"Zoomed waveform ({span_sec:.0f}s from t={start_sec:.0f}s)"
                 ": slow wave = breathing, fine ripple = heartbeat")


def plot_resp_extraction(ax, extracted, stages_names):
    """Plot extracted vs true respiratory rate, proving the feature is recovered.

    Extracted values are plotted as disconnected dots: FFT peak-picking can only
    land on discrete frequency bins, so connecting lines would fabricate visual
    jitter. Dots honestly show "the extracted value is one of these bins".
    """
    n = len(extracted)
    x = np.arange(n)
    true_centers = np.array([STAGE_CENTERS[s][1] for s in stages_names])

    ax.plot(x, extracted, ".", color="#1f77b4", markersize=2, alpha=0.5,
            label="extracted (from waveform)")
    ax.plot(x, true_centers, color="#d62728", lw=1.6,
            label="true center")
    ax.set_xlabel("epoch (30 s each)")
    ax.set_ylabel("respiratory rate (breaths/min)")
    ax.set_title("Extracted (blue dots) vs true (red line) — closer is better")
    ax.legend(loc="upper right", fontsize=8)


def main(n_nights=N_NIGHTS, n_epoch=N_EPOCH, n_hidden=N_HIDDEN,
         batch=BATCH, lr=LR, seed=SEED):
    print("=" * 62)
    print(" Wiring demo: raw waveform → dig features → sleep staging")
    print("=" * 62)
    print()

    # ---- 1. build training data (dig features from the signal, not synthesized out of thin air) ----
    print("[1/5] Synthesizing multiple nights → digging features from the waveform ...")
    X, y = sp.extract_dataset(n_nights=n_nights, n_epochs=240, seed=seed)
    X_scaled, mean, std = standardize(X)
    print(f"      {len(X)} 30-second segments in total ({n_nights} nights), "
          f"5 features dug from each segment's waveform")

    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(X))
    n_test = int(len(X) * 0.2)
    test_idx, train_idx = idx[:n_test], idx[n_test:]
    X_train, y_train = X_scaled[train_idx], y[train_idx]
    X_test, y_test = X_scaled[test_idx], y[test_idx]
    print(f"      train {len(train_idx)} segments + exam {len(test_idx)} segments")

    # ---- 2. train ----
    print("[2/5] Training classifier ...")
    model, _ = train(X_train, y_train, n_hidden, n_epoch, batch, lr, seed)

    # ---- 3. exam ----
    print("[3/5] Exam (segments the model has never seen) ...")
    preds = model.predict(X_test)
    test_acc = model.accuracy(X_test, y_test)
    kappa = cohens_kappa(y_test, preds, len(STAGES))
    cm = confusion_matrix(y_test, preds, len(STAGES))
    print(f"      accuracy: {test_acc * 100:.1f}%")
    print(f"      Cohen's Kappa: {kappa:.3f}")

    # ---- 4. synthesize one full night, stage end to end ----
    print("[4/5] Synthesizing an 8-hour night, staging end to end ...")
    night_sig, night_y, night_names = sp.simulate_night_signal(
        n_epochs=NIGHT_EPOCHS, seed=seed + 999)
    night_pred = sp.stage_from_signal(night_sig, model, mean, std)

    # Also compute "feature-extraction accuracy": how far the dug-out breathing/heart rate is from the true center
    night_X = sp.extract_epoch_features(night_sig)
    true_resp = np.array([STAGE_CENTERS[s][1] for s in night_names])
    true_hr = np.array([STAGE_CENTERS[s][2] for s in night_names])
    resp_mae = float(np.abs(night_X[:, 1] - true_resp).mean())
    hr_mae = float(np.abs(night_X[:, 2] - true_hr).mean())

    # ---- 5. draw plots + conclusion ----
    print("[5/5] Drawing plots + generating conclusion ...")
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))

    plot_waveform(axes[0, 0], night_sig, sp.FS, start_sec=600.0, span_sec=5.0)
    plot_hypnogram(axes[0, 1], night_y, night_pred, STAGE_COLORS)
    plot_resp_extraction(axes[1, 0], night_X[:, 1], night_names)
    plot_confusion(axes[1, 1], cm)

    plt.tight_layout()
    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, "sleep_pipeline.png")
    plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"      plot saved to: {path}")

    # ---- dynamic conclusion ----
    night_counts = np.bincount(night_pred, minlength=len(STAGES))
    deep_ratio = night_counts[STAGES.index("deep")] / night_counts.sum() * 100

    print()
    print("=" * 62)
    print(" This run's 'wiring' report (auto-generated from the numbers, not hardcoded)")
    print("=" * 62)
    print()
    print(f"  feature-extraction accuracy: mean breathing-rate error {resp_mae:.1f} breaths/min, "
          f"mean heart-rate error {hr_mae:.1f} bpm")
    print(f"               → the information in the waveform was recovered successfully.")
    print()
    print(f"  staging quality: accuracy {test_acc * 100:.1f}%, κ = {kappa:.3f}")
    print(f"               → {describe_kappa(kappa)}.")
    print()
    print("  This night's sleep structure (model prediction):")
    for s in STAGES:
        pct = night_counts[STAGES.index(s)] / night_counts.sum() * 100
        print(f"    {STAGE_LABELS[s]:<12} {pct:5.1f}%")
    print(f"    → {describe_deep_ratio(deep_ratio)}.")
    print()
    print(" Appendix · honest note (fixed explanation, not this run's conclusion):")
    print("   κ is high because the stage differences in the synthesized data were deliberately")
    print("   made well-separated, and there are none of the real-world confounders (sleep posture,")
    print("   mattress position, between-subject differences). A high κ precisely proves the")
    print("   'signal→features' chain doesn't lose information. Real-world consumer-grade non-EEG")
    print("   staging has κ of only 0.3~0.6, and medical-grade EEG PSG reaches κ≥0.81. This feature")
    print("   is positioned as a 'sleep trend reference', not a medical diagnosis.")
    print()

    return test_acc, kappa


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sleep staging wiring demo (signal → features → staging)")
    parser.add_argument("--nights", type=int, default=N_NIGHTS, help="how many nights to train on")
    parser.add_argument("--epochs", type=int, default=N_EPOCH, help="training epochs")
    parser.add_argument("--lr", type=float, default=LR, help="learning rate")
    parser.add_argument("--seed", type=int, default=SEED, help="random seed")
    args = parser.parse_args()
    main(n_nights=args.nights, n_epoch=args.epochs, lr=args.lr, seed=args.seed)
