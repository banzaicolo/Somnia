#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=============================================================================
PoPu real-data visualization — plot the pressure map of a real person on a
mattress so you can see it
=============================================================================

[What is it for?]

All the earlier parts of the project used "synthetic data", where the body shape
was drawn with Gaussian formulas. This script is the first to plot **real pressure
data from a real person**, letting you see for yourself:
  - what a real pressure map actually looks like when a person lies down
  - the real differences between the four poses (supine/left/right/prone)
  - how critical "zero-point calibration" is — without it, the body shape is
    completely drowned by the fake value of 482

[How to run]

    python3 src/pupu_demo.py

It generates outputs/pupu_real_poses.png.

[How to read the figure]

Two rows:
  top row = raw readings (before calibration): the four are nearly identical,
            all ~485, and the body shape is invisible
  bottom row = after baseline subtraction (after calibration): the body shape is
               clear — supine bright in the middle, side poses shifted to one
               side, prone broadly bright
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


# Output filenames are centralized in sensor_config (consistent with the other figures)
OUTPUT_NAME = FILE_PUPU_REAL


def pick_sample_per_pose(data_dir, pose, seed=0):
    """
    Pick one "representative frame" for a pose from the real data to plot.

    Approach: find the first file for that pose and take its first frame (need both
    before and after baseline subtraction).
    Returns: (raw readings matrix 12×6, baseline-subtracted matrix 12×6, volunteer metadata)
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
    print(" PoPu real-data visualization: real pressure map + effect of zero-point calibration")
    print("=" * 62)
    print()

    data_dir = pl.DEFAULT_DATA_DIR

    # Pick one representative frame per class
    samples = {}
    for pose in pl.POSE_ORDER:
        r = pick_sample_per_pose(data_dir, pose)
        if r is None:
            print(f"  ⚠️ No sample found for {pose}")
            continue
        samples[pose] = r
        raw, cal, meta = r
        print(f"  {pl.POSE_LABELS_CN[pose]:>4}: volunteer {meta['volunteer_id']}, "
              f"raw readings {raw.min():.0f}~{raw.max():.0f}, "
              f"after baseline {cal.min():.0f}~{cal.max():.0f}")

    print()
    print("  The line above is the evidence for 'zero-point calibration':")
    print("  raw readings all sit at ~485 and the four poses barely separate;")
    print("  only after subtracting the baseline does the signal surface and the body shape become visible.")

    # ---- Plot: top row raw, bottom row after baseline subtraction ----
    os.makedirs(OUT_DIR, exist_ok=True)
    n = len(samples)
    fig, axes = plt.subplots(2, n, figsize=(4.2 * n, 8.5))

    # Unified color scale: max pressure value after baseline subtraction (ignore noisy negatives)
    vmax = max(np.max(s[1]) for s in samples.values())
    vmax = max(vmax, 1.0)

    for col, pose in enumerate(pl.POSE_ORDER):
        if pose not in samples:
            continue
        raw, cal, meta = samples[pose]

        # Top row: raw readings (before calibration)
        ax_raw = axes[0, col]
        im_raw = ax_raw.imshow(raw, cmap="hot", vmin=0, vmax=raw.max(),
                               interpolation="bicubic")
        ax_raw.set_title(f"{pl.POSE_LABELS_CN[pose]}\nRaw readings (before calibration)", fontsize=13)
        ax_raw.set_xticks([])
        ax_raw.set_yticks([])
        fig.colorbar(im_raw, ax=ax_raw, fraction=0.046)

        # Bottom row: after baseline subtraction (after calibration)
        ax_cal = axes[1, col]
        im_cal = ax_cal.imshow(cal, cmap="hot", vmin=0, vmax=vmax,
                               interpolation="bicubic")
        ax_cal.set_title(f"{pl.POSE_LABELS_CN[pose]}\nAfter baseline subtraction (after calibration)", fontsize=13)
        ax_cal.set_xticks([])
        ax_cal.set_yticks([])
        fig.colorbar(im_cal, ax=ax_cal, fraction=0.046)

    fig.suptitle(
        "Pressure map of a real person lying on the mattress (PoPu dataset)\n"
        "Top: before calibration the body shape is invisible (drowned by ~485 zero offset); bottom: after baseline subtraction, "
        "supine centered, side poses shifted to one side, prone broadly bright",
        fontsize=15,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.93])

    path = os.path.join(OUT_DIR, OUTPUT_NAME)
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print()
    print(f"  ✅ Figure generated: {path}")


if __name__ == "__main__":
    main()
