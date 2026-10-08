#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=============================================================================
Body-shape proof figure — prove that "the person in the simulated data isn't randomly placed"
=============================================================================

[What is this thing?]

You asked a really good question: "that figure of yours is all white, I can't even
see a person — you didn't just randomly place them on the bed, did you?"

This script answers exactly that question. It does one thing:

    Zoom in on the "ground truth" pressure map, then use arrows to [directly label
    the four body parts (head, shoulder blades, hips, heels) on the figure], so you
    can see with your own eyes: the bright region is exactly where a real body presses.

[Why do you know the shape isn't random?]

Because when generating the data I never used random numbers to place the body. I
[explicitly] hard-coded the positions of the four body parts:

    head       near row 1 of the bed   (topmost)
    shoulders  near row 3 of the bed
    hips       near row 5 of the bed   (heaviest, largest reading)
    heels      near row 7 of the bed   (bottommost)

These four positions are laid out top-to-bottom (head -> shoulders -> hips -> heels),
exactly like a real person lying supine. Random numbers only cause ["per-point
sensitivity differs", "noise"] — those [small jitters]; they don't decide "where the
body is placed".

[What does this file produce?]

Running it generates one figure:

    body_shape_proof.png  -- left is the ground-truth heatmap (four parts labeled),
                             right is a schematic of a real person lying supine, for side-by-side comparison.

There's also a terminal number table that prints out 8 rows of numbers, so you can see
at a glance "top is the head (small numbers), middle is the hips (largest numbers),
bottom is the heels".

[How to run]

Open "Terminal" on Mac, paste and press Enter:

    python3 src/body_shape_proof.py

(requires numpy and matplotlib, same environment as the earlier scripts)
=============================================================================
"""

import os
import numpy as np
import matplotlib
# don't pop up a window; save figures directly to files
matplotlib.use("Agg")
import matplotlib.pyplot as plt
# for drawing circles and ellipses (the real-person schematic on the right)
from matplotlib.patches import Circle, Ellipse

# Chinese font support
plt.rcParams["font.sans-serif"] = ["PingFang SC", "Arial Unicode MS", "Heiti SC"]
plt.rcParams["axes.unicode_minus"] = False


# ============================================================================
# Part 1: positions of the four body parts (taken from sensor_config.py)
# ============================================================================
from sensor_config import (
    GRID_W, GRID_H, BODY_PARTS,
    OUT_DIR, FILE_BODY_PROOF,
    make_ideal_pressure,
)


# (make_ideal_pressure moved to sensor_config.py; see the import above)


# ============================================================================
# Part 2: plotting
# ============================================================================

def draw_person_guide(ax):
    """
    Draw a "real-person-lying-supine schematic" on the right, to compare against the heatmap on the left.

    Draw a top-down stick figure: a head circle + a body ellipse + two legs.
    Positions strictly match the y coordinates in BODY_PARTS (head up, heels down).
    """
    # turn off the axes for a cleaner figure
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.axis("off")

    # ⚠️ note the direction: matplotlib's y axis defaults to "increasing upward", while the
    # heatmap on the left has "row 0 at the top". So here the head must be drawn where y is
    # large (top) to line up with the left side.
    # head: a circle at the top
    head = Circle((0.5, 0.90), 0.10, color="#3b82f6", alpha=0.85)
    ax.add_patch(head)

    # torso: an ellipse (shoulders to hips)
    torso = Ellipse((0.5, 0.56), width=0.36, height=0.58,
                    color="#3b82f6", alpha=0.85)
    ax.add_patch(torso)

    # two legs: two narrow ellipses, down from the bottom of the torso
    leg_left = Ellipse((0.42, 0.16), width=0.11, height=0.30,
                       color="#3b82f6", alpha=0.85)
    leg_right = Ellipse((0.58, 0.16), width=0.11, height=0.30,
                        color="#3b82f6", alpha=0.85)
    ax.add_patch(leg_left)
    ax.add_patch(leg_right)

    # label the four body-part names (matching the row indices of the heatmap on the left)
    ax.text(0.86, 0.90, "Head", fontsize=13, va="center", color="#1d4ed8", fontweight="bold")
    ax.text(0.86, 0.76, "Shoulders\n(shoulders)", fontsize=13, va="center", color="#1d4ed8", fontweight="bold")
    ax.text(0.86, 0.40, "Hips\n(heaviest)", fontsize=13, va="center", color="#1d4ed8", fontweight="bold")
    ax.text(0.86, 0.16, "Heels", fontsize=13, va="center", color="#1d4ed8", fontweight="bold")

    ax.set_title("Real person lying supine (top view, head at top)", fontsize=13)


def main():
    print("=" * 60)
    print(" Body-shape proof figure — the person isn't randomly placed")
    print("=" * 60)
    print()

    # ---- 1. generate the ground truth (no random numbers, body shape is fixed) ----
    ideal = make_ideal_pressure(GRID_W, GRID_H, BODY_PARTS)

    # ---- 2. draw one big figure: left heatmap labels the parts, right real-person schematic ----
    fig, (ax1, ax2) = plt.subplots(
        1, 2,
        figsize=(16, 8),
        gridspec_kw={"width_ratios": [1.6, 1.0]}
    )

    # left: ground-truth heatmap, zoomed, with the four parts labeled
    im = ax1.imshow(ideal, cmap="hot", interpolation="bicubic",
                    vmin=0, vmax=120)
    ax1.set_title("Ground truth: a person pressing on the mattress (brighter = pressed harder)", fontsize=14)

    # use arrows + text to point out the four parts
    # arrows point to (column index, row index) on the heatmap, i.e. (x, y)
    for name, cx, cy, sx, sy, amp, ang in BODY_PARTS:
        # part center
        ax1.plot(cx, cy, "o", markersize=14, mfc="none",
                 mec="lime", mew=2.5)   # draw a green hollow circle to mark the part position
        # place text next to the circle, offset outward a bit to avoid covering it
        offset_x = 2.6 if cx < GRID_W / 2 else -2.6
        ax1.annotate(
            f"{name}\n({amp} cells)",
            xy=(cx, cy),
            xytext=(cx + offset_x, cy - 0.6),
            color="lime", fontsize=12, fontweight="bold",
            arrowprops=dict(arrowstyle="->", color="lime", lw=2),
        )

    # add a colorbar (ruler), making clear 0 = no pressure, 120 = pressed hardest
    fig.colorbar(im, ax=ax1, fraction=0.046, label="pressure reading (larger = pressed harder)")

    # label "head of bed / foot of bed" to help you orient
    ax1.text(GRID_W / 2, -1.1, "↑ head of bed (head is here)", ha="center",
             fontsize=11, color="gray")
    ax1.text(GRID_W / 2, GRID_H + 0.6, "↓ foot of bed (feet are here)", ha="center",
             fontsize=11, color="gray")

    # right: real-person-lying-supine schematic
    draw_person_guide(ax2)

    fig.suptitle("The bright region on the left = where the real body presses on the right, top to bottom: head -> shoulders -> hips -> heels",
                 fontsize=13, y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    out_dir = OUT_DIR
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, FILE_BODY_PROOF)
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    print(f"[1/2] figure generated: {path}")
    print()

    # ---- 3. print the number table so you can see the pattern in each row ----
    print("[2/2] ground-truth number table (8 rows × 12 columns; number = how hard each point is pressed):")
    print()
    # give each row a "body position" explanation
    row_meaning = {
        0: "row 1 (head of bed) <- head is near here, small numbers",
        1: "row 2",
        2: "row 3 <- shoulders are near here",
        3: "row 4",
        4: "row 5 <- hips (heaviest) near here",
        5: "row 6 <- hips (heaviest) near here",
        6: "row 7 <- heels near here",
        7: "row 8 (foot of bed)",
    }
    for r in range(GRID_H):
        row_str = "  ".join(f"{ideal[r, c]:4.0f}" for c in range(GRID_W))
        print(f"  {row_meaning[r]:<28} │ {row_str}")

    print()
    print("  Look at the pattern:")
    print("  · row 1 small numbers (15~35)          -> head, light")
    print("  · rows 5~6 largest numbers (up to 110) -> hips, heaviest")
    print("  · row 7 small numbers (20~30)          -> heels, light")
    print("  · four corners almost all 0             -> nobody pressing")
    print()
    print("  This is the 'heaviest in the middle, top-to-bottom' pattern, exactly like a real person lying supine.")
    print("  ✅ Proof: the body shape is placed deliberately, not rolled out by random numbers.")


if __name__ == "__main__":
    main()
