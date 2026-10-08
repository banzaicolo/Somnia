#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=============================================================================
BCG breathing/heartbeat monitoring demo — act out a mattress signal and see how the algorithm digs out breathing and heart rate
=============================================================================

What does this script demo?

Apply bcg_monitor.py's breathing/heartbeat monitoring to a synthesized mattress
signal so you can see clearly:
  1. what a mattress signal looks like (breathing + heartbeat + noise mixed together)
  2. how the algorithm "splits by frequency" to separate breathing and heartbeat
  3. whether the extracted breathing/heart rate matches the true values
  4. whether the algorithm can spot a "breathing pause" (infant suffocation and adult sleep apnea both rely on it)

The script (60 seconds):
  0~20 s    normal breathing (15 breaths/min) + heartbeat (72 beats/min)
  20~40 s   breathing pause (chest stopped, but heartbeat keeps going) — simulating suffocation
  40~60 s   breathing resumes

How to run

    python3 src/bcg_demo.py

After running, it generates bcg_monitor.png in "outputs".
=============================================================================
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")   # don't pop a window; save the plot directly
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["PingFang SC", "Arial Unicode MS", "Heiti SC"]
plt.rcParams["axes.unicode_minus"] = False

from sensor_config import OUT_DIR, FILE_BCG
import bcg_monitor as bcg


def main():
    print("=" * 64)
    print(" BCG breathing/heartbeat monitoring demo: normal → apnea → recovery")
    print("=" * 64)
    print()

    # ---- ① synthesize a mattress signal (ground truth: breathing 15 breaths/min, heartbeat 72 beats/min) ----
    TRUE_RESP = 15.0
    TRUE_HR = 72.0
    t, signal = bcg.simulate_bcg(
        duration_sec=60, resp_rate=TRUE_RESP, heart_rate=TRUE_HR,
        apnea=(20, 40),   # breathing pause during seconds 20~40
    )
    print("[1/4] Synthesized a 60-second mattress signal (breathing 15 breaths/min + heartbeat 72 beats/min + noise)")
    print(f"      seconds 20~40 were manually set as a 'breathing pause'")

    # ---- ② extract breathing rate and heart rate ----
    est_resp = bcg.estimate_breathing_rate(signal)
    est_hr = bcg.estimate_heart_rate(signal)
    print()
    print("[2/4] Extraction results vs true values:")
    print(f"      breathing rate: true {TRUE_RESP:>4.1f} breaths/min → extracted {est_resp:>4.1f} breaths/min"
          f" (error {abs(est_resp - TRUE_RESP):.1f})")
    print(f"      heart rate:    true {TRUE_HR:>4.1f} beats/min → extracted {est_hr:>4.1f} beats/min"
          f" (error {abs(est_hr - TRUE_HR):.1f})")

    # ---- ③ detect apnea ----
    flags, window_t = bcg.detect_apnea(signal, window_sec=10)
    print()
    print("[3/4] Apnea detection (one window per 10 seconds):")
    apnea_windows = []
    for i, (flag, wt) in enumerate(zip(flags, window_t)):
        mark = "⚠️ suspected apnea" if flag else "normal"
        if flag:
            apnea_windows.append(wt)
        print(f"      seconds {wt:>3.0f}~{wt + 10:>3.0f} → {mark}")
    if apnea_windows:
        print(f"      → successfully caught {len(apnea_windows)} pause(s) (exactly seconds 20~40)")
    else:
        print("      → no pause detected")

    # ---- ④ draw plots ----
    path = plot_signal(t, signal, flags, window_t)
    print()
    print(f" output plot: {path}")
    print()
    print(" ✅ done. Open that plot:")
    print("    top = raw signal (large breathing waves + fine heartbeat ripple; pause interval marked with a red box)")
    print("    middle = zoom into the first 5 seconds, to see breathing (slow wave) and heartbeat (fast wave) superimposed")
    print("    bottom = FFT spectrum; the two peaks are breathing (0.25 Hz) and heartbeat (1.2 Hz)")


def plot_signal(t, signal, apnea_flags, window_t):
    """Draw three subplots: full signal / zoomed detail / FFT spectrum."""
    os.makedirs(OUT_DIR, exist_ok=True)
    fs = bcg.FS

    fig, axes = plt.subplots(3, 1, figsize=(13, 11))

    # ---- subplot 1: 60 s full view, marking the pause interval ----
    ax = axes[0]
    ax.plot(t, signal, color="#2c5f8a", linewidth=1.0, label="mattress pressure signal")
    # draw a red box over the pause interval
    for i, flag in enumerate(apnea_flags):
        if flag:
            ax.axvspan(window_t[i], window_t[i] + 10, color="red", alpha=0.25)
    ax.set_title("① full signal (60 s): red shading = detected apnea")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("pressure reading")
    ax.set_ylim(bottom=signal.min() - 2, top=signal.max() + 2)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # ---- subplot 2: zoom into the first 5 s, to see the slow breathing wave + fast heartbeat ripple ----
    ax = axes[1]
    seg = t < 5
    ax.plot(t[seg], signal[seg], color="#2c5f8a", linewidth=1.4)
    ax.set_title("② zoom into the first 5 s: large slow waves = breathing, fine fast ripple = heartbeat")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("pressure reading")
    ax.grid(True, alpha=0.3)

    # ---- subplot 3: FFT spectrum, marking the two peaks ----
    ax = axes[2]
    n = len(signal)
    spectrum = np.abs(np.fft.rfft(signal))
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)
    # log scale so the large breathing peak and small heartbeat peak are both visible; +1 avoids log(0)
    ax.plot(freqs, np.log10(spectrum + 1), color="#7a4d8f", linewidth=1.4)
    # mark the breathing peak and heartbeat peak
    resp_peak = bcg.estimate_breathing_rate(signal) / 60.0
    hr_peak = bcg.estimate_heart_rate(signal) / 60.0
    ax.axvline(resp_peak, color="green", linestyle="--", linewidth=1.4,
               label=f"breathing peak {resp_peak:.2f} Hz")
    ax.axvline(hr_peak, color="red", linestyle="--", linewidth=1.4,
               label=f"heartbeat peak {hr_peak:.2f} Hz")
    ax.set_xlim(0, 2.0)   # only look at 0~2 Hz, enough to cover breathing and heartbeat
    ax.set_title("③ FFT spectrum: two peaks = breathing (low freq) and heartbeat (high freq) each in its own band")
    ax.set_xlabel("frequency (Hz)")
    ax.set_ylabel("energy (log scale)")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(OUT_DIR, FILE_BCG)
    plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return path


if __name__ == "__main__":
    main()
