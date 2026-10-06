#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=============================================================================
bcg_monitor.py —— 呼吸/心跳监测（BCG，心冲击图）
=============================================================================

【这东西解决什么问题？】

人躺在床上，心脏每跳一下、胸廓每呼吸一次，身体都会产生极其微小的
震动，通过床垫传到下面的压力传感器。BCG（心冲击图，英文
Ballistocardiography 的缩写）就是"靠这些微震动，无感地测出呼吸和心跳"。

跟手表不一样：手表要贴着皮肤、用光照测血流；BCG 什么都不用戴，
人往床上一躺，传感器就在床垫下面把呼吸和心跳"听"出来。

【能测什么、不能测什么？（诚实的边界）】

  ✅ 呼吸率 —— 胸廓起伏的震动大，最好测，甚至比手表还准。
  ✅ 心率   —— 心脏跳动的震动小，但能测（误差约 ±2 次/分）。
  ✅ 呼吸暂停 —— 胸廓不动了，能发现（婴儿窒息、成人睡眠呼吸暂停都靠它）。
  ❌ 血氧（SpO2）—— 血氧必须用光透过皮肤测（手表那种），
     床垫只有震动信号，物理上测不到血氧。别承诺这个。

【核心原理：一段信号里，混着三个"节奏"】

床垫读到的压力信号，是几样东西叠加在一起的：

  1. 静态体压（直流）—— 人躺上去的"基础重量"，是个不变的常数，去掉即可。
  2. 呼吸波动（低频）—— 胸廓一起一伏，让压力缓慢地上下波动。
     频率低：成人静息约 12~20 次/分 = 0.2~0.33 Hz（赫兹，每秒几次）。
  3. 心跳波动（高频）—— 心脏泵血的微小震动。
     频率高：约 60~100 次/分 = 1~1.67 Hz。
  4. 噪声 —— 传感器自己的底噪，像老电视雪花。

要把呼吸率和心率"挖"出来，诀窍就是：**把信号按频率切开**。
呼吸在低频段、心跳在高频段，各找各的，互不干扰。
这把"按频率切"的工具，叫 FFT（快速傅里叶变换，英文 Fast Fourier
Transform 的缩写）——它能把一段波形，拆成"每个频率各有多少能量"的一张表。

【怎么用？】

    import bcg_monitor as bcg
    t, signal = bcg.simulate_bcg()               # 合成一段床垫信号
    resp = bcg.estimate_breathing_rate(signal)   # 呼吸率（次/分）
    hr   = bcg.estimate_heart_rate(signal)       # 心率（次/分）

=============================================================================
"""

import numpy as np


# ============================================================================
# 监测参数（业务参数，想调只改这里）
# ============================================================================

FS = 50        # 采样率：每秒采 50 个点。真实 BCG 系统常用 50~1000 Hz。
               # 采样率越高越精细，但计算量越大；50 对演示和测试都够用。

# 呼吸和心跳各占一个"频段"（单位 Hz，赫兹 = 每秒几次）。
# 频段范围要罩住所有正常情况，又不能让两个频段重叠打架：
RESP_LOW = 0.1    # 呼吸频段下限：0.1 Hz = 6 次/分（慢呼吸）
RESP_HIGH = 0.6   # 呼吸频段上限：0.6 Hz = 36 次/分（快呼吸、婴儿）
HR_LOW = 0.7      # 心跳频段下限：0.7 Hz = 42 次/分（运动员静息心率）
HR_HIGH = 3.0     # 心跳频段上限：3.0 Hz = 180 次/分（婴儿、运动）


# ============================================================================
# 一、合成信号：用数学"演"一段床垫压力信号（没有硬件也能开发）
# ============================================================================

def simulate_bcg(duration_sec=60.0, fs=FS, resp_rate=15.0, heart_rate=72.0,
                 noise_std=0.05, apnea=None, seed=42):
    """
    合成一段床垫下的压力信号：静态体压 + 呼吸 + 心跳 + 噪声。

    真实 BCG 信号就是这几个成分叠在一起。这里用数学把它们"演"出来，
    作为标准答案，用来开发、验证提取算法（跟标定那一套思路一样：
    先有标准答案，才能判断算法挖得准不准）。

    参数：
      duration_sec —— 信号时长（秒）
      fs           —— 采样率（每秒采几个点）
      resp_rate    —— 真实呼吸率（次/分），这是"标准答案"
      heart_rate   —— 真实心率（次/分），这也是"标准答案"
      noise_std    —— 噪声强度（越大越脏）
      apnea        —— 可选，(start_sec, end_sec)，表示这段"呼吸暂停"
                     （呼吸幅度降为 0，但心跳还在——呼吸暂停 ≠ 心跳停止）。
                     默认 None = 全程正常呼吸。
      seed         —— 随机种子，固定它，每次结果一样，方便对比

    返回：
      t      —— 时间轴（秒）
      signal —— 合成的压力信号（一维数组）
    """
    rng = np.random.default_rng(seed)
    n = int(duration_sec * fs)
    t = np.arange(n) / fs

    resp_freq = resp_rate / 60.0   # 呼吸：次/分 → Hz（每秒几次）
    hr_freq = heart_rate / 60.0    # 心跳：次/分 → Hz

    baseline = 100.0               # 静态体压（直流），代表人的重量
    # 呼吸幅度大（胸廓起伏明显），心跳幅度小（约为呼吸的 1/10）
    respiration = 5.0 * np.sin(2 * np.pi * resp_freq * t)
    heartbeat = 0.5 * np.sin(2 * np.pi * hr_freq * t)

    # 呼吸暂停：把 [start, end) 这段时间的呼吸幅度压成 0
    if apnea is not None:
        start, end = apnea
        mask = (t >= start) & (t < end)
        respiration = np.where(mask, 0.0, respiration)

    noise = rng.normal(0, noise_std, size=n)

    signal = baseline + respiration + heartbeat + noise
    return t, signal


# ============================================================================
# 二、内部工具：FFT 找"某个频段里能量最强的频率"
# ============================================================================

def _dominant_freq(signal, fs, low, high):
    """
    用 FFT（快速傅里叶变换）把信号拆成"每个频率各有多少能量"，
    然后在 [low, high] 这个频段里，找出能量最强的那个频率。

    这个频率 × 60，就是这个频段对应的"每分钟多少次"。
    比如心跳频段里最强的是 1.2 Hz，那心率就是 1.2 × 60 = 72 次/分。

    参数：
      signal —— 一维信号
      fs     —— 采样率
      low, high —— 目标频段（Hz）

    返回：
      主峰频率（Hz）。如果频段里一个点都没有，返回 0。
    """
    n = len(signal)
    spectrum = np.abs(np.fft.rfft(signal))          # 每个频率的能量
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)          # 每个频率是多少 Hz
    mask = (freqs >= low) & (freqs <= high)
    if not mask.any():
        return 0.0
    band_spectrum = spectrum[mask]
    band_freqs = freqs[mask]
    # 频段里一点能量都没有（比如全零信号）→ 没信号，报 0，
    # 而不是让 argmax 无脑返回频段里第一个点
    if band_spectrum.max() <= 0:
        return 0.0
    # 频段内能量最大的那个频率，就是我们要找的"主峰"
    return float(band_freqs[np.argmax(band_spectrum)])


def _band_energy(signal, fs, low, high):
    """算 [low, high] 频段的总能量 = 该频段所有 FFT 幅度之和。"""
    n = len(signal)
    spectrum = np.abs(np.fft.rfft(signal))
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)
    mask = (freqs >= low) & (freqs <= high)
    return float(spectrum[mask].sum())


# ============================================================================
# 三、提取呼吸率、心率：各在各自频段里找主峰
# ============================================================================

def estimate_breathing_rate(signal, fs=FS):
    """
    估算呼吸率（次/分）。

    做法：呼吸是低频震动，所以在「呼吸频段」（0.1~0.6 Hz）里找能量
    最强的频率，再 ×60 换成"每分钟多少次"。

    参数：
      signal —— 床垫压力信号
      fs     —— 采样率

    返回：
      呼吸率（次/分，浮点数）
    """
    peak_hz = _dominant_freq(signal, fs, RESP_LOW, RESP_HIGH)
    return peak_hz * 60.0


def estimate_heart_rate(signal, fs=FS):
    """
    估算心率（次/分）。

    做法：心跳是高频微小震动，所以在「心跳频段」（0.7~3.0 Hz）里找
    能量最强的频率，再 ×60 换成"每分钟多少次"。

    参数：
      signal —— 床垫压力信号
      fs     —— 采样率

    返回：
      心率（次/分，浮点数）
    """
    peak_hz = _dominant_freq(signal, fs, HR_LOW, HR_HIGH)
    return peak_hz * 60.0


# ============================================================================
# 四、呼吸暂停检测：某段时间胸廓不动了
# ============================================================================

def detect_apnea(signal, fs=FS, window_sec=10.0, ratio=0.25):
    """
    检测呼吸暂停。

    原理：呼吸正常时，呼吸频段（0.1~0.6 Hz）里能量充足；呼吸一暂停，
    胸廓不动了，这个频段的能量就塌到接近 0。

    做法（滑窗）：
      1. 把整段信号按 window_sec 切成一段段（不重叠）；
      2. 每段算「呼吸频段的能量」；
      3. 能量 < 全局最大能量的 ratio 倍 → 判为「疑似呼吸暂停」。

    为什么用比例 ratio 不用固定数？因为不同人呼吸幅度差别很大，
    胖瘦、睡姿都会变。用"相对全局最大值"的比例，天然自适应。

    参数：
      signal     —— 床垫压力信号
      fs         —— 采样率
      window_sec —— 每个窗口多长（秒）。太短噪声大，太长不灵敏，10 秒较合适
      ratio      —— 能量低于全局峰值的多少倍算暂停（默认 0.25）

    返回：
      apnea_flags —— list[bool]，每个窗口是否疑似暂停
      window_t    —— list[float]，每个窗口的起始时间（秒）
    """
    win = int(window_sec * fs)
    n_windows = len(signal) // win
    if n_windows == 0:
        return [], []

    energies = []
    for i in range(n_windows):
        seg = signal[i * win:(i + 1) * win]
        energies.append(_band_energy(seg, fs, RESP_LOW, RESP_HIGH))
    energies = np.array(energies)

    peak = energies.max()
    if peak <= 0:
        flags = [False] * n_windows
    else:
        flags = (energies < ratio * peak).tolist()

    window_t = [i * window_sec for i in range(n_windows)]
    return flags, window_t
