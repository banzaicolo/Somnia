#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=============================================================================
sleep_pipeline.py —— 打通「信号 → 特征 → 分期」的完整闭环
=============================================================================

【这东西解决什么问题？】

前面两个模块各干一件事，但没有连起来：

  · bcg_monitor.py —— 从床垫压力波形里「挖」出呼吸率、心率（用 FFT 按频率切）
  · sleep_staging.py —— 睡眠分期，但它那 5 个特征（体动/呼吸率/心率/呼吸变异/
                       心率变异）是「凭空合成」的，相当于假装传感器已经把
                       数字算好了、直接递给分期器。

现实里没那么便宜的事——真实的床垫只会给你一段「压力波形」，
你必须自己从波形里把这 5 个数字挖出来。本模块就是把这条链路打通：

    原始压力波形  ──FFT──▶  5 个特征  ──神经网络──▶  睡眠阶段
    （一整夜）      （体动/呼吸/心率/变异）          （醒/浅/深/梦）

做完这一步，整个项目才是一条真正能跑的流水线，而不是几个孤立的零件。

【怎么打通？分三步】

  第一步·合成整夜信号（simulate_night_signal）
      —— 先让睡眠分期模块「排」好一整晚的阶段顺序（浅睡→深睡→做梦…），
        再按每个阶段的生理特点（清醒呼吸快、深睡呼吸慢、做梦心率乱），
        用数学合成对应的压力波形。这样波形和阶段标签是「配对的」。

  第二步·从波形挖特征（extract_epoch_features）
      —— 把整夜波形按 30 秒切段，每一段用 FFT（快速傅里叶变换）：
          体动   → 最低频段（比呼吸还慢的翻身/挪动）的能量
          呼吸率 → 呼吸频段的主峰频率
          心率   → 心跳频段的主峰频率
          呼吸/心率变异 → 段内滑窗，看主峰频率稳不稳
        得到 (N, 5) 的特征矩阵，和睡眠分期模块的 5 列一一对应。

  第三步·训练 + 分期（extract_dataset + stage_from_signal）
      —— 用挖出来的特征训练同一个手写神经网络，再对一整夜分期。

【诚实说明：两个「变异」特征是工程近似】

呼吸变异、心率变异（HRV 的简化）这两个量，真实系统是用「逐拍」测量的——
精确到每一次心跳/每一次呼吸的间隔，需要峰值检测那套更复杂的算法。
本模块为控制复杂度，用「30 秒段内滑窗主峰频率的波动」来做工程近似。
方向是对的（深睡最稳、做梦最乱），但精度有限。这里如实标出，不吹。

【怎么用？】

    import sleep_pipeline as sp
    signal, y, stages = sp.simulate_night_signal(480)   # 合成一整夜波形
    X = sp.extract_epoch_features(signal)               # 从波形挖出 5 特征
    # X: (480, 5)，列顺序同 sleep_staging：体动/呼吸率/心率/呼吸变异/心率变异

=============================================================================
"""

import numpy as np

# 复用 BCG 模块的信号处理（呼吸率/心率提取、频段能量、主峰）
import bcg_monitor as bcg
# 复用睡眠分期模块的阶段定义、生理参数、特征标准化
from sleep_staging import (
    STAGES, STAGE_CENTERS, EPOCH_SEC, standardize, _build_sleep_structure,
)


# ============================================================================
# 一、信号合成参数（可调旋钮都集中在这里）
# ============================================================================

FS = bcg.FS                    # 采样率：每秒 50 个点（跟 BCG 模块一致）
RESP_AMP = 5.0                 # 呼吸的波形幅度（胸廓起伏大，幅度大）
HR_AMP = 0.5                   # 心跳的波形幅度（微震动，约为呼吸的 1/10）
BASELINE = 100.0               # 静态体压（直流），代表人的重量

# 体动：翻身、挪动是「比呼吸更慢」的低频摆动
MOV_FREQ = 0.05                # 体动频率（Hz）= 每 20 秒摆一下
MOV_AMP_SCALE = 10.0           # 把「归一化体动幅度(0~0.7)」放大到波形尺度
                               # 清醒体动 0.70→7.0（翻身明显），深睡 0.03→0.3（几乎不动）

# 体动提取用的「体动频段」（Hz）—— 比呼吸频段(0.1~0.6)更低，避开呼吸
MOV_LOW = 0.02
MOV_HIGH = 0.09

# 变异特征提取用的滑窗（秒）：在 30 秒段内，用 15 秒窗、每 5 秒挪一步
VARIABILITY_WIN_SEC = 15.0
VARIABILITY_HOP_SEC = 5.0


# ============================================================================
# 二、合成「一整夜」的压力波形（带阶段标签）
# ============================================================================

def _synthesize_epoch(resp_rate, hr_rate, resp_var, hr_var, mov_amp, fs, rng):
    """
    合成「一个 30 秒段」的压力波形。

    这段波形 = 静态体压 + 呼吸 + 心跳 + 体动 + 噪声，跟 BCG 模块一个模型，
    但多了两点「睡眠分期」需要的花样：

      1. 呼吸率、心率会「慢漂移」——用 resp_var / hr_var 控制漂移幅度。
         深睡几乎不漂移（稳），做梦/清醒漂移大（乱）。这就是「变异」的真相。
      2. 体动用 mov_amp 控制强度——清醒翻身幅度大，深睡几乎不动。

    漂移用「相位积分」实现（不是硬分段）：瞬时频率随时间平滑变化，
    相位就平滑累积，波形不会在分段处跳变。

    参数：
      resp_rate, hr_rate —— 呼吸率/心率中心值（次/分）
      resp_var, hr_var   —— 呼吸率/心率的漂移幅度（次/分），越大越不稳
      mov_amp            —— 体动幅度（波形尺度）
      fs                 —— 采样率
      rng                —— 随机数生成器（保证可复现）

    返回：一段 (n,) 的压力波形
    """
    n = int(EPOCH_SEC * fs)
    t = np.arange(n) / fs

    # 瞬时呼吸率：中心 + 慢漂移（0.04 Hz 的调制 = 约 25 秒一个起伏）
    resp_inst = resp_rate + resp_var * np.sin(2 * np.pi * 0.04 * t
                                              + rng.uniform(0, 2 * np.pi))
    # 瞬时心率：中心 + 慢漂移（0.06 Hz）
    hr_inst = hr_rate + hr_var * np.sin(2 * np.pi * 0.06 * t
                                        + rng.uniform(0, 2 * np.pi))

    # 相位积分：频率(次/分) ÷ 60 = Hz，累加再 ×2π 得到连续相位
    resp_phase = 2 * np.pi * np.cumsum(resp_inst / 60.0) / fs
    hr_phase = 2 * np.pi * np.cumsum(hr_inst / 60.0) / fs

    respiration = RESP_AMP * np.sin(resp_phase)
    heartbeat = HR_AMP * np.sin(hr_phase)
    movement = mov_amp * np.sin(2 * np.pi * MOV_FREQ * t
                                + rng.uniform(0, 2 * np.pi))
    noise = rng.normal(0, 0.05, n)

    return BASELINE + respiration + heartbeat + movement + noise


def simulate_night_signal(n_epochs=480, fs=FS, seed=42):
    """
    合成「一整夜」的原始压力波形，并带上每个 30 秒段的真实阶段标签。

    这是打通的关键第一步：先让睡眠分期模块排好阶段顺序，再按阶段合成波形，
    于是「波形」和「阶段标签」天然配对——这正是训练和监督要的东西。

    参数：
      n_epochs —— 这一晚多少个 30 秒段（8 小时 = 960）
      fs       —— 采样率
      seed     —— 随机种子，固定它结果可复现

    返回：
      signal —— (n_epochs × 30 × fs,) 的一维压力波形
      y      —— (n_epochs,) 每个段的真实阶段编号（0醒/1浅/2深/3梦）
      stages —— (n_epochs,) 阶段名列表（方便调试查看）
    """
    rng = np.random.default_rng(seed)

    # 1) 先排好整晚阶段顺序（复用睡眠分期模块，深睡前半夜多、做梦后半夜多）
    stage_names = _build_sleep_structure(n_epochs, rng)

    # 2) 逐段按阶段合成波形
    n_per = int(EPOCH_SEC * fs)
    signal = np.zeros(n_epochs * n_per)
    for i, stage in enumerate(stage_names):
        c = STAGE_CENTERS[stage]          # [体动, 呼吸率, 心率, 呼吸变异, 心率变异]
        mov_amp = c[0] * MOV_AMP_SCALE    # 归一化体动 → 波形幅度
        seg = _synthesize_epoch(c[1], c[2], c[3], c[4], mov_amp, fs, rng)
        signal[i * n_per:(i + 1) * n_per] = seg

    y = np.array([STAGES.index(s) for s in stage_names], dtype=int)
    return signal, y, stage_names


# ============================================================================
# 三、从波形「挖」出 5 个特征（打通的关键第二步）
# ============================================================================

def _rate_variability(seg, fs, low, high):
    """
    估「某个频段主峰频率的波动」，作为呼吸变异/心率变异的工程近似。

    做法：把这段信号按 15 秒窗、5 秒步长切成几个重叠小窗，每个小窗用
    FFT 找主峰频率，再算这几个主峰的「标准差」。深睡时主峰稳稳不动
    （标准差小），做梦/清醒时主峰忽快忽慢（标准差大）。

    诚实说明：这是工程近似（见模块头部的说明），不是逐拍测量的真 HRV。

    参数：
      seg       —— 一段信号（一个 30 秒段）
      fs        —— 采样率
      low, high —— 目标频段（呼吸频段或心跳频段）

    返回：
      变异值（次/分）。窗口不足 2 个时返回 0。
    """
    win = int(VARIABILITY_WIN_SEC * fs)
    hop = int(VARIABILITY_HOP_SEC * fs)
    if win <= 0 or win > len(seg):
        return 0.0

    rates = []
    for start in range(0, len(seg) - win + 1, hop):
        window = seg[start:start + win]
        peak_hz = bcg.dominant_freq(window, fs, low, high)
        rates.append(peak_hz * 60.0)     # Hz → 次/分

    if len(rates) < 2:
        return 0.0
    return float(np.std(rates))


def extract_epoch_features(signal, fs=FS, epoch_sec=EPOCH_SEC):
    """
    把一整夜原始波形按 30 秒切段，每段挖出 5 个特征。

    5 列顺序跟睡眠分期模块完全一致（这样可以直接对接它的分类器）：
      [0] 体动幅度   —— 体动频段（比呼吸更低频）的能量
      [1] 呼吸率     —— 呼吸频段主峰频率 × 60
      [2] 心率       —— 心跳频段主峰频率 × 60
      [3] 呼吸变异   —— 段内滑窗主峰频率的波动
      [4] 心率变异   —— 段内滑窗主峰频率的波动

    关键细节：提特征前先「去直流」（减去均值）。因为体动频段非常靠近
    0 Hz，若不去掉代表体重的直流分量，直流会泄漏进来污染体动测量。

    参数：
      signal    —— 一整夜波形（一维）
      fs        —— 采样率
      epoch_sec —— 每段多长（默认 30 秒，AASM 分期标准时长）

    返回：
      X —— (n_epochs, 5) 特征矩阵
    """
    n_per = int(epoch_sec * fs)
    n_epochs = len(signal) // n_per

    X = np.zeros((n_epochs, 5))
    for i in range(n_epochs):
        seg = signal[i * n_per:(i + 1) * n_per]
        seg = seg - seg.mean()          # 去直流，别让体重污染低频测量

        X[i, 0] = bcg.band_energy(seg, fs, MOV_LOW, MOV_HIGH)     # 体动
        X[i, 1] = bcg.estimate_breathing_rate(seg, fs)            # 呼吸率
        X[i, 2] = bcg.estimate_heart_rate(seg, fs)                # 心率
        X[i, 3] = _rate_variability(seg, fs, bcg.RESP_LOW, bcg.RESP_HIGH)
        X[i, 4] = _rate_variability(seg, fs, bcg.HR_LOW, bcg.HR_HIGH)

    return X


# ============================================================================
# 四、造多晚训练集 + 端到端分期（第三步）
# ============================================================================

def extract_dataset(n_nights=6, n_epochs=240, fs=FS, seed=42):
    """
    合成多个晚上，逐晚「波形 → 特征」，拼成训练分类器用的大数据集。

    参数：
      n_nights —— 合成几个晚上
      n_epochs —— 每晚上多少个段
      fs       —— 采样率
      seed     —— 随机种子（每晚用 seed+i，模拟「不同的夜」）

    返回：
      X —— (总段数, 5) 特征矩阵
      y —— (总段数,) 阶段编号
    """
    X_list, y_list = [], []
    for i in range(n_nights):
        signal, y, _ = simulate_night_signal(n_epochs=n_epochs, fs=fs,
                                             seed=seed + i)
        X = extract_epoch_features(signal, fs=fs)
        X_list.append(X)
        y_list.append(y)
    return np.vstack(X_list), np.concatenate(y_list)


def stage_from_signal(signal, model, mean, std, fs=FS):
    """
    端到端分期：原始波形 → 特征 → 标准化 → 模型预测。

    参数：
      signal —— 一整夜波形
      model  —— 训练好的分类器（TinyMLP）
      mean, std —— 训练集算出的标准化参数（用同一把尺子）
      fs     —— 采样率

    返回：
      preds —— (n_epochs,) 预测的阶段编号
    """
    X = extract_epoch_features(signal, fs=fs)
    Xs, _, _ = standardize(X, mean, std)
    return model.predict(Xs)
