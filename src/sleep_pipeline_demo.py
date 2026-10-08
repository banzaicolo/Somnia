#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=============================================================================
sleep_pipeline_demo.py —— 演示「信号 → 特征 → 分期」完整闭环
=============================================================================

【干嘛的？】

前一个模块 sleep_staging_demo.py 里，特征是「凭空合成」的（假装传感器
已经把数字算好了）。本演示做真正打通：从原始压力波形出发，用 FFT 把
5 个特征挖出来，再喂给神经网络分期。完整走一遍：

    合成整夜波形 → 挖特征 → 训练分类器 → 对一整夜分期 → 画图 + 结论

【画四张图】

  ① 波形放大（5 秒）—— 看清「呼吸慢波 + 心跳快波」长什么样
  ② 睡眠分期图 —— 一整夜，上=真实、下=模型预测
  ③ 呼吸率提取对比 —— 从波形挖出的呼吸率（锯齿线） vs 真实中心（阶梯线）
       这张图是「打通成功」的铁证：挖出来的呼吸率稳稳跟着真实值起伏
  ④ 混淆矩阵 —— 模型考试时把什么错认成什么

【怎么运行】

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
    STAGES, STAGE_CN, STAGE_CENTERS, EPOCH_SEC,
)
from mlp import TinyMLP
from verdict import describe_kappa, describe_deep_ratio
# 复用睡眠分期演示里现成的绘图函数和配色，避免重复造轮子
from sleep_staging_demo import STAGE_COLORS, plot_hypnogram, plot_confusion


# ---- 训练超参数（命令行可覆盖）----
N_NIGHTS = 6      # 训练用几个晚上
N_EPOCH = 200     # 训练轮数
N_HIDDEN = 16     # 隐层神经元
BATCH = 64        # 批大小
LR = 0.3          # 学习率
SEED = 42         # 随机种子

NIGHT_EPOCHS = 960   # 演示用的整夜：960 段 × 30 秒 = 8 小时


def train(X_train, y_train, n_hidden, n_epoch, batch, lr, seed):
    """训练分类器，返回模型和每轮损失。"""
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
            print(f"     第 {epoch + 1:>3} 轮：损失 {losses[-1]:.4f}，"
                  f"训练集准确率 {model.accuracy(X_train, y_train) * 100:.1f}%")
    return model, losses


def plot_waveform(ax, signal, fs, start_sec=120.0, span_sec=5.0):
    """放大一小段波形，看清「呼吸慢波 + 心跳快波」。"""
    s = int(start_sec * fs)
    n = int(span_sec * fs)
    t = np.arange(n) / fs
    ax.plot(t, signal[s:s + n], color="#1f77b4", lw=1.2)
    ax.set_xlabel("时间（秒）")
    ax.set_ylabel("压力读数")
    ax.set_title(f"波形放大（第 {start_sec:.0f} 秒起的 {span_sec:.0f} 秒）"
                 "：大起伏=呼吸，细锯齿=心跳")


def plot_resp_extraction(ax, extracted, stages_names):
    """画「挖出的呼吸率 vs 真实中心」，证明特征被准确挖出。

    提取值画成散点不连线——FFT 挖出的频率只能落在离散档位上，
    连线反而会制造视觉毛刺；散点如实反映「挖出的值就这几档」。
    """
    n = len(extracted)
    x = np.arange(n)
    true_centers = np.array([STAGE_CENTERS[s][1] for s in stages_names])

    ax.plot(x, extracted, ".", color="#1f77b4", markersize=2, alpha=0.5,
            label="从波形挖出的呼吸率")
    ax.plot(x, true_centers, color="#d62728", lw=1.6,
            label="真实呼吸率中心")
    ax.set_xlabel("时间（每 30 秒一段）")
    ax.set_ylabel("呼吸率（次/分）")
    ax.set_title("呼吸率：从波形挖出的（蓝点） vs 真实（红线）——蓝点越贴红线越准")
    ax.legend(loc="upper right", fontsize=8)


def main(n_nights=N_NIGHTS, n_epoch=N_EPOCH, n_hidden=N_HIDDEN,
         batch=BATCH, lr=LR, seed=SEED):
    print("=" * 62)
    print(" 打通演示：原始波形 → 挖特征 → 睡眠分期")
    print("=" * 62)
    print()

    # ---- 1. 造训练数据（从信号挖特征，不是凭空合成）----
    print("[1/5] 合成多个晚上 → 从波形挖特征……")
    X, y = sp.extract_dataset(n_nights=n_nights, n_epochs=240, seed=seed)
    X_scaled, mean, std = standardize(X)
    print(f"      共 {len(X)} 个 30 秒段（{n_nights} 晚），每段从波形挖 5 个特征")

    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(X))
    n_test = int(len(X) * 0.2)
    test_idx, train_idx = idx[:n_test], idx[n_test:]
    X_train, y_train = X_scaled[train_idx], y[train_idx]
    X_test, y_test = X_scaled[test_idx], y[test_idx]
    print(f"      训练 {len(train_idx)} 段 + 考试 {len(test_idx)} 段")

    # ---- 2. 训练 ----
    print("[2/5] 训练分类器……")
    model, _ = train(X_train, y_train, n_hidden, n_epoch, batch, lr, seed)

    # ---- 3. 考试 ----
    print("[3/5] 考试（模型从没见过的段）……")
    preds = model.predict(X_test)
    test_acc = model.accuracy(X_test, y_test)
    kappa = cohens_kappa(y_test, preds, len(STAGES))
    cm = confusion_matrix(y_test, preds, len(STAGES))
    print(f"      准确率：{test_acc * 100:.1f}%")
    print(f"      Cohen's Kappa：{kappa:.3f}")

    # ---- 4. 合成一整夜，端到端分期 ----
    print("[4/5] 合成一个 8 小时整夜，端到端分期……")
    night_sig, night_y, night_names = sp.simulate_night_signal(
        n_epochs=NIGHT_EPOCHS, seed=seed + 999)
    night_pred = sp.stage_from_signal(night_sig, model, mean, std)

    # 顺带算「特征提取精度」：挖出的呼吸率/心率 离 真实中心 差多少
    night_X = sp.extract_epoch_features(night_sig)
    true_resp = np.array([STAGE_CENTERS[s][1] for s in night_names])
    true_hr = np.array([STAGE_CENTERS[s][2] for s in night_names])
    resp_mae = float(np.abs(night_X[:, 1] - true_resp).mean())
    hr_mae = float(np.abs(night_X[:, 2] - true_hr).mean())

    # ---- 5. 画图 + 结论 ----
    print("[5/5] 画图 + 生成结论……")
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
    print(f"      图已保存：{path}")

    # ---- 动态结论 ----
    night_counts = np.bincount(night_pred, minlength=len(STAGES))
    deep_ratio = night_counts[STAGES.index("deep")] / night_counts.sum() * 100

    print()
    print("=" * 62)
    print(" 本次「打通」报告（根据数字自动生成，非写死）")
    print("=" * 62)
    print()
    print(f"  特征提取精度：呼吸率平均误差 {resp_mae:.1f} 次/分，"
          f"心率平均误差 {hr_mae:.1f} 次/分")
    print(f"               → 波形里的信息被成功挖出。")
    print()
    print(f"  模型分期质量：准确率 {test_acc * 100:.1f}%，κ = {kappa:.3f}")
    print(f"               → {describe_kappa(kappa)}。")
    print()
    print("  这一晚的睡眠结构（模型预测）：")
    for s in STAGES:
        pct = night_counts[STAGES.index(s)] / night_counts.sum() * 100
        print(f"    {STAGE_CN[s]:<12} {pct:5.1f}%")
    print(f"    → {describe_deep_ratio(deep_ratio)}。")
    print()
    print(" 附 · 诚实说明（固定讲解，不是本次结论）：")
    print("   κ 很高，是因为合成数据里各阶段的信号差异故意分得开、且没有真实")
    print("   世界那些混杂因素（睡姿、床垫位置、个体差异）。κ 高恰恰证明")
    print("   「信号→特征」这条链路没丢信息。真实世界消费级非脑电分期 κ 只有")
    print("   0.3~0.6，医疗级脑电 PSG 才到 κ≥0.81。本功能定位是「睡眠趋势参考」，")
    print("   不是医疗诊断。")
    print()

    return test_acc, kappa


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="睡眠分期打通演示（信号→特征→分期）")
    parser.add_argument("--nights", type=int, default=N_NIGHTS, help="训练用几个晚上")
    parser.add_argument("--epochs", type=int, default=N_EPOCH, help="训练轮数")
    parser.add_argument("--lr", type=float, default=LR, help="学习率")
    parser.add_argument("--seed", type=int, default=SEED, help="随机种子")
    args = parser.parse_args()
    main(n_nights=args.nights, n_epoch=args.epochs, lr=args.lr, seed=args.seed)
