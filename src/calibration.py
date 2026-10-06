#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=============================================================================
传感器标定工具链 —— 把"脏数据"修干净的完整招式
=============================================================================

【先说人话：标定到底是啥？】

传感器会骗人（你已经在《压力模拟器》里亲眼见过了：没人压的四个角
也读出 15~40 的假数）。"标定"就是：想出一套数学动作，把传感器吐出来的
脏数据，还原成接近"人真实压上去"的干净数据。

【为什么这个文件最重要？】

整个 AI 医疗床项目，最容易翻车的地方不是 AI 模型，是传感器标定。
很多团队模型做得挺强，最后死在这。这个文件就是把标定的几招全部写齐，
而且每一招都能单独测、能串成一条完整流水线。

【标定有哪几招？】

传感器读到的脏数据，是这么"变脏"的（从左到右，一步一步污染）：

    真实压力 → ①乘灵敏度 → ②加温漂 → ③加零点漂移 → ④串扰漏电 → ⑤加噪声
    （ideal）                                                          （raw 脏数据）

所以"洗干净"就是反着来，一步一步往回推：

    脏数据 → 去噪声 → ④去串扰 → ③②去加性偏移 → ①去增益 → 干净数据

本文件实现的就是上面这个反向过程。每一招一个函数，最后用 calibrate()
把她们串起来。

【怎么用？】

    import calibration as cal
    cleaned = cal.calibrate(raw, baseline=..., gain_map=..., c=0.1)

=============================================================================
"""

import numpy as np


# ============================================================================
# 零、正向模型（先理解传感器是怎么"污染"数据的，才懂怎么"洗干净"）
# ============================================================================

def forward(ideal, gain, offset, c=0.0, noise=0.0):
    """
    正向模型：把"真实压力"变成"传感器读数"。

    这是整个标定的"反面教材"——传感器真实工作时的污染过程。

    参数：
      ideal  —— 真实体压图（标准答案），(H, W) 的二维数组
      gain   —— 每个点的灵敏度，(H, W)。真实世界每个点不一样（批次不一致）
      offset —— 每个点的零点偏移，(H, W)。温漂 + 老化造成的"假基线"
      c      —— 串扰系数，0~1。0.1 表示 10% 信号漏给邻居
      noise  —— 噪声，(H, W) 或标量。随机抖动

    返回：传感器读到的脏数据（raw），(H, W)
    """
    # ① 乘灵敏度 + ②③ 加偏移（乘性和加性都在这）
    x = ideal * gain + offset

    # ④ 串扰：每个点把一部分信号漏给上下左右四个邻居
    if c > 0:
        padded = np.pad(x, 1, mode="edge")   # 四周垫一圈，边界点也能取邻居
        neighbors = (padded[:-2, 1:-1] + padded[2:, 1:-1] +
                     padded[1:-1, :-2] + padded[1:-1, 2:]) / 4.0
        x = (1 - c) * x + c * neighbors

    # ⑤ 加噪声
    return x + noise


# ============================================================================
# 一、四个校正函数（反向过程，每个对应一类毛病）
# ============================================================================

def correct_crosstalk(raw, c, n_iter=30):
    """
    招④：串扰校正（去"邻居漏电"）。

    串扰的正向是：raw = (1-c)·x + c·(四个邻居的平均)。
    要还原 x，就是解这个方程。因为 c 通常很小（0.1），用"迭代法"最直观：

        第一次猜 x≈raw，然后反复用  x_new = (raw - c·邻居(x)) / (1-c)
        每迭代一次，就更接近真实的 x。

    参数：
      raw    —— 被串扰污染的数据
      c      —— 串扰系数（和正向模型里同一个数）
      n_iter —— 迭代几次。c 越小，收敛越快；30 次足够稳
    """
    x = np.array(raw, dtype=float)
    for _ in range(n_iter):
        padded = np.pad(x, 1, mode="edge")
        neighbors = (padded[:-2, 1:-1] + padded[2:, 1:-1] +
                     padded[1:-1, :-2] + padded[1:-1, 2:]) / 4.0
        x = (raw - c * neighbors) / (1 - c)
    return x


def correct_zero(raw, baseline):
    """
    招②③：零点校准（减法）。

    人下床时，读数本该是 0，但传感器读出一个"假基线"（温漂 + 零点漂移）。
    把空载时的读数记下来当 baseline，之后每个读数都减掉它。

    这就是你之前在《零点校准演示》里见过的"一个减法消掉 82% 误差"。
    """
    return raw - baseline


def correct_gain(raw, gain_map):
    """
    招①：增益校准（除法）。

    每个点的灵敏度不一样（同一个压力，A 点读 110、B 点读 90）。
    标定时用"已知的标准压力"测出每个点的放大倍数（gain_map），
    之后读数除以它，就还原成真实压力。

    注意：gain_map 里如果有接近 0 的值，直接除会爆掉（除 0），
    所以加了个保护：太小的值当成 1（不放大也不缩小）。
    """
    safe_gain = np.where(np.abs(gain_map) < 1e-6, 1.0, gain_map)
    return raw / safe_gain


def correct_temperature_offset(raw, temp, t_ref, tco_map):
    """
    招②进阶：温漂的零点补偿（减一个随温度变的量）。

    传感器的"假基线"会随温度变：温度每升高 1 度，基线飘一点点。
    这个"每度飘多少"叫 TCO（温度-零点系数），每个点不一样。

    标定时在参考温度 t_ref 测了基线；现在运行温度变成了 temp，
    基线多漂了 tco_map × (temp - t_ref) 这么多，把它减掉。

    参数：
      raw     —— 读数
      temp    —— 现在的温度（比如 40）
      t_ref   —— 标定时的参考温度（比如 25）
      tco_map —— 每个点的温度-零点系数，(H, W)
    """
    return raw - tco_map * (temp - t_ref)


def correct_temperature_gain(raw, temp, t_ref, tcs_map):
    """
    招①进阶：温漂的增益补偿（除一个随温度变的量）。

    灵敏度也会随温度变：温度升 1 度，放大倍数变一点点。
    这个叫 TCS（温度-灵敏度系数）。同理，把它除回去。

    tcs_map 是"每度变多少比例"，比如 0.002 表示每度变 0.2%。
    """
    return raw / (1.0 + tcs_map * (temp - t_ref))


def correct_hysteresis(reading, direction, h):
    """
    招（附赠）：迟滞补偿。

    迟滞就是传感器"记仇"：同一个压力，加压时读偏高、减压时读偏低。
    数学模型（跟《压力模拟器》里那条回线一致）：

        加压：reading = p·(1 - h/100) + h
        减压：reading = p·(1 + h/100) - h

    还原真实压力 p，就是把上面两个式子反着解：

        加压：p = (reading - h) / (1 - h/100)
        减压：p = (reading + h) / (1 + h/100)

    参数：
      reading   —— 传感器读数（可以是数组）
      direction —— 方向，"load"=加压（读数偏高） 或 "unload"=减压（读数偏低）
      h         —— 迟滞宽度（跟 HYSTERESIS 同一个数）
    """
    reading = np.asarray(reading, dtype=float)
    if direction == "load":
        return (reading - h) / (1 - h / 100.0)
    elif direction == "unload":
        return (reading + h) / (1 + h / 100.0)
    else:
        # 方向没给对，宁可原样返回也不瞎改
        raise ValueError("direction 必须是 'load' 或 'unload'")


# ============================================================================
# 二、标定参数估计（模拟"真实标定动作"：空载测基线、砝码测增益）
# ============================================================================

def estimate_baseline(offset, c=0.0, noise_std=0.0, rng=None):
    """
    模拟"空载标定"：人下床，读到的就是假基线。

    真实硬件动作：设备一开机、人还没躺上去时，采若干帧读数，取平均，
    得到 baseline。这里我们用已知的 offset（零点偏移图）走一遍正向模型。

    返回：baseline 的估计值（含串扰和噪声，跟真实场景一样"不完美"）。
    """
    if rng is None:
        rng = np.random.default_rng(0)
    noise = rng.normal(0, noise_std, size=np.shape(offset))
    # 空载：ideal = 0，所以 forward(0, gain, offset) = offset 经过串扰和噪声
    return forward(np.zeros_like(offset), np.ones_like(offset), offset, c, noise)


def estimate_gain(ideal_ref, raw_ref, baseline, c=0.0):
    """
    模拟"增益标定"：用已知标准压力压上去，反推每个点的灵敏度。

    真实硬件动作：用一个已知重量的标准砝码（或已知压力垫）压每个点，
    读数 ÷ 标准压力 = 这个点的灵敏度。

    这里：
      ideal_ref —— 标定时用的"标准压力图"（已知）
      raw_ref   —— 压标准压力时传感器读到的数
      baseline  —— 之前空载测的基线
      c         —— 串扰系数

    步骤：先去基线 → 再去串扰 → 除以标准压力 = 每个点的增益。
    """
    # 去加性偏移（减基线）
    x = raw_ref - baseline
    # 去串扰（解漏电）
    if c > 0:
        x = correct_crosstalk(x, c)
    # 除以标准压力，得到增益；标准压力为 0 的地方跳过（避免除 0）
    safe = np.where(np.abs(ideal_ref) < 1e-6, 1.0, ideal_ref)
    return x / safe


# ============================================================================
# 三、完整流水线：把上面所有招串起来
# ============================================================================

def calibrate(raw, baseline=None, gain_map=None, c=0.0,
              temp=None, t_ref=25.0, tco_map=None, tcs_map=None,
              direction=None, h=0.0):
    """
    完整标定流水线：输入脏数据，输出接近"真实压力"的干净数据。

    顺序（和污染顺序正好相反）：

        ① 串扰校正（先解空间上的"邻居漏电"）
        ② 温漂零点补偿（减随温度飘的基线，可选）
        ③ 零点校准（减空载基线）
        ④ 温漂增益补偿（除随温度变的灵敏度，可选）
        ⑤ 增益校准（除每个点的灵敏度）
        ⑥ 迟滞补偿（可选，方向已知时）

    所有参数都可以省略（None 就跳过那一步），方便你单独调某一招。
    """
    result = np.array(raw, dtype=float)

    if c > 0:
        result = correct_crosstalk(result, c)

    if tco_map is not None and temp is not None:
        result = correct_temperature_offset(result, temp, t_ref, tco_map)

    if baseline is not None:
        result = correct_zero(result, baseline)

    if tcs_map is not None and temp is not None:
        result = correct_temperature_gain(result, temp, t_ref, tcs_map)

    if gain_map is not None:
        result = correct_gain(result, gain_map)

    if direction is not None and h > 0:
        result = correct_hysteresis(result, direction, h)

    return result


# ============================================================================
# 四、误差衡量（标定好不好，用数字说话）
# ============================================================================

def mean_abs_error(cleaned, ideal):
    """平均绝对误差：每个点"跟正确答案差多少"再取平均。越小越好。"""
    return float(np.abs(np.asarray(cleaned) - np.asarray(ideal)).mean())


def improvement(err_before, err_after):
    """误差缩小了百分之几。正数=变好了。"""
    if err_before == 0:
        return 0.0
    return float((err_before - err_after) / err_before * 100)
