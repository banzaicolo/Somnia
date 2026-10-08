#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=============================================================================
PoPu 真实数据读取器 —— 把下载下来的真人床垫数据，读成能训练的压力图
=============================================================================

【这个文件干嘛的？】

之前项目一直用「合成数据」（数学公式造的）。现在有了 PoPu 真实数据集，
这个文件负责把它读进来、洗干净、变成跟合成数据一样的格式：
一张「压力图」摊平成一串数字 + 一个「睡姿标签」。

【PoPu 数据长什么样？（踩过的坑都记在这）】

  - 目录：pupu_data/PoPu_data/sensomatt_data/<志愿者编号>/<姿势><编号>_<变体>.json
    共 60 个志愿者（文件夹 1~60），每人 86 个文件。
  - 每个 JSON 里：volunteer_id / sex / height / weight /
    sensomatt_columns=6 / sensomatt_rows=12 / position / variation / snapshots
  - snapshots 是一个字典，key 是 '0'~'9'（字符串），每个 value 是一帧：
        {'id': ..., 'sensomatt_readings': [72 个数字]}
    72 = 12 行 × 6 列，前 6 个是第 1 行，再 6 个是第 2 行……（行优先）
  - 姿势从「文件名」解析，不是从 position 字段（others 文件的 position 是 None）。
    supine=仰卧  left=左侧卧  right=右侧卧  prone=俯卧
    另外还有 empty（空载，没人躺）和 others（其他姿势，本文件排除）。

【最关键的一步：减基线（零点校准）】

直接读出来的数是「原始读数」，没人躺也有约 482 的「假数」（零点偏移）。
人压上去只把读数抬高 0~12 —— 信号很弱，全被那个 482 淹没。

所以每个志愿者都有 1 个 empty 文件（10 帧空载），取它的平均当「基线」，
把每个读数减掉基线，人形才真正显出来。这正是标定链里第一招「零点校准」
在真实数据上的真刀真枪应用。

【四种睡姿怎么区分？（看形状，不看幅度）】

  仰卧：中间亮、左右对称（人平躺，肩臀压中间）
  左侧卧：亮区偏左（朝左侧躺，压在左半边）
  右侧卧：亮区偏右
  俯卧：大片均匀亮（胸腹大面积贴床）

=============================================================================
"""

import json
import os
import glob

import numpy as np


# ============================================================================
# 一、常量：姿势顺序、归一化、数据目录
# ============================================================================

# 四种睡姿的顺序（标签 0/1/2/3 就按这个顺序对应）。顺序固定，训练和预测共用。
POSE_ORDER = ["supine", "left", "right", "prone"]

# 中文名（画图、打印用）
POSE_LABELS_CN = {
    "supine": "仰卧",
    "left": "左侧卧",
    "right": "右侧卧",
    "prone": "俯卧",
}

# 英文名（README / 论文配图用）
POSE_LABELS_EN = {
    "supine": "Supine",
    "left": "Left",
    "right": "Right",
    "prone": "Prone",
}

# 数据默认位置（相对项目根目录）
DEFAULT_DATA_DIR = "pupu_data/PoPu_data/sensomatt_data"

# 归一化基准：减基线后信号约 -2 ~ +12，除以 20 拉到 0~0.6 之间。
# 跟合成数据用 /200 是同一个道理：把数变小，让神经网络训练稳定。
SCALE = 20.0

# 传感器阵列尺寸（PoPu 的 SensoMatt 是 12 行 × 6 列 = 72 点）
GRID_ROWS = 12
GRID_COLS = 6


# ============================================================================
# 二、底层读取：单个文件
# ============================================================================

def parse_pose(filename):
    """
    从文件名解析睡姿标签。文件名形如：
        supine1_0.json  → "supine"
        left3_1.json    → "left"
        right5_2.json   → "right"
        prone2_0.json   → "prone"
        empty1.json     → "empty"（空载，不算睡姿）
        others1.json    → "others"（其他姿势，排除）

    为什么不用 JSON 里的 position 字段？因为 others 文件的 position 是 None，
    而文件名始终可靠。文件名就是「标准答案」。
    """
    # 取第一个下划线或数字之前的那段字母
    prefix = ""
    for ch in filename:
        if ch.isalpha():
            prefix += ch
        else:
            break
    return prefix.lower()


def load_snapshots(path):
    """
    读一个 JSON 文件，返回它所有的「帧」。

    返回：
      matrices —— list，每个元素是一张 (12, 6) 的压力图（原始读数，未减基线）
      meta     —— dict，志愿者的身高体重等元数据
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # snapshots 的 key 是字符串 '0'~'9'，按数字大小排序，保证帧顺序正确
    keys = sorted(data["snapshots"].keys(), key=int)
    matrices = []
    for k in keys:
        readings = data["snapshots"][k]["sensomatt_readings"]
        # 真实数据偶有「脏帧」：读数不是 72 个（缺 1~2 个值）。
        # 这种帧跳过即可——占比不到万分之四，硬补反而引入假数据。
        if len(readings) != GRID_ROWS * GRID_COLS:
            continue
        # 72 个数 → 12 行 × 6 列（行优先：前 6 个是第一行）
        matrices.append(np.asarray(readings, dtype=float).reshape(GRID_ROWS, GRID_COLS))

    meta = {
        "volunteer_id": data.get("volunteer_id"),
        "sex": data.get("sex"),
        "height": data.get("height"),
        "weight": data.get("weight"),
        "position": data.get("position"),
    }
    return matrices, meta


def load_baseline(vol_dir):
    """
    读一个志愿者的「空载基线」：把 empty 文件的几帧取平均。

    为什么用平均？空载读数也有噪声，多帧平均能压掉随机抖动，
    得到一个更接近真值的「零点」。

    返回：(12, 6) 的基线图。若该志愿者没有 empty 文件，返回全 0。
    """
    empties = sorted(glob.glob(os.path.join(vol_dir, "empty*.json")))
    if not empties:
        return np.zeros((GRID_ROWS, GRID_COLS))

    matrices, _ = load_snapshots(empties[0])
    # 所有帧叠起来取平均 → 一个稳定的基线
    return np.mean(np.stack(matrices), axis=0)


# ============================================================================
# 三、核心：把整个数据集读成 (样本, 标签)
# ============================================================================

def load_dataset(data_dir=DEFAULT_DATA_DIR, frames_per_file=1, seed=0,
                 shuffle=True):
    """
    遍历全部 60 个志愿者，把四种睡姿读成「训练数据集」。

    流程（每个志愿者都一样）：
      1. 读 empty 文件，算基线（这个志愿者的「零点」）
      2. 遍历姿势文件（supine/left/right/prone 开头，排除 empty/others）
      3. 每帧读数减基线 → 这就是「洗干净」后的压力信号
      4. 摊平成 72 个数 + 归一化，配上睡姿标签

    参数：
      data_dir        —— 数据目录
      frames_per_file —— 每个文件取几帧当样本（默认 1，取第一帧）。
                         每文件通常有 10 帧，取 1 帧就是 1260 样本/类。
      seed            —— 打乱的随机种子
      shuffle         —— 是否打乱

    返回：
      X       —— (样本数, 72) 的二维数组，已减基线、已归一化
      y       —— (样本数,) 的类别编号（0仰卧/1左侧卧/2右侧卧/3俯卧）
      classes —— 姿势名列表 ["supine","left","right","prone"]
      meta    —— 每个样本的元数据列表（志愿者、身高体重等），跟 X 一一对应
    """
    # 志愿者文件夹（按数字排序，1~60）
    vol_dirs = sorted(
        [d for d in glob.glob(os.path.join(data_dir, "*")) if os.path.isdir(d)],
        key=lambda p: int(os.path.basename(p)) if os.path.basename(p).isdigit() else 999999,
    )

    X_list, y_list, meta_list = [], [], []

    for vol_dir in vol_dirs:
        baseline = load_baseline(vol_dir)   # 这个志愿者的零点

        for path in sorted(glob.glob(os.path.join(vol_dir, "*.json"))):
            filename = os.path.basename(path)
            pose = parse_pose(filename)
            if pose not in POSE_ORDER:       # 排除 empty / others
                continue

            matrices, meta = load_snapshots(path)
            # 取前 frames_per_file 帧当样本（每帧 = 一次独立测量）
            for mat in matrices[:frames_per_file]:
                signal = mat - baseline          # 减基线 = 零点校准
                X_list.append(signal.ravel())    # 12×6 摊平成 72
                y_list.append(POSE_ORDER.index(pose))
                meta_list.append(meta)

    X = np.array(X_list, dtype=float) / SCALE   # 归一化
    y = np.array(y_list, dtype=int)

    if shuffle:
        rng = np.random.default_rng(seed)
        idx = rng.permutation(len(X))
        X, y = X[idx], y[idx]
        meta_list = [meta_list[i] for i in idx]

    return X, y, list(POSE_ORDER), meta_list


def split_train_test(X, y, test_ratio=0.2, seed=0):
    """
    切成「训练集」和「考试卷」。用没见过的数据考试，分数才真实。
    """
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(X))
    n_test = int(len(X) * test_ratio)
    test_idx, train_idx = idx[:n_test], idx[n_test:]
    return X[train_idx], y[train_idx], X[test_idx], y[test_idx]


def split_by_volunteer(X, y, meta, test_ratio=0.2, seed=0):
    """
    按「志愿者」切分：一部分人训练，另一部分人考试。

    为什么需要它？——split_train_test 是随机切样本，同一个人的数据
    可能一半在训练、一半在考试，模型等于「见过这个人」，分数会虚高。
    按人切分后，考试里全是模型没见过的陌生人——这才是真实部署场景
    （新用户买回家，模型没见过他）的真正性能。

    返回：X_train, y_train, X_test, y_test
    """
    vol = np.array([int(m["volunteer_id"]) for m in meta])
    rng = np.random.default_rng(seed)
    vols = np.unique(vol)
    rng.shuffle(vols)
    n_test_v = int(len(vols) * test_ratio)
    test_v = set(vols[:n_test_v])
    te = np.array([v in test_v for v in vol])
    tr = ~te
    return X[tr], y[tr], X[te], y[te]
