# PressureLab

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![License](https://img.shields.io/badge/License-MIT-green)
![Tests](https://img.shields.io/badge/tests-48%20passed-brightgreen)

> A no-hardware playground for flexible pressure-sensor simulation, calibration & edge deployment.
>
> 一张**离线 AI 医疗床**的软件起步包：没有真硬件，用数学「演」出柔性压力传感器的脏数据 → 用完整的标定链洗干净 → 再训练第一个睡姿分类模型。三大环节全部带测试、可复现。

<p align="center">
  <img src="assets/zero_calibration.png" width="85%" alt="零点校准前后对比">
</p>

## 📖 目录

- [🎯 这是什么](#-这是什么)
- [🚀 快速开始](#-快速开始)
- [🧪 跑测试](#-跑测试)
- [🩺 标定工具链](#-标定工具链)
- [😴 睡姿分类（第一个真模型）](#-睡姿分类第一个真模型)
- [🩹 传感器五种毛病](#-传感器五种毛病)
- [📊 示例输出](#-示例输出)
- [🗺️ 路线图](#️-路线图)
- [📁 目录结构](#-目录结构)
- [📚 致谢与引用](#-致谢与引用)
- [📄 License](#-license)

## 🎯 这是什么

这张床的核心，是床垫下的一排**柔性压力传感器**。但真正让团队翻车的，往往不是 AI 模型，而是**传感器会骗人**：它会温漂、会零点漂移、会「记仇」（迟滞）、同一批货灵敏度还不一样。不先把这些毛病校准掉，AI 学到的全是假规律。

这个仓库把「没硬件也能开工」的部分先做出来，共三大块：

| 模块 | 干什么 | 关键结果 |
|---|---|---|
| **传感器模拟器** | 高斯鼓包模拟人躺上去 + 叠加 5 种传感器毛病，输出「标准答案 vs 脏数据」 | 三种睡姿（仰卧/侧卧/俯卧）、参数集中可调 |
| **标定工具链** | 串扰校正 + 温漂补偿 + 零点校准 + 增益校准，串成完整流水线 | 系统误差消掉 **99%**；温度漂 15 度不补偿只剩 31% |
| **睡姿分类** | 纯 numpy 手写神经网络（前向+反向传播），不依赖任何深度学习框架 | 测试集 **100%**；坏传感器下 88%→标定洗回 100% |
| **边缘主控骨架** | 采集线程 → 队列 → 推理线程，含断线重连 + 看门狗 | 真硬件到位后只换 3 个函数 |

## 🚀 快速开始

```bash
# 1. 装依赖（只需要 numpy + matplotlib）
pip install -r requirements.txt

# 2. 跑传感器模拟器，生成「标准答案 + 脏数据」
python src/pressure_simulator.py

# 3. 标定全流程演示：四招连发洗脏数据（含温度漂移剧情）
python src/calibration_demo.py

# 4. 训练睡姿分类器（想调参不用改代码，命令行传参）
python src/train_pose_classifier.py
python src/train_pose_classifier.py --epochs 100 --lr 0.3 --seed 7

# 5.（可选）下载 PoPu 真实压力数据：先 2.5MB 预览版验证，再 82MB 完整版
python src/download_pupu.py preview
python src/download_pupu.py full
```

跑完的图会生成在 `outputs/` 目录里。

## 🧪 跑测试

```bash
pip install -r requirements-dev.txt
pytest          # pyproject.toml 已配置好路径，直接跑就是全部 48 个测试
```

测试覆盖：五种毛病叠加、四招标定的数学正确性（串扰往返、迟滞逆模型、增益还原）、三种睡姿的形状特征（侧卧偏一侧 / 仰卧对称）、神经网络能否学会异或、数据集可复现且不泄题。

## 🩺 标定工具链（这个项目真正的护城河）

传感器把数据「弄脏」的顺序和标定「洗回来」的顺序正好相反：

```
弄脏：真实压力 → ×灵敏度 → +温漂 → +零点漂移 → 串扰漏电 → +噪声 → 脏数据
洗回：脏数据  → 解串扰 → 温漂补偿 → 零点校准 → 增益校准 → 干净数据
```

`src/calibration.py` 每一招都是一个独立函数，可单测、可单独使用：

| 招式 | 函数 | 数学动作 |
|---|---|---|
| 串扰校正 | `correct_crosstalk()` | 迭代解线性混合 |
| 零点校准 | `correct_zero()` | 减空载基线 |
| 增益校准 | `correct_gain()` | 除每点灵敏度 |
| 温漂补偿 | `correct_temperature_offset/gain()` | 减/除随温度变化的量 |
| 迟滞补偿 | `correct_hysteresis()` | 逆模型查表 |

实测（`calibration_demo.py`，标定 25℃ / 运行 40℃）：

- 不做温度补偿：误差只缩小 **31%**
- 完整标定链：缩小 **82%**（剩余是随机噪声，交给滤波）
- 关掉噪声看纯系统误差：**99%** 被消掉

## 😴 睡姿分类（第一个真模型）

`src/mlp.py` 是**纯 numpy 手写的两层神经网络**——前向传播、反向传播、梯度下降全部摊开写，注释拉满。不装 PyTorch 也能跑通一个完整的「造数据 → 训练 → 考试」闭环：

- 数据：三种睡姿各 200 张合成压力图（位置/体重随机抖动，模拟不同的人）
- 模型：96 → 32（ReLU）→ 3（softmax），约 3300 个权重
- 结果：训练集 100%，测试集 100%（混淆矩阵见 `outputs/pose_training.png`）

**附加实验（实测，结果反直觉）**：给考试卷加上「逐点增益漂移 + 假基线 + 重度串扰」，准确率掉到 **88.3%**；用标定链洗回后恢复 **100%**。同时诚实地告诉你：分类模型对「幅度类」毛病天然鲁棒（它看形状不看绝对值），标定真正的不可替代性在**看绝对压力值的任务**——比如压疮预警：增益漂 +30%，真实 25 mmHg（安全）会被读成 32.5（危险线以上）。

## 🩹 传感器有五种「毛病」（这个项目真正的敌人）

| 毛病 | 大白话 | 怎么修 |
|---|---|---|
| 批次不一致 | 同一批货，每个点灵敏度差 15% | 逐个标定（增益校准） |
| 温漂 | 天热天冷，基线整体偏移 | 温度补偿（TCO/TCS） |
| 零点漂移 | 用久了，没人压读数也自己飘 | 离床自动归零 |
| 串扰 | 压 A 点，隔壁 B 点也读到了 | 串扰校正（迭代解混） |
| 噪声 | 信号里随机抖动 | 滤波（标定消不掉它） |

<p align="center">
  <img src="assets/sensor_imperfections.png" width="85%" alt="传感器五种毛病叠加">
</p>

## 📊 示例输出

零点校准：一个减法，误差 33.9 → 6.4（缩小 81%）。

<p align="center">
  <img src="assets/body_shape_proof.png" width="85%" alt="人形证明图">
</p>

## 🗺️ 路线图

- [x] 传感器模拟器（高斯体压 + 5 种毛病 + 三种睡姿）
- [x] 零点校准（减法）
- [x] **完整标定链**（串扰 / 温漂 / 零点 / 增益 / 迟滞，系统误差消 99%）
- [x] 边缘主控骨架（采集 → 推理 → 报警，含断线重连 + 看门狗）
- [x] **睡姿分类模型**（纯 numpy 手写神经网络，测试集 100%）
- [x] **PoPu 真实数据下载入口**（CC0 可商用）
- [x] 测试套件（48 项）+ CI
- [ ] 用 PoPu 真实数据训练（替换合成数据）
- [ ] 增益标定的「真硬件版」：多帧平均去噪 + 砝码自动标定流程
- [ ] 压疮风险量化任务（压力×时长，对标定最敏感的场景）
- [ ] PSG 预训练 → 床垫数据迁移（对标 BCGNet）

## 📁 目录结构

```
.
├── src/
│   ├── sensor_config.py          # 集中配置：所有参数 + 三种睡姿定义（单一数据源）
│   ├── pressure_simulator.py     # 传感器模拟器（体压 + 五种毛病）
│   ├── calibration.py            # ⭐ 标定工具链（五招 + 流水线）
│   ├── calibration_demo.py       # 标定全流程演示（含温度剧情）
│   ├── zero_calibration.py       # 零点校准单独演示（入门版）
│   ├── pose_dataset.py           # 睡姿合成数据集（可无限造）
│   ├── mlp.py                    # ⭐ 纯 numpy 手写神经网络
│   ├── train_pose_classifier.py  # 训练入口（支持命令行调参 + 日志）
│   ├── download_pupu.py          # PoPu 真实数据下载器
│   ├── body_shape_proof.py       # 人形证明图
│   └── main_controller.py        # 边缘主控骨架
├── tests/                        # 48 个测试
├── assets/                       # README 用的示例图
├── examples/                     # 示例输出（标准答案 + 脏数据）
├── .github/workflows/            # CI 自动跑测试
├── pyproject.toml                # 项目元数据 + pytest 配置
├── requirements.txt / requirements-dev.txt
├── CITATION.cff                  # 学术引用格式
└── LICENSE
```

## 📚 致谢与引用

这个项目站在这些公开资源的肩膀上：

- [PoPu-Data](https://github.com/rdionisio1403/PoPu) — 床垫上下两层压力阵列公开数据集（CC0，可商用）
- [BCGNet](https://github.com/FiveSeasonsMedical/BCGNet) — 「海量 PSG 预训练 → 床垫数据微调迁移」的思路来源
- [Stanford-STAGES](https://github.com/Stanford-STAGES/stanford-stages) — 自动睡眠分期的参考实现

### 引用

如果你在研究中用到了这个项目，请这样引用：

```bibtex
@misc{pressurelab,
  author = {{PressureLab Contributors}},
  title = {PressureLab: A No-Hardware Playground for Pressure-Sensor Simulation and Calibration},
  year = {2026},
  url = {https://github.com/<your-name>/<repo-name>}
}
```

## 📄 License

[MIT](./LICENSE)
