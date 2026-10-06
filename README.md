# AI 医疗床 · 传感器标定起步包

> 一张**离线 AI 医疗床**的软件起步包：没有真硬件，先用数学「演」出柔性压力传感器的脏数据，再演示怎么把它校准干净。

<p align="center">
  <img src="assets/zero_calibration.png" width="85%" alt="零点校准前后对比">
</p>

## 一句话说明

这张床的核心，是床垫下的一排**柔性压力传感器**。但真正让团队翻车的，往往不是 AI 模型，而是**传感器会骗人**：它会温漂、会零点漂移、会「记仇」（迟滞）、同一批货灵敏度还不一样。不先把这些毛病校准掉，AI 学到的全是假规律。

这个仓库把「没硬件也能开工」的部分先做出来，共四样东西：

| 脚本 | 干什么 |
|---|---|
| `src/pressure_simulator.py` | 用高斯鼓包模拟人躺上去的体压图，再叠加 5 种传感器毛病，输出「标准答案 vs 脏数据」 |
| `src/zero_calibration.py` | 演示第一招校准——零点归零（一个减法），把平均误差砍掉 80% 以上 |
| `src/body_shape_proof.py` | 证明模拟出来的人形不是随机的，并标注头 / 肩 / 臀 / 脚跟位置 |
| `src/main_controller.py` | 边缘主控程序骨架：采集线程 → 队列 → 推理线程，含断线重连 + 看门狗 |

## 快速开始

```bash
# 1. 装依赖（只需要 numpy + matplotlib）
pip install -r requirements.txt

# 2. 跑传感器模拟器，生成「标准答案 + 脏数据」
python src/pressure_simulator.py

# 3. 跑零点校准演示，看脏数据被修干净
python src/zero_calibration.py

# 4. 看人形证明图
python src/body_shape_proof.py
```

跑完的图会生成在 `outputs/` 目录里。

## 传感器有五种「毛病」（这个项目真正的敌人）

| 毛病 | 大白话 | 怎么修 |
|---|---|---|
| 批次不一致 | 同一批货，每个点灵敏度差 15% | 逐个标定（增益校准） |
| 温漂 | 天热天冷，基线整体偏移 | 温度补偿 |
| 零点漂移 | 用久了，没人压读数也自己飘 | 离床自动归零 |
| 串扰 | 压 A 点，隔壁 B 点也读到了 | 串扰校正 |
| 噪声 | 信号里随机抖动 | 滤波 |

<p align="center">
  <img src="assets/sensor_imperfections.png" width="85%" alt="传感器五种毛病叠加">
</p>

## 目录结构

```
.
├── src/                      # 所有代码
│   ├── main_controller.py    # 边缘主控骨架
│   ├── pressure_simulator.py # 传感器模拟器
│   ├── zero_calibration.py   # 零点校准演示
│   └── body_shape_proof.py   # 人形证明图
├── assets/                   # README 用的示例图
├── examples/                 # 示例输出（标准答案 + 脏数据）
├── requirements.txt
└── LICENSE
```

## 背景：为什么从「传感器标定」开始

这个项目的完整目标是做一张离线 AI 医疗床：柔性压力传感器阵列 + 多模态感知 + 端侧离线推理，覆盖睡姿识别、跌倒/离床预警、压疮预防等场景。

但第一步不碰 AI 模型，先做传感器标定——因为**模型错了可以重训，传感器漂了整台设备就废了**。标定是这个项目真正的壁垒，也是很多团队栽跟头的地方。

## License

[MIT](./LICENSE)
