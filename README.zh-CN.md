# Somnia

[English](README.md)

[![CI](https://github.com/banzaicolo/Somnia/actions/workflows/ci.yml/badge.svg)](https://github.com/banzaicolo/Somnia/actions)
![License](https://img.shields.io/badge/license-MIT-blue)

一张床垫，读懂睡眠。不穿戴，不摄像，不上云。

这是柔性压力传感器床垫的研究项目。先用数学模拟出脏的传感器数据，再用标定流程洗干净，最后在上面做出第一批应用：睡姿识别、离床监测、呼吸与心跳、睡眠分期。睡姿分类器还用了 PoPu 数据集里 60 位真人的数据训练，对没见过的陌生人准确率 93.9%。

![五种模拟的传感器毛病](assets/sensor_imperfections.png)

## 快速开始

```bash
pip install -r requirements.txt

python src/pressure_simulator.py
python src/calibration_demo.py
python src/train_pose_classifier.py
python src/bed_monitor_demo.py
python src/bcg_demo.py
python src/sleep_staging_demo.py
python src/sleep_pipeline_demo.py
python src/download_pupu.py full
python src/pupu_demo.py
python src/train_real_pose.py
```

图片输出到 `outputs/`。

## 模块

| 文件 | 作用 |
|---|---|
| `pressure_simulator.py` | 模拟压力图，带上五种传感器毛病 |
| `calibration.py` | 串扰、温漂、零点、增益、迟滞 |
| `train_pose_classifier.py` | 睡姿分类器，纯 NumPy 写的小网络 |
| `bed_monitor.py` | 离床监测，「超时未归」报警 |
| `bcg_monitor.py` | 从心冲击图里提取呼吸和心跳 |
| `sleep_staging.py` | 睡眠分期，四分类（清醒 / 浅睡 / 深睡 / REM） |
| `sleep_pipeline.py` | 整条链路，从原始波形到睡眠分期 |
| `pupu_loader.py` | 读取 PoPu 数据并去掉零点偏移 |
| `train_real_pose.py` | 用 60 位真人训练睡姿 |
| `main_controller.py` | 边缘主控的多线程骨架 |

![完整标定链](assets/calibration_full.png)

## 数据

大部分演示跑在合成信号上，这类信号有已知的标准答案，用来证明算法逻辑正确，但不代表真实性能。

睡姿分类器还用了 [PoPu-Data](https://github.com/rdionisio1403/PoPu)（CC0 许可）的真实压力图训练，60 位真人。有两种评估方式：

| 评估方式 | 准确率 | 含义 |
|---|---|---|
| 随机切分 | 98.3% | 模型部分见过本人 |
| 按人切分 | 93.9% | 考试的人从未在训练里出现——这才是关键的数字 |

![真实数据训练的睡姿分类器](assets/pupu_training.png)

下载数据集：

```bash
python src/download_pupu.py preview
python src/download_pupu.py full
```

![端到端链路](assets/sleep_pipeline.png)

## 测试

```bash
pip install -r requirements-dev.txt
pytest
```

## 免责声明

这是研究用的软件，不是医疗器械。报警逻辑跑在合成数据上，没做过临床验证，公开数据集上的准确率也不代表临床表现。请勿用于诊断或监护。

## 数据与隐私

默认离线运行，数据不出房间。真实部署时，床垫压力信号属于敏感健康数据，运营方要遵守适用的法规（《个人信息保护法》、GDPR 等）。本仓库不含任何可识别的受试者数据，PoPu 数据集也未在此二次分发。

## 许可证

[MIT](LICENSE)

## 引用

见 [CITATION.cff](CITATION.cff)。
