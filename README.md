# Somnia

> [中文版](README.zh-CN.md)

> ⚠️ **Research & educational software only.**
> This project is an algorithm-research playground, **not a medical device**, and must not be
> used for clinical diagnosis, monitoring, or any medical decision-making. All alarm logic
> (bed-exit, apnea, pressure-ulcer risk) runs on **synthetic data** and has **not been
> clinically validated**. Accuracy figures on public datasets do not represent clinical performance.

[![CI](https://github.com/banzaicolo/Somnia/actions/workflows/ci.yml/badge.svg)](https://github.com/banzaicolo/Somnia/actions)
![License](https://img.shields.io/badge/license-MIT-blue)

**One mattress under the sleeper. No wearables, no cameras, no cloud.**

A **no-hardware playground** for a flexible-pressure-sensor bed platform: simulate dirty sensor data with math → clean it with a full calibration pipeline → build the first applications (posture, bed-exit, vitals, sleep staging) → and finally **train on real open data**. The posture classifier reaches **93.9% accuracy on volunteers it has never seen** (PoPu dataset, 60 people). Everything is fully tested.

![Five sensor imperfections](assets/sensor_imperfections.png)

*The five sensor defects the simulator generates — everything downstream must survive them.*

## What's inside

| Module | Entry point | What it does |
|---|---|---|
| Sensor simulator | `pressure_simulator.py` | Gaussian body-pressure maps + 5 sensor defects |
| Calibration pipeline | `calibration.py` | Crosstalk, temperature drift, zero, gain, hysteresis |
| Posture classifier | `train_pose_classifier.py` | Hand-written NumPy MLP (supine / side / prone) |
| Bed-exit monitor | `bed_monitor.py` | Total-pressure state machine + "away too long" alarm |
| BCG vitals | `bcg_monitor.py` | Breathing/heart rate via FFT + apnea detection |
| Sleep staging | `sleep_staging_demo.py` | 4-stage classifier (wake / light / deep / REM) |
| End-to-end pipeline | `sleep_pipeline_demo.py` | Raw BCG waveform → 5 features → sleep staging |
| Real-data loader | `pupu_loader.py` | PoPu JSON → calibrated pressure maps (zero offset removed) |
| Real-data training | `train_real_pose.py` | 4-pose classification on 60 real volunteers: **93.9% on unseen people** |
| Edge controller | `main_controller.py` | Threaded skeleton (capture → infer → alarm) |

![Full calibration pipeline](assets/calibration_full.png)

*Full calibration chain: crosstalk → temperature drift → zero offset → gain → hysteresis.*

## Quick start

```bash
pip install -r requirements.txt

python src/pressure_simulator.py       # simulate dirty sensor data
python src/calibration_demo.py         # full calibration pipeline
python src/train_pose_classifier.py    # train the posture classifier
python src/bed_monitor_demo.py         # bed-exit monitoring
python src/bcg_demo.py                 # breathing / heart rate + apnea
python src/sleep_staging_demo.py       # sleep staging (hypnogram)
python src/sleep_pipeline_demo.py      # full pipeline: waveform → features → staging
python src/download_pupu.py full       # fetch real data (82 MB, once)
python src/pupu_demo.py                # visualize real pressure maps
python src/train_real_pose.py          # train on real data: 93.9% on strangers
```

Plots are written to `outputs/`.

## Tests

```bash
pip install -r requirements-dev.txt
pytest     # 130 tests
```

## Data: synthetic vs. real

Most demos run on **synthetic signals** — mathematically generated "ground truth" used to prove the algorithms are logically correct. They do **not** measure real-world performance.

The **posture classifier additionally trains on real data** ([PoPu-Data](https://github.com/rdionisio1403/PoPu), CC0: 60 volunteers, ~50k real mattress pressure frames) and reports two honest scores:

| Evaluation | Accuracy | Meaning |
|---|---|---|
| Random split | 98.3% | Model has partly "seen" the person — inflated |
| **Volunteer split** | **93.9%** | Test people were never seen in training — **the honest deployment number** |

(4 classes: supine / left side / right side / prone; Cohen's κ = 0.919 on unseen volunteers.)

![Posture classifier trained on 60 real volunteers](assets/pupu_training.png)

*Left: training loss. Right: confusion matrix on 12 volunteers never seen during training.*

PoPu-Data is downloaded from its official repository:
**github.com/rdionisio1403/PoPu** (CC0 license — free to use and modify).

Fetch it (the downloader is already in this repo):

```bash
python src/download_pupu.py preview   # 2.6 MB — verify the pipeline first
python src/download_pupu.py full      # 82 MB — real training data
```

> Honest note: there is **no public dataset that pairs mattress pressure with
> sleep-stage labels**. Sleep-EDF is EEG, so it cannot train the mattress model
> directly. That paired dataset is the moat this project must build itself
> (PSG pretrain → mattress fine-tune, BCGNet-style).

![End-to-end pipeline: raw waveform → features → sleep staging](assets/sleep_pipeline.png)

*The full chain: raw BCG waveform → 5 features → sleep stages.*

## Roadmap

- [x] Sensor simulator + 5 defect models
- [x] Full calibration pipeline (crosstalk / temp / zero / gain / hysteresis)
- [x] Posture classifier (NumPy-only MLP)
- [x] Bed-exit monitoring
- [x] BCG vitals (breathing & heart rate, apnea detection)
- [x] Sleep staging
- [x] End-to-end pipeline (raw waveform → features → staging)
- [x] Train on real PoPu pressure data (**93.9%** on unseen volunteers, 4 poses)
- [ ] Hardware-in-the-loop calibration
- [ ] PSG pretrain → mattress fine-tune (BCGNet-style)

## Data & Privacy

This project is designed **offline by default** — no cloud, no data leaves the room.

In real deployments, mattress pressure signals are **sensitive personal health data**. Operators
must comply with applicable regulations (China PIPL, EU GDPR, etc.). This repository contains
**no identifiable subject data**. The PoPu-Data dataset used for training is CC0 (public domain)
and is **not redistributed here** — use `src/download_pupu.py` to fetch it from the official source.

## License

[MIT](LICENSE)

## Citation

See [CITATION.cff](CITATION.cff).
