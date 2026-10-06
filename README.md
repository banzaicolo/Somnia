# PressureLab

![Tests](https://img.shields.io/badge/tests-112%20passed-brightgreen)
![License](https://img.shields.io/badge/license-MIT-blue)

A **no-hardware playground** for a flexible-pressure-sensor medical bed.

Simulate dirty sensor data with math → clean it with a full calibration pipeline → build the first AI applications (posture, bed-exit, vitals, sleep staging). Everything runs on synthetic data and is fully tested — no real sensors required.

## What's inside

| Module | Entry point | What it does |
|---|---|---|
| Sensor simulator | `pressure_simulator.py` | Gaussian body-pressure maps + 5 sensor defects |
| Calibration pipeline | `calibration.py` | Crosstalk, temperature drift, zero, gain, hysteresis |
| Posture classifier | `train_pose_classifier.py` | Hand-written NumPy MLP (supine / side / prone) |
| Bed-exit monitor | `bed_monitor.py` | Total-pressure state machine + "away too long" alarm |
| BCG vitals | `bcg_monitor.py` | Breathing/heart rate via FFT + apnea detection |
| Sleep staging | `sleep_staging_demo.py` | 4-stage classifier (wake / light / deep / REM) |
| Edge controller | `main_controller.py` | Threaded skeleton (capture → infer → alarm) |

## Quick start

```bash
pip install -r requirements.txt

python src/pressure_simulator.py       # simulate dirty sensor data
python src/calibration_demo.py         # full calibration pipeline
python src/train_pose_classifier.py    # train the posture classifier
python src/bed_monitor_demo.py         # bed-exit monitoring
python src/bcg_demo.py                 # breathing / heart rate + apnea
python src/sleep_staging_demo.py       # sleep staging (hypnogram)
```

Plots are written to `outputs/`.

## Tests

```bash
pip install -r requirements-dev.txt
pytest     # 112 tests
```

## Roadmap

- [x] Sensor simulator + 5 defect models
- [x] Full calibration pipeline (crosstalk / temp / zero / gain / hysteresis)
- [x] Posture classifier (NumPy-only MLP)
- [x] Bed-exit monitoring
- [x] BCG vitals (breathing & heart rate, apnea detection)
- [x] Sleep staging
- [ ] Train on real PoPu pressure data
- [ ] Hardware-in-the-loop calibration
- [ ] PSG pretrain → mattress fine-tune (BCGNet-style)

## License

[MIT](LICENSE)

## Citation

See [CITATION.cff](CITATION.cff).
