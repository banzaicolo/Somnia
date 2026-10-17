.DEFAULT_GOAL := help
PY ?= python3

.PHONY: help install test demo real validate clean

help:
	@echo "make install   install runtime + dev dependencies"
	@echo "make test      run the test suite"
	@echo "make demo      run every synthetic demo (writes outputs/)"
	@echo "make real      visualize and train on PoPu real data"
	@echo "make validate  what CI runs: test + whitespace check"
	@echo "make clean     delete generated outputs"

install:
	$(PY) -m pip install -r requirements-dev.txt

test:
	$(PY) -m pytest -q

demo:
	$(PY) src/pressure_simulator.py
	$(PY) src/calibration_demo.py
	$(PY) src/train_pose_classifier.py
	$(PY) src/bed_monitor_demo.py
	$(PY) src/bcg_demo.py
	$(PY) src/sleep_staging_demo.py
	$(PY) src/sleep_pipeline_demo.py

real:
	$(PY) src/pupu_demo.py
	$(PY) src/train_real_pose.py

validate: test
	git diff --check

clean:
	rm -f outputs/*.png outputs/*.csv
