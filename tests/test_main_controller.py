# -*- coding: utf-8 -*-
"""
Tests for the edge main-controller skeleton script (src/main_controller.py).

This is pure standard library (threading/queue/time/random); importing has no
side effects. Here we only do a smoke test: confirm the key constants, the
exception class, and the behavior of a few pure functions that don't involve
multi-threading. The full multi-threaded behavior is not tested here (too heavy).
"""
import pytest
import main_controller as mc


def test_key_constants_correct():
    """Sensor point count, sample rate, and batch size are "contract" numbers that must not silently change."""
    assert mc.SENSOR_POINTS == 96
    assert mc.SAMPLE_HZ == 20
    assert mc.INFER_BATCH == 4


def test_disconnect_exception_is_exception_subclass():
    """SensorDisconnected must be an exception so try/except can catch it."""
    assert issubclass(mc.SensorDisconnected, Exception)


def test_read_one_frame_returns_96_points(monkeypatch):
    """After turning off "random disconnects", reading one frame should reliably return 96 points."""
    monkeypatch.setattr(mc, "SENSOR_DROP_PROB", 0.0)
    frame = mc.read_one_frame_from_sensor()
    assert len(frame) == 96


def test_preprocess_computes_average():
    """preprocess flattens the frames and averages them: (1+2+3+4+5+6)/6 = 3.5."""
    avg = mc.preprocess([[1, 2, 3], [4, 5, 6]])
    assert avg == 3.5


def test_model_maps_pressure_to_posture():
    """The fake model buckets by average pressure: <300 off_bed, 300-500 supine, 500-700 side, >700 prone."""
    assert mc.run_model_on_npu(100)["posture"] == "off_bed"
    assert mc.run_model_on_npu(400)["posture"] == "supine"
    assert mc.run_model_on_npu(600)["posture"] == "side"
    assert mc.run_model_on_npu(800)["posture"] == "prone"
