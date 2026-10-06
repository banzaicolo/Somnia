# -*- coding: utf-8 -*-
"""
测试「边缘主控骨架」脚本（src/main_controller.py）。

这个是纯标准库（threading/queue/time/random），import 无副作用。
这里只做"冒烟测试"（smoke test）：确认关键常量、异常类、以及几个
不涉及多线程的纯函数行为正确。多线程的完整行为不在这里测（太重了）。
"""
import pytest
import main_controller as mc


def test_关键常量正确():
    """传感器点数、采样率、批大小这些"合同"数字不能悄悄变。"""
    assert mc.SENSOR_POINTS == 96
    assert mc.SAMPLE_HZ == 20
    assert mc.INFER_BATCH == 4


def test_断线异常是异常类的子类():
    """SensorDisconnected 必须是个异常，才能被 try/except 接住。"""
    assert issubclass(mc.SensorDisconnected, Exception)


def test_读一帧返回96个点(monkeypatch):
    """关掉"随机断线"后，读一帧应该稳定返回 96 个点。"""
    monkeypatch.setattr(mc, "SENSOR_DROP_PROB", 0.0)
    frame = mc.read_one_frame_from_sensor()
    assert len(frame) == 96


def test_预处理算的是平均值():
    """preprocess 把一沓帧摊平后算平均，(1+2+3+4+5+6)/6 = 3.5。"""
    avg = mc.preprocess([[1, 2, 3], [4, 5, 6]])
    assert avg == 3.5


def test_模型按压力映射睡姿():
    """假模型按平均压力分档：<300 离床，300-500 仰卧，500-700 侧卧，>700 俯卧。"""
    assert mc.run_model_on_npu(100)["睡姿"] == "离床"
    assert mc.run_model_on_npu(400)["睡姿"] == "仰卧"
    assert mc.run_model_on_npu(600)["睡姿"] == "侧卧"
    assert mc.run_model_on_npu(800)["睡姿"] == "俯卧"
