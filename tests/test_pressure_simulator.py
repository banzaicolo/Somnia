# -*- coding: utf-8 -*-
"""
Tests for the "simulated pressure sensor" script (src/pressure_simulator.py).

Three things are verified:
1. The ideal body-pressure map is a "placed human shape", not random numbers (correct shape, hips heaviest)
2. After adding the 5 sensor faults, the readings really are "contaminated" (differ from ideal, unoccupied areas lifted)
3. Hysteresis: at the same pressure, loading reading > unloading reading
"""
import numpy as np
import pressure_simulator as ps


def test_ideal_body_shape_correct():
    """The ideal map should be 8 rows × 12 columns (=96 points), and pressure should never be negative."""
    ideal = ps.make_ideal_pressure(ps.GRID_W, ps.GRID_H, ps.BODY_PARTS)
    assert ideal.shape == (ps.GRID_H, ps.GRID_W)
    assert ideal.min() >= 0


def test_hips_are_heaviest():
    """The peak should appear near the hips (rows 5~6, i.e. indices 4~5)."""
    ideal = ps.make_ideal_pressure(ps.GRID_W, ps.GRID_H, ps.BODY_PARTS)
    row, _col = np.unravel_index(np.argmax(ideal), ideal.shape)
    assert 4 <= row <= 5


def test_faults_make_reading_differ_from_ideal():
    """Sensor faults will definitely make the final reading deviate from the ideal map."""
    ideal = ps.make_ideal_pressure(ps.GRID_W, ps.GRID_H, ps.BODY_PARTS)
    steps = ps.add_imperfections(ideal)
    final = list(steps.values())[-1]
    assert not np.allclose(final, ideal)


def test_unoccupied_corner_is_lifted():
    """The top-left corner has nobody pressing; the ideal value is near 0, and after faults it should be clearly above 0 (that's the "lying")."""
    ideal = ps.make_ideal_pressure(ps.GRID_W, ps.GRID_H, ps.BODY_PARTS)
    steps = ps.add_imperfections(ideal)
    final = list(steps.values())[-1]
    assert ideal[0, 0] < 5          # nobody presses the corner; ideal value is small (only the tail residue of the Gaussian bumps)
    assert final[0, 0] > ideal[0, 0]  # lifted after faults are added


def test_hysteresis_loading_higher_than_unloading():
    """Hysteresis loop: at the same pressure, the loading reading is strictly higher than the unloading reading (constant width equal to HYSTERESIS)."""
    p, load, unload = ps.hysteresis_curve()
    assert p.shape == load.shape == unload.shape
    assert (load >= unload).all()            # loading never below unloading
    assert (load - unload > 0).all()         # always a positive width (constant offset model)


def test_exactly_four_body_parts():
    """The body is made of four parts: head, shoulders, hips, heels."""
    assert len(ps.BODY_PARTS) == 4
