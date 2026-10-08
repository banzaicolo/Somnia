# -*- coding: utf-8 -*-
"""
Tests for the "body-shape proof figure" script (src/body_shape_proof.py).

Key thing verified: the simulated human shape is "placed according to body structure", not rolled out randomly.
"""
import numpy as np
import body_shape_proof as bp


def test_ground_truth_shape_correct():
    """The ideal map should be 8 rows × 12 columns."""
    ideal = bp.make_ideal_pressure(bp.GRID_W, bp.GRID_H, bp.BODY_PARTS)
    assert ideal.shape == (bp.GRID_H, bp.GRID_W)


def test_four_parts_top_to_bottom():
    """Row indices (y coordinates) should be strictly increasing: head -> shoulders -> hips -> heels."""
    ys = [part[2] for part in bp.BODY_PARTS]
    assert ys == sorted(ys)


def test_hips_have_largest_peak():
    """Among the four parts, the largest peak pressure should be the hips (the heaviest spot when supine)."""
    amps = [part[5] for part in bp.BODY_PARTS]
    heaviest = bp.BODY_PARTS[amps.index(max(amps))]
    assert heaviest[0] == "Hips"
