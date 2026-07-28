# tests/matching/test_set_position_unstub.py
"""SET_POSITION is a real MP-only centroid constraint (was a stub).

Residual = [x−x_mm, x′−xp_mrad, y−y_mm, y′−yp_mrad] at the card's own
recorder row (LatticeCommands get recorder rows).  Envelope results
carry no centroid → four zeros, flagged inert by the engine audit.
"""
from types import SimpleNamespace

import numpy as np
import pytest

from linac_gen.core.lattice import Lattice
from linac_gen.elements.drift import Drift
from linac_gen.elements.lattice_commands import SetPosition
from linac_gen.matching.constraints import collect_constraints


def _lat():
    lat = Lattice()
    lat.add(Drift("D1", 100.0))
    lat.add(SetPosition("SP", k=2.0, x_mm=0.5, xp_mrad=0.1,
                        y_mm=-0.5, yp_mrad=-0.1))
    lat.add(Drift("D2", 100.0))
    return lat


def _mp_results(lat, vec):
    n = len(lat.elements)
    cent = [np.zeros(6)] + [np.asarray(vec, dtype=float)] * n
    return SimpleNamespace(centroid=cent,
                           element_exit_idx=list(range(1, n + 1)))


def test_four_residuals_at_card_row():
    lat = _lat()
    (c,) = collect_constraints(lat)
    assert c.label == "SET_POSITION" and c.requires_mp
    res = c.evaluator(_mp_results(lat, [1.0, 0.3, 0.0, 0.2, 0, 0]), lat)
    assert res == pytest.approx([0.5, 0.2, 0.5, 0.3])


def test_weight_is_k():
    (c,) = collect_constraints(_lat())
    assert c.weight == pytest.approx(2.0)
    res = c.evaluate(_mp_results(_lat(), [1.0, 0.1, -0.5, -0.1, 0, 0]),
                     _lat())
    # weight multiplies the raw residual exactly once
    assert res == pytest.approx([2 * 0.5, 0.0, 0.0, 0.0])


def test_envelope_inert_four_zeros():
    (c,) = collect_constraints(_lat())
    res = c.evaluate(SimpleNamespace(sigma_x=[1.0]), _lat())
    assert res.shape == (4,)
    assert np.allclose(res, 0.0)
