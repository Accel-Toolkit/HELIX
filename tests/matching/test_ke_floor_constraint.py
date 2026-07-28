"""Tests for the one-sided ``SET_KE_OUT_MIN`` constraint."""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from linac_gen.core.lattice import Lattice
from linac_gen.elements.lattice_commands import SetKeOutMin
from linac_gen.matching.constraints import (
    _make_set_ke_out_min_evaluator,
    collect_constraints,
)


def _results(w_kin_out: float):
    return SimpleNamespace(ref_w_kin=[2.0, w_kin_out])


def test_residual_zero_when_above_floor():
    ev = _make_set_ke_out_min_evaluator(SetKeOutMin("k", 3.5, 1.0))
    assert ev(_results(4.0), None)[0] == 0.0


def test_residual_positive_when_below_floor():
    ev = _make_set_ke_out_min_evaluator(SetKeOutMin("k", 3.5, 1.0))
    res = ev(_results(2.5), None)
    assert res[0] == pytest.approx(1.0, abs=1e-9)


def test_residual_exactly_at_floor_is_zero():
    ev = _make_set_ke_out_min_evaluator(SetKeOutMin("k", 3.5, 1.0))
    assert ev(_results(3.5), None)[0] == 0.0


def test_empty_results_gives_zero_residual():
    """A lattice that produced no records yet (engine quirk) should not crash."""
    ev = _make_set_ke_out_min_evaluator(SetKeOutMin("k", 3.5, 1.0))
    res = ev(SimpleNamespace(ref_w_kin=[]), None)
    assert res[0] == 0.0


def test_weight_applied():
    lat = Lattice()
    lat.add(SetKeOutMin("k", energy_mev=3.5, weight=10.0))
    cs = collect_constraints(lat)
    assert len(cs) == 1
    res = cs[0].evaluate(_results(3.0), lat)
    # raw = 0.5 ; weighted = 5.0
    assert res[0] == pytest.approx(5.0, abs=1e-9)


def test_collect_skips_zero_weight():
    lat = Lattice()
    lat.add(SetKeOutMin("k", energy_mev=3.5, weight=0.0))
    cs = collect_constraints(lat)
    assert cs == []


def test_label_includes_energy():
    lat = Lattice()
    lat.add(SetKeOutMin("k", energy_mev=4.5, weight=1.0))
    cs = collect_constraints(lat)
    assert "4.5" in cs[0].label
