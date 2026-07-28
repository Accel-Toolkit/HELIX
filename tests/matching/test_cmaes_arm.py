"""Integration tests for the CMA-ES algorithm arm in :mod:`linac_gen.matching`.

These tests are quick (<10 s total) — they use the same 2-quad lattice
that ``test_engine.py`` already exercises so the optimisation target is
known and the regression is fully scoped.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from linac_gen.core.config import BeamConfig
from linac_gen.core.lattice import Lattice
from linac_gen.elements.drift import Drift
from linac_gen.elements.lattice_commands import (
    Adjust, MinEmitGrowth, SetKeOutMin, SetSize,
)
from linac_gen.elements.quadrupole import Quadrupole
from linac_gen.matching import MATCH_ALGORITHMS, match
from linac_gen.matching.constraints import collect_constraints
from linac_gen.matching.variables import collect_variables


@pytest.fixture
def beam_cfg():
    return BeamConfig(
        species="proton", energy=3.0, frequency=352.21,
        current=0.0, duty_cycle=100.0,
        n_particles=10, distribution="waterbag", cutoff=3.0,
        emit_nx=0.25, alpha_x=0.0, beta_x=1.0,
        emit_ny=0.25, alpha_y=0.0, beta_y=1.0,
        emit_z=0.30, alpha_z=0.0, beta_z=1.0,
    )


def _build_two_quad_lattice(g0_qf=5.0, g0_qd=-5.0,
                            vmin_qf=0.0, vmax_qf=15.0,
                            vmin_qd=-15.0, vmax_qd=0.0,
                            target_x=2.0, target_y=2.0):
    """Drift–QF–drift–QD–drift with bounded ADJUSTs on each quad."""
    lat = Lattice()
    lat.add(Drift("D1", length=100.0))
    lat.add(Adjust("A1", target="QF", param_idx=2, link_group=1,
                   vmin=vmin_qf, vmax=vmax_qf, start_step=0.5))
    lat.add(Quadrupole("QF", length=80.0, gradient=g0_qf, aperture=20.0))
    lat.add(Drift("D2", length=200.0))
    lat.add(Adjust("A2", target="QD", param_idx=2, link_group=2,
                   vmin=vmin_qd, vmax=vmax_qd, start_step=0.5))
    lat.add(Quadrupole("QD", length=80.0, gradient=g0_qd, aperture=20.0))
    lat.add(Drift("D3", length=100.0))
    lat.add(SetSize("C1", k=1.0, x_mm=target_x, y_mm=target_y))
    return lat


# ---------------------------------------------------------------------------
# Algorithm registration
# ---------------------------------------------------------------------------
def test_cmaes_in_algorithm_registry():
    assert "cmaes" in MATCH_ALGORITHMS


# ---------------------------------------------------------------------------
# Convergence on a known-good 2-quad problem
# ---------------------------------------------------------------------------
def test_cmaes_converges_close_to_least_squares(beam_cfg):
    """On a problem where LS converges cleanly, CMA-ES + LS polish should
    land within a small tolerance of the LS-only result.
    """
    lat_ls = _build_two_quad_lattice()
    res_ls = match(lat_ls, beam_cfg, algorithm="least_squares",
                   max_iter=200, xtol=1e-9, ftol=1e-9)
    assert res_ls.success or res_ls.cost < 1e-6

    lat_cma = _build_two_quad_lattice()
    res_cma = match(lat_cma, beam_cfg, algorithm="cmaes",
                    max_iter=40, cmaes_sigma=0.3, refine=True,
                    xtol=1e-8, ftol=1e-8)
    assert res_cma.success or res_cma.cost < 1e-6

    # CMA-ES + LS polish should land within ~5 % of LS-only cost
    # (LS is the gold standard on this convex problem).
    assert res_cma.cost < max(1e-8, 1.05 * res_ls.cost + 1e-8)


def test_cmaes_without_refine_still_works(beam_cfg):
    """``--no-refine`` skips the polish; CMA-ES alone should still pull
    cost down at least 10× from x0."""
    # Baseline cost at x0: one LS evaluation only.
    lat0 = _build_two_quad_lattice(g0_qf=10.0, g0_qd=-10.0)
    res0 = match(lat0, beam_cfg, algorithm="least_squares", max_iter=1)
    cost0 = float(res0.cost)
    assert cost0 > 1e-3, "baseline already too good; pick a more mismatched seed"

    lat = _build_two_quad_lattice(g0_qf=10.0, g0_qd=-10.0)
    res = match(lat, beam_cfg, algorithm="cmaes",
                max_iter=40, cmaes_sigma=0.3, refine=False)
    assert res.cost < 0.1 * cost0


# ---------------------------------------------------------------------------
# Bound enforcement
# ---------------------------------------------------------------------------
def test_cmaes_rejects_infinite_bounds(beam_cfg):
    """Open-ended ADJUST cards should raise — CMA-ES needs a finite box."""
    lat = Lattice()
    lat.add(Drift("D1", length=100.0))
    lat.add(Adjust("A1", target="QF", param_idx=2,
                   vmin=0.0, vmax=0.0))   # zero-zero = infinite per parser
    lat.add(Quadrupole("QF", length=80.0, gradient=5.0))
    lat.add(SetSize("C1", k=1.0, x_mm=2.0))
    with pytest.raises(ValueError, match="finite bounds"):
        match(lat, beam_cfg, algorithm="cmaes", max_iter=5)


def test_cmaes_respects_bounds(beam_cfg):
    """Final variable values must lie inside the declared [vmin, vmax]."""
    lat = _build_two_quad_lattice(g0_qf=5.0, g0_qd=-5.0,
                                  vmin_qf=2.0, vmax_qf=8.0,
                                  vmin_qd=-8.0, vmax_qd=-2.0)
    res = match(lat, beam_cfg, algorithm="cmaes",
                max_iter=30, cmaes_sigma=0.3)
    for var, x in zip(res.variables, res.x_final):
        assert var.vmin - 1e-9 <= x <= var.vmax + 1e-9, (
            f"{var.label}: {x} outside [{var.vmin}, {var.vmax}]"
        )


# ---------------------------------------------------------------------------
# New constraint types end-to-end
# ---------------------------------------------------------------------------
def test_cmaes_with_emit_growth_and_ke_floor(beam_cfg):
    """End-to-end: lattice carrying MIN_EMIT_GROWTH + SET_KE_OUT_MIN
    should converge under CMA-ES without raising."""
    lat = Lattice()
    lat.add(Drift("D1", length=100.0))
    lat.add(Adjust("A1", target="QF", param_idx=2,
                   vmin=0.0, vmax=15.0, start_step=0.5))
    lat.add(Quadrupole("QF", length=80.0, gradient=5.0, aperture=20.0))
    lat.add(Drift("D2", length=200.0))
    lat.add(Adjust("A2", target="QD", param_idx=2,
                   vmin=-15.0, vmax=0.0, start_step=0.5))
    lat.add(Quadrupole("QD", length=80.0, gradient=-5.0, aperture=20.0))
    lat.add(Drift("D3", length=100.0))
    lat.add(MinEmitGrowth("e_x", plane="X", weight=1.0))
    lat.add(MinEmitGrowth("e_y", plane="Y", weight=1.0))
    lat.add(MinEmitGrowth("e_z", plane="Z", weight=1.0))
    lat.add(SetKeOutMin("ke_floor", energy_mev=2.5, weight=10.0))

    res = match(lat, beam_cfg, algorithm="cmaes",
                max_iter=20, cmaes_sigma=0.3, refine=True)
    # No exception; report contains all four constraints
    labels = {c.label for c in res.constraints}
    assert {"MIN_EMIT_GROWTH:X", "MIN_EMIT_GROWTH:Y",
            "MIN_EMIT_GROWTH:Z"} <= labels
    assert any("SET_KE_OUT_MIN" in l for l in labels)
    # No bound violations
    for var, x in zip(res.variables, res.x_final):
        assert var.vmin - 1e-9 <= x <= var.vmax + 1e-9
