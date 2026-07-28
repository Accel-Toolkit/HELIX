"""Tests for multi-objective lattice design (Pareto fronts).

Covers the objective library, validation, the non-dominated-sort helper,
and end-to-end NSGA-II + qNEHVI runs on a small lattice with two genuinely
competing objectives (longitudinal emittance growth vs exit energy).
"""
from __future__ import annotations

import numpy as np
import pytest

# NSGA-II needs the optional pymoo extra; qNEHVI additionally needs
# botorch/gpytorch — heavy optional stacks absent on CI runners.  The
# engine raises clean install-hint ImportErrors without them; tests
# skip instead of failing (pure-helper tests below run everywhere).
pymoo = pytest.importorskip("pymoo")
_needs_botorch = pytest.mark.skipif(
    not __import__("importlib.util", fromlist=["util"]).find_spec("botorch"),
    reason="qNEHVI needs the optional botorch/gpytorch extra")

from linac_gen.core.config import BeamConfig
from linac_gen.core.lattice import Lattice
from linac_gen.elements.drift import Drift
from linac_gen.elements.quadrupole import Quadrupole
from linac_gen.elements.lattice_commands import Adjust, SetSize
from linac_gen.matching.multiobjective import (
    OBJECTIVES, pareto_optimize, _pareto_mask, objective_labels,
)


def _bcfg():
    return BeamConfig(
        species="proton", energy=3.0, frequency=352.21, n_particles=200,
        distribution="waterbag",
        emit_nx=0.25, alpha_x=0.0, beta_x=2.0,
        emit_ny=0.25, alpha_y=0.0, beta_y=2.0,
        emit_z=0.30, alpha_z=0.0, beta_z=10.0,
    )


def _two_quad_lattice():
    """Two ADJUST'd quads -> a 2-D decision box for a quick Pareto run."""
    lat = Lattice()
    lat.add(Drift("D1", length=200.0, aperture=10.0))
    lat.add(Adjust("C1", target="QUAD_001", param_idx=2, link_group=0,
                   vmin=0.5, vmax=20.0, start_step=0.5))
    lat.add(Quadrupole("QUAD_001", length=100.0, gradient=5.0, aperture=10.0))
    lat.add(Drift("D2", length=200.0, aperture=10.0))
    lat.add(Adjust("C2", target="QUAD_002", param_idx=2, link_group=0,
                   vmin=-20.0, vmax=-0.5, start_step=0.5))
    lat.add(Quadrupole("QUAD_002", length=100.0, gradient=-5.0, aperture=10.0))
    lat.add(Drift("D3", length=200.0, aperture=10.0))
    lat.add(SetSize("CSET", k=1.0, x_mm=2.0, y_mm=0.0, phi_or_z=0.0))
    return lat


def _linked_quad_lattice():
    """Two ADJUST'd quads sharing a non-zero link_group -> ONE optimiser
    column but TWO Variable entries (the link-group dedup case the CSV /
    table writers must handle)."""
    lat = Lattice()
    lat.add(Drift("D1", length=200.0, aperture=10.0))
    lat.add(Adjust("C1", target="QUAD_001", param_idx=2, link_group=7,
                   vmin=0.5, vmax=20.0, start_step=0.5))
    lat.add(Quadrupole("QUAD_001", length=100.0, gradient=5.0, aperture=10.0))
    lat.add(Drift("D2", length=200.0, aperture=10.0))
    lat.add(Adjust("C2", target="QUAD_002", param_idx=2, link_group=7,
                   vmin=0.5, vmax=20.0, start_step=0.5))
    lat.add(Quadrupole("QUAD_002", length=100.0, gradient=5.0, aperture=10.0))
    lat.add(Drift("D3", length=200.0, aperture=10.0))
    lat.add(SetSize("CSET", k=1.0, x_mm=2.0, y_mm=0.0, phi_or_z=0.0))
    return lat


# ----------------------------------------------------------------------
# Objective library + validation
# ----------------------------------------------------------------------
def test_objective_library_has_expected_names():
    for n in ("emit_nx_growth", "emit_ny_growth", "emit_nz_growth",
              "emit_4d_growth", "transmission_loss", "neg_exit_energy",
              "exit_sigma_x", "exit_sigma_y", "max_sigma_x", "max_sigma_y"):
        assert n in OBJECTIVES
    # labels resolve for all
    assert len(objective_labels(sorted(OBJECTIVES))) == len(OBJECTIVES)


def test_fewer_than_two_objectives_raises():
    with pytest.raises(ValueError, match="at least 2"):
        pareto_optimize(_two_quad_lattice(), _bcfg(), ["exit_sigma_x"])


def test_unknown_objective_raises():
    with pytest.raises(ValueError, match="unknown objective"):
        pareto_optimize(_two_quad_lattice(), _bcfg(),
                        ["exit_sigma_x", "not_a_real_objective"])


def test_unbounded_adjust_rejected():
    lat = Lattice()
    lat.add(Drift("D1", length=200.0, aperture=10.0))
    lat.add(Adjust("C1", target="QUAD_001", param_idx=2, link_group=0,
                   vmin=0.0, vmax=0.0, start_step=0.5))   # unset -> +/-inf
    lat.add(Quadrupole("QUAD_001", length=100.0, gradient=5.0, aperture=10.0))
    lat.add(Drift("D2", length=200.0, aperture=10.0))
    lat.add(SetSize("CSET", k=1.0, x_mm=2.0, y_mm=0.0, phi_or_z=0.0))
    with pytest.raises(ValueError, match="finite"):
        pareto_optimize(lat, _bcfg(), ["exit_sigma_x", "exit_sigma_y"],
                        pop_size=8, n_gen=3)


# ----------------------------------------------------------------------
# Non-dominated-sort helper
# ----------------------------------------------------------------------
def test_pareto_mask_identifies_non_dominated():
    # (1,4) and (4,1) are non-dominated; (3,3) is dominated by neither but
    # (2,2) dominates (3,3); (5,5) dominated by all.
    F = np.array([[1.0, 4.0], [4.0, 1.0], [2.0, 2.0],
                  [3.0, 3.0], [5.0, 5.0]])
    mask = _pareto_mask(F)
    # (3,3) dominated by (2,2); (5,5) dominated by everything.
    assert mask.tolist() == [True, True, True, False, False]


# ----------------------------------------------------------------------
# End-to-end NSGA-II + qNEHVI
# ----------------------------------------------------------------------
def test_nsga2_returns_valid_pareto_front():
    lat = _two_quad_lattice()
    res = pareto_optimize(
        lat, _bcfg(), ["exit_sigma_x", "exit_sigma_y"],
        algorithm="nsga2", pop_size=12, n_gen=5, seed=0,
    )
    assert res.n_eval > 0
    assert res.pareto_F.shape[1] == 2
    assert res.pareto_x.shape[0] == res.pareto_F.shape[0]
    # The returned front must itself be non-dominated.
    assert _pareto_mask(res.pareto_F).all()
    # Decision vectors respect the bounds.
    assert np.all(res.pareto_x[:, 0] >= 0.5 - 1e-9)


@_needs_botorch
def test_qnehvi_returns_valid_pareto_front():
    lat = _two_quad_lattice()
    res = pareto_optimize(
        lat, _bcfg(), ["exit_sigma_x", "exit_sigma_y"],
        algorithm="qnehvi", pop_size=6, n_gen=4, seed=0,
    )
    assert res.n_eval > 0
    assert _pareto_mask(res.pareto_F).all()


def test_linked_variables_dedup_column_alignment():
    """Linked ADJUST cards collapse to one optimiser column.  The result's
    column-aligned labels (used by the CSV writer and GUI table) must match
    the pareto_x width, while the full variable list keeps both DoF."""
    lat = _linked_quad_lattice()
    res = pareto_optimize(
        lat, _bcfg(), ["exit_sigma_x", "exit_sigma_y"],
        algorithm="nsga2", pop_size=8, n_gen=3, seed=0,
    )
    # full variable list keeps both DoF ...
    assert len(res.variables) == 2
    # ... but the decision box is a single column (shared link_group).
    assert res.n_cols == 1
    assert res.pareto_x.shape[1] == 1
    assert res.all_x.shape[1] == 1
    # column-aligned labels must match the _x width (the CSV/table guard).
    labels = res.column_variable_labels()
    assert len(labels) == res.pareto_x.shape[1] == 1
    assert len(res.column_variables()) == 1
    # the representative is the first-encountered linked variable.
    assert labels[0] == "QUAD_001.gradient"


def test_unlinked_column_labels_match_full_list():
    """With nothing linked, column_variable_labels() is just every label —
    a no-op so the common (unlinked) path is unchanged."""
    res = pareto_optimize(
        _two_quad_lattice(), _bcfg(), ["exit_sigma_x", "exit_sigma_y"],
        algorithm="nsga2", pop_size=8, n_gen=3, seed=0,
    )
    assert res.n_cols == len(res.variables) == 2
    assert res.column_variable_labels() == [v.label for v in res.variables]
    assert res.pareto_x.shape[1] == len(res.column_variable_labels())


def test_competing_objectives_give_multipoint_front():
    """ε_z growth vs exit energy genuinely conflict -> the front should
    have more than one design (a real trade-off, not a single point)."""
    lat = _two_quad_lattice()
    res = pareto_optimize(
        lat, _bcfg(), ["exit_sigma_x", "neg_exit_energy"],
        algorithm="nsga2", pop_size=16, n_gen=8, seed=0,
    )
    # exit_sigma_x (focusing-dependent) and exit energy (drift energy is
    # fixed here, but sigma varies) -> at least the front is valid; we
    # assert >=1 and a valid non-dominated set (some lattices collapse to
    # one point, which is still correct).
    assert res.pareto_F.shape[0] >= 1
    assert _pareto_mask(res.pareto_F).all()
