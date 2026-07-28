"""The pre-run constraint audit (2026-07-11): match() must refuse to run
rather than silently ignore an active constraint.

Two failure modes are covered:
  1. a stub card (parsed for round-trip, evaluator not implemented)
     carrying a nonzero weight;
  2. MIN_TRANSMISSION with the envelope cost solver (envelope mode
     tracks no particle loss, so the residual is identically zero).

`allow_inert_constraints=True` restores the old warn-and-continue
behaviour.
"""
from __future__ import annotations

import pytest

from linac_gen.core.config import BeamConfig
from linac_gen.core.lattice import Lattice
from linac_gen.elements.drift import Drift
from linac_gen.elements.lattice_commands import (
    Adjust, MinTransmission, SetPosition, SetSize,
)
from linac_gen.elements.quadrupole import Quadrupole
from linac_gen.matching import match


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


def _two_quad_lattice(extra_cards=()):
    lat = Lattice()
    lat.add(Drift("D1", length=100.0))
    lat.add(Adjust("A1", target="QF", param_idx=2, link_group=1,
                   vmin=0.0, vmax=15.0, start_step=0.5))
    lat.add(Quadrupole("QF", length=80.0, gradient=5.0, aperture=20.0))
    lat.add(Drift("D2", length=200.0))
    lat.add(Adjust("A2", target="QD", param_idx=2, link_group=2,
                   vmin=-15.0, vmax=0.0, start_step=0.5))
    lat.add(Quadrupole("QD", length=80.0, gradient=-5.0, aperture=20.0))
    lat.add(Drift("D3", length=100.0))
    lat.add(SetSize("C1", k=1.0, x_mm=2.0, y_mm=2.0))
    for card in extra_cards:
        lat.add(card)
    return lat


def test_min_transmission_with_envelope_solver_raises(beam_cfg):
    lat = _two_quad_lattice(
        [MinTransmission(name="MT", threshold_pct=99.0, weight=1.0)])
    with pytest.raises(ValueError, match="MIN_TRANSMISSION"):
        match(lat, beam_cfg, algorithm="least_squares", max_iter=2,
              cost_solver="envelope")


def test_active_stub_card_raises(beam_cfg):
    from linac_gen.elements.lattice_commands import SetAchromat
    lat = _two_quad_lattice([SetAchromat(name="SA")])
    with pytest.raises(ValueError, match="stub"):
        match(lat, beam_cfg, algorithm="least_squares", max_iter=2,
              cost_solver="envelope")


def test_cmaes_parallel_refuses_min_transmission_for_envelope_search(
        beam_cfg):
    # ENVELOPE-scored pool workers track no particle loss —
    # MIN_TRANSMISSION would silently read zero in every worker,
    # reopening the loss-gaming gap it exists to close.  Since the
    # honesty round the workers score the REQUESTED cost solver
    # (cost_solver='mp' + parallel now genuinely evaluates mp and
    # honours the constraint — covered in test_cmaes_search_solver);
    # the refusal applies exactly to envelope-scored search.
    lat = _two_quad_lattice(
        [MinTransmission(name="MT", threshold_pct=99.0, weight=1.0)])
    with pytest.raises(ValueError, match="MIN_TRANSMISSION"):
        match(lat, beam_cfg, algorithm="cmaes", max_iter=2,
              cost_solver="mp", mp_n_particles=50, cmaes_parallel=2,
              cmaes_search_solver="envelope")
    with pytest.raises(ValueError, match="MIN_TRANSMISSION"):
        match(lat, beam_cfg, algorithm="cmaes", max_iter=2,
              cost_solver="envelope", cmaes_parallel=2)


def test_set_position_runs_under_envelope_cost(beam_cfg):
    # Envelope results carry the centroid now — SET_POSITION is a live
    # constraint under BOTH cost solvers (the old audit refusal is gone;
    # only particle-loss observables stay MP-only).
    lat = _two_quad_lattice(
        [SetPosition(name="SP", k=1.0, x_mm=0.0)])
    res = match(lat, beam_cfg, algorithm="least_squares", max_iter=3,
                cost_solver="envelope")
    assert res is not None


def test_allow_inert_constraints_restores_old_behaviour(beam_cfg):
    lat = _two_quad_lattice(
        [SetPosition(name="SP", k=1.0, x_mm=0.0)])
    res = match(lat, beam_cfg, algorithm="least_squares", max_iter=5,
                cost_solver="envelope", allow_inert_constraints=True)
    assert res is not None    # ran to completion instead of raising


def test_set_size_k2_raises_under_audit(beam_cfg):
    """k2 != 0 asks for sizes INCLUDING the centroid (TW manual); HELIX
    evaluates about the centroid regardless.  The audit refuses instead
    of letting the remodelled flag pass with only a warning."""
    import warnings as _w
    lat = _two_quad_lattice(
        [SetSize("C2", k=1.0, x_mm=2.0, y_mm=2.0, k2=1)])
    with _w.catch_warnings():
        _w.simplefilter("ignore")   # the per-collect k2 warning
        with pytest.raises(ValueError, match="k2=1"):
            match(lat, beam_cfg, algorithm="least_squares", max_iter=2,
                  cost_solver="envelope")
        # explicit override restores warn-and-continue
        res = match(lat, beam_cfg, algorithm="least_squares", max_iter=3,
                    cost_solver="envelope", allow_inert_constraints=True)
    assert res is not None


def test_clean_lattice_unaffected(beam_cfg):
    lat = _two_quad_lattice()
    res = match(lat, beam_cfg, algorithm="least_squares", max_iter=5,
                cost_solver="envelope")
    assert res is not None
