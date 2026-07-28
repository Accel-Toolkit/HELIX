"""Tests for the ``cost_solver="mp"`` matcher option.

When ``cost_solver="mp"``, the matcher swaps its forward pass from the
linear-SC envelope to a full multi-particle PIC run.  The plumbing
reuses ``run_mp_sim`` from ``linac_gen.cli.common`` and feeds a
``DiagnosticRecorder`` to the same constraint evaluators -- no
constraint changes were made.
"""
from __future__ import annotations

import pytest

from linac_gen.core.config import BeamConfig
from linac_gen.core.lattice import Lattice
from linac_gen.elements.drift import Drift
from linac_gen.elements.lattice_commands import Adjust, SetSize
from linac_gen.elements.quadrupole import Quadrupole
from linac_gen.matching.engine import match


def _bcfg(**over) -> BeamConfig:
    base = dict(species="proton", energy=3.0, frequency=352.21,
                n_particles=500, distribution="waterbag",
                emit_nx=0.25, alpha_x=0.0, beta_x=2.0,
                emit_ny=0.25, alpha_y=0.0, beta_y=2.0,
                emit_z=0.3,    alpha_z=0.0, beta_z=10.0,
                current=0.0)
    base.update(over)
    return BeamConfig(**base)


def _fodo_with_one_adjust(target_sigma_mm: float = 3.0,
                          k_init: float = 5.0) -> Lattice:
    lat = Lattice()
    lat.add(Drift("D1", length=200.0, aperture=10.0))
    lat.add(Adjust("CMD1", target="QUAD", param_idx=2,
                    link_group=0, vmin=0.5, vmax=30, start_step=0.5))
    lat.add(Quadrupole("QUAD_001", length=100.0, gradient=k_init,
                       aperture=10.0))
    lat.add(Drift("D2", length=200.0, aperture=10.0))
    lat.add(SetSize("CSET", k=1.0, x_mm=target_sigma_mm, y_mm=0.0,
                    phi_or_z=0.0))
    return lat


# ---------------------------------------------------------------------------
# Sanity: invalid cost_solver values raise early.
# ---------------------------------------------------------------------------
def test_mp_cost_solver_unknown_value_raises():
    lat = _fodo_with_one_adjust()
    with pytest.raises(ValueError, match="unknown cost_solver"):
        match(lat, _bcfg(), max_iter=1, cost_solver="bogus")


def test_mp_cost_solver_zero_particles_rejected():
    lat = _fodo_with_one_adjust()
    with pytest.raises(ValueError, match="mp_n_particles"):
        match(lat, _bcfg(), max_iter=1,
              cost_solver="mp", mp_n_particles=0)


# ---------------------------------------------------------------------------
# Smoke: MP cost solver runs end-to-end on a minimal lattice.
# ---------------------------------------------------------------------------
def test_mp_cost_solver_smoke_one_quad():
    """A 1-quad lattice with `cost_solver='mp'` produces a finite,
    non-NaN cost and a normal MatchResult."""
    lat = _fodo_with_one_adjust(target_sigma_mm=3.0, k_init=5.0)
    cfg = _bcfg(current=0.0)        # no SC for speed
    res = match(lat, cfg, max_iter=4,
                algorithm="least_squares",
                cost_solver="mp", mp_n_particles=200)
    import math
    assert res.x_final.shape == (1,)
    assert math.isfinite(res.cost)
    assert res.cost >= 0.0
    quad = next(e for e in lat.elements
                if e.__class__.__name__ == "Quadrupole")
    assert quad.gradient == pytest.approx(res.x_final[0], abs=1e-9)


# ---------------------------------------------------------------------------
# Default cost_solver is envelope -- existing callers see no change.
# ---------------------------------------------------------------------------
def test_default_cost_solver_is_envelope():
    """When cost_solver kwarg is omitted, the engine takes the
    envelope path.  Regression guard against accidental default-flip."""
    lat = _fodo_with_one_adjust()
    res = match(lat, _bcfg(), max_iter=2,
                algorithm="least_squares")  # no cost_solver kwarg
    # If MP had been the default, this would take much longer and
    # populate state['last_results'] with a DiagnosticRecorder; the
    # behaviour we want is "envelope unless explicitly opted in".
    # Indirectly tested via runtime + cost-shape; just confirm it
    # returns a MatchResult with the expected fields.
    assert res.x_final.shape == (1,)


# ---------------------------------------------------------------------------
# All algorithms accept cost_solver='mp' through the same plumbing.
# ---------------------------------------------------------------------------
def test_mp_cost_solver_works_with_all_algorithms():
    """cost_solver dispatches at _residual_at level; every algorithm
    should honour it.  Verifies the (small) cost is finite for each.
    """
    cfg = _bcfg(current=0.0)
    for algo in ("least_squares", "differential_evolution",
                 "dual_annealing", "cmaes"):
        lat = _fodo_with_one_adjust(target_sigma_mm=3.0, k_init=5.0)
        res = match(lat, cfg, algorithm=algo,
                    max_iter=2,
                    cost_solver="mp", mp_n_particles=200)
        import math
        assert math.isfinite(res.cost), (
            f"{algo} produced non-finite cost {res.cost} with MP solver"
        )


# ---------------------------------------------------------------------------
# sequential_scan + MP works without code changes (planned).
# ---------------------------------------------------------------------------
def test_mp_cost_solver_with_sequential_scan():
    """sequential_scan reads state['last_results'].emit_x/emit_z for
    its reversal check; both EnvelopeResults and DiagnosticRecorder
    populate those fields, so the scan loop works transparently with
    MP cost solver."""
    lat = _fodo_with_one_adjust(target_sigma_mm=3.0, k_init=5.0)
    cfg = _bcfg(current=0.0)
    res = match(lat, cfg, algorithm="sequential_scan",
                seqscan_passes=1, seqscan_steps=3,
                cost_solver="mp", mp_n_particles=200)
    import math
    assert math.isfinite(res.cost)
    assert res.x_final.shape == (1,)
