"""Tests for the ``bayesopt`` matcher algorithm (BoTorch GP Bayesian
optimisation), the 7th optimiser in the matching engine.

Covers: registration, convergence on a single-quad SET_SIZE fixture,
finite-bounds requirement, the StopIteration cancellation contract, the
no-refine path, and that the result leaves the lattice at x_final.
"""
from __future__ import annotations

import numpy as np
import pytest

# bayesopt is an OPTIONAL algorithm (heavy botorch/gpytorch stack) — the
# engine raises a clean install-hint ImportError without it; these tests
# skip on machines/CI without the extra instead of failing.
pytest.importorskip("botorch")
pytest.importorskip("gpytorch")

from linac_gen.core.config import BeamConfig
from linac_gen.core.lattice import Lattice
from linac_gen.elements.drift import Drift
from linac_gen.elements.quadrupole import Quadrupole
from linac_gen.elements.lattice_commands import Adjust, SetSize
from linac_gen.matching import match, MATCH_ALGORITHMS


def _bcfg():
    return BeamConfig(
        species="proton", energy=3.0, frequency=352.21, n_particles=100,
        distribution="waterbag",
        emit_nx=0.25, alpha_x=0.0, beta_x=2.0,
        emit_ny=0.25, alpha_y=0.0, beta_y=2.0,
        emit_z=0.30, alpha_z=0.0, beta_z=10.0,
    )


def _one_quad_lattice(k_init=5.0, target_sigma_mm=2.0, vmin=0.5, vmax=30.0):
    lat = Lattice()
    lat.add(Drift("D1", length=200.0, aperture=10.0))
    lat.add(Adjust("CMD1", target="QUAD_001", param_idx=2,
                   link_group=0, vmin=vmin, vmax=vmax, start_step=0.5))
    lat.add(Quadrupole("QUAD_001", length=100.0, gradient=k_init,
                       aperture=10.0))
    lat.add(Drift("D2", length=200.0, aperture=10.0))
    lat.add(SetSize("CSET", k=1.0, x_mm=target_sigma_mm, y_mm=0.0,
                    phi_or_z=0.0))
    return lat


def test_bayesopt_in_algorithm_registry():
    assert "bayesopt" in MATCH_ALGORITHMS


def test_bayesopt_converges_and_records_baseline():
    """BO should reduce the cost well below the baseline and leave the
    lattice at the reported x_final."""
    lat = _one_quad_lattice()
    cfg = _bcfg()
    res = match(lat, cfg, algorithm="bayesopt", max_iter=20, refine=True)
    assert res.success
    assert res.baseline_cost is not None
    # Improved on (or matched) the baseline.
    assert res.cost <= res.baseline_cost + 1e-12
    # Lattice left at x_final.
    quad = next(e for e in lat.elements
                if e.__class__.__name__ == "Quadrupole")
    assert quad.gradient == pytest.approx(res.x_final[0], abs=1e-9)


def test_bayesopt_respects_bounds():
    lat = _one_quad_lattice(vmin=0.5, vmax=12.0)
    res = match(lat, _bcfg(), algorithm="bayesopt", max_iter=15,
                refine=False)
    assert 0.5 <= res.x_final[0] <= 12.0


def test_bayesopt_requires_finite_bounds():
    """vmin=vmax=0 is the 'unset' convention -> +/-inf bounds, which BO
    (like the other globals) must reject with a clear ValueError."""
    lat = Lattice()
    lat.add(Drift("D1", length=200.0, aperture=10.0))
    lat.add(Adjust("CMD1", target="QUAD_001", param_idx=2,
                   link_group=0, vmin=0.0, vmax=0.0, start_step=0.5))
    lat.add(Quadrupole("QUAD_001", length=100.0, gradient=5.0,
                       aperture=10.0))
    lat.add(Drift("D2", length=200.0, aperture=10.0))
    lat.add(SetSize("CSET", k=1.0, x_mm=2.0, y_mm=0.0, phi_or_z=0.0))
    with pytest.raises(ValueError, match="finite"):
        match(lat, _bcfg(), algorithm="bayesopt", max_iter=5)


def test_bayesopt_cancellation_preserves_best_x():
    """A callback raising StopIteration mid-run yields a cancelled
    MatchResult whose x_final is the best-cost x seen (not the trial
    point at cancellation), with the lattice left there."""
    lat = _one_quad_lattice()
    samples = []

    def cb(it, x, cost):
        samples.append((float(x[0]), float(cost)))
        if it >= 6:
            raise StopIteration("cancelled by user")

    res = match(lat, _bcfg(), algorithm="bayesopt", max_iter=30,
                refine=False, callback=cb)
    assert not res.success
    assert "cancel" in res.message.lower()
    best_x = min(samples, key=lambda p: p[1])[0]
    quad = next(e for e in lat.elements
                if e.__class__.__name__ == "Quadrupole")
    assert quad.gradient == pytest.approx(best_x, abs=1e-9)
    assert res.x_final[0] == pytest.approx(best_x, abs=1e-9)


def test_bayesopt_no_refine_runs():
    """The no-refine path returns a BO-only result (no LS polish)."""
    lat = _one_quad_lattice()
    res = match(lat, _bcfg(), algorithm="bayesopt", max_iter=12,
                refine=False)
    assert res.success
    assert "LS refine" not in res.message
