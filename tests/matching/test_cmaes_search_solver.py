# tests/matching/test_cmaes_search_solver.py
"""Explicit CMA-ES search scoring (honesty round, 2026-07-19).

Parallel CMA-ES used to score the population search with the ENVELOPE
objective regardless of cost_solver — silently optimizing a different
problem than the one requested (PRAB review finding).  Now the search
objective is an explicit choice:

* ``cmaes_search_solver="auto"`` (default) — the search uses the
  requested cost solver faithfully; parallel + mp genuinely evaluates
  the multiparticle objective in the workers.
* ``cmaes_search_solver="envelope"`` — the fast envelope-guided hybrid
  (envelope-scored search, mp-scored baseline/polish/final), as an
  explicit opt-in.

The strongest correctness pin: CMA-ES is seeded (seed=1) and the mp
forward pass is seeded per evaluation, so parallel-mp and serial-mp
must produce IDENTICAL search trajectories and final knobs.
"""
import numpy as np
import pytest

from linac_gen.core.config import BeamConfig
from linac_gen.core.lattice import Lattice
from linac_gen.elements.drift import Drift
from linac_gen.elements.quadrupole import Quadrupole
from linac_gen.elements.lattice_commands import Adjust, MinTransmission, SetSize
from linac_gen.matching import match


def _lat():
    # Two knobs: single-variable CMA-ES trips a pre-existing 0-dim
    # sigma_vec bug inside the cma package (unrelated to this round).
    lat = Lattice()
    lat.add(Drift("D1", length=100.0, aperture=20.0))
    lat.add(Adjust("A1", target="QF", param_idx=2, link_group=1,
                   vmin=0.0, vmax=15.0, start_step=0.5))
    lat.add(Quadrupole("QF", length=80.0, gradient=5.0, aperture=20.0))
    lat.add(Drift("D2", length=200.0, aperture=20.0))
    lat.add(Adjust("A2", target="QD", param_idx=2, link_group=2,
                   vmin=-15.0, vmax=0.0, start_step=0.5))
    lat.add(Quadrupole("QD", length=80.0, gradient=-5.0, aperture=20.0))
    lat.add(Drift("D3", length=100.0, aperture=20.0))
    lat.add(SetSize("CSET", k=1.0, x_mm=2.0, y_mm=2.0, phi_or_z=0.0))
    return lat


def _cfg():
    return BeamConfig(species="proton", energy=3.0, frequency=352.21,
                      n_particles=100, distribution="waterbag",
                      emit_nx=0.25, alpha_x=0.0, beta_x=2.0,
                      emit_ny=0.25, alpha_y=0.0, beta_y=2.0,
                      emit_z=0.3, alpha_z=0.0, beta_z=10.0)


# max_iter=2 with the default sigma trips a cma-internal sigma-reinit
# IndexError on 1-D problems; 3 generations at sigma 0.3 (the config the
# existing cmaes tests use) is stable and still fast.
_FAST = dict(algorithm="cmaes", max_iter=3, cmaes_sigma=0.3,
             cmaes_popsize=4, refine=False,
             cost_solver="mp", mp_n_particles=100)


@pytest.mark.slow
def test_parallel_mp_search_matches_serial_mp_exactly(capfd):
    """cost_solver='mp' + parallel: workers must evaluate the REAL mp
    objective — pinned by exact trajectory equality with the serial mp
    run (same CMA seed, same per-eval mp seed).  The capfd assertions
    prove the pool genuinely ran (a silent sequential fallback would
    make the equality vacuous)."""
    r_serial = match(_lat(), _cfg(), cmaes_parallel=1, **_FAST)
    r_par = match(_lat(), _cfg(), cmaes_parallel=2, **_FAST)
    err = capfd.readouterr().err
    assert "falling back to sequential" not in err, err
    assert "MULTI-PARTICLE cost in 2 workers" in err, err
    assert "(parallel x2)" in r_par.message, r_par.message
    assert np.array_equal(r_serial.x_final, r_par.x_final), \
        (r_serial.x_final, r_par.x_final)
    assert r_serial.cost == r_par.cost


def test_hybrid_envelope_search_is_an_explicit_opt_in():
    """The old silent behavior survives ONLY as a named choice."""
    r = match(_lat(), _cfg(), cmaes_parallel=2,
              cmaes_search_solver="envelope", **_FAST)
    assert np.isfinite(r.cost)


def test_search_heavier_than_cost_refused():
    with pytest.raises(ValueError, match="heavier"):
        match(_lat(), _cfg(), algorithm="cmaes", cost_solver="envelope",
              cmaes_search_solver="mp", max_iter=2)


def test_unknown_search_solver_refused():
    with pytest.raises(ValueError, match="cmaes_search_solver"):
        match(_lat(), _cfg(), algorithm="cmaes",
              cmaes_search_solver="typo", max_iter=2)


def _lat_with_transmission():
    lat = _lat()
    lat.add(MinTransmission("CMT", threshold_pct=99.0, weight=1.0))
    return lat


def test_min_transmission_refused_with_envelope_scored_search():
    """Envelope-scored workers track no particle loss — the guard must
    still refuse MIN_TRANSMISSION for the envelope-search hybrid."""
    with pytest.raises(ValueError, match="MIN_TRANSMISSION"):
        match(_lat_with_transmission(), _cfg(), cmaes_parallel=2,
              cmaes_search_solver="envelope", **_FAST)


@pytest.mark.slow
def test_min_transmission_allowed_with_mp_scored_search():
    """With genuinely-mp workers the observable exists — the historical
    blanket refusal must not fire."""
    r = match(_lat_with_transmission(), _cfg(), cmaes_parallel=2, **_FAST)
    assert np.isfinite(r.cost)
