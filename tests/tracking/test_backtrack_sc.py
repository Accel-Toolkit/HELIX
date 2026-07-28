# tests/tracking/test_backtrack_sc.py
"""Backward tracking with space charge — the SC sign/grid traps.

Key physics locked in here:
* The negated kick (momentum-column flip around the forward kernel)
  must subtract EXACTLY what the forward Strang bundle added.  A sign
  error doubles the SC defocus and shows up at any current.
* PIC grid statefulness: grid_mode="fixed" freezes the forward grid on
  the run's first kick.  Reusing the forward solver reproduces it
  (exact undo); building a fresh solver from a fixed-grid config cannot
  and must fall back to adaptive with a warning.
"""
import warnings

import numpy as np
import pytest

from linac_gen.core.beam import Beam
from linac_gen.core.config import SpaceChargeConfig
from linac_gen.core.particle import PROTON
from linac_gen.core.reference import ReferenceParticle
from linac_gen.core.lattice import Lattice
from linac_gen.elements.drift import Drift
from linac_gen.elements.quadrupole import Quadrupole
from linac_gen.pic.pic_solver import PicSolver
from linac_gen.tracking.tracker import Tracker
from linac_gen.tracking.backtrack import (
    BacktrackWarning, backtrack_distribution,
)

W_KIN = 3.0
FREQ = 352.21


def _make_ref():
    return ReferenceParticle(species=PROTON, w_kin=W_KIN, frequency=FREQ)


def _make_beam(n=600, current=20.0, seed=1, continuous=False):
    beam = Beam(ref=_make_ref(), n_particles=n, current=current)
    beam.continuous = continuous
    rng = np.random.default_rng(seed)
    for j, s in enumerate([1.0, 0.3, 1.0, 0.3, 4.0, 0.003]):
        beam.particles[:, j] = rng.normal(0, s, n)
    return beam


def _fodo():
    lat = Lattice()
    lat.add(Quadrupole("QF", 50.0, gradient=5.0, n_steps=5))
    lat.add(Drift("D1", 200.0))
    lat.add(Quadrupole("QD", 50.0, gradient=-5.0, n_steps=5))
    lat.add(Drift("D2", 200.0))
    return lat


def test_pic_roundtrip_reused_solver_exact():
    """Fixed grid + the forward run's own solver → machine precision."""
    lat = _fodo()
    beam = _make_beam()
    p_in = beam.particles.copy()
    sc = SpaceChargeConfig(nx=32, ny=32, nz=32)     # default fixed grid
    solver = PicSolver(sc)
    Tracker(lat, beam, pic_solver=solver).run()
    assert beam.n_alive == beam.n_particles
    backtrack_distribution(lat, beam, _make_ref(), pic_solver=solver)
    np.testing.assert_allclose(beam.particles, p_in, rtol=1e-8, atol=1e-10)


def test_pic_roundtrip_adaptive_grid():
    """Adaptive grids rebuild identically from identical positions —
    round trip to ~1e-6 (float accumulation in the FFT solve only).
    A backward SIGN error would show up at the full kick magnitude
    (~1e-1 mrad here), 5 orders louder."""
    lat = _fodo()
    beam = _make_beam()
    p_in = beam.particles.copy()
    sc = SpaceChargeConfig(nx=32, ny=32, nz=32, grid_mode="adaptive")
    Tracker(lat, beam, pic_solver=PicSolver(sc)).run()
    backtrack_distribution(lat, beam, _make_ref(), sc_config=sc)
    np.testing.assert_allclose(beam.particles, p_in, rtol=1e-4, atol=2e-5)


def test_fixed_config_without_solver_warns_adaptive_fallback():
    lat = _fodo()
    beam = _make_beam()
    sc = SpaceChargeConfig(nx=32, ny=32, nz=32)     # fixed
    Tracker(lat, beam, pic_solver=PicSolver(sc)).run()
    with pytest.warns(BacktrackWarning, match="ADAPTIVE"):
        backtrack_distribution(lat, beam, _make_ref(), sc_config=sc)


def test_halo_backend_refused_without_approximate_optin():
    """sc_backend='halo' has no backward model: silently swapping in
    the plain PicSolver used to present an approximate undo as exact.
    Both regimes: refuse by default; warn-and-swap under the existing
    approximate_backtracking opt-in (same contract as CSR)."""
    lat = _fodo()
    beam = _make_beam()
    sc = SpaceChargeConfig(nx=32, ny=32, nz=32, grid_mode="adaptive")
    Tracker(lat, beam, pic_solver=PicSolver(sc)).run()
    import dataclasses
    sc_halo = dataclasses.replace(sc, sc_backend="halo")
    saved = beam.particles.copy()
    with pytest.raises(ValueError, match="sc_backend='halo'"):
        backtrack_distribution(lat, beam, _make_ref(), sc_config=sc_halo)
    # refusal must not have touched the beam
    np.testing.assert_array_equal(beam.particles, saved)
    with pytest.warns(BacktrackWarning, match="halo"):
        backtrack_distribution(lat, beam, _make_ref(), sc_config=sc_halo,
                               approximate_backtracking=True)


def test_halo_forward_solver_refused_on_backward_walk():
    """The SHIPPED path (Simulation.run_backtrack) passes the forward
    solver directly as pic_solver, bypassing the sc_config rebuild
    branch — the halo gate must also fire on a HaloPicSolver INSTANCE
    (adversarial review 2026-07: the stateful learned corrector would
    otherwise silently drive the backward walk mid-state)."""
    import dataclasses

    from linac_gen.pic.ml.solver import HaloPicSolver

    lat = _fodo()
    beam = _make_beam()
    sc = SpaceChargeConfig(nx=32, ny=32, nz=32, grid_mode="adaptive")
    Tracker(lat, beam, pic_solver=PicSolver(sc)).run()

    halo_cfg = dataclasses.replace(sc, sc_backend="halo")
    halo = HaloPicSolver(halo_cfg)
    saved = beam.particles.copy()
    with pytest.raises(ValueError, match="halo"):
        backtrack_distribution(lat, beam, _make_ref(), pic_solver=halo)
    np.testing.assert_array_equal(beam.particles, saved)
    with pytest.warns(BacktrackWarning, match="halo"):
        backtrack_distribution(lat, beam, _make_ref(), pic_solver=halo,
                               approximate_backtracking=True)


def test_dc_kernel_roundtrip():
    """Continuous (DC) beam: the analytic 2-D kick is stateless, so the
    undo is exact.  No bunching element in the lattice → the continuous
    flag is consistent end to end."""
    lat = _fodo()
    beam = _make_beam(current=10.0, continuous=True)
    p_in = beam.particles.copy()
    sc = SpaceChargeConfig(nx=32, ny=32, nz=32)
    Tracker(lat, beam, pic_solver=PicSolver(sc)).run()
    assert beam.continuous is True
    backtrack_distribution(lat, beam, _make_ref(), sc_config=sc)
    np.testing.assert_allclose(beam.particles, p_in, rtol=1e-8, atol=1e-10)


def test_sc_off_at_zero_current():
    """current=0 → no SC kicks either way; plain matrix round trip."""
    lat = _fodo()
    beam = _make_beam(current=0.0)
    p_in = beam.particles.copy()
    sc = SpaceChargeConfig(nx=32, ny=32, nz=32)
    solver = PicSolver(sc)
    Tracker(lat, beam, pic_solver=solver).run()
    backtrack_distribution(lat, beam, _make_ref(), pic_solver=solver)
    np.testing.assert_allclose(beam.particles, p_in, rtol=1e-8, atol=1e-11)


def test_space_charge_comp_factor_honoured():
    """A SPACE_CHARGE_COMP element scales SC in force downstream of its
    position; the replay table must carry the same factor so the undo
    subtracts the same (scaled) kick."""
    from linac_gen.elements.space_charge_comp import SpaceChargeComp
    lat = Lattice()
    lat.add(Quadrupole("QF", 50.0, gradient=5.0, n_steps=5))
    lat.add(Drift("D1", 200.0))
    lat.add(SpaceChargeComp("SCC", factor=0.6))     # 60 % neutralised
    lat.add(Quadrupole("QD", 50.0, gradient=-5.0, n_steps=5))
    lat.add(Drift("D2", 200.0))
    beam = _make_beam()
    p_in = beam.particles.copy()
    sc = SpaceChargeConfig(nx=32, ny=32, nz=32)
    solver = PicSolver(sc)
    Tracker(lat, beam, pic_solver=solver).run()
    backtrack_distribution(lat, beam, _make_ref(), pic_solver=solver)
    np.testing.assert_allclose(beam.particles, p_in, rtol=1e-8, atol=1e-10)
