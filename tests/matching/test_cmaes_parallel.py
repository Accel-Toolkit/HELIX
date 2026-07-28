"""Regression tests for parallel CMA-ES matching.

The parallel branch evaluates the CMA-ES population across a
multiprocessing.Pool; we require it to land on the same matched
parameters as the sequential branch given a fixed seed and the same
lattice.  Skipped on platforms where multiprocessing.Pool isn't
available (some CI Windows runners).
"""
from __future__ import annotations

import math
import sys

import numpy as np
import pytest

from linac_gen.core.config import BeamConfig
from linac_gen.core.lattice import Lattice
from linac_gen.elements.drift import Drift
from linac_gen.elements.lattice_commands import Adjust, SetSize
from linac_gen.elements.quadrupole import Quadrupole
from linac_gen.matching import match


def _try_import_pool():
    """Skip cleanly on platforms that don't support multiprocessing."""
    try:
        from multiprocessing import Pool   # noqa: F401
    except Exception:
        pytest.skip("multiprocessing.Pool not available on this platform")


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


def _build_two_quad_lattice(g0_qf=5.0, g0_qd=-5.0):
    lat = Lattice()
    lat.add(Drift("D1", length=100.0))
    lat.add(Adjust("A1", target="QF", param_idx=2, link_group=1,
                   vmin=0.0, vmax=15.0, start_step=0.5))
    lat.add(Quadrupole("QF", length=80.0, gradient=g0_qf, aperture=20.0))
    lat.add(Drift("D2", length=200.0))
    lat.add(Adjust("A2", target="QD", param_idx=2, link_group=2,
                   vmin=-15.0, vmax=0.0, start_step=0.5))
    lat.add(Quadrupole("QD", length=80.0, gradient=g0_qd, aperture=20.0))
    lat.add(Drift("D3", length=100.0))
    lat.add(SetSize("C1", k=1.0, x_mm=2.0, y_mm=2.0))
    return lat


def test_parallel_matches_sequential(beam_cfg):
    """Parallel pool result agrees with sequential at fixed seed.

    Equivalence -- not optimality.  We're testing that the parallel
    code path produces the same trajectory as sequential, not that
    CMA-ES converges to a particular cost on this toy problem.
    """
    _try_import_pool()

    lat_seq = _build_two_quad_lattice()
    res_seq = match(lat_seq, beam_cfg, algorithm="cmaes",
                    max_iter=10, cmaes_sigma=0.3,
                    cmaes_parallel=1, refine=False)

    lat_par = _build_two_quad_lattice()
    res_par = match(lat_par, beam_cfg, algorithm="cmaes",
                    max_iter=10, cmaes_sigma=0.3,
                    cmaes_parallel=2, refine=False)

    # Same seed + same algorithm + same lattice → same trajectory.
    # We don't demand bit-identical (pool ordering can shuffle ties)
    # but the final cost should be within numerical noise.
    assert abs(res_par.cost - res_seq.cost) < 1e-6
    assert np.allclose(res_par.x_final, res_seq.x_final, atol=1e-5,
                       rtol=1e-5)


def test_parallel_zero_autodetects(beam_cfg):
    """cmaes_parallel=0 should auto-pick a positive worker count."""
    _try_import_pool()
    lat = _build_two_quad_lattice()
    res = match(lat, beam_cfg, algorithm="cmaes",
                max_iter=5, cmaes_sigma=0.3,
                cmaes_parallel=0, refine=False)
    # Just confirm it runs end-to-end.
    assert res.success or res.cost < 1e-4


def test_parallel_falls_back_to_sequential_for_unpicklable(beam_cfg, monkeypatch):
    """If pool creation fails (unpicklable lattice etc.) the match
    must fall back to sequential rather than crashing.

    Simulated by monkeypatching ``multiprocessing.Pool`` to raise."""
    _try_import_pool()
    import linac_gen.matching.engine as _engine
    # The engine looks up Pool via a `from multiprocessing import Pool`
    # inside the cmaes branch, so we patch the *module* not the engine.
    import multiprocessing as _mp
    orig = _mp.Pool

    def _raise_pool(*args, **kwargs):
        raise RuntimeError("simulated pickling failure")
    monkeypatch.setattr(_mp, "Pool", _raise_pool)

    lat = _build_two_quad_lattice()
    res = match(lat, beam_cfg, algorithm="cmaes",
                max_iter=5, cmaes_sigma=0.3,
                cmaes_parallel=4, refine=False)
    # Should still produce a valid result -- fell back to sequential.
    assert res.x_final.shape == (2,)
    monkeypatch.setattr(_mp, "Pool", orig)
