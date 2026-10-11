"""``PicSolver.kick`` split into stages must give the same particles bit for
bit as the single-block kick it replaced.

The split (``_rest_frame_field`` for grid set-up, charges, deposit, solve and
gather; the kick formula unchanged in ``kick``) exists so solver variants can
change how the field is obtained.  ``_kick_pre_split`` below is a literal
copy of the standard path of the kick as it was before the split.
"""
from __future__ import annotations

import numpy as np
import pytest

from linac_gen.core.beam import Beam
from linac_gen.core.config import SpaceChargeConfig
from linac_gen.core.particle import H_MINUS
from linac_gen.core.reference import ReferenceParticle
from linac_gen.pic import pic_solver as ps
from linac_gen.pic.macrocharge import macro_charge_for, macro_charges_weighted


def _kick_pre_split(solver, beam, ds):
    """Standard (isolated-bunch) path of the pre-split PicSolver.kick."""
    solver._train_kick_ordinal += 1
    if beam.current <= 0 or beam.n_alive < 2:
        return
    alive_mask = beam.alive_mask
    coords_lab = ps.beam_to_spatial(beam)
    gamma = beam.ref.gamma
    coords_rest = ps.boost_to_rest_frame(coords_lab, gamma)
    solver._setup_grid(coords_rest)
    n_alive = beam.n_alive
    macro_charge = macro_charge_for(beam)
    if solver._deposit_weights is None:
        charges = np.full(n_alive, macro_charge)
    else:
        charges = macro_charges_weighted(beam, solver._deposit_weights)
    deposit_fn, interpolate_fn = ps._select_kernel(
        getattr(solver.config, "kernel", "cic"))
    rho = deposit_fn(coords_rest, charges,
                     solver._grid_min, solver._grid_max, solver._n_grid)
    Ex_rest, Ey_rest, Ez_rest = solver._solver.solve(rho)
    E_at_particles = interpolate_fn(
        Ex_rest, Ey_rest, Ez_rest,
        coords_rest, solver._grid_min, solver._grid_max, solver._n_grid)
    ds_m = ds * 1e-3
    beta = beam.ref.beta
    mass = beam.ref.species.mass
    E_si = E_at_particles * 1e6
    z_state = abs(beam.ref.species.charge)
    factor_t = z_state * ds_m * 1e3 / (mass * 1e6 * beta**2 * gamma**2)
    factor_z = z_state * ds_m * 1e-6
    alive_indices = np.where(alive_mask)[0]
    beam.particles[alive_indices, 1] += factor_t * E_si[:, 0]
    beam.particles[alive_indices, 3] += factor_t * E_si[:, 1]
    beam.particles[alive_indices, 5] += factor_z * E_si[:, 2]


def _beam(n=12_000, seed=3, lost=0):
    rng = np.random.default_rng(seed)
    b = Beam(ref=ReferenceParticle(species=H_MINUS, w_kin=2.12, frequency=162.5),
             n_particles=n, current=5.0)
    b.bunch_frequency = 162.5
    b.particles[:] = rng.standard_normal((n, 6)) * [1.5, 1.5, 1.2, 1.5, 8.0, 0.01]
    if lost:
        b.lost[:lost] = True
    return b


def _cfg(kernel, mode, n=24):
    return SpaceChargeConfig(nx=n, ny=n, nz=n, kernel=kernel, grid_mode=mode,
                             grid_extent=6.0, use_gpu="cpu")


def _run(kick, cfg, *, kicks=3, weights=False, extent=None, lost=0):
    beam = _beam(lost=lost)
    solver = ps.PicSolver(cfg)
    if weights:
        solver._deposit_weights = np.random.default_rng(9).uniform(
            0.2, 1.0, beam.n_alive)
    for i in range(kicks):
        if extent is not None and i == 1:
            solver.set_grid_extent(extent)
        kick(solver, beam, 2.5)
        # move the beam so an adaptive grid really changes between kicks
        beam.particles[:, 0] += 0.05 * beam.particles[:, 1]
        beam.particles[:, 4] *= 1.01
    return beam.particles.copy()


CASES = [(k, m) for k in ("cic", "tsc") for m in ("fixed", "adaptive")]


@pytest.mark.parametrize("kernel,mode", CASES)
def test_kick_matches_pre_split_copy(kernel, mode):
    cfg = _cfg(kernel, mode)
    new = _run(lambda s, b, ds: s.kick(b, ds), cfg)
    old = _run(_kick_pre_split, cfg)
    assert not np.array_equal(new, _beam().particles)       # the kick acted
    assert np.array_equal(new, old)


@pytest.mark.parametrize("kernel,mode", CASES)
def test_weighted_deposit_matches_copy(kernel, mode):
    cfg = _cfg(kernel, mode)
    new = _run(lambda s, b, ds: s.kick(b, ds), cfg, weights=True)
    old = _run(_kick_pre_split, cfg, weights=True)
    assert np.array_equal(new, old)


@pytest.mark.parametrize("kernel,mode", CASES)
def test_extent_override_matches_copy(kernel, mode):
    cfg = _cfg(kernel, mode)
    new = _run(lambda s, b, ds: s.kick(b, ds), cfg, extent=9.0)
    old = _run(_kick_pre_split, cfg, extent=9.0)
    assert np.array_equal(new, old)


def test_lost_particles_matches_copy():
    cfg = _cfg("tsc", "adaptive")
    new = _run(lambda s, b, ds: s.kick(b, ds), cfg, lost=500)
    old = _run(_kick_pre_split, cfg, lost=500)
    assert np.array_equal(new, old)


def test_rest_frame_field_is_solver_units_at_alive_particles():
    beam = _beam(lost=100)
    solver = ps.PicSolver(_cfg("cic", "adaptive"))
    coords = ps.boost_to_rest_frame(ps.beam_to_spatial(beam), beam.ref.gamma)
    E = solver._rest_frame_field(coords, beam)
    assert E.shape == (beam.n_alive, 3)
    assert np.all(np.isfinite(E))


def test_bunch_train_path_still_runs_and_repeats():
    def once():
        beam = _beam()
        beam.bunch_train = True
        beam.particles[:, 4] *= 12.0          # core sigma_phi well above 35 deg
        solver = ps.PicSolver(_cfg("cic", "adaptive"))
        solver.kick(beam, 2.5)
        assert solver._train_engaged
        return beam.particles.copy()
    assert np.array_equal(once(), once())
