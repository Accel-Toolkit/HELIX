"""Field solved every K-th kick and rescaled in between
(linac_gen.pic.field_reuse.FieldReusePicSolver)."""
from __future__ import annotations

import dataclasses
import warnings

import numpy as np
import pytest

from linac_gen.core.beam import Beam
from linac_gen.core.config import SpaceChargeConfig
from linac_gen.core.particle import H_MINUS
from linac_gen.core.reference import ReferenceParticle
from linac_gen.pic import factory, methods
from linac_gen.pic.field_reuse import FieldReusePicSolver
from linac_gen.pic.pic_solver import PicSolver


def _beam(n=4000, seed=3):
    ref = ReferenceParticle(species=H_MINUS, w_kin=2.1, frequency=162.5)
    beam = Beam(ref=ref, n_particles=n, current=5.0)
    rng = np.random.default_rng(seed)
    beam.particles[:] = rng.standard_normal((n, 6)) * [1.2, 1.0, 1.0, 1.0,
                                                       9.0, 0.005]
    return beam


def _cfg(k, kernel="tsc", **kw):
    return SpaceChargeConfig(nx=16, ny=16, nz=16, grid_extent=6.0,
                             grid_mode="adaptive", kernel=kernel,
                             use_gpu="cpu", reuse_every=k, **kw)


def _rest(beam):
    from linac_gen.pic.coordinates import beam_to_spatial
    from linac_gen.pic.lorentz_boost import boost_to_rest_frame
    return boost_to_rest_frame(beam_to_spatial(beam), beam.ref.gamma)


# ---------------------------------------------------------------------------
# K = 1 and the solve kicks are the plain PIC, bit for bit
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("kernel", ["cic", "tsc"])
def test_reuse_every_1_is_the_plain_pic(kernel):
    a, b = _beam(), _beam()
    plain = PicSolver(_cfg(1, kernel))
    reuse = FieldReusePicSolver(_cfg(1, kernel))
    assert reuse.exact_backward
    for _ in range(3):
        plain.kick(a, 5.0)
        reuse.kick(b, 5.0)
        a.particles[:, 0] += 0.01 * a.particles[:, 1]
        b.particles[:, 0] += 0.01 * b.particles[:, 1]
    assert np.array_equal(a.particles, b.particles)
    assert reuse.n_solves == 0          # never left PicSolver's own step


@pytest.mark.parametrize("kernel", ["cic", "tsc"])
def test_solve_kick_is_the_plain_pic(kernel):
    a, b = _beam(), _beam()
    PicSolver(_cfg(1, kernel)).kick(a, 5.0)
    pic = FieldReusePicSolver(_cfg(2, kernel))
    pic.kick(b, 5.0)
    assert pic.n_solves == 1 and pic.n_reused == 0
    assert np.array_equal(a.particles, b.particles)


# ---------------------------------------------------------------------------
# the rescaling law
# ---------------------------------------------------------------------------
def _solved_then_reused(transform):
    """Field at the particles from a solve, and the reused field after the
    rest-frame positions are transformed (charges unchanged)."""
    beam = _beam()
    pic = FieldReusePicSolver(_cfg(2))
    c0 = _rest(beam)
    E0 = pic._rest_frame_field(c0, beam)
    c1 = transform(c0)
    E1 = pic._rest_frame_field(c1, beam)
    assert pic.n_solves == 1 and pic.n_reused == 1
    return c0, E0, c1, E1


def test_unchanged_beam_gets_back_its_solved_field():
    _, E0, _, E1 = _solved_then_reused(lambda c: c.copy())
    np.testing.assert_allclose(E1, E0, rtol=1e-11, atol=1e-12 * abs(E0).max())


def test_translation_leaves_the_field_unchanged():
    _, E0, _, E1 = _solved_then_reused(lambda c: c + [0.3, -0.2, 1.5])
    np.testing.assert_allclose(E1, E0, rtol=1e-11, atol=1e-12 * abs(E0).max())


def test_uniform_scaling_scales_the_field_by_one_over_a_squared():
    a = 1.3
    _, E0, _, E1 = _solved_then_reused(
        lambda c: c.mean(axis=0) + a * (c - c.mean(axis=0)))
    np.testing.assert_allclose(E1, E0 / a**2, rtol=1e-11,
                               atol=1e-12 * abs(E0).max())


def test_anisotropic_scaling_follows_the_rms_sizes():
    """E_i = g_i(u) Q / (s_bar s_i): a stretch of one axis by a changes E_i
    by s_bar0 s_i0 / (s_bar s_i)."""
    a = np.array([1.0, 1.0, 2.0])
    c0, E0, c1, E1 = _solved_then_reused(
        lambda c: c.mean(axis=0) + a * (c - c.mean(axis=0)))
    s0, s1 = c0.std(axis=0), c1.std(axis=0)
    factor = (np.prod(s0) ** (1 / 3) * s0) / (np.prod(s1) ** (1 / 3) * s1)
    np.testing.assert_allclose(E1, E0 * factor, rtol=1e-11,
                               atol=1e-12 * abs(E0).max())


def test_charge_scales_the_reused_field():
    beam = _beam()
    pic = FieldReusePicSolver(_cfg(2))
    c = _rest(beam)
    E0 = pic._rest_frame_field(c, beam)
    beam.current *= 0.5                     # half the charge per particle
    E1 = pic._rest_frame_field(c, beam)
    assert pic.n_reused == 1
    np.testing.assert_allclose(E1, 0.5 * E0, rtol=1e-11,
                               atol=1e-12 * abs(E0).max())


# ---------------------------------------------------------------------------
# cadence and the events that force a solve
# ---------------------------------------------------------------------------
def test_cadence_k4():
    beam = _beam()
    pic = FieldReusePicSolver(_cfg(4))
    order = []
    for _ in range(8):
        before = pic.n_solves
        pic.kick(beam, 2.0)
        order.append("S" if pic.n_solves > before else "r")
        beam.particles[:, 0] += 0.05 * beam.particles[:, 1]    # drift
    assert "".join(order) == "SrrrSrrr"
    assert pic.stats()["solves"] == 2 and pic.stats()["reused"] == 6
    # one Green's-function build per solve (the cell size changed)
    assert pic.stats()["green_builds"] == 2


def _forces_a_solve(event):
    beam = _beam()
    pic = FieldReusePicSolver(_cfg(4))
    pic.kick(beam, 2.0)
    event(pic, beam)
    before = pic.n_solves
    pic.kick(beam, 2.0)
    return pic.n_solves > before


def test_grid_extent_card_forces_a_solve():
    assert _forces_a_solve(lambda pic, beam: pic.set_grid_extent(8.0))


def test_travel_forces_a_solve():
    def travel(pic, beam):
        beam.ref.s += 150.0
    assert _forces_a_solve(travel)
    assert not _forces_a_solve(lambda pic, beam: None)


def test_loss_forces_a_solve():
    def lose(pic, beam):
        beam.lost[: int(0.02 * beam.n_particles)] = True
    assert _forces_a_solve(lose)


def test_bunch_train_kick_drops_the_stored_field():
    beam = _beam()
    pic = FieldReusePicSolver(_cfg(4))
    pic.kick(beam, 2.0)
    assert pic._store is not None
    pic.train_force_engage = True
    beam.bunch_train = True
    pic.kick(beam, 2.0)                 # through the train path: full solve
    assert pic._store is None
    pic.train_force_engage = False
    beam.bunch_train = False
    before = pic.n_solves
    pic.kick(beam, 2.0)
    assert pic.n_solves == before + 1


# ---------------------------------------------------------------------------
# configuration, registry, factory, backward walk
# ---------------------------------------------------------------------------
def test_config_checks():
    with pytest.raises(ValueError, match="reuse_every"):
        SpaceChargeConfig(reuse_every=0)
    with pytest.raises(ValueError, match="adaptive"):
        SpaceChargeConfig(reuse_every=2)                  # fixed grid
    with pytest.raises(ValueError, match="numpy"):
        SpaceChargeConfig(reuse_every=2, grid_mode="adaptive",
                          sc_backend="torch")
    assert SpaceChargeConfig().reuse_every == 1


@pytest.mark.parametrize("kern", ["cic", "tsc"])
@pytest.mark.parametrize("k", [2, 4])
def test_reuse_methods_are_named(kern, k):
    name = f"{kern}_adaptive_reuse{k}"
    cfg = SpaceChargeConfig(method=name)
    assert (cfg.kernel, cfg.grid_mode, cfg.reuse_every) == (kern, "adaptive", k)
    assert cfg.method == name
    assert methods.infer(dataclasses.asdict(cfg)) == name
    assert SpaceChargeConfig(method=f"{kern}_adaptive").reuse_every == 1


def test_factory_builds_the_reuse_solver_forward_only():
    cfg = SpaceChargeConfig(method="tsc_adaptive_reuse2")
    pic = factory.make_sc_solver(cfg)
    assert type(pic) is FieldReusePicSolver and not pic.exact_backward
    assert type(factory.make_sc_solver(cfg, purpose="backward")) is PicSolver
    assert type(factory.make_sc_solver(
        SpaceChargeConfig(method="tsc_adaptive"))) is PicSolver


def test_backward_walk_solves_every_kick_with_a_warning():
    from linac_gen.core.lattice import Lattice
    from linac_gen.elements.drift import Drift
    from linac_gen.tracking.backtrack import (BacktrackWarning,
                                              backtrack_distribution)
    from linac_gen.tracking.tracker import Tracker
    lat = Lattice()
    lat.add(Drift("D1", 100.0))
    beam = _beam()
    ref0 = ReferenceParticle(species=H_MINUS, w_kin=2.1, frequency=162.5)
    solver = FieldReusePicSolver(_cfg(2))
    Tracker(lat, beam, pic_solver=solver).run()
    with pytest.warns(BacktrackWarning, match="every 2 kicks"):
        rec = backtrack_distribution(lat, beam, ref0, pic_solver=solver)
    assert rec.sc_config.reuse_every == 1
    assert rec.sc_config.method == "tsc_adaptive"
    # rebuilt from the config: same rule
    with pytest.warns(BacktrackWarning, match="every 2 kicks"):
        backtrack_distribution(lat, beam, ref0, sc_config=_cfg(2))


def test_simulation_records_the_counters(tmp_path):
    import json

    import h5py

    from linac_gen.core.config import BeamConfig
    from linac_gen.core.lattice import Lattice
    from linac_gen.core.simulation import Simulation
    from linac_gen.distributions.factory import create_beam
    from linac_gen.elements.drift import Drift
    from linac_gen.io.hdf5_output import save_results_hdf5
    lat = Lattice()
    lat.add(Drift(name="D1", length=60.0, aperture=50.0))
    bc = BeamConfig(species="H-", energy=2.0, frequency=162.5, current=5.0,
                    n_particles=300, emit_nx=0.2, beta_x=0.5, emit_ny=0.2,
                    beta_y=0.5, emit_z=0.06, beta_z=500.0)
    sc = SpaceChargeConfig(method="cic_adaptive_reuse2", nx=8, ny=8, nz=8,
                           use_gpu="cpu")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        rec = Simulation(lat, create_beam(bc, seed=1), space_charge=sc).run()
    st = rec.sc_solver_stats
    assert st["solves"] >= 1 and st["reused"] >= 1
    assert abs(st["solves"] - st["reused"]) <= 1
    out = tmp_path / "r.h5"
    save_results_hdf5(rec, str(out))
    with h5py.File(out, "r") as f:
        assert json.loads(f["provenance"].attrs["sc_solver_stats"]) == st
        assert f["provenance"].attrs["sc_method"] == "cic_adaptive_reuse2"


# ---------------------------------------------------------------------------
# review additions
# ---------------------------------------------------------------------------
def test_a_later_solve_kick_is_the_plain_pic_field():
    """Every solve kick, not only the first, computes PicSolver's field."""
    beam = _beam()
    pic = FieldReusePicSolver(_cfg(4))
    for _ in range(4):                       # solve + 3 reuses
        pic.kick(beam, 2.0)
        beam.particles[:, 0] += 0.05 * beam.particles[:, 1]
    c = _rest(beam)
    before = pic.n_solves
    got = pic._rest_frame_field(c, beam)     # the 5th kick solves
    assert pic.n_solves == before + 1
    want = PicSolver(_cfg(1))._rest_frame_field(c, beam)
    assert np.array_equal(got, want)


def test_small_loss_keeps_the_stored_field():
    def lose_a_little(pic, beam):
        beam.lost[: int(0.005 * beam.n_particles)] = True
    assert not _forces_a_solve(lose_a_little)


def test_zero_charge_forces_a_solve():
    def no_charge(pic, beam):
        beam.current = 0.0
    beam = _beam()
    pic = FieldReusePicSolver(_cfg(4))
    c = _rest(beam)
    pic._rest_frame_field(c, beam)
    no_charge(pic, beam)
    assert pic._reused_field(c, beam) is None


def test_zero_extent_forces_a_solve():
    beam = _beam()
    pic = FieldReusePicSolver(_cfg(4))
    c = _rest(beam)
    pic._rest_frame_field(c, beam)
    flat = c.copy()
    flat[:, 2] = flat[0, 2]                  # no longitudinal extent
    assert pic._reused_field(flat, beam) is None


def test_counters_record_the_limits():
    st = FieldReusePicSolver(_cfg(2)).stats()
    assert st["reuse_every"] == 2 and st["max_reuse_travel_mm"] == 100.0
    assert st["max_reuse_loss"] == 0.01


def test_backward_walk_with_a_halo_solver_on_a_numpy_config_is_unchanged():
    """Regression (review): the reuse branch must key on reuse_every; a
    HaloPicSolver built on a numpy config keeps the HALO refusal."""
    from linac_gen.core.lattice import Lattice
    from linac_gen.elements.drift import Drift
    from linac_gen.pic.ml.solver import HaloPicSolver
    from linac_gen.tracking.backtrack import backtrack_distribution
    lat = Lattice()
    lat.add(Drift("D1", 100.0))
    beam = _beam(n=500)
    ref0 = ReferenceParticle(species=H_MINUS, w_kin=2.1, frequency=162.5)
    halo = HaloPicSolver(SpaceChargeConfig(nx=16, ny=16, nz=16,
                                           grid_extent=5.0,
                                           grid_mode="adaptive"))
    with pytest.raises(ValueError, match="approximate_backtracking"):
        backtrack_distribution(lat, beam, ref0, pic_solver=halo)
