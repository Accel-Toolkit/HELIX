# tests/tracking/test_drift_leftover_sc.py
"""Space charge over a drift's leftover sub-steps.

A drift is walked in space-charge bundles of ``sc_every`` integration sub-steps.
When step1 / step2 leave sub-steps after the last bundle (150/50, 200/50, 100/30
…; never at 1:1 or 2:1, so not at the default 100/50), those sub-steps used to run
without a kick: 0.9–2.7 % of the drift length of a superconducting linac.  They now form a
short last bundle with its own kick over its own length — in the tracker, the
backtracker and the differentiable step tracker.  The rounding remainder the old
code also produced when the sub-steps divide evenly (up to ~1e-13 mm) stays a plain
drift, so runs at those settings keep their bits.
"""
import copy

import numpy as np
import pytest

from linac_gen.core.beam import Beam
from linac_gen.core.config import SpaceChargeConfig
from linac_gen.core.lattice import Lattice
from linac_gen.core.particle import PROTON
from linac_gen.core.reference import ReferenceParticle
from linac_gen.core.simulation import Simulation
from linac_gen.core.step_config import StepConfig
from linac_gen.elements.drift import Drift
from linac_gen.elements.quadrupole import Quadrupole
from linac_gen.pic.pic_solver import PicSolver
from linac_gen.tracking import backtrack as bt
from linac_gen.tracking import tracker as T
from linac_gen.tracking.tracker import Tracker

SC_ADAPTIVE = SpaceChargeConfig(nx=16, ny=16, nz=16, grid_extent=4.0,
                                use_gpu="cpu", grid_mode="adaptive")


def _ref():
    return ReferenceParticle(species=PROTON, w_kin=3.0, frequency=352.21)


def _beam(n=600, current=20.0, seed=1, continuous=False):
    beam = Beam(ref=_ref(), n_particles=n, current=current)
    beam.continuous = continuous
    rng = np.random.default_rng(seed)
    for j, s in enumerate([1.0, 0.3, 1.0, 0.3, 4.0, 0.003]):
        beam.particles[:, j] = rng.normal(0, s, n)
    return beam


def _lattice(step1, step2, *elements):
    lat = Lattice()
    for e in elements:
        lat.add(e)
    lat.step_config = StepConfig(step1, step2)
    return lat


def _schedule(L, step1, step2):
    st = StepConfig(step1, step2)
    n_int = st.integration_steps_for_length_mm(L)
    n_sc = st.sc_steps_for_length_mm(L)
    sc_every = max(1, n_int // n_sc)
    return n_int, L / n_int, sc_every, n_int // sc_every


@pytest.fixture
def kick_lengths(monkeypatch):
    lengths = []
    orig = T.Tracker._apply_sc_kick

    def spy(self, ds_mm, *a, **k):
        lengths.append(ds_mm)
        return orig(self, ds_mm, *a, **k)
    monkeypatch.setattr(T.Tracker, "_apply_sc_kick", spy)
    return lengths


@pytest.mark.parametrize("L, step1, step2", [(42.0, 150, 50), (250.0, 100, 30),
                                             (250.0, 200, 50)])
def test_kicks_cover_the_whole_drift(kick_lengths, L, step1, step2):
    n_int, ds, sc_every, nb = _schedule(L, step1, step2)
    assert n_int > nb * sc_every                     # sub-steps are left over
    Simulation(_lattice(step1, step2, Drift("D", L)), _beam(),
               space_charge=SC_ADAPTIVE).run()
    assert len(kick_lengths) == nb + 1
    assert kick_lengths[:nb] == [sc_every * ds] * nb
    assert kick_lengths[-1] == pytest.approx((n_int - nb * sc_every) * ds, rel=1e-12)
    assert sum(kick_lengths) == pytest.approx(L, rel=1e-12)


@pytest.mark.parametrize("L", [42.0, 172.5, 250.0, 24.02])   # 24.02: rounding remainder
@pytest.mark.parametrize("step1, step2", [(100, 50), (100, 100)])
def test_whole_bundles_keep_the_old_schedule(kick_lengths, L, step1, step2):
    n_int, ds, sc_every, nb = _schedule(L, step1, step2)
    assert n_int == nb * sc_every
    if L == 24.02 and step2 == 50:
        assert 0.0 < L - nb * sc_every * ds < 1e-13   # the guard below is exercised
    Simulation(_lattice(step1, step2, Drift("D", L)), _beam(n=200),
               space_charge=SC_ADAPTIVE).run()
    assert kick_lengths == [sc_every * ds] * nb      # no kick on a rounding remainder


def test_the_short_bundle_restores_the_missing_impulse():
    """42 mm at 150/50 leaves 6 mm (1/7 of the drift) after the last bundle.  The
    space-charge growth of the rms divergence must now match a fine 1500/1500 walk
    (ratio 1.00006); without the short bundle's kick it was 0.72 — the growth of an
    uncorrelated beam's rms divergence goes as the square of the impulse, (6/7)²."""
    def divergence_change(step1, step2):
        beam = _beam(n=2000, current=40.0, seed=3, continuous=True)
        x0 = beam.particles[:, 1].std()
        Simulation(_lattice(step1, step2, Drift("D", 42.0)), beam,
                   space_charge=SC_ADAPTIVE).run()
        return beam.particles[:, 1].std() - x0
    fine = divergence_change(1500, 1500)
    assert fine > 0
    assert divergence_change(150, 50) / fine == pytest.approx(1.0, abs=0.005)


def test_backtrack_undoes_the_short_bundle(monkeypatch):
    lat = _lattice(150, 50, Drift("D0", 42.0),
                   Quadrupole("QF", 50.0, gradient=5.0, n_steps=5),
                   Drift("D1", 41.91))
    forward, backward = [], []
    orig_f, orig_b = T.Tracker._apply_sc_kick, bt._Backtracker._apply_sc_kick_negated

    def spy_f(self, ds_mm, *a, **k):
        forward.append(round(float(ds_mm), 9))
        return orig_f(self, ds_mm, *a, **k)

    def spy_b(self, ds_mm):
        backward.append(round(float(ds_mm), 9))
        return orig_b(self, ds_mm)
    monkeypatch.setattr(T.Tracker, "_apply_sc_kick", spy_f)
    monkeypatch.setattr(bt._Backtracker, "_apply_sc_kick_negated", spy_b)
    beam = _beam()
    p_in = beam.particles.copy()
    solver = PicSolver(SpaceChargeConfig(nx=32, ny=32, nz=32))     # fixed grid, reused
    Tracker(lat, beam, pic_solver=solver).run()
    assert beam.n_alive == beam.n_particles
    bt.backtrack_distribution(lat, beam, _ref(), pic_solver=solver)
    assert sorted(backward) == sorted(forward)
    np.testing.assert_allclose(beam.particles, p_in, rtol=1e-8, atol=1e-10)


def test_differentiable_step_tracker_matches_the_tracker():
    torch = pytest.importorskip("torch")
    from linac_gen.tracking.torch_step_tracker import track_beam_torch_stepwise

    def lattice():
        return _lattice(150, 50, Drift("D0", 42.0),
                        Quadrupole("QF", 100.0, gradient=8.0),
                        Drift("D1", 41.91))
    beam = _beam(n=200, current=12.0, seed=7)
    x0 = beam.particles.copy()
    beam_np = copy.deepcopy(beam)
    Simulation(lattice(), beam_np, space_charge=SC_ADAPTIVE).run()
    assert beam_np.n_alive == beam.n_particles
    macro = (beam.current * 1e-3) / (beam.bunch_frequency * 1e6) / beam.n_particles
    out = track_beam_torch_stepwise(lattice(), _ref(), x0, sc_cfg=SC_ADAPTIVE,
                                    macro_charge=macro)
    assert isinstance(out, torch.Tensor)
    scale = float(np.abs(beam_np.particles).max())
    np.testing.assert_allclose(out.detach().numpy(), beam_np.particles,
                               rtol=1e-6, atol=1e-6 * scale)
