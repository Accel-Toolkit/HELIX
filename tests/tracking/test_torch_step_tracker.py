"""M4 — the step-by-step differentiable tracker.

SC-off it must reproduce the pre-composed ``track_beam_torch`` exactly;
SC-on it must reproduce the numpy ``Tracker`` + ``PicSolver``; and the
whole forward pass (with space charge) must stay autograd-differentiable.
"""
import copy
import os

import numpy as np
import pytest
import torch

from linac_gen.core.beam import Beam
from linac_gen.core.config import SpaceChargeConfig
from linac_gen.core.lattice import Lattice
from linac_gen.core.particle import PROTON
from linac_gen.core.reference import ReferenceParticle
from linac_gen.core.simulation import Simulation
from linac_gen.elements.drift import Drift
from linac_gen.elements.quadrupole import Quadrupole
from linac_gen.io.tracewin_parser import parse_tracewin
from linac_gen.tracking.torch_step_tracker import (
    DifferentiableStepTracker, track_beam_torch_stepwise,
)
from linac_gen.tracking.torch_tracking import track_beam_torch

BTL = "examples/pipii/btl/btl.dat"


def _ref():
    return ReferenceParticle(species=PROTON, w_kin=3.0, frequency=352.21)


def _fodo():
    lat = Lattice()
    lat.add(Drift("D1", length=200.0))
    lat.add(Quadrupole("QF", length=100.0, gradient=8.0))
    lat.add(Drift("D2", length=400.0))
    lat.add(Quadrupole("QD", length=100.0, gradient=-8.0))
    lat.add(Drift("D3", length=200.0))
    return lat


def _bunched_beam(n=200, current=12.0, seed=7):
    beam = Beam(ref=_ref(), n_particles=n, current=current)
    rng = np.random.default_rng(seed)
    beam.particles[:, 0] = rng.normal(0.0, 1.0, n)
    beam.particles[:, 1] = rng.normal(0.0, 0.1, n)
    beam.particles[:, 2] = rng.normal(0.0, 1.0, n)
    beam.particles[:, 3] = rng.normal(0.0, 0.1, n)
    beam.particles[:, 4] = rng.normal(0.0, 5.0, n)
    beam.particles[:, 5] = rng.normal(0.0, 0.01, n)
    return beam


def _macro_charge(beam):
    return ((beam.current * 1e-3) / (beam.bunch_frequency * 1e6)
            / beam.n_particles)


# ---------------------------------------------------------------- SC off
def test_sc_off_matches_precomposed_fodo():
    """SC-off stepwise tracking == the pre-composed transfer matrix."""
    ref, lat = _ref(), _fodo()
    X = np.random.default_rng(0).normal(0.0, 1.0, size=(64, 6))
    out_pre = track_beam_torch(lat, ref, X).detach().numpy()
    out_step = track_beam_torch_stepwise(lat, ref, X).detach().numpy()
    np.testing.assert_allclose(out_step, out_pre, rtol=1e-9,
                               atol=1e-10 * np.abs(out_pre).max())


@pytest.mark.skipif(not os.path.exists(BTL), reason="btl.dat not found")
def test_sc_off_matches_precomposed_btl():
    """SC-off stepwise == pre-composed on the real 960-element BTL."""
    lattice, _ = parse_tracewin(BTL)
    ref = _ref()
    X = np.random.default_rng(1).normal(0.0, 0.5, size=(32, 6))
    out_pre = track_beam_torch(lattice, ref, X,
                               on_nonlinear="ignore").detach().numpy()
    out_step = track_beam_torch_stepwise(lattice, ref, X,
                                         on_nonlinear="ignore").detach().numpy()
    np.testing.assert_allclose(out_step, out_pre, rtol=1e-7,
                               atol=1e-7 * np.abs(out_pre).max())


# ---------------------------------------------------------------- SC on
def test_sc_on_matches_numpy_tracker_fodo():
    """SC-on stepwise tracking reproduces the numpy Tracker + PicSolver.

    Both use the 2-substep cadence (no step config) and an adaptive,
    CPU-FP64 PIC, so the comparison is bit-level meaningful.
    """
    ref = _ref()
    beam = _bunched_beam()
    X0 = beam.particles.copy()
    beam_np = copy.deepcopy(beam)

    cfg = SpaceChargeConfig(nx=16, ny=16, nz=16, grid_extent=4.0,
                            use_gpu="cpu", grid_mode="adaptive")
    Simulation(_fodo(), beam_np, space_charge=cfg).run()
    assert beam_np.n_alive == beam.n_particles, "no particles should be lost"

    out = track_beam_torch_stepwise(_fodo(), ref, X0, sc_cfg=cfg,
                                    macro_charge=_macro_charge(beam))
    scale = float(np.abs(beam_np.particles).max())
    np.testing.assert_allclose(out.detach().numpy(), beam_np.particles,
                               rtol=1e-6, atol=1e-6 * scale)


def test_autograd_through_sc_tracker():
    """A scalar of the SC-on tracked beam is differentiable w.r.t. a quad
    gradient — autograd vs central finite differences (FD-limited)."""
    ref = _ref()
    beam = _bunched_beam(n=300, seed=11)
    X0 = beam.particles.copy()
    mc = _macro_charge(beam)
    cfg = SpaceChargeConfig(nx=12, ny=12, nz=12, grid_extent=4.0,
                            use_gpu="cpu", grid_mode="adaptive")
    fodo = _fodo()
    qf = next(e for e in fodo.elements if e.name == "QF")
    g0 = float(qf.gradient)

    def loss_at(grad_value, requires_grad=False):
        t = torch.tensor(grad_value, dtype=torch.float64,
                         requires_grad=requires_grad)
        out = track_beam_torch_stepwise(
            fodo, ref, X0, overrides={id(qf): t},
            sc_cfg=cfg, macro_charge=mc)
        return (out[:, 0] ** 2).mean(), t

    loss, t = loss_at(g0, requires_grad=True)
    loss.backward()
    g_auto = float(t.grad)
    assert np.isfinite(g_auto) and g_auto != 0.0

    eps = 1e-2
    lp, _ = loss_at(g0 + eps)
    lm, _ = loss_at(g0 - eps)
    g_fd = (float(lp) - float(lm)) / (2 * eps)
    np.testing.assert_allclose(g_auto, g_fd, rtol=5e-2,
                               atol=1e-3 * abs(g_fd))


def test_differentiable_step_tracker_api():
    """The DifferentiableStepTracker convenience class tracks with SC and
    backpropagates to a registered tunable."""
    ref = _ref()
    beam = _bunched_beam(n=150, seed=5)
    cfg = SpaceChargeConfig(nx=12, ny=12, nz=12, grid_extent=4.0,
                            use_gpu="cpu", grid_mode="adaptive")
    tracker = DifferentiableStepTracker(_fodo(), ref)
    params = tracker.set_tunables([("QF", "gradient")])
    out = tracker.track(beam.particles.copy(), sc_cfg=cfg,
                        macro_charge=_macro_charge(beam))
    assert out.shape == (150, 6)
    (out[:, 0] ** 2).mean().backward()
    assert params[0].tensor.grad is not None
    assert torch.isfinite(params[0].tensor.grad).all()


@pytest.mark.skipif(not os.path.exists(BTL), reason="btl.dat not found")
def test_btl_sc_on_smoke():
    """SC-on stepwise tracking runs end to end on the 960-element BTL."""
    lattice, _ = parse_tracewin(BTL)
    ref = _ref()
    beam = _bunched_beam(n=32, seed=3)
    cfg = SpaceChargeConfig(nx=8, ny=8, nz=8, grid_extent=4.0,
                            use_gpu="cpu", grid_mode="adaptive")
    out = track_beam_torch_stepwise(lattice, ref, beam.particles.copy(),
                                    sc_cfg=cfg, macro_charge=_macro_charge(beam),
                                    on_nonlinear="ignore")
    assert out.shape == (32, 6)
    assert torch.isfinite(out).all()


# ----------------------------------------------------------- M6: checkpoint
def test_checkpoint_identical_gradients():
    """Gradient checkpointing must not change the gradient — it only trades
    memory for recomputation in the backward pass."""
    ref = _ref()
    beam = _bunched_beam(n=150, seed=21)
    cfg = SpaceChargeConfig(nx=12, ny=12, nz=12, grid_extent=4.0,
                            use_gpu="cpu", grid_mode="adaptive")
    mc = _macro_charge(beam)
    x0 = beam.particles.copy()

    def grad(checkpoint):
        tr = DifferentiableStepTracker(_fodo(), ref)
        p = tr.set_tunables([("QF", "gradient")])
        out = tr.track(x0, sc_cfg=cfg, macro_charge=mc, checkpoint=checkpoint)
        (out[:, 0] ** 2).mean().backward()
        return float(p[0].tensor.grad)

    g_plain = grad(False)
    g_ckpt = grad(True)
    assert g_plain != 0.0
    np.testing.assert_allclose(g_ckpt, g_plain, rtol=1e-9, atol=1e-12)


def test_checkpoint_enables_long_lattice_backward():
    """A long lattice backpropagates with checkpointing without building
    the full O(n_kicks) autograd graph."""
    ref = _ref()
    lat = Lattice()
    for i in range(20):                       # 20 FODO cells -> 100 elements
        lat.add(Drift(f"D1_{i}", length=200.0))
        lat.add(Quadrupole(f"QF_{i}", length=100.0, gradient=8.0))
        lat.add(Drift(f"D2_{i}", length=400.0))
        lat.add(Quadrupole(f"QD_{i}", length=100.0, gradient=-8.0))
        lat.add(Drift(f"D3_{i}", length=200.0))
    beam = _bunched_beam(n=64, seed=31)
    cfg = SpaceChargeConfig(nx=8, ny=8, nz=8, grid_extent=4.0,
                            use_gpu="cpu", grid_mode="adaptive")
    tr = DifferentiableStepTracker(lat, ref)
    p = tr.set_tunables([("QF_0", "gradient")])
    out = tr.track(beam.particles.copy(), sc_cfg=cfg,
                   macro_charge=_macro_charge(beam), checkpoint=True)
    (out[:, 0] ** 2).mean().backward()
    assert p[0].tensor.grad is not None
    assert torch.isfinite(p[0].tensor.grad).all()
    assert float(p[0].tensor.grad) != 0.0


def test_step_tracker_fails_loud_on_nonlinear_element():
    """With on_nonlinear='error' the tracker raises rather than silently
    mis-tracking an element it cannot differentiate (RF gaps and field maps
    are the real concern — a Marker stands in here)."""
    from linac_gen.elements.marker import Marker
    lat = Lattice()
    lat.add(Drift("D1", length=100.0))
    lat.add(Marker(name="MK"))
    lat.add(Drift("D2", length=100.0))
    x = np.random.default_rng(0).normal(0.0, 1.0, size=(16, 6))
    with pytest.raises(TypeError, match="not a differentiable"):
        track_beam_torch_stepwise(lat, _ref(), x, on_nonlinear="error")
