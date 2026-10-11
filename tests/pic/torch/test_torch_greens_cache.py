"""Torch IGF: the shared-vertex build (no gradient through dx) equals the
per-corner build bit for bit; with a gradient the per-corner graph is used, so
gradients are unchanged.  torch_pic_sc_kick reuses a solver only when no
gradient flows through the grid, the reused solver gives identical fields, and
the cache is bounded by entry count and by bytes.
"""
from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

# torch.func and torch.jit.trace call torch.jit.script internally, which
# torch 2.10 marks deprecated; that notice is torch's, not this code's.
pytestmark = pytest.mark.filterwarnings(
    "ignore:`torch.jit.script` is deprecated:DeprecationWarning")

from linac_gen.core.config import SpaceChargeConfig            # noqa: E402
from linac_gen.core.particle import H_MINUS                    # noqa: E402
from linac_gen.core.reference import ReferenceParticle         # noqa: E402
from linac_gen.pic.torch import greens as G                    # noqa: E402
from linac_gen.pic.torch import sc_kick as SK                  # noqa: E402

DX = [(0.5, 0.5, 0.5), (0.2, 0.3, 0.7), (0.071, 0.083, 0.157)]


@pytest.mark.parametrize("n", [(3, 3, 3), (4, 3, 5), (8, 8, 8), (24, 20, 32), (33, 17, 9)])
@pytest.mark.parametrize("dx", DX)
def test_vertex_equals_corners(n, dx):
    d = torch.tensor(dx, dtype=torch.float64)
    a = G._igf_torch_vertex(n, d, dtype=torch.float64, device=None)
    b = G._igf_torch_corners(n, d, dtype=torch.float64, device=None)
    assert torch.equal(a, b)


def test_vertex_equals_corners_float32():
    d = torch.tensor([0.071, 0.083, 0.157], dtype=torch.float32)
    a = G._igf_torch_vertex((16, 12, 20), d, dtype=torch.float32, device=None)
    b = G._igf_torch_corners((16, 12, 20), d, dtype=torch.float32, device=None)
    assert torch.equal(a, b)


def test_public_builder_uses_corner_graph_when_dx_requires_grad():
    d1 = torch.tensor([0.071, 0.083, 0.157], dtype=torch.float64, requires_grad=True)
    d2 = d1.detach().clone().requires_grad_(True)
    G.build_integrated_greens_torch((12, 12, 12), d1).pow(2).sum().backward()
    G._igf_torch_corners((12, 12, 12), d2, dtype=torch.float64, device=None).pow(2).sum().backward()
    assert torch.equal(d1.grad, d2.grad)


def test_public_builder_under_no_grad_matches_corners():
    d = torch.tensor([0.071, 0.083, 0.157], dtype=torch.float64, requires_grad=True)
    with torch.no_grad():
        a = G.build_integrated_greens_torch((10, 10, 10), d)
    b = G._igf_torch_corners((10, 10, 10), d.detach(), dtype=torch.float64, device=None)
    assert torch.equal(a, b)


def _kick(X, requires_grad=False, extent=5.0, macro_charge=1e-15):
    ref = ReferenceParticle(species=H_MINUS, w_kin=2.12, frequency=162.5)
    Xt = torch.tensor(X, dtype=torch.float64, requires_grad=requires_grad)
    cfg = SpaceChargeConfig(nx=16, ny=16, nz=16, grid_extent=extent, green_kind="igf", kernel="cic")
    out = SK.torch_pic_sc_kick(Xt, cfg, ds_mm=10.0, gamma=ref.gamma, beta=ref.beta,
                               mass_mev=H_MINUS.mass, wavelength_mm=ref.wavelength,
                               macro_charge=macro_charge)
    return Xt, out


def _beam(seed, n):
    return np.random.default_rng(seed).standard_normal((n, 6)) * [1.5, 1.5, 1.2, 1.5, 8.0, 0.01]


@pytest.fixture(autouse=True)
def _empty_cache():
    SK._SOLVER_CACHE.clear()
    yield
    SK._SOLVER_CACHE.clear()


def _solvers():
    return [v[0] for v in SK._SOLVER_CACHE.values()]


def test_solver_cache_hits_misses_and_is_exact():
    X = _beam(4, 4000)
    _, first = _kick(X)
    assert len(SK._SOLVER_CACHE) == 1
    solver = _solvers()[0]
    _, second = _kick(X)                                      # same grid: reused
    assert _solvers() == [solver]
    assert torch.equal(first, second)
    SK._SOLVER_CACHE.clear()
    _, fresh = _kick(X)                                       # rebuilt from scratch
    assert torch.equal(first, fresh)
    _kick(X, extent=6.0)                                      # different spacing: new entry
    assert len(SK._SOLVER_CACHE) == 2


def test_cached_result_equals_uncached_solver():
    """The kick through the cache equals the pre-cache code path exactly."""
    X = _beam(7, 3000)
    _, cached = _kick(X)
    orig = SK._solver_for
    try:
        SK._solver_for = lambda gmin, gmax, n, kind, dtype, device: SK.TorchPoissonSolverFFT(
            gmin, gmax, n, green_kind=kind, dtype=dtype, device=device)
        _, direct = _kick(X)
    finally:
        SK._solver_for = orig
    assert torch.equal(cached, direct)


def test_no_cache_when_gradient_flows_through_the_grid():
    X = _beam(5, 3000)
    Xt, out = _kick(X, requires_grad=True)
    assert len(SK._SOLVER_CACHE) == 0
    (out[:, 1] ** 2).sum().backward()
    assert Xt.grad is not None and torch.isfinite(Xt.grad).all()


@pytest.mark.filterwarnings("ignore:Converting a tensor with requires_grad=True to a scalar:UserWarning")
def test_gradient_through_charge_only_uses_cache_and_backpropagates():
    X = _beam(8, 2000)
    q = torch.tensor(1e-15, dtype=torch.float64, requires_grad=True)
    _, out = _kick(X, macro_charge=q)
    assert len(SK._SOLVER_CACHE) == 1
    (out[:, 1] ** 2).sum().backward()
    assert q.grad is not None and torch.isfinite(q.grad)


@pytest.mark.filterwarnings("ignore:Converting a tensor with requires_grad=True to a scalar:UserWarning")
def test_inference_mode_entries_kept_apart():
    X = _beam(9, 2000)
    with torch.inference_mode():
        _, a = _kick(X)
    assert len(SK._SOLVER_CACHE) == 1
    q = torch.tensor(1e-15, dtype=torch.float64, requires_grad=True)
    _, b = _kick(X, macro_charge=q)                           # normal mode: its own entry
    assert len(SK._SOLVER_CACHE) == 2
    (b[:, 1] ** 2).sum().backward()                           # no inference tensor saved
    assert torch.equal(a, b.detach())


def test_cache_is_bounded_by_count():
    X = _beam(6, 2000)
    for ext in (4.0, 4.5, 5.0, 5.5, 6.0, 6.5):
        _kick(X, extent=ext)
    assert len(SK._SOLVER_CACHE) == SK._SOLVER_CACHE_MAX


def test_cache_is_bounded_by_bytes(monkeypatch):
    X = _beam(10, 2000)
    one = 32 * 32 * 17 * 16                                   # complex128 FFT of the 16^3 doubled grid
    monkeypatch.setattr(SK, "_SOLVER_CACHE_MAX_BYTES", one)
    _kick(X, extent=4.0)
    _kick(X, extent=5.0)
    assert len(SK._SOLVER_CACHE) == 1                         # the older entry was evicted
    monkeypatch.setattr(SK, "_SOLVER_CACHE_MAX_BYTES", one - 1)
    SK._SOLVER_CACHE.clear()
    _kick(X, extent=4.0)
    assert len(SK._SOLVER_CACHE) == 0                         # larger than the cap: not kept


# ---------------------------------------------------------------------------
# Contexts in which a reused solver would be wrong (adversarial review
# 2026-10-10): forward-mode AD, torch.func transforms, tracing, a default
# device switch.  The cache must stay out of all of them.
# ---------------------------------------------------------------------------
_REF = ReferenceParticle(species=H_MINUS, w_kin=2.12, frequency=162.5)
_KW = dict(ds_mm=10.0, gamma=_REF.gamma, beta=_REF.beta, mass_mev=H_MINUS.mass,
           wavelength_mm=_REF.wavelength, macro_charge=1e-15)


def _cfg(n=8):
    return SpaceChargeConfig(nx=n, ny=n, nz=n, grid_extent=5.0, green_kind="igf", kernel="cic")


def _uncached(fn):
    """Run fn with the pre-cache code path (a fresh solver every call)."""
    orig = SK._solver_for
    SK._solver_for = lambda gmin, gmax, n, kind, dtype, device: SK.TorchPoissonSolverFFT(
        gmin, gmax, n, green_kind=kind, dtype=dtype, device=device)
    try:
        return fn()
    finally:
        SK._solver_for = orig


def test_forward_mode_ad_bypasses_cache():
    import torch.autograd.forward_ad as fwAD
    X = torch.tensor(_beam(11, 400))
    t = torch.zeros_like(X)
    t[:, 0] = 1.0                                             # stretches x: the spacing moves

    def jvp():
        with fwAD.dual_level():
            out = SK.torch_pic_sc_kick(fwAD.make_dual(X, t), _cfg(), **_KW)
            return fwAD.unpack_dual(out).tangent.clone()

    got = jvp()
    jvp()                                                     # a second call must not hit either
    assert len(SK._SOLVER_CACHE) == 0
    assert torch.equal(got, _uncached(jvp))


def test_func_jvp_bypasses_cache():
    from torch.func import jvp
    X = torch.tensor(_beam(12, 300))
    t = torch.zeros_like(X)
    t[:, 2] = X[:, 2]
    f = lambda x: SK.torch_pic_sc_kick(x, _cfg(), **_KW)      # noqa: E731
    _, tan = jvp(f, (X,), (t,))
    assert len(SK._SOLVER_CACHE) == 0
    _, ref = _uncached(lambda: jvp(f, (X,), (t,)))
    assert torch.equal(tan, ref)


@pytest.mark.filterwarnings("ignore::torch.jit.TracerWarning")
@pytest.mark.filterwarnings("ignore:`torch.jit.trace` is deprecated:DeprecationWarning")
def test_jit_trace_bypasses_cache():
    X1 = torch.tensor(_beam(13, 300))
    X2 = X1 * torch.tensor([1.7, 1.0, 0.6, 1.0, 1.3, 1.0], dtype=torch.float64)
    f = lambda x: SK.torch_pic_sc_kick(x, _cfg(), **_KW)      # noqa: E731
    f(X1)                                                     # X1's grid is now cached
    keys = list(SK._SOLVER_CACHE)
    traced = torch.jit.trace(f, (X1,), check_trace=False)
    assert list(SK._SOLVER_CACHE) == keys                     # tracing neither hit nor stored
    assert torch.equal(traced(X2), f(X2))


def test_cache_key_names_the_actual_device():
    _kick(_beam(14, 500))                                     # device=None
    assert next(iter(SK._SOLVER_CACHE))[4] == "cpu"


@pytest.mark.skipif(not (getattr(torch.backends, "mps", None) is not None
                         and torch.backends.mps.is_available()), reason="needs Apple MPS")
@pytest.mark.filterwarnings("ignore:An output with one or more elements was resized:UserWarning")
def test_default_device_switch_does_not_reuse_a_cpu_solver():
    Xc = torch.tensor(_beam(15, 300), dtype=torch.float32)
    kw = dict(_KW, dtype=torch.float32)
    SK.torch_pic_sc_kick(Xc, _cfg(), **kw)
    prev = torch.get_default_device()
    torch.set_default_device("mps")
    try:
        out = SK.torch_pic_sc_kick(Xc.to("mps"), _cfg(), **kw)
    finally:
        torch.set_default_device(prev)
    assert out.device.type == "mps"
    assert len(SK._SOLVER_CACHE) == 2


def test_free_memory_clears_the_cache():
    from linac_gen.pic.torch.solver import TorchPicSolver
    _kick(_beam(16, 300))
    assert len(SK._SOLVER_CACHE) == 1
    TorchPicSolver(SpaceChargeConfig(nx=8, ny=8, nz=8, grid_mode="adaptive")).free_memory()
    assert len(SK._SOLVER_CACHE) == 0
