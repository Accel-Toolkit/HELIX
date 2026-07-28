"""Bit-identity contract of the fused C++ field-map sampler.

Three claims pinned:

1. ``interp3_multi`` is BITWISE identical to scipy
   ``RegularGridInterpolator(method="linear", bounds_error=False,
   fill_value=0.0)`` — including out-of-bounds points (exact 0.0) and
   points exactly on grid nodes.
2. ``FieldMap3D._sample_channel_at`` returns bitwise-identical fields
   with the fused kernel enabled vs disabled (the scipy fallback).
3. End-to-end multi-particle tracking through a FieldMap3D is bitwise
   identical with the kernel enabled vs disabled.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from linac_gen.core.particle import PROTON
from linac_gen.core.reference import ReferenceParticle
from linac_gen.core.beam import Beam
from linac_gen.elements import field_map_3d as fm3_mod
from linac_gen.elements.field_map_3d import FieldMap3D
from linac_gen.io.field_map_data import FieldMapData

kernel = pytest.importorskip("linac_gen._fieldmap_kernels")


def _pillbox(n_x=11, n_y=13, n_z=51, L_mm=100.0, r_max_mm=20.0,
             E0=1.0) -> FieldMapData:
    x = np.linspace(-r_max_mm, r_max_mm, n_x)
    y = np.linspace(-r_max_mm, r_max_mm, n_y)
    z = np.linspace(0.0, L_mm, n_z)
    ez = E0 * np.cos(math.pi * z / L_mm)
    Ez = np.broadcast_to(ez[None, None, :], (n_x, n_y, n_z)).copy()
    d = np.gradient(ez, z)
    Ex = -0.5 * x[:, None, None] * np.broadcast_to(d, (n_x, n_y, n_z))
    Ey = -0.5 * y[None, :, None] * np.broadcast_to(d, (n_x, n_y, n_z))
    return FieldMapData(x=x, y=y, z=z, Ex=Ex, Ey=Ey, Ez=Ez, symmetry="3d")


def test_kernel_bitwise_vs_scipy():
    from scipy.interpolate import RegularGridInterpolator as RGI
    rng = np.random.default_rng(0)
    gx = np.linspace(-3.0, 2.0, 9)
    gy = np.linspace(0.0, 5.0, 7)
    gz = np.linspace(-1.0, 1.0, 21)
    fields = rng.normal(size=(3, 9, 7, 21))
    N = 5000
    xs = rng.uniform(-4.0, 3.0, N)          # includes out-of-bounds
    ys = rng.uniform(-1.0, 6.0, N)
    zs = rng.uniform(-1.2, 1.2, N)
    xs[:9] = gx; ys[:7] = gy[:7]; zs[:21] = gz    # exact node hits
    out = kernel.interp3_multi(xs, ys, zs, gx, gy, gz,
                               np.ascontiguousarray(fields))
    pts = np.column_stack([xs, ys, zs])
    for m in range(3):
        rgi = RGI((gx, gy, gz), fields[m], method="linear",
                  bounds_error=False, fill_value=0.0)
        np.testing.assert_array_equal(out[m], rgi(pts))
    # out-of-bounds is exactly zero
    assert out[0][np.abs(xs) > 3.5].max(initial=0.0) == 0.0


def _element():
    return FieldMap3D(name="P3D", length=100.0, field_data=_pillbox(),
                      scale=1.0, phase=0.0, frequency=352.21, n_steps=40)


def _beam(n=200, seed=3):
    rng = np.random.default_rng(seed)
    ref = ReferenceParticle(species=PROTON, w_kin=3.0, frequency=352.21)
    b = Beam(ref=ref, n_particles=n, current=0.0)
    b.particles[:, 0] = rng.normal(0, 3.0, n)
    b.particles[:, 2] = rng.normal(0, 3.0, n)
    b.particles[:, 4] = rng.normal(0, 5.0, n)
    b.particles[:, 1] = rng.normal(0, 1.0, n)
    b.particles[:, 3] = rng.normal(0, 1.0, n)
    b.particles[:, 5] = rng.normal(0, 0.001, n)
    return b


def test_sample_channel_dispatch_bitwise(monkeypatch):
    """The runtime switch flips the code path (verified with a spy) and
    both paths return bitwise-identical fields."""
    el = _element()
    assert el._fused_packs, "packs built whenever the kernel is available"
    rng = np.random.default_rng(1)
    xs = rng.uniform(-25, 25, 3000)       # includes OOB
    ys = rng.uniform(-25, 25, 3000)
    zs = np.full(3000, 42.0)
    calls = {"n": 0}
    real = fm3_mod._interp3_multi
    def spy(*a, **k):
        calls["n"] += 1
        return real(*a, **k)
    monkeypatch.setattr(fm3_mod, "_interp3_multi", spy)
    fm3_mod.use_fused_kernel(True)
    try:
        on = {ch: el._sample_channel_at(ch, ci, xs, ys, zs)
              for ch, ci in el._interpolators.items()}
        assert calls["n"] == len(el._interpolators)
        fm3_mod.use_fused_kernel(False)          # legacy scipy path
        off = {ch: el._sample_channel_at(ch, ci, xs, ys, zs)
               for ch, ci in el._interpolators.items()}
        assert calls["n"] == len(el._interpolators)   # kernel NOT called
    finally:
        fm3_mod.use_fused_kernel(True)
    for ch in on:
        for u, v in zip(on[ch], off[ch]):
            np.testing.assert_array_equal(u, v)


def test_tracking_bitwise_identical(monkeypatch):
    """Track the SAME element+beam under each sampling path and compare.

    _USE_FUSED is read at sample time, so the two loops must run under
    different switch states; a call-spy proves the kernel path really
    executed (guards against this test silently comparing scipy to
    scipy, which an earlier version did)."""
    calls = {"n": 0}
    real = fm3_mod._interp3_multi
    def spy(*a, **k):
        calls["n"] += 1
        return real(*a, **k)
    monkeypatch.setattr(fm3_mod, "_interp3_multi", spy)

    el_on, b_on = _element(), _beam()
    monkeypatch.setattr(fm3_mod, "_USE_FUSED", True)
    ds = el_on.length / el_on.n_steps
    for _ in range(el_on.n_steps):
        el_on.track_rk4(b_on, ds)
    kernel_calls = calls["n"]
    assert kernel_calls > 0, "kernel path never executed"

    el_off, b_off = _element(), _beam()
    monkeypatch.setattr(fm3_mod, "_USE_FUSED", False)
    for _ in range(el_off.n_steps):
        el_off.track_rk4(b_off, ds)
    assert calls["n"] == kernel_calls          # scipy loop: no new calls
    np.testing.assert_array_equal(b_on.particles, b_off.particles)


def test_kernel_nan_inf_and_nonfinite_fields():
    """Adversarial edges pinned: NaN coordinates win over out-of-bounds
    (scipy applies the NaN mask after the fill), Inf coordinates are
    plain OOB -> 0.0, and exact-node hits next to a non-finite map value
    reproduce scipy's 0*inf = NaN semantics (right-side index search)."""
    from scipy.interpolate import RegularGridInterpolator as RGI
    rng = np.random.default_rng(4)
    # non-uniform axes on purpose
    gx = np.cumsum(rng.uniform(0.5, 1.5, 9)); gx -= gx[0]
    gy = np.cumsum(rng.uniform(0.5, 1.5, 7)); gy -= gy[0]
    gz = np.cumsum(rng.uniform(0.5, 1.5, 11)); gz -= gz[0]
    F = rng.normal(size=(2, 9, 7, 11))
    F[1, 4, 3, 5] = np.inf                     # non-finite map value
    nan, inf = np.nan, np.inf
    xs = np.array([nan,  gx[2], inf,   gx[4], gx[4], -5.0, gx[0], gx[-1]])
    ys = np.array([100., nan,   gy[2], gy[3], gy[3], gy[1], gy[0], gy[-1]])
    zs = np.array([1.0,  1.0,   nan,   gz[5], gz[4], gz[2], gz[0], gz[-1]])
    out = kernel.interp3_multi(xs, ys, zs, gx, gy, gz,
                               np.ascontiguousarray(F))
    pts = np.column_stack([xs, ys, zs])
    for m in range(2):
        rgi = RGI((gx, gy, gz), F[m], method="linear",
                  bounds_error=False, fill_value=0.0)
        np.testing.assert_array_equal(out[m], rgi(pts))
