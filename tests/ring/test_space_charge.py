"""Space charge in ring mode (linac_gen.ring.sc): the 2.5-D model against
analytic fields, kicks, images and tune shifts; the kernels' bookkeeping
(no self-force, momentum conservation, the same bits for any number of
threads, a restart, the CUDA simulator); the settings and their refusals."""
from __future__ import annotations

import math
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.integrate import quad
from scipy.special import erf, exp1, ive, wofz

from linac_gen.core.config import BeamConfig
from linac_gen.core.lattice import Lattice
from linac_gen.elements.drift import Drift
from linac_gen.elements.thin_lens import ThinLens
from linac_gen.ring import RingConfig, RingSimulation
from linac_gen.ring.compile import compile_ring
from linac_gen.ring.optics import twiss
from linac_gen.ring.reference import RingReference
from linac_gen.ring.sc import greens
from linac_gen.ring.sc.boundary import FittedPipe, RoundPipe, harmonic_on_grid
from linac_gen.ring.sc.config import SpaceChargeConfig
from linac_gen.ring.sc.placement import kick_positions
from linac_gen.ring.sc.solver import SpaceCharge, classical_radius

pytest.importorskip("numba")

ROOT = Path(__file__).resolve().parents[2]
BOOSTER = ROOT / "tests" / "ring" / "fixtures" / "simple_booster.madx"


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
class _OneKickRing:
    """The least a SpaceCharge needs: one element, one kick of the whole length."""

    def __init__(self, length, ref):
        self.length = float(length)
        self.reference = ref
        self.node_element = np.array([0])
        self.node_s = np.array([0.0])
        self.ref_beta0 = np.array([ref.beta0])
        self.ref_bg2 = np.array([(ref.p0c / ref.mass) ** 2])


def _beam_arrays(X):
    n = X.shape[1]
    return X, np.zeros(n), np.ones(n), np.ones(n), np.ones(n, dtype=np.int8)


def _kick_once(cfg, X, weight, *, ref, length, rf_on=True):
    """X (6, n) after one kick of the whole ``length``; returns (X, ptau)."""
    from linac_gen.ring.sc.beams import CpuBeam
    sc = SpaceCharge(cfg, _OneKickRing(length, ref), zwrap=length, rf_on=rf_on)
    X, pt, rp, rv, st = _beam_arrays(X.copy())
    sc.kick(0, CpuBeam(X, pt, rp, rv, st, np.full(X.shape[1], float(weight))))
    return X, pt, sc


def _be_field(x, y, sx, sy):
    """F = (field of a unit Gaussian line charge) 2 pi eps0, Bassetti-Erskine
    (round beam: the exact limit)."""
    if abs(sx - sy) < 1e-12 * sx:
        r2 = x * x + y * y
        with np.errstate(invalid="ignore", divide="ignore"):
            g = np.where(r2 > 0, (1 - np.exp(-r2 / (2 * sx * sx))) / r2, 1 / (2 * sx * sx))
        return g * x, g * y
    swap = sy > sx
    if swap:
        sx, sy, x, y = sy, sx, y, x
    K = 2 * (sx * sx - sy * sy)
    sk = math.sqrt(K)
    xa, ya = np.abs(x), np.abs(y)
    z1 = (xa + 1j * ya) / sk
    z2 = ((sy / sx) * xa + 1j * (sx / sy) * ya) / sk
    dec = np.exp(-xa * xa / (2 * sx * sx) - ya * ya / (2 * sy * sy))
    psi = (math.pi / math.sqrt(math.pi * K)) * (wofz(z1) - dec * wofz(z2))
    fy, fx = psi.real * np.sign(y), psi.imag * np.sign(x)
    return (fy, fx) if swap else (fx, fy)


def _gauss_cells(n, x0, dx, s):
    e = x0 + (np.arange(n + 1) - 0.5) * dx
    return 0.5 * np.diff(erf(e / (math.sqrt(2) * s)))


# ---------------------------------------------------------------------------
# settings
# ---------------------------------------------------------------------------
def test_settings_defaults_round_trip_and_text():
    c = SpaceChargeConfig()
    assert (c.model, c.grid, c.deposit, c.pipe, c.longitudinal) == ("pic", (64, 64), "tsc",
                                                                    "open", "potential")
    d = SpaceChargeConfig(intensity="8e10", grid="128x64", kicks_per_turn="50").to_dict()
    assert d["intensity"] == 8e10 and d["grid"] == [128, 64] and d["kicks_per_turn"] == 50
    assert SpaceChargeConfig.from_dict(d).to_dict() == d
    rc = RingConfig(space_charge={"intensity": 1e10})
    assert RingConfig.from_dict(rc.to_dict()).space_charge == rc.space_charge


@pytest.mark.parametrize("bad, msg", [
    ({"model": "3d"}, "model"), ({"grid": [4, 64]}, "grid"), ({"grid": [64]}, "grid"),
    ({"pipe": "round"}, "pipe_a_m"), ({"pipe": "ellipse", "pipe_a_m": 0.05}, "pipe_b_m"),
    ({"longitudinal": "g_factor"}, "g_factor"), ({"intensity": -1}, "intensity"),
    ({"intensity": 0}, "leave it out"), ({"grid": [2048, 64]}, "grid"),
    ({"intensity": True}, "intensity"), ({"kicks_per_turn": 0}, "kicks_per_turn"),
    ({"box_sigma": 0.5}, "box_sigma"), ({"z_bins": 4}, "z_bins"), ({"bogus": 1}, "unknown"),
    ({"refresh_turns": -1}, "refresh_turns"), ({"intensity": float("nan")}, "intensity"),
    ({"kicks_per_turn": 1_000_001}, "kicks_per_turn"),
    ({"refresh_turns": 2_000_000_001}, "refresh_turns"), ({"box_sigma": 2e6}, "box_sigma"),
    ({"z_smoothing_bins": 2e6}, "z_smoothing_bins"),
])
def test_settings_refused(bad, msg):
    with pytest.raises(ValueError, match=msg):
        SpaceChargeConfig.from_dict(bad)
    with pytest.raises(ValueError, match="ring config: space charge"):
        RingConfig(space_charge=bad)


# ---------------------------------------------------------------------------
# the field solve
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("ratio", [1.0, 3.0])
def test_field_of_a_gaussian_is_bassetti_erskine(ratio):
    # the solver on the exact cell charges of a Gaussian (no particle noise):
    # 1e-3 of the largest field inside 3 sigma on 128 x 128 over 8 sigma
    sy = 1e-3
    sx = ratio * sy
    n = 128
    dx, dy = 8 * sx / (n - 4), 8 * sy / (n - 4)
    x0, y0 = -(n // 2) * dx, -(n // 2) * dy
    q = np.outer(_gauss_cells(n, x0, dx, sx), _gauss_cells(n, y0, dy, sy))
    phi, fx, fy = greens.solve(q, dx, dy)
    X, Y = np.meshgrid(x0 + np.arange(n) * dx, y0 + np.arange(n) * dy, indexing="ij")
    ex, ey = _be_field(X, Y, sx, sy)
    inside = (X / sx) ** 2 + (Y / sy) ** 2 < 9
    scale = max(np.abs(ex[inside]).max(), np.abs(ey[inside]).max())
    err = max(np.abs(fx - ex)[inside].max(), np.abs(fy - ey)[inside].max()) / scale
    assert err < 1e-3
    if ratio == 1.0:
        # the potential of a round Gaussian: -(E1(r^2/2 s^2) + ln r^2)/2
        r2 = X * X + Y * Y
        with np.errstate(divide="ignore", invalid="ignore"):
            ph = -0.5 * (exp1(r2 / (2 * sx * sx)) + np.log(r2))
        ok = inside & (r2 > 0)
        span = np.ptp(ph[ok])
        assert np.abs((phi - ph)[ok]).max() / span < 1e-3


def test_field_converges_as_the_cell_size_squared():
    errs = []
    for n in (32, 64, 128):
        s = 1e-3
        d = 8 * s / (n - 4)
        x0 = -(n // 2) * d
        q = np.outer(_gauss_cells(n, x0, d, s), _gauss_cells(n, x0, d, s))
        _phi, fx, _fy = greens.solve(q, d, d, potential=False)
        X, Y = np.meshgrid(x0 + np.arange(n) * d, x0 + np.arange(n) * d, indexing="ij")
        ex, _ey = _be_field(X, Y, s, s)
        rr = X ** 2 + Y ** 2
        inside = rr < 9 * s * s
        errs.append(np.abs(fx - ex)[inside].max() / np.abs(ex[inside]).max())
    assert 3.3 < errs[0] / errs[1] < 5.0 and 3.3 < errs[1] / errs[2] < 5.0


def test_field_greens_function_is_exactly_antisymmetric():
    gp, gx, gy = greens.cell_greens(16, 12, 1e-4, 3e-4)
    # offsets (m, n) and (-m, -n): index k and (-k) mod size
    rx = np.roll(gx[::-1, ::-1], 1, axis=(0, 1))
    ry = np.roll(gy[::-1, ::-1], 1, axis=(0, 1))
    rp = np.roll(gp[::-1, ::-1], 1, axis=(0, 1))
    m = np.ones_like(gx, dtype=bool)
    m[16, :] = m[:, 12] = False                 # (the unused offset -n)
    assert np.array_equal(gx[m], -rx[m]) and np.array_equal(gy[m], -ry[m])
    assert np.array_equal(gp[m], rp[m])


@pytest.mark.parametrize("deposit", ["tsc", "cic"])
def test_a_particle_alone_feels_no_kick(deposit):
    ref = RingReference.from_kinetic("proton", 800.0)
    rng = np.random.default_rng(4)
    for _ in range(5):
        X = np.zeros((6, 1))
        X[0], X[2] = rng.normal(0, 1e-3, 2)
        X1, _pt, sc = _kick_once({"deposit": deposit, "grid": [16, 16], "kicks_per_turn": 1},
                                 X, 1e10, ref=ref, length=10.0, rf_on=False)
        # (rounding of the FFT only: against the kick its charge gives a
        # particle one cell away)
        k = 2 * classical_radius(ref.mass) * 1e10 * sc.ds[0] / (ref.beta0 ** 2 * ref.gamma0 ** 3)
        scale = k / 10.0 / min(sc.last_box.dx, sc.last_box.dy)
        assert abs(X1[1, 0]) < 1e-12 * scale and abs(X1[3, 0]) < 1e-12 * scale


@pytest.mark.parametrize("deposit", ["tsc", "cic"])
def test_kicks_conserve_the_bunchs_transverse_momentum(deposit):
    ref = RingReference.from_kinetic("proton", 800.0)
    rng = np.random.default_rng(7)
    X = np.zeros((6, 4000))
    X[0] = rng.normal(0, 2e-3, 4000)
    X[2] = rng.normal(0, 1e-3, 4000)
    X1, _pt, sc = _kick_once({"deposit": deposit, "grid": [32, 32], "kicks_per_turn": 1,
                              "box_sigma": 10.0}, X, 1e7, ref=ref, length=10.0, rf_on=False)
    assert sc.n_outside_max == 0
    dpx, dpy = X1[1] - X[1], X1[3] - X[3]
    assert abs(dpx.sum()) < 1e-12 * np.abs(dpx).sum()
    assert abs(dpy.sum()) < 1e-12 * np.abs(dpy).sum()


def test_one_kick_is_the_analytic_kick_of_a_long_gaussian_bunch():
    # J. Qiang, Phys. Rev. Accel. Beams 28, 114602 (2025), eqs. 31-32: a cold
    # round Gaussian bunch, one kick; the reference code's own test of this
    # case asks 2e-2 (transverse) and 0.2 (longitudinal), rms over the bunch
    ref = RingReference.from_kinetic("proton", 100.0)
    n, sig, sz, L = 400_000, 4e-5, 0.5, 10.0
    rng = np.random.default_rng(1)
    X = np.zeros((6, n))
    X[0], X[2] = rng.normal(0, sig, n), rng.normal(0, sig, n)
    X[4] = np.clip(rng.normal(0, sz, n), -L / 2 + 1e-9, L / 2 - 1e-9)
    N = 6.241509e9                                   # 1 nC
    X1, pt, sc = _kick_once({"grid": [96, 96], "kicks_per_turn": 1, "z_bins": 256,
                             "z_smoothing_bins": 2.0}, X, N / n, ref=ref, length=L)
    b0, g0 = ref.beta0, ref.gamma0
    k = 2 * classical_radius(ref.mass) * N * sc.ds[0]
    x, y, z = X[0], X[2], X[4]
    r2 = x * x + y * y
    f = np.exp(-z * z / (2 * sz * sz)) / (math.sqrt(2 * math.pi) * sz)
    want_px = k * f * x * (1 - np.exp(-r2 / (2 * sig ** 2))) / r2 / (b0 ** 2 * g0 ** 3)
    phi = -0.5 * (exp1(r2 / (2 * sig ** 2)) + np.log(r2))      # (zero at 1 m)
    want_pt = -k * (-z / sz ** 2 * f) * phi / (b0 * g0 ** 3)
    got_px = X1[1] - X[1]
    assert np.sqrt(np.mean((got_px - want_px) ** 2)) / np.abs(want_px).max() < 1e-2
    assert np.sqrt(np.mean((pt - want_pt) ** 2)) / np.abs(want_pt).max() < 5e-2


# ---------------------------------------------------------------------------
# the pipe
# ---------------------------------------------------------------------------
def test_round_pipe_images_are_exact():
    # node charges are point charges at the nodes; their images in a grounded
    # circle of radius b: -q at b^2/conj(w)
    b = 0.05
    rng = np.random.default_rng(2)
    X, Y = np.meshgrid(np.linspace(-0.01, 0.012, 9), np.linspace(-0.008, 0.006, 7),
                       indexing="ij")
    q = rng.random(X.shape)
    q /= q.sum()
    pipe = RoundPipe(b)
    c = pipe.coefficients(q, X, Y)
    px, py = rng.uniform(-0.03, 0.03, 50), rng.uniform(-0.03, 0.03, 50)
    h, hx, hy = harmonic_on_grid(c, pipe.scale, pipe.centre, px, py, potential=True)
    w = X.ravel() + 1j * Y.ravel()
    img = b * b / np.conj(w)
    z = px + 1j * py
    # the image charges' potential (-q at img: +q ln|z - img|) and constant
    want_h = np.array([np.sum(q.ravel() * (np.log(np.abs(zz - img)) + np.log(np.abs(w) / b)))
                       for zz in z])
    e = np.array([np.sum(-q.ravel() / (zz - img)) for zz in z])     # F_x - i F_y
    assert np.allclose(h, want_h, rtol=0, atol=1e-12 * np.abs(want_h).max())
    assert np.allclose(hx - 1j * hy, e, rtol=0, atol=1e-11 * np.abs(e).max())
    # and on the wall the free-space potential plus h is zero
    t = np.linspace(0, 2 * np.pi, 13)
    zw = b * np.exp(1j * t)
    hw, _, _ = harmonic_on_grid(c, pipe.scale, pipe.centre, zw.real, zw.imag, potential=True)
    free = np.array([-np.sum(q.ravel() * np.log(np.abs(zz - w))) for zz in zw])
    assert np.abs(free + hw).max() < 1e-12


def _rect_potential(x, y, x0, y0, a, b, nmax=400):
    """Phi (dimensionless: -ln r near the charge) of a unit line charge at
    (x0, y0) in the grounded rectangle |x| < a, |y| < b (Fourier series in x)."""
    out = np.zeros_like(x, dtype=float)
    ylo, yhi = np.minimum(y, y0), np.maximum(y, y0)
    for k in range(1, nmax + 1):
        kk = k * math.pi / (2 * a)
        sx = math.sin(kk * (x0 + a)) * np.sin(kk * (x + a))
        # sinh(kk (ylo + b)) sinh(kk (b - yhi)) / (kk sinh(2 kk b)), without overflow
        g = (np.exp(-kk * (yhi - ylo)) * (1 - np.exp(-2 * kk * (ylo + b)))
             * (1 - np.exp(-2 * kk * (b - yhi))) / (2 * kk * (1 - math.exp(-4 * kk * b))))
        out += sx * g
    return out * (2 * math.pi / a)


@pytest.mark.parametrize("shape", ["rectangle", "ellipse"])
def test_fitted_pipe_grounds_the_wall(shape):
    a, b = 0.06, 0.025
    s = 3e-3
    n = 48
    d = 8 * s / (n - 4)
    x0 = -(n // 2) * d
    X, Y = np.meshgrid(x0 + np.arange(n) * d, x0 + np.arange(n) * d, indexing="ij")
    q = np.outer(_gauss_cells(n, x0, d, s), _gauss_cells(n, x0, d, s))
    pipe = FittedPipe(shape, a, b, 24)

    def wall(px, py):
        return greens.potential_at(px, py, X, Y, q, d, d)
    c = pipe.coefficients(q, X, Y, potential_at=wall)
    # 400 other points on the wall: free potential + harmonic close to zero
    from linac_gen.ring.sc.boundary import _wall_points
    px, py = _wall_points(shape, a, b, 400)
    hw, _, _ = harmonic_on_grid(c, pipe.scale, pipe.centre, px, py, potential=True)
    free = wall(px, py)
    assert np.abs(free + hw).max() < 2e-3 * np.ptp(free)


def test_rectangular_pipe_field_matches_the_exact_series():
    # a narrow beam off centre in a rectangle: the fitted harmonic's field at
    # the beam against the image field from the exact series solution
    a, b = 0.05, 0.03
    xq, yq = 0.008, -0.004
    n = 24
    d = 2e-4
    x0, y0 = xq - (n // 2) * d, yq - (n // 2) * d
    X, Y = np.meshgrid(x0 + np.arange(n) * d, y0 + np.arange(n) * d, indexing="ij")
    q = np.zeros((n, n))
    q[n // 2, n // 2] = 1.0                        # a point charge at a node
    pipe = FittedPipe("rectangle", a, b, 32)
    c = pipe.coefficients(q, X, Y, potential_at=lambda px, py: -0.5 * np.log(
        (px - xq) ** 2 + (py - yq) ** 2))
    pts_x = np.array([0.0, 0.01, -0.02, 0.03])
    pts_y = np.array([0.0, -0.01, 0.015, 0.02])
    h, hx, hy = harmonic_on_grid(c, pipe.scale, pipe.centre, pts_x, pts_y, potential=True)
    exact = _rect_potential(pts_x, pts_y, xq, yq, a, b)
    free = -0.5 * np.log((pts_x - xq) ** 2 + (pts_y - yq) ** 2)
    want_h = exact - free
    assert np.abs(h - want_h).max() < 1e-9 * np.abs(free).max()          # (6e-11 measured)
    # and the image field, by differences of the exact series
    e = 1e-6
    ex = -(_rect_potential(pts_x + e, pts_y, xq, yq, a, b) - _rect_potential(
        pts_x - e, pts_y, xq, yq, a, b)) / (2 * e)
    fx_free = (pts_x - xq) / ((pts_x - xq) ** 2 + (pts_y - yq) ** 2)
    assert np.abs(hx - (ex - fx_free)).max() < 1e-6 * np.abs(fx_free).max()  # (1e-8: the differences)


def test_an_ellipse_with_equal_axes_is_the_round_pipe():
    b = 0.04
    rng = np.random.default_rng(9)
    X, Y = np.meshgrid(np.linspace(-0.006, 0.006, 13), np.linspace(-0.006, 0.006, 13),
                       indexing="ij")
    q = np.exp(-(X ** 2 + Y ** 2) / (2 * 0.002 ** 2))
    q /= q.sum()
    ell = FittedPipe("ellipse", b, b, 24)
    w = X.ravel() + 1j * Y.ravel()
    c_e = ell.coefficients(q, X, Y, potential_at=lambda px, py: np.array([
        -np.sum(q.ravel() * np.log(np.abs(p - w))) for p in px + 1j * py]))
    c_r = RoundPipe(b).coefficients(q, X, Y)
    px, py = rng.uniform(-0.02, 0.02, 20), rng.uniform(-0.02, 0.02, 20)
    he, ex, ey = harmonic_on_grid(c_e, ell.scale, ell.centre, px, py, potential=True)
    hr, rx, ry = harmonic_on_grid(c_r, b, (0.0, 0.0), px, py, potential=True)
    assert np.abs(he - hr).max() < 1e-9
    assert np.abs(ex - rx).max() < 1e-7 * np.abs(rx).max()


# ---------------------------------------------------------------------------
# the longitudinal kick
# ---------------------------------------------------------------------------
def test_longitudinal_kick_of_a_uniform_beam_in_a_round_pipe():
    # a uniform round beam (radius a) in a pipe (radius b): Phi = (1 - r^2/a^2)/2
    # + ln(b/a) inside, g = 2 Phi(0) = 1 + 2 ln(b/a) on the axis
    ref = RingReference.from_kinetic("proton", 400.0)
    n, a, bp, sz, L = 400_000, 2e-3, 0.03, 0.6, 12.0
    rng = np.random.default_rng(3)
    r = a * np.sqrt(rng.random(n))
    t = 2 * np.pi * rng.random(n)
    X = np.zeros((6, n))
    X[0], X[2] = r * np.cos(t), r * np.sin(t)
    X[4] = np.clip(rng.normal(0, sz, n), -L / 2 + 1e-9, L / 2 - 1e-9)
    N = 1e11
    cfg = {"grid": [64, 64], "kicks_per_turn": 1, "z_bins": 256, "z_smoothing_bins": 2.0,
           "pipe": "round", "pipe_a_m": bp}
    _X1, pt, sc = _kick_once(cfg, X, N / n, ref=ref, length=L)
    b0, g0 = ref.beta0, ref.gamma0
    k = 2 * classical_radius(ref.mass) * N * sc.ds[0] / (b0 * g0 ** 3)
    z = X[4]
    # (the line density as the model makes it: the Gaussian bunch widened by
    # the filter and by the linear deposit and interpolation, dz^2/6 each)
    dz = L / 256
    se = math.sqrt(sz ** 2 + (2.0 * dz) ** 2 + dz ** 2 / 3)
    fp = -z / se ** 2 * np.exp(-z * z / (2 * se * se)) / (math.sqrt(2 * math.pi) * se)
    phi = 0.5 * (1 - r * r / (a * a)) + math.log(bp / a)
    want = -k * fp * phi
    core = (r < 0.8 * a) & (np.abs(z) < 2 * sz)
    ratio = np.sum(pt[core] * want[core]) / np.sum(want[core] ** 2)
    assert ratio == pytest.approx(1.0, abs=0.01)                          # (1.0014 measured)
    # the g-factor model gives every particle the axis value
    g = 1 + 2 * math.log(bp / a)
    _X2, pt_g, _ = _kick_once(dict(cfg, longitudinal="g_factor", g_factor=g), X, N / n, ref=ref,
                              length=L)
    want_g = -k * fp * (g / 2)
    ratio_g = np.sum(pt_g * want_g) / np.sum(want_g ** 2)
    assert ratio_g == pytest.approx(1.0, abs=0.01)                        # (1.0018)
    # and none when it is off
    _X3, pt_off, _ = _kick_once(dict(cfg, longitudinal="off"), X, N / n, ref=ref, length=L)
    assert not np.any(pt_off)


def test_a_coasting_beam_has_a_uniform_line_density_and_no_longitudinal_kick():
    ref = RingReference.from_kinetic("proton", 800.0)
    rng = np.random.default_rng(5)
    n, L = 5000, 100.0
    X = np.zeros((6, n))
    X[0], X[2] = rng.normal(0, 2e-3, n), rng.normal(0, 2e-3, n)
    X[4] = rng.uniform(-L / 2, L / 2, n)
    X1, pt, sc = _kick_once({"kicks_per_turn": 1}, X, 1e8, ref=ref, length=L, rf_on=False)
    assert not sc.bunched and not np.any(pt)
    # the transverse kick: the same for particles at the same (x, y) wherever in zeta
    Y = X.copy()
    Y[4] = 0.0
    Y1, _pt, _sc = _kick_once({"kicks_per_turn": 1}, Y, 1e8, ref=ref, length=L, rf_on=False)
    assert np.array_equal(X1[1], Y1[1]) and np.array_equal(X1[3], Y1[3])


# ---------------------------------------------------------------------------
# tune shifts (L. J. Laslett 1963): a round beam in a ring of identical cells
# focusing both planes alike, so beta_x = beta_y everywhere and the integral
# of beta / (sigma_x (sigma_x + sigma_y)) is C / (2 eps) exactly, kicks or not
# ---------------------------------------------------------------------------
W_LAS, EPS_LAS = 1000.0, 1e-6


@pytest.fixture(scope="module")
def round_ring():
    ref = RingReference.from_kinetic("proton", W_LAS)
    ncell, lc, mu = 40, 2.0, math.radians(9.9)
    f = lc / (2 * (1 - math.cos(mu)))
    lat = Lattice()
    for i in range(ncell):
        lat.add(Drift(name=f"d{i}a", length=lc / 2 * 1e3))
        lat.add(ThinLens(name=f"l{i}", fx=f, fy=f))
        lat.add(Drift(name=f"d{i}b", length=lc / 2 * 1e3))
    rl = compile_ring(lat, ref, lattice_energy=W_LAS)
    return lat, rl, twiss(rl, chromaticity=False), ref


def _round_beam(o, rl, kind, n, seed):
    rng = np.random.default_rng(seed)
    g = rng.standard_normal((4, n))
    if kind == "kv":
        g = g / np.linalg.norm(g, axis=0) * math.sqrt(4 * EPS_LAS)   # edge emittance 4 eps
    else:
        g = g * math.sqrt(EPS_LAS)
    bx, ax, by, ay = (o.start[k] for k in ("betx", "alfx", "bety", "alfy"))
    X = np.zeros((6, n))
    X[0], X[1] = math.sqrt(bx) * g[0], (g[1] - ax * g[0]) / math.sqrt(bx)
    X[2], X[3] = math.sqrt(by) * g[2], (g[3] - ay * g[2]) / math.sqrt(by)
    X[4] = (rng.random(n) - 0.5) * rl.length
    return X


def _detuning(ax, ay):
    """A round Gaussian beam's tune shift at amplitudes (ax, ay) sigma over its
    centre's: ∫0^1 exp(-(a1+a2)u)(I0 - I1)(a1 u) I0(a2 u) du, a = amplitude^2/4."""
    a1, a2 = ax * ax / 4, ay * ay / 4
    return quad(lambda u: (ive(0, a1 * u) - ive(1, a1 * u)) * ive(0, a2 * u), 0, 1)[0]


def _tune_shift_run(round_ring, kind, model, n, grid, amp, monkeypatch):
    import linac_gen.ring.engine as E
    lat, rl, o, ref = round_ring
    monkeypatch.setattr(E, "PROBE_SMALL_SIGMA_SC", amp)
    N = 1.1e12 if kind == "kv" else 0.55e12
    beam = BeamConfig(species="proton", energy=W_LAS, frequency=100.0, current=0.0,
                      emit_nx=1, emit_ny=1, emit_z=1, beta_z=100)
    res = RingSimulation(lat, beam, RingConfig(
        n_turns=256, n_particles=n, tune_probes=1, stats_every=256,
        space_charge={"model": model, "intensity": N, "kicks_per_turn": 20,
                      "grid": [grid, grid], "refresh_turns": 0}),
        coords=_round_beam(o, rl, kind, n, 1), ring_lattice=rl).run()
    b2g3 = ref.beta0 ** 2 * ref.gamma0 ** 3
    dq0 = -(1.0 if kind == "kv" else 2.0) * classical_radius(ref.mass) * N / (
        8 * math.pi * b2g3 * EPS_LAS)
    want = dq0 if kind == "kv" else dq0 * _detuning(amp, amp)
    return (res.tunes["qx"] - o.qx % 1) / want, (res.tunes["qy"] - o.qy % 1) / want, res


@pytest.mark.slow
def test_laslett_tune_shift_of_a_kv_beam(round_ring, monkeypatch):
    # inside a KV beam the force is linear: every particle has the same shift
    rx, ry, res = _tune_shift_run(round_ring, "kv", "pic", 100_000, 32, 0.5, monkeypatch)
    assert rx == pytest.approx(1.0, abs=0.02) and ry == pytest.approx(1.0, abs=0.02)
    assert res.meta["space_charge"]["laslett_gaussian"] is not None


@pytest.mark.slow
def test_laslett_tune_shift_of_a_gaussian_beam(round_ring, monkeypatch):
    rx, ry, _res = _tune_shift_run(round_ring, "gauss", "pic", 100_000, 64, 0.2, monkeypatch)
    assert 0.5 * (rx + ry) == pytest.approx(1.0, abs=0.03)
    fx, fy, _res = _tune_shift_run(round_ring, "gauss", "frozen", 20_000, 64, 0.1, monkeypatch)
    assert fx == pytest.approx(1.0, abs=0.02) and fy == pytest.approx(1.0, abs=0.02)


def test_analytic_tune_shift_estimate_of_the_round_ring(round_ring):
    from linac_gen.ring.sc.laslett import tune_shift
    _lat, rl, o, ref = round_ring
    N = 1e12
    dq = tune_shift(o, emit_x=EPS_LAS, emit_y=EPS_LAS, intensity=N, line_density=1 / rl.length,
                    mass_mev=ref.mass)
    want = -2 * classical_radius(ref.mass) * N / (8 * math.pi * ref.beta0 ** 2
                                                 * ref.gamma0 ** 3 * EPS_LAS)
    assert dq[0] == pytest.approx(want, rel=1e-9) and dq[1] == pytest.approx(want, rel=1e-9)


# ---------------------------------------------------------------------------
# the kicks in a run
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def booster():
    from linac_gen.cli.common import load_lattice
    return load_lattice(str(BOOSTER))


def _booster_beam():
    return BeamConfig(species="proton", energy=800.0, frequency=44.7044096, current=0.0,
                      emit_nx=1.5, emit_ny=1.5, emit_z=5.0, beta_z=100.0)


def test_kicks_sit_at_element_boundaries(booster):
    from linac_gen.ring.compile import compile_from_config
    rl = compile_from_config(booster, RingConfig(), beam=_booster_beam())
    nodes, s, ds = kick_positions(rl, 96)
    assert len(nodes) == 96 and np.all(np.diff(nodes) > 0)
    assert ds.sum() == pytest.approx(rl.length, rel=1e-12)
    ne = rl.node_element
    assert all(j == 0 or ne[j] != ne[j - 1] for j in nodes)
    assert np.array_equal(s, rl.node_s[nodes])
    n1, s1, ds1 = kick_positions(rl, 1)
    assert len(n1) == 1 and ds1[0] == rl.length


SC_BOOSTER = {"intensity": 2e10, "kicks_per_turn": 48}


def test_same_bits_for_any_number_of_threads(booster):
    out = []
    for th in (1, 4):
        cfg = RingConfig(n_turns=4, n_particles=4000, tune_probes=1, threads=th,
                         space_charge=SC_BOOSTER, snapshot_turns=["last"])
        out.append(RingSimulation(booster, _booster_beam(), cfg).run())
    a, b = out
    assert np.array_equal(a.cov, b.cov)
    assert np.array_equal(a.snapshots[4]["coords"], b.snapshots[4]["coords"])
    assert np.array_equal(a.probes["tbt"], b.probes["tbt"], equal_nan=True)
    m = a.meta["space_charge"]
    assert m["kicks"] == 48 and m["bunched"] and m["longitudinal_kick"]


@pytest.mark.parametrize("model", ["pic", "frozen"])
def test_restart_with_space_charge_is_bit_for_bit(booster, tmp_path, model):
    sc = dict(SC_BOOSTER, model=model, refresh_turns=2)
    cfg = RingConfig(n_turns=6, n_particles=3000, tune_probes=1, space_charge=sc,
                     snapshot_turns=["last"], checkpoint_every=3)
    ck = tmp_path / "ck.npz"
    full = RingSimulation(booster, _booster_beam(), cfg).run(checkpoint_path=ck)
    cont = RingSimulation(booster, _booster_beam(), cfg).run(restart_from=ck)
    assert np.array_equal(full.cov, cont.cov)
    assert np.array_equal(full.snapshots[6]["coords"], cont.snapshots[6]["coords"])
    other = RingConfig(n_turns=6, n_particles=3000, tune_probes=1,
                       space_charge=dict(sc, intensity=3e10), checkpoint_every=3)
    with pytest.raises(ValueError, match="space-charge settings"):
        RingSimulation(booster, _booster_beam(), other).run(restart_from=ck)


def _core(z):
    return 1.4826 * float(np.median(np.abs(z - np.median(z))))


def test_a_bunch_the_line_density_does_not_resolve_gets_a_note(booster):
    """The shortest bunch core over the statistics rows (1.4826 times the
    median absolute deviation of zeta: a halo does not hide it) against the
    line density's bins and filter (a Gaussian of rms sigma looks like one of
    rms sqrt(sigma^2 + filter^2 + dz^2/3)): with 16 bins over the 5.6 m RF
    period the run notes how much of a Gaussian's peak — and of the
    space-charge kick at its centre — they keep, with the core's length and
    the bins; with 1024 bins it does not; the record keeps both
    (meta["space_charge"]["line_density_resolution"]).  A 3 cm core in a 5 %
    halo spread over the period, 256 bins: noted (its rms, 37 cm, would not
    be).  A coasting beam has no line density: no record."""
    from linac_gen.ring.compile import compile_from_config
    out = {}
    for nz in (16, 1024):
        cfg = RingConfig(n_turns=2, n_particles=1000, tune_probes=0, snapshot_turns=[0, 1, 2],
                         space_charge=dict(SC_BOOSTER, kicks_per_turn=8, z_bins=nz))
        out[nz] = RingSimulation(booster, _booster_beam(), cfg).run()
    lr = {nz: r.meta["space_charge"]["line_density_resolution"] for nz, r in out.items()}
    rl = compile_from_config(booster, RingConfig(), beam=_booster_beam())
    for nz in (16, 1024):
        sz, dz = lr[nz]["zeta_core_rms_m"], rl.length / rl.harmonic / nz
        assert sz == pytest.approx(min(_core(sn["coords"][4]) for sn in out[nz].snapshots.values()),
                                   rel=1e-12)
        assert lr[nz]["bins_per_sigma"] == pytest.approx(sz / dz, rel=1e-12)
        assert lr[nz]["peak_ratio"] == pytest.approx(
            sz / math.sqrt(sz * sz + dz * dz + dz * dz / 3.0), rel=1e-12)
    assert lr[16]["peak_ratio"] < 0.97 < 0.999 < lr[1024]["peak_ratio"]
    noted = [n for n in out[16].meta["notes"] if "line-density bins" in n]
    assert len(noted) == 1 and "more z_bins (with enough particles in each)" in noted[0]
    assert f"about {lr[16]['peak_ratio'] * 100:.0f} % of the peak of a Gaussian" in noted[0]
    assert not any("line-density bins" in n for n in out[1024].meta["notes"])
    # a short core in a halo: the core is what is judged
    rng = np.random.default_rng(3)
    n, period = 4000, rl.length / rl.harmonic
    X0 = np.zeros((6, n))
    X0[0], X0[2] = rng.normal(0.0, 1e-3, n), rng.normal(0.0, 1e-3, n)
    X0[4] = rng.normal(0.0, 0.03, n)
    halo = rng.random(n) < 0.05
    X0[4, halo] = (rng.random(int(halo.sum())) - 0.5) * period
    X0[5] = rng.normal(0.0, 1e-4, n)
    cfg = RingConfig(n_turns=1, n_particles=n, tune_probes=0,
                     space_charge=dict(SC_BOOSTER, kicks_per_turn=8, z_bins=256))
    res = RingSimulation(booster, _booster_beam(), cfg, coords=X0).run()
    h = res.meta["space_charge"]["line_density_resolution"]
    assert h["zeta_core_rms_m"] == pytest.approx(0.03, rel=0.1)
    assert math.sqrt(res.cov[0][4, 4]) > 0.3                 # (the rms: the halo's)
    assert h["peak_ratio"] < 0.9 and any("line-density bins" in m for m in res.meta["notes"])
    # a coasting beam
    cfg = RingConfig(n_turns=1, n_particles=500, tune_probes=0, zeta_wrap="off",
                     space_charge=dict(SC_BOOSTER, kicks_per_turn=8, z_bins=16))
    coast = RingSimulation(booster, _booster_beam(), cfg, ring_lattice=rl.with_rf(False)).run()
    assert not coast.meta["space_charge"]["bunched"]
    assert "line_density_resolution" not in coast.meta["space_charge"]
    assert not any("line-density bins" in n for n in coast.meta["notes"])


def test_the_peak_fraction_kept_is_the_codes_own_line_density(booster):
    """SpaceCharge.peak_ratio against the line density the kicks use: a
    Gaussian bunch (noise-free: weighted points on a fine grid) deposited by
    the space charge's own kernel, filtered, and gathered at the bunch's
    centre, its value averaged over where the centre falls in a bin — within
    1 % of the formula for bunches 1 to 5 bins long and filters of 0, 1 and 3
    bins."""
    from linac_gen.ring.compile import compile_from_config
    from linac_gen.ring.sc import kernels as K
    from linac_gen.ring.sc.solver import Box, SpaceCharge
    rl = compile_from_config(booster, RingConfig(), beam=_booster_beam())
    period, nz = rl.length / rl.harmonic, 64
    deposit = K.sc_backend()[1]
    for sb, r in ((1.0, 1.0), (1.0, 2.0), (1.0, 5.0), (0.0, 2.0), (3.0, 2.0)):
        sc = SpaceCharge({"intensity": 1e10, "kicks_per_turn": 8, "z_bins": nz,
                          "z_smoothing_bins": sb}, rl, zwrap=period, rf_on=True)
        sigma = r * sc.dz
        kept = []
        for off in (0.0, 0.25, 0.5):
            zc = -0.5 * period + (nz // 2 + 0.5 + off) * sc.dz
            z = np.linspace(zc - 8.0 * sigma, zc + 8.0 * sigma, 200001)
            w = np.exp(-0.5 * ((z - zc) / sigma) ** 2)
            w /= w.sum()
            coords = np.zeros((6, z.size))
            coords[4] = z
            kp, ki = sc._params(0, Box(sc.nx, sc.ny, 0.0, 0.0, 1.0, 1.0), (0.0, 0.0), 1.0)
            grid = np.zeros((1, sc.nx + 3, sc.ny + 3), dtype=np.int64)
            line = np.zeros((1, nz), dtype=np.int64)
            scale = 2.0 ** 52
            deposit(coords, w, np.ones(z.size, dtype=np.int8), z.size, kp, ki, scale, grid, line,
                    np.zeros(1, dtype=np.int64), False)
            fl, _ = sc.line_density(line.sum(axis=0), scale, 1.0)
            k0, k1, w0, w1 = K.line_cell(zc, kp[K.KP_Z0], kp[K.KP_DZ], nz)
            kept.append((w0 * fl[k0] + w1 * fl[k1]) * math.sqrt(2.0 * math.pi) * sigma)
        assert np.mean(kept) == pytest.approx(sc.peak_ratio(sigma), rel=1e-2), (sb, r, kept)


def test_space_charge_needs_an_intensity(booster):
    cfg = RingConfig(n_turns=1, n_particles=100, space_charge={"kicks_per_turn": 4})
    with pytest.raises(ValueError, match="intensity"):
        RingSimulation(booster, _booster_beam(), cfg).run()


def test_a_bunched_beam_needs_zeta_wrapped(booster):
    cfg = RingConfig(n_turns=1, n_particles=100, zeta_wrap="off",
                     space_charge={"intensity": 1e10, "kicks_per_turn": 4})
    with pytest.raises(ValueError, match="wrapped"):
        RingSimulation(booster, _booster_beam(), cfg).run()


def test_space_charge_results_round_trip_through_hdf5(booster, tmp_path):
    from linac_gen.ring.results import load_ring_hdf5, save_ring_hdf5
    cfg = RingConfig(n_turns=2, n_particles=500, tune_probes=0, space_charge=SC_BOOSTER)
    res = RingSimulation(booster, _booster_beam(), cfg).run()
    p = tmp_path / "r.h5"
    save_ring_hdf5(res, p)
    back = load_ring_hdf5(p)
    assert back.meta["space_charge"]["kicks"] == 48
    assert back.meta["config"]["space_charge"]["intensity"] == 2e10


def test_ramp_with_space_charge_uses_each_kicks_reference(booster, tmp_path):
    ramp = {"energy": {"kind": "table", "times_s": [0.0, 1e-4], "values": [800.0, 806.0]},
            "voltage": {"kind": "table", "times_s": [0.0, 1.0], "values_mv": [1.2, 1.2]},
            "gamma_t": 5.449167323}
    cfg = RingConfig(n_turns=6, n_particles=2000, tune_probes=0, ramp=ramp,
                     space_charge=SC_BOOSTER, checkpoint_every=3)
    ck = tmp_path / "ck.npz"
    full = RingSimulation(booster, _booster_beam(), cfg).run(checkpoint_path=ck)
    assert full.ramp["kinetic_mev"][-1] > 800.0
    cont = RingSimulation(booster, _booster_beam(), cfg).run(restart_from=ck)
    assert np.array_equal(full.cov, cont.cov)


# ---------------------------------------------------------------------------
# the CUDA kernels (numba's simulator on a machine without a GPU)
# ---------------------------------------------------------------------------
def test_cuda_space_charge_kernels_in_the_simulator_match_the_cpu():
    code = r"""
import numpy as np
from numba import cuda
from linac_gen.ring.reference import RingReference
from linac_gen.ring.sc.beams import CpuBeam, CudaBeam
from linac_gen.ring.sc.solver import SpaceCharge, combine_moments, half_widths, make_box, \
    _power_of_two_scale
class R:
    def __init__(s, L, ref):
        s.length = L; s.reference = ref; s.node_element = np.array([0]); s.node_s = np.array([0.0])
        s.ref_beta0 = np.array([ref.beta0]); s.ref_bg2 = np.array([(ref.p0c / ref.mass) ** 2])
ref = RingReference.from_kinetic("proton", 800.0)
rl = R(5.0, ref)
n = 300
rng = np.random.default_rng(2)
X = np.zeros((6, n)); X[0] = rng.normal(0, 2e-3, n); X[2] = rng.normal(0, 1e-3, n)
X[4] = np.clip(rng.normal(0, 0.5, n), -2.49, 2.49); X[0, :2] = [0.02, -0.03]
worst = 0.0
for model in ("pic", "frozen"):
    out = []
    for kind in ("cpu", "cuda"):
        Y = X.copy(); arrs = [Y, np.zeros(n), np.ones(n), np.ones(n), np.ones(n, np.int8)]
        sc = SpaceCharge({"model": model, "grid": [16, 16], "kicks_per_turn": 1, "z_bins": 16,
                          "pipe": "round", "pipe_a_m": 0.05}, rl, zwrap=5.0, rf_on=True)
        if kind == "cpu":
            sc.kick(0, CpuBeam(*arrs, np.full(n, 1e8))); out.append(np.vstack([Y, arrs[1]]))
        else:
            dev = [cuda.to_device(a) for a in arrs]
            sc.kick(0, CudaBeam(dev, np.full(n, 1e8)))
            out.append(np.vstack([dev[0].copy_to_host(), dev[1].copy_to_host()]))
    dk = np.abs((out[0] - out[1]))
    ref_k = np.abs(out[0] - np.vstack([X, np.zeros(n)])).max(axis=1, keepdims=True)
    worst = max(worst, float(np.max(dk / np.maximum(ref_k, 1e-300))))
st = np.ones(n, np.int8); w = np.full(n, 1e8)
sc = SpaceCharge({"grid": [16, 16], "kicks_per_turn": 1, "z_bins": 16}, rl, zwrap=5.0, rf_on=True)
cb = CpuBeam(X.copy(), np.zeros(n), np.ones(n), np.ones(n), st, w)
m = combine_moments(cb.moments(0, 0), 0, 0)
box = make_box(16, 16, m["x"], m["y"], *half_widths(m, 4.0))
kp, ki = sc._params(0, box, (m["x"], m["y"]), m["w"])
s = _power_of_two_scale(m["w"])
g1, l1, o1 = cb.deposit(kp, ki, s, True)
dev = [cuda.to_device(a) for a in (X.copy(), np.zeros(n), np.ones(n), np.ones(n), st)]
cbd = CudaBeam(dev, w)
g2, l2, o2 = cbd.deposit(kp, ki, s, True)
same = np.array_equal(g1, g2) and np.array_equal(l1, l2) and o1 == o2 \
    and np.array_equal(cb.moments(0, 0), cbd.moments(0, 0))
# (unequal weights, a particle lost: the moments weighted alike)
w2 = np.where(np.arange(n) % 3 == 0, 3e8, 1e8); st2 = st.copy(); st2[5] = 0
dev2 = [cuda.to_device(a) for a in (X.copy(), np.zeros(n), np.ones(n), np.ones(n), st2)]
same = same and np.array_equal(
    CpuBeam(X.copy(), np.zeros(n), np.ones(n), np.ones(n), st2, w2).moments(0, 0),
    CudaBeam(dev2, w2).moments(0, 0))
print("CUDASIM_SC", worst, int(same), flush=True)
"""
    env = dict(os.environ)
    env.update({"NUMBA_ENABLE_CUDASIM": "1", "PYTHONPATH": str(ROOT)})
    r = subprocess.run([sys.executable, "-c", code], cwd=ROOT, env=env, capture_output=True,
                       text=True, timeout=900)
    line = [ln for ln in r.stdout.splitlines() if ln.startswith("CUDASIM_SC")]
    assert line, (r.returncode, r.stderr[-3000:])
    worst, same = float(line[-1].split()[1]), int(line[-1].split()[2])
    assert same == 1                                # deposits and moments: the same integers
    assert worst < 1e-14                            # kicks: rounding


# ---------------------------------------------------------------------------
# z_bins "auto": the line density's bins follow the bunch's core
# ---------------------------------------------------------------------------
def test_auto_bins_are_the_default_and_round_trip():
    assert SpaceChargeConfig().z_bins == "auto"
    assert SpaceChargeConfig.from_dict({"z_bins": " Auto "}).z_bins == "auto"
    d = SpaceChargeConfig(z_bins="auto").to_dict()
    assert d["z_bins"] == "auto" and SpaceChargeConfig.from_dict(d).to_dict() == d
    assert SpaceChargeConfig.from_dict({"z_bins": "512"}).z_bins == 512
    for bad in ("automatic", 7, 65537, 64.5, True, None):
        with pytest.raises(ValueError, match="z_bins"):
            SpaceChargeConfig.from_dict({"z_bins": bad})


def _periodic_gaussian_counts(sigma, centre, nz, period, total=1e7):
    """A Gaussian bunch's cloud-in-cell deposit on nz bins over the period
    (bin k centred at (k + 1/2 - nz/2) dz), as the space charge's kernel
    deposits it — its images one period either side; 64 points a bin."""
    dz = period / nz
    sub = 64
    z = (np.arange(nz * sub) + 0.5) / sub * dz - 0.5 * period
    f = sum(np.exp(-0.5 * ((z - centre - k * period) / sigma) ** 2) for k in (-1, 0, 1))
    u = (z + 0.5 * period) / dz - 0.5                  # (from bin 0's centre, in bins)
    c = np.floor(u)
    w1 = u - c
    k0 = np.mod(c.astype(np.int64), nz)
    out = np.bincount(k0, f * (1.0 - w1), nz) + np.bincount(np.mod(k0 + 1, nz), f * w1, nz)
    return out * (total / out.sum())


def test_the_core_rms_of_a_line_wherever_the_bunch_sits():
    """1.4826 times the median absolute deviation, from the counts: sigma for
    a Gaussian, at the middle of the period or lying across its ends (the
    bunch moved to the middle by its circular mean first); a halo spread over
    the period hardly moves it; an empty line gives 0."""
    from linac_gen.ring.sc.solver import _core_rms
    period, nz = 5.645, 1024
    dz = period / nz
    for sigma, centre in ((0.05, 0.0), (0.05, 1.3), (0.2, -2.0), (0.05, 0.5 * period - 0.01),
                          (0.3, 0.5 * period)):
        c = _periodic_gaussian_counts(sigma, centre, nz, period)
        assert _core_rms(c, dz) == pytest.approx(sigma, rel=2e-3), (sigma, centre)
    core = _periodic_gaussian_counts(0.05, 0.0, nz, period, total=0.95e7)
    halo = np.full(nz, 0.05e7 / nz)
    # (the median absolute deviation d of the mixture: 0.95 erf(d / (sqrt(2) sigma))
    # + 0.05 * 2 d / period = 1/2)
    from scipy.optimize import brentq
    d = brentq(lambda d: 0.95 * erf(d / (math.sqrt(2.0) * 0.05)) + 0.1 * d / period - 0.5,
               0.0, 1.0)
    assert _core_rms(core + halo, dz) == pytest.approx(1.4826 * d, rel=2e-3)
    assert 1.4826 * d < 1.07 * 0.05
    assert _core_rms(np.zeros(nz), dz) == 0.0


def _auto_sc(booster, **kw):
    from linac_gen.ring.compile import compile_from_config
    rl = compile_from_config(booster, RingConfig(), beam=_booster_beam())
    period = rl.length / rl.harmonic
    sc = SpaceCharge(dict({"intensity": 1e10, "kicks_per_turn": 8}, **kw), rl, zwrap=period,
                     rf_on=True)
    return sc, period


def test_the_auto_rule_its_limits_and_its_hysteresis(booster):
    """AUTO_BINS_PER_SIGMA bins per rms of the core, a power of two within
    AUTO_NZ_MIN..AUTO_NZ_MAX; the line density (and with it its slope)
    smoothed over AUTO_WIDTH sigma N^(-1/5), never under z_smoothing_bins
    bins.  A turn's line sets the bins and the width
    of the next turn; the bins change only once it asks for more than
    AUTO_HYSTERESIS times those in use (or less than their 1/AUTO_HYSTERESIS)."""
    from linac_gen.ring.sc import solver as S
    sc, period = _auto_sc(booster)
    assert sc.auto_z and sc.nz == S.AUTO_NZ_START and sc.nz_history == [(0, S.AUTO_NZ_START)]
    assert pytest.approx(1.0592, abs=1e-4) == S.AUTO_WIDTH             # (4/3)^(1/5)
    sig, n = 0.05, 20000
    assert sc.auto_target(sig, n) == pytest.approx(period * S.AUTO_BINS_PER_SIGMA / sig)
    assert sc.auto_target(sig, 3) == sc.auto_target(sig, n)          # (not the particles')
    wl = sc.auto_width(sig, n)
    assert wl == pytest.approx(S.AUTO_WIDTH * sig * n ** -0.2, rel=1e-15)
    for bad in ((0.0, n), (sig, 0), (float("nan"), n)):
        assert sc.auto_target(*bad) is None and sc.auto_bins(*bad) is None
        assert sc.auto_width(*bad) is None
    x = period * S.AUTO_BINS_PER_SIGMA / sig
    assert sc.auto_bins(sig, n) == 2 ** round(math.log2(x))
    assert sc.auto_bins(1e-9, n) == S.AUTO_NZ_MAX
    assert sc.auto_bins(1e3, n) == S.AUTO_NZ_MIN
    # (before a beam: z_smoothing_bins bins for both)
    assert sc.filter_width() == sc.dz

    def turn(t, sigma, count=n):
        sc._first_line = (_periodic_gaussian_counts(sigma, 0.0, sc.nz, period), count)
        sc.end_turn(t)
        before = sc.nz
        sc.begin_turn(t + 1, refresh=False)
        return before, sc.nz

    before, after = turn(0, sig)
    assert (before, after) == (S.AUTO_NZ_START, sc.auto_bins(sig, n))
    assert sc.nz_history[-1] == (1, after) and sc.dz == pytest.approx(period / after)
    # (the width from the line's core rms: read on the start's 256 bins, 2.3
    # bins per rms, to 1 %; on the bins that follow the bunch, to 0.1 %)
    assert sc.filter_width() == pytest.approx(max(sc.dz, wl), rel=1e-2)
    turn(1, sig)
    assert sc.filter_width() == pytest.approx(max(sc.dz, wl), rel=1e-3)
    # few particles (the fewest judged): wide; many: the floor of a bin
    turn(2, sig, count=S.AUTO_MIN_PARTICLES)
    assert sc.filter_width() > 0.4 * sig
    turn(3, sig, count=10 ** 12)
    assert sc.filter_width() == sc.dz
    n0 = sc.nz
    for factor, moves in ((1.5, False), (0.7, False), (1.75, True)):
        sc._set_nz(n0, 10)
        sigma = period * S.AUTO_BINS_PER_SIGMA / (factor * n0)
        _b, a = turn(10, sigma)
        assert (a != n0) == moves, (factor, a)
    assert a == 2 * n0
    sc._set_nz(n0, 20)
    assert turn(20, period * S.AUTO_BINS_PER_SIGMA / (0.55 * n0))[1] == n0 // 2
    # an empty line, or fewer than AUTO_MIN_PARTICLES (one particle's line is
    # a spike), asks for nothing; a fixed z_bins never moves, its filter
    # z_smoothing_bins
    nz, w = sc.nz, sc.width
    for counts, count in ((np.zeros(sc.nz), 0),
                          (_periodic_gaussian_counts(1e-4, 0.0, sc.nz, period), 1),
                          (_periodic_gaussian_counts(0.01, 0.0, sc.nz, period),
                           S.AUTO_MIN_PARTICLES - 1)):
        sc._first_line = (counts, count)
        sc.end_turn(30)
        sc.begin_turn(31, refresh=False)
        assert sc.nz == nz and sc.width == w
    sc.start_bins(0.001, 1)
    assert sc.nz == nz and sc.width == w
    fixed, _ = _auto_sc(booster, z_bins=128, z_smoothing_bins=2.0)
    fixed._first_line = (_periodic_gaussian_counts(0.01, 0.0, 128, period), 10 ** 6)
    fixed.end_turn(0)
    fixed.begin_turn(1, refresh=False)
    assert not fixed.auto_z and fixed.nz == 128 and fixed.nz_history == [(0, 128)]
    assert fixed.width is None and fixed.filter_width() == 2 * fixed.dz


def test_the_line_density_and_its_slope_through_the_auto_filter(booster):
    """"auto": f and f' through one Gaussian filter of the auto width (in
    Fourier space); a fixed z_bins gives the bits of the filter of
    z_smoothing_bins bins, as before "auto" existed."""
    sc, period = _auto_sc(booster)
    rng = np.random.default_rng(4)
    counts = rng.poisson(_periodic_gaussian_counts(0.1, 0.2, sc.nz, period, total=4000.0))
    sc.width = 0.05
    sc._filters = None
    fl, dfl = sc.line_density(counts, 1.0, float(counts.sum()))
    f = counts / (counts.sum() * sc.dz)
    F = np.fft.rfft(f)
    k = np.arange(F.size)
    om = 2.0 * np.pi * k / period
    G = np.exp(-0.5 * (om * 0.05) ** 2)
    want_f = np.fft.irfft(F * G, n=sc.nz)
    D = 1j * om * F * G
    D[-1] = 0.0
    want_d = np.fft.irfft(D, n=sc.nz)
    assert np.max(np.abs(fl - want_f)) < 1e-12 * np.max(np.abs(want_f))
    assert np.max(np.abs(dfl - want_d)) < 1e-12 * np.max(np.abs(want_d))
    # (fixed bins: the earlier filter, bit for bit)
    fixed, _ = _auto_sc(booster, z_bins=256, z_smoothing_bins=1.5)
    c2 = rng.poisson(_periodic_gaussian_counts(0.1, 0.0, 256, period, total=4000.0))
    got = fixed.line_density(c2, 1.0, float(c2.sum()))
    f2 = np.asarray(c2, dtype=float) / (1.0 * c2.sum() * fixed.dz)
    F2 = np.fft.rfft(f2)
    fr, fi = np.ascontiguousarray(F2.real), np.ascontiguousarray(F2.imag)
    g = np.array([math.exp(-0.5 * (2.0 * math.pi * kk * 1.5 / 256) ** 2)
                  for kk in range(F2.size)])
    fr, fi = fr * g, fi * g
    Fs = np.empty(F2.size, dtype=complex)
    Fs.real, Fs.imag = fr, fi
    om2 = 2.0 * np.pi * np.arange(F2.size) / period
    D2 = np.empty(F2.size, dtype=complex)
    D2.real, D2.imag = -fi * om2, fr * om2
    D2[-1] = 0.0
    assert np.array_equal(got[0], np.fft.irfft(Fs, n=256))
    assert np.array_equal(got[1], np.fft.irfft(D2, n=256))


def _slope_error(booster, n, seed):
    """The rms error of "auto"'s f' at the particles of a Gaussian bunch (rms
    0.3 m, N particles) against the bunch's own slope, relative to its rms
    — through the space charge's own deposit, filter and gather."""
    from linac_gen.ring.sc import kernels as K
    from linac_gen.ring.sc.solver import Box
    sc, period = _auto_sc(booster)
    rng = np.random.default_rng(seed)
    z = rng.normal(0.0, 0.3, n)
    sc.start_bins(0.3, n)
    coords = np.zeros((6, n))
    coords[4] = z
    w = np.full(n, 1.0 / n)
    kp, ki = sc._params(0, Box(sc.nx, sc.ny, 0.0, 0.0, 1.0, 1.0), (0.0, 0.0), 1.0)
    grid = np.zeros((1, sc.nx + 3, sc.ny + 3), dtype=np.int64)
    line = np.zeros((1, sc.nz), dtype=np.int64)
    scale = 2.0 ** 52
    K.sc_backend()[1](coords, w, np.ones(n, dtype=np.int8), n, kp, ki, scale, grid, line,
                      np.zeros(1, dtype=np.int64), False)
    _fl, dfl = sc.line_density(line.sum(axis=0), scale, 1.0)
    # (the kernel's gather, K.line_cell, for every particle at once)
    u = (z - kp[K.KP_Z0]) / kp[K.KP_DZ]
    c = np.floor(u)
    k0 = np.mod(c.astype(np.int64), sc.nz)
    k1 = np.mod(k0 + 1, sc.nz)
    got = (1.0 - (u - c)) * dfl[k0] + (u - c) * dfl[k1]
    want = -z / 0.3 ** 2 * np.exp(-0.5 * (z / 0.3) ** 2) / (math.sqrt(2.0 * math.pi) * 0.3)
    return float(np.sqrt(np.mean((got - want) ** 2)) / np.sqrt(np.mean(want ** 2)))


def test_auto_slope_noise_falls_as_particles_grow(booster):
    """The longitudinal kick follows the line density's slope: with "auto"
    its error at the particles (noise and smoothing together) falls as the
    particles grow, 5000 -> 20000 -> 80000 (a floor on the particles in the
    busiest bin, which narrows the bins as particles are added, made it grow)."""
    err = {n: np.mean([_slope_error(booster, n, s) for s in (1, 2)])
           for n in (5000, 20000, 80000)}
    assert err[5000] > err[20000] > err[80000], err
    assert err[80000] < 0.12, err


def test_the_frozen_model_changes_its_bins_when_it_measures_the_beam_again(booster):
    """(its stored line densities all have the bins they were made with)"""
    sc, period = _auto_sc(booster, model="frozen", refresh_turns=4)
    sc._first_line = (_periodic_gaussian_counts(0.05, 0.0, sc.nz, period), 1e9)
    sc.end_turn(0)
    want = sc.auto_bins(0.05, 1e9)
    assert want != sc.nz
    for t in (1, 2, 3):
        sc.begin_turn(t, refresh=False)
        assert sc.nz != want
    sc.begin_turn(4, refresh=True)
    assert sc.nz == want and sc.nz_history[-1] == (4, want)


def _rotating_bunch(n, seed=5):
    """A bunch with a sixth of its matched energy spread (the bucket's
    small-amplitude ratio sigma_z / sigma_delta is about 822 m at 800 MeV): a
    flat ellipse, it rotates in the bucket from 25 cm to about 4 cm in ten
    turns (without the longitudinal space-charge kick, which weakens the
    bucket's focusing)."""
    rng = np.random.default_rng(seed)
    X0 = np.zeros((6, n))
    X0[0], X0[2] = rng.normal(0.0, 1e-3, n), rng.normal(0.0, 1e-3, n)
    X0[4] = rng.normal(0.0, 0.25, n)
    X0[5] = rng.normal(0.0, 0.25 / 822.0 / 6.0, n)
    return X0


def test_auto_bins_follow_a_bunch_that_shortens(booster, tmp_path):
    """The bins grow as the bunch rotates shorter; the same bits and the same
    bins for any number of threads, and from a checkpoint written while a
    change was pending."""
    from linac_gen.ring.sc import solver as S
    X0 = _rotating_bunch(4000)
    # (the transverse kick alone: the RF rotates the bunch as it would without)
    sc = dict(SC_BOOSTER, kicks_per_turn=8, longitudinal="off")

    def cfg(n_turns, **kw):
        return RingConfig(n_turns=n_turns, n_particles=4000, tune_probes=0, space_charge=sc,
                          snapshot_turns=["last"], stats_every=1, **kw)

    a = RingSimulation(booster, _booster_beam(), cfg(12, threads=1), coords=X0).run()
    assert math.sqrt(a.cov[10][4, 4]) < 0.06                     # (the bunch did shorten)
    hist = a.meta["space_charge"]["z_bins_used"]
    assert hist[0] == [0, S.AUTO_NZ_START] and a.meta["space_charge"]["z_bins_auto"] is True
    start = [b for t, b in hist if t == 0][-1]
    later = [b for t, b in hist if t > 0]
    assert len(later) >= 2 and all(b2 > b1 for b1, b2 in zip([start] + later[:-1], later,
                                                             strict=True)), hist
    assert later[-1] >= 4 * start
    b = RingSimulation(booster, _booster_beam(), cfg(12, threads=4), coords=X0).run()
    assert np.array_equal(a.cov, b.cov) and b.meta["space_charge"]["z_bins_used"] == hist
    # a checkpoint at the end of the turn that asked for the first change (the
    # change pending in it)
    k = [t for t, _b in hist if t > 0][0]      # (bins from turn k on: asked for at k - 1)
    ck = tmp_path / "ck.npz"
    # (checkpoints are written before the last turn: a run one turn longer)
    RingSimulation(booster, _booster_beam(), cfg(k + 1, checkpoint_every=k),
                   coords=X0).run(checkpoint_path=ck)
    import json
    with np.load(ck) as z:
        assert json.loads(str(z["meta_json"]))["sc_bins"]["pending"] == later[0]
    c = RingSimulation(booster, _booster_beam(), cfg(12), coords=X0).run(restart_from=ck)
    assert np.array_equal(a.cov, c.cov)
    assert np.array_equal(a.snapshots[12]["coords"], c.snapshots[12]["coords"])
    assert c.meta["space_charge"]["z_bins_used"] == hist


def test_a_bunch_too_few_particles_resolve_gets_the_auto_note(booster):
    """With "auto" few particles widen the line density's filter: the note
    says how wide, and why."""
    from linac_gen.ring.sc import solver as S
    rng = np.random.default_rng(9)
    n = 600
    period = 474.202752 / 84.0
    X0 = np.zeros((6, n))
    X0[0], X0[2] = rng.normal(0.0, 1e-3, n), rng.normal(0.0, 1e-3, n)
    X0[4], X0[5] = rng.normal(0.0, 0.03, n), rng.normal(0.0, 0.03 / 825.0, n)
    cfg = RingConfig(n_turns=2, n_particles=n, tune_probes=0,
                     space_charge=dict(SC_BOOSTER, kicks_per_turn=8))
    res = RingSimulation(booster, _booster_beam(), cfg, coords=X0).run()
    lr = res.meta["space_charge"]["line_density_resolution"]
    assert lr["peak_ratio"] < 0.97 and lr["limited_by"] == "particles"
    assert lr["particles"] == n and lr["line_filter_m"] > 2.0 * period / lr["z_bins"]
    # (what the note says the line density was smoothed over is the auto width)
    assert lr["line_filter_m"] == pytest.approx(S.AUTO_WIDTH * 0.03 * n ** -0.2, rel=0.05)
    noted = [m for m in res.meta["notes"] if "line-density bins" in m]
    assert len(noted) == 1
    assert (f"the line density was smoothed over {lr['line_filter_m'] * 1e3:.3g} mm (rms) to "
            f"hold down the noise of its {n} particles — more particles resolve it") in noted[0]


def test_what_kept_auto_bins_coarse(booster):
    from linac_gen.ring.sc import solver as S
    sc, period = _auto_sc(booster)
    floor = period / 256
    assert sc.auto_limit(0.05, 100, 256, line_width=3.0 * floor) == "particles"
    assert sc.auto_limit(0.05, 1e9, 256, line_width=floor) == "follow"
    assert sc.auto_limit(0.05, 1e9, 256) == "follow"
    assert sc.auto_limit(0.05, 1e9, S.AUTO_NZ_MAX, line_width=period / S.AUTO_NZ_MAX) \
        == "largest"


# ---------------------------------------------------------------------------
# edge cases
# ---------------------------------------------------------------------------
def test_one_particle_runs_as_without_space_charge(booster):
    X = np.zeros((6, 1))
    X[0, 0], X[2, 0] = 1e-3, 5e-4
    out = []
    for sc in (SC_BOOSTER, None):
        cfg = RingConfig(n_turns=5, n_particles=1, tune_probes=0, space_charge=sc,
                         snapshot_turns=["last"])
        out.append(RingSimulation(booster, _booster_beam(), cfg, coords=X).run())
    a, b = out[0].snapshots[5]["coords"][:, 0], out[1].snapshots[5]["coords"][:, 0]
    # (rounding of its own field only: x, px, y, py, zeta [m], delta)
    assert np.all(np.abs(a - b) <= np.array([1e-12, 1e-12, 1e-12, 1e-12, 1e-11, 1e-14]))
    assert out[0].meta["space_charge"]["laslett_gaussian"] is None      # (no size)


def test_every_particle_lost_mid_turn(booster):
    import copy

    from linac_gen.elements.aperture import Aperture
    lat = copy.deepcopy(booster)
    lat.elements.insert(len(lat.elements) // 2, Aperture("pinhole", dx=1e-6,
                                                         aperture_type=Aperture.CIRCULAR))
    cfg = RingConfig(n_turns=4, n_particles=300, tune_probes=1, space_charge=SC_BOOSTER)
    res = RingSimulation(lat, _booster_beam(), cfg).run()
    assert res.n_alive[-1] == 0 and res.meta["all_lost_at_turn"] == 1
    assert res.ledger["lost"] == 300


def test_a_coasting_beam_needs_no_wrapped_zeta(booster):
    import copy

    from linac_gen.ring.compile import compile_from_config
    rl = compile_from_config(copy.deepcopy(booster), RingConfig(), beam=_booster_beam())
    cfg = RingConfig(n_turns=2, n_particles=400, tune_probes=0, zeta_wrap="off",
                     space_charge=SC_BOOSTER)
    res = RingSimulation(booster, _booster_beam(), cfg, ring_lattice=rl.with_rf(False)).run()
    m = res.meta["space_charge"]
    assert not m["bunched"] and not m["longitudinal_kick"] and res.n_alive[-1] == 400


def test_off_the_grid_the_multipole_series_is_the_charges_field():
    # kernels.far_field (particles beyond the grid) against the direct sums
    # F_x - i F_y = sum q/(z - z_j), Phi = -sum q ln|z - z_j|
    from linac_gen.ring.sc.kernels import far_field
    from linac_gen.ring.sc.solver import far_moments
    rng = np.random.default_rng(11)
    X, Y = rng.normal(0, 1e-3, (2, 400))
    q = rng.random(400)
    q /= q.sum()
    xc, yc = 1e-4, -2e-4
    far = far_moments(q, X, Y, xc, yc)
    r_max = float(np.max(np.hypot(X - xc, Y - yc)))
    for k in range(20):
        a = r_max * (2.0 + k * 0.2)
        t = rng.uniform(0, 2 * np.pi)
        dx, dy = a * math.cos(t), a * math.sin(t)
        fx, fy, phi = far_field(dx, dy, far, 16, True)
        z = complex(dx, dy)
        w = (X - xc) + 1j * (Y - yc)
        e = np.sum(q / (z - w))
        want_phi = -np.sum(q * np.log(np.abs(z - w)))
        tol = 2.0 * (r_max / a) ** 17
        assert abs(complex(fx, -fy) - e) <= tol * abs(e)
        assert abs(phi - want_phi) <= tol * abs(want_phi) + 1e-12


def test_the_kernels_pipe_series_is_the_hosts():
    # kernels.boundary_field (off-grid particles) and boundary.harmonic_on_grid
    # (the nodes) evaluate the same harmonic
    from linac_gen.ring.sc.kernels import boundary_field
    rng = np.random.default_rng(12)
    c = rng.normal(size=9) + 1j * rng.normal(size=9)
    bnd = np.vstack([c.real, c.imag])
    R = 0.04
    for _ in range(10):
        x, y = rng.uniform(-0.03, 0.03, 2)
        fx, fy, phi = boundary_field(x - 0.001, y + 0.002, bnd, 8, 1.0 / R, True)
        h, hx, hy = harmonic_on_grid(c, R, (0.001, -0.002), np.array([x]), np.array([y]),
                                     potential=True)
        assert phi == pytest.approx(h[0], rel=1e-13, abs=1e-15)
        assert fx == pytest.approx(hx[0], rel=1e-12, abs=1e-15)
        assert fy == pytest.approx(hy[0], rel=1e-12, abs=1e-15)


def test_charge_beyond_a_round_pipe_is_left_out_of_its_images():
    b = 0.02
    X, Y = np.meshgrid(np.linspace(-0.025, 0.025, 11), np.linspace(-0.004, 0.004, 3),
                       indexing="ij")
    q = np.zeros(X.shape)
    q[5, 1] = 0.9                                   # (0, 0)
    q[0, 1] = 0.1                                   # (-0.025, 0): beyond the wall
    pipe = RoundPipe(b)
    c = pipe.coefficients(q, X, Y)
    assert pipe.charge_beyond                       # (the kick is counted)
    inside = np.zeros(X.shape)
    inside[5, 1] = 0.9
    assert np.array_equal(c, pipe.coefficients(inside, X, Y))
    assert not pipe.charge_beyond


@pytest.mark.parametrize("model", ["pic", "frozen"])
@pytest.mark.parametrize("ratio", [3.0, 6.0, 30.0])
def test_particles_just_off_a_flat_beams_grid_get_its_field(model, ratio):
    # a flat beam's grid reaches 4 rms sizes in each plane: a particle just
    # beyond it in the narrow plane is nearer the centre than the charge at
    # the grid's far ends, where the multipole series does not converge —
    # it must get the beam's field all the same (here against
    # Bassetti-Erskine), as must particles far out
    ref = RingReference.from_kinetic("proton", 800.0)
    n, sy = 200_000, 1e-3
    sx = ratio * sy
    rng = np.random.default_rng(21)
    X = np.zeros((6, n))
    X[0], X[2] = rng.normal(0, sx, n), rng.normal(0, sy, n)
    keep = (X[0] / sx) ** 2 + (X[2] / sy) ** 2 < 16.0          # (the grid holds the rest)
    X = X[:, keep]
    probes = np.array([[0.0, 4.6], [0.5, 4.8], [0.0, 6.0], [-0.3, -5.0], [4.6, 0.0],
                       [3.0, 4.5], [0.0, 30.0], [12.0, 1.0]]).T
    T = np.zeros((6, probes.shape[1]))
    T[0], T[2] = probes[0] * sx, probes[1] * sy
    X = np.hstack([X, T])
    m = X.shape[1]
    X1, _pt, sc = _kick_once({"model": model, "grid": [64, 64], "kicks_per_turn": 1},
                             X, 1e10 / m, ref=ref, length=100.0, rf_on=False)
    k = 2 * classical_radius(ref.mass) * 1e10 * sc.ds[0] / 100.0 / (
        ref.beta0 ** 2 * ref.gamma0 ** 3)
    fx, fy = _be_field(T[0], T[2], sx, sy)
    got_x, got_y = (X1[1] - X[1])[-T.shape[1]:] / k, (X1[3] - X[3])[-T.shape[1]:] / k
    scale = np.hypot(fx, fy)
    err = np.hypot(got_x - fx, got_y - fy) / scale
    assert np.all(err < 0.03), err


def test_one_particle_through_a_ramp_runs_as_without_space_charge(booster):
    # zeta wrapped once a turn, also with space charge: in a ramp the RF
    # period in zeta changes within the turn, and a particle crossing the
    # period's edge must not be wrapped by C/h between two kicks
    ramp = {"energy": {"kind": "table", "times_s": [0.0, 1e-4], "values": [800.0, 830.0]},
            "voltage": {"kind": "table", "times_s": [0.0, 1.0], "values_mv": [1.2, 1.2]},
            "gamma_t": 5.449167323}
    X = np.zeros((6, 1))
    X[4, 0], X[5, 0] = 2.6, 2e-3          # (near the period's edge, outside the bucket)
    out = []
    for sc in (SC_BOOSTER, None):
        cfg = RingConfig(n_turns=40, n_particles=1, tune_probes=0, ramp=ramp, space_charge=sc,
                         snapshot_turns=["last"])
        out.append(RingSimulation(booster, _booster_beam(), cfg, coords=X).run())
    a, b = out[0].snapshots[40]["coords"][:, 0], out[1].snapshots[40]["coords"][:, 0]
    assert np.all(np.abs(a - b) <= np.array([1e-12, 1e-12, 1e-12, 1e-12, 1e-10, 1e-13])), a - b


def _misaligned(a, offset):
    """A copy of ``a`` starting ``offset`` bytes into a fresh buffer (8-byte
    aligned only): numpy's SIMD loops take other paths for such arrays."""
    buf = np.zeros(a.size * 8 + 64, dtype=np.uint8)
    out = np.frombuffer(buf[offset:offset + a.size * 8].data, dtype=float).reshape(a.shape)
    out[...] = a
    return out


@pytest.mark.parametrize("n", [16, 64])
def test_the_field_solve_gives_the_same_bits_wherever_its_arrays_sit(n):
    # (numpy's complex multiply rounds differently for differently aligned
    # arrays: the solve and the series are written in real arithmetic)
    from linac_gen.ring.sc.solver import far_moments
    rng = np.random.default_rng(5)
    q = rng.random((n, n))
    q /= q.sum()
    X, Y = np.meshgrid(np.arange(n) * 1e-4, np.arange(n) * 2e-4, indexing="ij")
    ref = greens.solve(q, 1e-4, 2e-4)
    ref_far = far_moments(q, X, Y, 1e-3, 2e-3)
    ref_img = RoundPipe(0.05).coefficients(q, X, Y)
    for k in range(300):
        qq = _misaligned(q, 8 * (k % 4))
        got = greens.solve(qq, 1e-4, 2e-4)
        assert all(np.array_equal(a, b) for a, b in zip(got, ref, strict=True)), k
        assert np.array_equal(far_moments(qq, X, Y, 1e-3, 2e-3), ref_far), k
        assert np.array_equal(RoundPipe(0.05).coefficients(qq, X, Y), ref_img), k


def test_a_run_on_a_small_grid_repeats_bit_for_bit(booster):
    sc = dict(SC_BOOSTER, grid=[16, 16], pipe="round", pipe_a_m=0.05)
    out = [RingSimulation(booster, _booster_beam(), RingConfig(
        n_turns=3, n_particles=2000, tune_probes=1, space_charge=sc,
        snapshot_turns=["last"])).run() for _ in range(3)]
    for r in out[1:]:
        assert np.array_equal(r.cov, out[0].cov)
        assert np.array_equal(r.snapshots[3]["coords"], out[0].snapshots[3]["coords"])


def test_a_fitted_pipe_does_not_depend_on_blas_threads():
    from threadpoolctl import threadpool_limits
    out = []
    for th in (1, 3):
        with threadpool_limits(th):
            out.append(FittedPipe("ellipse", 0.06, 0.025, 64).pinv)
    assert np.array_equal(out[0], out[1])



def test_a_coasting_beams_intensity_is_the_current_over_a_turn(booster):
    import copy

    from linac_gen.ring.compile import compile_from_config
    rl = compile_from_config(copy.deepcopy(booster), RingConfig(), beam=_booster_beam())
    beam = _booster_beam()
    beam.current = 5.0                                   # mA
    for rf in (False, True):
        cfg = RingConfig(n_turns=1, n_particles=200, tune_probes=0,
                         space_charge={"kicks_per_turn": 4})
        r = rl.with_rf(rf)
        res = RingSimulation(booster, beam, cfg, ring_lattice=r).run()
        got = res.meta["space_charge"]["intensity_used"]
        if rf:                                           # one bunch: I / f
            assert got == pytest.approx(5e-3 / (44.7044096e6 * 1.602176634e-19), rel=1e-12)
        else:                                            # the ring: I T_rev
            assert got == pytest.approx(5e-3 * r.revolution_time / 1.602176634e-19, rel=1e-12)


def test_the_frozen_models_fields_solved_again_give_the_same_bits(booster, monkeypatch):
    import linac_gen.ring.sc.solver as S
    sc = dict(SC_BOOSTER, model="frozen", refresh_turns=0)
    cfg = RingConfig(n_turns=3, n_particles=1000, tune_probes=0, space_charge=sc,
                     snapshot_turns=["last"])
    a = RingSimulation(booster, _booster_beam(), cfg).run()
    monkeypatch.setattr(S, "FROZEN_CACHE_BYTES", 1)      # (one kick's fields kept at a time)
    b = RingSimulation(booster, _booster_beam(), cfg).run()
    assert np.array_equal(a.snapshots[3]["coords"], b.snapshots[3]["coords"])
    assert a.meta["space_charge"]["outside_grid_max"] is None    # (not counted when frozen)


def _kick_positions_by_scan(rl, n):
    """The placement rule written out plainly (each target against every
    element start; a start at the very end of the ring is the start of the
    ring, node 0's place)."""
    C = float(rl.length)
    ne = list(np.asarray(rl.node_element))
    ns = list(np.asarray(rl.node_s, dtype=float))
    starts = [(i, ns[i]) for i in range(len(ne)) if i == 0 or ne[i] != ne[i - 1]]
    starts = [st for k, st in enumerate(starts) if k == 0 or st[1] < C * (1.0 - 1e-12)]
    nodes = np.array([i for i, _ in starts])
    s = np.array([v for _, v in starts])
    chosen = {}
    for k in range(n):
        t = k * C / n
        d = np.abs(s - t)
        d = np.minimum(d, C - d)
        i = int(np.argmin(d))
        chosen[float(s[i])] = int(nodes[i])
    pos = np.array(sorted(chosen))
    return np.array([chosen[p] for p in pos]), pos


def test_the_placement_is_the_plain_rule(booster):
    from linac_gen.ring.compile import compile_from_config
    from linac_gen.ring.sc.placement import element_starts, placed_count
    rl = compile_from_config(booster, RingConfig(), beam=_booster_beam())
    for n in (1, 7, 24, 96, 300, 1000, 5000):
        nodes, s, _ds = kick_positions(rl, n)
        want_nodes, want_s = _kick_positions_by_scan(rl, n)
        assert np.array_equal(nodes, want_nodes) and np.array_equal(s, want_s), n
        assert placed_count(element_starts(rl)[1], rl.length, n) == len(nodes)
    # a huge request is cheap: no more kicks than element starts
    nodes, _s, ds = kick_positions(rl, 10_000_000)
    assert len(nodes) == len(np.unique(element_starts(rl)[1]))
    assert ds.sum() == pytest.approx(rl.length, rel=1e-12)


class _Ring:
    """Just the node table the placement reads."""

    def __init__(self, node_element, node_s, length):
        self.node_element = np.asarray(node_element)
        self.node_s = np.asarray(node_s, dtype=float)
        self.length = float(length)


def test_a_zero_length_element_at_the_end_of_the_ring_gets_no_kick_of_its_own(booster):
    # its start is s = C, the start of the ring: no kick there (node 0's
    # place), and the kicks' lengths still sum to the ring
    from linac_gen.ring.sc.placement import element_starts, placed_count
    rl = _Ring([0, 0, 1, 1, 2, 3], [0.0, 0.5, 1.0, 1.5, 2.0, 2.0], 2.0)
    nodes, s = element_starts(rl)
    assert np.array_equal(nodes, [0, 2]) and np.array_equal(s, [0.0, 1.0])
    for n in (1, 2, 4, 9, 100):
        k_nodes, k_s, ds = kick_positions(rl, n)
        want_nodes, want_s = _kick_positions_by_scan(rl, n)
        assert np.array_equal(k_nodes, want_nodes) and np.array_equal(k_s, want_s), n
        assert np.all(k_s < rl.length) and ds.sum() == pytest.approx(rl.length, rel=1e-15)
        assert placed_count(s, rl.length, n) == len(k_nodes)
    # (a real ring: the booster with a marker after its last element)
    import copy

    from linac_gen.elements.marker import Marker
    from linac_gen.ring.compile import compile_from_config
    lat = copy.deepcopy(booster)
    lat.elements.append(Marker("end_marker"))
    rl = compile_from_config(lat, RingConfig(), beam=_booster_beam())
    nodes, s = element_starts(rl)
    assert np.asarray(rl.node_s)[-1] == pytest.approx(rl.length, rel=1e-15)
    assert s.max() < rl.length * (1 - 1e-12)
    k_nodes, k_s, ds = kick_positions(rl, 10_000_000)
    assert len(k_nodes) == len(np.unique(s)) and np.all(k_s < rl.length * (1 - 1e-12))
    assert ds.sum() == pytest.approx(rl.length, rel=1e-12)


def test_a_fitted_pipe_is_the_same_whatever_was_allocated_before():
    # (the fit is built with elementwise operations and sums only: the same
    # bits wherever its arrays land)
    rng = np.random.default_rng(3)
    ref = FittedPipe("ellipse", 0.05, 0.04, 8)
    X, Y = np.meshgrid(np.linspace(-0.01, 0.01, 9), np.linspace(-0.005, 0.005, 7), indexing="ij")
    q = rng.random(X.shape)
    q /= q.sum()
    d = (X[1, 0] - X[0, 0], Y[0, 1] - Y[0, 0])

    def coeffs(pipe, qq):
        return pipe.coefficients(qq, X, Y, potential_at=lambda px, py: greens.potential_at(
            px, py, X, Y, qq, *d))
    c_ref = coeffs(ref, q)
    junk = []
    for k in range(200):
        junk.append(np.empty(int(rng.integers(1, 4000))))       # (allocations of all sizes)
        if len(junk) > 20:
            junk.pop(int(rng.integers(0, len(junk))))
        pipe = FittedPipe("ellipse", 0.05, 0.04, 8)
        assert np.array_equal(pipe.pinv, ref.pinv), k
        assert np.array_equal(coeffs(pipe, _misaligned(q, 8 * (k % 4))), c_ref), k


def test_a_particle_beyond_a_round_pipes_wall_feels_the_beam_alone():
    # the images are the field inside the wall only: beyond it (a particle
    # about to be lost) the beam's own field, not the images' series, which
    # does not converge there
    ref = RingReference.from_kinetic("proton", 800.0)
    rng = np.random.default_rng(8)
    n = 20_000
    X = np.zeros((6, n + 4))
    X[0, :n], X[2, :n] = rng.normal(0, 2e-3, n), rng.normal(0, 1.5e-3, n)
    X[0, n:], X[2, n:] = [0.08, -0.06, 0.0, 0.07], [0.0, 0.06, -0.1, 0.07]
    out = {}
    for pipe in ({"pipe": "open"}, {"pipe": "round", "pipe_a_m": 0.05}):
        X1, _pt, _sc = _kick_once(dict(pipe, grid=[32, 32], kicks_per_turn=1), X, 1e11 / n,
                                  ref=ref, length=50.0, rf_on=False)
        out[pipe["pipe"]] = (X1 - X)[[1, 3], n:]
    k = 2 * classical_radius(ref.mass) * 1e11 / (ref.beta0 ** 2 * ref.gamma0 ** 3)
    r2 = X[0, n:] ** 2 + X[2, n:] ** 2
    free = k * np.vstack([X[0, n:] / r2, X[2, n:] / r2])        # (a line charge, far out)
    assert np.all(np.isfinite(out["round"]))
    assert np.array_equal(out["round"], out["open"])
    assert np.allclose(out["round"], free, rtol=0.02)


def test_the_frozen_cache_keeps_the_first_kicks_and_solves_the_rest_each_time(booster,
                                                                              monkeypatch):
    # (kept while they fit, never evicted: a turn visits the kicks in order,
    # so evicting the oldest would keep none it needs)
    import linac_gen.ring.sc.solver as S
    built, sizes = [], {}
    orig = S.SpaceCharge._frozen_fields

    def spy(self, k):
        f = orig(self, k)
        built.append(int(k))
        sizes[int(k)] = self._cached_bytes(f)
        return f

    monkeypatch.setattr(S.SpaceCharge, "_frozen_fields", spy)
    sc = dict(SC_BOOSTER, model="frozen", refresh_turns=0, kicks_per_turn=8)
    cfg = RingConfig(n_turns=3, n_particles=500, tune_probes=0, space_charge=sc,
                     snapshot_turns=["last"])
    a = RingSimulation(booster, _booster_beam(), cfg).run()
    m = len(sizes)
    assert built == list(range(m))                       # (each solved once)
    built.clear()
    monkeypatch.setattr(S, "FROZEN_CACHE_BYTES", sizes[0] + sizes[1])
    b = RingSimulation(booster, _booster_beam(), cfg).run()
    assert built == list(range(m)) + 2 * list(range(2, m))
    assert np.array_equal(a.snapshots[3]["coords"], b.snapshots[3]["coords"])


def test_a_ring_of_identical_cells_gets_the_same_kicks_in_every_cell(round_ring):
    # the targets k C/n start at the start of the ring: in a ring of 40 cells,
    # each with starts 1 m apart, 40 or 80 targets land on starts — no ties
    # (targets halfway between two starts used to split by rounding and drop
    # kicks)
    _lat, rl, _o, _ref = round_ring
    C = float(rl.length)
    for n in (40, 80):
        nodes, s, ds = kick_positions(rl, n)
        assert len(nodes) == n and np.allclose(ds, C / n, rtol=1e-12, atol=0.0), n
    # denser than the starts: every start, without making the targets
    nodes, _s, ds = kick_positions(rl, 1_000_000)
    assert len(nodes) == 80 and np.allclose(ds, 1.0, rtol=1e-12)


def test_an_element_of_negative_length_is_refused():
    rl = _Ring([0, 1, 2, 3], [0.0, 1.0, 0.6, 1.5], 2.0)
    with pytest.raises(ValueError, match="negative length"):
        kick_positions(rl, 3)


def test_a_pipe_the_harmonics_cannot_fit_is_refused():
    # the check: unit charges out to half the aperture leave < 0.01 on the wall
    # (its corners included: a rectangle four times wider than high leaves
    # 0.026 with 24 harmonics and 0.011 with 32, 0.0056 with 40)
    with pytest.raises(ValueError, match="half its aperture"):
        FittedPipe("rectangle", 0.08, 0.01, 24)
    for order in (24, 32):
        with pytest.raises(ValueError, match="half its aperture"):
            FittedPipe("rectangle", 0.04, 0.01, order)
    assert FittedPipe("rectangle", 0.04, 0.01, 40).fit_residual < 0.01
    assert FittedPipe("rectangle", 0.06, 0.025, 24).fit_residual < 0.01


def test_the_fit_check_finds_the_largest_potential_left_on_the_wall():
    # the check's points are evenly spaced with a rectangle's corners (where
    # the harmonics fit it worst) and an ellipse's axis ends among them: it
    # reports what a walk of 200 000 points around the wall finds, to 0.1 %
    # (points half a step from the corners missed 13 % of it on a rectangle
    # 2.4 times wider than high)
    from linac_gen.ring.sc.boundary import _check_points

    def walk(shape, a, b, n):
        if shape == "ellipse":
            t = np.linspace(0.0, 2.0 * np.pi, n)
            return a * np.cos(t), b * np.sin(t)
        s = np.linspace(-1.0, 1.0, n // 4)
        return (np.concatenate([a * s, np.full_like(s, a), a * s, np.full_like(s, -a)]),
                np.concatenate([np.full_like(s, -b), b * s, np.full_like(s, b), b * s]))

    for shape, a, b, order in (("rectangle", 0.03, 0.0125, 24), ("rectangle", 0.04, 0.01, 48),
                               ("rectangle", 0.05, 0.03, 24), ("ellipse", 0.06, 0.025, 24)):
        pipe = FittedPipe(shape, a, b, order)
        px, py = walk(shape, a, b, 200_000)
        worst = 0.0
        for x0, y0 in ((0.0, 0.0), (0.3 * a, -0.2 * b), (0.5 * a, 0.0), (0.0, 0.5 * b),
                       (0.35 * a, 0.35 * b)):
            def free(qx, qy, x0=x0, y0=y0):
                return -0.5 * np.log((qx - x0) ** 2 + (qy - y0) ** 2)
            c = pipe.coefficients(None, None, None, potential_at=free)
            h, _fx, _fy = harmonic_on_grid(c, pipe.scale, pipe.centre, px, py, potential=True)
            worst = max(worst, float(np.max(np.abs(free(px, py) + h))))
        assert pipe.fit_residual == pytest.approx(worst, rel=1e-3), (shape, a, b, order)
    qx, qy = _check_points("rectangle", 0.04, 0.01, 100)
    assert {(sx * 0.04, sy * 0.01) for sx in (-1, 1) for sy in (-1, 1)} <= set(zip(
        qx.tolist(), qy.tolist(), strict=True))
    qx, qy = _check_points("ellipse", 0.06, 0.025, 98)
    assert qx.size == 100 and qx.max() == 0.06 and qy.max() == pytest.approx(0.025, rel=1e-15)


def _phi_round_gaussian(r, s):
    """The potential -int rho ln|r - r'| of a round Gaussian of rms s (unit charge)."""
    if r == 0.0:
        return 0.5 * (np.euler_gamma - math.log(2 * s * s))
    return -0.5 * (float(exp1(r * r / (2 * s * s))) + math.log(r * r))


def test_particles_near_a_round_pipes_wall_get_the_images_at_their_place():
    # the images' field and potential at each particle, not interpolated from
    # nodes some of which lie beyond the wall: near the wall (0.99 to 0.999 of
    # the radius) a centred round beam's potential is Phi_free(r) - Phi_free(b)
    # (its images: a constant), an off-centre beam's field its free field plus
    # that of its image line charge at b^2/x0
    # (the beam inside the wall — at 5 rms sizes — as the analytic potential
    # assumes: a round pipe leaves charge beyond its wall out of its images;
    # one particle beyond the wall makes the grid reach past it, so that the
    # particles near the wall gather nodes on both sides of it)
    ref = RingReference.from_kinetic("proton", 800.0)
    b = 0.02
    rng = np.random.default_rng(31)
    n = 200_000
    radii = np.array([0.0, 0.99, 0.995, 0.999]) * b
    s = 4e-3
    X = np.zeros((6, n + radii.size))
    X[0, :n], X[2, :n] = rng.normal(0, s, n), rng.normal(0, s, n)
    X[4, :n] = rng.normal(0, 0.3, n)
    X[0, 0], X[2, 0] = -0.026, 0.0
    X[0, n:] = radii
    X[4, n:] = 0.3                          # (where the line density has a slope)
    cfg = {"pipe": "round", "pipe_a_m": b, "grid": [64, 64], "kicks_per_turn": 1,
           "z_bins": 64, "box_sigma": 6.0}
    _X1, pt, sc = _kick_once(cfg, X, 1e10 / n, ref=ref, length=5.0, rf_on=True)
    assert sc.last_box.x0 < -b < b + 2 * sc.last_box.dx < sc.last_box.x0 + 63 * sc.last_box.dx
    got = pt[n:] / pt[n]                    # (the same line-density slope for all)
    pb = _phi_round_gaussian(b, s)
    want = np.array([(_phi_round_gaussian(r, s) - pb) / (_phi_round_gaussian(0.0, s) - pb)
                     for r in radii])
    assert np.all(np.abs(got - want) < 3e-3), (got, want)
    # an off-centre beam: the field near the wall on its side
    s2, x0 = 2.5e-3, 0.01
    X = np.zeros((6, n + 3))
    X[0, :n], X[2, :n] = x0 + rng.normal(0, s2, n), rng.normal(0, s2, n)
    X[0, n:] = np.array([0.99, 0.995, 0.999]) * b
    X1, _pt, sc = _kick_once(dict(cfg, box_sigma=5.0, longitudinal="off"), X, 1e10 / n,
                             ref=ref, length=100.0, rf_on=False)
    k = 2 * classical_radius(ref.mass) * 1e10 * sc.ds[0] / 100.0 / (
        ref.beta0 ** 2 * ref.gamma0 ** 3)
    got = (X1[1] - X[1])[n:] / k
    d = X[0, n:] - x0
    free = (1.0 - np.exp(-d * d / (2 * s2 * s2))) / d
    image = -1.0 / (X[0, n:] - b * b / x0)
    assert np.allclose(got, free + image, rtol=0.01), (got, free + image)


def test_a_round_pipes_series_at_a_particle_needs_fewer_terms_nearer_the_centre():
    # each particle sums the terms its radius needs: the same field and
    # potential as the whole series (to its tolerance), far fewer terms at
    # the beam's core
    from linac_gen.ring.sc.kernels import boundary_field, round_terms
    b = 0.02
    rng = np.random.default_rng(13)
    X, Y = np.meshgrid(np.linspace(-0.015, 0.015, 41), np.linspace(-0.015, 0.015, 41),
                       indexing="ij")
    q = np.exp(-(X ** 2 + Y ** 2) / (2 * 0.004 ** 2))
    q[np.hypot(X, Y) > 0.0149] = 0.0
    q /= q.sum()
    c = RoundPipe(b).coefficients(q, X, Y)
    bnd = np.vstack([c.real, c.imag])
    nb = len(c) - 1
    assert nb > 40
    for r in (0.0, 1e-5, 0.002, 0.005, 0.01, 0.015, 0.019, 0.0199):
        t = rng.uniform(0.0, 2 * np.pi)
        x, y = r * math.cos(t), r * math.sin(t)
        n_p = round_terms(x, y, nb, 1.0 / b)
        assert 1 <= n_p <= nb
        full = boundary_field(x, y, bnd, nb, 1.0 / b, True)
        part = boundary_field(x, y, bnd, n_p, 1.0 / b, True)
        assert abs(part[0] - full[0]) <= 1e-11 / b and abs(part[1] - full[1]) <= 1e-11 / b, r
        assert abs(part[2] - full[2]) <= 1e-11, r
        if r <= 0.002:
            assert n_p <= nb // 3, (r, n_p, nb)



# ---------------------------------------------------------------------------
# unequal weights: the moments are the charge's
# ---------------------------------------------------------------------------
def _plain_moments(X, w, state, block):
    """The moments as plain sums of the offsets, block after block in order —
    what the kernels summed before the weights entered them."""
    tot = np.zeros(6)
    for lo in range(0, X.shape[1], block):
        s = [0.0] * 6
        for i in range(lo, min(lo + block, X.shape[1])):
            if state[i] == 1:
                x, y = float(X[0, i]), float(X[2, i])
                s[0] += 1.0
                s[1] += float(w[i])
                s[2] += x
                s[3] += y
                s[4] += x * x
                s[5] += y * y
        tot += np.array(s)
    n = tot[0]
    mx, my = tot[2] / n, tot[3] / n
    return {"n": int(n), "w": float(tot[1]), "x": mx, "y": my,
            "sx": math.sqrt(max(tot[4] / n - mx * mx, 0.0)),
            "sy": math.sqrt(max(tot[5] / n - my * my, 0.0))}


def test_with_equal_weights_the_moments_are_the_plain_sums_bit_for_bit():
    # (each weight enters relative to its block's largest: exactly 1 when the
    # weights are equal — weights of 0 included — so such a run keeps its bits)
    from linac_gen.ring.sc import kernels as K
    from linac_gen.ring.sc.beams import CpuBeam
    from linac_gen.ring.sc.solver import combine_moments
    rng = np.random.default_rng(17)
    n = 2 * K.MOMENT_BLOCK + 377
    X = np.zeros((6, n))
    X[0], X[2] = rng.normal(1e-3, 3e-3, n), rng.normal(-2e-4, 1e-3, n)
    st = np.ones(n, dtype=np.int8)
    st[rng.random(n) < 0.2] = 0
    for w0 in (1.0, 0.37, 6.6e6, 0.0):
        w = np.full(n, w0)
        rows = CpuBeam(X, np.zeros(n), np.ones(n), np.ones(n), st, w).moments(0.0, 0.0)
        got = combine_moments(rows, 0.0, 0.0)
        want = _plain_moments(X, w, st, K.MOMENT_BLOCK)
        assert all(got[k] == want[k] for k in want), (w0, got, want)


def _two_populations(n_a, s_a, w_a, n_b, s_b, w_b, length, x_b=0.0):
    rng = np.random.default_rng(3)
    X = np.zeros((6, n_a + n_b))
    X[0, :n_a], X[2, :n_a] = rng.normal(0, s_a, n_a), rng.normal(0, s_a, n_a)
    X[0, n_a:], X[2, n_a:] = rng.normal(x_b, s_b, n_b), rng.normal(0, s_b, n_b)
    X[4] = rng.uniform(-length / 2, length / 2, n_a + n_b)
    return X, np.concatenate([np.full(n_a, float(w_a)), np.full(n_b, float(w_b))])


def _kick_weighted(cfg, X, w, *, ref, length):
    """X (6, n) of weights ``w`` after one kick of a coasting beam; returns (X, sc)."""
    from linac_gen.ring.sc.beams import CpuBeam
    sc = SpaceCharge(cfg, _OneKickRing(length, ref), zwrap=length, rf_on=False)
    Y, pt, rp, rv, st = _beam_arrays(X.copy())
    sc.kick(0, CpuBeam(Y, pt, rp, rv, st, w))
    return Y, sc


def test_the_frozen_gaussian_and_the_grid_are_the_charges_with_unequal_weights():
    # two populations of different weights (a beam at the start and an
    # injected one, say): the frozen model's Gaussian has the charge's
    # centroid and rms sizes — counted as particles, its rms was 2.4 times
    # the charge's here ...
    ref = RingReference.from_kinetic("proton", 800.0)
    L = 5.0
    X, w = _two_populations(20000, 4e-3, 1e5, 20000, 1e-3, 3e6, L, x_b=5e-4)
    _Y, sc = _kick_weighted({"model": "frozen", "kicks_per_turn": 1, "grid": [64, 64]}, X, w,
                            ref=ref, length=L)
    x, y, sx, sy = sc._frozen_state["gauss"][0]
    xm, ym = np.average(X[0], weights=w), np.average(X[2], weights=w)
    assert x == pytest.approx(xm, rel=1e-9) and abs(y - ym) < 1e-15 + 1e-9 * abs(ym)
    assert sx == pytest.approx(math.sqrt(np.average((X[0] - xm) ** 2, weights=w)), rel=1e-9)
    assert sy == pytest.approx(math.sqrt(np.average((X[2] - ym) ** 2, weights=w)), rel=1e-9)
    assert sx < 0.45 * np.std(X[0])
    # ... and the particle-in-cell grid at 4 rms sizes holds the charge of a
    # few heavy particles far out: the field beyond them is the beam's (with
    # the grid sized by the particles' count, a third of the charge fell off
    # it and that field was 0.71 of the beam's)
    X, w = _two_populations(40000, 1e-3, 1e5, 2000, 6e-3, 5e6, L)
    out = {}
    for bs in (4.0, 1e6):                       # (1e6: the grid holds every particle)
        Y, sc = _kick_weighted({"kicks_per_turn": 1, "grid": [128, 128], "box_sigma": bs}, X, w,
                               ref=ref, length=L)
        out[bs] = (Y[1] - X[1], sc.n_outside_max)
    far = np.hypot(X[0], X[2]) > 15e-3
    assert out[4.0][1] <= 10 and far.sum() > 50
    assert np.median(out[4.0][0][far] / out[1e6][0][far]) == pytest.approx(1.0, abs=3e-3)


# ---------------------------------------------------------------------------
# the run's diagnostic counters
# ---------------------------------------------------------------------------
def test_the_diagnostic_counters_go_through_a_checkpoint_as_one_runs():
    """diagnostics_state / restore_diagnostics: kicks split by a checkpoint
    (the counters through JSON, the frozen model's measurements as a
    checkpoint keeps them) count as one run's — the most particles off the
    grid, the kicks with charge at or beyond a round pipe's wall (per kick,
    also in the frozen model, which solves its fields once per measurement)
    and the kicks made."""
    import json

    from linac_gen.ring.sc.beams import CpuBeam
    from linac_gen.ring.sc.solver import combine_moments
    ref = RingReference.from_kinetic("proton", 800.0)
    L = 5.0
    rng = np.random.default_rng(23)
    n = 4000
    X0 = np.zeros((6, n))
    X0[0], X0[2] = rng.normal(0, 3e-3, n), rng.normal(0, 2e-3, n)
    X0[4] = rng.uniform(-L / 2, L / 2, n)
    w = np.full(n, 1e6)
    scales = [1.0, 2.0, 0.5, 2.5, 0.6]

    def kicks(sc, which):
        for i in which:
            X = X0.copy()
            X[[0, 2]] *= scales[i]
            if i == 1:
                X[0, :400] = 0.02                   # (a halo off the grid: the most)
            beam = CpuBeam(*_beam_arrays(X), w)
            n_b = combine_moments(beam.moments(0.0, 0.0), 0.0, 0.0)["w"]
            sc.kick(0, beam, n_b=n_b, refresh=i == 0)

    for model, beyond in (("pic", 3), ("frozen", 5)):
        cfg = {"model": model, "pipe": "round", "pipe_a_m": 0.008, "grid": [32, 32],
               "kicks_per_turn": 1, "box_sigma": 2.0}

        def make(cfg=cfg):
            return SpaceCharge(cfg, _OneKickRing(L, ref), zwrap=L, rf_on=False)
        whole = make()
        kicks(whole, range(5))
        first = make()
        kicks(first, range(2))
        state = json.loads(json.dumps(first.diagnostics_state()))
        rest = make()
        rest.restore_frozen(first.frozen_arrays())
        rest.restore_diagnostics(state)
        kicks(rest, range(2, 5))
        d = whole.diagnostics_state()
        assert rest.diagnostics_state() == d, model
        assert d["kicks_beyond_wall"] == beyond and d["kicks_done"] == 5, (model, d)
        assert rest.summary()["kicks_charge_beyond_wall"] == beyond
        if model == "pic":
            assert d["outside_grid_max"] >= 400
        rest.restore_diagnostics(None)              # (a checkpoint without them)
        rest.restore_diagnostics({})
        assert rest.diagnostics_state() == d


def test_the_frozen_model_counts_the_kicks_with_charge_beyond_the_wall(booster, monkeypatch):
    # measured once (refresh_turns 0), its fields solved once a kick: every
    # kick of the 3 turns counted, as particle-in-cell counts them — also
    # with its fields solved again at every kick (no room to keep them)
    import linac_gen.ring.sc.solver as S
    beam = _booster_beam()
    beam.emit_nx = beam.emit_ny = 20.0
    sc = {"intensity": 2e10, "kicks_per_turn": 12, "pipe": "round", "pipe_a_m": 0.012,
          "grid": [32, 32], "model": "frozen", "refresh_turns": 0}
    cfg = RingConfig(n_turns=3, n_particles=500, tune_probes=0, space_charge=sc)
    got = [RingSimulation(booster, beam, cfg).run().meta["space_charge"]]
    monkeypatch.setattr(S, "FROZEN_CACHE_BYTES", 1)
    got.append(RingSimulation(booster, beam, cfg).run().meta["space_charge"])
    assert [m["kicks_charge_beyond_wall"] for m in got] == [36, 36]
    assert got[0]["kicks"] == 12


@pytest.mark.parametrize("model", ["pic", "frozen"])
def test_a_restarted_run_reports_the_uninterrupted_runs_counters(booster, tmp_path, model):
    # the counters go through the checkpoint (SpaceCharge.diagnostics_state):
    # the most particles off the grid and the kicks with charge at or beyond a
    # round pipe's wall
    beam = _booster_beam()
    beam.emit_nx = beam.emit_ny = 20.0
    sc = {"intensity": 2e10, "kicks_per_turn": 12, "pipe": "round", "pipe_a_m": 0.012,
          "grid": [32, 32], "box_sigma": 2.0, "model": model, "refresh_turns": 2}
    cfg = RingConfig(n_turns=6, n_particles=1000, tune_probes=0, space_charge=sc,
                     snapshot_turns=["last"], checkpoint_every=3)
    ck = tmp_path / "ck.npz"
    full = RingSimulation(booster, beam, cfg).run(checkpoint_path=ck)
    cont = RingSimulation(booster, beam, cfg).run(restart_from=ck)
    assert np.array_equal(full.snapshots[6]["coords"], cont.snapshots[6]["coords"])
    a, b = full.meta["space_charge"], cont.meta["space_charge"]
    assert a["kicks_charge_beyond_wall"] == b["kicks_charge_beyond_wall"] == 72
    assert a["outside_grid_max"] == b["outside_grid_max"]
    assert (a["outside_grid_max"] > 0) if model == "pic" else a["outside_grid_max"] is None
