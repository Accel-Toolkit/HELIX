"""The field of a 2-D Gaussian line charge (``pic_solver._gauss_field_2d``), used by the tune
footprint and by the DC kernel ``dc_kernel="gaussian"``.

The reference is the integral representation, evaluated by adaptive quadrature after the
substitution u = 1/(2 s_big² + q):

    E_x = λx/(2πε0) ∫_0^∞ exp(−x²/(2σx²+q) − y²/(2σy²+q)) / ((2σx²+q)^{3/2} (2σy²+q)^{1/2}) dq

(E_y with x ↔ y).  Before (2026-11-07): the Bassetti-Erskine closed form was evaluated with
signed coordinates, and on the side where Im z < 0 its two terms cancel to round-off times
exp((y²−x²)/K): beyond 2.3σ just past 5 % asymmetry, 3.2σ at a size ratio of 1.1, 7.8σ at 1.5
and 12σ at 2 the field was more than 1 % wrong, further out infinite or NaN; below 5 %
asymmetry the round-beam field stood in for it, up to 2.5 % off (3.5 % in a component), with a
2.6 % jump at the switch; and round-off crossed the axes, where the field across them is zero.
"""
from __future__ import annotations

import math

import numpy as np
import pytest
from scipy.integrate import quad

from linac_gen.pic import pic_solver as ps

EPS0 = 8.8541878128e-12
LAM = 2.0e-9
S = 1.0e-3
KS = (0.0, 1e-3, 0.1, 0.5, 1.0, 2.0, 3.0, 5.0, 8.0)          # in sigma; 1e-3: near the axes
KK = tuple(sorted(set(KS) | {-k for k in KS}))


def _ref(x, y, sx, sy, lam=LAM):
    sb2 = max(sx, sy) ** 2
    U = 1.0 / (2.0 * sb2)
    cx, cy = 2.0 * (sb2 - sx * sx), 2.0 * (sb2 - sy * sy)
    rho = x * x + y * y
    pts = [k / rho for k in (1.0, 5.0, 20.0, 60.0) if rho > 0 and k / rho < U] or None

    def f(u, px, py):
        ax, ay = 1.0 - cx * u, 1.0 - cy * u
        return math.exp(-x * x * u / ax - y * y * u / ay) / (ax ** px * ay ** py)

    ix = quad(f, 0.0, U, args=(1.5, 0.5), epsabs=0.0, epsrel=1e-13, limit=500, points=pts)[0]
    iy = quad(f, 0.0, U, args=(0.5, 1.5), epsabs=0.0, epsrel=1e-13, limit=500, points=pts)[0]
    pre = lam / (2.0 * math.pi * EPS0)
    return pre * x * ix, pre * y * iy


def _grid(ratio):
    sx, sy = ratio * S, S
    X, Y = np.meshgrid(np.array(KK) * sx, np.array(KK) * sy, indexing="ij")
    return sx, sy, X, Y


# sigma_x/sigma_y: round, near-round on the expansion side of the 1e-4 switch, just past it,
# the old 5 % switch on both sides, moderate and flat beams -- and every inverse (the swap)
_R = (1.0, 1 + 1e-6, 1 + 2e-5, 1 + 5e-5, 1 / (1 - 1.00001e-4), 1 + 2e-4, 1 + 1e-3, 1.01, 1.03,
      1.049, 1.0527, 1.1, 1.5, 2.0, 5.0, 10.0)
RATIOS = tuple(sorted(set(_R) | {1.0 / r for r in _R}))


@pytest.mark.parametrize("ratio", RATIOS)
def test_field_is_the_integral_everywhere(ratio):
    """In all four quadrants: 1e-10 of the local field where it is at least 1e-2 of the peak,
    5e-10 where at least 1e-3, and 1e-12 of the peak everywhere."""
    sx, sy, X, Y = _grid(ratio)
    ex, ey = ps._gauss_field_2d(X, Y, sx, sy, LAM)
    ref = np.array([[_ref(float(x), float(y), sx, sy) for x, y in zip(rx, ry, strict=True)]
                    for rx, ry in zip(X, Y, strict=True)])
    rx, ry = ref[..., 0], ref[..., 1]
    epk = np.hypot(rx, ry).max()
    for e, r in ((ex, rx), (ey, ry)):
        assert np.all(np.isfinite(e))
        rel = np.abs(e - r) / np.where(r != 0.0, np.abs(r), np.inf)
        assert np.max(rel[np.abs(r) >= 1e-2 * epk]) < 1e-10
        assert np.max(rel[np.abs(r) >= 1e-3 * epk]) < 5e-10
        assert np.max(np.abs(e - r)) < 1e-12 * epk


@pytest.mark.parametrize("asym", [0.5e-4, 0.999e-4, 1.001e-4, 1.05e-4, 2e-4])
def test_field_near_the_switch_at_random_points(asym):
    """The grid above misses where the closed form is weakest: just past the 1e-4 switch, near
    the centre (2.3-2.7e-10 where the field is 1e-3 of its peak, measured 2026-11-08).  Random
    points there, from the centre out to 8 sigma."""
    rng = np.random.default_rng(11)
    r = S * 10.0 ** rng.uniform(-4.0, math.log10(8.0), 400)
    th = rng.uniform(0.0, math.pi / 2.0, 400)
    X, Y = r * np.cos(th), r * np.sin(th)
    sx, sy = S / (1.0 - asym), S
    ex, ey = ps._gauss_field_2d(X, Y, sx, sy, LAM)
    ref = np.array([_ref(float(x), float(y), sx, sy) for x, y in zip(X, Y, strict=True)])
    epk = np.hypot(ref[:, 0], ref[:, 1]).max()
    for e, rr in ((ex, ref[:, 0]), (ey, ref[:, 1])):
        rel = np.abs(e - rr) / np.where(rr != 0.0, np.abs(rr), np.inf)
        assert np.max(rel[np.abs(rr) >= 1e-2 * epk]) < 1e-10
        assert np.max(rel[np.abs(rr) >= 1e-3 * epk]) < 5e-10
        assert np.max(np.abs(e - rr)) < 1e-12 * epk


@pytest.mark.parametrize("ratio", [1.0, 1.0 + 5e-5, 1.1])
def test_a_scalar_position_gives_the_field_there(ratio):
    """Both branches take a scalar (or 0-d) position as well as an array."""
    x, y = 0.7 * S, -1.3 * S
    want = [float(v[0]) for v in ps._gauss_field_2d(np.array([x]), np.array([y]), ratio * S, S,
                                                     LAM)]
    for xx, yy in ((x, y), (np.float64(x), np.float64(y)), (np.array(x), np.array(y))):
        got = ps._gauss_field_2d(xx, yy, ratio * S, S, LAM)
        assert [float(v) for v in got] == pytest.approx(want, rel=1e-14, abs=0.0)


@pytest.mark.parametrize("ratio", RATIOS)
def test_field_has_the_exact_symmetry(ratio):
    """E_x odd in x and even in y, E_y odd in y and even in x -- bit for bit; zero across the
    axes (E_x at x = 0, E_y at y = 0), exactly."""
    sx, sy, X, Y = _grid(ratio)
    ex, ey = ps._gauss_field_2d(X, Y, sx, sy, LAM)
    assert np.array_equal(ex, -ex[::-1, :]) and np.array_equal(ex, ex[:, ::-1])
    assert np.array_equal(ey, -ey[:, ::-1]) and np.array_equal(ey, ey[::-1, :])
    i0 = KK.index(0.0)
    assert np.all(ex[i0, :] == 0.0) and np.all(ey[:, i0] == 0.0)


def test_field_is_continuous_through_round():
    """sigma_x/sigma_y swept through 1 at fixed points: within 1e-10 of the peak of the
    reference everywhere, across the 1e-4 switch and the old 5 % one."""
    pts = [(0.3, 0.2), (-1.7, -0.083), (2.0, -3.0), (-5.0, 1.0), (-1.0, -1.0), (0.0, -2.0)]
    ratios = np.concatenate([np.linspace(0.94, 1.06, 121),
                             1 + np.array([-1.0001e-4, -0.9999e-4, 0.9999e-4, 1.0001e-4]),
                             1 / (1 - np.array([0.05 - 1e-9, 0.05 + 1e-9]))])
    for kx, ky in pts:
        x, y = kx * S, ky * S
        got = np.array([np.array(ps._gauss_field_2d(np.array([x]), np.array([y]), r * S, S,
                                                    LAM))[:, 0]
                        for r in ratios])
        ref = np.array([_ref(x, y, r * S, S) for r in ratios])
        assert np.max(np.abs(got - ref)) < 1e-10 * np.abs(ref).max()


@pytest.mark.parametrize("ratio", (1.0527, 1.0 / 1.0527, 3.0))
def test_far_field_is_gauss_law(ratio):
    """1000 sigma out in all four quadrants: the multipole series of a Gaussian line charge,
    E_r 2πε0 r/λ = 1 + D cos2θ/r² + 3 D² cos4θ/r⁴, D = σx² − σy² (an anchor that does not
    use the quadrature)."""
    sx, sy = ratio * S, S
    D = sx * sx - sy * sy
    for th in np.deg2rad([20.0, 110.0, 200.0, 290.0, 270.0]):
        r = 1000.0 * S
        x, y = r * math.cos(th), r * math.sin(th)
        ex, ey = (float(v[0])
                  for v in ps._gauss_field_2d(np.array([x]), np.array([y]), sx, sy, LAM))
        er = (ex * x + ey * y) / r
        want = 1.0 + D * math.cos(2.0 * th) / r ** 2 + 3.0 * D * D * math.cos(4.0 * th) / r ** 4
        assert er * 2.0 * math.pi * EPS0 * r / LAM == pytest.approx(want, rel=1e-10)


def test_dc_gauss_kick_of_a_near_round_beam_is_the_field():
    """The DC kick of a 7 %-asymmetric beam, the band where one side of the beam blew up beyond
    about 3σ, is q E ds/(β²γ m c²) of the reference field for every particle, tails included."""
    from linac_gen.core.beam import Beam
    from linac_gen.core.constants import C_LIGHT
    from linac_gen.core.particle import H_MINUS
    from linac_gen.core.reference import ReferenceParticle
    rng = np.random.default_rng(5)
    n = 4000
    b = Beam(ref=ReferenceParticle(species=H_MINUS, w_kin=0.030, frequency=162.5),
             n_particles=n, current=5.0)
    b.continuous = True
    b.particles[:, 0] = rng.normal(0.0, 1.07, n)
    b.particles[:, 2] = rng.normal(0.0, 1.00, n)
    xp0 = b.particles[:, [1, 3]].copy()
    ps.kick_continuous_2d_gauss(b, 10.0)
    kick = b.particles[:, [1, 3]] - xp0
    xs, ys = b.particles[:, 0], b.particles[:, 2]
    cx, cy = xs.mean(), ys.mean()
    sx, sy = xs.std() * 1e-3, ys.std() * 1e-3
    ref = b.ref
    lam = 5.0e-3 / (ref.beta * C_LIGHT)
    pre = (abs(ref.species.charge) * ps.E_CHARGE * 10.0e-3
           / (ref.beta ** 2 * ref.gamma * ref.species.mass * 1e6 * ps.E_CHARGE))
    r = np.hypot((xs - cx) / 1.07, ys - cy)
    pick = np.concatenate([np.argsort(r)[-40:], rng.choice(n, 40, replace=False)])
    for i in pick:
        ex, ey = _ref((xs[i] - cx) * 1e-3, (ys[i] - cy) * 1e-3, sx, sy, lam)
        assert kick[i, 0] == pytest.approx(pre * ex * 1e3, rel=1e-9, abs=1e-15)
        assert kick[i, 1] == pytest.approx(pre * ey * 1e3, rel=1e-9, abs=1e-15)


@pytest.mark.parametrize("continuous, dc_kernel, called", [
    (True, "uniform", False), (True, "pic2d", False), (False, "uniform", False),
    (False, "gaussian", False), (True, "gaussian", True)])
@pytest.mark.parametrize("walk", ["forward", "backward"])
def test_only_the_gaussian_dc_kernel_evaluates_the_field(monkeypatch, continuous, dc_kernel,
                                                         called, walk):
    """A pin, not a fix: bunched beams (with dc_kernel="gaussian" too) and the other DC kernels
    never reach this field, forward or in the backward walk, so the change above leaves them as
    they were."""
    from linac_gen.core.config import BeamConfig, SpaceChargeConfig
    from linac_gen.core.lattice import Lattice
    from linac_gen.core.simulation import Simulation
    from linac_gen.distributions.factory import create_beam
    from linac_gen.elements.drift import Drift
    from linac_gen.elements.quadrupole import Quadrupole
    n = [0]
    real = ps._gauss_field_2d

    def spy(*a, **k):
        n[0] += 1
        return real(*a, **k)
    monkeypatch.setattr(ps, "_gauss_field_2d", spy)
    lat = Lattice()
    for g in (8.0, -8.0):
        lat.add(Drift(name="D", length=100.0, aperture=30.0))
        lat.add(Quadrupole(name="Q", length=50.0, gradient=g, aperture=30.0))
    cfg = BeamConfig(species="H-", energy=0.030 if continuous else 2.1, frequency=162.5,
                     current=5.0, n_particles=2000, continuous=continuous, emit_nx=0.2,
                     alpha_x=0.0, beta_x=0.5, emit_ny=0.2, alpha_y=0.0, beta_y=0.6, emit_z=0.06,
                     alpha_z=0.0, beta_z=500.0)
    beam = create_beam(cfg, seed=3)
    sc = SpaceChargeConfig(nx=32, ny=32, nz=32, dc_kernel=dc_kernel)
    if walk == "forward":
        Simulation(lat, beam, space_charge=sc).run()
    else:
        import warnings

        from linac_gen.tracking.backtrack import backtrack_distribution
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            backtrack_distribution(lat, beam, beam.ref.copy(), sc_config=sc)
    assert (n[0] > 0) == called
