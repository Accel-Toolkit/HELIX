"""Time loop: drift exactness, energy conservation, losses, stragglers."""
from __future__ import annotations

import math

import numpy as np
import pytest

from linac_gen.rfq_td.fieldmap import RfqFieldMap
from linac_gen.rfq_td.geometry import RfqElectrodes
from linac_gen.rfq_td.tracker import (REASON_METAL, REASON_STALLED,
                                      TimeDomainTracker)

MC2 = 938.272       # proton, MeV


def _map(phi_fn, h=0.1, nx=81, z0=-30.0, z1=130.0, hz=0.5):
    xs = h * np.arange(nx)
    zs = np.arange(z0 - 2 * hz, z1 + 3 * hz - 1e-9, hz)
    X, Y, Z = np.meshgrid(xs, xs, zs, indexing="ij")
    phi = phi_fn(X, Y, Z).astype(np.float32)
    return RfqFieldMap.from_blocks(h, [(z0, z1)], [(float(zs[0]), hz, phi)])


def _electrodes(R=5.0, rho=4.0, z0=0.0, z1=100.0):
    z = np.linspace(z0, z1, 11)
    return RfqElectrodes(z=z, R_x=np.full(11, R), R_y=np.full(11, R),
                         rho_x=np.full(11, rho), rho_y=np.full(11, rho),
                         V_x=-1.0, V_y=1.0)


def _start(n, W_mev, rng):
    g = 1 + W_mev / MC2
    bg = math.sqrt(g * g - 1)
    xp = rng.normal(0, 2e-3, n)
    yp = rng.normal(0, 2e-3, n)
    uz = bg / np.sqrt(1 + xp ** 2 + yp ** 2)
    return (rng.normal(0, 0.5, n), rng.normal(0, 0.5, n),
            np.full(n, -10.0), uz * xp, uz * yp, uz.copy(), xp, yp)


def test_field_free_drift_is_exact():
    rng = np.random.default_rng(1)
    X, Y, Z, ux, uy, uz, xp, yp = _start(500, 0.05, rng)
    x0, y0 = X.copy(), Y.copy()
    m = _map(lambda X, Y, Z: 0.0 * X)
    el = _electrodes(R=50.0, rho=4.0)       # far away: no losses
    trk = TimeDomainTracker(m, el, freq_mhz=162.5, mass_mev=MC2, charge=1.0,
                            steps_per_period=24)
    planes = np.array([10.0, 37.5, 90.0])
    res = trk.run(X, Y, Z, ux, uy, uz, 0.0, planes)
    assert not res.lost.any()
    for k, zp in enumerate(planes):
        dz = zp - (-10.0)
        np.testing.assert_allclose(res.rec[k, :, 0], x0 + xp * dz,
                                   rtol=0, atol=1e-11)
        np.testing.assert_allclose(res.rec[k, :, 2], y0 + yp * dz,
                                   rtol=0, atol=1e-11)
        np.testing.assert_allclose(res.rec[k, :, 1], xp, rtol=1e-12)
        np.testing.assert_allclose(res.rec[k, :, 5], 0.05, rtol=1e-12)


def test_static_field_conserves_energy():
    """sin(omega t + 90 deg) ~ 1 for a tiny frequency: an electrostatic
    quadrupole + axial field; W + q phi is conserved to O(dt^2)."""
    a, b = 3.0, 0.4                     # V/mm^2, V/mm
    phi = lambda X, Y, Z: a * (X * X - Y * Y) + b * Z
    m = _map(phi)
    el = _electrodes(R=50.0)
    rng = np.random.default_rng(2)
    X, Y, Z, ux, uy, uz, _, _ = _start(200, 0.05, rng)
    w0 = 0.05 + phi(X, Y, Z) * 1e-6             # MeV (q = +1)
    trk = TimeDomainTracker(m, el, freq_mhz=1e-6, mass_mev=MC2, charge=1.0,
                            steps_per_period=4_000_000_000, phi0_deg=90.0)
    # T = 1e9 ns, dt = T/steps = 0.25 ns; sin(omega t + 90 deg) = 1 - O(1e-14)
    res = trk.run(X, Y, Z, ux, uy, uz, 0.0, np.array([50.0]))
    ok = ~res.lost
    xe, ye, We = res.rec[0, ok, 0], res.rec[0, ok, 2], res.rec[0, ok, 5]
    w1 = We + phi(xe, ye, 50.0) * 1e-6
    assert ok.sum() > 150
    np.testing.assert_allclose(w1, w0[ok], rtol=0, atol=2e-7)


def test_metal_and_straggler_losses():
    m = _map(lambda X, Y, Z: 0.0 * X)
    el = _electrodes(R=5.0, rho=4.0)
    g = 1 + 0.05 / MC2
    bg = math.sqrt(g * g - 1)
    X = np.array([0.0, 6.0, 0.0, 3.0])        # #1 starts inside the +x pole body
    Y = np.array([0.0, 0.0, 6.0, 3.0])        # #2 inside +y; #3 in the open corner
    Z = np.array([5.0, 5.0, 5.0, 5.0])
    ux = np.zeros(4); uy = np.zeros(4); uz = np.full(4, bg)
    trk = TimeDomainTracker(m, el, freq_mhz=162.5, mass_mev=MC2, charge=1.0,
                            steps_per_period=24)
    res = trk.run(X, Y, Z, ux, uy, uz, 0.0, np.array([90.0]))
    assert list(res.lost) == [False, True, True, False]
    assert list(res.loss_reason[[1, 2]]) == [REASON_METAL, REASON_METAL]
    # a particle that cannot reach the exit before t_max is a straggler
    X = np.zeros(1); Y = np.zeros(1); Z = np.zeros(1)
    ux = np.zeros(1); uy = np.zeros(1); uz = np.full(1, bg)
    trk = TimeDomainTracker(m, el, freq_mhz=162.5, mass_mev=MC2, charge=1.0,
                            steps_per_period=24, t_max_ns=5.0)
    res = trk.run(X, Y, Z, ux, uy, uz, 0.0, np.array([90.0]))
    assert res.lost[0] and res.loss_reason[0] == REASON_STALLED


def test_electrode_geometry_test():
    el = _electrodes(R=5.0, rho=4.0)
    # tip at (5,0); tip circle centre (9,0) radius 4; body |y|<=4 beyond x=9
    pts = np.array([[4.9, 0.0], [5.1, 0.0], [9.5, 3.9], [9.5, 4.1],
                    [0.0, 5.1], [3.5, 3.5], [-5.1, 0.0], [0.0, -5.1]])
    inside = el.inside(pts[:, 0], pts[:, 1], np.full(len(pts), 50.0))
    assert list(inside) == [False, True, True, False, True, False, True, True]
    # no vane metal outside the tabulated span
    assert not el.inside(np.array([6.0]), np.array([0.0]), np.array([-1.0]))[0]


def _modulated_electrodes():
    """A short modulated RFQ-like table with end plates (for exactness
    tests of the metal test)."""
    from linac_gen.rfq_td.geometry import EndPlate
    z = np.linspace(0.0, 100.0, 801)
    c = np.cos(np.pi * z / 9.5)
    return RfqElectrodes(z=z, R_x=3.0 * (1 + 0.1 * c),
                         R_y=3.0 * (1 - 0.1 * c),
                         rho_x=np.full(z.size, 2.25),
                         rho_y=np.full(z.size, 2.25), V_x=-1.0, V_y=1.0,
                         plates=[EndPlate(-5.0, -1), EndPlate(105.0, +1)])


def test_metal_test_prefilter_is_exact():
    """Only points that can be in metal get the full signed-distance test;
    the answer is the full test's everywhere, including points on the
    tips, on the bin edges of the prefilter and on the plate faces."""
    el = _modulated_electrodes()
    rng = np.random.default_rng(4)
    n = 400_000
    z = rng.uniform(-12.0, 112.0, n)
    r = np.abs(rng.normal(0.0, 3.0, n))
    th = rng.uniform(0.0, 2 * np.pi, n)
    x, y = r * np.cos(th), r * np.sin(th)
    d, _, _ = el.sdf_quadrant(np.abs(x), np.abs(y), z)
    assert 0 < (d < 0).sum() < n
    np.testing.assert_array_equal(el.inside(x, y, z), d < 0)
    z0, inv_dz, rm2 = el._bore()
    edges = z0 + np.arange(rm2.size + 1) / inv_dz
    zz = np.concatenate([edges, np.nextafter(edges, np.inf),
                         np.nextafter(edges, -np.inf), el.z])
    zz = zz[(zz >= el.z[0]) & (zz <= el.z[-1])]
    for R in (el.R_x, el.R_y):
        Rz = np.interp(zz, el.z, R)
        for eps in (-1e-9, -1e-13, 0.0, 1e-13, 1e-9):
            for xs, ys in ((Rz + eps, 0 * zz), (0 * zz, Rz + eps)):
                dd, _, _ = el.sdf_quadrant(xs, ys, zz)
                np.testing.assert_array_equal(el.inside(xs, ys, zz), dd < 0)
    for zf in (-5.0, -7.5, -5.0 - 1e-13, 105.0, 107.5, 105.0 + 1e-13):
        xs = np.linspace(0.0, 10.0, 2001)
        dd, _, _ = el.sdf_quadrant(xs, 0 * xs, np.full_like(xs, zf))
        np.testing.assert_array_equal(el.inside(xs, 0 * xs,
                                                np.full_like(xs, zf)), dd < 0)


def _quad_run(monkeypatch, mode):
    """A focusing channel with an axial ripple, 24 steps per period:
    ``mode`` = "kernel" (fused compiled step), "gather" (numpy step with
    the compiled gather) or "numpy" (everything numpy)."""
    from linac_gen.rfq_td import fieldmap
    if mode == "numpy":
        monkeypatch.setattr(fieldmap, "_KERNEL", False)
    elif mode == "gather":
        monkeypatch.setattr(fieldmap.RfqFieldMap, "push",
                            lambda self, *a, **k: None)
    a, b = 2.0e3, 3.0e2
    m = _map(lambda X, Y, Z: a * (X * X - Y * Y) * 1e-2
             + b * np.sin(2 * np.pi * Z / 9.5), nx=61)
    el = _modulated_electrodes()
    rng = np.random.default_rng(6)
    X, Y, Z, ux, uy, uz, _, _ = _start(3000, 0.05, rng)
    trk = TimeDomainTracker(m, el, freq_mhz=162.5, mass_mev=MC2, charge=1.0,
                            steps_per_period=24)
    return trk.run(X, Y, Z, ux, uy, uz, 0.0, np.linspace(10.0, 100.0, 7))


def test_fused_kernel_equals_the_numpy_step(monkeypatch):
    from linac_gen.rfq_td import fieldmap
    k = fieldmap._kernel()
    if not k or not hasattr(k, "push"):
        pytest.skip("compiled time-domain RFQ kernel not built")
    fused = _quad_run(monkeypatch, "kernel")
    with monkeypatch.context() as mp:
        step = _quad_run(mp, "gather")
    with monkeypatch.context() as mp:
        ref = _quad_run(mp, "numpy")
    assert 0 < fused.lost.sum() < fused.lost.size
    # the fused step is the numpy step with the compiled gather, bit for bit
    np.testing.assert_array_equal(fused.lost, step.lost)
    np.testing.assert_array_equal(fused.rec, step.rec)
    np.testing.assert_array_equal(fused.loss_z, step.loss_z)
    # and agrees with the all-numpy path to rounding
    np.testing.assert_array_equal(fused.lost, ref.lost)
    np.testing.assert_allclose(fused.rec, ref.rec, rtol=1e-10, atol=1e-12)


def test_leapfrog_is_time_reversible():
    """Kick-drift-kick in a static field: N steps forward, velocities
    reversed, N steps back returns every particle to its start (to
    rounding)."""
    from linac_gen.rfq_td import fieldmap
    a, b = 2.0e3, 3.0e2
    m = _map(lambda X, Y, Z: a * (X * X - Y * Y) * 1e-2
             + b * np.sin(2 * np.pi * Z / 9.5), nx=61)
    rng = np.random.default_rng(9)
    n = 400
    X, Y = rng.normal(0, 0.5, n), rng.normal(0, 0.5, n)
    Z = rng.uniform(0.0, 20.0, n)
    ux, uy = rng.normal(0, 1e-4, n), rng.normal(0, 1e-4, n)
    uz = np.full(n, 0.0103)
    start = [v.copy() for v in (X, Y, Z)]
    act = np.ones(n, bool)
    c = 1e-6                              # dt/2 * field factor (static)
    cdt = 299.792458 * 0.05

    def steps(k, X, Y, Z, ux, uy, uz):
        Ex, Ey, Ez, _ = m.field(X, Y, Z, act)
        for _ in range(k):
            if m.push(X, Y, Z, ux, uy, uz, act, Ex, Ey, Ez, c, c,
                      cdt) is None:
                pytest.skip("compiled time-domain RFQ kernel not built")

    if not fieldmap._kernel():
        pytest.skip("compiled time-domain RFQ kernel not built")
    steps(400, X, Y, Z, ux, uy, uz)
    assert np.abs(Z - start[2]).max() > 1.0          # it moved
    ux *= -1.0
    uy *= -1.0
    uz *= -1.0
    steps(400, X, Y, Z, ux, uy, uz)
    for v, v0 in zip((X, Y, Z), start):
        np.testing.assert_allclose(v, v0, rtol=0, atol=1e-9)


def test_long_rf_quadrupole_channel_keeps_the_rms_emittance():
    """An unmodulated RF quadrupole channel (linear in x, y) over ~60 RF
    periods: particles that cross a plane at one RF phase all see the same
    linear map, so the rms emittance at the exit equals the entrance's to
    integration accuracy (< 0.5 %)."""
    a = 0.4e3                                       # V/mm^2 at the crest
    m = _map(lambda X, Y, Z: a * (X * X - Y * Y), nx=81, z1=900.0, hz=2.0)
    el = _electrodes(R=50.0, rho=4.0, z1=900.0)
    rng = np.random.default_rng(10)
    n = 3000
    g = 1 + 0.05 / MC2
    bg = math.sqrt(g * g - 1)
    xp = rng.normal(0, 1.5e-3, n)
    yp = rng.normal(0, 1.5e-3, n)
    X = rng.normal(0, 0.6, n)
    Y = rng.normal(0, 0.6, n)
    uz = bg / np.sqrt(1 + xp ** 2 + yp ** 2)
    Z = np.full(n, -1.0)                            # one bunch, one phase
    trk = TimeDomainTracker(m, el, freq_mhz=162.5, mass_mev=MC2, charge=1.0,
                            steps_per_period=48)
    planes = np.array([0.0, 850.0])
    res = trk.run(X, Y, Z, uz * xp, uz * yp, uz, 0.0, planes)
    assert not res.lost.any()

    def emit(k, i, j):
        c = np.cov(res.rec[k, :, i].astype(float),
                   res.rec[k, :, j].astype(float))
        return math.sqrt(np.linalg.det(c))

    for i, j in ((0, 1), (2, 3)):
        e0, e1 = emit(0, i, j), emit(1, i, j)
        assert abs(e1 / e0 - 1.0) < 5e-3, (i, e0, e1)
