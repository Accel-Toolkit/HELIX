"""Periodic-z / open-xy space charge against analytic fields."""
from __future__ import annotations

import numpy as np
from scipy.special import erf

from linac_gen.rfq_td.space_charge import EPS0_MM, PeriodicTrainSpaceCharge

F_MHZ = 162.5
LAM = 299.792458 / (F_MHZ * 1e-3)          # mm
Q_PERIOD = 5e-3 / (F_MHZ * 1e6)            # C per RF period at 5 mA


def test_dc_round_gaussian_beam():
    """Uniform in z over the period: the field is the 2-D field of a
    Gaussian line charge, E_r = lambda/(2 pi eps0 r) (1 - exp(-r^2/2s^2))."""
    rng = np.random.default_rng(1)
    n, beta = 300_000, 0.008
    P = beta * LAM
    x = rng.normal(0, 1.0, n)
    y = rng.normal(0, 1.0, n)
    z = rng.uniform(0, P, n)
    sc = PeriodicTrainSpaceCharge(Q_PERIOD / n, LAM, nx=64, extent=5.0)
    ex, ey, ez = sc.field(x, y, z, np.full(n, beta))
    lam_line = Q_PERIOD / P
    r = np.hypot(x, y)
    er_th = lam_line / (2 * np.pi * EPS0_MM * r) * (1 - np.exp(-r ** 2 / 2))
    er = (ex * x + ey * y) / r
    m = (r > 0.3) & (r < 3.5)
    rel = (er[m] - er_th[m]) / er_th[m].max()
    assert np.sqrt(np.mean(rel ** 2)) < 0.02
    assert ez.std() < 0.02 * er_th.max()          # no longitudinal field


def test_periodic_train_of_gaussian_bunches():
    """Spherical Gaussian bunches every P = 8 sigma: direct sum over
    +-400 images of the analytic single-bunch field."""
    rng = np.random.default_rng(2)
    n, P = 300_000, 8.0
    beta = P / LAM
    x, y, z = rng.normal(0, 1.0, (3, n))
    z = np.mod(z, P)
    sc = PeriodicTrainSpaceCharge(Q_PERIOD / n, LAM, nx=64, extent=5.0)
    ex, ey, ez = sc.field(x, y, z, np.full(n, beta))
    zc = (z + P / 2) % P - P / 2

    def e_sph(dx, dy, dz):
        R = np.sqrt(dx * dx + dy * dy + dz * dz) + 1e-30
        g = Q_PERIOD / (4 * np.pi * EPS0_MM * R ** 2) * (
            erf(R / np.sqrt(2)) - np.sqrt(2 / np.pi) * R * np.exp(-R * R / 2))
        return g * dx / R, g * dy / R, g * dz / R

    sel = rng.choice(n, 2000, replace=False)
    t = np.zeros((3, sel.size))
    for m in range(-400, 401):
        t += np.array(e_sph(x[sel], y[sel], zc[sel] - m * P))
    scale = np.abs(t[0]).max()
    for got, ref in zip((ex[sel], ey[sel], ez[sel]), t):
        assert np.sqrt(np.mean((got - ref) ** 2)) / scale < 0.02


def test_longitudinal_resolution_follows_bunch_length():
    rng = np.random.default_rng(4)
    n, beta = 20_000, 0.05
    P = beta * LAM
    x = rng.normal(0, 1, n)
    y = rng.normal(0, 1, n)
    sc = PeriodicTrainSpaceCharge(1e-17, LAM)
    sc.field(x, y, rng.uniform(0, P, n), np.full(n, beta))      # DC
    nz_dc = sc.last["nz"]
    sc.field(x, y, rng.normal(0, 0.02 * P, n), np.full(n, beta))  # short bunch
    nz_short = sc.last["nz"]
    assert nz_dc == 32 and nz_short > nz_dc


def test_green_functions_are_reused():
    rng = np.random.default_rng(6)
    n, beta = 20_000, 0.02
    P = beta * LAM
    x = rng.normal(0, 1, n)
    y = rng.normal(0, 1, n)
    z = rng.uniform(0, P, n)
    sc = PeriodicTrainSpaceCharge(1e-17, LAM)
    for _ in range(5):
        sc.field(x * (1 + 0.003 * rng.normal()), y, z, np.full(n, beta))
    assert sc.n_green == 1 and sc.n_solve == 5


def test_kick_is_repulsive_for_either_sign():
    """|q|E: an off-axis particle is pushed outwards."""
    rng = np.random.default_rng(7)
    n, beta = 50_000, 0.01
    P = beta * LAM
    X = rng.normal(0, 1, n); Y = rng.normal(0, 1, n); Z = rng.uniform(0, P, n)
    g = 1.00005
    uz = np.full(n, np.sqrt(g * g - 1)); ux = np.zeros(n); uy = np.zeros(n)
    sc = PeriodicTrainSpaceCharge(Q_PERIOD / n, LAM)
    sc.kick(X, Y, Z, ux, uy, uz, np.ones(n, bool), 1.0, 3.19e-7)
    far = np.hypot(X, Y) > 1.5
    assert np.all(ux[far] * X[far] + uy[far] * Y[far] > 0)


def test_cached_spectra_are_guarded_and_solves_repeat_bitwise():
    """The Green's-function spectra sit inside an allocation one element
    larger on each side and the product is taken in place, so numpy's
    complex multiply always takes its vectorised loop: repeated solves of
    one beam are bitwise equal while spectrum-sized arrays come and go
    (a PXIE run differed at the 1e-11 level between identical runs before).
    """
    rng = np.random.default_rng(5)
    n, beta = 20_000, 0.03
    P = beta * LAM
    x, y = rng.normal(0, 1.0, n), rng.normal(0, 1.0, n)
    z = rng.normal(0, 0.1 * P, n)
    sc = PeriodicTrainSpaceCharge(Q_PERIOD / n, LAM, nx=32, extent=5.0)
    first = sc.field(x, y, z, np.full(n, beta))
    (G, _ks), = sc._cache.values()
    assert G.base is not None and G.base.ndim == 1
    assert G.base.size == G.size + 2
    churn = []
    for i in range(60):
        churn = [np.empty(G.shape, complex) for _ in range(i % 4)]
        again = sc.field(x, y, z, np.full(n, beta))
        assert all(np.array_equal(a, b) for a, b in zip(first, again))
    del churn
