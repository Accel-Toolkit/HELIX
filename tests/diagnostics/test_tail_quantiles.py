"""Tail diagnostics (HALO-PIC M1): analytic pins for fractional emittance
and radial quantiles.

External anchors (not round-trips): for a matched Gaussian phase plane the
CS action satisfies J/eps_rms ~ chi^2_2, so eps_q/eps_rms = -ln(1-q); the
normalized radius r^2 = (x/sx)^2+(y/sy)^2 ~ chi^2_2, so r_q = sqrt(-2 ln(1-q)).
"""
from __future__ import annotations

import numpy as np
import pytest

from linac_gen.diagnostics.tail import (
    compute_fractional_emittance,
    compute_radial_quantiles,
    cs_actions,
    frac_key,
)


def _gaussian_beam(n=400_000, seed=7, alpha=1.3, beta=4.2, emit=2.5):
    """Correlated Gaussian (x, x') with prescribed Twiss; other columns
    filled with an independent Gaussian pair for y/y'."""
    rng = np.random.default_rng(seed)
    gamma_t = (1 + alpha**2) / beta
    cov = emit * np.array([[beta, -alpha], [-alpha, gamma_t]])
    L = np.linalg.cholesky(cov)
    p = np.zeros((n, 6))
    p[:, 0:2] = rng.standard_normal((n, 2)) @ L.T
    p[:, 2:4] = rng.standard_normal((n, 2)) @ L.T
    return p


def test_mean_action_is_twice_rms_emittance():
    p = _gaussian_beam()
    J = cs_actions(p, "x")
    from linac_gen.diagnostics.moments import compute_emittance
    eps = compute_emittance(p, "x")
    assert np.mean(J) == pytest.approx(2.0 * eps, rel=1e-3)


@pytest.mark.parametrize("q, factor", [(0.99, -np.log(0.01)),
                                       (0.999, -np.log(0.001))])
def test_gaussian_fractional_emittance_pin(q, factor):
    p = _gaussian_beam()
    from linac_gen.diagnostics.moments import compute_emittance
    eps = compute_emittance(p, "x")
    out = compute_fractional_emittance(p, (q,), "x")
    # 400k samples: the 99.9% quantile carries ~2-3% sampling error
    tol = 0.01 if q <= 0.99 else 0.03
    assert out[q] == pytest.approx(factor * eps, rel=tol)


def test_gaussian_radial_quantile_pin():
    p = _gaussian_beam()
    out = compute_radial_quantiles(p, (0.99, 0.999))
    assert out[0.99] == pytest.approx(np.sqrt(-2 * np.log(0.01)), rel=0.01)
    assert out[0.999] == pytest.approx(np.sqrt(-2 * np.log(0.001)), rel=0.03)


def test_bi_gaussian_mixture_tail_mass():
    """Core + 5% halo at 4x rms: eps_99 must sit far above the pure-core
    value (the tail carries the quantile) — a discrimination test that a
    core-only metric would fail."""
    rng = np.random.default_rng(11)
    n_core, n_halo = 190_000, 10_000
    p = np.zeros((n_core + n_halo, 6))
    p[:n_core, 0:2] = rng.standard_normal((n_core, 2))
    p[n_core:, 0:2] = 4.0 * rng.standard_normal((n_halo, 2))
    pure = _gaussian_beam(n=200_000, seed=12, alpha=0.0, beta=1.0, emit=1.0)
    e_mix = compute_fractional_emittance(p, (0.999,), "x")[0.999]
    e_pure = compute_fractional_emittance(pure, (0.999,), "x")[0.999]
    from linac_gen.diagnostics.moments import compute_emittance
    # normalize by each beam's own rms emittance
    r_mix = e_mix / compute_emittance(p, "x")
    r_pure = e_pure / compute_emittance(pure, "x")
    assert r_mix > 1.5 * r_pure


def test_weighted_equals_replication():
    """A particle with weight w must count exactly like w unit-weight
    replicas (the M5-readiness identity)."""
    rng = np.random.default_rng(3)
    base = _gaussian_beam(n=20_000, seed=3)
    idx = rng.choice(len(base), size=2_000, replace=False)
    # replicate the chosen particles 3x
    replicated = np.vstack([base, base[idx], base[idx]])
    weights = np.ones(len(base))
    weights[idx] = 3.0
    for fn, kwargs in ((compute_fractional_emittance, {"plane": "x"}),
                       (compute_radial_quantiles, {})):
        a = fn(replicated, (0.99, 0.999), **kwargs)
        b = fn(base, (0.99, 0.999), weights=weights, **kwargs)
        for q in (0.99, 0.999):
            # weighted midpoint-CDF vs np.quantile linear interpolation
            # differ by O(1/N_tail) at the extreme quantile — the identity
            # holds to estimator convention, not to machine precision
            assert b[q] == pytest.approx(a[q], rel=1e-2), (fn.__name__, q)


def test_empty_and_zero_emittance():
    assert compute_fractional_emittance(np.zeros((0, 6)), (0.99,), "x") == {0.99: 0.0}
    cold = np.zeros((100, 6))
    assert compute_fractional_emittance(cold, (0.99,), "x")[0.99] == 0.0


def test_frac_key():
    assert frac_key(0.99) == "q99"
    assert frac_key(0.999) == "q999"
    assert frac_key(0.9) == "q9"


def test_recorder_opt_in():
    """configure_tail records aligned lists; unconfigured recorder is
    untouched (zero-overhead default)."""
    from linac_gen.diagnostics.recorder import DiagnosticRecorder

    class _Ref:
        w_kin = 2.1; phi_s = 0.0; beta = 0.067; gamma = 1.002
        bg = 0.0672; frequency = 162.5
        wavelength = 299792.458 / 162.5   # mm
        class species:  # noqa: N801
            mass = 939.294

    class _Beam:
        particles = _gaussian_beam(n=5_000, seed=1)
        lost = np.zeros(5_000, dtype=bool)
        n_particles = 5_000
        n_alive = 5_000
        alive_particles = particles
        ref = _Ref()
        continuous = False

    rec = DiagnosticRecorder()
    rec.record(_Beam(), 0.0)
    assert not hasattr(rec, "tail") or not rec.tail   # default: nothing

    rec2 = DiagnosticRecorder()
    rec2.configure_tail((0.99, 0.999))
    rec2.record(_Beam(), 0.0)
    rec2.record(_Beam(), 1.0)
    assert len(rec2.tail["emit_x_q99"]) == 2
    assert len(rec2.tail["r_q999"]) == 2
    assert rec2.tail["emit_x_q999"][0] > rec2.tail["emit_x_q99"][0] > 0
