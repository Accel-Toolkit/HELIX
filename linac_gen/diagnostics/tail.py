"""Tail / halo diagnostics: fractional (quantile) emittances and radial
quantiles.

Fractional emittance follows the Courant–Snyder action convention:

    J_i = gamma_t*u_i^2 + 2*alpha*u_i*u'_i + beta*u'_i^2

with (alpha, beta, gamma_t) the rms Twiss of the SAME distribution, so
<J> = 2*eps_rms identically for any distribution.  We define

    eps_q  =  Quantile_q(J) / 2

so that eps_{q->1} is the full emittance and, for a matched Gaussian
where J/eps_rms ~ chi^2_2 (Exp with mean 2),

    eps_q / eps_rms = -ln(1 - q)          (analytic pin used in tests)

e.g. eps_99 = 4.6052*eps_rms, eps_99.9 = 6.9078*eps_rms.

All functions accept an optional per-particle ``weights`` array (for
future importance-weighted beams); ``weights=None`` is the exact
unweighted fast path.
"""
from __future__ import annotations

import numpy as np

_PLANES = {"x": (0, 1), "y": (2, 3), "z": (4, 5)}


def _weighted_quantile(values: np.ndarray, q, weights=None) -> np.ndarray:
    """Quantile(s) of ``values`` with optional weights.

    Uses the inverse of the weighted empirical CDF with midpoint
    convention (reduces to ``np.quantile(..., method='linear')``-like
    behaviour for uniform weights on large N).
    """
    q = np.atleast_1d(np.asarray(q, dtype=float))
    v = np.asarray(values, dtype=float)
    if v.size == 0:
        return np.zeros_like(q)
    if weights is None:
        return np.quantile(v, q)
    w = np.asarray(weights, dtype=float)
    order = np.argsort(v)
    v, w = v[order], w[order]
    cw = np.cumsum(w)
    # midpoint positions of each sample in the CDF
    pos = (cw - 0.5 * w) / cw[-1]
    return np.interp(q, pos, v)


def cs_actions(particles: np.ndarray, plane: str = "x",
               weights=None) -> np.ndarray:
    """Single-particle Courant–Snyder actions J_i w.r.t. the rms Twiss of
    the given (weighted) distribution.  <J> = 2*eps_rms by construction."""
    i, j = _PLANES[plane]
    u = np.asarray(particles[:, i], dtype=float)
    up = np.asarray(particles[:, j], dtype=float)
    if weights is None:
        um, upm = u.mean(), up.mean()
        u, up = u - um, up - upm
        uu, uup, upup = (u * u).mean(), (u * up).mean(), (up * up).mean()
    else:
        w = np.asarray(weights, dtype=float)
        wsum = w.sum()
        um, upm = (w * u).sum() / wsum, (w * up).sum() / wsum
        u, up = u - um, up - upm
        uu = (w * u * u).sum() / wsum
        uup = (w * u * up).sum() / wsum
        upup = (w * up * up).sum() / wsum
    emit_sq = uu * upup - uup * uup
    emit = np.sqrt(max(emit_sq, 0.0))
    if emit < 1e-30:
        return np.zeros_like(u)
    beta = uu / emit
    alpha = -uup / emit
    gamma_t = upup / emit
    return gamma_t * u * u + 2.0 * alpha * u * up + beta * up * up


def compute_fractional_emittance(particles: np.ndarray,
                                 fractions=(0.99, 0.999),
                                 plane: str = "x",
                                 weights=None) -> dict:
    """{fraction: eps_q} — quantile emittances of one transverse plane."""
    fr = tuple(float(f) for f in fractions)
    if len(particles) == 0:
        return {f: 0.0 for f in fr}
    J = cs_actions(particles, plane=plane, weights=weights)
    qs = _weighted_quantile(J, fr, weights=weights)
    return {f: float(qv) / 2.0 for f, qv in zip(fr, qs)}


def compute_radial_quantiles(particles: np.ndarray,
                             fractions=(0.99, 0.999),
                             weights=None,
                             normalize: bool = True) -> dict:
    """{fraction: r_q} of the transverse radius.

    ``normalize=True`` uses r = sqrt((x/sigma_x)^2 + (y/sigma_y)^2)
    (dimensionless, the halo-relevant coordinate); ``False`` uses the raw
    sqrt(x^2+y^2) in mm.
    """
    fr = tuple(float(f) for f in fractions)
    if len(particles) == 0:
        return {f: 0.0 for f in fr}
    x = np.asarray(particles[:, 0], dtype=float)
    y = np.asarray(particles[:, 2], dtype=float)
    if weights is None:
        xm, ym = x.mean(), y.mean()
    else:
        w = np.asarray(weights, dtype=float)
        xm = (w * x).sum() / w.sum()
        ym = (w * y).sum() / w.sum()
    x, y = x - xm, y - ym
    if normalize:
        if weights is None:
            sx = np.sqrt((x * x).mean())
            sy = np.sqrt((y * y).mean())
        else:
            sx = np.sqrt((w * x * x).sum() / w.sum())
            sy = np.sqrt((w * y * y).sum() / w.sum())
        sx = max(sx, 1e-30)
        sy = max(sy, 1e-30)
        r = np.sqrt((x / sx) ** 2 + (y / sy) ** 2)
    else:
        r = np.sqrt(x * x + y * y)
    qs = _weighted_quantile(r, fr, weights=weights)
    return {f: float(qv) for f, qv in zip(fr, qs)}


def frac_key(fraction: float) -> str:
    """Canonical recorder/metrics key suffix: 0.99 -> 'q99', 0.999 -> 'q999'."""
    s = f"{fraction:.6f}".rstrip("0")
    return "q" + s.split(".")[1]
