"""Hofmann anisotropic coherent-mode dispersion solver (``l = 2, 3, 4``).

Implements the dispersion relations of I. Hofmann, Phys. Rev. E **57**, 4713
(1998), for non-oscillatory coherent space-charge instability modes
``l = 2, 3, 4`` of a KV beam in a linear focusing channel, in their full
anisotropic forms.  The chart coordinates are the depressed tune ratio
``R = nu_z/nu_x``, the transverse tune depression ``Y = nu_x/nu_0x``, and the
geometric emittance ratio ``eps_z/eps_x`` — exactly the per-cell quantities
HELIX's phase-probe channel model produces (see
:mod:`linac_gen.analysis.hofmann_stability` for the adapter).

This is a verbatim physics port of the corrected solver from the analysis
package of A. Pathak, *"Higher-order anisotropic Hofmann stability charts
with probabilistic margins and a machine-learning surrogate, applied to
PIP-II"* (PRAB manuscript; ``PRAB_Hofmann/analysis/dispersion.py``).  Keep
the numerics byte-diffable against that source.  It carries the manuscript's
three corrections to the printed 1998 equations:

1. **Coordinate map** (Eq. (24) + p. 4716): once the plane carrying the
   larger envelope is identified as Hofmann's "x", BOTH ``h = a/b`` and
   ``alpha = nu_y/nu_x`` follow from that same assignment
   (``ALPHA = R_eff`` pairing), restoring the emittance identity
   ``eps_x/eps_y = h^2/alpha``.
2. **Odd-branch interchange** (p. 4719): interchanging ``nu_x <-> nu_y`` in
   Eqs. (24)/(25) rescales the ``nu_x``-normalised quantities as well:
   ``l3_odd_D(s, a, e, sp2) = l3_even_D(s/a^2, 1/a, 1/e, sp2/a^2)``.
3. **Eq. (41) sign**: the ``sp^4`` block prefactor is ``+sp^4`` (required by
   the isotropic-limit reduction to Eq. (42)).

Public API
----------

::

    from linac_gen.analysis.hofmann_dispersion import DispersionSolver

    s = DispersionSolver()
    g = s.growth_rate(R=0.6, Y=0.46, eps_ratio=1.475, l_modes=(2, 3, 4))
    raw = s.solve_branch("l4_even", R=1.50, Y=0.25, eps_ratio=1.475)

Branches available: ``l2`` (closed-form quadratic), ``l3_even``, ``l3_odd``,
``l4_even``, ``l4_odd``.

Provenance: for ``l=2,3,4_even`` both the ``S^2`` block AND the higher-order
``S^4`` / ``S^6`` blocks are the full anisotropic forms of Hofmann's Eqs. (36)
and (41) (the ``l=3_even`` ``S^4`` block carries the (1 -+ 2 eta^2/alpha)
factors required for its iso-limit reduction). Only the ``l=4_odd`` ``S^4`` /
``S^6`` block is retained in the isotropic-limit closed form Eq. (46), and
that branch is disabled by default in the aggregator. The isotropic-limit
reduction at ``alpha=eta=1`` to Eqs. (37)/(42) and the off-isotropic map
identities are pinned by ``tests/analysis/test_hofmann_dispersion.py``,
including bitwise golden values cross-checked against the source package.

The perturbative-validity gate ``S^2 <= 10`` and the per-cell driver live in
:mod:`linac_gen.analysis.hofmann_stability`; the Monte-Carlo probabilistic
layer lives in :mod:`linac_gen.analysis.hofmann_probabilistic`.
"""

from __future__ import annotations

from typing import Callable, Iterable, Tuple

import numpy as np


class ChartAborted(RuntimeError):
    """Raised by :meth:`DispersionSolver.chart` when ``should_stop`` fires."""


# =============================================================================
# Parametrization: (R, Y, eps_ratio) -> (alpha, eta, S^2, R_eff, gscale)
# =============================================================================

def map_to_pre_grid(
    R: float | np.ndarray,
    Y: float | np.ndarray,
    eps_ratio: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Map chart coordinates to Hofmann internal variables.

    Derived directly from Hofmann (1998) Eqs. (5), (24), (25). Writing his
    envelope ratio as ``h = a/b >= 1`` to avoid collision with the chart's
    tune depression, Eq. (24) reads::

        sp2 = wp^2 / nu_x^2 ,   alpha = nu_y / nu_x ,   h = a/b (>= 1)

    and the paper states the companion identity ``eps_x/eps_y = h^2/alpha``
    (p. 4716). Those three are not independent: once the plane carrying the
    LARGER envelope is identified as Hofmann's "x", both ``h`` and ``alpha``
    follow from that same assignment. Pairing an ``h`` taken from one
    assignment with an ``alpha`` taken from the other violates the identity by
    a factor ``R_eff^2`` — the defect fixed here.

    For a matched KV beam ``eps = (envelope)^2 * nu``, so the transverse /
    longitudinal envelope ratio is ``(a_x/a_z)^2 = R / (eps_z/eps_x)``.

    * ``R >= eps_ratio`` (no flip): Hofmann's x-plane is the transverse one,
      ``h = sqrt(R/eps)``, ``alpha = R``, and Eq. (5) gives
      ``sp2 = (1+h)(1-Y^2)/Y^2``. A root ``s`` is normalised to ``nu_x``, so
      ``Im(omega)/nu_0x = sqrt(-s) * Y``.
    * ``R < eps_ratio`` (flip): Hofmann's x-plane is the longitudinal one,
      ``h = sqrt(eps/R)``, ``alpha = 1/R``. Eq. (5) applied to the *other*
      plane gives ``sp2 = (1+h)(1-Y^2)/(h R^2 Y^2)``, and now ``s`` is
      normalised to ``nu_z``, so ``Im(omega)/nu_0x = sqrt(-s) * R * Y``.

    Both branches reduce to the isotropic values ``alpha = h = 1``,
    ``sp2 = 2(1-Y^2)/Y^2``, ``gscale = Y`` at ``R = eps_ratio = 1``, which is
    why the isotropic-limit regression tests cannot distinguish this map from
    the previous one.

    Parameters
    ----------
    R : tune ratio nu_z/nu_x, from the DEPRESSED tunes (Hofmann Eq. (24)
        defines alpha from the depressed frequencies; see
        :func:`linac_gen.analysis.hofmann_stability.hofmann_coordinates`).
    Y : transverse tune-depression eta = nu_x / nu_{0x}
    eps_ratio : eps_z / eps_x  (geometric)

    Returns
    -------
    alpha, h, sp2, R_eff, gscale  — NumPy arrays of the broadcast shape.
    """
    R = np.clip(R, 1e-9, None)
    u = np.sqrt(np.maximum(R / eps_ratio, 1e-30))     # a_x / a_z
    flip = u < 1.0
    R_eff = np.where(flip, 1.0 / R, R)
    ETA = np.where(flip, 1.0 / u, u)                  # Hofmann h >= 1
    ALPHA = R_eff                                     # pairs with ETA (Eq. 24)

    Y2 = Y * Y
    sp2_noflip = (1.0 + ETA) * (1.0 - Y2) / (Y2 + 1e-30)
    sp2_flip = (1.0 + ETA) * (1.0 - Y2) / (ETA * R * R * Y2 + 1e-30)
    SP2 = np.maximum(np.where(flip, sp2_flip, sp2_noflip), 0.0)

    # gscale recovers Im(omega)/nu_{0x} from the raw root sqrt(-s):
    #   no flip: s is normalised to nu_x -> *Y
    #   flip:    s is normalised to nu_z -> *R*Y
    GSCALE = np.where(flip, R * Y, Y)
    return ALPHA, ETA, SP2, R_eff, GSCALE


# =============================================================================
# l=2 closed-form quadratic (reproduced from hoffmann.py)
# =============================================================================

def _l2_even_coeffs(ALPHA: np.ndarray, ETA: np.ndarray, SP2: np.ndarray):
    """Hofmann Eq. (28) times (4-s)(4 alpha^2 - s)."""
    c2 = (1.0 + ETA) ** 2
    c1 = -(1.0 + ETA) ** 2 * (4.0 + 4.0 * ALPHA ** 2) - SP2 * (
        1.0 + 4.0 * ETA + ETA ** 2
    )
    c0 = (
        (1.0 + ETA) ** 2 * (16.0 * ALPHA ** 2)
        + SP2 * (4.0 * ALPHA ** 2 * (1.0 + 2.0 * ETA) + 4.0 * (2.0 * ETA + ETA ** 2))
        + (SP2 ** 2) * (2.0 * ETA)
    )
    return c2, c1, c0


def _l2_odd_coeffs(ALPHA: np.ndarray, ETA: np.ndarray, SP2: np.ndarray):
    """Hofmann Eq. (32) times ((1-alpha)^2 - s)((1+alpha)^2 - s)."""
    B = (1.0 - ALPHA) ** 2
    C = (1.0 + ALPHA) ** 2
    t1 = (1.0 - ALPHA) * (1.0 - (ETA ** 2) / ALPHA)
    t2 = (1.0 + ALPHA) * (1.0 + (ETA ** 2) / ALPHA)
    c2 = (1.0 + ETA) ** 2
    c1 = -2.0 * (1.0 + ETA) ** 2 * (1.0 + ALPHA ** 2) - 0.5 * SP2 * (t1 + t2)
    c0 = (1.0 + ETA) ** 2 * (1.0 - 2.0 * ALPHA ** 2 + ALPHA ** 4) + 0.5 * SP2 * (
        t1 * C + t2 * B
    )
    return c2, c1, c0


def _quad_roots(c2, c1, c0):
    D = np.maximum(c1 * c1 - 4.0 * c2 * c0, 0.0)
    sqrtD = np.sqrt(D)
    denom = 2.0 * np.where(np.abs(c2) < 1e-30, np.sign(c2) * 1e-30, c2)
    return (-c1 + sqrtD) / denom, (-c1 - sqrtD) / denom


def _l2_growth_grid(R, Y, eps_ratio):
    """Closed-form maximum l=2 growth rate (envelope + quadrupole)."""
    ALPHA, ETA, SP2, _, GSCALE = map_to_pre_grid(R, Y, eps_ratio)
    c2e, c1e, c0e = _l2_even_coeffs(ALPHA, ETA, SP2)
    s1e, s2e = _quad_roots(c2e, c1e, c0e)
    ge = np.maximum(
        np.where(s1e < 0.0, np.sqrt(np.maximum(-s1e, 0.0)) * GSCALE, 0.0),
        np.where(s2e < 0.0, np.sqrt(np.maximum(-s2e, 0.0)) * GSCALE, 0.0),
    )
    c2o, c1o, c0o = _l2_odd_coeffs(ALPHA, ETA, SP2)
    s1o, s2o = _quad_roots(c2o, c1o, c0o)
    go = np.maximum(
        np.where(s1o < 0.0, np.sqrt(np.maximum(-s1o, 0.0)) * GSCALE, 0.0),
        np.where(s2o < 0.0, np.sqrt(np.maximum(-s2o, 0.0)) * GSCALE, 0.0),
    )
    return np.maximum(ge, go)


# =============================================================================
# Higher-order dispersion functions D_l(s; alpha, eta, sp2)
# =============================================================================

def l3_even_D(s, alpha, eta, sp2):
    """D_{3,e}(s) — Hofmann (1998) Eq. (36), full anisotropic.

    Leading power (1+eta)^3, poles at 1, 9, (1-2 alpha)^2, (1+2 alpha)^2.
    Both the sp^2 single-pole block and the sp^4 product-pole block are
    written in full anisotropic form. The sp^4 anisotropic terms require
    the factors (1 -+ 2 eta^2 / alpha) on the alpha-dependent residues,
    which are absent from the printed Eq. (36) but are mandated by the
    exact isotropic-limit reduction to Eq. (37): at alpha=eta=1 the
    block must contribute residue 27/8 at the (9-s)^-2 pole, achievable
    only with these factors. Verified analytically:
    -(9-s)^2 + 6(1-s)(9-s) + 27(1-s)^2 = -32 s (3-s)  ✓ matches Eq. 37.

    Accepts scalars or numpy arrays (broadcasting-friendly).
    """
    a = alpha
    e = eta
    sp4 = sp2 * sp2
    a_safe = np.maximum(np.abs(a), 1e-30)

    g1 = 1.0 - s
    g2 = 9.0 - s
    g3 = (1.0 - 2.0 * a) ** 2 - s
    g4 = (1.0 + 2.0 * a) ** 2 - s

    D = (1.0 + e) ** 3

    N1 = (1.0 - 5.0 * e)
    N2 = (9.0 + 27.0 * e + 24.0 * e * e)
    N3 = (1.0 - 2.0 * a) * (1.0 - 2.0 * e * e / a_safe) * (3.0 + e)
    N4 = (1.0 + 2.0 * a) * (1.0 + 2.0 * e * e / a_safe) * (3.0 + e)
    D = D + (sp2 / 8.0) * (N1 / g1 + N2 / g2 + N3 / g3 + N4 / g4)

    # sp^4 anisotropic block — Hofmann Eq. (36), corrected for the
    # missing (1 -+ 2 eta^2 / alpha) factors on the alpha-dependent terms.
    M1 = -1.0 / (g1 * g1)
    M2 = 3.0 / (g1 * g2)
    M3 = 3.0 * (1.0 - 2.0 * a) * (1.0 - 2.0 * e * e / a_safe) / (g2 * g3)
    M4 = 3.0 * (1.0 + 2.0 * a) * (1.0 + 2.0 * e * e / a_safe) / (g2 * g4)
    D = D + (sp4 / 8.0) * (M1 + M2 + M3 + M4)
    return D


def l3_odd_D(s, alpha, eta, sp2):
    """D_{3,o}(s) — Eq. (36) under Hofmann's odd-mode interchange (p. 4719).

    Hofmann, p. 4719: "The odd mode dispersion relation is obtained by
    interchanging nu_x and nu_y as well as a and b **in Eqs. (24) and (25)**,
    and solving Eq. (36) with the new variables."

    Eqs. (24)/(25) define ``sp2 = wp^2/nu_x^2`` and ``s = (omega/nu_x)^2``, i.e.
    both are normalised to ``nu_x``. Interchanging ``nu_x <-> nu_y`` therefore
    rescales them as well as inverting alpha and h::

        alpha -> 1/alpha ,  h -> 1/h ,  sp2 -> sp2/alpha^2 ,  s -> s/alpha^2

    The plasma frequency ``wp^2`` itself is of course unchanged by a
    relabelling; it is the *normaliser* ``nu_x^2`` that changes, so the
    dimensionless ``sp2`` does move. (An earlier version held ``sp2`` and ``s``
    fixed on the grounds that the beam intensity is invariant — correct for
    ``wp^2``, but not for ``sp2``.)

    Because the returned expression is still written as a function of the
    original ``s``, the root scan needs no additional rescaling.
    """
    a_safe = np.maximum(np.abs(alpha), 1e-30)
    a2 = a_safe * a_safe
    return l3_even_D(s / a2, 1.0 / a_safe,
                     1.0 / np.maximum(eta, 1e-30), sp2 / a2)


def l4_even_D(s, alpha, eta, sp2):
    """D_{4,e}(s) — Hofmann (1998) Eq. (41), full anisotropic.

    Leading (1+eta)^4; all of sp^2, sp^4, sp^6 blocks transcribed in their
    full anisotropic form from Hofmann's Eq. (41). The prefactor sign on
    the sp^4 block is taken as +sp^4 (the printed equation appears to use
    -sp^4, but the iso-limit reduction to Eq. (42) only succeeds with +;
    we attribute the sign in the printed paper to a typesetting/typo
    issue and adopt +sp^4 in accord with Eq. (42) and physical
    consistency).
    """
    a = alpha
    e = eta
    e2 = e * e
    e3 = e2 * e
    e4 = e2 * e2
    a2 = a * a
    sp4 = sp2 * sp2
    sp6 = sp4 * sp2
    a_safe = np.maximum(np.abs(a), 1e-30)

    g_16 = 16.0 - s
    g_16a2 = 16.0 * a2 - s
    g_4p = 4.0 * (1.0 + a) ** 2 - s
    g_4m = 4.0 * (1.0 - a) ** 2 - s
    g_4 = 4.0 - s
    g_4a2 = 4.0 * a2 - s

    D = (1.0 + e) ** 4

    # sp^2 block — anisotropic, 6 terms (Hofmann Eq. 41).
    T1 = (5.0 / 4 + 5.0 * e + 29.0 * e2 / 4 + 4.0 * e3) / g_16
    T2 = 3.0 * (1.0 - a) * (1.0 - e2 / a_safe) * (1.0 + 4.0 * e + e2) / 8.0 / g_4m
    T3 = 3.0 * (1.0 + a) * (1.0 + e2 / a_safe) * (1.0 + 4.0 * e + e2) / 8.0 / g_4p
    T4 = (4.0 * e + 29.0 * e2 / 4 + 5.0 * e3 + 5.0 * e4 / 4) / g_16a2
    T5 = -2.0 * e2 / g_4
    T6 = -2.0 * e2 / g_4a2
    D = D + sp2 * (T1 + T2 + T3 + T4 + T5 + T6)

    # sp^4 block — full anisotropic, 12 terms (Hofmann Eq. 41).
    M1  = (0.25 + e) / (g_4 * g_4)
    M2  = -(0.5 + 2.0 * e) / (g_16 * g_4)
    M3  = -(5.0 * e + 9.0 * e2 + 5.0 * e3) / (g_16 * g_16a2)
    M4  = (5.0 * e2 / 2.0) / (g_4 * g_16a2)
    M5  = (5.0 * e2 / 2.0) / (g_16 * g_4a2)
    M6  = -(e2 / 2.0) / (g_4 * g_4a2)
    M7  = (e3 + e4 / 4.0) / (g_4a2 * g_4a2)
    M8  = -(2.0 * e3 + e4 / 2.0) / (g_16a2 * g_4a2)
    M9  = -3.0 * (1.0 - a) * (1.0 - e2 / a_safe) * (1.0 + 4.0 * e) / 8.0 / (g_16 * g_4m)
    M10 = -3.0 * (1.0 + a) * (1.0 + e2 / a_safe) * (1.0 + 4.0 * e) / 8.0 / (g_16 * g_4p)
    M11 = -3.0 * (1.0 - a) * (1.0 - e2 / a_safe) * (4.0 * e + e2) / 8.0 / (g_16a2 * g_4m)
    M12 = -3.0 * (1.0 + a) * (1.0 + e2 / a_safe) * (4.0 * e + e2) / 8.0 / (g_16a2 * g_4p)
    D = D + sp4 * (M1 + M2 + M3 + M4 + M5 + M6 + M7 + M8 + M9 + M10 + M11 + M12)

    # sp^6 block — full anisotropic, 6 terms (Hofmann Eq. 41).
    K1 = -e / (g_4 * g_4 * g_16a2)
    K2 = 2.0 * e / (g_16 * g_4 * g_16a2)
    K3 = 3.0 * (1.0 - a) * (1.0 - e2 / a_safe) * (e / 2.0) / (g_16 * g_4m * g_16a2)
    K4 = 3.0 * (1.0 + a) * (1.0 + e2 / a_safe) * (e / 2.0) / (g_16 * g_4p * g_16a2)
    K5 = -e3 / (g_16 * g_4a2 * g_4a2)
    K6 = 2.0 * e3 / (g_16 * g_16a2 * g_4a2)
    D = D + sp6 * (K1 + K2 + K3 + K4 + K5 + K6)
    return D


def l4_odd_D(s, alpha, eta, sp2):
    """D_{4,o}(s) — Hofmann (1998) Eq. (45).

    Anisotropic ``sp2`` exact; ``sp4`` in isotropic limit (Eq. 46).
    """
    a = alpha
    e = eta
    e2 = e * e
    sp4 = sp2 * sp2
    a_safe = np.maximum(np.abs(a), 1e-30)

    g_1m = (1.0 - a) ** 2 - s
    g_1p = (1.0 + a) ** 2 - s
    g_1m3 = (1.0 - 3.0 * a) ** 2 - s
    g_1p3 = (1.0 + 3.0 * a) ** 2 - s
    g_3m = (3.0 - a) ** 2 - s
    g_3p = (3.0 + a) ** 2 - s

    D = (1.0 + e) ** 4

    T1_num = 2.0 * (1.0 - a) * (1.0 - e2 / a_safe) * (1.0 - 4.0 * e + e2)
    T2_num = (1.0 + a) * (1.0 + e2 / a_safe) * (1.0 - 4.0 * e + e2) * 2.0
    T3_num = (1.0 - 3.0 * a) * (1.0 - 3.0 * e2 / a_safe) * (5.0 + 4.0 * e + e2)
    T4_num = (1.0 + 3.0 * a) * (1.0 + 3.0 * e2 / a_safe) * (5.0 + 4.0 * e + e2)
    T5_num = (3.0 - a) * (3.0 - e2 / a_safe) * (1.0 + 4.0 * e + 5.0 * e2)
    T6_num = (3.0 + a) * (3.0 + e2 / a_safe) * (1.0 + 4.0 * e + 5.0 * e2)

    D = D + (sp2 / 16.0) * (
        T1_num / g_1m + T2_num / g_1p
        + T3_num / g_1m3 + T4_num / g_1p3
        + T5_num / g_3m + T6_num / g_3p
    )

    # sp^4 isotropic (Eq. 46)
    g16 = 16.0 - s
    g4 = 4.0 - s
    D = D + sp4 * (4.0 / (g16 * g16) + 4.0 / (g16 * g4))
    return D


_BRANCH_TABLE: dict[str, Callable] = {
    "l3_even": l3_even_D,
    "l3_odd": l3_odd_D,
    "l4_even": l4_even_D,
    "l4_odd": l4_odd_D,
}


# =============================================================================
# Root finder
# =============================================================================

def _find_max_growth(
    D_func: Callable[[float], float],
    s_min: float = -200.0,
    n_scan: int = 2000,
    n_bisect: int = 50,
) -> float:
    """Scan ``D(s)`` on ``[s_min, 0)`` and bisect every sign change.

    Returns sqrt(-s_root) for the most negative real root, or 0 if D has no
    sign change in the search domain (stable mode).
    """
    s_grid = np.linspace(s_min, -1e-10, n_scan)
    D_vals = np.array([D_func(s) for s in s_grid])

    signs = np.sign(D_vals)
    sign_changes = np.where(
        np.isfinite(D_vals[:-1]) & np.isfinite(D_vals[1:])
        & (signs[:-1] * signs[1:] < 0)
    )[0]
    if len(sign_changes) == 0:
        return 0.0

    max_growth = 0.0
    for idx in sign_changes:
        a, b = s_grid[idx], s_grid[idx + 1]
        fa = D_vals[idx]
        for _ in range(n_bisect):
            mid = 0.5 * (a + b)
            fm = D_func(mid)
            if not np.isfinite(fm):
                break
            if fa * fm < 0:
                b = mid
            else:
                a = mid
                fa = fm
        s_root = 0.5 * (a + b)
        if s_root < -1e-12:
            max_growth = max(max_growth, np.sqrt(-s_root))
    return max_growth


def _find_max_growth_batch(
    D_func: Callable,
    alpha: np.ndarray,
    eta: np.ndarray,
    sp2: np.ndarray,
    n_scan: int = 2000,
    n_bisect: int = 50,
    max_signs: int = 4,
) -> np.ndarray:
    """Vectorized version of ``_find_max_growth`` over a batch of N points.

    For each i in 0..N-1 returns ``sqrt(-s_root_i)`` for the most-unstable
    root of ``D_func(s; alpha_i, eta_i, sp2_i) = 0`` on
    ``s in [s_min_i, 0)``, where ``s_min_i = -max(200, 20*sp2*(1+eta)^2)``
    — matching the scalar ``_find_max_growth`` exactly when the per-row
    scan grids coincide.  Returns 0 for rows with no sign change.

    ``max_signs`` caps the number of sign changes processed per row; the
    dispersion polynomials of Hofmann (1998) Eqs. (36), (41), (45) admit
    at most 3 real roots on the negative-s axis in the physically relevant
    parameter range, so the default ``max_signs=4`` is conservative.
    """
    alpha = np.asarray(alpha, dtype=float)
    eta = np.asarray(eta, dtype=float)
    sp2 = np.asarray(sp2, dtype=float)
    N = alpha.shape[0]
    if N == 0:
        return np.zeros(0)

    s_min_arr = -np.maximum(200.0, 20.0 * sp2 * (1.0 + eta) ** 2)  # (N,)
    s_max = -1e-10
    t = np.linspace(0.0, 1.0, n_scan)                              # (n_scan,)
    s_grid = s_min_arr[:, None] + (s_max - s_min_arr[:, None]) * t[None, :]   # (N, n_scan)

    Dv = D_func(s_grid, alpha[:, None], eta[:, None], sp2[:, None])           # (N, n_scan)

    finite = np.isfinite(Dv)
    signs = np.sign(Dv)
    sc = finite[:, :-1] & finite[:, 1:] & (signs[:, :-1] * signs[:, 1:] < 0)  # (N, n_scan-1)

    rows = np.arange(N)
    max_growth = np.zeros(N)

    for _ in range(max_signs):
        any_true = sc.any(axis=1)             # (N,)
        idx = sc.argmax(axis=1)               # (N,) — 0 if no True

        a_b = s_grid[rows, idx]
        b_b = s_grid[rows, idx + 1]
        fa = Dv[rows, idx]
        for _ in range(n_bisect):
            mid = 0.5 * (a_b + b_b)
            fm = D_func(mid, alpha, eta, sp2)
            finite_m = np.isfinite(fm)
            tighten_left = (fa * fm < 0) & finite_m
            # tighten_right := finite_m & ~tighten_left
            b_b = np.where(tighten_left, mid, b_b)
            a_b = np.where(finite_m & ~tighten_left, mid, a_b)
            fa = np.where(finite_m & ~tighten_left, fm, fa)
        s_root = 0.5 * (a_b + b_b)
        growth_here = np.where(
            any_true & (s_root < -1e-12),
            np.sqrt(np.maximum(-s_root, 0.0)),
            0.0,
        )
        max_growth = np.maximum(max_growth, growth_here)

        # consume this sign-change so the next iteration finds the next one
        sc[rows, idx] = False

    return max_growth


# =============================================================================
# DispersionSolver — public façade
# =============================================================================

class DispersionSolver:
    """Hofmann l=2,3,4 coherent-mode dispersion solver."""

    # static accessors so the regression suite can probe branches directly
    l3_even_D = staticmethod(l3_even_D)
    l3_odd_D = staticmethod(l3_odd_D)
    l4_even_D = staticmethod(l4_even_D)
    l4_odd_D = staticmethod(l4_odd_D)

    @staticmethod
    def l3_even_dispersion(alpha, eta, sp2):
        return lambda s: l3_even_D(s, alpha, eta, sp2)

    @staticmethod
    def l3_odd_dispersion(alpha, eta, sp2):
        return lambda s: l3_odd_D(s, alpha, eta, sp2)

    @staticmethod
    def l4_even_dispersion(alpha, eta, sp2):
        return lambda s: l4_even_D(s, alpha, eta, sp2)

    @staticmethod
    def l4_odd_dispersion(alpha, eta, sp2):
        return lambda s: l4_odd_D(s, alpha, eta, sp2)

    # ---- single-branch raw root --------------------------------------------

    def solve_branch_raw(
        self,
        branch: str,
        alpha: float,
        eta: float,
        sp2: float,
    ) -> float:
        """Return raw sqrt(-s_root) for one branch — no gscale, for testing."""
        if branch not in _BRANCH_TABLE:
            raise KeyError(f"unknown branch {branch!r}; expected one of {list(_BRANCH_TABLE)}")
        D = _BRANCH_TABLE[branch]
        D_func = lambda s: D(s, alpha, eta, sp2)
        s_min = -max(200.0, 20.0 * sp2 * (1.0 + eta) ** 2)
        return _find_max_growth(D_func, s_min=s_min)

    def solve_mode(self, dispersion_fn, alpha, eta, sp2) -> float:
        """Legacy-compatible alias: pass a dispersion factory + (a,e,sp2)."""
        D_func = dispersion_fn(alpha, eta, sp2)
        s_min = -max(200.0, 20.0 * sp2 * (1.0 + eta) ** 2)
        return _find_max_growth(D_func, s_min=s_min)

    def solve_branch(self, branch: str, R: float, Y: float, eps_ratio: float) -> float:
        """Return scaled Im(omega)/nu_{0x} growth for one branch at (R, Y, eps)."""
        a, e, sp2, _, gs = map_to_pre_grid(
            np.array([R]), np.array([Y]), eps_ratio
        )
        return self.solve_branch_raw(branch, a.item(), e.item(), sp2.item()) * gs.item()

    # ---- aggregate over chosen modes ---------------------------------------

    def growth_rate(
        self,
        R: float,
        Y: float,
        eps_ratio: float,
        l_modes: Iterable[int] = (2, 3, 4),
        include_l4_odd: bool = False,
    ) -> float:
        """Maximum coherent growth rate across the requested l-modes.

        Parameters
        ----------
        l_modes : iterable of {2, 3, 4}
        include_l4_odd : include Hofmann l=4 odd branch (default False —
            retained only in its isotropic-limit closed form, Eq. (46)).
        """
        a, e, sp2, _, gs = map_to_pre_grid(
            np.array([R]), np.array([Y]), eps_ratio
        )
        a_v, e_v, sp2_v, gs_v = a.item(), e.item(), sp2.item(), gs.item()

        if Y >= 1.0 or sp2_v < 1e-15:
            return 0.0

        gmax = 0.0
        if 2 in l_modes:
            gmax = max(gmax, _l2_growth_grid(np.array([[R]]), np.array([[Y]]), eps_ratio).item())
        if 3 in l_modes:
            for branch in ("l3_even", "l3_odd"):
                gmax = max(gmax, self.solve_branch_raw(branch, a_v, e_v, sp2_v) * gs_v)
        if 4 in l_modes:
            gmax = max(gmax, self.solve_branch_raw("l4_even", a_v, e_v, sp2_v) * gs_v)
            if include_l4_odd:
                gmax = max(gmax, self.solve_branch_raw("l4_odd", a_v, e_v, sp2_v) * gs_v)
        return gmax

    # ---- batched API for hot loops (probabilistic chart, ML labelling) -----

    def growth_rate_batch(
        self,
        R: np.ndarray,
        Y: np.ndarray,
        eps_ratio: float | np.ndarray,
        l_modes: Iterable[int] = (2, 3, 4),
        include_l4_odd: bool = False,
    ) -> np.ndarray:
        """Vectorized :meth:`growth_rate` over arrays of (R, Y).

        Returns the maximum coherent growth rate across the requested
        modes for each (R_i, Y_i) — equivalent to calling :meth:`growth_rate`
        in a loop but ~50--100x faster by virtue of numpy broadcasting.

        Parameters
        ----------
        R, Y : array-like, same shape
        eps_ratio : scalar or array broadcastable to ``R``
        l_modes : iterable of {2, 3, 4}
        include_l4_odd : include the l=4 odd branch (default False).
        """
        R = np.asarray(R, dtype=float)
        Y = np.asarray(Y, dtype=float)
        out = np.zeros_like(R)

        a, e, sp2, _, gs = map_to_pre_grid(R, Y, eps_ratio)
        active = (Y < 1.0) & (sp2 >= 1e-15)
        if not active.any():
            return out

        if 2 in l_modes:
            # Mask exactly as the scalar growth_rate() does: at Y >= 1 or
            # sp2 ~ 0 the point is space-charge-free and returns 0. The l=2
            # closed form already yields 0 there (both quadratics have
            # non-negative roots at sp2 = 0), so this is a no-op numerically,
            # but it keeps the batched and scalar paths structurally identical
            # rather than relying on that coincidence.
            out = np.maximum(out, np.where(active, _l2_growth_grid(R, Y, eps_ratio), 0.0))

        if 3 in l_modes or 4 in l_modes:
            a_a = a[active]; e_a = e[active]; sp2_a = sp2[active]; gs_a = gs[active]
            higher = np.zeros_like(a_a)
            if 3 in l_modes:
                for D in (l3_even_D, l3_odd_D):
                    higher = np.maximum(
                        higher,
                        _find_max_growth_batch(D, a_a, e_a, sp2_a) * gs_a,
                    )
            if 4 in l_modes:
                higher = np.maximum(
                    higher,
                    _find_max_growth_batch(l4_even_D, a_a, e_a, sp2_a) * gs_a,
                )
                if include_l4_odd:
                    higher = np.maximum(
                        higher,
                        _find_max_growth_batch(l4_odd_D, a_a, e_a, sp2_a) * gs_a,
                    )
            tmp = out.copy()
            tmp[active] = np.maximum(out[active], higher)
            out = tmp
        return out

    # ---- legacy-compatible alias (used by some test paths) ----------------

    def growth_at_point(self, R, Y, eps_ratio, l_modes=(2, 3, 4), include_l4_odd=False):
        """Alias for :meth:`growth_rate` to mirror the legacy API."""
        return self.growth_rate(R, Y, eps_ratio, l_modes, include_l4_odd)

    # ---- internal scanner exposed for refinement tests ---------------------

    @staticmethod
    def _find_growth_rate_func(D_func, s_min=-200.0, n_scan=2000):
        return _find_max_growth(D_func, s_min=s_min, n_scan=n_scan)

    # ---- chart-building -----------------------------------------------------

    def chart(
        self,
        eps_ratio: float,
        r_max: float = 2.0,
        y_max: float = 1.0,
        steps: int = 200,
        l_modes: Iterable[int] = (2, 3, 4),
        include_l4_odd: bool = False,
        y_min: float = 0.05,
        should_stop: Callable[[], bool] | None = None,
        progress: Callable[[int, int], None] | None = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute the Hofmann growth-rate chart on a (R, Y) grid.

        ``should_stop`` (checked once per 2500-point chunk) raises
        :class:`ChartAborted` when it returns True; ``progress`` receives
        ``(chunks_done, n_chunks)`` after each chunk.  Both default to None,
        leaving the computation identical to the source package.

        Returns
        -------
        r_arr : (nx,)  R-axis (= nu_z/nu_x)
        y_arr : (ny,)  Y-axis (= nu_x/nu_{0x})
        G     : (ny, nx)  Im(omega)/nu_{0x} maximum across selected modes
        """
        nx = max(100, int(steps))
        ny = max(70, int(0.55 * steps * (y_max - y_min) / max(r_max, 1e-6)))
        r_arr = np.linspace(1e-6, r_max, nx)
        y_arr = np.linspace(y_min, y_max, ny)

        RR, YY = np.meshgrid(r_arr, y_arr)
        r_flat = RR.ravel()
        y_flat = YY.ravel()
        g_flat = np.empty_like(r_flat)
        chunk = 2500
        n_chunks = (r_flat.size + chunk - 1) // chunk
        for i, start in enumerate(range(0, r_flat.size, chunk)):
            if should_stop is not None and should_stop():
                raise ChartAborted("Hofmann chart computation aborted")
            stop = min(start + chunk, r_flat.size)
            g_flat[start:stop] = self.growth_rate_batch(
                r_flat[start:stop],
                y_flat[start:stop],
                eps_ratio,
                l_modes=l_modes,
                include_l4_odd=include_l4_odd,
            )
            if progress is not None:
                progress(i + 1, n_chunks)
        G = g_flat.reshape(ny, nx)
        return r_arr, y_arr, G


def solver_fingerprint() -> str:
    """Short hash of the physics that determines a chart, for cache keys.

    Any cached chart/bootstrap product is only valid for the coordinate map and
    dispersion functions that produced it. Keying a cache on grid size and
    sample counts alone lets a stale product survive a physics change silently
    -- which is exactly what happened across the Eq. (24) coordinate-map
    correction, where a cached B=200 bootstrap from the previous map was
    reloaded and reported as a current result.

    The fingerprint hashes the source of the map and of every dispersion
    function, so editing any of them changes the key and orphans stale caches.

    PACKAGE-LOCAL: because the hash covers docstrings, this fingerprint
    intentionally differs from the source manuscript package's even though
    the numerics are bitwise identical — never compare fingerprints across
    packages as a physics-equality test (the golden pins in
    ``tests/analysis/test_hofmann_dispersion.py`` are that test).
    """
    import hashlib
    import inspect

    src = "".join(
        inspect.getsource(fn)
        for fn in (map_to_pre_grid, _l2_even_coeffs, _l2_odd_coeffs,
                   l3_even_D, l3_odd_D, l4_even_D, l4_odd_D)
    )
    return hashlib.sha256(src.encode()).hexdigest()[:10]


__all__ = [
    "ChartAborted",
    "DispersionSolver",
    "solver_fingerprint",
    "map_to_pre_grid",
    "l3_even_D",
    "l3_odd_D",
    "l4_even_D",
    "l4_odd_D",
]
