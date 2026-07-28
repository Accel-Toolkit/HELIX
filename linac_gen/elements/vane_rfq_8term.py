"""Matcher-aware r₀(z) for RFQ cells (M2 milestone for ``VaneRFQ``).

Engaged via ``field_model="8term"``.  In non-accelerating matcher cells
(``A₁₀ = 0, m = 1``), the .dat's constant ``r₀`` of 5.576 mm misses the
entrance/exit aperture flare (up to ~20 mm in the .vane data).  When
``A₁₀ = 0`` and ``m = 1``, Crandall's 2-term coefficient ``X`` reduces
to ``1`` independently of ``r₀``, so substituting the local
``r₀(z) = √(a₁(z)·a₂(z))`` from the .vane is mathematically exact and
gives the physically correct (weaker) focusing in flared matcher
regions.  In accelerating cells (``A₁₀ ≠ 0, m > 1``) the cell-level
coefficients were derived assuming a constant ``r₀_dat``; per-z r₀
would violate that closure, so the cell-constant value is kept.

Caveat — empirical observation
------------------------------
On the PXIE LEBT+RFQ benchmark with correct initial conditions
(σ_x_in = 4.87 mm, see :mod:`diag_vane_rfq_8term`), the matcher-aware
refinement *worsens* σ_x_exit (M1 1.84 mm → M2 3.66 mm vs TW 0.51 mm).
Reason: the flared-aperture matcher physics is correct, but it lets the
beam expand more before the AG-focusing accelerating cells take over.
TraceWin's accelerating-cell focusing is ~3× stronger than LG's because
TraceWin uses the full Crandall X = m·(1 − A·I₀(ka)) coefficient (and
higher-order multipole terms) instead of LG's simple ``(1 − A₁₀)``
short-form.  Naively replacing LG's coefficient with the full Crandall
form pushes per-cell phase advance through an AG-resonance and σ
explodes by 7 orders of magnitude — the 8-term multipole modes
(A_{1,2}, A_{3,0}, etc.) are required to stay below the resonance.

Implication: M2 alone is the right *direction* for matcher physics but
cannot close the σ_x gap.  M3 (numerical 2-D Laplace per z, true
Toutatis equivalent) is needed to fully match TraceWin.

Module name and ``field_model="8term"`` keyword are kept for forward
compatibility with a future true 8-term Crandall multipole expansion.
"""
from __future__ import annotations

import numpy as np
from scipy.special import iv


# Threshold below which a cell is treated as a matcher (no acceleration).
# 1e-6 catches the {A₁₀ = 0, m = 1} matcher cells in the PXIE .dat (cells
# 1, 201, 202, 203 have exact zeros) without false-positives — the
# smallest non-zero A₁₀ in the design is ~1.3e-4.
A10_MATCHER_THRESHOLD = 1e-6


def is_matcher_cell(A10: float) -> bool:
    """True if a cell is a non-accelerating matcher (A₁₀ ≈ 0)."""
    return abs(A10) < A10_MATCHER_THRESHOLD


def effective_r0_mm(z_mid_mm: float, cell_r0_dat_mm: float, A10: float,
                    vane_lookup) -> float:
    """Effective r₀ for the DC quadrupole at a substep midpoint.

    Returns ``vane_lookup(z_mid_mm)`` for matcher cells (A₁₀ ≈ 0)
    and ``cell_r0_dat_mm`` otherwise — see module docstring for why.
    """
    if is_matcher_cell(A10):
        return float(vane_lookup(z_mid_mm))
    return float(cell_r0_dat_mm)


def crandall_X(A10: float, m: float, r0_mm: float, L_mm: float) -> float:
    """Crandall 2-term DC quadrupole coefficient ``X``.

    ``X = m·(1 − A₁₀·I₀(ka))`` with ``a = r₀/√m`` and ``k = π/L``
    (Wangler 8.13 / Crandall LA-11968-MS Eq. 2-7).

    Limits:
    * matcher (A₁₀=0, m=1): X = 1 (pure DC quadrupole)
    * trivial L=0 or r₀=0:  X = 0
    """
    if L_mm <= 0 or r0_mm <= 0 or m <= 0:
        return 0.0
    if is_matcher_cell(A10) and abs(m - 1.0) < 1e-9:
        return 1.0
    a = r0_mm / np.sqrt(m)
    k = np.pi / L_mm
    ka = k * a
    return m * (1.0 - A10 * float(iv(0, ka)))


def A_quad_local(A10: float, m: float, r0_eff_mm: float,
                 r0_dat_mm: float, L_mm: float) -> float:
    """DC quadrupole coefficient ``A_quad = (1−A₁₀)/r₀_eff²`` (1/mm²).

    Uses the *short form* ``X ≈ 1−A₁₀`` (the M1 Wangler/RfqCell choice),
    not the full Crandall ``X = m·(1−A·I₀(ka))``.  See module docstring
    for why: the physically correct Crandall X is ~1.4–1.7× stronger and
    pushes per-cell phase advance through an AG-resonance unless the
    higher-order multipole modes (A_{1,2}, A_{3,0}, …) are included —
    which the 2-term model isn't.

    The matcher-aware refinement is in ``r0_eff_mm`` only: matcher cells
    take the local ``√(a₁·a₂)`` from the .vane data; accelerating cells
    keep ``r0_dat_mm`` so the active path stays bit-equivalent to M1.

    Parameters
    ----------
    A10, m, r0_eff_mm, r0_dat_mm, L_mm
        Cell parameters.  ``m``, ``r0_dat_mm``, ``L_mm`` are accepted to
        keep the call site future-proof for an M3 path that needs the
        full Crandall X — they are not used in the short-form branch.
    """
    if r0_eff_mm <= 0:
        return 0.0
    return max(0.0, 1.0 - A10) / (r0_eff_mm * r0_eff_mm)
