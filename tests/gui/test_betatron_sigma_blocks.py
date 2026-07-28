"""Unit tests for the Schur-complement dispersion correction helper
in the Results tab.

The helper builds the betatron (dispersion-corrected) 4×4 Σ
sub-block entries from the recorded 6×6 σ matrices:

    Σ_β,ii = Σ_ii − Σ_i5² / Σ_55

These tests check:

* The math: a synthetic Σ with known D_x, σ_δ contribution gives
  σ_β² = σ² − D²·σ_δ² to numerical precision.
* The fallback: when Σ[5,5] ≤ ε (DC beam / zero energy spread),
  Σ_β reduces to Σ (no division by zero).
* Off-diagonal cross-terms (σ_xxp,β) are computed correctly.
* The result is consistent with the projected betatron emittance
  formula ε_β = √(σxx,β · σxpxp,β − σxxp,β²).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

# results_tab is a GUI module — make sure the gui/ path is importable
_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO / "gui"))


def _import_helper():
    """Import the helper inside a function to keep the file collectable
    even on systems where the QApplication can't initialise (Qt is
    imported by results_tab at module load)."""
    from PyQt6.QtWidgets import QApplication
    if QApplication.instance() is None:
        QApplication([])
    from linac_gen_gui.interphase.tabs.results_tab import _betatron_sigma_blocks
    return _betatron_sigma_blocks


def test_dispersion_cancellation_known_values():
    """Build Σ so that Σ[0,5]² / Σ[5,5] equals the raw Σ[0,0].
    Schur complement must give exactly 0 (within float epsilon)."""
    _betatron_sigma_blocks = _import_helper()
    S = np.zeros((1, 6, 6))
    # Raw x-block
    S[0, 0, 0] = 4.0   # σ_x² = 4 mm²
    S[0, 1, 1] = 1.0
    S[0, 2, 2] = 4.0
    S[0, 3, 3] = 1.0
    # Dispersion: σ[0,5] = 2 mm·MeV, σ[5,5] = 1 MeV² → contribution = 4 mm²
    S[0, 0, 5] = 2.0
    S[0, 5, 0] = 2.0
    S[0, 5, 5] = 1.0
    sxx, sxxp, sxpxp, syy, syyp, sypyp = _betatron_sigma_blocks(S)
    # σ_x,β² = 4 - 4 = 0
    assert sxx[0] == pytest.approx(0.0, abs=1e-12)
    # y untouched: σ[2,5] = 0
    assert syy[0] == pytest.approx(4.0, abs=1e-12)


def test_dispersion_partial_subtraction():
    """Less dramatic case: Σ[0,5]² / Σ[5,5] = 1 mm², so σ_β² = 3 mm²."""
    _betatron_sigma_blocks = _import_helper()
    S = np.zeros((1, 6, 6))
    S[0, 0, 0] = 4.0
    S[0, 5, 5] = 1.0
    S[0, 0, 5] = 1.0
    S[0, 5, 0] = 1.0
    sxx, *_ = _betatron_sigma_blocks(S)
    assert sxx[0] == pytest.approx(3.0, abs=1e-12)


def test_dc_beam_fallback_no_zero_division():
    """When Σ[5,5] = 0 the dispersive contribution is undefined.
    Helper must return the raw diagonal (no NaN, no zero division)."""
    _betatron_sigma_blocks = _import_helper()
    S = np.zeros((1, 6, 6))
    S[0, 0, 0] = 4.0
    S[0, 5, 5] = 0.0
    S[0, 0, 5] = 0.7   # would explode if divided by 0
    S[0, 5, 0] = 0.7
    sxx, *_ = _betatron_sigma_blocks(S)
    assert np.isfinite(sxx[0])
    assert sxx[0] == pytest.approx(4.0, abs=1e-12)


def test_off_diagonal_cross_term():
    """σ_xxp,β = σ_xxp − σ_x5·σ_xp5 / σ_55."""
    _betatron_sigma_blocks = _import_helper()
    S = np.zeros((1, 6, 6))
    S[0, 0, 0] = 4.0; S[0, 1, 1] = 1.0
    S[0, 0, 1] = 0.5; S[0, 1, 0] = 0.5
    S[0, 5, 5] = 2.0
    S[0, 0, 5] = 1.0; S[0, 5, 0] = 1.0
    S[0, 1, 5] = 0.5; S[0, 5, 1] = 0.5
    sxx, sxxp, sxpxp, *_ = _betatron_sigma_blocks(S)
    # σ_xx,β  = 4 - 1²/2 = 3.5
    # σ_xpxp,β= 1 - 0.5²/2 = 0.875
    # σ_xxp,β = 0.5 - 1·0.5/2 = 0.25
    assert sxx[0] == pytest.approx(3.5, abs=1e-12)
    assert sxpxp[0] == pytest.approx(0.875, abs=1e-12)
    assert sxxp[0] == pytest.approx(0.25, abs=1e-12)


def test_betatron_emittance_self_consistency():
    """ε_x,β = √(σ_xx,β · σ_xpxp,β − σ_xxp,β²) must be invariant under
    the dispersive shift (the Schur complement is precisely the
    construction that makes this so)."""
    _betatron_sigma_blocks = _import_helper()
    S = np.zeros((2, 6, 6))
    # Snapshot A: zero dispersion
    S[0, 0, 0] = 4.0; S[0, 1, 1] = 1.0
    S[0, 0, 1] = 0.0; S[0, 1, 0] = 0.0
    S[0, 5, 5] = 1.0
    # Snapshot B: same betatron Σ but with Σ[i,5] non-zero — must give
    # the SAME ε_β as snapshot A.
    S[1, 0, 0] = 4.0 + 1.0       # raw inflated by dispersion contribution
    S[1, 1, 1] = 1.0 + 0.25
    S[1, 0, 1] = 0.5             # raw cross term inflated
    S[1, 1, 0] = 0.5
    S[1, 5, 5] = 1.0
    S[1, 0, 5] = 1.0; S[1, 5, 0] = 1.0
    S[1, 1, 5] = 0.5; S[1, 5, 1] = 0.5
    sxx, sxxp, sxpxp, *_ = _betatron_sigma_blocks(S)
    eps_b_A = np.sqrt(sxx[0] * sxpxp[0] - sxxp[0] ** 2)
    eps_b_B = np.sqrt(sxx[1] * sxpxp[1] - sxxp[1] ** 2)
    assert eps_b_A == pytest.approx(eps_b_B, rel=1e-10)


def test_stack_shape_and_diagonal_clamp():
    """Multi-step stack: result arrays match N.  Negative diagonal
    Σ_β (from floating-point subtraction) is clamped to 0 so
    downstream √ doesn't NaN."""
    _betatron_sigma_blocks = _import_helper()
    rng = np.random.default_rng(seed=42)
    N = 5
    S = np.zeros((N, 6, 6))
    for k in range(N):
        # Plant Σ[0,0] slightly less than Σ[0,5]²/Σ[5,5] so the
        # Schur subtraction goes negative — the clamp must save us.
        S[k, 5, 5] = 1.0
        S[k, 0, 5] = 1.0001
        S[k, 5, 0] = 1.0001
        S[k, 0, 0] = 1.0   # raw — note this is LESS than 1.0001²/1
    sxx, *_ = _betatron_sigma_blocks(S)
    assert sxx.shape == (N,)
    # Clamped: all entries non-negative even though the raw subtraction
    # gave slightly negative numbers.
    assert (sxx >= 0.0).all()
