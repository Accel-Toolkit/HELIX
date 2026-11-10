"""Tests for (Δφ, ΔW) <-> (z, δ) basis conversion."""
import math

import numpy as np
import pytest

from linac_gen.core.particle import H_MINUS
from linac_gen.core.reference import ReferenceParticle
from linac_gen.elements.drift import Drift
from linac_gen.tracking.longitudinal_coords import (
    matrix_to_tracewin, sigma_to_tracewin,
)


def _ref():
    return ReferenceParticle(species=H_MINUS, w_kin=2.1226695, frequency=162.5)


def test_drift_matrix_in_tracewin_basis_matches_published_form():
    """After the basis change, the drift should have R_zz = [[1, Δs/γ²], [0, 1]]."""
    ref = _ref()
    L_mm = 50.0
    M = Drift("D", length=L_mm).transfer_matrix(ref)
    M_TW = matrix_to_tracewin(M, ref)

    # TraceWin's R_zz[0, 1] = Δs / γ^2  (with Δs in METERS — TraceWin convention)
    expected = (L_mm / 1000.0) / ref.gamma**2
    assert M_TW[4, 5] == pytest.approx(expected, rel=1e-10)
    assert M_TW[4, 4] == pytest.approx(1.0, abs=1e-12)
    assert M_TW[5, 5] == pytest.approx(1.0, abs=1e-12)
    assert abs(M_TW[5, 4]) < 1e-12


def test_round_trip_matrix():
    """Converting Ours -> TW -> Ours should recover the original matrix."""
    from linac_gen.tracking.longitudinal_coords import _transform
    ref = _ref()
    M = Drift("D", length=75.0).transfer_matrix(ref)
    M_TW = matrix_to_tracewin(M, ref)
    # Apply the inverse transformation manually
    T = _transform(ref.beta, ref.gamma, ref.species.mass, ref.wavelength)
    Tinv = np.diag(1.0 / np.diag(T))
    M_back = T @ M_TW @ Tinv
    np.testing.assert_allclose(M_back, M, atol=1e-14)


def test_sigma_round_trip():
    """Σ round-trip through the basis change returns the original Σ."""
    from linac_gen.tracking.longitudinal_coords import _transform
    ref = _ref()
    # Build a realistic-ish Σ
    rng = np.random.default_rng(0)
    raw = rng.standard_normal((6, 10000))
    Sigma = (raw @ raw.T) / 10000

    Sigma_TW = sigma_to_tracewin(Sigma, ref)
    T = _transform(ref.beta, ref.gamma, ref.species.mass, ref.wavelength)
    Sigma_back = T @ Sigma_TW @ T.T
    np.testing.assert_allclose(Sigma_back, Sigma, atol=1e-12)


def test_transverse_unchanged():
    """The 4x4 transverse block must be untouched by the basis change."""
    ref = _ref()
    M = Drift("D", length=50.0).transfer_matrix(ref)
    M_TW = matrix_to_tracewin(M, ref)
    np.testing.assert_allclose(M_TW[:4, :4], M[:4, :4], atol=1e-14)


def test_longitudinal_determinant_preserved():
    """Both bases are connected by a diagonal linear transform, so the
    determinant of the full 6×6 should match."""
    ref = _ref()
    M = Drift("D", length=100.0).transfer_matrix(ref)
    M_TW = matrix_to_tracewin(M, ref)
    assert np.linalg.det(M_TW) == pytest.approx(np.linalg.det(M), rel=1e-10)


def test_a_map_across_a_frequency_card_in_the_tracewin_basis():
    """z and δ are physical: a card moves neither, so R55 stays 1 and a drift's R56 stays
    L/γ² on each side, with each side converted on its own clock."""
    from linac_gen.core.lattice import Lattice
    from linac_gen.elements.drift import Drift
    from linac_gen.elements.lattice_commands import Freq
    from linac_gen.tracking.matrix_tracking import compute_transfer_matrix, reference_at
    lat = Lattice()
    for el in (Drift("D0", 100.0), Freq("F", 325.0), Drift("D1", 150.0)):
        lat.add(el)
    ref = ReferenceParticle(species=H_MINUS, w_kin=2.1, frequency=162.5)
    M = compute_transfer_matrix(lat, ref)
    tw = matrix_to_tracewin(M, reference_at(lat, ref, 0), reference_at(lat, ref, 3))
    assert tw[4, 4] == pytest.approx(1.0, rel=1e-13)
    assert tw[4, 5] == pytest.approx(0.25 / ref.gamma ** 2, rel=1e-12)
    assert tw[5, 5] == pytest.approx(1.0, rel=1e-13)
    # one reference: the same as before, with both sides on that clock
    window = compute_transfer_matrix(lat, ref, start=2)
    after = reference_at(lat, ref, 2)
    np.testing.assert_array_equal(matrix_to_tracewin(window, after),
                                  matrix_to_tracewin(window, after, after))
    assert matrix_to_tracewin(window, after)[4, 5] == pytest.approx(0.15 / ref.gamma ** 2, rel=1e-12)
