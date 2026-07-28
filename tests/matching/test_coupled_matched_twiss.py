"""Coupled (4×4) matched Twiss for solenoid-focused HWR-style lattices.

The pre-existing find_periodic_twiss decoupled the x and y planes by
extracting Twiss from the 2×2 diagonal blocks of the 4×4 one-turn map.
That refused (correctly) to return a meaningful answer for solenoid
lattices because the off-diagonal x↔y blocks are non-zero -- the 2×2
block trace is not the one-turn phase advance.

This test file covers the coupled fallback path
(find_coupled_matched_twiss + auto-route in find_periodic_twiss):

* Decoupled lattices still take the old path and return coupled=False
  with values matching the previous behaviour.
* Solenoid lattices route to the eigenvector method, return a finite
  matched Σ that satisfies M·Σ·Mᵀ = Σ to numerical precision, and
  surface coupled=True.
"""
from __future__ import annotations

import numpy as np
import pytest

from linac_gen.core.particle import PROTON
from linac_gen.core.reference import ReferenceParticle
from linac_gen.core.lattice import Lattice
from linac_gen.elements.drift import Drift
from linac_gen.elements.quadrupole import Quadrupole
from linac_gen.elements.solenoid import Solenoid
from linac_gen.matching.periodic import (
    find_coupled_matched_twiss, find_periodic_twiss,
    _build_coupled_matched_sigma, _project_twiss_from_sigma,
)
from linac_gen.tracking.matrix_tracking import compute_transfer_matrix


def _ref():
    return ReferenceParticle(species=PROTON, w_kin=3.0, frequency=352.21)


def _fodo_lattice():
    """Decoupled FODO -- old decoupled path should still handle it."""
    lat = Lattice()
    lat.add(Quadrupole("QF", length=100, gradient=+10, aperture=10))
    lat.add(Drift("D1", length=200, aperture=10))
    lat.add(Quadrupole("QD", length=100, gradient=-10, aperture=10))
    lat.add(Drift("D2", length=200, aperture=10))
    return lat


def _solenoid_lattice():
    """Drift + solenoid + drift -- transverse-coupled via solenoid focus."""
    lat = Lattice()
    lat.add(Drift("D1", length=100, aperture=10))
    lat.add(Solenoid("SOL", length=200, field=2.0, aperture=10))
    lat.add(Drift("D2", length=100, aperture=10))
    return lat


def _hwr_style_lattice():
    """Solenoid + cavity-like (modelled here as drift) + solenoid -- the
    coupling signature of an HWR cryomodule period."""
    lat = Lattice()
    lat.add(Solenoid("SOL1", length=200, field=1.5, aperture=10))
    lat.add(Drift("GAP", length=300, aperture=10))
    lat.add(Solenoid("SOL2", length=200, field=-1.5, aperture=10))
    lat.add(Drift("D", length=100, aperture=10))
    return lat


# ----------------------------------------------------------------------
# Decoupled path: must still work and report coupled=False
# ----------------------------------------------------------------------
def test_decoupled_fodo_still_uses_decoupled_path():
    """A FODO cell has no x↔y coupling.  find_periodic_twiss must
    take the old fast path and return coupled=False."""
    lat = _fodo_lattice()
    r = find_periodic_twiss(lat, _ref())
    assert r["coupled"] is False
    # Sanity: αx and αy are non-zero, βx and βy are positive.
    assert r["beta_x"] > 0
    assert r["beta_y"] > 0


# ----------------------------------------------------------------------
# Coupled path: solenoid lattice
# ----------------------------------------------------------------------
def test_solenoid_lattice_routes_to_coupled_path():
    """Single solenoid couples x and y -- old decoupled path raises;
    the auto-router must catch and route to find_coupled_matched_twiss
    transparently."""
    lat = _solenoid_lattice()
    r = find_periodic_twiss(lat, _ref())
    assert r["coupled"] is True
    assert "sigma4" in r
    assert r["sigma4"].shape == (4, 4)
    # The two normal-mode phase advances must exist and be in (0, 360°).
    assert 0 < r["mu_1"] < 360
    assert 0 < r["mu_2"] < 360


def test_solenoid_matched_sigma_satisfies_periodic_condition():
    """The matched Σ must satisfy M·Σ·Mᵀ = Σ (periodic condition).
    Numerical tolerance ~1e-12 -- the eigenvector approach is exact
    up to floating-point precision."""
    lat = _solenoid_lattice()
    ref = _ref()
    r = find_coupled_matched_twiss(lat, ref)
    M = compute_transfer_matrix(lat, ref.copy())[0:4, 0:4]
    Sigma = r["sigma4"]
    Sigma_propagated = M @ Sigma @ M.T
    np.testing.assert_allclose(Sigma_propagated, Sigma, atol=1e-10)


def test_solenoid_matched_sigma_is_symmetric():
    """Σ is symmetric by construction; ensure no numerical drift."""
    lat = _solenoid_lattice()
    r = find_coupled_matched_twiss(lat, _ref())
    Sigma = r["sigma4"]
    np.testing.assert_allclose(Sigma, Sigma.T, atol=1e-12)


def test_solenoid_matched_sigma_is_positive_definite():
    """A physical Σ must have positive eigenvalues (variances are
    non-negative)."""
    lat = _solenoid_lattice()
    r = find_coupled_matched_twiss(lat, _ref())
    Sigma = r["sigma4"]
    eigvals = np.linalg.eigvalsh(Sigma)
    assert (eigvals > 0).all(), f"Σ eigenvalues: {eigvals}"


def test_single_symmetric_solenoid_yields_equal_xy_optics():
    """A single, symmetric solenoid has rotational symmetry -- the
    matched Σ_xx must equal Σ_yy and Σ_xpxp must equal Σ_ypyp."""
    lat = _solenoid_lattice()
    r = find_coupled_matched_twiss(lat, _ref())
    Sigma = r["sigma4"]
    assert Sigma[0, 0] == pytest.approx(Sigma[2, 2], rel=1e-6)
    assert Sigma[1, 1] == pytest.approx(Sigma[3, 3], rel=1e-6)
    # Projected optics also equal
    assert r["beta_x"] == pytest.approx(r["beta_y"], rel=1e-6)


def test_hwr_style_lattice_finds_matched_sigma():
    """End-to-end smoke on an HWR-period-shape lattice: two solenoids
    of opposite sign with a drift between them.  Such an anti-symmetric
    pair *cancels* the net rotation at the period boundary, so the
    auto-router may take the decoupled path (correct physics) -- accept
    either route, but require the result to be finite, stable, and
    consistent with the periodic condition.
    """
    lat = _hwr_style_lattice()
    r = find_periodic_twiss(lat, _ref())
    assert r["beta_x"] > 0
    assert r["beta_y"] > 0
    if r.get("coupled"):
        # Coupled path: verify the matched Σ
        M = compute_transfer_matrix(lat, _ref().copy())[0:4, 0:4]
        Sigma = r["sigma4"]
        np.testing.assert_allclose(M @ Sigma @ M.T, Sigma, atol=1e-9)
    # Else: decoupled path returned a per-plane α/β -- the existing
    # find_periodic_twiss invariants apply and are already tested
    # elsewhere.


# ----------------------------------------------------------------------
# Σ → α/β projection
# ----------------------------------------------------------------------
def test_project_twiss_from_decoupled_sigma_matches_diagonal_blocks():
    """The projection function should agree with the standard 2×2 Twiss
    extraction when the input Σ is already block-diagonal."""
    # Build a block-diagonal Σ from known α, β per plane.
    alpha_x, beta_x = 0.5, 2.0
    alpha_y, beta_y = -0.3, 3.5
    gamma_x = (1 + alpha_x ** 2) / beta_x
    gamma_y = (1 + alpha_y ** 2) / beta_y
    Sigma = np.zeros((4, 4))
    Sigma[0, 0] = beta_x;  Sigma[1, 1] = gamma_x
    Sigma[0, 1] = -alpha_x; Sigma[1, 0] = -alpha_x
    Sigma[2, 2] = beta_y;  Sigma[3, 3] = gamma_y
    Sigma[2, 3] = -alpha_y; Sigma[3, 2] = -alpha_y
    p = _project_twiss_from_sigma(Sigma)
    assert p["alpha_x"] == pytest.approx(alpha_x, abs=1e-9)
    assert p["beta_x"] == pytest.approx(beta_x, abs=1e-9)
    assert p["alpha_y"] == pytest.approx(alpha_y, abs=1e-9)
    assert p["beta_y"] == pytest.approx(beta_y, abs=1e-9)
    # Unit emittance (by construction).
    assert p["emit_x_proj"] == pytest.approx(1.0, abs=1e-9)
    assert p["emit_y_proj"] == pytest.approx(1.0, abs=1e-9)


# ----------------------------------------------------------------------
# Stability guard
# ----------------------------------------------------------------------
def test_unstable_lattice_raises_value_error():
    """A lattice with eigenvalues off the unit circle is dynamically
    unstable.  _build_coupled_matched_sigma must raise rather than
    return garbage.

    Since the det(M4)^(1/4) normalization, UNIFORM scaling (a conformal
    map = ordinary adiabatic damping/growth) is normalized exactly, so
    a genuinely unstable probe must be plane-DEPENDENT: growth in x,
    damping in y, det = 1 so no normalization applies."""
    import math as _m
    th = 0.7
    R = np.array([[_m.cos(th), _m.sin(th)],
                  [-_m.sin(th), _m.cos(th)]])
    M_unstable = np.block([[R * 1.5, np.zeros((2, 2))],
                           [np.zeros((2, 2)), R / 1.5]])
    with pytest.raises(ValueError, match="unstable"):
        _build_coupled_matched_sigma(M_unstable)
    # The old uniform-scaling probe normalizes to the identity — pure
    # resonance — and is still rejected, now via the degeneracy guard.
    with pytest.raises(ValueError, match="degenerate"):
        _build_coupled_matched_sigma(np.eye(4) * 1.5)


# ----------------------------------------------------------------------
# Accelerating-section tolerance: 4-15% off the unit circle is normal
# physics, not an instability.  Must accept + warn.
# ----------------------------------------------------------------------
def test_accelerating_section_within_tolerance_succeeds():
    """An accelerating section produces eigenvalues with modulus < 1
    due to adiabatic damping.  At a few percent deviation this is the
    expected physics of a cryomodule period, not an instability --
    _build_coupled_matched_sigma must accept it within stab_tol."""
    # Build a near-unit-circle matrix: take a known stable map and
    # scale slightly to mimic ~5% damping.
    # Start from a 4×4 rotation matrix (symplectic, eigvals on unit circle)
    # and damp by 0.96 to push moduli inside the unit circle.
    import math as _m
    th_x, th_y = 0.7, 0.9
    Rx = np.array([[_m.cos(th_x), _m.sin(th_x)],
                   [-_m.sin(th_x), _m.cos(th_x)]])
    Ry = np.array([[_m.cos(th_y), _m.sin(th_y)],
                   [-_m.sin(th_y), _m.cos(th_y)]])
    M = np.block([[Rx, np.zeros((2, 2))],
                  [np.zeros((2, 2)), Ry]]) * 0.96
    # Should NOT raise (4% damping is within default stab_tol = 0.15)
    Sigma = _build_coupled_matched_sigma(M)
    assert Sigma.shape == (4, 4)
    # Σ remains symmetric and positive-definite by construction
    np.testing.assert_allclose(Sigma, Sigma.T, atol=1e-10)
    assert (np.linalg.eigvalsh(Sigma) > 0).all()


def test_accelerating_section_beyond_tolerance_rejected():
    """A far-off-unit-circle NON-conformal lattice must raise.

    (Uniform 50% damping — the old probe — is conformal and now
    normalizes exactly; plane-dependent damping is what survives the
    det^(1/4) normalization and must be bounded by stab_tol.)"""
    import math as _m
    th = 0.7
    R = np.array([[_m.cos(th), _m.sin(th)],
                  [-_m.sin(th), _m.cos(th)]])
    # x damped 50%, y undamped → ±~29% residual after normalization.
    M = np.block([[R * 0.5, np.zeros((2, 2))],
                  [np.zeros((2, 2)), R]])
    with pytest.raises(ValueError, match="unstable"):
        _build_coupled_matched_sigma(M)


def test_conformal_damping_is_normalized_exactly_no_warning():
    """Uniform (conformal) adiabatic damping is removed exactly by the
    det(M4)^{1/4} normalization — no smooth-approximation warning, and
    the matched Σ equals the undamped map's Σ."""
    import io as _io
    from contextlib import redirect_stderr
    import math as _m
    th = 0.7
    R = np.array([[_m.cos(th), _m.sin(th)],
                  [-_m.sin(th), _m.cos(th)]])
    M_sym = np.block([[R, np.zeros((2, 2))],
                      [np.zeros((2, 2)), R]])
    M_damped = M_sym * 0.95            # 5% conformal damping
    buf = _io.StringIO()
    with redirect_stderr(buf):
        S_damped = _build_coupled_matched_sigma(M_damped)
        S_sym = _build_coupled_matched_sigma(M_sym)
    assert "smooth approximation" not in buf.getvalue()
    np.testing.assert_allclose(S_damped, S_sym, rtol=1e-9, atol=1e-12)


def test_nonconformal_damping_emits_smooth_approx_warning():
    """PLANE-DEPENDENT damping survives det-normalization and must
    produce the stderr smooth-approximation notice."""
    import io as _io
    from contextlib import redirect_stderr
    import math as _m
    th = 0.7
    R = np.array([[_m.cos(th), _m.sin(th)],
                  [-_m.sin(th), _m.cos(th)]])
    # x-plane damped 10%, y-plane undamped: det^(1/4) removes the
    # geometric mean, leaving a ±5%-ish residual on each mode.
    M = np.block([[R * 0.90, np.zeros((2, 2))],
                  [np.zeros((2, 2)), R]])
    buf = _io.StringIO()
    with redirect_stderr(buf):
        _build_coupled_matched_sigma(M)
    text = buf.getvalue()
    assert "non-conformal" in text
    assert "smooth approximation" in text


# ----------------------------------------------------------------------
# Solenoid / FieldMap-solenoid periodicity in find_fodo_cells
# ----------------------------------------------------------------------
def test_find_fodo_cells_detects_solenoid_periodicity():
    """A lattice of Solenoid + drift + Solenoid + drift + Solenoid
    must be detected as having one periodic cell (sol1 → sol3 span),
    just like 3 Quadrupoles would."""
    from linac_gen.matching.periodic import find_fodo_cells
    lat = Lattice()
    lat.add(Solenoid("SOL1", length=200, field=2.0, aperture=10))
    lat.add(Drift("D1", length=300, aperture=10))
    lat.add(Solenoid("SOL2", length=200, field=2.0, aperture=10))
    lat.add(Drift("D2", length=300, aperture=10))
    lat.add(Solenoid("SOL3", length=200, field=2.0, aperture=10))
    cells = find_fodo_cells(lat)
    assert len(cells) == 1
    # Cell spans from just after SOL1 to SOL3, inclusive
    cs, ce = cells[0]
    assert cs == 1   # element index after SOL1 (idx 0)
    assert ce == 4   # element index of SOL3


def test_find_fodo_cells_still_detects_quadrupole_periodicity():
    """Backward-compat: pure-Quadrupole lattice must still be detected."""
    from linac_gen.matching.periodic import find_fodo_cells
    lat = Lattice()
    lat.add(Quadrupole("Q1", length=100, gradient=10, aperture=10))
    lat.add(Drift("D1", length=200, aperture=10))
    lat.add(Quadrupole("Q2", length=100, gradient=-10, aperture=10))
    lat.add(Drift("D2", length=200, aperture=10))
    lat.add(Quadrupole("Q3", length=100, gradient=10, aperture=10))
    cells = find_fodo_cells(lat)
    assert len(cells) == 1


def test_find_fodo_cells_too_few_focusers_returns_empty():
    """Fewer than 3 focusing elements -> no period detection."""
    from linac_gen.matching.periodic import find_fodo_cells
    lat = Lattice()
    lat.add(Quadrupole("Q1", length=100, gradient=10, aperture=10))
    lat.add(Drift("D1", length=200, aperture=10))
    lat.add(Quadrupole("Q2", length=100, gradient=-10, aperture=10))
    assert find_fodo_cells(lat) == []
