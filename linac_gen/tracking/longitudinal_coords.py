"""Convert transfer and sigma matrices between our (Δφ, ΔW) basis and
TraceWin's (z, δ) basis.

Our code carries the longitudinal pair as (phase-lag in deg, kinetic-energy
deviation in MeV) which is convenient for RF tracking.  TraceWin's transport
matrices are published with z in **meters** and δ = Δp/p dimensionless
(while its transverse block stays in mm, mrad — a documented asymmetry of
the TraceWin display).  Both are valid; the transformation is a pair of
diagonal linear maps on the (5, 6) block.

Linear transform that converts TraceWin-basis coordinates to our basis::

    Δφ [deg]   = k_phi  · z  [m],         k_phi = −360 / (β_s · λ_m)
    ΔW [MeV]   = k_w    · δ,               k_w   = β_s² · γ_s · m

So if X_ours = T · X_TW  with  T = diag(1, 1, 1, 1, k_phi, k_w), then

    M_TW  = T⁻¹ · M_ours · T
    Σ_TW  = T⁻¹ · Σ_ours · T⁻ᵀ  (= T⁻¹ · Σ · T⁻¹, since T is diagonal)

Both conversions use a *single* reference particle state (β_s, γ_s, m, λ).
For pure transverse / non-accelerating cases this is exact; for accelerating
structures the input and output reference states differ and one should use
each at the respective end -- a refinement we don't need for the GUI viewer.
"""
from __future__ import annotations

import numpy as np


def _transform(beta: float, gamma: float, mass_MeV: float, wavelength_mm: float) -> np.ndarray:
    """Diagonal 6×6 T such that X_ours = T · X_TW (Δφ, ΔW vs z [m], δ).

    TraceWin's transport matrix uses z in **meters** (not mm) even though
    its transverse block is in mm/mrad — so k_phi carries the 1/1000 factor
    (wavelength_mm / 1000 = wavelength_m).
    """
    if beta <= 0.0 or wavelength_mm <= 0.0:
        raise ValueError("beta and wavelength must be positive to convert bases")
    wavelength_m = wavelength_mm / 1000.0
    k_phi = -360.0 / (beta * wavelength_m)            # deg/m
    k_w   = beta * beta * gamma * mass_MeV            # MeV per dimensionless
    T = np.eye(6)
    T[4, 4] = k_phi
    T[5, 5] = k_w
    return T


def matrix_to_tracewin(M: np.ndarray, ref) -> np.ndarray:
    """Return ``M`` expressed in TraceWin's (x, x', y, y', z, δ) basis.

    ``ref`` must expose .beta, .gamma, .wavelength and .species.mass (in MeV).
    """
    T = _transform(ref.beta, ref.gamma, ref.species.mass, ref.wavelength)
    Tinv = np.diag(1.0 / np.diag(T))
    return Tinv @ M @ T


def sigma_to_tracewin(sigma: np.ndarray, ref) -> np.ndarray:
    """Return ``sigma`` expressed in TraceWin's (x, x', y, y', z, δ) basis."""
    T = _transform(ref.beta, ref.gamma, ref.species.mass, ref.wavelength)
    Tinv = np.diag(1.0 / np.diag(T))
    return Tinv @ sigma @ Tinv.T


def matrix_to_tracewin_custom(M: np.ndarray, *, beta: float, gamma: float,
                              mass_MeV: float, wavelength_mm: float) -> np.ndarray:
    """Same as :func:`matrix_to_tracewin` but takes the four numbers directly
    (for when you have β, γ per step but no full ReferenceParticle)."""
    T = _transform(beta, gamma, mass_MeV, wavelength_mm)
    Tinv = np.diag(1.0 / np.diag(T))
    return Tinv @ M @ T


def sigma_to_tracewin_custom(sigma: np.ndarray, *, beta: float, gamma: float,
                             mass_MeV: float, wavelength_mm: float) -> np.ndarray:
    T = _transform(beta, gamma, mass_MeV, wavelength_mm)
    Tinv = np.diag(1.0 / np.diag(T))
    return Tinv @ sigma @ Tinv.T


# ---------------------------------------------------------------------------
# MAD-X canonical basis  (x [m], px, y [m], py, t [m], pt)
# ---------------------------------------------------------------------------
# Pinned against MAD-X 5.09.03 (cpymad TRACK, tests/io/test_madx_conventions):
#   * t  = −c·Δt — position-like, a particle AHEAD of the reference has
#     t > 0 — so with Δφ = 360·f·Δt,  t = −λ_free·Δφ/360  (λ_free = c/f,
#     no β: MAD-X's t is c times the time advance, not the path offset);
#   * pt = ΔE/(p₀c) with p₀ the FIXED sequence reference momentum, so
#     ΔW [MeV] = p₀c [MeV] · pt = β₀γ₀m · pt;
#   * px = P_x/p₀ (paraxial): x' [mrad] = 10³ · (p₀/p_local) · px.  The
#     momentum ratio matters only for a map that changes the reference
#     energy — MAD-X cannot move its reference, so an accelerating map is
#     expressed about the entrance p₀ with the exit angles rescaled by
#     p₀/p_out.  That is the canonical (symplectic) form, which MAD-X
#     TWISS leaves alone (it symplectifies MATRIX elements) and TRACK
#     applies verbatim.
def madx_transform(ref0, *, p_local_over_p0: float = 1.0) -> np.ndarray:
    """Diagonal 6×6 T such that ``X_helix = T · X_madx`` at a point where
    the local reference momentum is ``p_local_over_p0`` times the
    sequence reference ``ref0`` (species, kinetic energy, frequency)."""
    if ref0.beta <= 0.0 or ref0.wavelength <= 0.0:
        raise ValueError("beta and wavelength must be positive to convert bases")
    if p_local_over_p0 <= 0.0:
        raise ValueError("p_local/p0 must be positive")
    lambda_free_m = ref0.wavelength / 1000.0     # ref.wavelength is c/f [mm]
    p0c_MeV = ref0.bg * ref0.species.mass
    T = np.eye(6)
    T[0, 0] = T[2, 2] = 1000.0
    T[1, 1] = T[3, 3] = 1000.0 / p_local_over_p0
    T[4, 4] = -360.0 / lambda_free_m
    T[5, 5] = p0c_MeV
    return T


def _p_ratio(ref_local, ref0) -> float:
    return (ref_local.bg * ref_local.species.mass) / (ref0.bg * ref0.species.mass)


def matrix_to_madx(M: np.ndarray, ref0, ref_in=None, ref_out=None) -> np.ndarray:
    """HELIX-basis map ``M`` → MAD-X canonical map about the sequence
    reference ``ref0``.  ``ref_in`` / ``ref_out`` are the local reference
    states at the element's entrance / exit (default: ``ref0`` — the
    non-accelerating case, where the transform is a similarity)."""
    T_in = madx_transform(ref0, p_local_over_p0=_p_ratio(ref_in or ref0, ref0))
    T_out = madx_transform(ref0, p_local_over_p0=_p_ratio(ref_out or ref0, ref0))
    return np.diag(1.0 / np.diag(T_out)) @ np.asarray(M, dtype=float) @ T_in


def matrix_from_madx(R: np.ndarray, ref0, ref_in=None, ref_out=None) -> np.ndarray:
    """Inverse of :func:`matrix_to_madx` (same reference arguments)."""
    T_in = madx_transform(ref0, p_local_over_p0=_p_ratio(ref_in or ref0, ref0))
    T_out = madx_transform(ref0, p_local_over_p0=_p_ratio(ref_out or ref0, ref0))
    return T_out @ np.asarray(R, dtype=float) @ np.diag(1.0 / np.diag(T_in))


def vector_to_madx(v: np.ndarray, ref0, ref_local=None) -> np.ndarray:
    """HELIX phase-space offset → MAD-X canonical (e.g. MATRIX kick1..6)."""
    T = madx_transform(ref0, p_local_over_p0=_p_ratio(ref_local or ref0, ref0))
    return np.asarray(v, dtype=float) / np.diag(T)


def vector_from_madx(k: np.ndarray, ref0, ref_local=None) -> np.ndarray:
    T = madx_transform(ref0, p_local_over_p0=_p_ratio(ref_local or ref0, ref0))
    return np.asarray(k, dtype=float) * np.diag(T)
