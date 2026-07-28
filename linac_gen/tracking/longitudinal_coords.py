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
