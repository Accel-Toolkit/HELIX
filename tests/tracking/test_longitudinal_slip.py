"""Physics regression for the longitudinal phase-slip term M[4, 5].

The 6×6 transfer matrix of a drift / quadrupole / solenoid / dipole carries
a non-zero (4, 5) entry that captures the RF phase slip accrued by a
particle with kinetic-energy deviation ΔW over a length Δs.

This test derives the expected Δφ directly from kinematics
(Δt = Δs/v_particle - Δs/v_sync, Δφ = -360 · f · Δt) and compares to the
matrix element — i.e. it catches any bug in the β / γ power counts in the
analytical formula by going back to first principles, not by re-stating
the formula.

Historical note: an earlier version of the code had β² (instead of β³)
in the denominator, which under-estimated the phase slip by a factor
1/β ≈ 15 at H⁻ 2 MeV.  This test locks in the correct β³ dependence.
"""
import math

import numpy as np
import pytest

from linac_gen.core.constants import C_LIGHT
from linac_gen.core.particle import H_MINUS, PROTON
from linac_gen.core.reference import ReferenceParticle
from linac_gen.elements.drift import Drift
from linac_gen.elements.dipole import Dipole
from linac_gen.elements.quadrupole import Quadrupole
from linac_gen.elements.solenoid import Solenoid


def _direct_dphi_deg(L_m: float, W: float, dW: float, m: float, f_Hz: float) -> float:
    """Phase slip Δφ (deg) from pure kinematics (no approximations).

    Sign convention matches the rest of the code: ``Δφ > 0`` means the
    particle arrives *later* than the synchronous particle -- so a
    positive kinetic-energy deviation (faster → earlier arrival) yields
    a negative Δφ.  That's why the formula is ``+360·f·Δt`` with
    ``Δt = t_particle − t_sync``.
    """
    gamma_s = 1.0 + W / m
    beta_s  = math.sqrt(1.0 - 1.0 / gamma_s**2)
    gamma_p = 1.0 + (W + dW) / m
    beta_p  = math.sqrt(1.0 - 1.0 / gamma_p**2)
    dt = L_m / (beta_p * C_LIGHT) - L_m / (beta_s * C_LIGHT)
    return 360.0 * f_Hz * dt


@pytest.mark.parametrize("species,W,f_MHz,dW", [
    (H_MINUS, 2.1226695, 162.5,  0.001),    # low-energy H-
    (H_MINUS, 2.1226695, 162.5, -0.001),
    (PROTON,  3.0,       352.21, 0.002),    # low-energy proton
    (PROTON,  100.0,     805.0,  0.01),     # medium-energy proton
])
def test_drift_phase_slip_matches_kinematics(species, W, f_MHz, dW):
    ref = ReferenceParticle(species=species, w_kin=W, frequency=f_MHz)
    L_mm = 200.0

    drift = Drift("D", length=L_mm)
    M = drift.transfer_matrix(ref)
    dphi_matrix = M[4, 5] * dW

    dphi_direct = _direct_dphi_deg(
        L_m=L_mm * 1e-3, W=W, dW=dW, m=species.mass, f_Hz=f_MHz * 1e6,
    )
    # Linear approximation OK for small dW -- tolerate 0.1% relative error.
    assert abs(dphi_matrix - dphi_direct) / max(abs(dphi_direct), 1e-30) < 1e-3, (
        f"Δφ mismatch for drift: matrix={dphi_matrix:+.6e} deg, "
        f"direct={dphi_direct:+.6e} deg"
    )


def test_drift_matches_tracewin_rzz():
    """Verify our M[4,5] equals TraceWin's R_zz = [[1, Δs/γ²], [0, 1]]
    after transforming from (z, δ) to (Δφ, ΔW)."""
    ref = ReferenceParticle(species=H_MINUS, w_kin=2.1226695, frequency=162.5)
    L_mm = 50.0

    drift = Drift("D", length=L_mm)
    M = drift.transfer_matrix(ref)

    # TraceWin: R_zz[0,1] = Δs/γ² = dz/dδ
    # Transform: Δφ = -360·z/(β·λ),  ΔW = β²·γ·m·δ
    # => M[4,5] = dΔφ/dΔW = -(360/(β·λ)) · (Δs/γ²) · 1/(β²·γ·m)
    #          = -360·Δs / (β³·γ³·m·λ)
    expected = -360.0 * L_mm / (
        ref.beta**3 * ref.gamma**3 * ref.species.mass * ref.wavelength
    )
    assert abs(M[4, 5] - expected) < 1e-10


def test_quadrupole_longitudinal_slip_matches_drift():
    """A quad has the same longitudinal phase slip as a drift of equal length."""
    ref = ReferenceParticle(species=H_MINUS, w_kin=2.1226695, frequency=162.5)
    L_mm = 100.0

    q = Quadrupole("Q", length=L_mm, gradient=5.0)
    d = Drift("D", length=L_mm)
    M_q = q.transfer_matrix(ref)
    M_d = d.transfer_matrix(ref)
    assert abs(M_q[4, 5] - M_d[4, 5]) < 1e-12


def test_solenoid_longitudinal_slip_matches_drift():
    """A solenoid has the same longitudinal phase slip as a drift."""
    ref = ReferenceParticle(species=H_MINUS, w_kin=2.1226695, frequency=162.5)
    L_mm = 150.0

    s = Solenoid("S", length=L_mm, field=0.2)
    d = Drift("D", length=L_mm)
    assert abs(s.transfer_matrix(ref)[4, 5] - d.transfer_matrix(ref)[4, 5]) < 1e-12


def test_dipole_longitudinal_slip_matches_drift():
    """Dipole's M[4,5] should equal a drift of equal arc length."""
    # Geometry: rho=1000 mm, angle=5.73 deg ⇒ arc length = pi/180 * rho * angle_deg
    ref = ReferenceParticle(species=PROTON, w_kin=3.0, frequency=352.21)
    angle_deg = 5.73
    rho_mm = 1000.0
    arc_mm = math.pi / 180.0 * rho_mm * angle_deg

    bend = Dipole("B", angle=angle_deg, rho=rho_mm)
    drift = Drift("D", length=arc_mm)
    M_bend = bend.transfer_matrix(ref)
    M_drift = drift.transfer_matrix(ref)
    # Pure sector bend (field_index=0) should have the same phase slip as a
    # drift of equal arc length (dispersion is separate, in M[0,5]/M[1,5]).
    assert abs(M_bend[4, 5] - M_drift[4, 5]) < 1e-10
