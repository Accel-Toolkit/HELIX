"""Tests for the magnetic-stripping (Lorentz-stripping) loss analyzer.

Reference paper: Folsom, Eshraqi, Blaskovic-Kraljevic, Gålnander,
*"Stripping mechanisms and remediation for H- beams"*, Phys. Rev. Accel.
Beams **24**, 074201 (2021), Eqs. (4) and (5).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List

import numpy as np
import pytest

from linac_gen.analysis.magnetic_stripping import (
    A1_SVPM, A2_VPM, MagStripResult,
    magnetic_stripping_loss, _design_b_for_element,
)


# ---------------------------------------------------------------------------
# Synthetic Results / lattice scaffolding so the analyzer doesn't depend on a
# full tracking run.
# ---------------------------------------------------------------------------
@dataclass
class _FakeResults:
    s: List[float] = field(default_factory=list)
    sigma_x: List[float] = field(default_factory=list)
    sigma_y: List[float] = field(default_factory=list)
    ref_gamma: List[float] = field(default_factory=list)
    ref_beta: List[float] = field(default_factory=list)
    ref_w_kin: List[float] = field(default_factory=list)
    element_names: List[str] = field(default_factory=list)
    mass_mev: float = 938.27208816   # synthetic fixture value (the analyzer
    # takes mass from results.mass_mev, so any value is self-consistent here;
    # the physical H⁻ ion mass is 939.294 = m_p + 2·m_e)
    sigma_matrix: object = None      # optional (N,6,6); enables θ_rms from σ'


@dataclass
class _FakeBeamConfig:
    species: str = "H-"
    current: float = 5.0          # mA
    duty_cycle: float = 100.0     # %
    frequency: float = 162.5      # MHz


@dataclass
class _FakeLattice:
    elements: list = field(default_factory=list)


# ---------------------------------------------------------------------------
# 1. Hand-calculation parity for the Folsom Eq. (5) formula.
# ---------------------------------------------------------------------------
def _dipole_for_B(B_target: float, gamma: float, beta: float,
                  mass_mev: float = 938.272):
    """A Dipole whose bend field B = brho/rho equals ``B_target`` at this γβ
    (a genuinely perpendicular field, unlike a solenoid)."""
    from linac_gen.elements.dipole import Dipole
    brho = gamma * beta * mass_mev * 1.0e6 / (2.99792458e8 * 1.0)
    rho_mm = brho / B_target * 1.0e3
    return Dipole(name="D", angle=30.0, rho=rho_mm)


def test_constant_b_folsom_eq5_gamma_beta_1_b_5T():
    """γβ=1, B⊥=5T → E*=1.5e9 V/m, rate = (5/A1)·exp(-A2/E*) ≈ 8.58e4/m.
    Uses a dipole — a field genuinely perpendicular to v — to drive Eq.(5)."""
    gamma = math.sqrt(2.0)            # γβ = 1 exactly
    beta = 1.0 / gamma
    assert gamma * beta == pytest.approx(1.0, rel=1e-12)
    dip = _dipole_for_B(5.0, gamma, beta)

    res = _FakeResults(
        s=[0.0, 1000.0],                # 1 m
        sigma_x=[1.0, 1.0], sigma_y=[1.0, 1.0],
        ref_gamma=[gamma, gamma], ref_beta=[beta, beta],
        ref_w_kin=[(gamma - 1.0) * 938.272] * 2,
        element_names=["D", "D"], mass_mev=938.272,
    )
    out = magnetic_stripping_loss(res, _FakeLattice([dip]), _FakeBeamConfig())

    expected_E_star = 1.0 * 2.99792458e8 * 5.0
    expected_rate = (5.0 / A1_SVPM) * math.exp(-A2_VPM / expected_E_star)

    assert out.B_T[0] == pytest.approx(5.0, rel=1e-9)
    assert out.E_star_V_per_m[0] == pytest.approx(expected_E_star, rel=1e-9)
    assert out.loss_rate_per_m[0] == pytest.approx(expected_rate, rel=1e-9)
    assert out.integral_loss[-1] == pytest.approx(expected_rate * 1.0, rel=1e-9)


def test_constant_b_low_field_negligible_loss():
    """γβ=0.5, B⊥=1T → E*=1.5e8 V/m → exp(-29.4) → rate ≈ 5.4e-8/m (tiny)."""
    gamma = math.sqrt(1.25)          # γβ = 0.5
    beta = 0.5 / gamma
    assert gamma * beta == pytest.approx(0.5, rel=1e-12)
    dip = _dipole_for_B(1.0, gamma, beta)

    res = _FakeResults(
        s=[0.0, 1000.0],
        sigma_x=[1.0, 1.0], sigma_y=[1.0, 1.0],
        ref_gamma=[gamma, gamma], ref_beta=[beta, beta],
        ref_w_kin=[(gamma - 1) * 938.272] * 2,
        element_names=["D", "D"], mass_mev=938.272,
    )
    out = magnetic_stripping_loss(res, _FakeLattice([dip]), _FakeBeamConfig())

    assert out.B_T[0] == pytest.approx(1.0, rel=1e-9)
    assert out.E_star_V_per_m[0] == pytest.approx(0.5 * 2.99792458e8, rel=1e-9)
    assert out.loss_rate_per_m[0] < 1.0e-7    # tiny loss rate
    assert out.loss_rate_per_m[0] > 1.0e-9    # but non-zero


# ---------------------------------------------------------------------------
# 2. Element-type B-field dispatch (the helper).
# ---------------------------------------------------------------------------
def test_dispatch_dipole_uses_brho_over_rho():
    from linac_gen.elements.dipole import Dipole
    # 30° bend with ρ = 1000 mm = 1 m.
    dip = Dipole(name="D1", angle=30.0, rho=1000.0)
    # At brho = 5 T·m, expected B = 5 / 1.0 = 5 T.
    b, kind = _design_b_for_element(dip, brho=5.0, sigma_x_mm=1.0, sigma_y_mm=1.0,
                                     quad_b_scale="2sigma", fieldmap_sample="max")
    assert kind == "dipole"
    assert b == pytest.approx(5.0, rel=1e-12)


def test_dispatch_solenoid_uses_perp_field_from_divergence():
    """Solenoid B_z is axial (∥ v); only B_z·θ_rms is perpendicular, so the
    dispatch must return field·θ_rms, NOT the full on-axis field."""
    from linac_gen.elements.solenoid import Solenoid
    sol = Solenoid(name="S1", length=300.0, field=2.5)
    b, kind = _design_b_for_element(sol, brho=10.0, sigma_x_mm=1.0, sigma_y_mm=1.0,
                                     quad_b_scale="2sigma", fieldmap_sample="max",
                                     theta_rms=2.0e-3)
    assert kind == "solenoid"
    assert b == pytest.approx(2.5 * 2.0e-3, rel=1e-12)   # B_z·θ, not 2.5


def test_solenoid_loss_is_negligible_not_catastrophic():
    """Regression for the B_z-as-B⊥ bug: a 6 T SC solenoid at 800 MeV must give
    a NEGLIGIBLE loss rate (B⊥=B_z·θ_rms ~ mT), not the ~1e5/m the old code
    produced by treating the axial field as perpendicular."""
    from linac_gen.elements.solenoid import Solenoid
    sol = Solenoid(name="SOL", length=300.0, field=6.0)
    gamma = 1.0 + 800.0 / 938.272          # 800 MeV H-
    beta = math.sqrt(1.0 - 1.0 / gamma ** 2)
    sm = np.zeros((2, 6, 6))               # σ_x' = σ_y' = 1 mrad → Σ22=Σ44=1
    sm[:, 1, 1] = 1.0
    sm[:, 3, 3] = 1.0
    res = _FakeResults(
        s=[0.0, 300.0], sigma_x=[1, 1], sigma_y=[1, 1],
        ref_gamma=[gamma, gamma], ref_beta=[beta, beta],
        ref_w_kin=[800.0, 800.0], element_names=["SOL", "SOL"],
        mass_mev=938.272, sigma_matrix=sm,
    )
    out = magnetic_stripping_loss(res, _FakeLattice([sol]), _FakeBeamConfig())
    theta = math.hypot(1.0e-3, 1.0e-3)
    assert out.B_T[0] == pytest.approx(6.0 * theta, rel=1e-9)   # ~8.5 mT, not 6 T
    assert out.loss_rate_per_m[0] < 1.0e-6                      # negligible


def test_dispatch_quad_2sigma_radius():
    from linac_gen.elements.quadrupole import Quadrupole
    # G = 30 T/m, σ_x = σ_y = 2 mm → 2σ_xy radius = 2·√(2²+2²) = 5.66 mm = 5.66e-3 m.
    q = Quadrupole(name="Q1", length=100.0, gradient=30.0)
    b, kind = _design_b_for_element(q, brho=5.0, sigma_x_mm=2.0, sigma_y_mm=2.0,
                                     quad_b_scale="2sigma", fieldmap_sample="max")
    assert kind == "quad"
    expected = 30.0 * (2.0 * math.sqrt(2.0 ** 2 + 2.0 ** 2)) * 1.0e-3
    assert b == pytest.approx(expected, rel=1e-12)


def test_dispatch_quad_pole_uses_aperture():
    from linac_gen.elements.quadrupole import Quadrupole
    q = Quadrupole(name="Q2", length=100.0, gradient=20.0, aperture=10.0)  # 10 mm bore
    b, kind = _design_b_for_element(q, brho=5.0, sigma_x_mm=1.0, sigma_y_mm=1.0,
                                     quad_b_scale="pole", fieldmap_sample="max")
    assert kind == "quad"
    assert b == pytest.approx(20.0 * 0.010, rel=1e-12)   # 0.2 T


def test_dispatch_drift_returns_zero():
    from linac_gen.elements.drift import Drift
    d = Drift(name="DR", length=500.0)
    b, kind = _design_b_for_element(d, brho=5.0, sigma_x_mm=1.0, sigma_y_mm=1.0,
                                     quad_b_scale="2sigma", fieldmap_sample="max")
    assert b == 0.0
    assert kind == "passive"


# ---------------------------------------------------------------------------
# 3. Guards & error paths.
# ---------------------------------------------------------------------------
def test_non_H_minus_species_raises():
    res = _FakeResults(
        s=[0.0, 1.0], sigma_x=[1, 1], sigma_y=[1, 1],
        ref_gamma=[1.5, 1.5], ref_beta=[0.7, 0.7],
        ref_w_kin=[100, 100], element_names=["X", "X"],
    )
    bc = _FakeBeamConfig(species="proton")
    with pytest.raises(ValueError, match="H-"):
        magnetic_stripping_loss(res, _FakeLattice([]), bc)


def test_drift_only_lattice_zero_loss():
    from linac_gen.elements.drift import Drift
    drift = Drift(name="D", length=1000.0)
    res = _FakeResults(
        s=[0.0, 500.0, 1000.0],
        sigma_x=[1, 1, 1], sigma_y=[1, 1, 1],
        ref_gamma=[2.0, 2.0, 2.0], ref_beta=[0.866, 0.866, 0.866],
        ref_w_kin=[938.272, 938.272, 938.272],
        element_names=["D", "D", "D"], mass_mev=938.272,
    )
    out = magnetic_stripping_loss(res, _FakeLattice([drift]), _FakeBeamConfig())

    assert np.all(out.B_T == 0.0)
    assert np.all(out.loss_rate_per_m == 0.0)
    assert out.integral_loss[-1] == 0.0
    assert out.integral_power_loss_W[-1] == 0.0


def test_zero_b_does_not_blow_up():
    """B=0 in the formula would give exp(-∞) — guard explicitly."""
    from linac_gen.elements.solenoid import Solenoid
    sol = Solenoid(name="S0", length=1000.0, field=0.0)   # turned off
    res = _FakeResults(
        s=[0.0, 1000.0],
        sigma_x=[1, 1], sigma_y=[1, 1],
        ref_gamma=[2.0, 2.0], ref_beta=[0.866, 0.866],
        ref_w_kin=[938, 938],
        element_names=["S0", "S0"], mass_mev=938.272,
    )
    out = magnetic_stripping_loss(res, _FakeLattice([sol]), _FakeBeamConfig())
    assert np.all(np.isfinite(out.loss_rate_per_m))
    assert np.all(out.loss_rate_per_m == 0.0)


def test_length_mismatch_raises():
    res = _FakeResults(
        s=[0.0, 1.0],
        sigma_x=[1.0],   # wrong length on purpose
        sigma_y=[1, 1],
        ref_gamma=[2.0, 2.0], ref_beta=[0.866, 0.866],
        ref_w_kin=[1.0, 1.0], element_names=["X", "X"], mass_mev=938.272,
    )
    with pytest.raises(ValueError, match="length mismatch"):
        magnetic_stripping_loss(res, _FakeLattice([]), _FakeBeamConfig())


# ---------------------------------------------------------------------------
# 4. Power-loss + cumulative integrals.
# ---------------------------------------------------------------------------
def test_power_scales_with_duty_and_current():
    """At fixed rate, doubling current OR duty doubles W/m, but not the
    fractional rate or the cumulative-loss curve."""
    gamma = 1.5
    beta = math.sqrt(1.0 - 1.0 / gamma ** 2)
    dip = _dipole_for_B(2.0, gamma, beta)   # real perpendicular 2 T → nonzero rate
    res = _FakeResults(
        s=[0.0, 1000.0], sigma_x=[1, 1], sigma_y=[1, 1],
        ref_gamma=[gamma, gamma], ref_beta=[beta, beta],
        ref_w_kin=[(gamma - 1) * 938.272] * 2,
        element_names=["D", "D"], mass_mev=938.272,
    )
    bc = _FakeBeamConfig(current=5.0, duty_cycle=10.0)

    a = magnetic_stripping_loss(res, _FakeLattice([dip]), bc)
    b = magnetic_stripping_loss(res, _FakeLattice([dip]), bc, current_mA=10.0)
    c = magnetic_stripping_loss(res, _FakeLattice([dip]), bc, duty_factor=0.20)

    # Fractional rate independent of current/duty
    np.testing.assert_allclose(a.loss_rate_per_m, b.loss_rate_per_m)
    np.testing.assert_allclose(a.loss_rate_per_m, c.loss_rate_per_m)
    # Power doubles with current
    np.testing.assert_allclose(b.power_loss_per_m_W, 2.0 * a.power_loss_per_m_W)
    # Power doubles with duty (a was duty=0.10, c is 0.20)
    np.testing.assert_allclose(c.power_loss_per_m_W, 2.0 * a.power_loss_per_m_W)
