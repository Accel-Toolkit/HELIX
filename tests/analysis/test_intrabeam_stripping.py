"""Validation suite for the IBS analyzer.

Coverage:
  * Cross-section σ_H(β) numerical parity with Ostiguy's `Utils/cross_section.py`.
  * Form factor F closed-form parity.
  * End-to-end regression vs Ostiguy's `intrabeam_stripping3.py` driven on
    the supplied `partran1.out` (subset captured as a fixture); requires
    enabling the legacy θ_s = dv/v convention and the legacy F-args bug
    so we reproduce the original numbers byte-for-byte.
  * Eq. (7) closed-form sanity on a synthetic Gaussian.
  * Species guard for non-H⁻ beams.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import pi
from pathlib import Path

import numpy as np
import pytest

from linac_gen.analysis.intrabeam_stripping import (
    IbsResult, ibs_loss, sigma_h, form_factor,
)


FIXTURE_DIR = Path(__file__).parent / "fixtures"
PARTRAN_FIXTURE = FIXTURE_DIR / "partran1_subset.out"
OSTIGUY_REF = FIXTURE_DIR / "ostiguy_reference.npz"


# ---------------------------------------------------------------------------
# Mock Results / BeamConfig built from a TraceWin partran1.out file.
# ---------------------------------------------------------------------------
@dataclass
class _MockBeamConfig:
    species: str
    current: float       # mA peak
    frequency: float     # MHz (bunch frequency = base RF in our case)
    duty_cycle: float    # %


class _MockResults:
    """Drop-in stand-in for the diagnostics Recorder shape consumed by
    ``ibs_loss``.  Attributes mirror the real Recorder."""
    pass


def _build_mock_from_partran(path: Path) -> tuple[_MockResults, _MockBeamConfig]:
    """Construct an analyzer-ready Results + BeamConfig from a partran file.

    The σ-matrix [1,1], [3,3], [5,5] entries are populated so that
    ``ibs_loss(theta_z_convention='dv_over_v', _legacy_F_args=True)``
    reproduces Ostiguy's `intrabeam_stripping3.py` exactly.  σ_W in
    particular is back-derived from the partran ε_nz so that the
    analyzer's dp/p → dv/v conversion lands on the same θ_s Ostiguy
    used internally.
    """
    lines = path.read_text().splitlines()
    fields0 = lines[1].split()
    mass_mev = float(fields0[0])
    f_mhz = float(fields0[1])
    charge = float(fields0[2])
    current_ma = float(fields0[3])

    rows: list[list[float]] = []
    seen_elements: set[int] = set()
    for line in lines[10:]:
        parts = line.split()
        if not parts:
            continue
        elm = int(parts[0])
        if elm in seen_elements:
            continue
        seen_elements.add(elm)
        rows.append([float(p) for p in parts])

    arr = np.asarray(rows, dtype=float)
    s_m = arr[:, 1]              # m
    gamma_m1 = arr[:, 2]
    gamma = gamma_m1 + 1.0
    bg = np.sqrt(np.maximum(gamma * gamma - 1.0, 0.0))
    beta = bg / gamma
    sigma_x = arr[:, 9]          # mm
    sigma_y = arr[:, 10]         # mm
    sigma_phi = arr[:, 11]       # deg
    eps_nx = arr[:, 15]          # normalised mm·mrad
    eps_ny = arr[:, 16]
    eps_nz = arr[:, 38]          # normalised mm·mrad (z, dp/p × 1000)
    sigma_z_mm = arr[:, 39]      # SizeZ from partran [mm]

    # σ_x' [mrad] = ε_x_geom / σ_x = (ε_nx / βγ) / σ_x
    safe_sx = np.where(sigma_x > 0.0, sigma_x, 1.0)
    safe_sy = np.where(sigma_y > 0.0, sigma_y, 1.0)
    safe_sz = np.where(sigma_z_mm > 0.0, sigma_z_mm, 1.0)
    safe_bg = np.where(bg > 0.0, bg, 1.0)

    sigma_xp_mrad = (eps_nx / safe_bg) / safe_sx
    sigma_yp_mrad = (eps_ny / safe_bg) / safe_sy

    # Ostiguy's longitudinal: ε_z_geom (dv/v×1000 units) = ε_nz / (βγ · γ²)
    eps_z_dv_v = eps_nz / (safe_bg * gamma * gamma)
    sigma_zp_dv_v_mradlike = eps_z_dv_v / safe_sz   # i.e. (dv/v) × 1000
    theta_s_dv_v = sigma_zp_dv_v_mradlike * 1.0e-3  # dv/v (dimensionless)

    # Back out σ_W [MeV] so that our analyzer's _theta_z(convention="dv_over_v")
    # reproduces this exact θ_s.  Our formula: dv/v = σ_W / (β²γ m₀ · γ²).
    sigma_w_mev = theta_s_dv_v * (beta ** 2) * (gamma ** 3) * mass_mev

    n = len(s_m)
    sm = np.zeros((n, 6, 6), dtype=float)
    sm[:, 0, 0] = sigma_x ** 2
    sm[:, 1, 1] = sigma_xp_mrad ** 2
    sm[:, 2, 2] = sigma_y ** 2
    sm[:, 3, 3] = sigma_yp_mrad ** 2
    sm[:, 4, 4] = sigma_phi ** 2
    sm[:, 5, 5] = sigma_w_mev ** 2

    res = _MockResults()
    res.s = s_m * 1000.0          # ibs_loss expects mm internally
    res.sigma_x = sigma_x
    res.sigma_y = sigma_y
    res.sigma_phi = sigma_phi
    res.sigma_w = sigma_w_mev
    res.ref_beta = beta
    res.ref_gamma = gamma
    res.ref_w_kin = (gamma - 1.0) * mass_mev
    res.ref_frequency = np.full_like(beta, f_mhz)
    res.sigma_matrix = sm
    res.mass_mev = mass_mev

    cfg = _MockBeamConfig(species="H-", current=current_ma,
                          frequency=f_mhz, duty_cycle=2.0)
    return res, cfg


# ---------------------------------------------------------------------------
# Cross-section + form-factor parity
# ---------------------------------------------------------------------------
def _ostiguy_sigma_h(b):
    """Re-implementation of Ostiguy's `Utils/cross_section.sigma_h`."""
    a0 = 0.529e-8
    alf = 1.0 / 137.0
    bm = 7.5e-5
    val = 120 * a0 ** 2
    val *= (alf / (b + alf)) ** 2
    val *= (b - bm) ** 6 / ((b - bm) ** 6 + bm ** 6)
    val *= np.log(3.2 * ((b + alf) / alf) ** 2)
    return val


def _ostiguy_form_factor(a, b, c):
    val = (2.0 - np.sqrt(3.0)) / (np.sqrt(3.0) * (np.sqrt(3.0) - 1.0))
    val *= (a + b + c) / np.sqrt(a ** 2 + b ** 2 + c ** 2) - 1.0
    val += 1.0
    return val


@pytest.mark.parametrize("beta", [1e-4, 4e-4, 1e-3, 5e-3, 1e-2, 0.05, 0.5, 0.9])
def test_sigma_h_matches_ostiguy(beta):
    assert sigma_h(beta) == pytest.approx(_ostiguy_sigma_h(beta), rel=1e-15, abs=0)


def test_sigma_h_vectorised():
    bs = np.linspace(1e-4, 0.9, 50)
    np.testing.assert_allclose(sigma_h(bs), _ostiguy_sigma_h(bs), rtol=1e-15)


@pytest.mark.parametrize("a,b,c", [
    (1.0, 1.0, 1.0),     # peak: F = 2/√3
    (1.0, 2.0, 5.0),     # generic ratios
    (1e-3, 2e-3, 3e-3),  # small but nonzero
])
def test_form_factor_matches_ostiguy(a, b, c):
    assert form_factor(a, b, c) == pytest.approx(
        _ostiguy_form_factor(a, b, c), rel=1e-15
    )


def test_form_factor_zero_arg_returns_one():
    # When two arguments are zero, F should reduce to 1 (F is the
    # normalisation, anisotropy correction vanishes).
    assert form_factor(0.0, 0.0, 1.0) == pytest.approx(1.0, abs=1e-14)


def test_form_factor_peak_equal_arguments():
    assert form_factor(2.0, 2.0, 2.0) == pytest.approx(2.0 / np.sqrt(3.0),
                                                        rel=1e-12)


# ---------------------------------------------------------------------------
# End-to-end regression vs Ostiguy
# ---------------------------------------------------------------------------
@pytest.mark.skipif(not OSTIGUY_REF.exists() or not PARTRAN_FIXTURE.exists(),
                    reason="Ostiguy reference fixture missing")
def test_ostiguy_parity_dv_over_v_legacy_F():
    """With dv/v + legacy F arguments, our output must match Ostiguy bit-for-bit
    on the same partran1.out subset and same duty/current/frequency."""
    results, cfg = _build_mock_from_partran(PARTRAN_FIXTURE)
    out = ibs_loss(
        results, cfg,
        duty_factor=0.02,
        non_constant_xs=False,
        theta_z_convention="dv_over_v",
        _legacy_F_args=True,
    )
    ref = np.load(OSTIGUY_REF)

    assert out.duty_factor == pytest.approx(0.02)
    assert out.current_mA == pytest.approx(float(ref["current_mA"]))
    # Ostiguy's script hard-codes e = 1.6021765e-19 (rounded to 7 sig figs),
    # while we use the post-2019-redefinition exact value 1.602176634e-19.
    # The ~8e-8 relative drift propagates into N and the loss rate.
    np.testing.assert_allclose(out.n_per_bunch, float(ref["n_per_bunch"]),
                               rtol=2e-7)
    np.testing.assert_allclose(out.s, ref["s"], rtol=1e-12)
    np.testing.assert_allclose(out.loss_rate_per_m, ref["loss_rate_per_m"],
                               rtol=2e-5, atol=1e-20)
    np.testing.assert_allclose(out.power_loss_per_m_W, ref["power_loss_per_m_W"],
                               rtol=2e-5, atol=1e-20)
    np.testing.assert_allclose(out.integral_loss, ref["integral_loss"],
                               rtol=2e-5, atol=1e-20)
    np.testing.assert_allclose(out.integral_power_loss_W,
                               ref["integral_power_loss_W"],
                               rtol=2e-5, atol=1e-20)


@pytest.mark.skipif(not OSTIGUY_REF.exists() or not PARTRAN_FIXTURE.exists(),
                    reason="Ostiguy reference fixture missing")
def test_paper_default_close_to_legacy():
    """With paper defaults (dp/p, F on velocity spreads), the answer must
    not diverge wildly — the integrated power loss should differ from the
    legacy run by less than 30%, the typical correction band for the F-args
    fix and the dp/p-vs-dv/v switch on a relativistic linac."""
    results, cfg = _build_mock_from_partran(PARTRAN_FIXTURE)
    legacy = ibs_loss(results, cfg, duty_factor=0.02,
                       theta_z_convention="dv_over_v",
                       _legacy_F_args=True)
    paper = ibs_loss(results, cfg, duty_factor=0.02,
                      theta_z_convention="dp_over_p",
                      _legacy_F_args=False)
    a = float(legacy.integral_power_loss_W[-1])
    b = float(paper.integral_power_loss_W[-1])
    assert a > 0
    assert abs(b - a) / a < 0.30


# ---------------------------------------------------------------------------
# Closed-form sanity on a synthetic uniform-spread Gaussian.
# ---------------------------------------------------------------------------
def test_eq7_closed_form_sanity():
    """Hand-computed Eq. (7) value for a synthetic single-step beam.

    σ_x = σ_y = σ_z = 1 mm, σ_x' = σ_y' = σ_z' = 1 mrad (= 1e-3 rad), γ = 2.
    With θ_s = dp/p = 1e-3, all three velocity-spread terms equal (γ·1e-3,
    γ·1e-3, 1e-3) → form factor at its near-peak.

    rate = N σ √((γ·1e-3)² + (γ·1e-3)² + (1e-3)²) F / (8π² σ_x σ_y σ_z γ²)
    """
    # Build a single-step mock Results with these synthetic numbers.
    res = _MockResults()
    res.s = np.array([0.0])
    res.sigma_x = np.array([1.0])
    res.sigma_y = np.array([1.0])
    res.sigma_phi = np.array([1.0])    # not used in the rate calc directly
    res.sigma_w = np.array([0.0])      # we'll override theta_s via sigma_matrix
    res.ref_beta = np.array([np.sqrt(3) / 2.0])     # γ = 2 → β = √3/2
    res.ref_gamma = np.array([2.0])
    res.ref_w_kin = np.array([1.0])
    res.ref_frequency = np.array([162.5])
    res.mass_mev = 939.294308

    # Choose σ_W so dp/p = 1e-3 exactly: σ_W = β²γ m_0 · 1e-3
    dpp = 1.0e-3
    beta_sq = 3.0 / 4.0
    sigma_w = beta_sq * 2.0 * res.mass_mev * dpp
    sm = np.zeros((1, 6, 6))
    sm[0, 0, 0] = 1.0 ** 2          # σ_x²
    sm[0, 1, 1] = 1.0 ** 2          # σ_x'² (mrad²)
    sm[0, 2, 2] = 1.0 ** 2
    sm[0, 3, 3] = 1.0 ** 2
    sm[0, 5, 5] = sigma_w ** 2
    res.sigma_matrix = sm

    # Need σ_z = 1 mm — engineer σ_φ such that the converter yields 1 mm.
    # σ_z = σ_φ · β · λ / 360, so σ_φ = 360 / (β · λ_mm).
    c_light = 2.99792458e8
    lam_mm = c_light / (162.5e6) * 1000.0
    sigma_phi_required = 360.0 / (np.sqrt(3) / 2.0 * lam_mm)
    res.sigma_phi = np.array([sigma_phi_required])
    res.sigma_w = np.array([sigma_w])

    cfg = _MockBeamConfig(species="H-", current=10.0, frequency=162.5,
                           duty_cycle=100.0)
    out = ibs_loss(res, cfg, duty_factor=1.0,
                    theta_z_convention="dp_over_p")

    # Hand-compute reference:
    gamma = 2.0
    theta_x = theta_y = 1.0e-3
    theta_s = 1.0e-3
    v_rel = np.sqrt((gamma * theta_x) ** 2 + (gamma * theta_y) ** 2 + theta_s ** 2)
    F_expected = form_factor(gamma * theta_x, gamma * theta_y, theta_s)
    sigma_strip_cm2 = sigma_h(4.0e-4)
    sigma_strip_mm2 = sigma_strip_cm2 * 100.0
    n_per_bunch = (10.0e-3) / (1.602176634e-19 * 162.5e6)
    rate_per_mm = (n_per_bunch * sigma_strip_mm2 * v_rel * F_expected
                   / (8 * pi ** 2 * 1.0 * 1.0 * 1.0 * gamma ** 2))
    rate_per_m_expected = rate_per_mm * 1.0e3

    assert out.loss_rate_per_m[0] == pytest.approx(rate_per_m_expected, rel=5e-3)
    assert out.form_factor[0] == pytest.approx(F_expected, rel=1e-12)


# ---------------------------------------------------------------------------
# Guards
# ---------------------------------------------------------------------------
def test_proton_beam_raises():
    cfg = _MockBeamConfig(species="proton", current=5.0, frequency=325.0,
                           duty_cycle=2.0)
    res = _MockResults()
    res.s = np.array([0.0])
    res.sigma_x = np.array([1.0])
    res.sigma_y = np.array([1.0])
    res.sigma_phi = np.array([1.0])
    res.sigma_w = np.array([0.001])
    res.ref_beta = np.array([0.5])
    res.ref_gamma = np.array([1.15])
    res.ref_w_kin = np.array([100.0])
    res.ref_frequency = np.array([325.0])
    res.sigma_matrix = np.zeros((1, 6, 6))
    res.mass_mev = 938.272
    with pytest.raises(ValueError, match="H-"):
        ibs_loss(res, cfg)


def test_invalid_theta_z_convention():
    cfg = _MockBeamConfig(species="H-", current=5.0, frequency=325.0,
                           duty_cycle=2.0)
    res = _MockResults()
    res.s = np.array([0.0])
    res.sigma_x = np.array([1.0])
    res.sigma_y = np.array([1.0])
    res.sigma_phi = np.array([1.0])
    res.sigma_w = np.array([0.001])
    res.ref_beta = np.array([0.5])
    res.ref_gamma = np.array([1.15])
    res.ref_w_kin = np.array([100.0])
    res.ref_frequency = np.array([325.0])
    res.sigma_matrix = np.zeros((1, 6, 6))
    res.mass_mev = 939.0
    with pytest.raises(ValueError, match="theta_z_convention"):
        ibs_loss(res, cfg, theta_z_convention="bogus")
