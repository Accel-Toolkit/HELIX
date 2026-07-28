"""Physics regression tests for the PIC space-charge kick formulas.

Pins two claims that have subtle history:

1. The LONGITUDINAL kick has NO γ-factor suppression.  E_z is Lorentz-
   invariant under a z-boost and dW/dt = q·E_z·v_z gives ΔW = q·E_z·Δs
   (no γ).  A previous version had a spurious 1/γ² here.

2. The TRANSVERSE angular kick is q·E_rest·Δs/(β²γ²·mc²): one γ from
   v×B cancellation of the lab-frame Lorentz force, one βγ from the
   p_z used to convert Δp → Δx'.

Both claims are tested together by building a bunch that is *spherically
symmetric in the beam rest frame*: at any rest-frame point the three
rest-frame field components have the same statistical variance.  If the
kick factors are right, the inferred ⟨E²⟩ from ΔW and from Δx' agree.
"""
import math

import numpy as np
import pytest

from linac_gen.pic.pic_solver import PicSolver
from linac_gen.core.config import SpaceChargeConfig
from linac_gen.core.beam import Beam
from linac_gen.core.reference import ReferenceParticle
from linac_gen.core.particle import PROTON, H_MINUS


def _rest_spherical_beam(n: int, sigma_rest_mm: float, w_kin: float,
                         current_mA: float = 50.0, seed: int = 0) -> Beam:
    """Gaussian bunch whose rest-frame shape is σ_rest in x, y, z.

    Rest-frame z is stretched by γ vs lab z, so we set σ_z_lab = σ_rest / γ.
    Transverse is unchanged by a z-boost so σ_x_lab = σ_y_lab = σ_rest.
    """
    ref = ReferenceParticle(species=PROTON, w_kin=w_kin, frequency=352.21)
    gamma = ref.gamma
    beta = ref.beta
    wavelength_mm = ref.wavelength

    # z in rest frame has σ = sigma_rest_mm → σ_z_lab = σ_rest / γ
    sigma_z_lab_mm = sigma_rest_mm / gamma
    # Convert σ_z_lab (mm) to σ_dphi (deg): z_lab = −dphi · β · λ / 360
    sigma_dphi_deg = sigma_z_lab_mm * 360.0 / (beta * wavelength_mm)

    beam = Beam(ref=ref, n_particles=n, current=current_mA)
    rng = np.random.default_rng(seed)
    beam.particles[:, 0] = rng.normal(0.0, sigma_rest_mm, n)
    beam.particles[:, 1] = 0.0
    beam.particles[:, 2] = rng.normal(0.0, sigma_rest_mm, n)
    beam.particles[:, 3] = 0.0
    beam.particles[:, 4] = rng.normal(0.0, sigma_dphi_deg, n)
    beam.particles[:, 5] = 0.0
    return beam


def _apply_one_kick(beam: Beam, ds_mm: float) -> tuple[np.ndarray, np.ndarray]:
    """Apply one PIC kick with fixed grid and return (Δx', ΔW) arrays."""
    cfg = SpaceChargeConfig(nx=48, ny=48, nz=48, grid_extent=5.0,
                            grid_mode="fixed")
    solver = PicSolver(cfg)
    xp_before = beam.particles[:, 1].copy()
    dw_before = beam.particles[:, 5].copy()
    solver.kick(beam, ds_mm)
    return (beam.particles[:, 1] - xp_before,
            beam.particles[:, 5] - dw_before)


def test_rest_frame_spherical_symmetry_of_inferred_fields():
    """For a rest-frame-spherical bunch, inferring ⟨E²⟩ from the longitudinal
    and transverse kicks must give the same answer — otherwise the ratio
    factor_z / factor_t encodes a spurious γ."""
    n = 20_000
    sigma_rest = 2.0  # mm
    ds_mm = 5.0

    # Use a high-γ beam so a stray γ² in factor_z would clearly violate
    # the isotropy bound below.  w_kin = 469.65 MeV → γ ≈ 1.5.
    beam = _rest_spherical_beam(n=n, sigma_rest_mm=sigma_rest, w_kin=469.65)
    ref = beam.ref
    mass = ref.species.mass     # MeV
    beta, gamma = ref.beta, ref.gamma

    dxp, dW = _apply_one_kick(beam, ds_mm)

    # Invert the kick formulas to recover E at each particle.
    ds_m = ds_mm * 1e-3
    factor_t = ds_m * 1e3 / (mass * 1e6 * beta ** 2 * gamma ** 2)
    factor_z = ds_m * 1e-6
    Ex_inferred = dxp / factor_t           # V/m (rest frame)
    Ez_inferred = dW / factor_z            # V/m (rest frame)

    # Restrict to particles in a shell around the rms radius where the
    # statistics are cleanest.
    x = beam.particles[:, 0]
    y = beam.particles[:, 2]
    z_lab = -beam.particles[:, 4] * beta * ref.wavelength / 360.0
    z_rest = z_lab * gamma
    r_rest = np.sqrt(x * x + y * y + z_rest * z_rest)
    shell = (r_rest > 0.5 * sigma_rest) & (r_rest < 1.5 * sigma_rest)

    Ex2 = np.mean(Ex_inferred[shell] ** 2)
    Ez2 = np.mean(Ez_inferred[shell] ** 2)

    # Isotropy claim: the two variances agree to within PIC noise (≤ 15 %).
    ratio = Ez2 / Ex2
    assert 0.80 < ratio < 1.25, (
        f"rest-frame isotropy violated: <Ez²>/<Ex²> = {ratio:.3f}; "
        f"a ratio != 1 typically signals a spurious γ factor in factor_z "
        f"or factor_t. (Ex² = {Ex2:.3e}, Ez² = {Ez2:.3e})"
    )


def test_longitudinal_kick_is_gamma_invariant():
    """Same rest-frame bunch geometry at two different γ → identical ⟨ΔW²⟩.

    E_z is Lorentz-invariant and ΔW = q·E_z·Δs carries no explicit γ,
    so reshaping the lab bunch to preserve rest-frame geometry must keep
    ⟨ΔW²⟩ the same.  If the kick formula had a γ² factor, the ratio
    between the two runs would be (γ₁/γ₂)⁴ — easily detectable.
    """
    n = 20_000
    sigma_rest = 2.0   # mm
    ds_mm = 5.0
    current_mA = 50.0

    # Two kinetic energies giving γ ≈ 1.005 and γ ≈ 1.5 (proton).
    w1 = 5.0        # γ ≈ 1.0053
    w2 = 469.65     # γ ≈ 1.50

    def _dw2(w_kin):
        beam = _rest_spherical_beam(n=n, sigma_rest_mm=sigma_rest,
                                    w_kin=w_kin, current_mA=current_mA)
        _, dW = _apply_one_kick(beam, ds_mm)
        return np.mean(dW ** 2)

    dw2_lo = _dw2(w1)
    dw2_hi = _dw2(w2)
    ratio = dw2_hi / dw2_lo

    # Expected: identical (ratio = 1).  If the old 1/γ² bug returns the
    # ratio would be (γ_lo/γ_hi)⁴ ≈ (1.005/1.5)⁴ ≈ 2e-4 — four orders of
    # magnitude different.  Allow ±30 % for PIC discretisation noise.
    assert 0.7 < ratio < 1.3, (
        f"longitudinal kick is NOT γ-invariant: ratio {ratio:.3e} "
        f"(expected ≈ 1; a 1/γ² regression would give ≈ 2e-4)"
    )


def test_sc_is_repulsive_for_negative_species():
    """Space-charge between like charges is repulsive regardless of their sign.

    Historically the kick carried a species-sign prefix while the charge
    deposition was unsigned, producing an *attractive* force for H⁻ (the
    sign flipped once in the effective field and again in the kick).  A
    uniform-ish H⁻ bunch at zero initial divergence must gain outward
    momentum after one SC kick.
    """
    n = 5000
    sigma = 1.5  # mm
    current_mA = 40.0

    ref = ReferenceParticle(species=H_MINUS, w_kin=2.1227, frequency=162.5)
    beam = Beam(ref=ref, n_particles=n, current=current_mA)
    rng = np.random.default_rng(0)
    beam.particles[:, 0] = rng.normal(0.0, sigma, n)    # x
    beam.particles[:, 1] = 0.0                           # xp ← zero initially
    beam.particles[:, 2] = rng.normal(0.0, sigma, n)    # y
    beam.particles[:, 3] = 0.0
    beam.particles[:, 4] = rng.normal(0.0, 5.0, n)      # dphi
    beam.particles[:, 5] = 0.0

    cfg = SpaceChargeConfig(nx=48, ny=48, nz=48, grid_extent=5.0,
                            grid_mode="fixed")
    PicSolver(cfg).kick(beam, ds=10.0)

    # After a repulsive kick, the correlation ⟨x · x'⟩ must be *positive*
    # (particles at +x acquire positive x', particles at −x acquire
    # negative x'); an attractive force would give ⟨x·x'⟩ < 0.
    xxp_corr = float(np.mean(beam.particles[:, 0] * beam.particles[:, 1]))
    assert xxp_corr > 0.0, (
        f"H⁻ SC kick is ATTRACTIVE (⟨x·x'⟩ = {xxp_corr:.3e}); "
        f"like-charge space charge must be repulsive."
    )


def test_transverse_kick_scales_as_inv_beta2_gamma2():
    """Same rest-frame bunch at two γ → Δx'² ratio = (β²γ²)_lo² / (β²γ²)_hi².

    Pins the β²γ² in the transverse kick factor.  If that γ²
    were removed, the ratio would be off by (γ_hi/γ_lo)⁴.
    """
    n = 20_000
    sigma_rest = 2.0
    ds_mm = 5.0
    current_mA = 50.0

    w1 = 5.0
    w2 = 469.65

    def _run(w_kin):
        beam = _rest_spherical_beam(n=n, sigma_rest_mm=sigma_rest,
                                    w_kin=w_kin, current_mA=current_mA)
        dxp, _ = _apply_one_kick(beam, ds_mm)
        return np.mean(dxp ** 2), beam.ref.beta, beam.ref.gamma

    dxp2_lo, beta_lo, gamma_lo = _run(w1)
    dxp2_hi, beta_hi, gamma_hi = _run(w2)

    meas_ratio = dxp2_hi / dxp2_lo
    expected_ratio = ((beta_lo * gamma_lo) ** 2
                      / (beta_hi * gamma_hi) ** 2) ** 2

    # Expected ratio ≈ (β²γ²_lo / β²γ²_hi)² ≈ 3e-6 at these energies.
    # Allow ±30 % tolerance for PIC noise.
    rel_err = abs(meas_ratio - expected_ratio) / expected_ratio
    assert rel_err < 0.30, (
        f"transverse kick β²γ² scaling broken: "
        f"measured ratio {meas_ratio:.3e}, expected {expected_ratio:.3e} "
        f"(rel err {rel_err:.2%})"
    )


# ---------------------------------------------------------------------------
# Charge-state |Z| scaling (fixed 2026-07-10: the kick factors previously
# hard-coded |q| = e, under-kicking multiply charged ions by |Z|).
# ---------------------------------------------------------------------------
def test_kick_scales_linearly_with_charge_state():
    """Identical bunch coordinates and current (identical deposited source
    charge I/f, hence identical E-field) but |Z| = 2 test particles must
    receive exactly twice the |Z| = 1 kick, transverse AND longitudinal."""
    from linac_gen.core.particle import Particle

    n = 5_000
    ds_mm = 5.0
    he_like_z1 = Particle(mass=3727.379, charge=1, name="he-like Z=1")
    he_like_z2 = Particle(mass=3727.379, charge=2, name="alpha")

    def _run(species):
        ref = ReferenceParticle(species=species, w_kin=10.0, frequency=352.21)
        beam = Beam(ref=ref, n_particles=n, current=25.0)
        rng = np.random.default_rng(7)
        beam.particles[:, 0] = rng.normal(0.0, 2.0, n)
        beam.particles[:, 2] = rng.normal(0.0, 2.0, n)
        beam.particles[:, 4] = rng.normal(0.0, 5.0, n)
        beam.particles[:, [1, 3, 5]] = 0.0
        return _apply_one_kick(beam, ds_mm)

    dxp_z1, dW_z1 = _run(he_like_z1)
    dxp_z2, dW_z2 = _run(he_like_z2)

    # Same mass, energy, coordinates, seed and current -> identical field;
    # the ONLY difference is the |Z| factor in the kick, so the ratio is
    # exactly 2 up to floating-point roundoff.
    assert np.allclose(dxp_z2, 2.0 * dxp_z1, rtol=1e-12, atol=1e-18), \
        "transverse PIC kick does not scale linearly with |Z|"
    assert np.allclose(dW_z2, 2.0 * dW_z1, rtol=1e-12, atol=1e-18), \
        "longitudinal PIC kick does not scale linearly with |Z|"
    # And it is a real kick, not 0 == 2*0.
    assert np.max(np.abs(dxp_z1)) > 0.0


def test_torch_kick_scales_linearly_with_charge_state():
    """The differentiable PIC kick honours charge_state the same way."""
    torch = pytest.importorskip("torch")
    from linac_gen.pic.torch.sc_kick import torch_pic_sc_kick

    cfg = SpaceChargeConfig(nx=16, ny=16, nz=16, grid_extent=5.0,
                            grid_mode="fixed")
    rng = np.random.default_rng(11)
    n = 2_000
    parts = np.zeros((n, 6))
    parts[:, 0] = rng.normal(0.0, 2.0, n)
    parts[:, 2] = rng.normal(0.0, 2.0, n)
    parts[:, 4] = rng.normal(0.0, 5.0, n)
    X = torch.as_tensor(parts, dtype=torch.float64)

    kw = dict(ds_mm=5.0, gamma=1.01, beta=0.14, mass_mev=3727.379,
              wavelength_mm=851.0, macro_charge=1e-14)
    d1 = torch_pic_sc_kick(X.clone(), cfg, charge_state=1.0, **kw) - X
    d2 = torch_pic_sc_kick(X.clone(), cfg, charge_state=2.0, **kw) - X

    assert torch.allclose(d2, 2.0 * d1, rtol=1e-12, atol=1e-18), \
        "torch PIC kick does not scale linearly with charge_state"
    assert float(d1.abs().max()) > 0.0
