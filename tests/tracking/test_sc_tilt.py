"""Tilt-aware space-charge kernels (2026-08-07, S-R-anchor finding).

The analytic SC kernels were tilt-blind: a beam with ⟨xy⟩ != 0 got the
upright-ellipse field at projected sizes, missing the 45-deg quadrupole
component.  Found by the Struckmeier-Reiser 1984 external anchor: the
missing coupling fabricated an even⊗odd Krein "sum instability" in
solenoid channels at sigma_0 = 60 deg where exact K-V theory proves
stability (growth 1.24/period; vanishes with the correct field while
the REAL sigma_0 = 120 deg parametric band is untouched).

Pins here: rotational covariance of both kernels, bit-identity of the
upright default path, the solenoid Floquet regression, the MP sampling-
noise gate, and envelope<->MP parity of the tilted DC kick.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from linac_gen.core.config import BeamConfig
from linac_gen.distributions.factory import create_beam
from linac_gen.pic import pic_solver as ps
from linac_gen.tracking.envelope import (
    _sc_kick_matrix_2d_dc, _sc_kick_matrix_3d,
)

ARGS_2D = dict(current_mA=20.0, charge_state=1, mass_MeV=938.272,
               beta=0.08, gamma=1.0032, ds_mm=1.0)
ARGS_3D = dict(current_mA=5.0, charge_state=1, mass_MeV=938.272,
               beta=0.08, gamma=1.0032, frequency_MHz=162.5, ds_mm=1.0)


def _grad(M):
    """Transverse gradient tensor of a kick matrix."""
    return np.array([[M[1, 0], M[1, 2]], [M[3, 0], M[3, 2]]])


def _rotated_sigma(s1, s2, psi_deg):
    """Spatial moments of a (s1, s2) ellipse rotated by psi."""
    psi = math.radians(psi_deg)
    c, s = math.cos(psi), math.sin(psi)
    Q = np.array([[c, -s], [s, c]])
    S = Q @ np.diag([s1 * s1, s2 * s2]) @ Q.T
    return math.sqrt(S[0, 0]), math.sqrt(S[1, 1]), S[0, 1], Q


@pytest.mark.parametrize("psi_deg", [17.0, 45.0, -30.0])
def test_2d_dc_rotation_covariance(psi_deg):
    """G(Q Σ Qᵀ) == Q G(Σ) Qᵀ — the tilted kick is exactly the rotated
    upright kick, no new physics constants."""
    sx, sy, sxy, Q = _rotated_sigma(2.0, 1.0, psi_deg)
    M_up = _sc_kick_matrix_2d_dc(sigma_x_mm=2.0, sigma_y_mm=1.0,
                                 **ARGS_2D)
    M_t = _sc_kick_matrix_2d_dc(sigma_x_mm=sx, sigma_y_mm=sy,
                                sigma_xy_mm2=sxy, **ARGS_2D)
    np.testing.assert_allclose(_grad(M_t), Q @ _grad(M_up) @ Q.T,
                               rtol=0, atol=1e-15)


@pytest.mark.parametrize("psi_deg", [17.0, -30.0])
def test_3d_rotation_covariance(psi_deg):
    """Same covariance for the bunched ellipsoid; the longitudinal kick
    must equal the upright principal-axes value (rotation-invariant)."""
    sx, sy, sxy, Q = _rotated_sigma(2.0, 1.0, psi_deg)
    M_up = _sc_kick_matrix_3d(sigma_x_mm=2.0, sigma_y_mm=1.0,
                              sigma_phi_deg=5.0, **ARGS_3D)
    M_t = _sc_kick_matrix_3d(sigma_x_mm=sx, sigma_y_mm=sy,
                             sigma_phi_deg=5.0, sigma_xy_mm2=sxy,
                             **ARGS_3D)
    np.testing.assert_allclose(_grad(M_t), Q @ _grad(M_up) @ Q.T,
                               rtol=0, atol=1e-15)
    np.testing.assert_allclose(M_t[5, 4], M_up[5, 4], rtol=1e-12)


def test_upright_default_bit_identical():
    """sigma_xy_mm2 = 0 (the default) must reproduce the historical
    upright kick BIT-IDENTICALLY, with zero cross terms."""
    for maker, kw in ((_sc_kick_matrix_2d_dc, ARGS_2D),
                      (_sc_kick_matrix_3d, dict(ARGS_3D,
                                                sigma_phi_deg=5.0))):
        M_default = maker(sigma_x_mm=2.0, sigma_y_mm=1.0, **kw)
        M_explicit = maker(sigma_x_mm=2.0, sigma_y_mm=1.0,
                           sigma_xy_mm2=0.0, **kw)
        assert np.array_equal(M_default, M_explicit)
        assert M_default[1, 2] == 0.0 and M_default[3, 0] == 0.0


@pytest.mark.physics
def test_solenoid_floquet_krein_artifact_regression():
    """THE anchor pin: Maryland solenoid cell (S-R 1984) at sigma_0 =
    60 deg, deep depression.  Exact K-V theory proves stability; the
    tilt-blind kernel fabricated growth 1.24/period through spurious
    even<->odd coupling.  Must stay quiet forever."""
    pytest.importorskip(
        "linac_gen.analysis.envelope_floquet",
        reason="Floquet analyzer not present in this distribution")
    from linac_gen.analysis.envelope_floquet import envelope_floquet
    from linac_gen.analysis.period_detect import PeriodicStructure
    from linac_gen.core.lattice import Lattice
    from linac_gen.core.particle import PROTON
    from linac_gen.core.reference import ReferenceParticle
    from linac_gen.elements.drift import Drift
    from linac_gen.elements.solenoid import Solenoid

    lat = Lattice()
    for _ in range(3):
        lat.add(Drift("O1", 51.0, aperture=100.0))
        lat.add(Solenoid("SOL", 34.0, field=10.84422, aperture=100.0))
        lat.add(Drift("O2", 51.0, aperture=100.0))
    per = PeriodicStructure(start=0, end=9, inner_period_length=3,
                            inner_slice_end=3, n_repeats=3,
                            label="sr-sol", source="manual")
    ref = ReferenceParticle(species=PROTON, w_kin=3.0, frequency=162.5)
    init = dict(alpha_x=0.0, beta_x=1.0, emit_x=1.0,
                alpha_y=0.0, beta_y=1.0, emit_y=1.0,
                alpha_z=0.0, beta_z=2.0, emit_z=0.3, continuous=True)
    f = envelope_floquet(lat, ref, per, 780.0, init)
    assert f.get("reason") is None
    assert f["max_growth"] < 1.001, f["max_growth"]


def _dc_beam(beta_y=0.125):
    cfg = BeamConfig(species="H-", energy=0.030, frequency=162.5,
                     current=5.0, n_particles=20000, continuous=True,
                     emit_nx=0.2, alpha_x=0.0, beta_x=0.5,
                     emit_ny=0.2, alpha_y=0.0, beta_y=beta_y,
                     emit_z=0.06, alpha_z=0.0, beta_z=500.0)
    return create_beam(cfg, seed=3)


def _dc_beam_symmetrized(beta_y=0.125):
    """DC beam whose spatial cloud is 4-fold mirror-symmetrized:
    sample ⟨xy⟩ and centroid are EXACTLY zero (float-sum residual only),
    so the upright branch is guaranteed regardless of the noise gate —
    the clean reference for rotation/parity tests."""
    b = _dc_beam(beta_y)
    n = b.particles.shape[0]
    q = n // 4
    x = b.particles[:q, 0].copy()
    y = b.particles[:q, 2].copy()
    for k, (sx_, sy_) in enumerate(((1, 1), (1, -1), (-1, 1), (-1, -1))):
        b.particles[k * q:(k + 1) * q, 0] = sx_ * x
        b.particles[k * q:(k + 1) * q, 2] = sy_ * y
    return b


def test_mp_upright_beam_keeps_historical_path():
    """A random upright ensemble ALWAYS has |cov| ~ sigma^2/sqrt(N);
    the noise gate must keep it on the upright branch — per-particle
    kick exactly proportional to own x (no y admixture)."""
    b = _dc_beam()
    xs = b.particles[b.alive_mask, 0].copy()
    ys = b.particles[b.alive_mask, 2].copy()
    xp0 = b.particles[:, 1].copy()
    ps.kick_continuous_2d(b, 1.0)
    d = b.particles[b.alive_mask, 1] - xp0[b.alive_mask]
    xr = xs - xs.mean()
    yr = ys - ys.mean()
    A = np.column_stack([xr, yr])
    (a, bcoef), *_ = np.linalg.lstsq(A, d, rcond=None)
    # upright branch: kick depends on own x ONLY -> y-coefficient is
    # float noise; the tilted branch would put ~rho * a into bcoef
    assert abs(bcoef) < 1e-9 * abs(a), (a, bcoef)


def test_mp_tilted_kick_rotation_covariance():
    """Rotating the beam rotates the kick: kick(Q r) == Q kick(r)."""
    psi = math.radians(30.0)
    c, s = math.cos(psi), math.sin(psi)
    b_up = _dc_beam_symmetrized()
    b_rot = _dc_beam_symmetrized()
    x, y = b_up.particles[:, 0].copy(), b_up.particles[:, 2].copy()
    b_rot.particles[:, 0] = c * x - s * y
    b_rot.particles[:, 2] = s * x + c * y

    xp_u = b_up.particles[:, 1].copy(); yp_u = b_up.particles[:, 3].copy()
    xp_r = b_rot.particles[:, 1].copy(); yp_r = b_rot.particles[:, 3].copy()
    ps.kick_continuous_2d(b_up, 1.0)
    ps.kick_continuous_2d(b_rot, 1.0)
    du = b_up.particles[:, 1] - xp_u
    dv = b_up.particles[:, 3] - yp_u
    np.testing.assert_allclose(b_rot.particles[:, 1] - xp_r,
                               c * du - s * dv, rtol=0, atol=1e-12)
    np.testing.assert_allclose(b_rot.particles[:, 3] - yp_r,
                               s * du + c * dv, rtol=0, atol=1e-12)


def test_mp_tilted_matches_envelope_kernel():
    """Envelope<->MP parity for a tilted DC beam: the per-particle kick
    field of the MP branch must be the envelope kernel's G tensor at
    the same moments (lossless beam -> same current)."""
    psi = math.radians(30.0)
    c, s = math.cos(psi), math.sin(psi)
    b = _dc_beam_symmetrized()
    x, y = b.particles[:, 0].copy(), b.particles[:, 2].copy()
    b.particles[:, 0] = c * x - s * y
    b.particles[:, 2] = s * x + c * y
    xs = b.particles[b.alive_mask, 0]
    ys = b.particles[b.alive_mask, 2]
    sx = float(np.std(xs)); sy = float(np.std(ys))
    sxy = float(np.mean((xs - xs.mean()) * (ys - ys.mean())))

    xp0 = b.particles[:, 1].copy(); yp0 = b.particles[:, 3].copy()
    ps.kick_continuous_2d(b, 1.0)
    dxp = b.particles[b.alive_mask, 1] - xp0[b.alive_mask]
    dyp = b.particles[b.alive_mask, 3] - yp0[b.alive_mask]

    M = _sc_kick_matrix_2d_dc(
        current_mA=5.0, charge_state=b.ref.species.charge,
        mass_MeV=b.ref.species.mass, beta=b.ref.beta, gamma=b.ref.gamma,
        sigma_x_mm=sx, sigma_y_mm=sy, ds_mm=1.0, sigma_xy_mm2=sxy)
    xr = xs - xs.mean(); yr = ys - ys.mean()
    np.testing.assert_allclose(dxp, M[1, 0] * xr + M[1, 2] * yr,
                               rtol=1e-10, atol=1e-14)
    np.testing.assert_allclose(dyp, M[3, 0] * xr + M[3, 2] * yr,
                               rtol=1e-10, atol=1e-14)
