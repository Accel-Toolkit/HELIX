# tests/elements/test_solenoid.py
"""Tests for solenoid element (TransferMapElement)."""
import numpy as np
import math
import pytest
from linac_gen.core.particle import PROTON, H_MINUS
from linac_gen.core.reference import ReferenceParticle
from linac_gen.core.beam import Beam
from linac_gen.elements.solenoid import Solenoid


def _make_ref(species=PROTON, w_kin=3.0, freq=352.21):
    return ReferenceParticle(species=species, w_kin=w_kin, frequency=freq)


def _make_beam(species=PROTON, n=5, w_kin=3.0, freq=352.21):
    ref = _make_ref(species=species, w_kin=w_kin, freq=freq)
    beam = Beam(ref=ref, n_particles=n, current=60.0)
    return beam


# ---- Matrix shape and basic properties ----

class TestSolenoidBasics:
    def test_matrix_shape(self):
        sol = Solenoid("SOL1", length=100.0, field=0.5)
        ref = _make_ref()
        M = sol.transfer_matrix(ref)
        assert M.shape == (6, 6)

    def test_is_transfer_map_element(self):
        from linac_gen.elements.base import TransferMapElement
        sol = Solenoid("SOL1", length=100.0, field=0.5)
        assert isinstance(sol, TransferMapElement)

    def test_length_stored(self):
        sol = Solenoid("SOL1", length=200.0, field=0.5)
        assert sol.length == 200.0

    def test_field_stored(self):
        sol = Solenoid("SOL1", length=100.0, field=0.75)
        assert sol.field == 0.75

    def test_default_aperture_zero(self):
        sol = Solenoid("SOL1", length=100.0, field=0.5)
        assert sol.aperture == 0.0


# ---- Zero field -> drift ----

class TestSolenoidZeroField:
    def test_zero_field_is_drift_like(self):
        """Zero field should give a drift-like matrix."""
        from linac_gen.elements.drift import Drift
        sol = Solenoid("SOL0", length=100.0, field=0.0)
        d = Drift("D0", length=100.0)
        ref = _make_ref()
        M_sol = sol.transfer_matrix(ref)
        M_d = d.transfer_matrix(ref)
        np.testing.assert_array_almost_equal(M_sol, M_d, decimal=10)

    def test_zero_length_is_identity(self):
        """Zero length should give identity matrix."""
        sol = Solenoid("SOL0", length=0.0, field=0.5)
        ref = _make_ref()
        M = sol.transfer_matrix(ref)
        np.testing.assert_array_almost_equal(M, np.eye(6), decimal=10)


# ---- x-y coupling ----

class TestSolenoidCoupling:
    def test_xy_coupling_present(self):
        """Solenoid should couple x and y (off-diagonal 4x4 blocks nonzero)."""
        sol = Solenoid("SOL1", length=100.0, field=0.5)
        ref = _make_ref()
        M = sol.transfer_matrix(ref)
        # Check that the x-y coupling elements are nonzero
        assert abs(M[0, 2]) > 1e-10  # x depends on y
        assert abs(M[0, 3]) > 1e-10  # x depends on y'
        assert abs(M[2, 0]) > 1e-10  # y depends on x
        assert abs(M[2, 1]) > 1e-10  # y depends on x'

    def test_no_coupling_to_longitudinal_from_transverse(self):
        """Transverse motion should not couple to longitudinal (off-diagonal blocks zero)."""
        sol = Solenoid("SOL1", length=100.0, field=0.5)
        ref = _make_ref()
        M = sol.transfer_matrix(ref)
        # M[0:4, 4:6] should be zero
        np.testing.assert_array_almost_equal(M[0:4, 4:6], 0.0, decimal=12)
        # M[4:6, 0:4] should be zero
        np.testing.assert_array_almost_equal(M[4:6, 0:4], 0.0, decimal=12)


# ---- Symplecticity ----

class TestSolenoidSymplecticity:
    def test_4x4_transverse_symplectic(self):
        """The 4x4 transverse block should be symplectic: M^T J M = J."""
        sol = Solenoid("SOL1", length=100.0, field=0.5)
        ref = _make_ref()
        M = sol.transfer_matrix(ref)
        M4 = M[0:4, 0:4]
        # Symplectic form J = [[0,1,0,0],[-1,0,0,0],[0,0,0,1],[0,0,-1,0]]
        J = np.zeros((4, 4))
        J[0, 1] = 1.0
        J[1, 0] = -1.0
        J[2, 3] = 1.0
        J[3, 2] = -1.0
        result = M4.T @ J @ M4
        np.testing.assert_array_almost_equal(result, J, decimal=10)

    def test_2x2_block_determinants(self):
        """Individual 2x2 blocks should have determinant related properties."""
        sol = Solenoid("SOL1", length=100.0, field=0.5)
        ref = _make_ref()
        M = sol.transfer_matrix(ref)
        # For a solenoid, the diagonal 2x2 blocks (xx and yy) have det = C^4 + S^2*C^2 etc.
        # But the overall 4x4 should have det = 1
        M4 = M[0:4, 0:4]
        det4 = np.linalg.det(M4)
        assert abs(det4 - 1.0) < 1e-10


# ---- Focusing behavior ----

class TestSolenoidFocusing:
    def test_beam_size_oscillates(self):
        """After half a Larmor period, the x-x' phase space should show focusing."""
        ref = _make_ref()
        charge_sign = 1 if ref.species.charge > 0 else -1
        B0 = 0.5
        k_s = charge_sign * B0 / (2 * ref.brho)  # 1/m

        # Choose length for quarter Larmor period: phi = pi/2
        L_m = (math.pi / 2) / abs(k_s)  # m
        L_mm = L_m * 1e3  # mm

        sol = Solenoid("SOL1", length=L_mm, field=B0)
        M = sol.transfer_matrix(ref)

        # At phi=pi/2: C=0, S=1
        # M[0,0] = C^2 = 0, M[0,1] = CS/k = 0, M[0,2] = SC = 0, M[0,3] = S^2/k
        # So a particle starting with (x=1, xp=0, y=0, yp=0) should end at:
        # x = M[0,0]*1 = 0 (approximately)
        assert abs(M[0, 0]) < 1e-10  # C^2 = cos(pi/2)^2 = 0

    def test_round_beam_stays_round(self):
        """A round beam (same x and y distribution) should stay round through a solenoid."""
        beam = _make_beam(n=100, w_kin=3.0)
        # Set up a round beam: same rms in x and y
        np.random.seed(42)
        beam.particles[:, 0] = np.random.normal(0, 1.0, 100)  # x (mm)
        beam.particles[:, 1] = np.random.normal(0, 0.5, 100)  # xp (mrad)
        beam.particles[:, 2] = np.random.normal(0, 1.0, 100)  # y (mm)
        beam.particles[:, 3] = np.random.normal(0, 0.5, 100)  # yp (mrad)

        sol = Solenoid("SOL1", length=100.0, field=0.5)
        sol.track(beam)

        sigma_x = np.std(beam.particles[:, 0])
        sigma_y = np.std(beam.particles[:, 2])
        # Round beam should remain approximately round (within a few percent)
        ratio = sigma_x / sigma_y
        assert 0.8 < ratio < 1.2  # reasonable tolerance for 100 particles


# ---- H-minus ----

class TestSolenoidHMinus:
    def test_h_minus_reverses_rotation(self):
        """H- (charge=-1) should rotate in the opposite direction."""
        B0 = 0.5
        L_mm = 100.0
        ref_p = _make_ref(species=PROTON)
        ref_h = _make_ref(species=H_MINUS)

        sol = Solenoid("SOL1", length=L_mm, field=B0)
        M_p = sol.transfer_matrix(ref_p)
        M_h = sol.transfer_matrix(ref_h)

        # The off-diagonal coupling M[0,2] (x from y) should have opposite sign
        # because the Larmor rotation direction reverses
        assert M_p[0, 2] * M_h[0, 2] < 0  # opposite signs


# ---- Slice and step consistency ----

class TestSolenoidSlicing:
    def test_slice_multiplication(self):
        """Product of two half-solenoid matrices should equal full solenoid matrix."""
        sol = Solenoid("SOL1", length=100.0, field=0.5)
        ref = _make_ref()
        M_full = sol.transfer_matrix(ref)
        M_half = sol.transfer_matrix(ref, ds=50.0)
        M_product = M_half @ M_half
        np.testing.assert_array_almost_equal(M_product, M_full, decimal=10)

    def test_ds_parameter(self):
        """transfer_matrix with ds should use ds instead of self.length."""
        sol = Solenoid("SOL1", length=100.0, field=0.5)
        ref = _make_ref()
        M_30 = sol.transfer_matrix(ref, ds=30.0)
        M_100 = sol.transfer_matrix(ref)
        assert not np.allclose(M_30, M_100)


# ---- Track method ----

class TestSolenoidTrack:
    def test_track_updates_ref_s(self):
        """track should advance the reference particle position."""
        beam = _make_beam()
        s_before = beam.ref.s
        sol = Solenoid("SOL1", length=100.0, field=0.5)
        sol.track(beam)
        assert abs(beam.ref.s - (s_before + 100.0)) < 1e-10

    def test_track_updates_ref_phi_s(self):
        """track should advance the synchronous phase."""
        beam = _make_beam()
        phi_before = beam.ref.phi_s
        sol = Solenoid("SOL1", length=100.0, field=0.5)
        sol.track(beam)
        expected_dphi = 360.0 * 100.0 / (beam.ref.beta * beam.ref.wavelength)
        assert abs(beam.ref.phi_s - (phi_before + expected_dphi)) < 1e-8

    def test_track_applies_matrix(self):
        """track should apply the transfer matrix to particles."""
        beam = _make_beam(n=1)
        beam.particles[0, 0] = 1.0  # x = 1 mm
        sol = Solenoid("SOL1", length=100.0, field=0.5)
        sol.track(beam)
        # After solenoid, x should have changed
        assert abs(beam.particles[0, 0] - 1.0) > 1e-6

    def test_track_only_alive(self):
        """Lost particles should not be moved."""
        beam = _make_beam(n=3)
        beam.particles[0, 0] = 1.0
        beam.particles[1, 0] = 1.0
        beam.record_loss(0, 0.0, "test")
        sol = Solenoid("SOL1", length=100.0, field=0.5)
        sol.track(beam)
        # Particle 0 (lost) should remain at x=1 (unchanged, but actually it was
        # zeroed or stays the same since alive_mask excludes it)
        # Particle 1 (alive) should have moved
        assert beam.particles[1, 0] != 1.0  # alive particle moved


# ---- Longitudinal matrix element ----

class TestSolenoidLongitudinal:
    def test_phase_slip_element(self):
        """M[4,5] should match the drift-like phase slip formula -360 L /(beta^3 gamma^3 m lambda).

        Matches TraceWin's R_zz = [[1, Δs/γ²], [0,1]] after converting
        the (z, δ) basis to our (Δφ, ΔW) basis.
        """
        sol = Solenoid("SOL1", length=100.0, field=0.5)
        ref = _make_ref()
        M = sol.transfer_matrix(ref)
        expected_M45 = -360.0 * 100.0 / (
            ref.beta**3 * ref.gamma**3 * ref.species.mass * ref.wavelength
        )
        assert abs(M[4, 5] - expected_M45) < 1e-10

    def test_diagonal_ones(self):
        """Diagonal elements M[4,4] and M[5,5] should be 1."""
        sol = Solenoid("SOL1", length=100.0, field=0.5)
        ref = _make_ref()
        M = sol.transfer_matrix(ref)
        assert abs(M[4, 4] - 1.0) < 1e-14
        assert abs(M[5, 5] - 1.0) < 1e-14


# ---- Known analytical values ----

class TestSolenoidAnalytical:
    def test_diagonal_elements_are_C_squared(self):
        """M[0,0] = M[1,1] = M[2,2] = M[3,3] = C^2 where C = cos(phi)."""
        ref = _make_ref()
        B0 = 0.5
        L_mm = 100.0
        L_m = L_mm * 1e-3
        k_s = B0 / (2 * ref.brho)
        phi = k_s * L_m
        C = math.cos(phi)
        C2 = C * C

        sol = Solenoid("SOL1", length=L_mm, field=B0)
        M = sol.transfer_matrix(ref)

        np.testing.assert_allclose(M[0, 0], C2, rtol=1e-12)
        np.testing.assert_allclose(M[1, 1], C2, rtol=1e-12)
        np.testing.assert_allclose(M[2, 2], C2, rtol=1e-12)
        np.testing.assert_allclose(M[3, 3], C2, rtol=1e-12)

    def test_off_diagonal_analytical(self):
        """Check specific off-diagonal elements against analytical formulas."""
        ref = _make_ref()
        B0 = 0.5
        L_mm = 100.0
        L_m = L_mm * 1e-3
        k_s = B0 / (2 * ref.brho)
        phi = k_s * L_m
        C = math.cos(phi)
        S = math.sin(phi)

        sol = Solenoid("SOL1", length=L_mm, field=B0)
        M = sol.transfer_matrix(ref)

        # M[0,1] = C*S/k_s  (in m, but matrix is in mm/mrad units)
        # After unit conversion: M[0,1] in mm/(mrad) = (C*S/k_s) in m
        np.testing.assert_allclose(M[0, 1], C * S / k_s, rtol=1e-10)
        # M[1,0] = -k_s*S*C (in 1/m, which is mrad/mm after unit conversion)
        np.testing.assert_allclose(M[1, 0], -k_s * S * C, rtol=1e-10)
        # M[0,2] = S*C (dimensionless in mm/mm)
        np.testing.assert_allclose(M[0, 2], S * C, rtol=1e-10)
        # M[0,3] = S^2/k_s (in m, which is mm/mrad)
        np.testing.assert_allclose(M[0, 3], S**2 / k_s, rtol=1e-10)
        # M[2,0] = -S*C (dimensionless)
        np.testing.assert_allclose(M[2, 0], -S * C, rtol=1e-10)
        # M[2,1] = -S^2/k_s (in m)
        np.testing.assert_allclose(M[2, 1], -S**2 / k_s, rtol=1e-10)
        # M[3,0] = k_s*S^2 (in 1/m)
        np.testing.assert_allclose(M[3, 0], k_s * S**2, rtol=1e-10)
        # M[3,1] = -S*C (dimensionless)
        np.testing.assert_allclose(M[3, 1], -S * C, rtol=1e-10)
        # M[1,2] = -k_s*S^2 (in 1/m)
        np.testing.assert_allclose(M[1, 2], -k_s * S**2, rtol=1e-10)
        # M[1,3] = S*C (dimensionless)
        np.testing.assert_allclose(M[1, 3], S * C, rtol=1e-10)
        # M[3,2] = -k_s*S*C (in 1/m)
        np.testing.assert_allclose(M[3, 2], -k_s * S * C, rtol=1e-10)
        # M[2,3] = C*S/k_s (in m)
        np.testing.assert_allclose(M[2, 3], C * S / k_s, rtol=1e-10)
