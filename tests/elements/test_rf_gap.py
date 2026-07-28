# tests/elements/test_rf_gap.py
"""Tests for RF gap element (ThinKickElement)."""
import numpy as np
import math
import pytest
from linac_gen.core.particle import PROTON, H_MINUS
from linac_gen.core.reference import ReferenceParticle
from linac_gen.core.beam import Beam
from linac_gen.elements.rf_gap import RFGap


def _make_beam(species=PROTON, n=5, w_kin=3.0, freq=352.21):
    ref = ReferenceParticle(species=species, w_kin=w_kin, frequency=freq)
    beam = Beam(ref=ref, n_particles=n, current=60.0)
    return beam


# ---- Energy gain tests ----

class TestRFGapEnergyGain:
    def test_energy_gain_on_crest(self):
        """V0=1 MV, T=1, phi_s=0 deg -> dW = 1 MeV for proton."""
        beam = _make_beam(w_kin=3.0)
        gap = RFGap("G1", voltage=1.0, phase=0.0, frequency=352.21, ttf=1.0)
        w_before = beam.ref.w_kin
        gap.apply_kick(beam)
        assert abs(beam.ref.w_kin - (w_before + 1.0)) < 1e-12

    def test_energy_gain_negative_phase(self):
        """phi_s=-30 deg -> dW = V0*T*cos(-30) = 0.866 MeV."""
        beam = _make_beam(w_kin=3.0)
        gap = RFGap("G1", voltage=1.0, phase=-30.0, frequency=352.21, ttf=1.0)
        w_before = beam.ref.w_kin
        gap.apply_kick(beam)
        expected = math.cos(math.radians(-30.0))  # ~0.866
        assert abs(beam.ref.w_kin - (w_before + expected)) < 1e-10

    def test_energy_gain_with_ttf(self):
        """Transit time factor scales the energy gain."""
        beam = _make_beam(w_kin=3.0)
        gap = RFGap("G1", voltage=2.0, phase=0.0, frequency=352.21, ttf=0.5)
        w_before = beam.ref.w_kin
        gap.apply_kick(beam)
        # dW = 1*2.0*0.5*cos(0) = 1.0 MeV
        assert abs(beam.ref.w_kin - (w_before + 1.0)) < 1e-12

    def test_h_minus_energy_gain_negative(self):
        """H- has charge=-1 -> energy gain is negative (decelerated on crest)."""
        beam = _make_beam(species=H_MINUS, w_kin=10.0)
        gap = RFGap("G1", voltage=1.0, phase=0.0, frequency=352.21, ttf=1.0)
        w_before = beam.ref.w_kin
        gap.apply_kick(beam)
        # dW = -1 * 1.0 * 1.0 * cos(0) = -1.0 MeV
        assert abs(beam.ref.w_kin - (w_before - 1.0)) < 1e-12

    def test_h_minus_accelerated_at_180deg(self):
        """H- at phi_s=180 deg: dW = -1 * V0 * cos(180) = +V0."""
        beam = _make_beam(species=H_MINUS, w_kin=10.0)
        gap = RFGap("G1", voltage=1.0, phase=180.0, frequency=352.21, ttf=1.0)
        w_before = beam.ref.w_kin
        gap.apply_kick(beam)
        assert abs(beam.ref.w_kin - (w_before + 1.0)) < 1e-12


# ---- Particle energy deviation kick ----

class TestRFGapParticleKick:
    def test_on_axis_particle_no_energy_kick(self):
        """A particle with dphi=0 should get dW_i=0 (same as synchronous)."""
        beam = _make_beam(n=1, w_kin=3.0)
        beam.particles[0, :] = 0.0  # all zero deviations
        gap = RFGap("G1", voltage=1.0, phase=-30.0, frequency=352.21, ttf=1.0)
        gap.apply_kick(beam)
        # dW_i = cos(phi_s + 0) - cos(phi_s) = 0
        assert abs(beam.particles[0, 5]) < 1e-14

    def test_off_phase_particle_gets_energy_kick(self):
        """A particle with dphi != 0 should get a non-zero dW_i."""
        beam = _make_beam(n=1, w_kin=3.0)
        beam.particles[0, 4] = 5.0  # dphi = 5 deg
        gap = RFGap("G1", voltage=1.0, phase=-30.0, frequency=352.21, ttf=1.0)
        gap.apply_kick(beam)
        phi_s_rad = math.radians(-30.0)
        dphi_rad = math.radians(5.0)
        expected_dW = 1.0 * 1.0 * (math.cos(phi_s_rad + dphi_rad) - math.cos(phi_s_rad))
        assert abs(beam.particles[0, 5] - expected_dW) < 1e-12


# ---- Adiabatic damping ----

class TestRFGapAdiabaticDamping:
    def test_xp_decreases_after_acceleration(self):
        """x' should decrease (damp) after acceleration."""
        beam = _make_beam(n=1, w_kin=3.0)
        beam.particles[0, 1] = 10.0  # xp = 10 mrad
        xp_before = beam.particles[0, 1]
        bg_before = beam.ref.bg
        gap = RFGap("G1", voltage=1.0, phase=0.0, frequency=352.21, ttf=1.0)
        gap.apply_kick(beam)
        bg_after = beam.ref.bg
        # After damping: xp_new = xp_old * bg_old/bg_new (ignoring RF defocusing since x=0)
        expected_xp = xp_before * bg_before / bg_after
        # The RF defocusing term is k_rf * x, but x=0 so no contribution
        assert abs(beam.particles[0, 1] - expected_xp) < 1e-10

    def test_yp_decreases_after_acceleration(self):
        """y' should decrease (damp) after acceleration."""
        beam = _make_beam(n=1, w_kin=3.0)
        beam.particles[0, 3] = 10.0  # yp = 10 mrad
        yp_before = beam.particles[0, 3]
        bg_before = beam.ref.bg
        gap = RFGap("G1", voltage=1.0, phase=0.0, frequency=352.21, ttf=1.0)
        gap.apply_kick(beam)
        bg_after = beam.ref.bg
        expected_yp = yp_before * bg_before / bg_after
        assert abs(beam.particles[0, 3] - expected_yp) < 1e-10

    def test_damping_ratio_consistent(self):
        """Both planes should damp by the same ratio."""
        beam = _make_beam(n=1, w_kin=3.0)
        beam.particles[0, 1] = 5.0  # xp
        beam.particles[0, 3] = 8.0  # yp
        bg_before = beam.ref.bg
        gap = RFGap("G1", voltage=1.0, phase=0.0, frequency=352.21, ttf=1.0)
        gap.apply_kick(beam)
        bg_after = beam.ref.bg
        damping = bg_before / bg_after
        # x=y=0, so no RF defocusing contribution
        np.testing.assert_allclose(beam.particles[0, 1], 5.0 * damping, rtol=1e-10)
        np.testing.assert_allclose(beam.particles[0, 3], 8.0 * damping, rtol=1e-10)


# ---- RF defocusing ----

class TestRFGapRFDefocusing:
    def test_positive_x_gets_defocusing_kick(self):
        """With phi_s < 0 (stable linac), particle at positive x gets positive xp kick."""
        beam = _make_beam(n=1, w_kin=3.0)
        beam.particles[0, 0] = 1.0  # x = 1 mm
        beam.particles[0, 1] = 0.0  # xp = 0
        gap = RFGap("G1", voltage=1.0, phase=-30.0, frequency=352.21, ttf=1.0)
        gap.apply_kick(beam)
        # k_rf = -pi * q * V0 * T * sin(phi_s) / (m * bg^3 * lambda)
        # sin(-30 deg) < 0, so -sin < 0 gives k_rf > 0 (defocusing)
        # xp kick = k_rf * x * 1e3 > 0 (after damping)
        assert beam.particles[0, 1] > 0  # defocusing kick

    def test_rf_defocusing_symmetric_xy(self):
        """RF defocusing should be the same in x and y."""
        beam = _make_beam(n=1, w_kin=3.0)
        beam.particles[0, 0] = 1.0  # x = 1 mm
        beam.particles[0, 2] = 1.0  # y = 1 mm
        gap = RFGap("G1", voltage=1.0, phase=-30.0, frequency=352.21, ttf=1.0)
        gap.apply_kick(beam)
        # Both xp and yp kicks should be identical (damping + defocusing)
        np.testing.assert_allclose(beam.particles[0, 1], beam.particles[0, 3], rtol=1e-10)

    def test_rf_defocusing_proportional_to_position(self):
        """RF defocusing kick is linear in position."""
        beam1 = _make_beam(n=1, w_kin=3.0)
        beam2 = _make_beam(n=1, w_kin=3.0)
        beam1.particles[0, 0] = 1.0  # x = 1 mm
        beam2.particles[0, 0] = 2.0  # x = 2 mm
        gap = RFGap("G1", voltage=1.0, phase=-30.0, frequency=352.21, ttf=1.0)
        gap.apply_kick(beam1)
        gap2 = RFGap("G2", voltage=1.0, phase=-30.0, frequency=352.21, ttf=1.0)
        gap2.apply_kick(beam2)
        # xp(2mm) should be 2*xp(1mm)
        np.testing.assert_allclose(beam2.particles[0, 1], 2.0 * beam1.particles[0, 1], rtol=1e-10)

    def test_rf_defocusing_magnitude(self):
        """Verify the numerical value of the RF defocusing kick."""
        beam = _make_beam(n=1, w_kin=3.0)
        x0 = 1.0  # mm
        beam.particles[0, 0] = x0
        phi_s_deg = -30.0
        V0 = 1.0
        T = 1.0
        gap = RFGap("G1", voltage=V0, phase=phi_s_deg, frequency=352.21, ttf=T)

        bg_old = beam.ref.bg
        mass = beam.ref.species.mass
        charge = beam.ref.species.charge  # +1

        gap.apply_kick(beam)

        bg_new = beam.ref.bg
        wl = beam.ref.wavelength  # mm
        phi_s_rad = math.radians(phi_s_deg)

        # Expected: damping first, then defocusing
        # xp_after_damping = 0 * bg_old/bg_new = 0
        # k_rf = -pi * q * V0 * T * sin(phi_s) / (m * bg_new^3 * wl)
        k_rf = -math.pi * charge * V0 * T * math.sin(phi_s_rad) / (mass * bg_new**3 * wl)
        expected_xp = k_rf * x0 * 1e3  # mrad
        np.testing.assert_allclose(beam.particles[0, 1], expected_xp, rtol=1e-10)


# ---- advance_ref ----

class TestRFGapAdvanceRef:
    def test_advance_ref_updates_energy(self):
        """advance_ref should update ref.w_kin by the synchronous energy gain."""
        ref = ReferenceParticle(species=PROTON, w_kin=3.0, frequency=352.21)
        gap = RFGap("G1", voltage=1.0, phase=-30.0, frequency=352.21, ttf=1.0)
        w_before = ref.w_kin
        gap.advance_ref(ref)
        expected_dW = 1.0 * math.cos(math.radians(-30.0))
        assert abs(ref.w_kin - (w_before + expected_dW)) < 1e-12

    def test_advance_ref_updates_derived(self):
        """After advance_ref, beta/gamma/bg should be recomputed."""
        ref = ReferenceParticle(species=PROTON, w_kin=3.0, frequency=352.21)
        bg_before = ref.bg
        gap = RFGap("G1", voltage=1.0, phase=0.0, frequency=352.21, ttf=1.0)
        gap.advance_ref(ref)
        assert ref.bg > bg_before  # more energy = more momentum
        assert ref.gamma > 1.0 + 3.0 / ref.species.mass  # increased gamma

    def test_advance_ref_h_minus(self):
        """H- advance_ref gives negative energy change on crest."""
        ref = ReferenceParticle(species=H_MINUS, w_kin=10.0, frequency=352.21)
        w_before = ref.w_kin
        gap = RFGap("G1", voltage=1.0, phase=0.0, frequency=352.21, ttf=1.0)
        gap.advance_ref(ref)
        assert abs(ref.w_kin - (w_before - 1.0)) < 1e-12


# ---- kick_matrix ----

class TestRFGapKickMatrix:
    def test_kick_matrix_shape(self):
        gap = RFGap("G1", voltage=1.0, phase=-30.0, frequency=352.21, ttf=1.0)
        ref = ReferenceParticle(species=PROTON, w_kin=3.0, frequency=352.21)
        M = gap.kick_matrix(ref)
        assert M.shape == (6, 6)

    def test_kick_matrix_longitudinal_focusing(self):
        """M[5,4] should be the linearized longitudinal focusing."""
        gap = RFGap("G1", voltage=1.0, phase=-30.0, frequency=352.21, ttf=1.0)
        ref = ReferenceParticle(species=PROTON, w_kin=3.0, frequency=352.21)
        M = gap.kick_matrix(ref)
        phi_s_rad = math.radians(-30.0)
        expected_M54 = -1.0 * 1.0 * math.sin(phi_s_rad) * math.pi / 180.0
        assert abs(M[5, 4] - expected_M54) < 1e-12

    def test_kick_matrix_transverse_damping(self):
        """Diagonal x'/y' entries should reflect adiabatic damping."""
        gap = RFGap("G1", voltage=1.0, phase=0.0, frequency=352.21, ttf=1.0)
        ref = ReferenceParticle(species=PROTON, w_kin=3.0, frequency=352.21)
        M = gap.kick_matrix(ref)
        # M[1,1] and M[3,3] should be bg_old/bg_new < 1
        assert M[1, 1] < 1.0
        assert M[3, 3] < 1.0
        assert M[1, 1] > 0.0  # but still positive

    def test_kick_matrix_rf_defocusing(self):
        """M[1,0] and M[3,2] should have the RF defocusing strength."""
        gap = RFGap("G1", voltage=1.0, phase=-30.0, frequency=352.21, ttf=1.0)
        ref = ReferenceParticle(species=PROTON, w_kin=3.0, frequency=352.21)
        M = gap.kick_matrix(ref)
        # With phi_s < 0, k_rf > 0 (defocusing), so M[1,0] > 0
        assert M[1, 0] > 0
        assert M[3, 2] > 0

    def test_kick_matrix_does_not_modify_ref(self):
        """kick_matrix should not change the reference particle energy."""
        ref = ReferenceParticle(species=PROTON, w_kin=3.0, frequency=352.21)
        w_before = ref.w_kin
        gap = RFGap("G1", voltage=1.0, phase=-30.0, frequency=352.21, ttf=1.0)
        _ = gap.kick_matrix(ref)
        assert abs(ref.w_kin - w_before) < 1e-14


# ---- General properties ----

class TestRFGapGeneral:
    def test_is_thin_kick_element(self):
        from linac_gen.elements.base import ThinKickElement
        gap = RFGap("G1", voltage=1.0, phase=0.0, frequency=352.21)
        assert isinstance(gap, ThinKickElement)

    def test_zero_length(self):
        gap = RFGap("G1", voltage=1.0, phase=0.0, frequency=352.21)
        assert gap.length == 0.0

    def test_default_ttf_is_one(self):
        gap = RFGap("G1", voltage=1.0, phase=0.0, frequency=352.21)
        assert gap.ttf == 1.0

    def test_default_aperture_is_zero(self):
        gap = RFGap("G1", voltage=1.0, phase=0.0, frequency=352.21)
        assert gap.aperture == 0.0

    def test_only_alive_particles_kicked(self):
        """Lost particles should not be affected."""
        beam = _make_beam(n=4, w_kin=3.0)
        beam.particles[:, 0] = 1.0  # x = 1 mm for all
        beam.record_loss(0, 0.0, "test")
        gap = RFGap("G1", voltage=1.0, phase=-30.0, frequency=352.21, ttf=1.0)
        gap.apply_kick(beam)
        # Particle 0 (lost) should have xp unchanged (still 0)
        assert beam.particles[0, 1] == 0.0
        # Alive particles should have xp changed (RF defocusing)
        assert beam.particles[1, 1] != 0.0

    def test_apply_kick_and_advance_ref_give_same_energy(self):
        """apply_kick and advance_ref should give the same synchronous energy gain."""
        ref1 = ReferenceParticle(species=PROTON, w_kin=3.0, frequency=352.21)
        ref2 = ReferenceParticle(species=PROTON, w_kin=3.0, frequency=352.21)
        beam = Beam(ref=ref1, n_particles=1, current=60.0)
        gap = RFGap("G1", voltage=1.5, phase=-25.0, frequency=352.21, ttf=0.8)
        gap.apply_kick(beam)
        gap2 = RFGap("G2", voltage=1.5, phase=-25.0, frequency=352.21, ttf=0.8)
        gap2.advance_ref(ref2)
        assert abs(ref1.w_kin - ref2.w_kin) < 1e-12


def test_kick_matrix_matches_apply_kick():
    """kick_matrix linearization must match apply_kick for small deviations."""
    from linac_gen.core.beam import Beam
    gap = RFGap(name="GAP", voltage=1.0, phase=-30.0, frequency=352.21, ttf=1.0)
    ref = ReferenceParticle(species=PROTON, w_kin=10.0, frequency=352.21)
    
    # Small x offset, compute kick via apply_kick
    beam = Beam(ref=ref.copy(), n_particles=1, current=0.0)
    beam.particles[0, 0] = 0.01  # x = 0.01 mm (small)
    xp_before = beam.particles[0, 1]
    gap.apply_kick(beam)
    xp_after_particle = beam.particles[0, 1]
    
    # Same via kick_matrix
    M = gap.kick_matrix(ref)
    state = np.array([0.01, 0.0, 0.0, 0.0, 0.0, 0.0])
    xp_after_matrix = (M @ state)[1]
    
    # Should agree for small x
    assert abs(xp_after_particle - xp_after_matrix) < 0.001 * abs(xp_after_particle + 1e-30)
