# tests/elements/test_steerer.py
import numpy as np
import pytest
from linac_gen.core.particle import PROTON, H_MINUS
from linac_gen.core.reference import ReferenceParticle
from linac_gen.core.beam import Beam
from linac_gen.elements.steerer import Steerer


def _make_beam(species=PROTON, n=5, w_kin=3.0):
    ref = ReferenceParticle(species=species, w_kin=w_kin, frequency=352.21)
    beam = Beam(ref=ref, n_particles=n, current=60.0)
    # zero initial conditions
    return beam


class TestSteererHorizontal:
    def test_horizontal_kick_changes_xp(self):
        """By_L > 0 should kick x' (xp = col 1)."""
        beam = _make_beam()
        s = Steerer("SH", by_l=0.001)  # 0.001 T.m
        s.apply_kick(beam)
        # xp should be nonzero
        assert np.all(beam.particles[:, 1] != 0.0)

    def test_horizontal_kick_no_yp_change(self):
        """By_L should not affect y'."""
        beam = _make_beam()
        s = Steerer("SH", by_l=0.001)
        s.apply_kick(beam)
        np.testing.assert_array_equal(beam.particles[:, 3], 0.0)

    def test_horizontal_kick_magnitude(self):
        """xp kick = charge_sign * By_L / brho * 1e3 mrad."""
        beam = _make_beam()
        brho = beam.ref.brho
        by_l = 0.002
        s = Steerer("SH", by_l=by_l)
        s.apply_kick(beam)
        expected = by_l / brho * 1e3  # mrad, charge_sign=+1 for proton
        np.testing.assert_allclose(beam.particles[:, 1], expected, rtol=1e-10)


class TestSteererVertical:
    def test_vertical_kick_changes_yp(self):
        """Bx_L > 0 should kick y' (yp = col 3)."""
        beam = _make_beam()
        s = Steerer("SV", bx_l=0.001)
        s.apply_kick(beam)
        assert np.all(beam.particles[:, 3] != 0.0)

    def test_vertical_kick_no_xp_change(self):
        """Bx_L should not affect x'."""
        beam = _make_beam()
        s = Steerer("SV", bx_l=0.001)
        s.apply_kick(beam)
        np.testing.assert_array_equal(beam.particles[:, 1], 0.0)

    def test_vertical_kick_magnitude(self):
        """yp kick = charge_sign * Bx_L / brho * 1e3 mrad."""
        beam = _make_beam()
        brho = beam.ref.brho
        bx_l = 0.003
        s = Steerer("SV", bx_l=bx_l)
        s.apply_kick(beam)
        expected = bx_l / brho * 1e3
        np.testing.assert_allclose(beam.particles[:, 3], expected, rtol=1e-10)


class TestSteererChargeSign:
    def test_h_minus_reverses_xp_kick(self):
        """H- has charge=-1, so kick sign is reversed."""
        beam_p = _make_beam(species=PROTON)
        beam_h = _make_beam(species=H_MINUS)
        by_l = 0.001
        Steerer("S", by_l=by_l).apply_kick(beam_p)
        Steerer("S", by_l=by_l).apply_kick(beam_h)
        # Proton: positive kick; H-: negative kick
        assert beam_p.particles[0, 1] > 0
        assert beam_h.particles[0, 1] < 0

    def test_h_minus_reverses_yp_kick(self):
        beam_p = _make_beam(species=PROTON)
        beam_h = _make_beam(species=H_MINUS)
        bx_l = 0.001
        Steerer("S", bx_l=bx_l).apply_kick(beam_p)
        Steerer("S", bx_l=bx_l).apply_kick(beam_h)
        assert beam_p.particles[0, 3] > 0
        assert beam_h.particles[0, 3] < 0


class TestSteererKickMatrix:
    def test_kick_matrix_is_identity(self):
        """Steerer is a constant kick — no position-dependent focusing."""
        s = Steerer("S", bx_l=0.001, by_l=0.001)
        ref = ReferenceParticle(species=PROTON, w_kin=3.0, frequency=352.21)
        M = s.kick_matrix(ref)
        np.testing.assert_array_equal(M, np.eye(6))

    def test_kick_matrix_shape(self):
        s = Steerer("S", by_l=0.001)
        ref = ReferenceParticle(species=PROTON, w_kin=3.0, frequency=352.21)
        M = s.kick_matrix(ref)
        assert M.shape == (6, 6)


class TestSteererGeneral:
    def test_zero_length(self):
        s = Steerer("S")
        assert s.length == 0.0

    def test_zero_aperture_by_default(self):
        s = Steerer("S")
        assert s.aperture == 0.0

    def test_default_zero_kick(self):
        beam = _make_beam()
        s = Steerer("S")
        s.apply_kick(beam)
        np.testing.assert_array_equal(beam.particles, 0.0)

    def test_is_thin_kick_element(self):
        from linac_gen.elements.base import ThinKickElement
        s = Steerer("S")
        assert isinstance(s, ThinKickElement)

    def test_only_alive_particles_kicked(self):
        beam = _make_beam(n=4)
        # Mark first particle as lost
        beam.record_loss(0, 0.0, "test")
        s = Steerer("S", by_l=0.001)
        s.apply_kick(beam)
        # Particle 0 should remain at 0 (was already zeroed, and we set it lost)
        # Particles 1-3 should be kicked
        assert beam.particles[0, 1] == 0.0  # not kicked (lost)
        assert beam.particles[1, 1] != 0.0  # kicked


class TestSteererElectric:
    """THIN_STEERING Elec=1: operands are ∫E·dl in volts, kick is
    SAME-PLANE via the electric rigidity Eρ = βc·Bρ (TW manual)."""

    def test_electric_kick_same_plane_and_magnitude(self):
        beam = _make_beam()
        el_x = 500.0                        # volts
        s = Steerer("SE", bx_l=el_x, elec=True)
        s.apply_kick(beam)
        erho = beam.ref.beta * 299_792_458.0 * beam.ref.brho
        expected = el_x / erho * 1e3        # mrad, +q proton
        np.testing.assert_allclose(beam.particles[:, 1], expected,
                                   rtol=1e-12)
        # Same-plane: ELx must NOT touch y' (magnetic would).
        np.testing.assert_array_equal(beam.particles[:, 3], 0.0)

    def test_electric_vs_magnetic_energy_scaling(self):
        """Electric kick scales as 1/(β²γ) vs magnetic 1/(βγ): the
        electric/magnetic kick ratio at two energies differs by β2/β1
        — the wrong-quantity bug this guards against."""
        k = {}
        for w in (0.05, 3.0):
            beam = _make_beam(w_kin=w, n=1)
            se = Steerer("SE", bx_l=1000.0, elec=True)
            sm = Steerer("SM", by_l=0.001)
            se.apply_kick(beam)
            xp_e = float(beam.particles[0, 1])
            beam.particles[:, :] = 0.0
            sm.apply_kick(beam)
            k[w] = (xp_e, float(beam.particles[0, 1]),
                    beam.ref.beta)
        r_low = k[0.05][0] / k[0.05][1]
        r_high = k[3.0][0] / k[3.0][1]
        assert r_low / r_high == pytest.approx(
            k[3.0][2] / k[0.05][2], rel=1e-9)

    def test_electric_charge_sign(self):
        bp = _make_beam(PROTON, n=1)
        bm = _make_beam(H_MINUS, n=1, w_kin=3.0)
        s = Steerer("SE", bx_l=500.0, elec=True)
        s.apply_kick(bp)
        s.apply_kick(bm)
        assert bp.particles[0, 1] > 0 > bm.particles[0, 1]

    def test_electric_inverse_kick_exact(self):
        beam = _make_beam()
        s = Steerer("SE", bx_l=500.0, by_l=-200.0, elec=True)
        before = beam.particles.copy()
        s.apply_kick(beam)
        s.inverse_kick(beam, beam.ref)
        np.testing.assert_array_equal(beam.particles, before)

    def test_magnetic_default_unchanged(self):
        """elec defaults False and the magnetic numbers are bit-identical
        to the pre-flag implementation."""
        beam = _make_beam()
        s = Steerer("SM", bx_l=0.003, by_l=0.002)
        assert s.elec is False
        s.apply_kick(beam)
        brho = beam.ref.brho
        np.testing.assert_allclose(beam.particles[:, 1],
                                   0.002 / brho * 1e3, rtol=0, atol=0)
        np.testing.assert_allclose(beam.particles[:, 3],
                                   0.003 / brho * 1e3, rtol=0, atol=0)
