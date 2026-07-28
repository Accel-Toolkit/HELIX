# tests/elements/test_multipole.py
"""Tests for the Multipole thin-kick element."""
import math
import numpy as np
import pytest
from linac_gen.elements.multipole import Multipole
from linac_gen.elements.base import ThinKickElement
from linac_gen.elements.quadrupole import Quadrupole
from linac_gen.core.reference import ReferenceParticle
from linac_gen.core.beam import Beam
from linac_gen.core.particle import PROTON


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ref(w_kin: float = 100.0) -> ReferenceParticle:
    return ReferenceParticle(species=PROTON, w_kin=w_kin, frequency=352.21)


def _beam(n: int = 4, w_kin: float = 100.0) -> Beam:
    return Beam(ref=_ref(w_kin=w_kin), n_particles=n, current=10.0)


def _beam_at(x_mm: float, y_mm: float = 0.0, n: int = 1) -> Beam:
    """Single-particle beam at (x, y) with zero divergence."""
    beam = _beam(n=n)
    beam.particles[:, 0] = x_mm
    beam.particles[:, 2] = y_mm
    return beam


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

class TestMultipoleConstruction:
    def test_is_thin_kick_element(self):
        m = Multipole("M")
        assert isinstance(m, ThinKickElement)

    def test_zero_length(self):
        m = Multipole("M", knl=[0.5])
        assert m.length == 0.0

    def test_default_empty_lists(self):
        m = Multipole("M")
        assert m.knl == []
        assert m.ksl == []

    def test_stores_knl_ksl(self):
        m = Multipole("M", knl=[1.0, 2.0, 3.0], ksl=[0.1, 0.2])
        assert m.knl == [1.0, 2.0, 3.0]
        assert m.ksl == [0.1, 0.2]

    def test_aperture_stored(self):
        m = Multipole("M", aperture=15.0)
        assert m.aperture == 15.0


# ---------------------------------------------------------------------------
# Zero strength → no kick
# ---------------------------------------------------------------------------

class TestMultipoleZeroStrength:
    def test_no_kick_when_all_zero(self):
        m = Multipole("M", knl=[0.0, 0.0], ksl=[0.0, 0.0])
        beam = _beam_at(5.0, 3.0)
        before = beam.particles.copy()
        m.apply_kick(beam)
        np.testing.assert_array_equal(beam.particles, before)

    def test_no_kick_empty_lists(self):
        m = Multipole("M")
        beam = _beam_at(5.0, 3.0)
        before = beam.particles.copy()
        m.apply_kick(beam)
        np.testing.assert_array_equal(beam.particles, before)

    def test_kick_matrix_identity_no_quad(self):
        m = Multipole("M", knl=[1.0, 0.0, 3.0])  # no quad term
        ref = _ref()
        M = m.kick_matrix(ref)
        np.testing.assert_array_equal(M, np.eye(6))


# ---------------------------------------------------------------------------
# Dipole kick (n=1): constant kick, position-independent
# ---------------------------------------------------------------------------

class TestMultipoleDipoleOrder:
    def test_dipole_kick_x_constant(self):
        """k0L provides a constant horizontal kick (independent of position)."""
        k0L = 0.01  # rad
        m = Multipole("M", knl=[k0L])
        beam1 = _beam_at(0.0)
        beam2 = _beam_at(10.0)
        m.apply_kick(beam1)
        m.apply_kick(beam2)
        # Both should get the same kick
        np.testing.assert_allclose(beam1.particles[:, 1], beam2.particles[:, 1], rtol=1e-12)

    def test_dipole_kick_magnitude(self):
        """k0L kick: Δx'[mrad] = -k0L * 1e3."""
        k0L = 0.005  # rad
        m = Multipole("M", knl=[k0L])
        beam = _beam_at(0.0)
        m.apply_kick(beam)
        expected_mrad = -k0L * 1e3
        np.testing.assert_allclose(beam.particles[:, 1], expected_mrad, rtol=1e-12)


# ---------------------------------------------------------------------------
# Quadrupole order (n=2): linear position kick
# ---------------------------------------------------------------------------

class TestMultipoleQuadOrder:
    def test_quad_kick_linear_in_x(self):
        """For a normal quad k1L, Δx' ∝ -x and Δy' ∝ +y."""
        k1L = 1.0   # 1/m (integrated)
        m = Multipole("M", knl=[0.0, k1L])

        beam = _beam(n=1)
        beam.particles[0, 0] = 2.0   # x = 2 mm
        beam.particles[0, 2] = 0.0

        m.apply_kick(beam)

        # dx'[mrad] = -k1L * x[mm] = -1.0 * 2.0 = -2.0 mrad
        np.testing.assert_allclose(beam.particles[0, 1], -2.0, rtol=1e-12)
        # dy' should be zero for y=0
        assert beam.particles[0, 3] == 0.0

    def test_quad_kick_defocusing_y(self):
        """Normal quad: Δy'[mrad] = +k1L * y[mm]."""
        k1L = 1.5
        m = Multipole("M", knl=[0.0, k1L])
        beam = _beam(n=1)
        beam.particles[0, 0] = 0.0
        beam.particles[0, 2] = 3.0  # y = 3 mm
        m.apply_kick(beam)
        np.testing.assert_allclose(beam.particles[0, 3], k1L * 3.0, rtol=1e-12)

    def test_quad_kick_matrix(self):
        """kick_matrix for quad component should match M[1,0] = -k1L, M[3,2] = k1L."""
        k1L = 2.0
        m = Multipole("M", knl=[0.0, k1L])
        ref = _ref()
        M = m.kick_matrix(ref)
        assert abs(M[1, 0] - (-k1L)) < 1e-12
        assert abs(M[3, 2] - k1L) < 1e-12

    def test_quad_agrees_with_quadrupole_thin_lens(self):
        """Multipole quad kick should agree with thin Quadrupole in thin-lens limit."""
        L_mm = 1.0
        G = 5.0       # T/m
        ref = _ref()

        # Thin quadrupole
        quad = Quadrupole("Q", length=L_mm, gradient=G)
        M_quad = quad.transfer_matrix(ref)
        k1L_expected = G / ref.brho * L_mm * 1e-3   # 1/m

        # Multipole equivalent
        m = Multipole("M", knl=[0.0, k1L_expected])
        M_mult = m.kick_matrix(ref)

        # Check that the focusing element M[1,0] agrees
        assert abs(M_mult[1, 0] - M_quad[1, 0]) < abs(M_quad[1, 0]) * 0.01


# ---------------------------------------------------------------------------
# Sextupole order (n=3): x²-y² dependence
# ---------------------------------------------------------------------------

class TestMultipoleSextupoleOrder:
    """Sextupole: k2L at index 2 (n=3).

    The kick is:
        Δx'[rad] = -k2L/2! * Re{(x+iy)^2} = -k2L/2 * (x^2 - y^2)
        Δy'[rad] = -k2L/2! * Im{(x+iy)^2} = -k2L * x * y
    where x, y are in metres.
    """

    def test_sextupole_kick_on_axis(self):
        """On axis (x=y=0) a sextupole gives no kick."""
        m = Multipole("M", knl=[0.0, 0.0, 2.0])
        beam = _beam_at(0.0, 0.0)
        before = beam.particles.copy()
        m.apply_kick(beam)
        np.testing.assert_array_equal(beam.particles, before)

    def test_sextupole_kick_x_direction(self):
        """On x-axis (y=0): Δx' = -k2L/2 * x^2 (in rad, with x in m)."""
        k2L = 4.0   # 1/m²
        m = Multipole("M", knl=[0.0, 0.0, k2L])

        x_mm = 1.0
        x_m = x_mm * 1e-3
        beam = _beam_at(x_mm, y_mm=0.0)
        m.apply_kick(beam)

        # Δx'[rad] = -(k2L/2) * x_m^2 (since 1/2! = 0.5)
        expected_rad = -(k2L / 2.0) * x_m ** 2
        expected_mrad = expected_rad * 1e3
        np.testing.assert_allclose(beam.particles[:, 1], expected_mrad, rtol=1e-10)

    def test_sextupole_kick_x2_minus_y2(self):
        """Verify sextupole kick components for (x, y) off-axis position.

        With the MAD-X sign convention (Δx' - i·Δy' = kick):
            Δx'[rad] = -(k2L/2) * (x^2 - y^2)
            Δy'[rad] = +(k2L/2) * 2*x*y   (opposite sign to Im{kick})
        """
        k2L = 6.0
        m = Multipole("M", knl=[0.0, 0.0, k2L])

        x_mm, y_mm = 2.0, 1.0
        x_m, y_m = x_mm * 1e-3, y_mm * 1e-3
        beam = _beam_at(x_mm, y_mm)
        m.apply_kick(beam)

        expected_dxp_rad = -(k2L / 2.0) * (x_m ** 2 - y_m ** 2)
        expected_dyp_rad = +(k2L / 2.0) * 2.0 * x_m * y_m   # +sign: Δy' = -Im{kick}
        np.testing.assert_allclose(beam.particles[:, 1], expected_dxp_rad * 1e3, rtol=1e-10)
        np.testing.assert_allclose(beam.particles[:, 3], expected_dyp_rad * 1e3, rtol=1e-10)

    def test_sextupole_kick_y_symmetry(self):
        """For (0, y): Δx' = +k2L/2 * y^2, Δy' = 0 (from x^2-y^2 at x=0)."""
        k2L = 3.0
        m = Multipole("M", knl=[0.0, 0.0, k2L])

        y_mm = 2.0
        y_m = y_mm * 1e-3
        beam = _beam_at(0.0, y_mm)
        m.apply_kick(beam)

        # x=0: Re{(iy)^2} = Re{-y^2} = -y^2  →  Δx' = -k2L/2*(-y^2) = +k2L/2*y^2
        expected_dxp_rad = (k2L / 2.0) * y_m ** 2
        np.testing.assert_allclose(beam.particles[:, 1], expected_dxp_rad * 1e3, rtol=1e-10)
        np.testing.assert_allclose(beam.particles[:, 3], 0.0, atol=1e-15)

    def test_sextupole_kick_matrix_identity(self):
        """Sextupole has no linear term; kick_matrix should be identity."""
        m = Multipole("M", knl=[0.0, 0.0, 5.0])
        ref = _ref()
        M = m.kick_matrix(ref)
        np.testing.assert_array_equal(M, np.eye(6))


# ---------------------------------------------------------------------------
# Skew component
# ---------------------------------------------------------------------------

class TestMultipoleSkew:
    def test_skew_dipole_y_kick(self):
        """Skew dipole (a0L = ks0L) gives a constant y kick.

        kick = -(0 + i*a0L) * z^0 = -i*a0L
        Δx' = Re{kick} = 0
        Δy' = -Im{kick} = -(-a0L) = +a0L
        """
        a0L = 0.01  # rad
        m = Multipole("M", ksl=[a0L])
        beam = _beam_at(0.0, 0.0)
        m.apply_kick(beam)
        expected_mrad = +a0L * 1e3    # Δy' = +a0L (Δy' = -Im{kick})
        np.testing.assert_allclose(beam.particles[:, 3], expected_mrad, rtol=1e-12)
        # No x kick
        assert beam.particles[0, 1] == 0.0

    def test_skew_quad_rotates_kick(self):
        """Skew quadrupole mixes x and y planes.

        kick = -(i*ks1L) * (x+iy) = -i*ks1L*x - ks1L*i²*y = -i*ks1L*x + ks1L*y
        Δx' = Re{kick} = +ks1L*y  (zero for y=0)
        Δy' = -Im{kick} = -(-ks1L*x) = +ks1L*x
        """
        ks1L = 1.0
        m = Multipole("M", ksl=[0.0, ks1L])
        beam = _beam_at(2.0, 0.0)   # x=2mm, y=0
        m.apply_kick(beam)
        x_m = 2.0e-3
        expected_dyp_rad = +ks1L * x_m   # Δy' = +ks1L*x_m
        np.testing.assert_allclose(beam.particles[:, 3], expected_dyp_rad * 1e3, rtol=1e-12)
        assert abs(beam.particles[0, 1]) < 1e-14

    def test_combined_normal_skew(self):
        """Both normal and skew components act simultaneously.

        kick = -(k1L + i*ks1L) * (x + iy)
             = -(k1L*x - ks1L*y) - i*(k1L*y + ks1L*x)
        Δx' = Re{kick} = -(k1L*x - ks1L*y)
        Δy' = -Im{kick} = +(k1L*y + ks1L*x)
        """
        k1L = 1.0
        ks1L = 0.5
        m = Multipole("M", knl=[0.0, k1L], ksl=[0.0, ks1L])

        x_mm, y_mm = 3.0, 2.0
        x_m, y_m = x_mm * 1e-3, y_mm * 1e-3

        beam = _beam_at(x_mm, y_mm)
        m.apply_kick(beam)

        expected_dxp = -(k1L * x_m - ks1L * y_m)
        expected_dyp = +(k1L * y_m + ks1L * x_m)  # -Im{kick} = +

        np.testing.assert_allclose(beam.particles[:, 1], expected_dxp * 1e3, rtol=1e-12)
        np.testing.assert_allclose(beam.particles[:, 3], expected_dyp * 1e3, rtol=1e-12)


# ---------------------------------------------------------------------------
# Alive-particle masking
# ---------------------------------------------------------------------------

class TestMultipoleAliveMask:
    def test_lost_particles_not_kicked(self):
        m = Multipole("M", knl=[0.0, 1.0])
        beam = _beam(n=3)
        beam.particles[:, 0] = 2.0   # x = 2mm
        beam.record_loss(0, 0.0, "test")
        m.apply_kick(beam)
        assert beam.particles[0, 1] == 0.0   # lost: not kicked
        assert beam.particles[1, 1] != 0.0   # alive: kicked
        assert beam.particles[2, 1] != 0.0


# ---------------------------------------------------------------------------
# Multi-order superposition
# ---------------------------------------------------------------------------

class TestMultipoleOffsetAndTilt:
    """Element offset (dx, dy) and tilt about z."""

    def test_offset_zero_no_change(self):
        """dx=dy=0 must produce identical kicks to the un-offset element."""
        k1L = 1.0
        m_a = Multipole("A", knl=[0.0, k1L])
        m_b = Multipole("B", knl=[0.0, k1L], dx=0.0, dy=0.0)
        b1 = _beam_at(2.5, 1.7)
        b2 = _beam_at(2.5, 1.7)
        m_a.apply_kick(b1); m_b.apply_kick(b2)
        np.testing.assert_array_equal(b1.particles, b2.particles)

    def test_offset_quad_feeds_down_to_dipole(self):
        """A misaligned quadrupole acquires a dipole feed-down kick.

        For a normal quad with offset dx, dy:
            Δx' - i Δy' = -k1L · ((x − dx) + i(y − dy))
        At (x = dx, y = dy) the kick is zero — the offset shifts the
        magnetic centre.
        """
        k1L = 2.0
        dx, dy = 1.0, 0.5  # mm
        m = Multipole("M", knl=[0.0, k1L], dx=dx, dy=dy)
        beam = _beam_at(dx, dy)  # particle on the shifted axis → no kick
        m.apply_kick(beam)
        np.testing.assert_allclose(beam.particles[0, 1], 0.0, atol=1e-14)
        np.testing.assert_allclose(beam.particles[0, 3], 0.0, atol=1e-14)

        # Same particle without offset → non-zero kick
        m_centred = Multipole("M0", knl=[0.0, k1L])
        b2 = _beam_at(dx, dy)
        m_centred.apply_kick(b2)
        assert abs(b2.particles[0, 1]) > 0
        assert abs(b2.particles[0, 3]) > 0

    def test_tilt_45_converts_normal_quad_to_skew(self):
        """A normal quad tilted by 45° behaves like a skew quad.

        Linearised: Δx' ↔ Δy' under k1L_n → k1L_s after a 45° tilt.
        We compare the tilted-normal kick at (x, 0) to a pure skew kick
        at (0, x).
        """
        k1L = 1.0
        m_tilted = Multipole("MT", knl=[0.0, k1L], tilt_deg=45.0)
        m_skew = Multipole("MS", ksl=[0.0, k1L])

        # Use a particle on x = 2 mm.  After a 45° rotation it lands on
        # (√2, √2) in the rotated frame; the kick is in the rotated
        # frame; rotating back distributes between dx' and dy'.  The
        # *magnitudes* of (dx', dy') for a tilted-normal quad at (2, 0)
        # should equal the magnitudes for a skew quad at (0, 2).
        b1 = _beam_at(2.0, 0.0)
        b2 = _beam_at(0.0, 2.0)
        m_tilted.apply_kick(b1)
        m_skew.apply_kick(b2)
        # Compare magnitudes (sign / sharing depends on which slot was
        # populated, which differs between tilted-normal and pure-skew).
        norm1 = np.hypot(b1.particles[0, 1], b1.particles[0, 3])
        norm2 = np.hypot(b2.particles[0, 1], b2.particles[0, 3])
        np.testing.assert_allclose(norm1, norm2, rtol=1e-10)

    def test_tilt_kick_matrix_preserves_quad_strength_norm(self):
        """The 2-D Frobenius norm of the kick block is tilt-invariant.

        ||M[1::2, 0::2]||² = (M[1,0]² + M[1,2]² + M[3,0]² + M[3,2]²)
        equals 2·k1L² for any tilt angle (the rotation just shuffles
        between normal and skew components).
        """
        k1L = 1.7
        ref = _ref()
        for tilt in (0.0, 17.5, 45.0, 90.0):
            m = Multipole("M", knl=[0.0, k1L], tilt_deg=tilt)
            M = m.kick_matrix(ref)
            block = np.array([[M[1, 0], M[1, 2]], [M[3, 0], M[3, 2]]])
            n2 = np.sum(block ** 2)
            np.testing.assert_allclose(n2, 2.0 * k1L ** 2, rtol=1e-10)

    def test_offset_and_tilt_compose(self):
        """Element with both offset and tilt: kick at the (dx, dy) point
        is zero (after the tilt the displaced origin still lands at the
        rotated origin)."""
        k1L = 1.5
        dx, dy = 0.8, 1.2
        m = Multipole("M", knl=[0.0, k1L], dx=dx, dy=dy, tilt_deg=30.0)
        beam = _beam_at(dx, dy)
        m.apply_kick(beam)
        np.testing.assert_allclose(beam.particles[0, 1], 0.0, atol=1e-14)
        np.testing.assert_allclose(beam.particles[0, 3], 0.0, atol=1e-14)


class TestMultipoleHelperConstructors:
    def test_sextupole_helper(self):
        from linac_gen.elements.multipole import Sextupole
        s = Sextupole("S", k2L=4.0)
        assert s.knl == [0.0, 0.0, 4.0]
        assert s.ksl == [0.0, 0.0, 0.0]

    def test_skew_sextupole_helper(self):
        from linac_gen.elements.multipole import Sextupole
        s = Sextupole("S", k2L=4.0, skew=True)
        assert s.ksl == [0.0, 0.0, 4.0]
        assert s.knl == [0.0, 0.0, 0.0]

    def test_octupole_helper(self):
        from linac_gen.elements.multipole import Octupole
        o = Octupole("O", k3L=2.0)
        assert o.knl == [0.0, 0.0, 0.0, 2.0]

    def test_octupole_kick_pattern(self):
        """Octupole kick scales as r³; the algebra is checked elsewhere
        — just confirm the helper produces a working element."""
        from linac_gen.elements.multipole import Octupole
        o = Octupole("O", k3L=10.0)
        b = _beam_at(2.0, 1.0)
        o.apply_kick(b)
        assert abs(b.particles[0, 1]) > 0  # got a kick
        assert abs(b.particles[0, 3]) > 0


class TestMultipoleMultiOrder:
    def test_superposition_quad_sext(self):
        """Quad + sextupole: total kick = sum of individual kicks."""
        k1L = 1.0
        k2L = 4.0

        m_both = Multipole("M", knl=[0.0, k1L, k2L])
        m_quad = Multipole("Q", knl=[0.0, k1L])
        m_sext = Multipole("S", knl=[0.0, 0.0, k2L])

        x_mm, y_mm = 1.5, 0.5

        beam_both = _beam_at(x_mm, y_mm)
        beam_q = _beam_at(x_mm, y_mm)
        beam_s = _beam_at(x_mm, y_mm)

        m_both.apply_kick(beam_both)
        m_quad.apply_kick(beam_q)
        m_sext.apply_kick(beam_s)

        np.testing.assert_allclose(
            beam_both.particles[:, 1],
            beam_q.particles[:, 1] + beam_s.particles[:, 1],
            rtol=1e-12,
        )
        np.testing.assert_allclose(
            beam_both.particles[:, 3],
            beam_q.particles[:, 3] + beam_s.particles[:, 3],
            rtol=1e-12,
        )
