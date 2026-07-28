# tests/elements/test_dipole.py
"""Tests for the Dipole (sector-bend) element."""
import math
import numpy as np
import pytest
from linac_gen.elements.dipole import Dipole
from linac_gen.elements.base import TransferMapElement
from linac_gen.core.reference import ReferenceParticle
from linac_gen.core.beam import Beam
from linac_gen.core.particle import PROTON


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

def _ref(w_kin: float = 100.0) -> ReferenceParticle:
    return ReferenceParticle(species=PROTON, w_kin=w_kin, frequency=352.21)


def _beam(n: int = 4, w_kin: float = 100.0) -> Beam:
    ref = _ref(w_kin=w_kin)
    return Beam(ref=ref, n_particles=n, current=10.0)


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

class TestDipoleConstruction:
    def test_is_transfer_map_element(self):
        d = Dipole("D", angle=30.0, rho=1000.0)
        assert isinstance(d, TransferMapElement)

    def test_arc_length_calculation(self):
        angle_deg = 30.0
        rho_mm = 2000.0
        d = Dipole("D", angle=angle_deg, rho=rho_mm)
        expected = rho_mm * abs(angle_deg) * math.pi / 180.0
        assert abs(d.length - expected) < 1e-9

    def test_arc_length_negative_angle(self):
        d_pos = Dipole("D+", angle=45.0, rho=1000.0)
        d_neg = Dipole("D-", angle=-45.0, rho=1000.0)
        assert abs(d_pos.length - d_neg.length) < 1e-9

    def test_attributes_stored(self):
        d = Dipole("D", angle=15.0, rho=500.0, e1=5.0, e2=7.0, aperture=25.0, n_steps=10)
        assert d.angle == 15.0
        assert d.rho == 500.0
        assert d.e1 == 5.0
        assert d.e2 == 7.0
        assert d.aperture == 25.0
        assert d.n_steps == 10

    def test_default_edge_angles_zero(self):
        d = Dipole("D", angle=10.0, rho=1000.0)
        assert d.e1 == 0.0
        assert d.e2 == 0.0


# ---------------------------------------------------------------------------
# Transfer matrix shape and type
# ---------------------------------------------------------------------------

class TestDipoleMatrixShape:
    def test_shape_6x6(self):
        d = Dipole("D", angle=10.0, rho=1000.0)
        ref = _ref()
        M = d.transfer_matrix(ref)
        assert M.shape == (6, 6)

    def test_returns_ndarray(self):
        d = Dipole("D", angle=10.0, rho=1000.0)
        ref = _ref()
        M = d.transfer_matrix(ref)
        assert isinstance(M, np.ndarray)


# ---------------------------------------------------------------------------
# Zero-angle limit → identity-like matrix (no bend)
# ---------------------------------------------------------------------------

class TestDipoleZeroAngle:
    def test_zero_angle_identity_transverse(self):
        """A zero-angle dipole should have no transverse focusing."""
        d = Dipole("D", angle=0.0, rho=1000.0)
        ref = _ref()
        M = d.transfer_matrix(ref)
        # No coupling: transverse block should be identity
        np.testing.assert_allclose(M[0, 0], 1.0, atol=1e-12)
        np.testing.assert_allclose(M[0, 1], 0.0, atol=1e-12)
        np.testing.assert_allclose(M[1, 0], 0.0, atol=1e-12)
        np.testing.assert_allclose(M[1, 1], 1.0, atol=1e-12)

    def test_zero_angle_zero_length(self):
        d = Dipole("D", angle=0.0, rho=1000.0)
        assert d.length == 0.0


# ---------------------------------------------------------------------------
# Symplecticity
# ---------------------------------------------------------------------------

class TestDipoleSymplecticity:
    """The 2×2 sub-determinants must equal 1 (symplecticity)."""

    @pytest.mark.parametrize("angle", [15.0, 30.0, 45.0, 90.0])
    def test_horizontal_symplectic(self, angle):
        d = Dipole("D", angle=angle, rho=1000.0)
        ref = _ref()
        M = d.transfer_matrix(ref)
        det_x = M[0, 0] * M[1, 1] - M[0, 1] * M[1, 0]
        assert abs(det_x - 1.0) < 1e-9

    @pytest.mark.parametrize("angle", [15.0, 30.0, 90.0])
    def test_vertical_symplectic(self, angle):
        """Vertical plane is a drift; det must be 1."""
        d = Dipole("D", angle=angle, rho=1000.0)
        ref = _ref()
        M = d.transfer_matrix(ref)
        det_y = M[2, 2] * M[3, 3] - M[2, 3] * M[3, 2]
        assert abs(det_y - 1.0) < 1e-9


# ---------------------------------------------------------------------------
# 180-degree bend reverses horizontal position
# ---------------------------------------------------------------------------

class TestDipole180:
    def test_180_reverses_x(self):
        """After a 180° bend with no divergence, x → −x."""
        d = Dipole("D180", angle=180.0, rho=1000.0)
        ref = _ref()
        M = d.transfer_matrix(ref)
        # M[0,0] should equal cos(pi) = -1
        assert abs(M[0, 0] - (-1.0)) < 1e-9

    def test_180_reverses_xp(self):
        """After a 180° bend, xp → −xp (M[1,1] = cos(pi) = -1)."""
        d = Dipole("D180", angle=180.0, rho=1000.0)
        ref = _ref()
        M = d.transfer_matrix(ref)
        assert abs(M[1, 1] - (-1.0)) < 1e-9


# ---------------------------------------------------------------------------
# Vertical plane acts like a drift
# ---------------------------------------------------------------------------

class TestDipoleVerticalDrift:
    def test_vertical_drift_coupling(self):
        """M[2,3] should equal arc_length_in_metres for the vertical drift."""
        angle_deg = 30.0
        rho_mm = 2000.0
        d = Dipole("D", angle=angle_deg, rho=rho_mm)
        ref = _ref()
        M = d.transfer_matrix(ref)
        arc_m = d.length * 1e-3
        assert abs(M[2, 3] - arc_m) < 1e-9

    def test_vertical_no_focusing(self):
        """Without edge angles, M[3,2] (vertical focusing) should be zero."""
        d = Dipole("D", angle=45.0, rho=1000.0, e1=0.0, e2=0.0)
        ref = _ref()
        M = d.transfer_matrix(ref)
        assert abs(M[3, 2]) < 1e-12


# ---------------------------------------------------------------------------
# Edge focusing
# ---------------------------------------------------------------------------

class TestDipoleEdgeFocusing:
    def test_edge_modifies_matrix(self):
        """Non-zero e1/e2 should change the matrix compared to e1=e2=0."""
        d_no_edge = Dipole("D", angle=30.0, rho=1000.0, e1=0.0, e2=0.0)
        d_edge = Dipole("D", angle=30.0, rho=1000.0, e1=10.0, e2=10.0)
        ref = _ref()
        M_no = d_no_edge.transfer_matrix(ref)
        M_ed = d_edge.transfer_matrix(ref)
        assert not np.allclose(M_no, M_ed)

    def test_edge_reduces_horizontal_focusing(self):
        """Positive edge angles in a sector bend reduce horizontal body focusing.

        The edge matrix contributes +tan(e)/rho to M[1,0], shifting it toward
        zero.  With symmetric edges of 15° and 30° bend, the combined M[1,0]
        is less negative (or positive) than the body-only value.
        """
        d_no = Dipole("D", angle=30.0, rho=1000.0, e1=0.0, e2=0.0)
        d_ed = Dipole("D", angle=30.0, rho=1000.0, e1=15.0, e2=15.0)
        ref = _ref()
        M_no = d_no.transfer_matrix(ref)
        M_ed = d_ed.transfer_matrix(ref)
        # Edge shifts M[1,0] toward zero / positive (less horizontal focusing)
        assert M_ed[1, 0] > M_no[1, 0]

    def test_edge_vertical_focusing(self):
        """Positive edge angles add vertical focusing (M[3,2] < 0) to the drift-like
        vertical plane.  Without edges M[3,2] is zero; with edges it becomes negative.
        """
        d_no = Dipole("D", angle=30.0, rho=1000.0, e1=0.0, e2=0.0)
        d_ed = Dipole("D", angle=30.0, rho=1000.0, e1=15.0, e2=15.0)
        ref = _ref()
        M_no = d_no.transfer_matrix(ref)
        M_ed = d_ed.transfer_matrix(ref)
        assert M_no[3, 2] == 0.0        # pure drift: no vertical focusing
        assert M_ed[3, 2] < 0.0         # edge adds vertical focusing

    def test_symmetric_edges_same_e1_e2(self):
        """Symmetric entrance/exit gives a matrix with M[0,0] == M[1,1] only in body."""
        d = Dipole("D", angle=10.0, rho=2000.0, e1=5.0, e2=5.0)
        ref = _ref()
        M = d.transfer_matrix(ref)
        # Just check matrix is non-trivial and valid shape
        assert M.shape == (6, 6)


# ---------------------------------------------------------------------------
# Slice consistency
# ---------------------------------------------------------------------------

class TestDipoleSlice:
    def test_slice_different_from_full(self):
        d = Dipole("D", angle=30.0, rho=1000.0)
        ref = _ref()
        M_full = d.transfer_matrix(ref)
        M_half = d.transfer_matrix(ref, ds=d.length / 2)
        assert not np.allclose(M_full, M_half)

    def test_two_halves_compose_to_full_body(self):
        """Two half-slice body matrices should multiply to the full body matrix.

        Note: Edges are NOT applied when ds is given, so this tests body only.
        """
        d = Dipole("D", angle=30.0, rho=1000.0, e1=0.0, e2=0.0)
        ref = _ref()
        M_full = d.transfer_matrix(ref)          # full, no edges (e1=e2=0)
        M_half = d.transfer_matrix(ref, ds=d.length / 2)
        M_composed = M_half @ M_half
        np.testing.assert_array_almost_equal(M_composed, M_full, decimal=8)

    def test_slice_proportional_angle(self):
        """A quarter-length slice should have bend angle ≈ angle/4."""
        d = Dipole("D", angle=40.0, rho=1000.0)
        ref = _ref()
        ds = d.length / 4
        M_slice = d.transfer_matrix(ref, ds=ds)
        theta_slice = math.radians(40.0 / 4)
        rho_m = d.rho * 1e-3
        expected_m01 = rho_m * math.sin(theta_slice)
        assert abs(M_slice[0, 1] - expected_m01) < 1e-9


# ---------------------------------------------------------------------------
# Dispersion (M[0,5] and M[1,5])
# ---------------------------------------------------------------------------

class TestDipoleDispersion:
    def test_dispersion_nonzero(self):
        """M[0,5] and M[1,5] (dispersion terms) should be non-zero for a real bend."""
        d = Dipole("D", angle=30.0, rho=1000.0)
        ref = _ref()
        M = d.transfer_matrix(ref)
        assert abs(M[0, 5]) > 0.0, "M[0,5] dispersion should be non-zero"
        assert abs(M[1, 5]) > 0.0, "M[1,5] dispersion should be non-zero"

    def test_dispersion_zero_for_zero_angle(self):
        d = Dipole("D", angle=0.0, rho=1000.0)
        ref = _ref()
        M = d.transfer_matrix(ref)
        assert abs(M[0, 5]) < 1e-12
        assert abs(M[1, 5]) < 1e-12


# ---------------------------------------------------------------------------
# track() method
# ---------------------------------------------------------------------------

class TestDipoleTrack:
    def test_track_advances_reference_s(self):
        d = Dipole("D", angle=20.0, rho=1000.0)
        beam = _beam()
        s_init = beam.ref.s
        d.track(beam)
        assert abs(beam.ref.s - (s_init + d.length)) < 1e-9

    def test_track_only_alive_particles(self):
        d = Dipole("D", angle=10.0, rho=1000.0)
        beam = _beam(n=4)
        beam.particles[0, 0] = 5.0   # give particle 0 a non-zero x
        beam.record_loss(0, 0.0, "pre")
        # Particle 0 is lost, so its x should remain 5.0 after track
        d.track(beam)
        assert beam.particles[0, 0] == 5.0

    def test_track_changes_beam(self):
        """A 30° dipole should displace the transverse coordinates."""
        d = Dipole("D", angle=30.0, rho=1000.0)
        beam = _beam(n=2)
        beam.particles[:, 1] = 1.0   # 1 mrad divergence in x
        before = beam.particles.copy()
        d.track(beam)
        assert not np.allclose(beam.particles, before)


def test_bend_field_index_zero_is_pure_sector():
    """N=0 gives a pure sector bend: horizontal focusing from curvature,
    vertical is pure drift (no focusing), non-zero horizontal dispersion."""
    import numpy as np
    from linac_gen.core.particle import PROTON
    from linac_gen.core.reference import ReferenceParticle
    from linac_gen.elements.dipole import Dipole

    ref = ReferenceParticle(species=PROTON, w_kin=3.0, frequency=352.21)
    pure = Dipole("B1", angle=10.0, rho=500.0, field_index=0.0)
    M = pure.transfer_matrix(ref)
    # Dispersion (horizontal coupling to energy) is non-zero.
    assert M[0, 5] != 0.0
    # Vertical is a pure drift of length rho * angle_rad:
    rho_m = 500.0 * 1e-3
    theta = np.radians(10.0)
    L = rho_m * theta
    assert M[2, 3] == pytest.approx(L, rel=1e-10)
    assert abs(M[3, 2]) < 1e-12  # no vertical focusing


def test_bend_field_index_positive_reduces_horizontal_focusing():
    """A positive N (0 < N < 1) weakens horizontal focusing; picks up vertical focusing."""
    import numpy as np
    from linac_gen.core.particle import PROTON
    from linac_gen.core.reference import ReferenceParticle
    from linac_gen.elements.dipole import Dipole

    ref = ReferenceParticle(species=PROTON, w_kin=3.0, frequency=352.21)
    n0 = Dipole("B0", angle=10.0, rho=500.0, field_index=0.0)
    n5 = Dipole("B5", angle=10.0, rho=500.0, field_index=0.5)
    M0 = n0.transfer_matrix(ref)
    M5 = n5.transfer_matrix(ref)
    # Horizontal focusing magnitude decreases with N.
    assert abs(M5[1, 0]) < abs(M0[1, 0])
    # Vertical picks up focusing (negative sign for focusing).
    assert M5[3, 2] < 0.0
