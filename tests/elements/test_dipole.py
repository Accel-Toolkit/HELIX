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


# ---------------------------------------------------------------------------
# Bend direction, field-index branches and the pinned baseline
# (fix of 2026-09-06: |theta| and |rho| in the body trig, sign(theta) on the
# dispersion column only; hyperbolic and parabolic dispersion in mm/MeV,
# mrad/MeV like every other branch)
# ---------------------------------------------------------------------------

_S_X = np.diag([-1.0, -1.0, 1.0, 1.0, 1.0, 1.0])     # mirror x -> -x
_FIELD_INDICES = [0.0, 0.3, 1.0, 1.5, -400.0]


def _body_x(M):
    return M[:2, :2]


@pytest.mark.parametrize("N", _FIELD_INDICES)
@pytest.mark.parametrize("edges", [(0.0, 0.0), (10.0, 20.0)])
def test_mirror_symmetry_negative_angle(N, edges):
    """M(-theta) == S M(+theta) S exactly: the mirror image of a magnet has
    the same 4x4 focusing and an opposite-sign dispersion column."""
    ref = _ref(100.0)
    e1, e2 = edges
    M_pos = Dipole("b", angle=+20.0, rho=1000.0, e1=e1, e2=e2, field_index=N).transfer_matrix(ref)
    M_neg = Dipole("b", angle=-20.0, rho=1000.0, e1=e1, e2=e2, field_index=N).transfer_matrix(ref)
    assert np.array_equal(M_neg, _S_X @ M_pos @ _S_X)
    assert np.array_equal(M_neg[:4, :4], M_pos[:4, :4])
    assert M_neg[0, 5] == -M_pos[0, 5] and M_neg[1, 5] == -M_pos[1, 5]
    assert M_pos[0, 5] > 0.0 and M_pos[1, 5] > 0.0          # positive bend: D, D' > 0


@pytest.mark.parametrize("N", _FIELD_INDICES)
def test_signed_rho_convention_equals_positive_rho(N):
    """The Elegant importer keeps rho = L/angle signed; the sign of rho must
    add nothing the angle sign does not already say."""
    ref = _ref(100.0)
    M_a = Dipole("b", angle=-15.0, rho=-2000.0, field_index=N).transfer_matrix(ref)
    M_b = Dipole("b", angle=-15.0, rho=+2000.0, field_index=N).transfer_matrix(ref)
    assert np.array_equal(M_a, M_b)


@pytest.mark.parametrize("angle", [-30.0, -5.0, 5.0, 30.0])
@pytest.mark.parametrize("N", _FIELD_INDICES)
def test_two_by_two_determinants_all_branches(angle, N):
    ref = _ref(100.0)
    M = Dipole("b", angle=angle, rho=1500.0, field_index=N).transfer_matrix(ref)
    for blk in (M[:2, :2], M[2:4, 2:4]):
        # ch^2 - sh^2 cancels catastrophically in a strongly hyperbolic
        # block (entries ~1e5 for N = -400): scale the tolerance with |M|^2
        tol = 1e-12 * max(1.0, float(np.abs(blk).max()) ** 2)
        assert np.linalg.det(blk) == pytest.approx(1.0, abs=tol)


@pytest.mark.parametrize("angle", [-30.0, -20.0, 25.0])
@pytest.mark.parametrize("N", _FIELD_INDICES)
def test_half_slices_compose_to_full_body(angle, N):
    ref = _ref(100.0)
    d = Dipole("b", angle=angle, rho=1000.0, field_index=N)
    M_half = d.transfer_matrix(ref, ds=d.length / 2.0)
    M_full = d.transfer_matrix(ref, ds=d.length)
    np.testing.assert_allclose(M_half @ M_half, M_full, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("angle", [-20.0, 20.0])
def test_continuity_across_field_index_one(angle):
    """The parabolic N = 1 branch is the limit of the elliptic (N < 1) and
    hyperbolic (N > 1) branches — no jump and no unit change."""
    ref = _ref(100.0)
    Ms = {N: Dipole("b", angle=angle, rho=1000.0, field_index=N).transfer_matrix(ref)
          for N in (1.0 - 1e-6, 1.0, 1.0 + 1e-6)}
    # entries that vanish exactly at N = 1 (R21 = -kx^2 L) are O(dN) away
    np.testing.assert_allclose(Ms[1.0 - 1e-6], Ms[1.0], rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(Ms[1.0 + 1e-6], Ms[1.0], rtol=1e-5, atol=1e-6)


def test_field_index_one_dispersion_closed_form():
    """N = 1: the bend plane is a drift and D = L^2/(2 rho), D' = L/rho
    (mm/MeV, mrad/MeV) — the parabolic limit of (1 - cos kL)/(rho k^2)."""
    ref = _ref(100.0)
    rho, angle = 1000.0, 20.0
    d = Dipole("b", angle=angle, rho=rho, field_index=1.0)
    M = d.transfer_matrix(ref)
    L_m = d.length * 1e-3
    rho_m = rho * 1e-3
    beta2gm = ref.beta ** 2 * ref.gamma * ref.species.mass
    assert M[0, 1] == pytest.approx(L_m, rel=1e-12)
    assert M[0, 5] == pytest.approx(1000.0 * L_m * L_m / (2.0 * rho_m) / beta2gm, rel=1e-12)
    assert M[1, 5] == pytest.approx(1000.0 * L_m / rho_m / beta2gm, rel=1e-12)


def test_hyperbolic_dispersion_has_the_same_units_as_the_elliptic_branch():
    """N slightly above vs slightly below 1 must give dispersion of the same
    magnitude — the old hyperbolic branch was 1000x too small."""
    ref = _ref(100.0)
    M_lo = Dipole("b", angle=20.0, rho=1000.0, field_index=0.9).transfer_matrix(ref)
    M_hi = Dipole("b", angle=20.0, rho=1000.0, field_index=1.1).transfer_matrix(ref)
    assert M_hi[0, 5] == pytest.approx(M_lo[0, 5], rel=0.05)
    assert M_hi[1, 5] == pytest.approx(M_lo[1, 5], rel=0.05)
    assert M_hi[0, 5] > 0.0 and M_hi[1, 5] > 0.0


def test_near_unity_field_index_is_numerically_stable():
    """N = 1 +- 2e-16 (a MAD-X k1 = -1/rho^2 deck): the relative guard
    routes to the parabolic branch instead of dividing by a kx2 that has
    lost every digit."""
    ref = _ref(100.0)
    exact = Dipole("b", angle=20.0, rho=1000.0, field_index=1.0).transfer_matrix(ref)
    for N in (1.0 - 2e-16, 1.0 + 2e-16, 1.0 - 1e-12):
        M = Dipole("b", angle=20.0, rho=1000.0, field_index=N).transfer_matrix(ref)
        np.testing.assert_allclose(M, exact, rtol=1e-9, atol=1e-12)


def test_vertical_negative_bend_unchanged_by_the_horizontal_fix():
    """hv=1 was already right: |theta| goes to the body and the plane swap
    flips the dispersion by hand.  Pin the mirror identity there too."""
    ref = _ref(100.0)
    S_y = np.diag([1.0, 1.0, -1.0, -1.0, 1.0, 1.0])
    M_pos = Dipole("b", angle=+20.0, rho=1000.0, hv=1).transfer_matrix(ref)
    M_neg = Dipole("b", angle=-20.0, rho=1000.0, hv=1).transfer_matrix(ref)
    assert np.array_equal(M_neg, S_y @ M_pos @ S_y)


def test_dipole_baseline_bit_identical():
    """Positive-angle (rho > 0) matrices and the Elegant signed-rho
    convention are pinned bit for bit against the fixture written by
    ``regen_dipole_baseline.py`` from the tree before the 2026-09-06 fix."""
    from pathlib import Path
    from tests.elements.regen_dipole_baseline import matrices
    fixture = Path(__file__).parent / "fixtures" / "dipole_matrix_baseline.npz"
    base = np.load(fixture)
    now = matrices()
    assert set(base.files) == set(now)
    bad = [k for k in base.files if not np.array_equal(base[k], now[k])]
    assert not bad, f"{len(bad)} configurations moved, e.g. {bad[:5]}"


# ---------------------------------------------------------------------------
# Field error at fixed magnet length; thin slices; near-unity field index
# (review follow-ups, 2026-09-06)
# ---------------------------------------------------------------------------

def test_field_rel_keeps_the_arc_length():
    """B(1+δ) in a magnet of fixed length: θ → θ(1+δ), ρ → ρ/(1+δ), the
    vertical drift and the phase slip see the design length."""
    ref = _ref(100.0)
    d = Dipole("b", angle=12.0, rho=1500.0, field_rel=0.01)
    ideal = Dipole("b", angle=12.0 * 1.01, rho=1500.0 / 1.01)
    assert d.length == Dipole("b", angle=12.0, rho=1500.0).length
    M = d.transfer_matrix(ref)
    np.testing.assert_allclose(M, ideal.transfer_matrix(ref), rtol=1e-14, atol=1e-15)
    assert M[2, 3] == pytest.approx(d.length * 1e-3, rel=1e-12)          # vertical drift = L
    M_half = d.transfer_matrix(ref, ds=d.length / 2.0)
    assert M_half[2, 3] == pytest.approx(d.length * 0.5e-3, rel=1e-12)


def test_zero_field_is_a_drift_of_the_magnet_length():
    from linac_gen.elements.drift import Drift
    ref = _ref(100.0)
    d = Dipole("b", angle=12.0, rho=1500.0, field_rel=-1.0)
    M = d.transfer_matrix(ref)
    np.testing.assert_allclose(M, Drift("d", length=d.length).transfer_matrix(ref), rtol=0, atol=1e-15)


@pytest.mark.parametrize("N", [0.9, 0.99, 1.1, -0.5])
def test_thin_slices_keep_the_bend_plane_focusing(N):
    """500 slices of a 5° bend: the product must equal the full body (the
    old kx²L² guard silently dropped R21 = -kx²L for thin slices)."""
    ref = _ref(100.0)
    d = Dipole("b", angle=-5.0, rho=1000.0, field_index=N)
    n = 500
    ds = d.length / n
    M_slice = d.transfer_matrix(ref, ds=ds)
    P = np.eye(6)
    for _ in range(n):
        P = M_slice @ P
    np.testing.assert_allclose(P, d.transfer_matrix(ref, ds=d.length), rtol=1e-9, atol=1e-12)


def test_near_unity_field_index_dispersion_is_precise():
    """N = 1 ± 1e-8: the 2 sin²(kL/2) form keeps the dispersion within 1e-7 of
    the parabolic limit instead of losing eight digits to 1 - cos."""
    ref = _ref(100.0)
    exact = Dipole("b", angle=20.0, rho=1000.0, field_index=1.0).transfer_matrix(ref)
    for N in (1.0 - 1e-8, 1.0 + 1e-8):
        M = Dipole("b", angle=20.0, rho=1000.0, field_index=N).transfer_matrix(ref)
        np.testing.assert_allclose(M[:2, 5], exact[:2, 5], rtol=1e-7)
        # R21 = -kx^2 L is O(dN) = 3.5e-9 away from the exact zero
        np.testing.assert_allclose(M[:2, :2], exact[:2, :2], rtol=1e-7, atol=1e-8)


def test_signed_rho_flips_the_element_edges_like_h_tan_e():
    """The body ignores the sign of rho; the element's own e1/e2 follow
    h·tan e with the signed curvature, so rho < 0 with (e1, e2) equals
    rho > 0 with (-e1, -e2) exactly."""
    ref = _ref(100.0)
    M_neg = Dipole("b", angle=-15.0, rho=-2000.0, e1=10.0, e2=20.0).transfer_matrix(ref)
    M_pos = Dipole("b", angle=-15.0, rho=+2000.0, e1=-10.0, e2=-20.0).transfer_matrix(ref)
    assert np.array_equal(M_neg, M_pos)
    assert not np.array_equal(M_neg, Dipole("b", angle=-15.0, rho=2000.0, e1=10.0, e2=20.0).transfer_matrix(ref))


# ---------------------------------------------------------------------------
# Path-length row: R51, R52 and momentum compaction (2026-09-07)
# ---------------------------------------------------------------------------

def _k_phi(ref):
    """The factor tying the path-length row to the dispersion column."""
    return 0.36 * ref.beta * ref.gamma * ref.species.mass / ref.wavelength


@pytest.mark.parametrize("N", [0.0, 0.3, 1.0, 1.5, -400.0, 1.0 - 1e-8])
@pytest.mark.parametrize("angle", [20.0, -20.0, 0.5])
@pytest.mark.parametrize("hv", [0, 1])
def test_path_length_row_is_the_symplectic_partner_of_the_dispersion(N, angle, hv):
    """M[4,0] and M[4,1] are not free: symplecticity fixes them as a multiple
    of the dispersion column, with the planes exchanged.  Building them any
    other way lets the two rows drift apart."""
    ref = _ref(100.0)
    M = Dipole("b", angle=angle, rho=1000.0, field_index=N, hv=hv).transfer_matrix(ref)
    k = _k_phi(ref)
    row, col = (2, 2) if hv == 1 else (0, 0)
    assert M[4, row] == pytest.approx(k * M[col + 1, 5], rel=1e-12, abs=1e-15)
    assert M[4, row + 1] == pytest.approx(k * M[col, 5], rel=1e-12, abs=1e-15)
    # the other plane never contributes to the path length
    other = 0 if hv == 1 else 2
    assert M[4, other] == 0.0 and M[4, other + 1] == 0.0


def test_path_length_coefficients_are_pure_geometry():
    """dL/dx and dL/dx' depend on the magnet, not on the beam.  The matrix
    entries carry a 1/(beta lambda) from converting a length to a phase, so
    it is the coefficients that must be energy independent."""
    ref_lo, ref_hi = _ref(5.0), _ref(500.0)
    d = Dipole("b", angle=17.0, rho=1500.0, field_index=0.4)
    c_lo = d.transfer_matrix(ref_lo)[4, :2] * (ref_lo.beta * ref_lo.wavelength) / 360.0
    c_hi = d.transfer_matrix(ref_hi)[4, :2] * (ref_hi.beta * ref_hi.wavelength) / 360.0
    np.testing.assert_allclose(c_lo, c_hi, rtol=1e-12)
    assert c_lo[0] > 0.0 and c_lo[1] > 0.0


def test_sector_bend_path_length_matches_the_closed_form():
    """n = 0: dL/dx = sin(theta), dL/dx' = rho(1 - cos theta),
    dL/ddelta = rho(theta - sin theta) — the textbook sector bend."""
    ref = _ref(100.0)
    rho_mm, angle = 1500.0, 22.0
    d = Dipole("b", angle=angle, rho=rho_mm)
    M = d.transfer_matrix(ref)
    th, rho_m = math.radians(angle), rho_mm * 1e-3
    k = 360.0 / (ref.beta * ref.wavelength)
    assert M[4, 0] / k == pytest.approx(math.sin(th), rel=1e-12)
    assert M[4, 1] / k == pytest.approx(rho_m * (1.0 - math.cos(th)), rel=1e-9)
    c3 = M[4, 5] * (ref.beta ** 3 * ref.gamma * ref.species.mass * ref.wavelength) / 360000.0 \
        + d.length * 1e-3 / ref.gamma ** 2
    assert c3 == pytest.approx(rho_m * (th - math.sin(th)), rel=1e-9)


def test_compaction_does_not_depend_on_the_bend_direction():
    """dL/ddelta goes as h^2: an off-energy particle rides the outside of the
    arc whichever way the magnet bends."""
    ref = _ref(100.0)
    for hv in (0, 1):
        pos = Dipole("b", angle=+14.0, rho=1200.0, hv=hv).transfer_matrix(ref)
        neg = Dipole("b", angle=-14.0, rho=1200.0, hv=hv).transfer_matrix(ref)
        assert neg[4, 5] == pytest.approx(pos[4, 5], rel=1e-15)
        row = 2 if hv == 1 else 0
        assert neg[4, row] == pytest.approx(-pos[4, row], rel=1e-15)


def test_compaction_is_continuous_across_the_series_threshold():
    """J = (L - sin(kL)/k)/k^2 is evaluated by a series below
    |kx2 L^2| = 1e-3 and in closed form above; the two must agree there."""
    ref = _ref(100.0)
    rho_mm, angle = 1000.0, 20.0
    L = math.radians(angle) * rho_mm * 1e-3
    n_at = 1.0 - 1e-3 / (L * L)            # exactly |u| = 1e-3
    vals = [Dipole("b", angle=angle, rho=rho_mm,
                   field_index=n_at * (1.0 + eps)).transfer_matrix(ref)[4, 5]
            for eps in (-1e-6, -1e-9, 1e-9, 1e-6)]
    for a, b in zip(vals, vals[1:]):
        assert a == pytest.approx(b, rel=1e-8)
