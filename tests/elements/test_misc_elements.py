# tests/elements/test_misc_elements.py
"""Tests for SpaceChargeComp and ThinLens elements."""
import numpy as np
import pytest
from linac_gen.elements.space_charge_comp import SpaceChargeComp
from linac_gen.elements.thin_lens import ThinLens
from linac_gen.elements.base import PassiveElement, ThinKickElement
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


# ===========================================================================
# SpaceChargeComp
# ===========================================================================

class TestSpaceChargeCompConstruction:
    def test_is_passive_element(self):
        sc = SpaceChargeComp("SC")
        assert isinstance(sc, PassiveElement)

    def test_zero_length(self):
        sc = SpaceChargeComp("SC", factor=0.5)
        assert sc.length == 0.0

    def test_zero_aperture(self):
        sc = SpaceChargeComp("SC")
        assert sc.aperture == 0.0

    def test_default_factor_zero(self):
        sc = SpaceChargeComp("SC")
        assert sc.factor == 0.0

    def test_factor_stored(self):
        sc = SpaceChargeComp("SC", factor=0.75)
        assert sc.factor == 0.75

    def test_name_stored(self):
        sc = SpaceChargeComp("MY_SC")
        assert sc.name == "MY_SC"


class TestSpaceChargeCompApply:
    def test_apply_does_not_change_beam(self):
        """apply() is a no-op: beam must be unchanged."""
        sc = SpaceChargeComp("SC", factor=0.5)
        beam = _beam(n=4)
        beam.particles[:, 0] = 2.0
        beam.particles[:, 1] = 0.5
        before = beam.particles.copy()
        sc.apply(beam)
        np.testing.assert_array_equal(beam.particles, before)

    def test_apply_with_zero_factor(self):
        sc = SpaceChargeComp("SC", factor=0.0)
        beam = _beam(n=2)
        beam.particles[:, 0] = 1.0
        before = beam.particles.copy()
        sc.apply(beam)
        np.testing.assert_array_equal(beam.particles, before)

    def test_apply_with_full_neutralization(self):
        sc = SpaceChargeComp("SC", factor=1.0)
        beam = _beam(n=3)
        beam.particles[:, 2] = 3.0
        before = beam.particles.copy()
        sc.apply(beam)
        np.testing.assert_array_equal(beam.particles, before)

    def test_apply_does_not_modify_reference(self):
        sc = SpaceChargeComp("SC", factor=0.3)
        beam = _beam()
        s_before = beam.ref.s
        wkin_before = beam.ref.w_kin
        sc.apply(beam)
        assert beam.ref.s == s_before
        assert beam.ref.w_kin == wkin_before

    def test_apply_with_lost_particles(self):
        """apply() should not crash even when particles are lost."""
        sc = SpaceChargeComp("SC", factor=0.5)
        beam = _beam(n=4)
        beam.record_loss(0, 0.0, "test")
        before = beam.particles.copy()
        sc.apply(beam)
        np.testing.assert_array_equal(beam.particles, before)


class TestSpaceChargeCompFactor:
    def test_factor_readable(self):
        sc = SpaceChargeComp("SC", factor=0.4)
        assert sc.factor == 0.4

    def test_factor_mutable(self):
        sc = SpaceChargeComp("SC", factor=0.0)
        sc.factor = 0.9
        assert sc.factor == 0.9

    @pytest.mark.parametrize("f", [0.0, 0.25, 0.5, 0.75, 1.0])
    def test_factor_range(self, f):
        sc = SpaceChargeComp("SC", factor=f)
        assert sc.factor == f


# ===========================================================================
# ThinLens
# ===========================================================================

class TestThinLensConstruction:
    def test_is_thin_kick_element(self):
        t = ThinLens("TL", fx=1.0)
        assert isinstance(t, ThinKickElement)

    def test_zero_length(self):
        t = ThinLens("TL", fx=2.0)
        assert t.length == 0.0

    def test_default_infinite_focal_lengths(self):
        t = ThinLens("TL")
        assert t.fx == float('inf')
        assert t.fy == float('inf')

    def test_focal_lengths_stored(self):
        t = ThinLens("TL", fx=2.5, fy=3.0)
        assert t.fx == 2.5
        assert t.fy == 3.0

    def test_aperture_stored(self):
        t = ThinLens("TL", aperture=20.0)
        assert t.aperture == 20.0


class TestThinLensFocusing:
    def test_positive_fx_focuses_x(self):
        """Positive fx: a particle with x>0 should get a negative xp kick."""
        t = ThinLens("TL", fx=1.0)
        beam = _beam(n=1)
        beam.particles[0, 0] = 5.0   # x = 5 mm
        t.apply_kick(beam)
        assert beam.particles[0, 1] < 0.0   # xp decreases → focusing

    def test_negative_fx_defocuses_x(self):
        """Negative fx: a particle with x>0 should get a positive xp kick (diverging lens)."""
        t = ThinLens("TL", fx=-1.0)
        beam = _beam(n=1)
        beam.particles[0, 0] = 5.0
        t.apply_kick(beam)
        assert beam.particles[0, 1] > 0.0

    def test_positive_fy_focuses_y(self):
        """Positive fy: a particle with y>0 should get a negative yp kick."""
        t = ThinLens("TL", fy=2.0)
        beam = _beam(n=1)
        beam.particles[0, 2] = 3.0
        t.apply_kick(beam)
        assert beam.particles[0, 3] < 0.0

    def test_xp_kick_magnitude(self):
        """Δx'[mrad] = -x[mm] / fx[m]."""
        fx = 0.5   # m
        x_mm = 4.0
        t = ThinLens("TL", fx=fx)
        beam = _beam(n=1)
        beam.particles[0, 0] = x_mm
        t.apply_kick(beam)
        expected = -x_mm / fx
        np.testing.assert_allclose(beam.particles[0, 1], expected, rtol=1e-12)

    def test_yp_kick_magnitude(self):
        """Δy'[mrad] = -y[mm] / fy[m]."""
        fy = 0.8
        y_mm = 6.0
        t = ThinLens("TL", fy=fy)
        beam = _beam(n=1)
        beam.particles[0, 2] = y_mm
        t.apply_kick(beam)
        expected = -y_mm / fy
        np.testing.assert_allclose(beam.particles[0, 3], expected, rtol=1e-12)


class TestThinLensInfiniteF:
    def test_infinite_fx_no_x_kick(self):
        t = ThinLens("TL", fx=float('inf'), fy=1.0)
        beam = _beam(n=1)
        beam.particles[0, 0] = 5.0
        t.apply_kick(beam)
        assert beam.particles[0, 1] == 0.0   # no xp kick

    def test_infinite_fy_no_y_kick(self):
        t = ThinLens("TL", fx=1.0, fy=float('inf'))
        beam = _beam(n=1)
        beam.particles[0, 2] = 5.0
        t.apply_kick(beam)
        assert beam.particles[0, 3] == 0.0   # no yp kick

    def test_both_infinite_identity(self):
        """Default ThinLens (both f = inf) should be a no-op."""
        t = ThinLens("TL")
        beam = _beam(n=4)
        beam.particles[:, 0] = 2.0
        beam.particles[:, 2] = 3.0
        before = beam.particles.copy()
        t.apply_kick(beam)
        np.testing.assert_array_equal(beam.particles, before)


class TestThinLensKickMatrix:
    def test_matrix_shape(self):
        t = ThinLens("TL", fx=1.0)
        ref = _ref()
        M = t.kick_matrix(ref)
        assert M.shape == (6, 6)

    def test_matrix_m10_value(self):
        """M[1,0] = -1/fx in mrad/mm."""
        fx = 2.0
        t = ThinLens("TL", fx=fx)
        ref = _ref()
        M = t.kick_matrix(ref)
        assert abs(M[1, 0] - (-1.0 / fx)) < 1e-12

    def test_matrix_m32_value(self):
        """M[3,2] = -1/fy in mrad/mm."""
        fy = 3.0
        t = ThinLens("TL", fy=fy)
        ref = _ref()
        M = t.kick_matrix(ref)
        assert abs(M[3, 2] - (-1.0 / fy)) < 1e-12

    def test_matrix_identity_when_infinite(self):
        t = ThinLens("TL")
        ref = _ref()
        M = t.kick_matrix(ref)
        np.testing.assert_array_equal(M, np.eye(6))

    def test_matrix_other_elements_identity(self):
        """All matrix elements except M[1,0] and M[3,2] should be identity-like."""
        t = ThinLens("TL", fx=1.0, fy=2.0)
        ref = _ref()
        M = t.kick_matrix(ref)
        # Identity block minus the two kick elements
        M_copy = M.copy()
        M_copy[1, 0] = 0.0
        M_copy[3, 2] = 0.0
        np.testing.assert_array_equal(M_copy, np.eye(6))

    def test_matrix_consistent_with_apply_kick(self):
        """kick_matrix and apply_kick should agree for small displacements."""
        fx, fy = 1.5, 2.5
        t = ThinLens("TL", fx=fx, fy=fy)
        ref = _ref()
        M = t.kick_matrix(ref)

        x_mm, y_mm = 3.0, 2.0
        beam = _beam(n=1)
        beam.particles[0, 0] = x_mm
        beam.particles[0, 2] = y_mm
        t.apply_kick(beam)

        # From matrix: Δxp = M[1,0]*x, Δyp = M[3,2]*y
        expected_dxp = M[1, 0] * x_mm
        expected_dyp = M[3, 2] * y_mm
        np.testing.assert_allclose(beam.particles[0, 1], expected_dxp, rtol=1e-12)
        np.testing.assert_allclose(beam.particles[0, 3], expected_dyp, rtol=1e-12)


class TestThinLensAliveMask:
    def test_lost_particles_not_kicked(self):
        t = ThinLens("TL", fx=1.0, fy=1.0)
        beam = _beam(n=3)
        beam.particles[:, 0] = 5.0
        beam.particles[:, 2] = 5.0
        beam.record_loss(0, 0.0, "test")
        t.apply_kick(beam)
        assert beam.particles[0, 1] == 0.0   # lost: not kicked
        assert beam.particles[0, 3] == 0.0
        assert beam.particles[1, 1] != 0.0   # alive: kicked
        assert beam.particles[1, 3] != 0.0


class TestThinLensDecoupled:
    def test_fx_only_no_y_kick(self):
        """fx-only lens: x plane focused, y plane untouched."""
        t = ThinLens("TL", fx=1.0)
        beam = _beam(n=1)
        beam.particles[0, 0] = 3.0
        beam.particles[0, 2] = 4.0
        t.apply_kick(beam)
        assert beam.particles[0, 1] != 0.0   # x was kicked
        assert beam.particles[0, 3] == 0.0   # y untouched

    def test_fy_only_no_x_kick(self):
        """fy-only lens: y plane focused, x plane untouched."""
        t = ThinLens("TL", fy=1.0)
        beam = _beam(n=1)
        beam.particles[0, 0] = 3.0
        beam.particles[0, 2] = 4.0
        t.apply_kick(beam)
        assert beam.particles[0, 1] == 0.0   # x untouched
        assert beam.particles[0, 3] != 0.0   # y was kicked
