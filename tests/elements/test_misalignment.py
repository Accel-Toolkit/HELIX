"""Tests for the Misalignment + FieldError mixins applied to elements.

Each element with the mixin is checked for:
* zero-misalignment is a no-op vs the same element without mixin params
* dx/dy shift the centroid by exactly that amount
* tilt_deg of a normal quadrupole produces equivalent skew-quad coupling
* gradient_rel / field_rel / voltage_rel scale the kick proportionally
"""
from __future__ import annotations

import math
import numpy as np
import pytest

from linac_gen.core.beam import Beam
from linac_gen.core.particle import PROTON
from linac_gen.core.reference import ReferenceParticle
from linac_gen.elements.dipole import Dipole
from linac_gen.elements.drift import Drift
from linac_gen.elements.quadrupole import Quadrupole
from linac_gen.elements.rf_gap import RFGap
from linac_gen.elements.solenoid import Solenoid


def _ref(w_kin: float = 100.0) -> ReferenceParticle:
    return ReferenceParticle(species=PROTON, w_kin=w_kin, frequency=325.0)


def _on_axis_beam(n: int = 1) -> Beam:
    """A beam at the origin (no centroid offset, no spread)."""
    return Beam(ref=_ref(), n_particles=n, current=0.0)


def _shifted_beam(x_mm: float = 0.0, y_mm: float = 0.0,
                  xp_mr: float = 0.0, yp_mr: float = 0.0,
                  n: int = 1) -> Beam:
    b = Beam(ref=_ref(), n_particles=n, current=0.0)
    b.particles[:, 0] = x_mm
    b.particles[:, 2] = y_mm
    b.particles[:, 1] = xp_mr
    b.particles[:, 3] = yp_mr
    return b


# ---------------------------------------------------------------------------
class TestMisalignmentDefaults:
    """Default constructor (no kwargs) → all misalignment slots zero."""

    @pytest.mark.parametrize("ctor", [
        lambda: Quadrupole("Q", length=100, gradient=10),
        lambda: Solenoid("S", length=100, field=2.0),
        lambda: Dipole("B", angle=10, rho=1000),
        lambda: Drift("D", length=100),
        lambda: RFGap("G", voltage=1.0, phase=-30, frequency=325),
    ])
    def test_no_misalignment_by_default(self, ctor):
        elem = ctor()
        assert elem.dx == 0.0
        assert elem.dy == 0.0
        assert elem.dz == 0.0
        assert elem.tilt_deg == 0.0
        assert elem.pitch_deg == 0.0
        assert elem.yaw_deg == 0.0
        assert not elem.is_misaligned


class TestMisalignmentReprAndFlag:
    def test_is_misaligned_flag(self):
        q = Quadrupole("Q", length=100, gradient=10, dx=0.5)
        assert q.is_misaligned
        q2 = Quadrupole("Q", length=100, gradient=10)
        assert not q2.is_misaligned

    def test_misalignment_repr(self):
        q = Quadrupole("Q", length=100, gradient=10, dx=0.5, tilt_deg=2.0)
        rep = q._misalignment_repr()
        assert "dx=0.5" in rep
        assert "tilt=2" in rep


# ---------------------------------------------------------------------------
class TestFieldErrorScaling:
    def test_quad_gradient_rel_zero_is_no_op(self):
        q1 = Quadrupole("Q", length=100, gradient=10, gradient_rel=0.0)
        q2 = Quadrupole("Q", length=100, gradient=10)
        ref = _ref()
        np.testing.assert_array_equal(q1.transfer_matrix(ref),
                                      q2.transfer_matrix(ref))

    def test_quad_gradient_rel_scales_focusing(self):
        """gradient_rel = +0.05 → effective gradient 5 % stronger.

        Direct check on ``effective_gradient``; the matrix changes
        nonlinearly with gradient (sin/cos of kL), so we don't try to
        predict the matrix ratio in closed form here.
        """
        q0 = Quadrupole("Q", length=100, gradient=10, gradient_rel=0.0)
        q1 = Quadrupole("Q", length=100, gradient=10, gradient_rel=0.05)
        assert q0.effective_gradient == pytest.approx(10.0)
        assert q1.effective_gradient == pytest.approx(10.5)
        # And the resulting matrix actually differs.
        ref = _ref()
        assert not np.allclose(q0.transfer_matrix(ref),
                               q1.transfer_matrix(ref))

    def test_solenoid_field_rel_scales_kick(self):
        ref = _ref()
        s0 = Solenoid("S", length=100, field=2.0, field_rel=0.0)
        s1 = Solenoid("S", length=100, field=2.0, field_rel=0.10)
        m0 = s0.transfer_matrix(ref)
        m1 = s1.transfer_matrix(ref)
        # Solenoid coupling angle KL is ∝ B; a 10 % field error → 10 %
        # different rotation.  Pick the M[0,2] off-diagonal coupling.
        assert abs(m0[0, 2] - m1[0, 2]) > 1e-6

    def test_dipole_field_rel_scales_angle(self):
        d0 = Dipole("B", angle=30, rho=1000, field_rel=0.0)
        d1 = Dipole("B", angle=30, rho=1000, field_rel=-0.02)
        assert d0.effective_angle == pytest.approx(30.0, rel=1e-12)
        assert d1.effective_angle == pytest.approx(30.0 * 0.98, rel=1e-12)

    def test_rfgap_voltage_rel(self):
        g = RFGap("G", voltage=2.0, phase=-30, frequency=325, voltage_rel=0.05)
        assert g.effective_voltage == pytest.approx(2.10, rel=1e-12)
        assert g.effective_phase == pytest.approx(-30.0, abs=1e-12)
        assert g.effective_frequency == pytest.approx(325.0, abs=1e-12)

    def test_rfgap_phase_offset(self):
        g = RFGap("G", voltage=2.0, phase=-30, frequency=325,
                  phase_offset=2.5)
        assert g.effective_phase == pytest.approx(-27.5, rel=1e-12)


# ---------------------------------------------------------------------------
class TestQuadHigherMultipoles:
    """Wired g3/g4 quad multipole content (Tier 1.3)."""

    def test_zero_g3_is_no_op(self):
        """A quad with g3=0 (default) tracks like the base linear quad."""
        q1 = Quadrupole("Q", length=100, gradient=5, g3=0.0)
        q2 = Quadrupole("Q", length=100, gradient=5)
        beam1 = _shifted_beam(2.0, 1.0)
        beam2 = _shifted_beam(2.0, 1.0)
        q1.track(beam1)
        q2.track(beam2)
        np.testing.assert_array_equal(beam1.particles, beam2.particles)

    def test_g3_introduces_x2_dependence(self):
        """A quad with g3 > 0 produces an x²-dependent kick on top of the
        linear focusing.  Compare two beams differing only in initial x;
        the difference in x' kick should scale ~ x² as soon as g3 ≠ 0
        but remain linear when g3 = 0.
        """
        ref = _ref()
        q_no_g3 = Quadrupole("Q0", length=100, gradient=5, g3=0.0)
        q_with_g3 = Quadrupole("Q1", length=100, gradient=5, g3=4.0)

        # Two beams: x = 1mm and x = 2mm.  For pure linear quad, kick(2)
        # is exactly 2 × kick(1).  With g3, there's a quadratic excess.
        b1a = _shifted_beam(1.0, 0.0); b2a = _shifted_beam(2.0, 0.0)
        b1b = _shifted_beam(1.0, 0.0); b2b = _shifted_beam(2.0, 0.0)
        q_no_g3.track(b1a); q_no_g3.track(b2a)
        q_with_g3.track(b1b); q_with_g3.track(b2b)

        # Linear-only: ratio of x' kicks at x=2 vs x=1 is exactly 2.
        ratio_lin = (b2a.particles[0, 1] - 0) / (b1a.particles[0, 1] - 0)
        ratio_g3 = (b2b.particles[0, 1] - 0) / (b1b.particles[0, 1] - 0)
        # Ratio should differ between the two (g3 introduces nonlinearity).
        assert abs(ratio_lin - ratio_g3) > 1e-6


# ---------------------------------------------------------------------------
class TestTrackerMisalignment:
    """End-to-end: tracker honours dx, dy, tilt_deg via the pre/post block."""

    def test_dx_shifts_beam_centroid_through_drift(self):
        """A drift with dx=2mm should leave the beam unaffected (drift
        is field-free): tracker shifts in then shifts out → identity.
        """
        from linac_gen.tracking.tracker import Tracker
        from linac_gen.core.lattice import Lattice
        beam = _shifted_beam(0.0, 0.0)
        lat = Lattice()
        lat.add(Drift("D", length=100, dx=2.0, dy=0.5))
        Tracker(lat, beam).run()
        # Drift is field-free: pre-shift then post-shift exactly cancel.
        np.testing.assert_allclose(beam.particles[0, 0], 0.0, atol=1e-12)
        np.testing.assert_allclose(beam.particles[0, 2], 0.0, atol=1e-12)

    def test_dx_on_quad_moves_focal_point(self):
        """A misaligned quad does NOT cancel: a particle on the magnetic
        axis (x=dx, y=0) gets no kick, but a particle at the geometric
        origin gets a non-zero kick equal to a centred quad evaluated at -dx."""
        from linac_gen.tracking.tracker import Tracker
        from linac_gen.core.lattice import Lattice
        ref = _ref()
        q_centered = Quadrupole("Q", length=100, gradient=5)
        # Centered quad applied to particle at x=2, y=0
        beam_c = _shifted_beam(2.0, 0.0)
        lat_c = Lattice(); lat_c.add(q_centered)
        Tracker(lat_c, beam_c).run()
        kick_centered = beam_c.particles[0, 1]

        # Misaligned quad (dx=2) applied to particle at x=2, y=0 → on-axis → no kick
        q_shifted = Quadrupole("Q", length=100, gradient=5, dx=2.0)
        beam_s = _shifted_beam(2.0, 0.0)
        lat_s = Lattice(); lat_s.add(q_shifted)
        Tracker(lat_s, beam_s).run()
        kick_shifted = beam_s.particles[0, 1]

        assert abs(kick_centered) > 0.5  # got a real kick
        assert abs(kick_shifted) < abs(kick_centered) * 0.05  # ~no kick

    def test_tilt_45_normal_quad_couples_xy(self):
        """A tilt_deg=45° normal quad acts like a skew quad: a particle
        at x=2, y=0 receives a y' kick (not just x')."""
        from linac_gen.tracking.tracker import Tracker
        from linac_gen.core.lattice import Lattice
        beam = _shifted_beam(2.0, 0.0)
        q = Quadrupole("Q", length=100, gradient=5, tilt_deg=45.0)
        lat = Lattice(); lat.add(q)
        Tracker(lat, beam).run()
        # In tilted frame, particle is at (√2, √2); kick has both x and y
        # components after rotating back.
        assert abs(beam.particles[0, 1]) > 1e-6  # got x' kick
        assert abs(beam.particles[0, 3]) > 1e-6  # got y' kick


# ---------------------------------------------------------------------------
class TestZeroMisalignmentRoundTrip:
    """All-zero misalignment must produce identical results to no-mixin paths."""

    def test_quad_zero_misalignment_matches_baseline(self):
        from linac_gen.tracking.tracker import Tracker
        from linac_gen.core.lattice import Lattice
        b1 = _shifted_beam(1.0, 0.5, 0.1, 0.05)
        b2 = _shifted_beam(1.0, 0.5, 0.1, 0.05)
        lat1 = Lattice(); lat1.add(Quadrupole("Q", length=100, gradient=10))
        lat2 = Lattice(); lat2.add(Quadrupole("Q", length=100, gradient=10,
                                              dx=0.0, dy=0.0, tilt_deg=0.0))
        Tracker(lat1, b1).run()
        Tracker(lat2, b2).run()
        np.testing.assert_array_equal(b1.particles, b2.particles)
