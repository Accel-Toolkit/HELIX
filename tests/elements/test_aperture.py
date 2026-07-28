"""Aperture element: shape-aware loss marking.

Covers the TraceWin shape flags (``n``) supported by ``Aperture``:

    n = 0   Rectangular (dx, dy)
    n = 1   Circular (dx = radius)
    n = 2   Pepperpot (pass-through with warning)
    n = 3   Fraction (treated as rectangular)
    n = 4/5 Finger (treated as rectangular for now)
    n = 6   Ring (pass-through with warning)
"""
import numpy as np
import pytest

from linac_gen.core.beam import Beam
from linac_gen.core.particle import PROTON
from linac_gen.core.reference import ReferenceParticle
from linac_gen.elements.aperture import Aperture


def _make_beam_with_positions(xs, ys):
    """Helper: a beam with explicit transverse positions, zero current."""
    ref = ReferenceParticle(species=PROTON, w_kin=3.0, frequency=352.21)
    n = len(xs)
    beam = Beam(ref=ref, n_particles=n, current=0.0)
    if n:
        beam.particles[:, 0] = xs
        beam.particles[:, 2] = ys
    return beam


# ── TraceWin shape-flag semantics (Task 4.1) ────────────────────────────────

def test_rectangular_aperture_marks_outside_dxy():
    ref = ReferenceParticle(species=PROTON, w_kin=3.0, frequency=352.21)
    beam = Beam(ref=ref, n_particles=4, current=0.0)
    beam.particles[:, 0] = [5.0, 20.0, 0.0, 0.0]   # x
    beam.particles[:, 2] = [0.0, 0.0, 15.0, 30.0]  # y

    ap = Aperture("AP", dx=10.0, dy=20.0, aperture_type=0)
    ap.apply(beam)
    assert list(beam.lost) == [False, True, False, True]


def test_circular_aperture_uses_dx_as_radius():
    ref = ReferenceParticle(species=PROTON, w_kin=3.0, frequency=352.21)
    beam = Beam(ref=ref, n_particles=3, current=0.0)
    beam.particles[:, 0] = [3.0, 0.0, 5.0]
    beam.particles[:, 2] = [0.0, 5.0, 5.0]
    ap = Aperture("AP", dx=6.0, dy=0.0, aperture_type=1)
    ap.apply(beam)
    # r = 3, 5, 7.07 -> third is lost
    assert list(beam.lost) == [False, False, True]


def test_fraction_mode_accepts_any_beam_without_error():
    """ap_type=3 is a beam-fraction setting; we implement as rectangular."""
    ref = ReferenceParticle(species=PROTON, w_kin=3.0, frequency=352.21)
    beam = Beam(ref=ref, n_particles=2, current=0.0)
    beam.particles[0, 0] = 2.0
    beam.particles[1, 0] = 20.0
    ap = Aperture("AP", dx=10.0, dy=10.0, aperture_type=3)
    ap.apply(beam)   # must not raise
    # Rectangular semantics: |x|>10 -> lost; |x|<=10 -> alive.
    assert list(beam.lost) == [False, True]


def test_pepperpot_warns_and_passes_through():
    ref = ReferenceParticle(species=PROTON, w_kin=3.0, frequency=352.21)
    beam = Beam(ref=ref, n_particles=1, current=0.0)
    beam.particles[0, 0] = 50.0   # far outside any reasonable aperture
    ap = Aperture("AP", dx=1.0, dy=1.0, aperture_type=2)
    ap.apply(beam)
    # Pepperpot is not simulated -- particle stays alive.
    assert not beam.lost[0]


# ── Legacy behaviour retained: circular/rectangular loss mechanics ──────────

class TestApertureCircular:
    def test_circular_keeps_particles_inside(self):
        beam = _make_beam_with_positions([0.5, 0.0], [0.0, 0.5])
        ap = Aperture("AP", aperture_type=Aperture.CIRCULAR, dx=1.0)
        ap.apply(beam)
        assert beam.n_alive == 2

    def test_circular_loses_particles_outside(self):
        beam = _make_beam_with_positions([2.0, 0.0], [0.0, 0.0])
        ap = Aperture("AP", aperture_type=Aperture.CIRCULAR, dx=1.0)
        ap.apply(beam)
        assert beam.n_alive == 1

    def test_circular_boundary_lost(self):
        beam = _make_beam_with_positions([1.001], [0.0])
        ap = Aperture("AP", aperture_type=Aperture.CIRCULAR, dx=1.0)
        ap.apply(beam)
        assert beam.n_alive == 0

    def test_circular_default_b_equals_a(self):
        ap = Aperture("AP", aperture_type=Aperture.CIRCULAR, dx=5.0)
        # Legacy alias: when dy <= 0, vertical half-size falls back to dx.
        assert ap.b == 5.0
        assert ap.dy == 5.0

    def test_circular_loss_table_populated(self):
        beam = _make_beam_with_positions([2.0], [0.0])
        ap = Aperture("AP", aperture_type=Aperture.CIRCULAR, dx=1.0)
        ap.apply(beam)
        lt = beam.loss_table
        assert len(lt) == 1
        assert lt[0]["element_name"] == "AP"

    def test_circular_mixed(self):
        xs = [0.5, 1.5, 0.0]
        ys = [0.0, 0.0, 0.8]
        beam = _make_beam_with_positions(xs, ys)
        ap = Aperture("AP", aperture_type=Aperture.CIRCULAR, dx=1.0)
        ap.apply(beam)
        assert beam.n_alive == 2


class TestApertureRectangular:
    def test_rectangular_keeps_inside(self):
        beam = _make_beam_with_positions([0.5, -0.5], [0.3, -0.3])
        ap = Aperture("AP", aperture_type=Aperture.RECTANGULAR, dx=1.0, dy=0.5)
        ap.apply(beam)
        assert beam.n_alive == 2

    def test_rectangular_loses_outside_x(self):
        beam = _make_beam_with_positions([1.5, 0.0], [0.0, 0.0])
        ap = Aperture("AP", aperture_type=Aperture.RECTANGULAR, dx=1.0, dy=1.0)
        ap.apply(beam)
        assert beam.n_alive == 1

    def test_rectangular_loses_outside_y(self):
        beam = _make_beam_with_positions([0.0, 0.0], [0.0, 1.5])
        ap = Aperture("AP", aperture_type=Aperture.RECTANGULAR, dx=1.0, dy=1.0)
        ap.apply(beam)
        assert beam.n_alive == 1

    def test_rectangular_asymmetric(self):
        beam = _make_beam_with_positions([1.5], [0.6])
        ap = Aperture("AP", aperture_type=Aperture.RECTANGULAR, dx=2.0, dy=0.5)
        ap.apply(beam)
        assert beam.n_alive == 0

    def test_rectangular_loss_table_entry(self):
        beam = _make_beam_with_positions([2.0], [0.0])
        ap = Aperture("AP2", aperture_type=Aperture.RECTANGULAR, dx=1.0, dy=1.0)
        ap.apply(beam)
        assert beam.loss_table[0]["element_name"] == "AP2"


class TestApertureGeneral:
    def test_zero_length(self):
        ap = Aperture("AP", aperture_type=Aperture.CIRCULAR, dx=5.0)
        assert ap.length == 0.0

    def test_zero_aperture_element(self):
        ap = Aperture("AP", aperture_type=Aperture.CIRCULAR, dx=0.0)
        assert ap.dx == 0.0
        assert ap.a == 0.0   # legacy alias

    def test_empty_beam_no_error(self):
        beam = _make_beam_with_positions([], [])
        ap = Aperture("AP", aperture_type=Aperture.CIRCULAR, dx=1.0)
        ap.apply(beam)  # should not raise

    def test_all_alive_no_loss(self):
        beam = _make_beam_with_positions([0.1, 0.2], [0.1, 0.2])
        ap = Aperture("AP", aperture_type=Aperture.CIRCULAR, dx=5.0)
        ap.apply(beam)
        assert beam.n_alive == 2
        assert len(beam.loss_table) == 0

    def test_constants_match_tracewin_convention(self):
        """Constants follow TraceWin ``n`` ordering (0 rect, 1 circle, ...)."""
        assert Aperture.RECTANGULAR == 0
        assert Aperture.CIRCULAR == 1
        assert Aperture.PEPPERPOT == 2
        assert Aperture.FRACTION == 3
