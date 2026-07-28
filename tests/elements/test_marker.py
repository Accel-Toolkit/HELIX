# tests/elements/test_marker.py
import numpy as np
import pytest
from linac_gen.core.particle import PROTON
from linac_gen.core.reference import ReferenceParticle
from linac_gen.core.beam import Beam
from linac_gen.elements.marker import Marker


def _make_beam(n=10):
    ref = ReferenceParticle(species=PROTON, w_kin=3.0, frequency=352.21)
    beam = Beam(ref=ref, n_particles=n, current=60.0)
    rng = np.random.default_rng(0)
    beam.particles[:, 0] = rng.normal(0, 1.0, n)
    beam.particles[:, 1] = rng.normal(0, 0.5, n)
    beam.particles[:, 2] = rng.normal(0, 1.0, n)
    beam.particles[:, 3] = rng.normal(0, 0.5, n)
    return beam


def test_marker_has_zero_length():
    m = Marker("START")
    assert m.length == 0.0


def test_marker_has_zero_aperture():
    m = Marker("START")
    assert m.aperture == 0.0


def test_marker_has_zero_n_steps():
    m = Marker("START")
    assert m.n_steps == 0


def test_marker_does_not_change_particles():
    beam = _make_beam()
    before = beam.particles.copy()
    m = Marker("MID")
    m.apply(beam)
    np.testing.assert_array_equal(beam.particles, before)


def test_marker_does_not_cause_loss():
    beam = _make_beam()
    m = Marker("END")
    m.apply(beam)
    assert beam.n_alive == 10
    assert len(beam.loss_table) == 0


def test_marker_does_not_advance_ref():
    beam = _make_beam()
    s_before = beam.ref.s
    phi_before = beam.ref.phi_s
    m = Marker("M1")
    m.apply(beam)
    assert beam.ref.s == s_before
    assert beam.ref.phi_s == phi_before


def test_marker_default_snapshot_false():
    m = Marker("M1")
    assert m.snapshot is False


def test_marker_snapshot_flag_set():
    m = Marker("M1", snapshot=True)
    assert m.snapshot is True


def test_marker_name_stored():
    m = Marker("MY_MARKER")
    assert m.name == "MY_MARKER"


def test_marker_is_passive_element():
    from linac_gen.elements.base import PassiveElement
    m = Marker("M")
    assert isinstance(m, PassiveElement)
