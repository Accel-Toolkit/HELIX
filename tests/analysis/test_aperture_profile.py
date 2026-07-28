"""Tests for ``linac_gen.analysis.aperture_profile``.

The profile is the data backing the aperture overlay on envelope plots.
Coverage:

* Constant-aperture elements emit two endpoints at (s_start, ap) and
  (s_end, ap).
* Per-z FieldMap profiles (Ka=1 ``.ouv``) are unpacked at the right
  longitudinal slot.
* Zero-aperture and zero-length filler elements are skipped silently.
* ``aperture_y`` falls back to ``aperture`` when not specified
  (circular pipe).
* Empty / no-element lattices return empty arrays.
"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from linac_gen.analysis.aperture_profile import aperture_profile


# --------------------------------------------------------------------- #
def _elem(name: str, length: float, aperture: float = 0.0,
          aperture_y: float | None = None,
          field_data=None) -> SimpleNamespace:
    return SimpleNamespace(
        name=name, length=length,
        aperture=aperture, aperture_y=aperture_y,
        field_data=field_data,
    )


def _lattice(*elements) -> SimpleNamespace:
    return SimpleNamespace(elements=list(elements))


# --------------------------------------------------------------------- #
class TestConstantAperture:
    def test_emits_endpoints_per_element(self):
        lat = _lattice(
            _elem("D1", 100.0, aperture=20.0),
            _elem("D2", 200.0, aperture=20.0),
        )
        s, rx, ry = aperture_profile(lat)
        np.testing.assert_array_equal(s,  [0.0, 100.0, 100.0, 300.0])
        np.testing.assert_array_equal(rx, [20.0, 20.0, 20.0, 20.0])
        np.testing.assert_array_equal(ry, [20.0, 20.0, 20.0, 20.0])

    def test_aperture_y_falls_back_to_x_for_circular_pipe(self):
        lat = _lattice(_elem("D1", 100.0, aperture=15.0))   # no aperture_y
        s, rx, ry = aperture_profile(lat)
        np.testing.assert_array_equal(rx, [15.0, 15.0])
        np.testing.assert_array_equal(ry, [15.0, 15.0])

    def test_separate_x_and_y_apertures_for_rectangular_pipe(self):
        lat = _lattice(_elem("D1", 100.0, aperture=20.0, aperture_y=10.0))
        s, rx, ry = aperture_profile(lat)
        np.testing.assert_array_equal(rx, [20.0, 20.0])
        np.testing.assert_array_equal(ry, [10.0, 10.0])

    def test_aperture_changes_emit_step_transitions(self):
        lat = _lattice(
            _elem("D1", 100.0, aperture=30.0),  # wider before
            _elem("Q1",  50.0, aperture=15.0),  # narrower throat
            _elem("D2", 100.0, aperture=30.0),
        )
        s, rx, ry = aperture_profile(lat)
        # Expect two samples per element: 6 samples total, with the step
        # transitions at s = 100 and s = 150 carrying both old and new
        # aperture values.
        assert s.shape == rx.shape == ry.shape == (6,)
        # Adjacent samples at s=100: rx[1]=30 (D1 exit), rx[2]=15 (Q1 entry)
        np.testing.assert_array_equal(s[1:3],  [100.0, 100.0])
        np.testing.assert_array_equal(rx[1:3], [30.0,  15.0])
        # And at s=150
        np.testing.assert_array_equal(s[3:5],  [150.0, 150.0])
        np.testing.assert_array_equal(rx[3:5], [15.0,  30.0])


class TestSkipBehaviour:
    def test_zero_aperture_skipped(self):
        # Zero/negative aperture means "unknown / no pipe" — ignored.
        lat = _lattice(
            _elem("D1", 100.0, aperture=0.0),     # skipped
            _elem("D2", 100.0, aperture=20.0),    # kept
            _elem("D3", 100.0, aperture=-1.0),    # skipped
        )
        s, rx, ry = aperture_profile(lat)
        assert s.tolist() == [100.0, 200.0]
        assert rx.tolist() == [20.0, 20.0]

    def test_nan_aperture_skipped(self):
        lat = _lattice(_elem("D1", 100.0, aperture=float("nan")),
                       _elem("D2", 100.0, aperture=20.0))
        s, rx, ry = aperture_profile(lat)
        assert s.size == 2

    def test_zero_length_marker_emits_a_single_spike(self):
        lat = _lattice(
            _elem("D1", 100.0, aperture=20.0),
            _elem("M",    0.0, aperture=15.0),    # marker with aperture
            _elem("D2", 100.0, aperture=20.0),
        )
        s, rx, ry = aperture_profile(lat)
        # Marker contributes a single (s=100, rx=15) sample.
        assert 15.0 in rx.tolist()

    def test_empty_lattice_returns_empty_arrays(self):
        lat = SimpleNamespace(elements=[])
        s, rx, ry = aperture_profile(lat)
        assert s.size == 0 and rx.size == 0 and ry.size == 0

    def test_lattice_without_elements_attr_returns_empty(self):
        lat = SimpleNamespace()       # no .elements at all
        s, rx, ry = aperture_profile(lat)
        assert s.size == 0


class TestFieldMapProfile:
    def test_pipe_radius_profile_is_unpacked_at_correct_offset(self):
        # FieldMap with a per-z profile from a .ouv: emit samples at
        # cursor + z, not at the element endpoints alone.
        z   = np.array([0.0, 50.0, 100.0])     # mm
        rxp = np.array([5.0,  3.0,   5.0])
        ryp = np.array([5.0,  3.0,   5.0])
        fd = SimpleNamespace(pipe_radius_profile=(z, rxp, ryp))
        lat = _lattice(
            _elem("D1", 200.0, aperture=20.0),
            _elem("FM", 100.0, aperture=10.0, field_data=fd),
            _elem("D2", 100.0, aperture=20.0),
        )
        s, rx, ry = aperture_profile(lat)
        # D1 → 2 samples, FM unpacks to 3 samples shifted by 200, D2 → 2.
        assert s.size == 7
        # The FM block sits in the middle: indices 2, 3, 4.  At s=200,
        # 250, 300 with rx 5, 3, 5.
        np.testing.assert_array_equal(s[2:5],  [200.0, 250.0, 300.0])
        np.testing.assert_array_equal(rx[2:5], [5.0,   3.0,   5.0])
        np.testing.assert_array_equal(ry[2:5], [5.0,   3.0,   5.0])

    def test_short_or_malformed_profile_falls_back_to_constant(self):
        # Profile with only one z sample is unusable — the writer should
        # ignore it and use the element's constant aperture.
        fd = SimpleNamespace(pipe_radius_profile=(np.array([0.0]),
                                                  np.array([5.0]),
                                                  np.array([5.0])))
        lat = _lattice(_elem("FM", 100.0, aperture=10.0, field_data=fd))
        s, rx, ry = aperture_profile(lat)
        # Falls back: 2 endpoints at 10 mm.
        np.testing.assert_array_equal(rx, [10.0, 10.0])
