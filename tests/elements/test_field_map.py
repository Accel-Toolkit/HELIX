# tests/elements/test_field_map.py
"""Tests for FieldMap element with RK4 tracking (Task 7.2)."""
import numpy as np
import math
import pytest

from linac_gen.core.particle import PROTON, H_MINUS
from linac_gen.core.reference import ReferenceParticle
from linac_gen.core.beam import Beam
from linac_gen.elements.base import FieldMapElement as FieldMapBase
from linac_gen.elements.field_map import FieldMap
from linac_gen.io.field_map_reader import FieldMapData


def _make_ref(species=PROTON, w_kin=3.0, freq=352.21, phi_s=0.0):
    return ReferenceParticle(species=species, w_kin=w_kin, frequency=freq, phi_s=phi_s)


def _make_beam(n=10, species=PROTON, w_kin=3.0, freq=352.21, phi_s=0.0):
    ref = _make_ref(species=species, w_kin=w_kin, freq=freq, phi_s=phi_s)
    return Beam(ref=ref, n_particles=n, current=0.0)


def _zero_field(length_mm=100.0, nz=101):
    """Create a zero-field map (pure drift case)."""
    z = np.linspace(0, length_mm, nz)
    Ez = np.zeros(nz)
    return FieldMapData(z=z, Ez=Ez, symmetry="1d")


def _uniform_field(length_mm=100.0, nz=101, ez_val=1.0):
    """Create a uniform Ez field map (constant field along z)."""
    z = np.linspace(0, length_mm, nz)
    Ez = np.full(nz, ez_val)
    return FieldMapData(z=z, Ez=Ez, symmetry="1d")


def _sinusoidal_field(length_mm=100.0, nz=101):
    """Create a sinusoidal Ez field map."""
    z = np.linspace(0, length_mm, nz)
    Ez = np.sin(np.pi * z / length_mm)
    return FieldMapData(z=z, Ez=Ez, symmetry="1d")


# ---- Instance type tests ----

class TestFieldMapBasics:
    def test_is_field_map_element(self):
        fd = _zero_field()
        fm = FieldMap("FM1", length=100.0, field_data=fd)
        assert isinstance(fm, FieldMapBase)

    def test_attributes(self):
        fd = _zero_field()
        fm = FieldMap("FM1", length=100.0, field_data=fd,
                      scale=2.0, phase=-30.0, frequency=352.21,
                      aperture=15.0, n_steps=50)
        assert fm.name == "FM1"
        assert fm.length == 100.0
        assert fm.scale == 2.0
        assert fm.phase == -30.0
        assert fm.frequency == 352.21
        assert fm.aperture == 15.0
        assert fm.n_steps == 50

    def test_default_attributes(self):
        fd = _zero_field()
        fm = FieldMap("FM1", length=100.0, field_data=fd)
        assert fm.scale == 1.0
        assert fm.phase == 0.0
        assert fm.frequency == 0.0
        assert fm.aperture == 0.0
        assert fm.n_steps == 100


# ---- Zero field (pure drift) tests ----

class TestZeroFieldDrift:
    def test_zero_field_no_energy_gain(self):
        """Through a zero field, reference energy should not change."""
        fd = _zero_field(length_mm=100.0)
        fm = FieldMap("FM1", length=100.0, field_data=fd, n_steps=10)
        beam = _make_beam(n=1, w_kin=3.0)
        w_before = beam.ref.w_kin
        ds = fm.length / fm.n_steps
        for _ in range(fm.n_steps):
            fm.track_rk4(beam, ds)
        np.testing.assert_allclose(beam.ref.w_kin, w_before, atol=1e-14)

    def test_zero_field_particle_dw_unchanged(self):
        """Particle energy deviation should not change in zero field."""
        fd = _zero_field(length_mm=100.0)
        fm = FieldMap("FM1", length=100.0, field_data=fd, n_steps=10)
        beam = _make_beam(n=3, w_kin=3.0)
        beam.particles[0, 5] = 0.1  # dW = 0.1 MeV
        ds = fm.length / fm.n_steps
        for _ in range(fm.n_steps):
            fm.track_rk4(beam, ds)
        np.testing.assert_allclose(beam.particles[0, 5], 0.1, atol=1e-14)

    def test_zero_field_transverse_drift(self):
        """In zero field, x should evolve as x += xp * ds (drift-like)."""
        fd = _zero_field(length_mm=100.0)
        fm = FieldMap("FM1", length=100.0, field_data=fd, n_steps=100)
        beam = _make_beam(n=1, w_kin=3.0)
        beam.particles[0, 0] = 0.0   # x = 0
        beam.particles[0, 1] = 10.0  # xp = 10 mrad
        ds = fm.length / fm.n_steps
        for _ in range(fm.n_steps):
            fm.track_rk4(beam, ds)
        # x should be approximately xp * L in mm*mrad units
        # x(mm) += xp(mrad) * ds(mm) * 1e-3 = 10 * 100 * 1e-3 = 1.0 mm
        expected_x = 10.0 * 100.0 * 1e-3
        np.testing.assert_allclose(beam.particles[0, 0], expected_x, rtol=1e-3)

    def test_zero_field_s_advances(self):
        """Reference s-position should advance by total length."""
        fd = _zero_field(length_mm=100.0)
        fm = FieldMap("FM1", length=100.0, field_data=fd, n_steps=10)
        beam = _make_beam(n=1, w_kin=3.0)
        s_before = beam.ref.s
        ds = fm.length / fm.n_steps
        for _ in range(fm.n_steps):
            fm.track_rk4(beam, ds)
        np.testing.assert_allclose(beam.ref.s, s_before + 100.0, atol=1e-10)


# ---- Uniform field energy gain ----

class TestUniformField:
    def test_uniform_field_energy_gain(self):
        """Uniform Ez on crest should give energy change (phase-dependent).

        dW_per_step = q * Ez * scale * cos(phi_total) * ds * 1e-3  (MeV)
        With phase=0 and phi_s starting at 0, the total phase evolves and
        cos(phi_total) oscillates, so net energy change may be positive or
        negative. We just verify the energy actually changed (not zero).
        """
        length = 100.0  # mm
        fd = _uniform_field(length_mm=length, ez_val=1.0)
        fm = FieldMap("FM1", length=length, field_data=fd,
                      scale=1.0, phase=0.0, n_steps=100)
        beam = _make_beam(n=1, w_kin=10.0, phi_s=0.0)
        w_before = beam.ref.w_kin
        ds = fm.length / fm.n_steps
        for _ in range(fm.n_steps):
            fm.track_rk4(beam, ds)
        # Energy should have changed (not be exactly the same as before)
        assert abs(beam.ref.w_kin - w_before) > 1e-6

    def test_uniform_field_positive_energy(self):
        """With positive charge, positive Ez, phi=0: energy should increase."""
        fd = _uniform_field(length_mm=50.0, ez_val=1.0)
        fm = FieldMap("FM1", length=50.0, field_data=fd,
                      scale=0.5, phase=0.0, n_steps=50)
        beam = _make_beam(n=1, w_kin=10.0, phi_s=0.0)
        w_before = beam.ref.w_kin
        ds = fm.length / fm.n_steps
        for _ in range(fm.n_steps):
            fm.track_rk4(beam, ds)
        assert beam.ref.w_kin > w_before

    def test_h_minus_phase_180_accelerates(self):
        """H- at phase=180 should gain energy (charge*cos(180) < 0, but
        cos(phi_total) evolves)."""
        fd = _uniform_field(length_mm=50.0, ez_val=1.0)
        fm = FieldMap("FM1", length=50.0, field_data=fd,
                      scale=0.5, phase=180.0, n_steps=50)
        beam = _make_beam(n=1, species=H_MINUS, w_kin=10.0, phi_s=0.0)
        w_before = beam.ref.w_kin
        ds = fm.length / fm.n_steps
        for _ in range(fm.n_steps):
            fm.track_rk4(beam, ds)
        # charge=-1, cos(180+phi_s) at start = -1, so dW = -1 * 1 * (-1) * ds > 0
        assert beam.ref.w_kin > w_before


# ---- advance_ref tests ----

class TestAdvanceRef:
    def test_advance_ref_zero_field(self):
        """advance_ref through zero field should not change energy."""
        fd = _zero_field(length_mm=100.0)
        fm = FieldMap("FM1", length=100.0, field_data=fd, n_steps=50)
        ref = _make_ref(w_kin=3.0)
        w_before = ref.w_kin
        fm.advance_ref(ref)
        np.testing.assert_allclose(ref.w_kin, w_before, atol=1e-14)

    def test_advance_ref_updates_s(self):
        """advance_ref should advance s by the full element length."""
        fd = _zero_field(length_mm=100.0)
        fm = FieldMap("FM1", length=100.0, field_data=fd, n_steps=50)
        ref = _make_ref(w_kin=3.0)
        s_before = ref.s
        fm.advance_ref(ref)
        np.testing.assert_allclose(ref.s, s_before + 100.0, atol=1e-10)

    def test_advance_ref_uniform_field_energy(self):
        """advance_ref through uniform field should change energy."""
        fd = _uniform_field(length_mm=100.0, ez_val=1.0)
        fm = FieldMap("FM1", length=100.0, field_data=fd,
                      scale=1.0, phase=0.0, n_steps=100)
        ref = _make_ref(w_kin=10.0, phi_s=0.0)
        w_before = ref.w_kin
        fm.advance_ref(ref)
        # Energy should have changed (phase evolves, so net gain depends on
        # the accumulated phase over the element length)
        assert abs(ref.w_kin - w_before) > 1e-6

    def test_advance_ref_matches_tracking(self):
        """advance_ref should give same ref energy as full tracking."""
        fd = _sinusoidal_field(length_mm=100.0)
        fm = FieldMap("FM1", length=100.0, field_data=fd,
                      scale=0.5, phase=-30.0, n_steps=100)
        # Track beam
        beam = _make_beam(n=1, w_kin=10.0, phi_s=-10.0)
        w_before_track = beam.ref.w_kin
        ds = fm.length / fm.n_steps
        for _ in range(fm.n_steps):
            fm.track_rk4(beam, ds)
        w_after_track = beam.ref.w_kin

        # advance_ref only
        fm2 = FieldMap("FM2", length=100.0, field_data=fd,
                       scale=0.5, phase=-30.0, n_steps=100)
        ref = _make_ref(w_kin=10.0, phi_s=-10.0)
        fm2.advance_ref(ref)
        np.testing.assert_allclose(ref.w_kin, w_after_track, rtol=1e-10)

    def test_advance_ref_updates_frequency(self):
        """If element has nonzero frequency, ref frequency should be updated."""
        fd = _zero_field(length_mm=100.0)
        fm = FieldMap("FM1", length=100.0, field_data=fd,
                      frequency=704.42, n_steps=10)
        ref = _make_ref(w_kin=3.0, freq=352.21)
        fm.advance_ref(ref)
        assert ref.frequency == 704.42


# ---- fitted_matrix tests ----

class TestFittedMatrix:
    def test_fitted_matrix_zero_field_is_drift(self):
        """For zero field, fitted_matrix should approximate a drift matrix."""
        fd = _zero_field(length_mm=100.0)
        fm = FieldMap("FM1", length=100.0, field_data=fd, n_steps=50)
        ref = _make_ref(w_kin=10.0)
        M = fm.fitted_matrix(ref)
        assert M.shape == (6, 6)
        # For a drift, M[0,1] = L*1e-3 = 0.1
        np.testing.assert_allclose(M[0, 1], 0.1, atol=0.01)
        np.testing.assert_allclose(M[2, 3], 0.1, atol=0.01)

    def test_fitted_matrix_shape(self):
        fd = _sinusoidal_field()
        fm = FieldMap("FM1", length=100.0, field_data=fd,
                      scale=1.0, phase=-30.0, n_steps=50)
        ref = _make_ref(w_kin=10.0)
        M = fm.fitted_matrix(ref)
        assert M.shape == (6, 6)

    def test_fitted_matrix_nonzero_field_not_identity(self):
        """With a real field, the matrix should differ from identity."""
        fd = _sinusoidal_field()
        fm = FieldMap("FM1", length=100.0, field_data=fd,
                      scale=1.0, phase=-30.0, n_steps=50)
        ref = _make_ref(w_kin=10.0, phi_s=-10.0)
        M = fm.fitted_matrix(ref)
        # Should not be identity
        assert not np.allclose(M, np.eye(6), atol=1e-4)

    def test_fitted_matrix_does_not_modify_ref(self):
        """fitted_matrix should not change the reference particle."""
        fd = _sinusoidal_field()
        fm = FieldMap("FM1", length=100.0, field_data=fd,
                      scale=1.0, phase=-30.0, n_steps=50)
        ref = _make_ref(w_kin=10.0, phi_s=-10.0)
        w_before = ref.w_kin
        s_before = ref.s
        phi_before = ref.phi_s
        _ = fm.fitted_matrix(ref)
        assert ref.w_kin == w_before
        assert ref.s == s_before
        assert ref.phi_s == phi_before


# ---- Interpolation tests ----

class TestInterpolation:
    def test_interpolate_ez_at_grid_points(self):
        """Interpolated Ez at grid points should match exactly."""
        fd = _sinusoidal_field(length_mm=100.0, nz=101)
        fm = FieldMap("FM1", length=100.0, field_data=fd, n_steps=50)
        # Midpoint
        z_mid = 50.0
        ez_interp = fm._interpolate_Ez(z_mid)
        expected = np.sin(np.pi * z_mid / 100.0)
        np.testing.assert_allclose(ez_interp, expected, atol=1e-6)

    def test_interpolate_ez_between_grid_points(self):
        """Interpolation between grid points should be reasonably accurate."""
        fd = _sinusoidal_field(length_mm=100.0, nz=101)
        fm = FieldMap("FM1", length=100.0, field_data=fd, n_steps=50)
        z_test = 25.5  # between grid points
        ez_interp = fm._interpolate_Ez(z_test)
        expected = np.sin(np.pi * z_test / 100.0)
        np.testing.assert_allclose(ez_interp, expected, atol=0.01)

    def test_interpolate_ez_at_boundaries(self):
        """Interpolation at boundaries should work without error."""
        fd = _sinusoidal_field(length_mm=100.0, nz=101)
        fm = FieldMap("FM1", length=100.0, field_data=fd, n_steps=50)
        ez0 = fm._interpolate_Ez(0.0)
        ez_end = fm._interpolate_Ez(100.0)
        np.testing.assert_allclose(ez0, 0.0, atol=1e-10)
        np.testing.assert_allclose(ez_end, 0.0, atol=1e-10)


# ---- Phase advance tests ----

class TestPhaseAdvance:
    def test_phase_advances_through_element(self):
        """Reference phi_s should advance through the element."""
        fd = _zero_field(length_mm=100.0)
        fm = FieldMap("FM1", length=100.0, field_data=fd, n_steps=10)
        beam = _make_beam(n=1, w_kin=3.0, phi_s=0.0)
        phi_before = beam.ref.phi_s
        ds = fm.length / fm.n_steps
        for _ in range(fm.n_steps):
            fm.track_rk4(beam, ds)
        assert beam.ref.phi_s > phi_before

    def test_phase_advance_consistent(self):
        """Phase advance through zero-field element should match drift formula."""
        fd = _zero_field(length_mm=100.0)
        fm = FieldMap("FM1", length=100.0, field_data=fd, n_steps=100)
        ref = _make_ref(w_kin=3.0, phi_s=0.0)
        beta = ref.beta
        wl = ref.wavelength
        expected_dphi = 360.0 * 100.0 / (beta * wl)
        fm.advance_ref(ref)
        np.testing.assert_allclose(ref.phi_s, expected_dphi, rtol=1e-6)


# ---- Lost particles ----

class TestLostParticles:
    def test_lost_particles_not_tracked(self):
        """Lost particles should not be affected by tracking."""
        fd = _uniform_field(length_mm=100.0, ez_val=1.0)
        fm = FieldMap("FM1", length=100.0, field_data=fd,
                      scale=1.0, phase=0.0, n_steps=10)
        beam = _make_beam(n=3, w_kin=10.0, phi_s=0.0)
        beam.particles[0, 0] = 5.0
        beam.particles[0, 1] = 10.0
        beam.record_loss(0, 0.0, "test")
        x_before = beam.particles[0, 0]
        xp_before = beam.particles[0, 1]
        dw_before = beam.particles[0, 5]
        ds = fm.length / fm.n_steps
        for _ in range(fm.n_steps):
            fm.track_rk4(beam, ds)
        assert beam.particles[0, 0] == x_before
        assert beam.particles[0, 1] == xp_before
        assert beam.particles[0, 5] == dw_before


# ---- Integration with Tracker ----

class TestTrackerIntegration:
    def test_tracker_dispatches_field_map(self):
        """The Tracker should dispatch FieldMap elements correctly."""
        from linac_gen.core.lattice import Lattice
        from linac_gen.tracking.tracker import Tracker

        fd = _zero_field(length_mm=50.0)
        fm = FieldMap("FM1", length=50.0, field_data=fd, n_steps=10)
        lattice = Lattice()
        lattice.add(fm)
        beam = _make_beam(n=5, w_kin=3.0)
        s_before = beam.ref.s
        tracker = Tracker(lattice, beam)
        tracker.run()
        np.testing.assert_allclose(beam.ref.s, s_before + 50.0, atol=1e-10)


# ---- from_file class method ----

FIXTURE_PATH = "tests/io/fixtures/test_cavity_1d.edz"

import os as _os

_FIXTURE_ABS = _os.path.join(
    _os.path.dirname(_os.path.dirname(__file__)), "io", "fixtures", "test_cavity_1d.edz"
)


class TestFromFile:
    def test_from_file_creates_field_map(self):
        """from_file should return a FieldMap instance."""
        fm = FieldMap.from_file("FM_file", _FIXTURE_ABS)
        assert isinstance(fm, FieldMapBase)

    def test_from_file_infers_length(self):
        """When length=None, length should equal z[-1] - z[0] from the file."""
        fm = FieldMap.from_file("FM_file", _FIXTURE_ABS)
        # test_cavity_1d.edz: z spans 0..100 mm (10 cm converted)
        assert fm.length == pytest.approx(100.0, abs=1e-6)

    def test_from_file_explicit_length(self):
        """Explicit length overrides the file's z-range."""
        fm = FieldMap.from_file("FM_file", _FIXTURE_ABS, length=150.0)
        assert fm.length == 150.0

    def test_from_file_scale_and_phase(self):
        """scale and phase should be stored correctly."""
        fm = FieldMap.from_file("FM_file", _FIXTURE_ABS, scale=2.5, phase=-45.0)
        assert fm.scale == 2.5
        assert fm.phase == -45.0

    def test_from_file_frequency(self):
        """frequency parameter should be stored."""
        fm = FieldMap.from_file("FM_file", _FIXTURE_ABS, frequency=352.21)
        assert fm.frequency == 352.21

    def test_from_file_n_steps(self):
        """n_steps should be passed through correctly."""
        fm = FieldMap.from_file("FM_file", _FIXTURE_ABS, n_steps=200)
        assert fm.n_steps == 200

    def test_from_file_tracks_without_crash(self):
        """A FieldMap loaded from file should track a beam without error."""
        fm = FieldMap.from_file("FM_file", _FIXTURE_ABS, scale=1.0, n_steps=10)
        beam = _make_beam(n=5, w_kin=3.0)
        ds = fm.length / fm.n_steps
        for _ in range(fm.n_steps):
            fm.track_rk4(beam, ds)
        assert beam.ref.s > 0.0

    def test_from_file_missing_path_raises(self):
        """from_file should raise FileNotFoundError for non-existent files."""
        with pytest.raises(FileNotFoundError):
            FieldMap.from_file("FM_bad", "/no/such/file.edz")
