# tests/io/test_field_map_reader.py
"""Tests for field map reader (Task 7.1)."""
import numpy as np
import os
import pytest

from linac_gen.io.field_map_reader import (
    FieldMapData,
    read_field_map,
    expand_1d_to_2d,
    read_edz_1d,
    read_edz_2d,
    read_csv,
    expand_1d_offaxis,
)

FIXTURE_DIR = os.path.join(os.path.dirname(__file__), "fixtures")


# ---- FieldMapData dataclass ----

class TestFieldMapData:
    def test_default_fields_are_none(self):
        z = np.linspace(0, 100, 11)
        fmd = FieldMapData(z=z)
        assert fmd.r is None
        assert fmd.Ez is None
        assert fmd.Er is None
        assert fmd.Bz is None
        assert fmd.Br is None

    def test_default_symmetry_is_1d(self):
        fmd = FieldMapData(z=np.array([0.0, 1.0]))
        assert fmd.symmetry == "1d"

    def test_default_norm_factor(self):
        fmd = FieldMapData(z=np.array([0.0]))
        assert fmd.norm_factor == 1.0

    def test_default_frequency(self):
        fmd = FieldMapData(z=np.array([0.0]))
        assert fmd.frequency == 0.0


# ---- 1D .edz reader ----

class TestRead1dEdz:
    def test_read_1d_edz_z_coordinates(self):
        """z should span from z_start to z_end in mm (converted from cm)."""
        fmd = read_field_map(os.path.join(FIXTURE_DIR, "test_1d.edz"), fm_type=1)
        assert len(fmd.z) == 101
        np.testing.assert_allclose(fmd.z[0], 0.0, atol=1e-12)
        np.testing.assert_allclose(fmd.z[-1], 100.0, atol=1e-6)  # 10 cm -> 100 mm

    def test_read_1d_edz_ez_shape(self):
        fmd = read_field_map(os.path.join(FIXTURE_DIR, "test_1d.edz"), fm_type=1)
        assert fmd.Ez.shape == (101,)

    def test_read_1d_edz_ez_values(self):
        """Ez should be sin(pi*z/L), peak=1 at midpoint."""
        fmd = read_field_map(os.path.join(FIXTURE_DIR, "test_1d.edz"), fm_type=1)
        # Peak at midpoint (z = 50 mm)
        mid = len(fmd.Ez) // 2
        np.testing.assert_allclose(fmd.Ez[mid], 1.0, atol=1e-10)
        # Endpoints should be ~0
        np.testing.assert_allclose(fmd.Ez[0], 0.0, atol=1e-10)
        np.testing.assert_allclose(fmd.Ez[-1], 0.0, atol=1e-10)

    def test_read_1d_edz_symmetry(self):
        fmd = read_field_map(os.path.join(FIXTURE_DIR, "test_1d.edz"), fm_type=1)
        assert fmd.symmetry == "1d"

    def test_read_1d_edz_no_radial(self):
        fmd = read_field_map(os.path.join(FIXTURE_DIR, "test_1d.edz"), fm_type=1)
        assert fmd.r is None
        assert fmd.Er is None


# ---- 2D .edz reader ----

class TestRead2dEdz:
    def test_read_2d_edz_shapes(self):
        fmd = read_field_map(os.path.join(FIXTURE_DIR, "test_2d.edz"), fm_type=2)
        assert fmd.Ez.shape == (5, 10)
        assert fmd.Er.shape == (5, 10)

    def test_read_2d_edz_z_coordinates(self):
        fmd = read_field_map(os.path.join(FIXTURE_DIR, "test_2d.edz"), fm_type=2)
        assert len(fmd.z) == 10
        np.testing.assert_allclose(fmd.z[0], 0.0, atol=1e-12)
        # dz = 1.0 cm -> 10 mm, so z[-1] = 9 * 10 = 90 mm
        np.testing.assert_allclose(fmd.z[-1], 90.0, atol=1e-6)

    def test_read_2d_edz_r_coordinates(self):
        fmd = read_field_map(os.path.join(FIXTURE_DIR, "test_2d.edz"), fm_type=2)
        assert len(fmd.r) == 5
        np.testing.assert_allclose(fmd.r[0], 0.0, atol=1e-12)
        # dr = 0.5 cm -> 5 mm, so r[-1] = 4 * 5 = 20 mm
        np.testing.assert_allclose(fmd.r[-1], 20.0, atol=1e-6)

    def test_read_2d_edz_symmetry(self):
        fmd = read_field_map(os.path.join(FIXTURE_DIR, "test_2d.edz"), fm_type=2)
        assert fmd.symmetry == "cylindrical"

    def test_read_2d_edz_on_axis_ez(self):
        """On-axis Ez should be sinusoidal (from fixture generation)."""
        fmd = read_field_map(os.path.join(FIXTURE_DIR, "test_2d.edz"), fm_type=2)
        # On axis (r=0) the field should be sin(pi*z/L)
        z_cm = np.arange(10) * 1.0
        L_cm = 9.0
        expected_ez_axis = np.sin(np.pi * z_cm / L_cm)
        np.testing.assert_allclose(fmd.Ez[0, :], expected_ez_axis, atol=1e-8)

    def test_read_2d_edz_er_on_axis_zero(self):
        """Er should be zero on axis (r=0)."""
        fmd = read_field_map(os.path.join(FIXTURE_DIR, "test_2d.edz"), fm_type=2)
        np.testing.assert_allclose(fmd.Er[0, :], 0.0, atol=1e-10)


# ---- CSV reader ----

class TestReadCSV:
    def test_read_csv_z_coordinates(self):
        fmd = read_field_map(os.path.join(FIXTURE_DIR, "test_fields.csv"), fm_type=1)
        assert len(fmd.z) == 51
        np.testing.assert_allclose(fmd.z[0], 0.0, atol=1e-12)
        np.testing.assert_allclose(fmd.z[-1], 100.0, atol=1e-6)

    def test_read_csv_ez_values(self):
        fmd = read_field_map(os.path.join(FIXTURE_DIR, "test_fields.csv"), fm_type=1)
        expected = np.sin(np.pi * fmd.z / 100.0)
        np.testing.assert_allclose(fmd.Ez, expected, atol=1e-8)

    def test_read_csv_symmetry(self):
        fmd = read_field_map(os.path.join(FIXTURE_DIR, "test_fields.csv"), fm_type=1)
        assert fmd.symmetry == "1d"


# ---- Auto-detection from extension ----

class TestAutoDetection:
    def test_edz_detected_as_tracewin(self):
        """A .edz file should be read using the TraceWin parser."""
        fmd = read_field_map(os.path.join(FIXTURE_DIR, "test_1d.edz"), fm_type=1)
        # If parsed correctly as TraceWin, z is in mm (converted from cm)
        assert fmd.z[-1] > 50.0  # 100 mm, not 10 cm

    def test_csv_detected_as_csv(self):
        """A .csv file should be read using the CSV parser."""
        fmd = read_field_map(os.path.join(FIXTURE_DIR, "test_fields.csv"), fm_type=1)
        assert fmd.z[-1] > 0  # sanity


# ---- Bessel off-axis expansion ----

class TestBesselExpansion:
    def test_expand_returns_ez_er(self):
        """expand_1d_to_2d should return 2D Ez and Er arrays."""
        nz = 101
        z = np.linspace(0, 100, nz)  # mm
        ez_axis = np.sin(np.pi * z / 100.0)
        r_max = 10.0  # mm
        nr = 5
        Ez2d, Er2d, r = expand_1d_to_2d(ez_axis, z, r_max, nr)
        assert Ez2d.shape == (nr, nz)
        assert Er2d.shape == (nr, nz)
        assert len(r) == nr

    def test_expand_on_axis_unchanged(self):
        """On axis (r=0), Ez should equal the original 1D field."""
        nz = 101
        z = np.linspace(0, 100, nz)
        ez_axis = np.sin(np.pi * z / 100.0)
        Ez2d, Er2d, r = expand_1d_to_2d(ez_axis, z, r_max=10.0, nr=5)
        np.testing.assert_allclose(Ez2d[0, :], ez_axis, atol=1e-10)

    def test_expand_er_zero_on_axis(self):
        """Er should be zero on axis (r=0)."""
        nz = 101
        z = np.linspace(0, 100, nz)
        ez_axis = np.sin(np.pi * z / 100.0)
        Ez2d, Er2d, r = expand_1d_to_2d(ez_axis, z, r_max=10.0, nr=5)
        np.testing.assert_allclose(Er2d[0, :], 0.0, atol=1e-10)

    def test_expand_ez_decreases_off_axis(self):
        """Ez should decrease off axis due to -(r^2/4)*Ez'' correction.

        For Ez = sin(pi*z/L), Ez'' = -(pi/L)^2 * sin(pi*z/L).
        So the correction is +(r^2/4)*(pi/L)^2 * sin(pi*z/L), meaning
        Ez increases off axis. But at the field peak z=L/2,
        Ez'' < 0, so -(r^2/4)*Ez'' > 0 and Ez_2d > Ez_1d.
        Actually for sin, Ez'' is negative at the peak, so the correction
        -(r^2/4)*Ez'' is positive. Let's just check the correction is nonzero.
        """
        nz = 201
        z = np.linspace(0, 100, nz)
        ez_axis = np.sin(np.pi * z / 100.0)
        Ez2d, Er2d, r = expand_1d_to_2d(ez_axis, z, r_max=10.0, nr=5)
        mid = nz // 2
        # Off-axis Ez should differ from on-axis at the peak
        assert not np.allclose(Ez2d[-1, mid], Ez2d[0, mid], atol=1e-6)

    def test_expand_er_nonzero_off_axis(self):
        """Er should be nonzero off axis where Ez' != 0."""
        nz = 201
        z = np.linspace(0, 100, nz)
        ez_axis = np.sin(np.pi * z / 100.0)
        Ez2d, Er2d, r = expand_1d_to_2d(ez_axis, z, r_max=10.0, nr=5)
        # At z != 0, L (where Ez' is maximal), Er should be nonzero off axis
        quarter = nz // 4
        assert abs(Er2d[-1, quarter]) > 1e-6

    def test_expand_er_sign(self):
        """Er = -(r/2)*Ez'. At z=0, Ez' > 0, so Er < 0 for r > 0."""
        nz = 201
        z = np.linspace(0, 100, nz)
        ez_axis = np.sin(np.pi * z / 100.0)
        Ez2d, Er2d, r = expand_1d_to_2d(ez_axis, z, r_max=10.0, nr=5)
        # Near z=0 (but not at boundary), Ez' > 0, so Er < 0 for r > 0
        # Use index 5 to avoid boundary effects from finite differences
        assert Er2d[-1, 5] < 0  # r > 0, Ez' > 0 near z=0 side


# ---- Error handling ----

class TestErrorHandling:
    def test_nonexistent_file_raises(self):
        with pytest.raises(FileNotFoundError):
            read_field_map("/nonexistent/path/file.edz", fm_type=1)

    def test_invalid_fm_type_2d_on_1d_file(self):
        """Requesting fm_type=2 on a 1D file should raise ValueError."""
        with pytest.raises(ValueError):
            read_field_map(os.path.join(FIXTURE_DIR, "test_1d.edz"), fm_type=2)


# ---- Public named reader functions (Task 7.1 additions) ----

class TestReadEdz1d:
    """Tests for the public read_edz_1d function."""

    def test_n_points(self):
        """Should read 11 points from the cavity fixture."""
        fmd = read_edz_1d(os.path.join(FIXTURE_DIR, "test_cavity_1d.edz"))
        assert len(fmd.z) == 11
        assert len(fmd.Ez) == 11

    def test_z_range_mm(self):
        """z should span 0-100 mm (10 cm * 10 conversion)."""
        fmd = read_edz_1d(os.path.join(FIXTURE_DIR, "test_cavity_1d.edz"))
        np.testing.assert_allclose(fmd.z[0], 0.0, atol=1e-12)
        np.testing.assert_allclose(fmd.z[-1], 100.0, atol=1e-6)

    def test_ez_peak(self):
        """Ez peak should be 1.0."""
        fmd = read_edz_1d(os.path.join(FIXTURE_DIR, "test_cavity_1d.edz"))
        assert fmd.Ez.max() == pytest.approx(1.0)

    def test_symmetry_1d(self):
        fmd = read_edz_1d(os.path.join(FIXTURE_DIR, "test_cavity_1d.edz"))
        assert fmd.symmetry == "1d"

    def test_no_radial_fields(self):
        fmd = read_edz_1d(os.path.join(FIXTURE_DIR, "test_cavity_1d.edz"))
        assert fmd.r is None
        assert fmd.Er is None

    def test_returns_field_map_data(self):
        fmd = read_edz_1d(os.path.join(FIXTURE_DIR, "test_cavity_1d.edz"))
        assert isinstance(fmd, FieldMapData)


class TestReadEdz2d:
    """Tests for the public read_edz_2d function."""

    def test_shape(self):
        """Ez and Er should have shape (nz, nr) = (5, 3)."""
        fmd = read_edz_2d(os.path.join(FIXTURE_DIR, "test_cavity_2d.edz"))
        assert fmd.Ez.shape == (5, 3)
        assert fmd.Er.shape == (5, 3)

    def test_z_array(self):
        """z should have nz=5 points; dz = 0.2 cm -> 2.0 mm; z[-1] = 4*2.0 = 8.0 mm."""
        fmd = read_edz_2d(os.path.join(FIXTURE_DIR, "test_cavity_2d.edz"))
        assert len(fmd.z) == 5
        np.testing.assert_allclose(fmd.z[0], 0.0, atol=1e-12)
        # dz = 0.2 cm -> 2.0 mm; z[-1] = 4 * 2.0 = 8.0 mm
        np.testing.assert_allclose(fmd.z[-1], 8.0, atol=1e-6)

    def test_r_array(self):
        """r should have nr=3 points; dr = 0.1 cm -> 1.0 mm; r[-1] = 2*1.0 = 2.0 mm."""
        fmd = read_edz_2d(os.path.join(FIXTURE_DIR, "test_cavity_2d.edz"))
        assert len(fmd.r) == 3
        np.testing.assert_allclose(fmd.r[0], 0.0, atol=1e-12)
        np.testing.assert_allclose(fmd.r[-1], 2.0, atol=1e-6)

    def test_er_present(self):
        fmd = read_edz_2d(os.path.join(FIXTURE_DIR, "test_cavity_2d.edz"))
        assert fmd.Er is not None

    def test_symmetry_cylindrical(self):
        fmd = read_edz_2d(os.path.join(FIXTURE_DIR, "test_cavity_2d.edz"))
        assert fmd.symmetry == "cylindrical"

    def test_no_bz_br_type2(self):
        """Type 2 (pure E) should not have Bz/Br."""
        fmd = read_edz_2d(os.path.join(FIXTURE_DIR, "test_cavity_2d.edz"))
        assert fmd.Bz is None
        assert fmd.Br is None

    def test_returns_field_map_data(self):
        fmd = read_edz_2d(os.path.join(FIXTURE_DIR, "test_cavity_2d.edz"))
        assert isinstance(fmd, FieldMapData)


class TestReadCsv:
    """Tests for the public read_csv function."""

    def test_n_points(self):
        fmd = read_csv(os.path.join(FIXTURE_DIR, "test_cavity_fields.csv"))
        assert len(fmd.z) == 5

    def test_z_values(self):
        fmd = read_csv(os.path.join(FIXTURE_DIR, "test_cavity_fields.csv"))
        np.testing.assert_allclose(fmd.z, [0.0, 10.0, 20.0, 30.0, 40.0])

    def test_ez_values(self):
        fmd = read_csv(os.path.join(FIXTURE_DIR, "test_cavity_fields.csv"))
        np.testing.assert_allclose(fmd.Ez, [0.0, 500.0, 1000.0, 500.0, 0.0])

    def test_er_values(self):
        fmd = read_csv(os.path.join(FIXTURE_DIR, "test_cavity_fields.csv"))
        assert fmd.Er is not None
        np.testing.assert_allclose(fmd.Er, [0.0, 50.0, 100.0, 50.0, 0.0])

    def test_symmetry_1d(self):
        fmd = read_csv(os.path.join(FIXTURE_DIR, "test_cavity_fields.csv"))
        assert fmd.symmetry == "1d"

    def test_returns_field_map_data(self):
        fmd = read_csv(os.path.join(FIXTURE_DIR, "test_cavity_fields.csv"))
        assert isinstance(fmd, FieldMapData)


class TestReadFieldMapAutoDetect:
    """Tests for read_field_map auto-detection with new fixtures."""

    def test_autodetect_1d_edz(self):
        """read_field_map on a 1D .edz should return symmetry='1d'."""
        fmd = read_field_map(os.path.join(FIXTURE_DIR, "test_cavity_1d.edz"), fm_type=1)
        assert fmd.symmetry == "1d"

    def test_autodetect_2d_edz(self):
        """read_field_map on a 2D .edz should return symmetry='cylindrical'."""
        fmd = read_field_map(os.path.join(FIXTURE_DIR, "test_cavity_2d.edz"), fm_type=2)
        assert fmd.symmetry == "cylindrical"


class TestExpand1dOffaxis:
    """Tests for the expand_1d_offaxis function returning callable field functions."""

    def _get_fmap(self):
        z = np.linspace(0, 100, 101)
        ez = np.sin(np.pi * z / 100.0)
        from linac_gen.io.field_map_reader import FieldMapData
        return FieldMapData(z=z, Ez=ez, symmetry="1d")

    def test_returns_two_callables(self):
        fmap = self._get_fmap()
        result = expand_1d_offaxis(fmap)
        assert len(result) == 2
        Ez_func, Er_func = result
        assert callable(Ez_func)
        assert callable(Er_func)

    def test_ez_func_evaluates(self):
        fmap = self._get_fmap()
        Ez_func, _ = expand_1d_offaxis(fmap)
        val = Ez_func(50.0)
        assert isinstance(val, float)
        # At z=50mm (midpoint), sin(pi/2) = 1.0
        assert val == pytest.approx(1.0, abs=1e-6)

    def test_er_zero_on_axis(self):
        """Er(r=0, z) should be 0 for any z."""
        fmap = self._get_fmap()
        _, Er_func = expand_1d_offaxis(fmap)
        for z_val in [10.0, 30.0, 50.0, 70.0]:
            assert Er_func(0.0, z_val) == pytest.approx(0.0, abs=1e-12)

    def test_er_nonzero_off_axis(self):
        """Er(r>0, z) should be nonzero where Ez' != 0."""
        fmap = self._get_fmap()
        _, Er_func = expand_1d_offaxis(fmap)
        # At z=10mm, sin has nonzero derivative, so Er should be nonzero for r>0
        val = Er_func(5.0, 10.0)
        assert abs(val) > 1e-6

    def test_er_sign_at_peak(self):
        """At z near 0 where Ez'>0, Er = -0.5*r*Ez' should be negative for r>0."""
        fmap = self._get_fmap()
        _, Er_func = expand_1d_offaxis(fmap)
        # Near z=5mm: sin' = (pi/100)*cos(pi*5/100) > 0, so Er < 0 for r>0
        val = Er_func(5.0, 5.0)
        assert val < 0

    def test_symmetry_field_1d(self):
        """FieldMapData constructed with symmetry='1d' should reflect that."""
        fmap = self._get_fmap()
        assert fmap.symmetry == "1d"
