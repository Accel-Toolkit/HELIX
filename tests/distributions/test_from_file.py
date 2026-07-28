"""Tests for the from-file distribution loader."""
import os
import tempfile
import numpy as np
import pytest
from linac_gen.distributions.from_file import load_distribution
from linac_gen.distributions.gaussian import generate_gaussian

FIXTURES_DIR = os.path.join(os.path.dirname(__file__), "fixtures")
FIXTURE_WITH_HEADER = os.path.join(FIXTURES_DIR, "test_distribution_with_header.dat")
FIXTURE_NO_HEADER = os.path.join(FIXTURES_DIR, "test_distribution_no_header.dat")

# Reference values used in the fixture files
W_KIN_REF = 3.0        # MeV
PHI_REF = -30.0        # deg

# Expected absolute coordinates from the fixture
ABS_DATA = np.array([
    [ 0.123, -0.456,  0.789,  1.234, -45.600, 3.0012],
    [-0.567,  0.890, -0.345,  0.678, -38.200, 2.9987],
    [ 1.200,  0.300,  0.100, -0.800, -25.000, 3.0150],
    [-0.900, -0.200,  0.500,  0.400, -32.500, 2.9900],
    [ 0.050,  1.100, -0.700, -0.300, -28.750, 3.0050],
], dtype=np.float64)

# Expected deviations = absolute - ref (columns 4 and 5 only)
EXPECTED_DEVIATIONS = ABS_DATA.copy()
EXPECTED_DEVIATIONS[:, 4] = ABS_DATA[:, 4] - PHI_REF
EXPECTED_DEVIATIONS[:, 5] = ABS_DATA[:, 5] - W_KIN_REF


class TestLoadWithHeader:
    def test_returns_tuple(self):
        result = load_distribution(FIXTURE_WITH_HEADER)
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_particles_shape(self):
        particles, _ = load_distribution(FIXTURE_WITH_HEADER)
        assert particles.shape == (5, 6)

    def test_particles_dtype(self):
        particles, _ = load_distribution(FIXTURE_WITH_HEADER)
        assert particles.dtype == np.float64

    def test_header_dict_has_w_kin_ref(self):
        _, header = load_distribution(FIXTURE_WITH_HEADER)
        assert "w_kin_ref" in header
        assert abs(header["w_kin_ref"] - W_KIN_REF) < 1e-9

    def test_header_dict_has_phi_ref(self):
        _, header = load_distribution(FIXTURE_WITH_HEADER)
        assert "phi_ref" in header
        assert abs(header["phi_ref"] - PHI_REF) < 1e-9

    def test_transverse_coords_unchanged(self):
        """Columns 0-3 (x, xp, y, yp) should be copied verbatim."""
        particles, _ = load_distribution(FIXTURE_WITH_HEADER)
        np.testing.assert_allclose(particles[:, :4], ABS_DATA[:, :4], rtol=1e-10)

    def test_phi_deviation_correct(self):
        """Column 4: dphi = phi_abs - phi_ref."""
        particles, _ = load_distribution(FIXTURE_WITH_HEADER)
        expected_dphi = ABS_DATA[:, 4] - PHI_REF
        np.testing.assert_allclose(particles[:, 4], expected_dphi, rtol=1e-10)

    def test_energy_deviation_correct(self):
        """Column 5: dW = W_abs - w_kin_ref."""
        particles, _ = load_distribution(FIXTURE_WITH_HEADER)
        expected_dw = ABS_DATA[:, 5] - W_KIN_REF
        np.testing.assert_allclose(particles[:, 5], expected_dw, rtol=1e-10)

    def test_full_deviation_array(self):
        particles, _ = load_distribution(FIXTURE_WITH_HEADER)
        np.testing.assert_allclose(particles, EXPECTED_DEVIATIONS, rtol=1e-10)


class TestLoadNoHeader:
    def test_load_without_header_uses_provided_refs(self):
        """Without a header, ref values from arguments should be used."""
        particles, header = load_distribution(
            FIXTURE_NO_HEADER,
            ref_w_kin=W_KIN_REF,
            ref_phi_s=PHI_REF,
        )
        np.testing.assert_allclose(particles, EXPECTED_DEVIATIONS, rtol=1e-10)

    def test_load_without_header_defaults_to_zero(self):
        """Without a header and no ref args, defaults to ref=0 for both."""
        particles, header = load_distribution(FIXTURE_NO_HEADER)
        # With ref=0, deviations = absolute values
        np.testing.assert_allclose(particles[:, :4], ABS_DATA[:, :4], rtol=1e-10)
        np.testing.assert_allclose(particles[:, 4], ABS_DATA[:, 4], rtol=1e-10)
        np.testing.assert_allclose(particles[:, 5], ABS_DATA[:, 5], rtol=1e-10)

    def test_load_without_header_returns_empty_header(self):
        _, header = load_distribution(FIXTURE_NO_HEADER)
        assert isinstance(header, dict)


class TestArgRefOverridesHeader:
    def test_explicit_ref_args_override_file_header(self):
        """Explicit ref_w_kin / ref_phi_s args take precedence over file header."""
        override_w = 2.5
        override_phi = -20.0
        particles, header = load_distribution(
            FIXTURE_WITH_HEADER,
            ref_w_kin=override_w,
            ref_phi_s=override_phi,
        )
        expected_dphi = ABS_DATA[:, 4] - override_phi
        expected_dw = ABS_DATA[:, 5] - override_w
        np.testing.assert_allclose(particles[:, 4], expected_dphi, rtol=1e-10)
        np.testing.assert_allclose(particles[:, 5], expected_dw, rtol=1e-10)


class TestRoundTrip:
    """Generate a Gaussian, export as absolute coordinates, reload, check deviations."""

    def test_round_trip(self):
        n = 200
        emit_x, alpha_x, beta_x = 1.0, 0.5, 2.0
        emit_y, alpha_y, beta_y = 0.8, -0.3, 1.5
        emit_z, alpha_z, beta_z = 0.5, 0.0, 3.0
        ref_w = 3.0
        ref_phi = -30.0

        deviations = generate_gaussian(
            n,
            emit_x, alpha_x, beta_x,
            emit_y, alpha_y, beta_y,
            emit_z, alpha_z, beta_z,
            seed=99,
        )

        # Construct absolute coordinates
        abs_coords = deviations.copy()
        abs_coords[:, 4] += ref_phi
        abs_coords[:, 5] += ref_w

        # Write to a temporary file with header
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".dat", delete=False
        ) as f:
            tmpfile = f.name
            f.write("# Linac_Gen distribution file\n")
            f.write(f"# w_kin_ref: {ref_w:.6f} MeV\n")
            f.write(f"# phi_ref: {ref_phi:.6f} deg\n")
            f.write("# columns: x(mm) xp(mrad) y(mm) yp(mrad) phi_abs(deg) W_abs(MeV)\n")
            for row in abs_coords:
                f.write(f"{row[0]:12.6f} {row[1]:12.6f} {row[2]:12.6f} "
                        f"{row[3]:12.6f} {row[4]:12.6f} {row[5]:12.6f}\n")

        try:
            loaded, header = load_distribution(tmpfile)
            # Tolerance matches the 6-decimal-place format used when writing
            np.testing.assert_allclose(loaded, deviations, rtol=1e-4, atol=1e-6)
        finally:
            os.unlink(tmpfile)


class TestEdgeCases:
    def test_missing_file_raises(self):
        with pytest.raises((FileNotFoundError, OSError)):
            load_distribution("/nonexistent/path/to/file.dat")

    def test_single_particle(self):
        """Single-particle file should return shape (1, 6)."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".dat", delete=False
        ) as f:
            tmpfile = f.name
            f.write("# w_kin_ref: 3.0 MeV\n")
            f.write("# phi_ref: 0.0 deg\n")
            f.write(" 1.0  2.0  3.0  4.0  5.0  8.0\n")

        try:
            particles, _ = load_distribution(tmpfile)
            assert particles.shape == (1, 6)
        finally:
            os.unlink(tmpfile)
