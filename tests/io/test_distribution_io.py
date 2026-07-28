"""Tests for distribution I/O (Task 12.1)."""
import os
import tempfile
import numpy as np
import pytest

from linac_gen.io.distribution_io import export_distribution, import_distribution
from linac_gen.core.beam import Beam
from linac_gen.core.reference import ReferenceParticle
from linac_gen.core.particle import PROTON


def make_beam(n=20, n_lost=0, w_kin=3.0, phi_s=-30.0, freq=352.21, current=60.0):
    """Helper: create a small beam with deterministic particles."""
    rng = np.random.default_rng(42)
    ref = ReferenceParticle(species=PROTON, w_kin=w_kin, frequency=freq, phi_s=phi_s)
    beam = Beam(ref=ref, n_particles=n, current=current)
    beam.particles[:] = rng.standard_normal((n, 6)) * [1.0, 1.0, 1.0, 1.0, 2.0, 0.01]
    # Mark some particles as lost
    for i in range(n_lost):
        beam.lost[i] = True
    return beam


class TestExportDistribution:
    def test_file_is_created(self, tmp_path):
        beam = make_beam()
        fp = str(tmp_path / "dist.dat")
        export_distribution(beam, fp)
        assert os.path.exists(fp)

    def test_header_lines_present(self, tmp_path):
        beam = make_beam(w_kin=3.0, phi_s=-30.0, freq=352.21, current=60.0)
        fp = str(tmp_path / "dist.dat")
        export_distribution(beam, fp)
        with open(fp) as f:
            content = f.read()
        assert "# Linac_Gen distribution file" in content
        assert "# species: proton" in content
        assert "# w_kin_ref: 3.000000 MeV" in content
        assert "# phi_ref: -30.000000 deg" in content
        assert "# frequency: 352.210000 MHz" in content
        assert "# current: 60.000000 mA" in content

    def test_n_particles_header_reflects_alive_count(self, tmp_path):
        beam = make_beam(n=20, n_lost=5)
        fp = str(tmp_path / "dist.dat")
        export_distribution(beam, fp)
        with open(fp) as f:
            content = f.read()
        assert "# n_particles: 15" in content

    def test_columns_header_present(self, tmp_path):
        beam = make_beam()
        fp = str(tmp_path / "dist.dat")
        export_distribution(beam, fp)
        with open(fp) as f:
            content = f.read()
        assert "# columns:" in content
        assert "phi_abs" in content
        assert "W_abs" in content

    def test_data_row_count_equals_alive(self, tmp_path):
        beam = make_beam(n=20, n_lost=3)
        fp = str(tmp_path / "dist.dat")
        export_distribution(beam, fp)
        with open(fp) as f:
            data_lines = [l for l in f if not l.startswith("#") and l.strip()]
        assert len(data_lines) == 17

    def test_absolute_coordinates_written(self, tmp_path):
        """phi_abs = ref.phi_s + dphi, W_abs = ref.w_kin + dW."""
        beam = make_beam(n=5, w_kin=3.0, phi_s=-30.0)
        fp = str(tmp_path / "dist.dat")
        export_distribution(beam, fp)
        with open(fp) as f:
            data_lines = [l for l in f if not l.startswith("#") and l.strip()]
        ref = beam.ref
        alive = beam.alive_particles
        for i, line in enumerate(data_lines):
            vals = list(map(float, line.split()))
            assert len(vals) == 6
            phi_expected = ref.phi_s + alive[i, 4]
            w_expected = ref.w_kin + alive[i, 5]
            # Format writes 6 decimal places → ~1e-6 precision
            assert abs(vals[4] - phi_expected) < 1e-5
            assert abs(vals[5] - w_expected) < 1e-5

    def test_transverse_coords_unchanged(self, tmp_path):
        beam = make_beam(n=5)
        fp = str(tmp_path / "dist.dat")
        export_distribution(beam, fp)
        with open(fp) as f:
            data_lines = [l for l in f if not l.startswith("#") and l.strip()]
        alive = beam.alive_particles
        for i, line in enumerate(data_lines):
            vals = list(map(float, line.split()))
            # Format writes 6 decimal places → ~1e-6 precision
            assert abs(vals[0] - alive[i, 0]) < 1e-5
            assert abs(vals[1] - alive[i, 1]) < 1e-5
            assert abs(vals[2] - alive[i, 2]) < 1e-5
            assert abs(vals[3] - alive[i, 3]) < 1e-5

    def test_only_alive_particles_exported(self, tmp_path):
        """Lost particles must not appear in the output."""
        n, n_lost = 10, 4
        beam = make_beam(n=n, n_lost=n_lost)
        fp = str(tmp_path / "dist.dat")
        export_distribution(beam, fp)
        with open(fp) as f:
            data_lines = [l for l in f if not l.startswith("#") and l.strip()]
        assert len(data_lines) == n - n_lost


class TestRoundTrip:
    def test_roundtrip_particles_match(self, tmp_path):
        """Export then import: particle coords should match within 1e-5."""
        beam = make_beam(n=50)
        fp = str(tmp_path / "rt.dat")
        export_distribution(beam, fp)
        particles, header = import_distribution(fp)
        alive = beam.alive_particles
        # After import, particles are in relative coords (dphi, dW)
        # Export writes phi_abs = ref.phi_s + dphi, W_abs = ref.w_kin + dW
        # import subtracts w_kin_ref and phi_ref back → should recover original
        assert particles.shape == alive.shape
        np.testing.assert_allclose(particles, alive, atol=1e-5)

    def test_roundtrip_header_values(self, tmp_path):
        beam = make_beam(w_kin=5.5, phi_s=-25.0)
        fp = str(tmp_path / "rt.dat")
        export_distribution(beam, fp)
        _, header = import_distribution(fp)
        assert abs(header["w_kin_ref"] - 5.5) < 1e-5
        assert abs(header["phi_ref"] - (-25.0)) < 1e-5

    def test_roundtrip_with_losses(self, tmp_path):
        """Only alive particles survive the round-trip."""
        beam = make_beam(n=30, n_lost=10)
        fp = str(tmp_path / "rt.dat")
        export_distribution(beam, fp)
        particles, _ = import_distribution(fp)
        assert particles.shape[0] == beam.n_alive


class TestImportDistribution:
    def test_is_alias_for_load_distribution(self, tmp_path):
        """import_distribution must delegate to load_distribution."""
        from linac_gen.distributions.from_file import load_distribution
        beam = make_beam(n=10)
        fp = str(tmp_path / "alias.dat")
        export_distribution(beam, fp)
        p1, h1 = import_distribution(fp)
        p2, h2 = load_distribution(fp)
        np.testing.assert_array_equal(p1, p2)
        assert h1 == h2

    def test_explicit_ref_override(self, tmp_path):
        """Explicit ref args to import_distribution override file header."""
        beam = make_beam(n=5, w_kin=3.0, phi_s=0.0)
        fp = str(tmp_path / "override.dat")
        export_distribution(beam, fp)
        # Provide different ref values
        particles, _ = import_distribution(fp, ref_w_kin=4.0, ref_phi_s=10.0)
        # dphi should now be phi_abs - 10.0, dW = W_abs - 4.0
        alive = beam.alive_particles
        # phi_abs = 0.0 + alive[:,4]; with ref=10 → dphi = alive[:,4] - 10
        expected_dphi = alive[:, 4] - 10.0
        expected_dw = alive[:, 5] - 1.0  # W_abs = 3.0 + dW, dW_new = W_abs - 4.0 = dW - 1.0
        # Format writes 6 decimal places → ~1e-6 precision in absolute, ~1e-5 after subtraction
        np.testing.assert_allclose(particles[:, 4], expected_dphi, atol=1e-5)
        np.testing.assert_allclose(particles[:, 5], expected_dw, atol=1e-5)
