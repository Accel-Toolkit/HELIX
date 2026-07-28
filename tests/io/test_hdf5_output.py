"""Tests for HDF5 results output (Task 12.2)."""
import os
import numpy as np
import pytest

h5py = pytest.importorskip("h5py")

from linac_gen.io.hdf5_output import save_results_hdf5, load_results_hdf5
from linac_gen.diagnostics.recorder import DiagnosticRecorder
from linac_gen.core.beam import Beam
from linac_gen.core.reference import ReferenceParticle
from linac_gen.core.particle import PROTON


def make_recorder_with_data():
    """Return a DiagnosticRecorder pre-populated with a few steps."""
    rec = DiagnosticRecorder()
    rng = np.random.default_rng(0)
    for i, s in enumerate([0.0, 0.5, 1.0]):
        ref = ReferenceParticle(species=PROTON, w_kin=3.0 + i * 0.1,
                                frequency=352.21, phi_s=-30.0 + i)
        beam = Beam(ref=ref, n_particles=50, current=60.0)
        beam.particles[:] = rng.standard_normal((50, 6)) * [1, 1, 1, 1, 2, 0.01]
        rec.record(beam, s)
        rec.save_snapshot(beam, s)
    return rec


class TestSaveResultsHDF5:
    def test_file_is_created(self, tmp_path):
        rec = make_recorder_with_data()
        fp = str(tmp_path / "results.h5")
        save_results_hdf5(rec, fp)
        assert os.path.exists(fp)

    def test_envelope_group_exists(self, tmp_path):
        rec = make_recorder_with_data()
        fp = str(tmp_path / "results.h5")
        save_results_hdf5(rec, fp)
        with h5py.File(fp, "r") as f:
            assert "envelope" in f

    def test_reference_group_exists(self, tmp_path):
        rec = make_recorder_with_data()
        fp = str(tmp_path / "results.h5")
        save_results_hdf5(rec, fp)
        with h5py.File(fp, "r") as f:
            assert "reference" in f

    def test_particles_group_exists(self, tmp_path):
        rec = make_recorder_with_data()
        fp = str(tmp_path / "results.h5")
        save_results_hdf5(rec, fp)
        with h5py.File(fp, "r") as f:
            assert "particles" in f

    def test_envelope_s_array_stored(self, tmp_path):
        rec = make_recorder_with_data()
        fp = str(tmp_path / "results.h5")
        save_results_hdf5(rec, fp)
        with h5py.File(fp, "r") as f:
            np.testing.assert_allclose(f["envelope"]["s"][:], np.array(rec.s))

    def test_envelope_sigma_x_stored(self, tmp_path):
        rec = make_recorder_with_data()
        fp = str(tmp_path / "results.h5")
        save_results_hdf5(rec, fp)
        with h5py.File(fp, "r") as f:
            np.testing.assert_allclose(f["envelope"]["sigma_x"][:],
                                       np.array(rec.sigma_x))

    def test_envelope_longitudinal_twiss_stored(self, tmp_path):
        """alpha_z/beta_z (recorded since 2026-07) round-trip."""
        rec = make_recorder_with_data()
        fp = str(tmp_path / "results.h5")
        save_results_hdf5(rec, fp)
        with h5py.File(fp, "r") as f:
            np.testing.assert_allclose(f["envelope"]["alpha_z"][:],
                                       np.array(rec.alpha_z))
            np.testing.assert_allclose(f["envelope"]["beta_z"][:],
                                       np.array(rec.beta_z))
        loaded = load_results_hdf5(fp)      # flat dict keyed by name
        np.testing.assert_allclose(loaded["alpha_z"],
                                   np.array(rec.alpha_z))

    def test_envelope_emit_x_stored(self, tmp_path):
        rec = make_recorder_with_data()
        fp = str(tmp_path / "results.h5")
        save_results_hdf5(rec, fp)
        with h5py.File(fp, "r") as f:
            np.testing.assert_allclose(f["envelope"]["emit_x"][:],
                                       np.array(rec.emit_x))

    def test_envelope_transmission_stored(self, tmp_path):
        rec = make_recorder_with_data()
        fp = str(tmp_path / "results.h5")
        save_results_hdf5(rec, fp)
        with h5py.File(fp, "r") as f:
            np.testing.assert_allclose(f["envelope"]["transmission"][:],
                                       np.array(rec.transmission))

    def test_reference_history_stored(self, tmp_path):
        rec = make_recorder_with_data()
        fp = str(tmp_path / "results.h5")
        save_results_hdf5(rec, fp)
        with h5py.File(fp, "r") as f:
            np.testing.assert_allclose(f["reference"]["w_kin"][:],
                                       np.array(rec.ref_w_kin))
            np.testing.assert_allclose(f["reference"]["phi_s"][:],
                                       np.array(rec.ref_phi_s))
            np.testing.assert_allclose(f["reference"]["beta"][:],
                                       np.array(rec.ref_beta))
            np.testing.assert_allclose(f["reference"]["gamma"][:],
                                       np.array(rec.ref_gamma))

    def test_snapshots_stored_count(self, tmp_path):
        rec = make_recorder_with_data()
        fp = str(tmp_path / "results.h5")
        save_results_hdf5(rec, fp)
        with h5py.File(fp, "r") as f:
            assert len(f["particles"]) == len(rec._snapshots)

    def test_snapshot_data_shape(self, tmp_path):
        rec = make_recorder_with_data()
        fp = str(tmp_path / "results.h5")
        save_results_hdf5(rec, fp)
        with h5py.File(fp, "r") as f:
            for key in f["particles"]:
                grp = f["particles"][key]
                assert "data" in grp
                assert grp["data"].shape[1] == 6

    def test_snapshot_ref_attrs(self, tmp_path):
        rec = make_recorder_with_data()
        fp = str(tmp_path / "results.h5")
        save_results_hdf5(rec, fp)
        with h5py.File(fp, "r") as f:
            for key in f["particles"]:
                grp = f["particles"][key]
                for attr in ("s", "w_kin", "phi_s", "beta", "gamma"):
                    assert attr in grp.attrs, f"Missing attr '{attr}' in snapshot '{key}'"

    def test_snapshot_ref_w_kin_correct(self, tmp_path):
        rec = make_recorder_with_data()
        fp = str(tmp_path / "results.h5")
        save_results_hdf5(rec, fp)
        with h5py.File(fp, "r") as f:
            stored_w_kins = sorted(
                [f["particles"][k].attrs["w_kin"] for k in f["particles"]]
            )
        expected = sorted([v[1].w_kin for v in rec._snapshots.values()])
        np.testing.assert_allclose(stored_w_kins, expected, rtol=1e-10)

    def test_beam_config_stored_as_attrs(self, tmp_path):
        rec = make_recorder_with_data()
        fp = str(tmp_path / "results.h5")

        class FakeConfig:
            def __init__(self):
                self.current = 60.0
                self.species = "proton"
                self.frequency = 352.21
                self.n_particles = 50
                self.extra_none = None  # None values should be skipped

        save_results_hdf5(rec, fp, beam_config=FakeConfig())
        with h5py.File(fp, "r") as f:
            assert "beam_config" in f
            cfg = f["beam_config"]
            assert abs(cfg.attrs["current"] - 60.0) < 1e-9
            assert abs(cfg.attrs["frequency"] - 352.21) < 1e-9
            assert cfg.attrs["n_particles"] == 50
            # None value should not be stored
            assert "extra_none" not in cfg.attrs

    def test_no_beam_config_group_absent(self, tmp_path):
        rec = make_recorder_with_data()
        fp = str(tmp_path / "results.h5")
        save_results_hdf5(rec, fp)  # no beam_config
        with h5py.File(fp, "r") as f:
            assert "beam_config" not in f


class TestEmptyRecorder:
    def test_empty_recorder_produces_valid_file(self, tmp_path):
        rec = DiagnosticRecorder()
        fp = str(tmp_path / "empty.h5")
        save_results_hdf5(rec, fp)
        assert os.path.exists(fp)
        with h5py.File(fp, "r") as f:
            assert "envelope" in f

    def test_empty_recorder_no_particles_group(self, tmp_path):
        """No snapshots → no particles group."""
        rec = DiagnosticRecorder()
        fp = str(tmp_path / "empty.h5")
        save_results_hdf5(rec, fp)
        with h5py.File(fp, "r") as f:
            assert "particles" not in f

    def test_empty_recorder_s_dataset_empty(self, tmp_path):
        rec = DiagnosticRecorder()
        fp = str(tmp_path / "empty.h5")
        save_results_hdf5(rec, fp)
        with h5py.File(fp, "r") as f:
            assert f["envelope"]["s"].shape == (0,)


class TestLoadResultsHDF5:
    def test_load_returns_dict(self, tmp_path):
        rec = make_recorder_with_data()
        fp = str(tmp_path / "results.h5")
        save_results_hdf5(rec, fp)
        results = load_results_hdf5(fp)
        assert isinstance(results, dict)

    def test_load_has_s_key(self, tmp_path):
        rec = make_recorder_with_data()
        fp = str(tmp_path / "results.h5")
        save_results_hdf5(rec, fp)
        results = load_results_hdf5(fp)
        assert "s" in results

    def test_load_envelope_roundtrip(self, tmp_path):
        rec = make_recorder_with_data()
        fp = str(tmp_path / "results.h5")
        save_results_hdf5(rec, fp)
        results = load_results_hdf5(fp)
        np.testing.assert_allclose(results["s"], np.array(rec.s))
        np.testing.assert_allclose(results["sigma_x"], np.array(rec.sigma_x))
        np.testing.assert_allclose(results["emit_x"], np.array(rec.emit_x))
        np.testing.assert_allclose(results["transmission"], np.array(rec.transmission))

    def test_load_reference_roundtrip(self, tmp_path):
        rec = make_recorder_with_data()
        fp = str(tmp_path / "results.h5")
        save_results_hdf5(rec, fp)
        results = load_results_hdf5(fp)
        np.testing.assert_allclose(results["ref_w_kin"], np.array(rec.ref_w_kin))
        np.testing.assert_allclose(results["ref_beta"], np.array(rec.ref_beta))
        np.testing.assert_allclose(results["ref_gamma"], np.array(rec.ref_gamma))
