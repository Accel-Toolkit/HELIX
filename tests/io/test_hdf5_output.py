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


class TestRunCurrent:
    """Run-current provenance on the envelope group (fix item
    tune-depression-stale-banner).

    Writer: ``current_mA`` is written ONLY when the run current is known
    (finite), and a boolean ``run_current_known`` marker is always
    written — a legacy MP file's 0.0 sentinel is otherwise
    indistinguishable from a genuine 0 mA envelope run.

    Loader: legacy files (no marker) carrying 0.0 are treated as
    unknown and fall back to ``beam_config/attrs['current']``; a
    new-format 0.0 with the marker is trusted (genuine 0 mA is data);
    the transport-only marker never appears in the returned dict.
    """

    @staticmethod
    def _cfg(current):
        from types import SimpleNamespace
        return SimpleNamespace(current=current, n_particles=50)

    @staticmethod
    def _make_legacy(fp):
        """Rewrite a saved file to the pre-fix on-disk form."""
        with h5py.File(fp, "a") as f:
            env = f["envelope"]
            if "run_current_known" in env.attrs:
                del env.attrs["run_current_known"]
            env.attrs["current_mA"] = 0.0

    def test_writer_omits_current_when_unknown_and_marks_known(self, tmp_path):
        rec = make_recorder_with_data()          # bare recorder: unknown
        fp = str(tmp_path / "unknown.h5")
        save_results_hdf5(rec, fp)
        with h5py.File(fp, "r") as f:
            assert "current_mA" not in f["envelope"].attrs
            assert bool(f["envelope"].attrs["run_current_known"]) is False

    @pytest.mark.parametrize("value", [5.0, 0.0])
    def test_writer_persists_known_current(self, tmp_path, value):
        rec = make_recorder_with_data()
        rec.current_mA = value
        fp = str(tmp_path / "known.h5")
        save_results_hdf5(rec, fp)
        with h5py.File(fp, "r") as f:
            assert float(f["envelope"].attrs["current_mA"]) == value
            assert bool(f["envelope"].attrs["run_current_known"]) is True

    def test_writer_resolves_current_from_attached_beam(self, tmp_path):
        """MP recorder with only ``.beam`` attached persists correctly."""
        from types import SimpleNamespace
        rec = make_recorder_with_data()
        rec.beam = SimpleNamespace(current=7.0)
        fp = str(tmp_path / "beam_only.h5")
        save_results_hdf5(rec, fp)
        with h5py.File(fp, "r") as f:
            assert float(f["envelope"].attrs["current_mA"]) == 7.0
            assert bool(f["envelope"].attrs["run_current_known"]) is True

    def test_loader_legacy_zero_falls_back_to_beam_config_current(self, tmp_path):
        fp = str(tmp_path / "legacy_mp.h5")
        save_results_hdf5(make_recorder_with_data(), fp,
                          beam_config=self._cfg(60.0))
        self._make_legacy(fp)
        loaded = load_results_hdf5(fp)
        assert loaded["current_mA"] == 60.0
        assert "run_current_known" not in loaded

    def test_loader_legacy_zero_without_beam_config_is_unknown(self, tmp_path):
        fp = str(tmp_path / "legacy_nocfg.h5")
        save_results_hdf5(make_recorder_with_data(), fp)
        self._make_legacy(fp)
        loaded = load_results_hdf5(fp)
        assert "current_mA" not in loaded
        assert "run_current_known" not in loaded

    def test_loader_trusts_marker_for_genuine_zero(self, tmp_path):
        """New-format envelope file at a genuine 0 mA stays 0.0 even when
        the dump-time beam_config carried a different current."""
        rec = make_recorder_with_data()
        rec.current_mA = 0.0
        fp = str(tmp_path / "genuine_zero.h5")
        save_results_hdf5(rec, fp, beam_config=self._cfg(60.0))
        loaded = load_results_hdf5(fp)
        assert loaded["current_mA"] == 0.0
        assert "run_current_known" not in loaded

    def test_loader_known_value_wins_over_beam_config(self, tmp_path):
        rec = make_recorder_with_data()
        rec.current_mA = 5.0
        fp = str(tmp_path / "known_vs_cfg.h5")
        save_results_hdf5(rec, fp, beam_config=self._cfg(60.0))
        loaded = load_results_hdf5(fp)
        assert loaded["current_mA"] == 5.0
        assert "run_current_known" not in loaded

    def test_loader_unknown_with_beam_config_resolves_to_config(self, tmp_path):
        """New-format file from a hand-built recorder (marker False):
        the dump-time config current is the best available answer."""
        fp = str(tmp_path / "unknown_cfg.h5")
        save_results_hdf5(make_recorder_with_data(), fp,
                          beam_config=self._cfg(60.0))
        loaded = load_results_hdf5(fp)
        assert loaded["current_mA"] == 60.0
        assert "run_current_known" not in loaded
