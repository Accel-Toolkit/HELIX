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


class TestActionScan:
    """``action_scan/`` — the opt-in halo action scan group.

    Written only when the recorder carries a scan aligned with ``s``
    (every other file keeps its former tree); loads back as a nested
    ``results["action_scan"]`` dict; files without it load unchanged.
    """

    @staticmethod
    def _recorder(n=2000, steps=3, seed=1):
        rec = DiagnosticRecorder()
        rec.configure_action_scan()
        rng = np.random.default_rng(seed)
        for i in range(steps):
            ref = ReferenceParticle(species=PROTON, w_kin=3.0 + i,
                                    frequency=352.21)
            beam = Beam(ref=ref, n_particles=n, current=0.0)
            beam.particles[:] = rng.standard_normal((n, 6)) * [1, 1, 1, 1,
                                                               2, 0.01]
            beam.lost[:i * n // 20] = True           # 0 %, 5 %, 10 % lost
            beam.continuous = (i == 0)
            rec.record(beam, 100.0 * i, f"EL_{i}")
        return rec

    def test_round_trip(self, tmp_path):
        from linac_gen.diagnostics.action_scan import PLANES
        rec = self._recorder()
        fp = str(tmp_path / "scan.h5")
        save_results_hdf5(rec, fp)
        with h5py.File(fp, "r") as f:
            g = f["action_scan"]
            assert g["count_x"].dtype == np.int32
            assert g["count_x"].compression == "gzip"
            assert g["count_x"].shape == (3, len(rec.action_scan["n"]))
            assert int(g.attrs["schema_version"]) == 1
            assert g.attrs["planes"] == ",".join(PLANES)
        scan = load_results_hdf5(fp)["action_scan"]
        np.testing.assert_array_equal(scan["n"], rec.action_scan["n"])
        np.testing.assert_array_equal(scan["n_alive"], [2000, 1900, 1800])
        np.testing.assert_array_equal(scan["continuous"],
                                      [True, False, False])
        assert scan["element"] == ["EL_0", "EL_1", "EL_2"]
        assert "exp(-n/2)" in scan["attrs"]["definition"]
        assert scan["attrs"]["schema_version"] == 1
        for p in PLANES:
            np.testing.assert_array_equal(
                scan[f"count_{p}"], np.asarray(rec.action_scan[f"count_{p}"]))
            for key in ("eps", "eps_n", "n_max"):
                np.testing.assert_array_equal(
                    scan[f"{key}_{p}"], rec.action_scan[f"{key}_{p}"])

    def test_no_scan_no_group(self, tmp_path):
        rec = make_recorder_with_data()
        fp = str(tmp_path / "plain.h5")
        save_results_hdf5(rec, fp)
        with h5py.File(fp, "r") as f:
            assert "action_scan" not in f
        assert "action_scan" not in load_results_hdf5(fp)

    def test_group_is_the_only_addition(self, tmp_path):
        """Same recorder written with and without its scan: the object
        trees differ by exactly the action_scan group."""
        rec = self._recorder()
        with_fp, without_fp = str(tmp_path / "a.h5"), str(tmp_path / "b.h5")
        save_results_hdf5(rec, with_fp)
        del rec.action_scan
        save_results_hdf5(rec, without_fp)

        def tree(fp):
            names = []
            with h5py.File(fp, "r") as f:
                f.visit(names.append)
            return set(names)
        extra = tree(with_fp) - tree(without_fp)
        assert tree(without_fp) <= tree(with_fp)
        assert {n.split("/")[0] for n in extra} == {"action_scan"}

    def test_misaligned_scan_skipped_with_warning(self, tmp_path):
        rec = self._recorder()
        rec.action_scan["eps_x"].append(1.0)
        fp = str(tmp_path / "bad.h5")
        with pytest.warns(RuntimeWarning, match="action scan not written"):
            save_results_hdf5(rec, fp)
        with h5py.File(fp, "r") as f:
            assert "action_scan" not in f

    def test_misaligned_group_dropped_on_load(self, tmp_path):
        rec = self._recorder()
        fp = str(tmp_path / "cut.h5")
        save_results_hdf5(rec, fp)
        with h5py.File(fp, "a") as f:
            data = f["action_scan"]["eps_y"][:2]
            del f["action_scan"]["eps_y"]
            f["action_scan"].create_dataset("eps_y", data=data)
        with pytest.warns(RuntimeWarning, match="action_scan group ignored"):
            res = load_results_hdf5(fp)
        assert "action_scan" not in res
        assert len(res["s"]) == 3

    def test_saving_loaded_results_writes_the_group_back(self, tmp_path):
        """Loaded results (the dict form, e.g. re-saved by the assistant's
        write_results) round-trip: save -> load -> save -> load is equal,
        and the groups after action_scan/ (beam_config/) are written."""
        from types import SimpleNamespace
        rec = self._recorder()
        first, second = str(tmp_path / "1.h5"), str(tmp_path / "2.h5")
        cfg = SimpleNamespace(current=5.0, n_particles=2000)
        save_results_hdf5(rec, first, beam_config=cfg)
        a = load_results_hdf5(first)
        save_results_hdf5(SimpleNamespace(**a), second, beam_config=cfg)
        b = load_results_hdf5(second)
        sa, sb = a["action_scan"], b["action_scan"]
        assert set(sa) == set(sb)
        for key in sa:
            if key == "attrs":
                assert sa[key] == sb[key]
            elif key == "element":
                assert sa[key] == sb[key] == ["EL_0", "EL_1", "EL_2"]
            else:
                np.testing.assert_array_equal(sa[key], sb[key])
        with h5py.File(second, "r") as f:
            assert "beam_config" in f and "action_scan" in f

    def test_unknown_members_ignored_newer_schema_dropped(self, tmp_path):
        """Additions a later writer may make (sub-groups, scalar or
        differently-sized datasets, array attributes) are ignored; a newer
        schema or an incomplete group is dropped WITH its reason."""
        rec = self._recorder()
        fp = str(tmp_path / "extra.h5")
        save_results_hdf5(rec, fp)
        with h5py.File(fp, "a") as f:
            g = f["action_scan"]
            g.create_group("future_subgroup")
            g.create_dataset("scalar_note", data=3.0)
            g.create_dataset("future_grid_edges", data=np.arange(552.0))
            g.attrs["future_levels"] = [1.0, 0.1, 0.01]
        res = load_results_hdf5(fp)                    # still loads
        scan = res["action_scan"]
        assert "action_scan_error" not in res
        for extra in ("future_subgroup", "scalar_note", "future_grid_edges"):
            assert extra not in scan
        assert list(scan["attrs"]["future_levels"]) == [1.0, 0.1, 0.01]
        assert scan["count_x"].shape[0] == 3
        with h5py.File(fp, "a") as f:
            f["action_scan"].attrs["schema_version"] = 99
        with pytest.warns(RuntimeWarning, match="newer than this reader"):
            res = load_results_hdf5(fp)
        assert "action_scan" not in res and len(res["s"]) == 3
        assert "newer than this reader" in res["action_scan_error"]
        with h5py.File(fp, "a") as f:
            f["action_scan"].attrs["schema_version"] = 1
            del f["action_scan"]["eps_n_y"]
        with pytest.warns(RuntimeWarning, match="eps_n_y missing"):
            res = load_results_hdf5(fp)
        assert "eps_n_y missing" in res["action_scan_error"]

    def test_long_non_ascii_names_cut_on_a_character_boundary(self,
                                                              tmp_path):
        rec = self._recorder()
        long_name = "QUAD_" + "é" * 20                 # 45 UTF-8 bytes
        rec.element_names[1] = long_name
        fp = str(tmp_path / "names.h5")
        save_results_hdf5(rec, fp)
        got = load_results_hdf5(fp)["action_scan"]["element"][1]
        assert long_name.startswith(got) and "\ufffd" not in got
        assert len(got.encode("utf-8")) <= 32

    def test_gaussian_pin_through_the_file(self, tmp_path):
        """400k Gaussian: % outside n = 9.25 read back from the FILE equals
        100*exp(-9.25/2) within 4 binomial sigma."""
        from linac_gen.diagnostics.action_scan import action_scan_from_results
        n = 400_000
        rec = self._recorder(n=n, steps=1, seed=2)
        fp = str(tmp_path / "gauss.h5")
        save_results_hdf5(rec, fp)
        sc = action_scan_from_results(load_results_hdf5(fp))
        j = int(np.searchsorted(sc.n, 9.25))
        frac = np.exp(-9.25 / 2.0)
        sig = np.sqrt(n * frac * (1.0 - frac))
        for p in ("x", "y", "z"):
            assert abs(int(sc.counts[p][0, j]) - n * frac) < 4.0 * sig, p
