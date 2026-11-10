"""The GUI's envelope run starts a beam read from a .dst file where a multiparticle run
does: at the file's energy and clock, with the file's Twiss (it took the Beam tab's)."""
import os

import numpy as np
import pytest

pytest.importorskip("PyQt6")

from linac_gen.core.particle import PROTON  # noqa: E402
from linac_gen.io.tracewin_dst import load_dst, write_dst  # noqa: E402

FODO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "examples", "fodo_cell.dat"))


def test_the_envelope_run_starts_from_the_files_beam(qapp, win, tmp_path):
    p = np.random.default_rng(5).normal(size=(2000, 6)) * [1.0, 1.5, 0.8, 1.2, 3.0, 0.004]
    path = tmp_path / "beam.dst"
    write_dst(str(path), p, current_mA=0.0, frequency_MHz=325.0, mass_MeV=PROTON.mass,
              w_kin_ref=7.0003)
    win.open_lattice(FODO)
    cfg = win.state.beam_config
    cfg.species, cfg.energy, cfg.frequency, cfg.current = "proton", 6.5, 162.5, 0.0
    cfg.source, cfg.distribution_file = "file", str(path)
    win.state.set_beam_config(cfg)
    win._run_envelope()
    win.wait_worker(win._envelope_worker)
    res = win.state.results
    assert np.asarray(res.ref_w_kin)[0] == load_dst(str(path))[1]["w_kin_ref"] != 6.5
    assert np.asarray(res.ref_frequency)[0] == 325.0
    assert cfg.energy == 6.5                          # the Beam tab's settings stay


def _dst_config(tmp_path):
    from linac_gen.core.config import BeamConfig
    p = np.random.default_rng(5).normal(size=(2000, 6)) * [1.0, 1.5, 0.8, 1.2, 3.0, 0.004]
    path = tmp_path / "beam.dst"
    write_dst(str(path), p, current_mA=0.0, frequency_MHz=325.0, mass_MeV=PROTON.mass,
              w_kin_ref=7.0003)
    cfg = BeamConfig(species="proton", energy=6.5, frequency=162.5, n_particles=2000,
                     source="file", distribution_file=str(path))
    return cfg, load_dst(str(path))[1]["w_kin_ref"]


def test_the_envelope_overlay_on_a_multiparticle_run_starts_from_the_file(qapp, tmp_path):
    from linac_gen_gui.interphase.tabs.results_tab import _beam_inputs_from_config

    from linac_gen.distributions.factory import reference_config
    cfg, w_file = _dst_config(tmp_path)
    ref, initial = _beam_inputs_from_config(cfg)
    assert (ref.w_kin, ref.frequency) == (w_file, 325.0)
    assert initial["beta_x"] == reference_config(cfg).beta_x != cfg.beta_x


def test_the_dispersion_model_and_the_transfer_matrix_dialog_start_from_the_file(qapp, tmp_path):
    from types import SimpleNamespace

    from linac_gen_gui.dialogs.transfer_matrix_dialog import TransferMatrixDialog
    from linac_gen_gui.interphase.tabs.results_tab import _DispersionPopup

    from linac_gen.core.lattice import Lattice
    from linac_gen.elements.drift import Drift
    cfg, w_file = _dst_config(tmp_path)
    lat = Lattice()
    lat.add(Drift("D", 100.0))
    got, why = _DispersionPopup._model_inputs(SimpleNamespace(_state=SimpleNamespace(
        lattice=lat, beam_config=cfg)))
    assert why is None and (got[1].w_kin, got[1].frequency) == (w_file, 325.0)
    ref = TransferMatrixDialog._build_ref(SimpleNamespace(_beam_config=cfg))
    assert (ref.w_kin, ref.frequency) == (w_file, 325.0)


def test_a_parameter_scans_envelope_run_starts_from_the_file(qapp, tmp_path):
    from types import SimpleNamespace

    from linac_gen_gui.interphase.dialogs.parameter_scan import _ScanWorker

    from linac_gen.core.lattice import Lattice
    from linac_gen.elements.drift import Drift
    cfg, w_file = _dst_config(tmp_path)
    lat = Lattice()
    lat.add(Drift("D", 100.0, aperture=30.0))
    worker = _ScanWorker(SimpleNamespace(lattice=lat, beam_config=cfg), lat.elements[0], "length",
                         np.array([100.0]), "envelope", "ref_w_kin", "last")
    assert worker._run_envelope(worker._lattice, worker._beam) == w_file
