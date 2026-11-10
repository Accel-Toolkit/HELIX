"""The Beam tab's Import .dst… with a file that keeps lost particles or a DC beam."""
from __future__ import annotations

import struct

import numpy as np
import pytest

pytest.importorskip("PyQt6")


def _file(path, rows, current=6.8):
    arr = np.asarray(rows, dtype="<f8")
    path.write_bytes(struct.pack("<2sIddB", b"\x7e\x65", len(arr), current, 162.5, 0x65)
                     + arr.tobytes() + struct.pack("<d", 939.294))
    return str(path)


def _tab(monkeypatch, tmp_path, file_path):
    from linac_gen_gui.interphase.state import AppState
    from linac_gen_gui.interphase.tabs.beam_tab import BeamTab
    from PyQt6.QtWidgets import QFileDialog
    monkeypatch.setenv("HELIX_QSETTINGS_DIR", str(tmp_path))
    monkeypatch.setattr(QFileDialog, "getOpenFileName",
                        staticmethod(lambda *a, **k: (file_path, "")))
    tab = BeamTab(AppState())
    tab._continuous.setChecked(True)            # the file decides, either way
    tab._import_dst()
    return tab


def _rows(phi, w, lost=0):
    n = len(phi)
    rng = np.random.default_rng(5)
    good = [[rng.normal(0, 0.1), rng.normal(0, 1e-3), rng.normal(0, 0.1),
             rng.normal(0, 1e-3), phi[i], w[i], 0.0] for i in range(n)]
    return good + [[0.3, 0.0, 0.3, 0.0, 1.0, 0.02, 29.0]] * lost


def test_a_bunched_file_fills_the_form_and_clears_continuous(qapp, monkeypatch, tmp_path):
    phi = np.linspace(-0.2, 0.2, 150)
    path = _file(tmp_path / "b.dst", _rows(phi, 2.1 + 1e-3 * phi, lost=50))
    tab = _tab(monkeypatch, tmp_path, path)
    assert tab._npart.value() == 150
    assert tab._current.value() == pytest.approx(6.8 * 150 / 200, abs=1e-3)
    assert not tab._continuous.isChecked()
    assert "50 of 200 flagged lost" in tab._status.text()
    assert "Continuous" not in tab._status.text()


def test_a_dc_file_ticks_continuous_and_says_so(qapp, monkeypatch, tmp_path):
    phi = np.linspace(-np.pi, np.pi, 60)
    path = _file(tmp_path / "dc.dst", _rows(phi, 0.03 + 1e-6 * phi))
    tab = _tab(monkeypatch, tmp_path, path)
    assert tab._continuous.isChecked()
    assert "Continuous ticked" in tab._status.text()
    cfg = tab.get_beam_config()
    assert cfg.continuous is True and cfg.source == "file"


def test_the_status_reports_particles_far_from_the_median_energy(qapp, monkeypatch, tmp_path):
    phi = np.linspace(-0.2, 0.2, 60)
    w = np.full(60, 2.1)
    w[:3] = 0.03
    tab = _tab(monkeypatch, tmp_path, _file(tmp_path / "e.dst", _rows(phi, w)))
    assert "3 particles far from the median energy" in tab._status.text()
