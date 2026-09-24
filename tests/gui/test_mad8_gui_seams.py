"""GUI seams for the MAD8 importer: save-in-place guard (R1) and the
shared extension dispatch helper."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

REPO = Path(__file__).resolve().parents[2]
BTL_LAT = REPO / "BTL2025v0703.lat"


def _mock_window(lattice_path):
    """A stand-in `self` for InterphaseWindow._save_lattice."""
    m = MagicMock()
    m.state.lattice = object()
    m.state.lattice_path = lattice_path
    # Model the real AppState default — a bare MagicMock attribute is
    # truthy and would spuriously trigger the fitted-lattice Save-As
    # reroute for every case.
    m.state.lattice_fitted = False
    return m


@pytest.mark.parametrize("path, in_place", [
    ("line.dat", True),          # TraceWin: write in place
    ("line.lat", False),         # MAD8: must route to Save-As (R1)
    ("line.flat", False),
    ("line.madx", False),        # MAD-X: pre-existing guard
    ("line.seq", False),
    (None, False),               # no path yet
])
def test_save_lattice_guard(qapp, path, in_place):
    """Ctrl+S must never overwrite a MAD source with TraceWin text."""
    from linac_gen_gui.interphase.app import InterphaseWindow
    m = _mock_window(path)
    InterphaseWindow._save_lattice(m)
    assert m._write_lattice.called == in_place
    assert m._save_lattice_as.called == (not in_place)


def test_parse_lattice_file_dispatch(qapp, tmp_path):
    from linac_gen_gui.interphase.app import _parse_lattice_file
    mini = tmp_path / "mini.lat"
    mini.write_text("""BRHO := 4.881
D1: DRIFT, L=0.5
TOP: LINE=(D1)
""")
    lat, meta = _parse_lattice_file(str(mini))
    assert len(lat.elements) == 1
    assert "warnings" in meta          # GUI reads meta["warnings"]


@pytest.mark.skipif(not BTL_LAT.exists(), reason="BTL .lat not present")
def test_lattice_tab_reload_dispatch(qapp):
    """The editor Reload path must route a .lat through parse_mad8 —
    the TraceWin parser would silently mis-parse MAD8 label syntax."""
    from linac_gen_gui.interphase.app import _parse_lattice_file
    lat, _ = _parse_lattice_file(str(BTL_LAT))
    assert len(lat.elements) == 1125


def test_startup_restore_uses_last_session_beam(win, tmp_path):
    """A rigidity-less MAD8 lattice restored at start-up must be converted
    with the LAST SESSION'S beam (restored right after it), not the Beam
    tab's ~2 MeV start-up default — that mis-scaled every magnet 23x."""
    import json
    from linac_gen_gui.interphase import app as app_mod
    deck = tmp_path / "norig.mad8"
    deck.write_text("Q1: QUADRUPOLE, L=0.2, K1=1.0\nTOP: LINE=(Q1)\n")
    s = app_mod._settings()
    s.setValue(app_mod._SETTINGS_LAST_LATTICE, str(deck))
    s.setValue(app_mod._SETTINGS_SESSION_BEAM,
               json.dumps({"species": "H-", "energy": 800.0}))
    assert win.state.beam_config.energy < 10.0        # the start-up default
    win._restore_last_lattice()
    assert win.state.lattice_path == str(deck)
    q = [e for e in win.state.lattice.elements if type(e).__name__ == "Quadrupole"][0]
    assert q.gradient == pytest.approx(-4.8829, abs=1e-4)
