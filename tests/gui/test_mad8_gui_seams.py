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
