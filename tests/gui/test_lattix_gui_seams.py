"""GUI seams for the lattix-backed importers (Bmad / SciBmad / PALS):
save-in-place guard, the shared extension dispatch, the open-dialog
filter, File -> Open on a real SciBmad deck, and the New Project wizard
materialising a foreign deck as ``<name>.dat``.

Modelled on test_mad8_gui_seams.py.  Every modal is stubbed (an
offscreen modal never returns).
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

pytest.importorskip("PyQt6")

REPO = Path(__file__).resolve().parents[2]
EXAMPLES = REPO / "examples"
FODO_JL = EXAMPLES / "scibmad" / "fodo.jl"
FODO_BMAD = EXAMPLES / "bmad" / "fodo.bmad"


def _lattix_or_skip():
    from linac_gen.io.lattix_bridge import _import_lattix
    try:
        _import_lattix()
    except ImportError as exc:
        pytest.skip(f"lattix not available: {exc}")
    from lattix.formats.base import FORMATS
    if "scibmad" not in FORMATS:
        pytest.skip("this lattix checkout predates the SciBmad format")


def _mock_window(lattice_path):
    m = MagicMock()
    m.state.lattice = object()
    m.state.lattice_path = lattice_path
    m.state.lattice_fitted = False
    return m


@pytest.mark.parametrize("path, in_place", [
    ("line.dat", True),
    ("line.txt", True),               # unknown suffix: parsed as TraceWin, writable
    ("line.bmad", False),
    ("line.jl", False),
    ("line.scibmad", False),
    ("line.pals.yaml", False),        # double suffix: Path.suffix alone would say ".yaml"
    ("line.pals.yml", False),
    ("line.pals.json", False),
    ("line.lattix.json", False),
    ("line.madx", False),
    ("line.lat", False),
    ("line.lte", False),
    (None, False),
])
def test_save_lattice_guard(qapp, path, in_place):
    """Ctrl+S must never overwrite a Bmad/SciBmad/PALS source with TraceWin text."""
    from linac_gen_gui.interphase.app import InterphaseWindow
    m = _mock_window(path)
    InterphaseWindow._save_lattice(m)
    assert m._write_lattice.called == in_place
    assert m._save_lattice_as.called == (not in_place)


def test_parse_lattice_file_dispatches_to_the_bridge(qapp, tmp_path, monkeypatch):
    """Hermetic: the GUI dispatcher routes every lattix suffix to the bridge."""
    import linac_gen.io.lattix_bridge as bridge
    from linac_gen.core.lattice import Lattice
    from linac_gen.elements.drift import Drift
    from linac_gen_gui.interphase.app import _parse_lattice_file
    seen = []

    def fake(path, **kw):
        seen.append(Path(path).name)
        lat = Lattice()
        lat.add(Drift(name="d", length=10.0))
        return lat, {"warnings": ["w1"]}

    monkeypatch.setattr(bridge, "parse_with_lattix", fake)
    for name in ("a.bmad", "b.jl", "c.pals.yaml"):
        p = tmp_path / name
        p.write_text("stub\n")
        lat, meta = _parse_lattice_file(str(p))
        assert len(lat.elements) == 1 and meta["warnings"] == ["w1"]
    assert seen == ["a.bmad", "b.jl", "c.pals.yaml"]


def test_open_dialog_filter_lists_every_import_family(qapp):
    from linac_gen.io.formats import DIALOG_FILTER, IMPORT_SUFFIXES
    from linac_gen_gui.interphase import app as app_mod
    assert app_mod._LATTICE_DIALOG_FILTER == DIALOG_FILTER
    for suf in IMPORT_SUFFIXES:
        assert f"*{suf}" in DIALOG_FILTER, suf
    for family in ("Bmad", "SciBmad", "PALS", "MAD-X", "MAD8", "Elegant", "TraceWin"):
        assert family in DIALOG_FILTER


def test_open_lattice_loads_a_scibmad_deck(win):
    """File -> Open on examples/scibmad/fodo.jl through the real slot."""
    _lattix_or_skip()
    statuses = []
    win.state.status_message.connect(statuses.append)
    win.open_lattice(FODO_JL)
    assert win.state.lattice is not None
    assert win.state.lattice_path == str(FODO_JL)
    assert any(s.startswith("Loaded 14 elements") for s in statuses), statuses
    assert any("warning(s)" in s for s in statuses), statuses      # the Beam-tab note
    kinds = [type(e).__name__ for e in win.state.lattice.elements]
    assert kinds.count("Quadrupole") == 2 and kinds.count("Dipole") == 2


def test_new_project_import_materialises_a_foreign_deck(qapp, tmp_path):
    """The wizard turns fodo.jl into <name>.dat inside the project and hands
    the import warnings back; the .dat re-parses to the same lattice."""
    _lattix_or_skip()
    from linac_gen.io.tracewin_parser import parse_tracewin
    from linac_gen_gui.interphase.dialogs.new_project import NewProjectDialog
    d = NewProjectDialog(None, start_dir=str(tmp_path), examples=[])
    d._name.setText("from_scibmad")
    d._rb_import.setChecked(True)
    d._import_path.setText(str(FODO_JL))
    d._copy_in.setChecked(False)          # ignored for a foreign deck: always materialised
    d._accept()
    res = d.project_result()
    d.deleteLater()
    assert res and res["mode"] == "import"
    dat = Path(res["lattice_path"])
    assert dat == tmp_path / "from_scibmad" / "from_scibmad.dat" and dat.is_file()
    assert any("set the Beam tab" in w for w in res["import_warnings"])
    lat, _ = parse_tracewin(str(dat))
    kinds = [type(e).__name__ for e in lat.elements]
    assert kinds.count("Quadrupole") == 2 and kinds.count("Dipole") == 2
    assert sum(e.length for e in lat.elements) == pytest.approx(6600.0, rel=1e-9)
    assert FODO_JL.read_text().startswith("# ====")      # the source is untouched


def test_new_project_rejects_an_unknown_suffix(qapp, tmp_path, monkeypatch):
    from linac_gen_gui.interphase.dialogs import new_project as np_mod
    warned = []
    monkeypatch.setattr(np_mod.QMessageBox, "warning",
                        staticmethod(lambda *a, **k: warned.append(a[2]) or 0))
    odd = tmp_path / "cell.yaml"
    odd.write_text("x: 1\n")
    d = np_mod.NewProjectDialog(None, start_dir=str(tmp_path), examples=[])
    d._name.setText("odd")
    d._rb_import.setChecked(True)
    d._import_path.setText(str(odd))
    d._accept()
    assert d.project_result() is None
    assert warned and ".pals.yaml" in warned[0] and ".jl" in warned[0]
    assert not (tmp_path / "odd").exists()
    d.deleteLater()
