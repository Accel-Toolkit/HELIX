"""TraceWin ``.ini`` import through the REAL GUI seams: the Open-Lattice
prompt (Yes / No / no sibling), the Beam-tab button, the New Project
wizard checkbox and the ``_new_project`` handler, and the session-beam
consequence (the imported beam is what a restart restores)."""
from __future__ import annotations

import json
import shutil
import struct
from pathlib import Path

import pytest

pytest.importorskip("PyQt6")

from linac_gen.core.config import BeamConfig
from linac_gen.io.tracewin_ini import FIELDS

REPO = Path(__file__).resolve().parents[2]
ADS = REPO / "tests" / "io" / "fixtures" / "tracewin_ini" / "ads.ini"
FODO = REPO / "tests" / "io" / "fixtures" / "simple_fodo.dat"
OFF = {f.name: f.offset for f in FIELDS}


def _project(tmp_path, *, ini=True, name="deck"):
    shutil.copy(FODO, tmp_path / f"{name}.dat")
    if ini:
        shutil.copy(ADS, tmp_path / f"{name}.ini")
    return tmp_path / f"{name}.dat"


def _dc_ini(path: Path) -> None:
    buf = bytearray(path.read_bytes())
    for k in ("eps_z1", "betz1"):
        buf[OFF[k]:OFF[k] + 8] = struct.pack("<d", 0.0)
    path.write_bytes(bytes(buf))


def _session_beam() -> dict | None:
    from linac_gen_gui.interphase.app import _settings, _SETTINGS_SESSION_BEAM
    raw = _settings().value(_SETTINGS_SESSION_BEAM)
    return json.loads(raw) if raw else None


# ---------------------------------------------------------------------
# Open Lattice… prompt
# ---------------------------------------------------------------------
def test_open_lattice_offers_the_sibling_ini_and_imports_on_yes(win, tmp_path, gui_message_boxes):
    deck = _project(tmp_path)
    before = win.beam_tab._energy.value()
    assert before != 20.0
    win.open_lattice(deck)                      # autouse stub answers Yes
    kinds = [b[0] for b in gui_message_boxes]
    assert "question" in kinds
    q = next(b for b in gui_message_boxes if b[0] == "question")
    assert "deck.ini" in q[2] and "replaces" in q[2]
    assert win.beam_tab._energy.value() == 20.0
    assert win.beam_tab._species.currentText() == "proton"
    assert abs(win.beam_tab._alpha_z.value() + 0.17661) < 1e-6
    assert win.beam_tab._npart.value() == 50000
    # the imported beam is the session beam a restart restores
    sb = _session_beam()
    assert sb and sb["energy"] == 20.0 and sb["species"] == "proton"
    assert win.state.project_dirty


def test_open_lattice_no_keeps_the_current_beam(win, tmp_path, monkeypatch, gui_message_boxes):
    from PyQt6.QtWidgets import QMessageBox
    from linac_gen_gui.interphase import app as app_mod
    deck = _project(tmp_path)
    before = win.beam_tab._energy.value()
    monkeypatch.setattr(app_mod.QMessageBox, "question",
                        staticmethod(lambda *a, **k: QMessageBox.StandardButton.No))
    win.open_lattice(deck)
    assert win.state.lattice is not None
    assert win.beam_tab._energy.value() == before
    sb = _session_beam()
    assert not sb or sb.get("energy") != 20.0


def test_open_lattice_without_a_sibling_never_asks(win, tmp_path, gui_message_boxes):
    deck = _project(tmp_path, ini=False)
    win.open_lattice(deck)
    assert win.state.lattice is not None
    assert not [b for b in gui_message_boxes if b[0] == "question"]


def test_open_lattice_ignores_a_non_tracewin_ini_with_a_note(win, tmp_path, gui_message_boxes):
    """A text `deck.ini` (some other tool's config) is not TraceWin's
    options file: no prompt that asserts otherwise, a status note instead."""
    deck = _project(tmp_path, ini=False)
    (tmp_path / "deck.ini").write_text("[not]\nbinary=1\n")
    notes = []
    win.state.status_message.connect(notes.append)
    before = win.beam_tab._energy.value()
    win.open_lattice(deck)
    assert win.state.lattice is not None
    assert not [b for b in gui_message_boxes if b[0] in ("question", "warning")]
    assert any("not a TraceWin options file" in n for n in notes)
    assert win.beam_tab._energy.value() == before


def test_open_lattice_with_a_truncated_real_ini_prompts_then_warns(win, tmp_path, gui_message_boxes):
    deck = _project(tmp_path, ini=False)
    (tmp_path / "deck.ini").write_bytes(ADS.read_bytes()[:20000])   # magic OK, size not
    before = win.beam_tab._energy.value()
    win.open_lattice(deck)
    assert win.state.lattice is not None
    assert [b for b in gui_message_boxes if b[0] == "question"]
    warn = [b for b in gui_message_boxes if b[0] == "warning"]
    assert warn and "31624" in warn[-1][2] and "44824" in warn[-1][2]
    assert win.beam_tab._energy.value() == before


def test_prompt_is_gated_on_a_tracewin_deck(win, tmp_path, gui_message_boxes):
    """Only TraceWin projects carry a .ini: a MAD-X deck with a stray
    sibling never gets the TraceWin prompt."""
    shutil.copy(ADS, tmp_path / "line.ini")
    (tmp_path / "line.madx").write_text("! not parsed here\n")
    win._offer_tracewin_ini(str(tmp_path / "line.madx"))
    assert not [b for b in gui_message_boxes if b[0] == "question"]
    (tmp_path / "line.dat").write_text("DRIFT 100 20 0\nEND\n")
    win._offer_tracewin_ini(str(tmp_path / "line.dat"))
    assert [b for b in gui_message_boxes if b[0] == "question"]


# ---------------------------------------------------------------------
# Beam-tab button
# ---------------------------------------------------------------------
def test_beam_tab_button_imports(win, tmp_path, monkeypatch, gui_message_boxes):
    from linac_gen_gui.interphase.tabs import beam_tab as bt_mod
    ini = tmp_path / "beam.ini"
    shutil.copy(ADS, ini)
    monkeypatch.setattr(bt_mod.QFileDialog, "getOpenFileName",
                        staticmethod(lambda *a, **k: (str(ini), "")))
    win.beam_tab._cutoff.setValue(2.5)          # a field the .ini does not carry
    win.beam_tab._import_ini_btn.click()
    win.pump(0.1)
    assert win.beam_tab._energy.value() == 20.0
    assert win.beam_tab._freq.value() == 100.0
    assert abs(win.beam_tab._current.value() - 5.0) < 1e-9
    assert abs(win.beam_tab._emit_nx.value() - 0.2) < 1e-6
    assert abs(win.beam_tab._beta_z.value() - 37.11) < 0.01
    assert win.beam_tab._cutoff.value() == 2.5   # kept
    assert "beam.ini" in win.beam_tab._status.text()
    assert not win.beam_tab._continuous.isChecked()
    assert win.state.project_dirty
    assert _session_beam()["energy"] == 20.0
    assert not [b for b in gui_message_boxes if b[0] == "critical"]


def test_beam_tab_dc_project_switches_to_continuous(win, tmp_path, monkeypatch):
    from linac_gen_gui.interphase.tabs import beam_tab as bt_mod
    ini = tmp_path / "lebt.ini"
    shutil.copy(ADS, ini)
    _dc_ini(ini)
    monkeypatch.setattr(bt_mod.QFileDialog, "getOpenFileName",
                        staticmethod(lambda *a, **k: (str(ini), "")))
    win.beam_tab._import_ini_btn.click()
    win.pump(0.1)
    assert win.beam_tab._continuous.isChecked()
    assert win.beam_tab._energy.value() == 20.0
    assert "warning" in win.beam_tab._status.text()


def test_beam_tab_bad_file_is_a_message_not_a_crash(win, tmp_path, monkeypatch, gui_message_boxes):
    from linac_gen_gui.interphase.tabs import beam_tab as bt_mod
    bad = tmp_path / "bad.ini"
    bad.write_text("[beam]\n")
    before = win.beam_tab._energy.value()
    monkeypatch.setattr(bt_mod.QFileDialog, "getOpenFileName",
                        staticmethod(lambda *a, **k: (str(bad), "")))
    win.beam_tab._import_ini_btn.click()
    win.pump(0.1)
    crit = [b for b in gui_message_boxes if b[0] == "critical"]
    assert crit and crit[-1][1] == "Import .ini failed"
    assert win.beam_tab._energy.value() == before


def test_beam_tab_import_api_returns_config_and_warnings(win, tmp_path):
    ini = tmp_path / "beam.ini"
    shutil.copy(ADS, ini)
    cfg, warns = win.beam_tab.import_tracewin_ini(str(ini), species="H-")
    assert isinstance(cfg, BeamConfig) and cfg.species == "H-"
    assert any("'Proton'" in w for w in warns)
    assert win.beam_tab._species.currentText() == "H-"


# ---------------------------------------------------------------------
# New Project wizard
# ---------------------------------------------------------------------
def _dialog(tmp_path):
    from linac_gen_gui.interphase.dialogs.new_project import NewProjectDialog
    return NewProjectDialog(None, start_dir=str(tmp_path), examples=[])


def test_wizard_checkbox_follows_the_sibling(qapp, tmp_path, gui_message_boxes):
    (tmp_path / "src").mkdir()
    (tmp_path / "lonely").mkdir()
    deck = _project(tmp_path / "src", ini=True)
    lonely = _project(tmp_path / "lonely", ini=False)
    d = _dialog(tmp_path)
    assert not d._ini_in.isEnabled()
    d._rb_import.setChecked(True)
    assert not d._ini_in.isEnabled()            # no path yet
    d._import_path.setText(str(lonely))
    assert not d._ini_in.isEnabled()            # no sibling
    d._import_path.setText(str(deck))
    assert d._ini_in.isEnabled()                # sibling found
    d._rb_blank.setChecked(True)
    assert not d._ini_in.isEnabled()
    d.deleteLater()


def test_wizard_copies_the_ini_with_a_copied_deck(qapp, tmp_path, gui_message_boxes):
    (tmp_path / "src").mkdir()
    deck = _project(tmp_path / "src")
    d = _dialog(tmp_path)
    d._name.setText("p1")
    d._rb_import.setChecked(True)
    d._import_path.setText(str(deck))
    d._ini_in.setChecked(True)
    d._accept()
    res = d.project_result()
    assert res and res["mode"] == "import"
    proj = Path(res["project_dir"])
    assert Path(res["lattice_path"]) == proj / "deck.dat"
    assert res["tracewin_ini"] == str(proj / "deck.ini")
    assert (proj / "deck.ini").read_bytes() == ADS.read_bytes()
    d.deleteLater()


def test_wizard_references_the_ini_in_place_without_copy(qapp, tmp_path, gui_message_boxes):
    (tmp_path / "src").mkdir()
    deck = _project(tmp_path / "src")
    d = _dialog(tmp_path)
    d._name.setText("p2")
    d._rb_import.setChecked(True)
    d._import_path.setText(str(deck))
    d._copy_in.setChecked(False)
    d._ini_in.setChecked(True)
    d._accept()
    res = d.project_result()
    assert res["lattice_path"] == str(deck)
    assert res["tracewin_ini"] == str(deck.with_suffix(".ini"))
    assert not (Path(res["project_dir"]) / "deck.ini").exists()
    d.deleteLater()


def test_wizard_unchecked_box_hands_back_nothing(qapp, tmp_path, gui_message_boxes):
    (tmp_path / "src").mkdir()
    deck = _project(tmp_path / "src")
    d = _dialog(tmp_path)
    d._name.setText("p3")
    d._rb_import.setChecked(True)
    d._import_path.setText(str(deck))
    d._accept()
    res = d.project_result()
    assert res["tracewin_ini"] is None
    assert not (Path(res["project_dir"]) / "deck.ini").exists()
    d.deleteLater()


def test_new_project_handler_writes_the_imported_beam(win, tmp_path, monkeypatch):
    from linac_gen_gui.interphase.dialogs import new_project as np_mod
    proj = tmp_path / "np"
    proj.mkdir()
    deck = _project(proj)
    canned = {"name": "np", "project_dir": str(proj), "lattice_path": str(deck),
              "mode": "import", "import_warnings": [],
              "tracewin_ini": str(deck.with_suffix(".ini"))}

    class _Stub:
        def __init__(self, *a, **k):
            pass

        def exec(self):
            from PyQt6.QtWidgets import QDialog
            return QDialog.DialogCode.Accepted

        def project_result(self):
            return canned

    monkeypatch.setattr(np_mod, "NewProjectDialog", _Stub)
    win._new_project()
    fp = proj / "np.lgproj"
    assert fp.is_file()
    data = json.loads(fp.read_text(encoding="utf-8"))
    assert data["beam"]["energy"] == 20.0 and data["beam"]["species"] == "proton"
    assert abs(data["beam"]["alpha_z"] + 0.17661) < 1e-9
    assert win.beam_tab._energy.value() == 20.0


# ---------------------------------------------------------------------
# Clamping, the assistant seam, wizard robustness, cleanup
# ---------------------------------------------------------------------
def test_beam_tab_import_reports_spinbox_clamping(win, tmp_path):
    ini = tmp_path / "big.ini"
    buf = bytearray(ADS.read_bytes())
    buf[OFF["nbr_part1"]:OFF["nbr_part1"] + 4] = struct.pack("<i", 5_000_000)
    buf[OFF["current1"]:OFF["current1"] + 8] = struct.pack("<d", 2.0)     # 2 A
    ini.write_bytes(bytes(buf))
    cfg, warns = win.beam_tab.import_tracewin_ini(str(ini))
    assert cfg.n_particles == 2_000_000 and cfg.current == 1000.0        # the applied truth
    assert cfg == win.state.beam_config
    assert any("n_particles" in w and "clamped" in w for w in warns)
    assert any("current" in w and "clamped" in w for w in warns)
    assert "N=2000000" in win.beam_tab._status.text()


def test_assistant_beam_write_reaches_the_form_and_the_saved_project(win, tmp_path):
    """load_lattice(tracewin_ini=) from the assistant must land in the
    Beam-tab widgets: Save Project serialises them, and any later Apply
    rebuilds the state from them."""
    from linac_gen.assist.tools import TOOLS
    from linac_gen_gui.interphase.dialogs.assistant_panel import _make_context
    from linac_gen.io.project import load_project
    deck = _project(tmp_path)

    class _Nav:                      # the panel's GUI hop, same thread here
        _app = win
        tab_labels, subtab_map = [], {}

        def run_on_gui(self, fn, timeout=3.0):
            return fn()

    ctx = _make_context(win.state, str(tmp_path), nav=_Nav())
    res = TOOLS["load_lattice"].fn(ctx, path=str(deck), tracewin_ini="auto")
    assert res["status"] == "ok", res
    assert win.beam_tab._energy.value() == 20.0
    assert win.beam_tab._species.currentText() == "proton"
    assert win.state.beam_config.energy == 20.0 and win.state.project_dirty
    win.beam_tab._apply(quiet=True)                        # must NOT revert
    assert win.state.beam_config.energy == 20.0
    out = tmp_path / "q.lgproj"
    assert win._write_project_file(str(out))
    assert load_project(out).beam.energy == 20.0


def test_wizard_unreadable_ini_still_creates_the_project(qapp, tmp_path, gui_message_boxes):
    import os
    if hasattr(os, "geteuid") and os.geteuid() == 0:
        pytest.skip("root reads everything")
    (tmp_path / "src").mkdir()
    deck = _project(tmp_path / "src")
    ini = deck.with_suffix(".ini")
    ini.chmod(0)
    try:
        d = _dialog(tmp_path)
        d._name.setText("p9")
        d._rb_import.setChecked(True)
        d._import_path.setText(str(deck))
        assert d._ini_in.isEnabled()
        d._ini_in.setChecked(True)
        d._accept()
        res = d.project_result()
        assert res is not None and res["tracewin_ini"] is None
        assert (Path(res["project_dir"]) / "deck.dat").is_file()
        assert not (Path(res["project_dir"]) / "deck.ini").exists()
        assert any(b[0] == "warning" and "could not be copied" in b[2]
                   for b in gui_message_boxes)
        d.deleteLater()
    finally:
        ini.chmod(0o644)


def test_new_project_failure_cleanup_removes_the_copied_ini(win, tmp_path, monkeypatch, gui_message_boxes):
    from linac_gen_gui.interphase import app as app_mod
    from linac_gen_gui.interphase.dialogs import new_project as np_mod
    proj = tmp_path / "np"
    proj.mkdir()
    deck = _project(proj)
    canned = {"name": "np", "project_dir": str(proj), "lattice_path": str(deck),
              "mode": "import", "import_warnings": [],
              "tracewin_ini": str(deck.with_suffix(".ini"))}

    class _Stub:
        def __init__(self, *a, **k):
            pass

        def exec(self):
            from PyQt6.QtWidgets import QDialog
            return QDialog.DialogCode.Accepted

        def project_result(self):
            return canned

    monkeypatch.setattr(np_mod, "NewProjectDialog", _Stub)

    def _boom(_fp):
        raise RuntimeError("synthetic parse failure")
    monkeypatch.setattr(app_mod, "_parse_lattice_file", _boom)
    win._new_project()
    assert not proj.exists()                     # deck AND ini removed, folder gone
    assert any(b[0] == "critical" for b in gui_message_boxes)
