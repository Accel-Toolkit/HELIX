"""New Project wizard — dialog validation, window handler, save flow.

Every modal (QMessageBox / QFileDialog / the wizard itself at window
level) is monkeypatched: an offscreen modal never returns (documented
forever-hang), so tests must never let one open for real.
"""
from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

import pytest

pytest.importorskip("PyQt6")


@pytest.fixture()
def win(qapp):
    from linac_gen_gui.interphase.app import InterphaseWindow
    w = InterphaseWindow()
    yield w
    w.close()
    w.deleteLater()


@pytest.fixture()
def dlg_mod():
    from linac_gen_gui.interphase.dialogs import new_project
    return new_project


def _make_dialog(dlg_mod, tmp_path, examples=None):
    d = dlg_mod.NewProjectDialog(None, start_dir=str(tmp_path),
                                 examples=examples or [])
    return d


def _silence_warnings(dlg_mod, monkeypatch, sink: list):
    monkeypatch.setattr(
        dlg_mod.QMessageBox, "warning",
        staticmethod(lambda *a, **k: sink.append(a[2]) or 0))


# ---------------------------------------------------------------------
# Dialog level
# ---------------------------------------------------------------------

def test_blank_happy_path_writes_parsable_starter(dlg_mod, tmp_path, qapp):
    d = _make_dialog(dlg_mod, tmp_path)
    d._name.setText("demo_proj")
    d._accept()
    res = d.project_result()
    assert res and res["mode"] == "blank"
    proj = Path(res["project_dir"])
    assert proj == tmp_path / "demo_proj"
    dat = Path(res["lattice_path"])
    assert dat.is_file() and dat.parent == proj

    from linac_gen.io.tracewin_parser import parse_tracewin
    lattice, _meta = parse_tracewin(str(dat))
    assert len(lattice.elements) >= 1
    d.deleteLater()


@pytest.mark.parametrize("bad", ["", "a<b", "CON", "name.", "com1.dat"])
def test_name_veto_matrix(dlg_mod, tmp_path, qapp, monkeypatch, bad):
    warned: list = []
    _silence_warnings(dlg_mod, monkeypatch, warned)
    d = _make_dialog(dlg_mod, tmp_path)
    d._name.setText(bad)
    d._accept()
    assert d.project_result() is None
    assert warned, f"no veto for name {bad!r}"
    if bad:
        assert not (tmp_path / bad).exists()
    d.deleteLater()


def test_surrounding_whitespace_is_normalized(dlg_mod, tmp_path, qapp):
    """Leading/trailing spaces are stripped, not rejected — friendlier
    than a veto, and the on-disk name can never carry them."""
    d = _make_dialog(dlg_mod, tmp_path)
    d._name.setText("  padded  ")
    d._accept()
    res = d.project_result()
    assert res and res["name"] == "padded"
    assert (tmp_path / "padded").is_dir()
    d.deleteLater()


def test_nonempty_target_refused(dlg_mod, tmp_path, qapp, monkeypatch):
    warned: list = []
    _silence_warnings(dlg_mod, monkeypatch, warned)
    tgt = tmp_path / "busy"
    tgt.mkdir()
    (tgt / "occupied.txt").write_text("x", encoding="utf-8")
    d = _make_dialog(dlg_mod, tmp_path)
    d._name.setText("busy")
    d._accept()
    assert d.project_result() is None and warned
    d.deleteLater()


def _write_src_dat(tmp_path) -> Path:
    src = tmp_path / "src_lattice.dat"
    src.write_text("FREQ 162.5\nDRIFT 100 15 0\nEND\n", encoding="utf-8")
    return src


def test_import_copy_on_vs_off(dlg_mod, tmp_path, qapp):
    src = _write_src_dat(tmp_path)
    for copy_in, name in ((True, "cp_on"), (False, "cp_off")):
        d = _make_dialog(dlg_mod, tmp_path)
        d._name.setText(name)
        d._rb_import.setChecked(True)
        d._import_path.setText(str(src))
        d._copy_in.setChecked(copy_in)
        d._accept()
        res = d.project_result()
        assert res and res["mode"] == "import"
        inside = Path(res["lattice_path"]).parent == tmp_path / name
        assert inside is copy_in
        d.deleteLater()


def test_example_always_copied(dlg_mod, tmp_path, qapp):
    ex = _write_src_dat(tmp_path)
    d = _make_dialog(dlg_mod, tmp_path, examples=[("Demo", str(ex))])
    d._name.setText("from_example")
    d._rb_example.setChecked(True)
    d._accept()
    res = d.project_result()
    assert res and res["mode"] == "example"
    assert Path(res["lattice_path"]).parent == tmp_path / "from_example"
    assert ex.is_file()          # source untouched
    d.deleteLater()


def test_empty_examples_disable_radio(dlg_mod, tmp_path, qapp):
    d = _make_dialog(dlg_mod, tmp_path, examples=[])
    assert not d._rb_example.isEnabled()
    d.deleteLater()


# ---------------------------------------------------------------------
# Window level
# ---------------------------------------------------------------------

def _stub_dialog_factory(canned: dict | None):
    class _Stub:
        def __init__(self, *a, **k):
            _Stub.instantiated = True

        def exec(self):
            from PyQt6.QtWidgets import QDialog
            return (QDialog.DialogCode.Accepted if canned
                    else QDialog.DialogCode.Rejected)

        def project_result(self):
            return canned
    _Stub.instantiated = False
    return _Stub


def _premade_project(tmp_path, name="stubproj") -> dict:
    proj = tmp_path / name
    proj.mkdir()
    dat = proj / f"{name}.dat"
    dat.write_text("FREQ 162.5\nDRIFT 100 15 0\nEND\n", encoding="utf-8")
    return {"name": name, "project_dir": str(proj),
            "lattice_path": str(dat), "mode": "blank"}


def test_handler_assembles_project(win, tmp_path, monkeypatch):
    from linac_gen_gui.interphase.dialogs import new_project as np_mod
    from linac_gen_gui.interphase import app as app_mod

    canned = _premade_project(tmp_path)
    monkeypatch.setattr(np_mod, "NewProjectDialog",
                        _stub_dialog_factory(canned))
    win._new_project()

    fp = Path(canned["project_dir"]) / "stubproj.lgproj"
    assert fp.is_file()
    data = json.loads(fp.read_text(encoding="utf-8"))
    assert data["calc_dir"] == "runs"
    assert not os.path.isabs(data["lattice_path"])
    assert win.state.lattice is not None

    s = app_mod._settings()
    assert s.value(app_mod._SETTINGS_LAST_PROJECT) == str(fp)
    assert str(Path(canned["project_dir"]) / "runs") == \
        s.value(app_mod._SETTINGS_CALC_DIR)
    assert str(fp) in app_mod._recent_projects_load()
    assert win.state.project_dirty is False


def test_handler_respects_dirty_veto(win, tmp_path, monkeypatch):
    from linac_gen_gui.interphase.dialogs import new_project as np_mod
    stub = _stub_dialog_factory(_premade_project(tmp_path, "vetoed"))
    monkeypatch.setattr(np_mod, "NewProjectDialog", stub)
    monkeypatch.setattr(win, "_confirm_discard", lambda *a, **k: False)
    win._new_project()
    assert stub.instantiated is False
    assert not (tmp_path / "vetoed" / "vetoed.lgproj").exists()


def test_created_project_survives_folder_move(win, tmp_path, monkeypatch):
    from linac_gen_gui.interphase.dialogs import new_project as np_mod
    from linac_gen_gui.interphase import app as app_mod

    canned = _premade_project(tmp_path, "mover")
    monkeypatch.setattr(np_mod, "NewProjectDialog",
                        _stub_dialog_factory(canned))
    win._new_project()

    moved = tmp_path / "relocated"
    shutil.move(canned["project_dir"], moved)
    assert win._restore_last_project(str(moved / "mover.lgproj"))
    assert win.state.lattice is not None
    assert app_mod._settings().value(app_mod._SETTINGS_CALC_DIR) == \
        str(moved / "runs")


def test_save_project_silent_then_save_as_prompts(win, tmp_path, monkeypatch):
    from linac_gen_gui.interphase.dialogs import new_project as np_mod
    from linac_gen_gui.interphase import app as app_mod

    canned = _premade_project(tmp_path, "saver")
    monkeypatch.setattr(np_mod, "NewProjectDialog",
                        _stub_dialog_factory(canned))
    win._new_project()
    fp = Path(canned["project_dir"]) / "saver.lgproj"
    before = fp.stat().st_mtime_ns

    # Plain Save must not open any dialog now that a project is current.
    def _boom(*a, **k):
        raise AssertionError("Save Project must not prompt")
    monkeypatch.setattr(app_mod.QFileDialog, "getSaveFileName",
                        staticmethod(_boom))
    win.state.mark_project_dirty()
    win._save_project()
    assert fp.stat().st_mtime_ns >= before
    assert win.state.project_dirty is False

    # Save As always prompts; feed it a new path.
    other = tmp_path / "elsewhere.lgproj"
    monkeypatch.setattr(
        app_mod.QFileDialog, "getSaveFileName",
        staticmethod(lambda *a, **k: (str(other), "")))
    win._save_project_as()
    assert other.is_file()


def test_target_exists_as_plain_file_vetoed(dlg_mod, tmp_path, qapp,
                                            monkeypatch):
    """A FILE named like the project must veto cleanly, not raise
    NotADirectoryError from iterdir()."""
    warned: list = []
    _silence_warnings(dlg_mod, monkeypatch, warned)
    (tmp_path / "blocked").write_text("x", encoding="utf-8")
    d = _make_dialog(dlg_mod, tmp_path)
    d._name.setText("blocked")
    d._accept()
    assert d.project_result() is None and warned
    d.deleteLater()


def test_cancelled_write_ends_previous_project_session(win, tmp_path,
                                                       monkeypatch):
    """HIGH regression: if the .lgproj write is cancelled, the previous
    project must NOT stay current — plain Save would silently rewrite
    it around the newly adopted lattice."""
    from linac_gen_gui.interphase.dialogs import new_project as np_mod
    from linac_gen_gui.interphase import app as app_mod

    old_proj = tmp_path / "old.lgproj"
    old_proj.write_text("{}", encoding="utf-8")
    app_mod._settings().setValue(app_mod._SETTINGS_LAST_PROJECT,
                                 str(old_proj))

    canned = _premade_project(tmp_path, "halfway")
    monkeypatch.setattr(np_mod, "NewProjectDialog",
                        _stub_dialog_factory(canned))
    monkeypatch.setattr(win, "_write_project_file",
                        lambda *a, **k: False)
    win._new_project()
    assert app_mod._settings().value(
        app_mod._SETTINGS_LAST_PROJECT) is None


def test_plain_save_preserves_relative_calc_dir(win, tmp_path,
                                                monkeypatch):
    """HIGH regression: the first plain Save must not bake the absolute
    calc dir back into a wizard-created (portable) project."""
    import json
    from linac_gen_gui.interphase.dialogs import new_project as np_mod
    from linac_gen_gui.interphase import app as app_mod

    canned = _premade_project(tmp_path, "portable")
    monkeypatch.setattr(np_mod, "NewProjectDialog",
                        _stub_dialog_factory(canned))
    win._new_project()
    fp = Path(canned["project_dir"]) / "portable.lgproj"
    assert json.loads(fp.read_text(encoding="utf-8"))["calc_dir"] == "runs"

    win.state.mark_project_dirty()
    monkeypatch.setattr(
        app_mod.QFileDialog, "getSaveFileName",
        staticmethod(lambda *a, **k: (_ for _ in ()).throw(
            AssertionError("must not prompt"))))
    win._save_project()
    assert json.loads(fp.read_text(encoding="utf-8"))["calc_dir"] == "runs"


def test_corrupt_import_cleans_up_created_folder(win, tmp_path,
                                                 monkeypatch):
    """A lattice that fails to parse must not orphan the just-created
    project folder (which would block a retry under the same name)."""
    from linac_gen_gui.interphase.dialogs import new_project as np_mod
    from linac_gen_gui.interphase import app as app_mod

    proj = tmp_path / "corrupt"
    proj.mkdir()
    canned = {"name": "corrupt", "project_dir": str(proj),
              "lattice_path": str(proj / "missing.dat"), "mode": "blank"}
    monkeypatch.setattr(np_mod, "NewProjectDialog",
                        _stub_dialog_factory(canned))
    crits: list = []
    monkeypatch.setattr(
        app_mod.QMessageBox, "critical",
        staticmethod(lambda *a, **k: crits.append(a[2]) or 0))
    win._new_project()
    assert crits, "no error dialog for corrupt import"
    assert not proj.exists()
