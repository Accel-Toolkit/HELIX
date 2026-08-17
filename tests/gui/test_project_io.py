"""Project (.lgproj) save/load regressions at the InterphaseWindow level."""
from __future__ import annotations

from dataclasses import asdict, replace

import pytest

pytest.importorskip("PyQt6")


@pytest.fixture()
def win(qapp):
    from linac_gen_gui.interphase.app import InterphaseWindow
    w = InterphaseWindow()
    yield w
    w.close()
    w.deleteLater()


def test_project_load_pushes_beam_into_state(win):
    """Regression: loading a project only refreshed the Beam-tab WIDGETS;
    state.beam_config kept the previous project's applied beam, so every
    run silently used the old physics while the form showed the new."""
    from linac_gen.core.config import BeamConfig

    old = replace(BeamConfig(), current=1.0)
    win.state.set_beam_config(old)

    new = replace(BeamConfig(), current=7.5, energy=2.5)
    win._apply_project_dict(
        {"__kind__": "linac_gen_project", "beam": asdict(new)}, silent=True)

    assert win.state.beam_config is not old
    assert win.state.beam_config.current == 7.5
    assert win.state.beam_config.energy == 2.5
    # Widgets agree with state after the load.
    assert win.beam_tab.get_beam_config().current == 7.5


def test_collect_reports_invalid_beam(win, monkeypatch):
    """Regression: an invalid Beam form was swallowed by a bare
    ``except: pass`` — the project was written WITHOUT its beam section
    and nobody was told."""
    monkeypatch.setattr(
        win.beam_tab, "get_beam_config",
        lambda: (_ for _ in ()).throw(ValueError("mismatch_x must be > -100 %")))
    warnings: list[str] = []
    data = win._collect_project_dict(warnings)
    assert "beam" not in data
    assert warnings and "mismatch_x" in warnings[0]
    assert "convergence" in data          # the rest is still captured


def test_save_project_cancels_on_missing_section(win, monkeypatch, tmp_path):
    from linac_gen_gui.interphase import app as app_mod

    # Plain Save now writes silently to a known current project; this
    # test exercises the prompt path, so make sure none is current.
    app_mod._settings().remove(app_mod._SETTINGS_LAST_PROJECT)

    target = tmp_path / "broken.lgproj"
    monkeypatch.setattr(
        app_mod.QFileDialog, "getSaveFileName",
        staticmethod(lambda *a, **k: (str(target), "")))
    monkeypatch.setattr(
        win.beam_tab, "get_beam_config",
        lambda: (_ for _ in ()).throw(ValueError("bad form")))

    # User picks Cancel → nothing written.
    monkeypatch.setattr(
        app_mod.QMessageBox, "warning",
        staticmethod(lambda *a, **k: app_mod.QMessageBox.StandardButton.Cancel))
    win._save_project()
    assert not target.exists()

    # User picks Save → file written, minus the beam section.
    monkeypatch.setattr(
        app_mod.QMessageBox, "warning",
        staticmethod(lambda *a, **k: app_mod.QMessageBox.StandardButton.Save))
    win._save_project()
    assert target.exists()
    import json
    assert "beam" not in json.loads(target.read_text())


def test_project_load_consolidates_section_warnings(win, monkeypatch):
    """A bad section must not abort the load nor pop one dialog per
    failure — warnings are collected, ONE summary dialog appears, and
    the return value lets callers qualify their status message."""
    from linac_gen_gui.interphase import app as app_mod

    dialogs = []
    monkeypatch.setattr(
        app_mod.QMessageBox, "warning",
        staticmethod(lambda *a, **k: dialogs.append(a)))

    warns = win._apply_project_dict({
        "__kind__": "linac_gen_project",
        "beam": {"mismatch_x": -200.0},          # BeamConfig rejects
        "convergence": {"grid_nx": "not-an-int"},  # int() raises
    })

    assert len(warns) == 2
    assert any("Beam" in w for w in warns)
    assert any("Numerics" in w for w in warns)
    assert len(dialogs) == 1                     # consolidated, not per-section

    # Silent path: no dialogs at all, warnings still returned.
    dialogs.clear()
    warns = win._apply_project_dict(
        {"beam": {"mismatch_x": -200.0}}, silent=True)
    assert len(warns) == 1 and dialogs == []


def test_openpmd_export_extension_and_overwrite_guard(win, monkeypatch,
                                                      tmp_path):
    """Review finding: appending .opmd.h5 AFTER the save dialog could
    silently overwrite a different existing file (the dialog's own
    prompt covered only the typed name), and 'x.opmd' became
    'x.opmd.opmd.h5'."""
    from linac_gen_gui.interphase import app as app_mod

    win.state.set_results(object())
    saved = []
    monkeypatch.setattr(
        "linac_gen.io.openpmd_output.save_results_openpmd",
        lambda results, path, **k: saved.append(path))
    # The success path ends in a modal information box — stub it or the
    # offscreen test blocks forever in its exec loop.
    monkeypatch.setattr(
        app_mod.QMessageBox, "information",
        staticmethod(lambda *a, **k: app_mod.QMessageBox.StandardButton.Ok))

    def _pick(name):
        monkeypatch.setattr(
            app_mod.QFileDialog, "getSaveFileName",
            staticmethod(lambda *a, **k: (str(tmp_path / name), "")))

    # Bare name → canonical double extension.
    _pick("results")
    win._export_openpmd()
    assert saved[-1].endswith("results.opmd.h5")

    # 'x.opmd' → single .h5 appended, not .opmd.h5 twice.
    _pick("x.opmd")
    win._export_openpmd()
    assert saved[-1].endswith("x.opmd.h5")
    assert ".opmd.opmd." not in saved[-1]

    # Appended path collides with an existing file → user says No →
    # nothing written.
    (tmp_path / "taken.opmd.h5").write_text("precious")
    monkeypatch.setattr(
        app_mod.QMessageBox, "question",
        staticmethod(lambda *a, **k: app_mod.QMessageBox.StandardButton.No))
    n_before = len(saved)
    _pick("taken")
    win._export_openpmd()
    assert len(saved) == n_before
    assert (tmp_path / "taken.opmd.h5").read_text() == "precious"


def test_mismatch_spin_cannot_reach_invalid_floor(qapp):
    """The spin floor used to be exactly -100 %, which BeamConfig rejects
    (<= -100 zeroes the emittance) — dialing the spin to its minimum made
    get_beam_config() raise and fed the silent save-drop above."""
    from linac_gen_gui.interphase.state import AppState
    from linac_gen_gui.interphase.tabs.beam_tab import BeamTab

    tab = BeamTab(AppState())
    for spin in (tab._mx, tab._my, tab._mz):
        spin.setValue(-1e9)               # clamps to the widget minimum
    cfg = tab.get_beam_config()           # must not raise
    assert cfg.mismatch_x > -100.0
    tab.deleteLater()
