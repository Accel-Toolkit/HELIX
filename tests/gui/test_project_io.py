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
    assert "beam" not in json.loads(target.read_text(encoding="utf-8"))


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
    assert (tmp_path / "taken.opmd.h5").read_text(encoding="utf-8") == "precious"


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


def test_export_madx_slot_writes_the_loaded_lattice(win, monkeypatch, tmp_path):
    """File → Export Lattice as MAD-X…: the app-level slot writes a MAD-X
    sequence from state.lattice with the Beam tab's species/energy, tells
    the status bar, and re-imports to the same quadrupole gradients."""
    from pathlib import Path
    from dataclasses import replace

    from linac_gen.core.config import BeamConfig
    from linac_gen.elements.quadrupole import Quadrupole
    from linac_gen.io.madx_parser import parse_madx
    from linac_gen.io.tracewin_parser import parse_tracewin
    from linac_gen_gui.interphase import app as app_mod

    repo = Path(__file__).resolve().parents[2]
    dat = repo / "examples" / "fodo_cell.dat"
    lat = parse_tracewin(str(dat))[0]
    win.state.set_lattice(lat, str(dat))
    win.state.set_beam_config(replace(BeamConfig(), species="H-", energy=3.0))

    infos, statuses = [], []
    monkeypatch.setattr(
        app_mod.QMessageBox, "information",
        staticmethod(lambda *a, **k: infos.append(a[2])
                     or app_mod.QMessageBox.StandardButton.Ok))
    win.state.status_message.connect(statuses.append)
    # A bare name typed into the dialog gets the .madx extension.
    monkeypatch.setattr(
        app_mod.QFileDialog, "getSaveFileName",
        staticmethod(lambda *a, **k: (str(tmp_path / "fodo_export"), "")))

    win._export_madx()

    out = tmp_path / "fodo_export.madx"
    assert out.exists()
    assert infos and "Reference: H- 3 MeV" in infos[-1]
    assert any("Exported MAD-X" in m and "0 warning(s)" in m for m in statuses)
    back, meta = parse_madx(str(out))
    assert meta["reference"].species.charge == -1
    g0 = [e.gradient for e in lat.elements if isinstance(e, Quadrupole)]
    g1 = [e.gradient for e in back.elements if isinstance(e, Quadrupole)]
    assert g1 == g0


def test_export_madx_slot_without_lattice_warns(win, monkeypatch):
    from linac_gen_gui.interphase import app as app_mod
    warned = []
    monkeypatch.setattr(
        app_mod.QMessageBox, "warning",
        staticmethod(lambda *a, **k: warned.append(a[1])
                     or app_mod.QMessageBox.StandardButton.Ok))
    called = []
    monkeypatch.setattr(app_mod.QFileDialog, "getSaveFileName",
                        staticmethod(lambda *a, **k: called.append(1) or ("", "")))
    win.state.set_lattice(None)
    win._export_madx()
    assert warned == ["No lattice"] and called == []


def test_export_madx_never_overwrites_the_imported_source(win, monkeypatch, tmp_path):
    """A lattice imported from MAD-X: the dialog proposes ``<stem>_helix.madx``
    and picking the source file itself is refused (the manual promises
    the MAD-X source is never overwritten)."""
    from pathlib import Path
    from dataclasses import replace

    from linac_gen.core.config import BeamConfig
    from linac_gen.io.madx_parser import parse_madx
    from linac_gen_gui.interphase import app as app_mod

    repo = Path(__file__).resolve().parents[2]
    src = tmp_path / "ring.madx"
    src.write_text((repo / "examples" / "madx" / "fodo.madx").read_text(encoding="utf-8"))
    lat, meta = parse_madx(str(src))
    win.state.set_lattice(lat, str(src))
    win.state.set_beam_config(replace(BeamConfig(), species="proton",
                                      energy=meta["reference"].w_kin))
    proposed, warned = [], []
    monkeypatch.setattr(
        app_mod.QFileDialog, "getSaveFileName",
        staticmethod(lambda *a, **k: proposed.append(a[2]) or (str(src), "")))
    monkeypatch.setattr(
        app_mod.QMessageBox, "warning",
        staticmethod(lambda *a, **k: warned.append(a[1])
                     or app_mod.QMessageBox.StandardButton.Ok))
    before = src.read_text(encoding="utf-8")
    win._export_madx()
    assert proposed[-1].endswith("ring_helix.madx")
    assert warned == ["Export refused"]
    assert src.read_text(encoding="utf-8") == before


def test_drift_single_push_round_trips_through_the_project(win):
    ct = win.convergence_tab
    assert ct._drift_single_push.isChecked() is True
    win._apply_project_dict({"__kind__": "linac_gen_project",
                             "convergence": {"drift_single_push": False}}, silent=True)
    assert ct._drift_single_push.isChecked() is False
    data = win._collect_project_dict([], None)
    assert data["convergence"]["drift_single_push"] is False
    # a project written before the option existed leaves it where it is
    win._apply_project_dict({"__kind__": "linac_gen_project",
                             "convergence": {"step1_per_m": 100.0}}, silent=True)
    assert ct._drift_single_push.isChecked() is False


def test_drift_single_push_project_value_uses_the_cli_coercion(win):
    """A hand-edited "false" in the project file means off, as it does for the CLI."""
    ct = win.convergence_tab
    win._apply_project_dict({"__kind__": "linac_gen_project",
                             "convergence": {"drift_single_push": "false"}}, silent=True)
    assert ct._drift_single_push.isChecked() is False
    win._apply_project_dict({"__kind__": "linac_gen_project",
                             "convergence": {"drift_single_push": "on"}}, silent=True)
    assert ct._drift_single_push.isChecked() is True
