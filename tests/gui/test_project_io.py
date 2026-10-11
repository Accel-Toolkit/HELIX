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


def test_record_action_scan_round_trips_through_the_project(win):
    ct = win.convergence_tab
    assert ct._record_action_scan.isChecked() is True      # on by default
    win._apply_project_dict({"__kind__": "linac_gen_project",
                             "convergence": {"record_action_scan": False}},
                            silent=True)
    assert ct._record_action_scan.isChecked() is False     # project wins
    win._apply_project_dict({"__kind__": "linac_gen_project",
                             "convergence": {"record_action_scan": True}},
                            silent=True)
    assert ct._record_action_scan.isChecked() is True
    assert win._collect_project_dict([], None)["convergence"][
        "record_action_scan"] is True
    # hand-edited strings use the CLI coercion
    win._apply_project_dict({"__kind__": "linac_gen_project",
                             "convergence": {"record_action_scan": "false"}},
                            silent=True)
    assert ct._record_action_scan.isChecked() is False
    # a project written before the option existed leaves it where it is
    ct._record_action_scan.setChecked(True)
    win._apply_project_dict({"__kind__": "linac_gen_project",
                             "convergence": {"step1_per_m": 100.0}},
                            silent=True)
    assert ct._record_action_scan.isChecked() is True


@pytest.mark.parametrize("widget", ["_record_action_scan", "_record_density",
                                    "_record_substeps"])
def test_recording_switches_mark_the_project_dirty(win, widget):
    """They are saved in the project, so toggling one is an unsaved change
    (and loading a project still ends clean)."""
    ct = win.convergence_tab
    win.state.mark_project_clean()
    box = getattr(ct, widget)
    box.setChecked(not box.isChecked())
    assert win.state.project_dirty
    win._apply_project_dict({"__kind__": "linac_gen_project",
                             "convergence": {widget.lstrip("_"): False}},
                            silent=True)
    assert not win.state.project_dirty


def test_density_settings_mark_the_project_dirty(win):
    ct = win.convergence_tab
    win.state.mark_project_clean()
    ct._density_bins.setValue(ct._density_bins.value() + 8)
    assert win.state.project_dirty
    win.state.mark_project_clean()
    ct._density_extent.setValue(ct._density_extent.value() + 1.0)
    assert win.state.project_dirty


# ---------------------------------------------------------------------------
# space-charge method in the project file (same resolver as the CLI)
# ---------------------------------------------------------------------------
def _sc_fields(cfg):
    import dataclasses
    return {f.name: getattr(cfg, f.name) for f in dataclasses.fields(cfg)}


def test_space_charge_method_round_trips_through_the_project(win):
    from linac_gen.pic import methods
    ct = win.convergence_tab
    ct._sc_method.setCurrentIndex(ct._sc_method.findData("tsc_adaptive_reuse2"))
    ct._fixed_nx.setValue(64)
    conv = win._collect_project_dict([])["convergence"]
    assert conv["sc_method"] == "tsc_adaptive_reuse2"
    assert (conv["reuse_every"], conv["grid_card"], conv["green_build"]) == (
        2, "auto", "symmetric")
    gui = ct.current_sc_config(5.0)
    assert _sc_fields(methods.resolve_sc(conv, {})) == _sc_fields(gui)
    ct._sc_method.setCurrentIndex(ct._sc_method.findData("cic_fixed"))
    win._apply_project_dict({"__kind__": "linac_gen_project",
                             "convergence": conv}, silent=True)
    assert ct.sc_method_name() == "tsc_adaptive_reuse2"
    assert _sc_fields(ct.current_sc_config(5.0)) == _sc_fields(gui)


from tests.dataguard import needs  # noqa: E402


@needs("examples/pipii/mebt/mebt.lgproj")
def test_legacy_project_shows_its_method_and_runs_what_the_cli_runs(win):
    """A shipped project (no sc_method): the tab shows the method its
    settings match, and builds exactly the configuration the CLI builds."""
    import json
    import os

    from linac_gen.cli import common
    here = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(here, "..", "..", "examples", "pipii", "mebt",
                        "mebt.lgproj")
    data = json.load(open(path))
    assert "sc_method" not in data["convergence"]
    win._apply_project_dict(data, silent=True)
    ct = win.convergence_tab
    assert ct.sc_method_name() == "cic_fixed"

    class _Beam:
        current = 5.0
    cli = common.make_sc_config(_Beam(), data["convergence"], {})
    assert _sc_fields(ct.current_sc_config(5.0)) == _sc_fields(cli)


def test_project_without_space_charge_keys_does_not_keep_the_old_ones(win):
    """Regression: SC widgets kept the previously opened project's values
    when the next project did not store them."""
    from linac_gen.pic import methods
    ct = win.convergence_tab
    ct._sc_method.setCurrentIndex(ct._sc_method.findData("tsc_adaptive"))
    conv = {"step1_per_m": 100.0}
    win._apply_project_dict({"__kind__": "linac_gen_project",
                             "convergence": conv}, silent=True)
    want = methods.resolve_sc(conv, {})
    got = ct.current_sc_config(5.0)
    for key in ("kernel", "grid_mode", "nx", "grid_extent", "green_build",
                "grid_card", "reuse_every"):
        assert getattr(got, key) == getattr(want, key), key


def test_project_with_an_unknown_method_warns_and_keeps_loading(win,
                                                               monkeypatch):
    from linac_gen_gui.interphase import app as app_mod
    seen = []
    monkeypatch.setattr(app_mod.QMessageBox, "warning",
                        staticmethod(lambda *a, **k: seen.append(a)))
    win._apply_project_dict({"__kind__": "linac_gen_project",
                             "convergence": {"sc_method": "cic_fast",
                                             "step1_per_m": 70.0}},
                            silent=True)
    assert win.convergence_tab._fixed_step1.value() == 70.0



def test_legacy_project_with_the_torch_engine_still_runs_torch_in_the_gui(win):
    conv = {"grid_nx": 32, "grid_extent_sigma": 6.0, "kernel": "cic",
            "grid_mode": "fixed", "green_kind": "igf", "backend": "auto",
            "sc_backend": "torch", "dc_kernel": "uniform"}
    win._apply_project_dict({"__kind__": "linac_gen_project",
                             "convergence": conv}, silent=True)
    ct = win.convergence_tab
    sc = ct.current_sc_config(5.0)
    assert (sc.sc_backend, sc.grid_mode) == ("torch", "adaptive")
    assert ct.sc_method_name() == "differentiable_pic"


def test_settings_the_tab_cannot_hold_are_reported(win):
    conv = {"sc_method": "tsc_fixed", "grid_nx": 48,
            "grid_extent_sigma": 25.0, "csr_bins": 300}
    warns = win._apply_project_dict({"__kind__": "linac_gen_project",
                                     "convergence": conv}, silent=True)
    text = "\n".join(warns)
    assert "grid_extent" in text and "csr_bins" in text


def test_invalid_block_falls_back_to_the_stored_keys(win):
    ct = win.convergence_tab
    ct._sc_method.setCurrentIndex(ct._sc_method.findData("cic_adaptive_reuse4"))
    conv = {"sc_method": "cic_fast", "grid_nx": 40, "kernel": "tsc",
            "grid_mode": "fixed"}
    warns = win._apply_project_dict({"__kind__": "linac_gen_project",
                                     "convergence": conv}, silent=True)
    sc = ct.current_sc_config(5.0)
    assert (sc.nx, sc.kernel, sc.grid_mode, sc.reuse_every) == (40, "tsc",
                                                                "fixed", 1)
    assert any("cic_fast" in w for w in warns)


def test_a_save_keeps_the_migration_commands_undo_record(win, tmp_path):
    """scripts/sc_method_migrate.py --switch keeps the old space-charge
    keys under sc_previous; a save of the open project keeps them (also
    the null of a project that had no block), so --restore still works."""
    import json

    from linac_gen_gui.interphase import app as app_mod
    s = app_mod._settings()
    old = s.value(app_mod._SETTINGS_LAST_PROJECT)
    try:
        for prev in ({"grid_nx": 48, "kernel": "tsc", "grid_mode": "fixed"},
                     None):
            data = {"__kind__": "linac_gen_project",
                    "convergence": {"sc_method": "tsc_adaptive",
                                    "sc_previous": prev}}
            path = tmp_path / "p.lgproj"
            path.write_text(json.dumps(data))
            win._apply_project_dict(data, silent=True)
            s.setValue(app_mod._SETTINGS_LAST_PROJECT, str(path))
            out = win._collect_project_dict(None, project_path=str(path))
            assert out["convergence"]["sc_previous"] == prev
        s.remove(app_mod._SETTINGS_LAST_PROJECT)       # no open project
        assert "sc_previous" not in win._collect_project_dict(
            None)["convergence"]
    finally:
        if old:
            s.setValue(app_mod._SETTINGS_LAST_PROJECT, old)
        else:
            s.remove(app_mod._SETTINGS_LAST_PROJECT)
