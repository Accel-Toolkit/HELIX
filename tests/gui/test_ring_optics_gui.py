"""Ring mode in the GUI, milestone M1: the Ring & Optics page (through its
real button and worker), the element-model choices and their place in the
project, "use this Twiss in the Beam tab", and the Lattice tab's banner for a
lattice that closes on itself."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("PyQt6")

ROOT = Path(__file__).resolve().parents[2]
FIXTURE = ROOT / "tests" / "ring" / "fixtures" / "simple_booster.madx"
REF = json.loads((ROOT / "tests" / "ring" / "fixtures" / "refs" / "simple_booster_madx.json").read_text())
FODO = ROOT / "examples" / "fodo_cell.dat"


def _ring_on(win):
    act = win._toolbar._ring_action
    if not act.isChecked():
        act.trigger()
    win.pump(0.1)
    return win.ring_tab


def _compute(win, tab):
    tab._optics_btn.click()
    win.wait_worker(tab._optics_worker)
    assert tab._optics is not None, tab._optics_status.text()
    return tab._optics["optics"]


def test_optics_through_the_button(win):
    win.open_lattice(FIXTURE)
    tab = _ring_on(win)
    o = _compute(win, tab)
    assert o.qx == pytest.approx(REF["q1"], abs=1e-6) and o.qy == pytest.approx(REF["q2"], abs=1e-6)
    assert o.dqx == pytest.approx(REF["dq1_delta"], abs=0.02)
    assert "Q 6.7100 / 6.8100" in tab._chips.text()
    assert "Q'x -8.65" in tab._optics_text.text()
    assert len(tab._curve_bx.getData()[0]) == len(win.state.lattice.elements)
    assert tab._optics_btn.isEnabled() and tab._to_beam_btn.isEnabled()
    # the window's lattice is untouched (the worker compiles a copy)
    assert win.state.lattice.elements[0].name == "sa"


def test_model_choice_is_used_and_saved(win):
    win.open_lattice(FIXTURE)
    tab = _ring_on(win)
    tab._model.setCurrentIndex(1)                     # curvilinear
    assert not tab._edge.isEnabled() and tab._edge.currentData() == "linear"
    o = _compute(win, tab)
    assert o.dqx == pytest.approx(-7.99, abs=0.02)    # curvilinear bend multipoles
    cfg = win._collect_project_dict()["ring"]["config"]
    assert cfg["model"] == "curvilinear" and cfg["edge_model"] == "linear"
    # a project with these settings shows them again
    data = win._collect_project_dict()
    tab._model.setCurrentIndex(0)
    win._apply_project_dict(data, silent=True)
    assert win.ring_tab._model.currentData() == "curvilinear"
    # (and under the model's name of earlier versions)
    from linac_gen.ring.config import MODEL_ALIASES
    data["ring"]["config"]["model"] = next(k for k, v in MODEL_ALIASES.items()
                                           if v == "curvilinear")
    tab._model.setCurrentIndex(0)
    win._apply_project_dict(data, silent=True)
    assert win.ring_tab._model.currentData() == "curvilinear"


def test_six_d_optics_show_the_synchrotron_tune(win):
    win.open_lattice(FIXTURE)
    tab = _ring_on(win)
    tab._six_d.setChecked(True)
    o = _compute(win, tab)
    assert o.mode == "6d" and o.qs == pytest.approx(0.02366, abs=1e-4)
    assert "Qs 0.0236" in tab._optics_text.text() and "bucket" in tab._optics_text.text()


def _use_matched(win, tab):
    tab._to_beam_btn.click()
    win.wait_worker(tab._matched_worker)
    assert tab._matched_worker is None and tab._to_beam_btn.isEnabled()
    return win.beam_tab.get_beam_config()


def test_the_matched_beam_goes_to_the_beam_tab(win):
    # (no optics computed first: the button computes the run's first turn)
    import math

    from linac_gen.ring import RingConfig, RingSimulation
    win.open_lattice(FIXTURE)
    tab = _ring_on(win)
    assert tab._optics is None and tab._to_beam_btn.isEnabled()
    win.state.mark_project_clean()
    cfg = _use_matched(win, tab)
    text = tab._optics_status.text()
    assert text.startswith("Beam tab: the beam matched to the ring at the start of the run "
                           "(proton 800 MeV, fixed energy; RF 44.7044 MHz)")
    # the deck declares its beam: its species and energy come too
    assert cfg.species == "proton" and cfg.energy == pytest.approx(800.0, rel=1e-12)
    st = REF["start"]
    assert cfg.beta_x == pytest.approx(st["betx"], rel=1e-4)
    assert cfg.beta_y == pytest.approx(st["bety"], rel=1e-4)
    assert cfg.alpha_x == pytest.approx(st["alfx"], abs=1e-4)
    assert cfg.alpha_y == pytest.approx(st["alfy"], abs=1e-4)
    # dispersion per MeV: D [mm/MeV] = 1e3 D [m] / (beta0 p0c [MeV])
    per_mev = 1e3 / (REF["beta0"] * REF["pc_gev"] * 1e3)
    assert cfg.disp_x == pytest.approx(st["dx_delta"] * per_mev, rel=1e-3)
    assert cfg.disp_xp == pytest.approx(st["dpx_delta"] * per_mev, abs=1e-6)
    assert cfg.disp_y == pytest.approx(0.0, abs=1e-9) and cfg.disp_yp == pytest.approx(0.0, abs=1e-9)
    # a ring without errors: the beam on the axis
    assert (cfg.centroid_x, cfg.centroid_xp, cfg.centroid_y, cfg.centroid_yp, cfg.centroid_dphi,
            cfg.centroid_dw) == (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    # the longitudinal Twiss are the engine's, the phase at the Beam tab's frequency
    mb = RingSimulation(win.state.lattice, cfg, RingConfig()).matched_beam()
    assert mb["frequency_mhz"] == cfg.frequency == pytest.approx(162.5)
    for k in ("beta_x", "alpha_x", "beta_y", "alpha_y", "disp_x", "disp_xp", "alpha_z",
              "beta_z"):
        assert getattr(cfg, k) == mb[k], k
    # ... and those of the smooth approximation, beta_zeta = |eta| C / (2 pi Qs)
    # (within the thin-kick correction (2 pi Qs)^2/6 = 0.4 %)
    b0 = REF["beta0"]
    g0 = 1.0 / math.sqrt(1.0 - b0 * b0)
    eta = REF["alfa"] - 1.0 / g0 ** 2
    beta_zeta = abs(eta) * REF["length_m"] / (2.0 * math.pi * mb["qs"])
    e0 = REF["pc_gev"] * 1e3 / b0                       # total energy, MeV
    deg_per_m = 360.0 * cfg.frequency * 1e6 / (b0 * 299792458.0)
    assert cfg.beta_z == pytest.approx(beta_zeta * deg_per_m / (b0 * b0 * e0), rel=1e-2)
    assert abs(cfg.alpha_z) < 0.1                        # (the cavity is near s = 0)
    # the emittances and the number of particles stay the Beam tab's
    assert (cfg.emit_nx, cfg.emit_z, cfg.n_particles) == (0.21, 0.06231832, 100_000)
    assert win.state.project_dirty


def test_a_new_lattice_drops_the_optics(win):
    win.open_lattice(FIXTURE)
    tab = _ring_on(win)
    _compute(win, tab)
    win.state.mark_project_clean()
    win.open_lattice(FODO)
    win.pump(0.2)
    assert tab._optics is None and tab._chips.text() == ""
    # (the matched beam needs no optics shown: the new lattice's own)
    assert tab._to_beam_btn.isEnabled()


def test_a_lattice_ring_mode_cannot_track_says_why(win, tmp_path):
    deck = tmp_path / "sol.madx"
    deck.write_text("beam, particle=proton, energy=1.0+pmass;\n"
                    "s: solenoid, l=1, ks=0.1;\nd: drift, l=1;\nr: line=(s, d);\nuse, sequence=r;\n")
    win.open_lattice(deck)
    tab = _ring_on(win)
    tab._optics_btn.click()
    win.wait_worker(tab._optics_worker)
    assert "solenoids are not modelled in ring mode" in tab._optics_status.text()
    assert tab._optics is None and tab._optics_btn.isEnabled()


def test_banner_offers_ring_mode_for_a_closed_ring(win):
    win.open_lattice(FIXTURE)
    win.pump(0.1)
    banner = win.lattice_tab._ring_banner
    assert banner.isVisibleTo(win.lattice_tab) and "closes on itself" in banner._text.text()
    banner._why.click()
    assert banner._details.isVisibleTo(win.lattice_tab) and "never switched on" in banner._details.text()
    banner._use.click()
    win.pump(0.1)
    assert win.state.ring_mode and not banner.isVisibleTo(win.lattice_tab)
    win._toolbar._ring_action.trigger()        # ring mode off again: offered again
    win.pump(0.1)
    assert banner.isVisibleTo(win.lattice_tab)


def test_keep_as_line_is_remembered_and_saved(win):
    import copy
    win.open_lattice(FIXTURE)
    win.pump(0.1)
    banner = win.lattice_tab._ring_banner
    banner._keep.click()
    assert not banner.isVisibleTo(win.lattice_tab)
    data = win._collect_project_dict()
    # the project holds the choice and nothing else: ring mode was not used
    assert data["ring"] == {"keep_as_line": True} and not win.state.ring_used
    # (an edit, a reload or a matching Apply keep the choice: the same file)
    win.state.lattice_changed.emit(win.state.lattice)
    win.state.set_lattice(copy.deepcopy(win.state.lattice), win.state.lattice_path)
    win.pump(0.1)
    assert not banner.isVisibleTo(win.lattice_tab)
    assert win._collect_project_dict()["ring"] == {"keep_as_line": True}
    # a project without the choice shows the banner again; with it, not
    plain = {k: v for k, v in data.items() if k != "ring"}
    win._apply_project_dict(plain, silent=True)
    win.pump(0.1)
    assert banner.isVisibleTo(win.lattice_tab)
    win._apply_project_dict(data, silent=True)
    win.pump(0.1)
    assert not banner.isVisibleTo(win.lattice_tab)
    assert not win.state.ring_mode and not win.state.ring_used


def test_keep_as_line_and_ring_mode_leave_the_next_projects_alone(win):
    win.open_lattice(FIXTURE)
    win.pump(0.1)
    win.lattice_tab._ring_banner._keep.click()
    win.state.mark_project_clean()
    win.open_lattice(FODO)                      # (a bare Open Lattice: a new session)
    win.pump(0.1)
    assert win.state.lattice_path == str(FODO)
    assert "ring" not in win._collect_project_dict()
    # ring mode used with the ring, then switched off: a later linac
    # lattice's project has no ring section either
    win.state.mark_project_clean()
    win.open_lattice(FIXTURE)
    win.pump(0.1)
    assert not win.lattice_tab._ring_banner.isVisibleTo(win.lattice_tab)   # (still kept)
    _ring_on(win)
    win._toolbar._ring_action.trigger()
    win.pump(0.1)
    assert win._collect_project_dict()["ring"]["enabled"] is False
    win.state.mark_project_clean()
    win.open_lattice(FODO)
    win.pump(0.1)
    assert win.state.lattice_path == str(FODO)
    assert "ring" not in win._collect_project_dict()


def test_a_ring_section_without_turns_keeps_the_ten_turns(win):
    win.open_lattice(FIXTURE)
    tab = _ring_on(win)
    assert tab._turns.value() == 10
    data = win._collect_project_dict()
    assert data["ring"] == {"enabled": True, "config": {"source": "beam"}}
    win._apply_project_dict(data, silent=True)
    win.pump(0.1)
    assert win.ring_tab._turns.value() == 10 and "n_turns" not in win.state.ring_config
    # (a number of turns no integer can hold is an invalid setting, not a crash)
    data["ring"]["config"] = {"n_turns": 1e999}
    warns = win._apply_project_dict(data, silent=True)
    assert any("ring settings are not valid" in w for w in warns)


def test_a_changed_setting_drops_the_optics(win):
    win.open_lattice(FIXTURE)
    tab = _ring_on(win)
    tab._edge.setCurrentIndex(1)                     # full
    _compute(win, tab)
    assert "pole faces full" in tab._optics_text.text()
    tab._model.setCurrentIndex(1)                    # curvilinear: other optics
    assert tab._optics is None
    assert tab._chips.text() == "" and "compute the optics again" in tab._optics_status.text()
    assert tab._edge.currentData() == "linear" and not tab._edge.isEnabled()
    tab._model.setCurrentIndex(0)                    # back to cartesian: its own pole faces
    assert tab._edge.currentData() == "full" and tab._edge.isEnabled()
    _compute(win, tab)
    tab._six_d.setChecked(True)
    assert tab._optics is None and "compute the optics again" in tab._optics_status.text()
    _compute(win, tab)
    tab._constants.setCurrentIndex(2)
    assert tab._optics is None


def test_a_lattice_change_stops_the_computation(win, monkeypatch):
    import threading

    import linac_gen.ring.optics as ro
    from linac_gen.core.cancelled import OperationCancelled
    started, release = threading.Event(), threading.Event()
    real = ro.twiss

    def held_twiss(rl, *, should_stop=None, **kw):
        started.set()
        release.wait(120)
        if should_stop is not None and should_stop():
            raise OperationCancelled("stopped")
        return real(rl, should_stop=should_stop, **kw)

    monkeypatch.setattr(ro, "twiss", held_twiss)
    win.open_lattice(FIXTURE)
    tab = _ring_on(win)
    tab._optics_btn.click()
    try:
        assert started.wait(300)
        worker = tab._optics_worker
        win.state.mark_project_clean()
        win.open_lattice(FODO)
        win.pump(0.1)
        assert win.state.lattice_path == str(FODO)
        assert tab._optics_worker is None and tab._optics_btn.isEnabled()
        assert tab._optics_status.text() == "Stopped: the lattice changed."
    finally:
        release.set()
    assert worker.wait(120_000)
    win.pump(0.3)
    assert tab._optics is None and tab._optics_status.text() == "Stopped: the lattice changed."


def test_optics_of_a_lattice_without_a_beam_follow_the_beam_tab(win):
    # a TraceWin deck declares no beam: its magnets are scaled with the Beam
    # tab's — optics of another species or energy go, and the matched beam is
    # that of the Beam tab's energy as it stands, which stays
    from linac_gen.ring import RingConfig, RingSimulation
    bt = win.beam_tab
    bt._species.setCurrentIndex(bt._species.findText("proton"))
    bt._energy.setValue(100.0)
    bt._apply_btn.click()
    win.state.mark_project_clean()
    win.open_lattice(FODO)
    tab = _ring_on(win)
    o = _compute(win, tab)
    assert o.gamma_t is None and "100 MeV" in tab._optics_text.text()
    bt._energy.setValue(200.0)                        # (the form, not applied yet)
    cfg = _use_matched(win, tab)
    assert cfg.energy == 200.0 and cfg.species == "proton"
    mb = RingSimulation(win.state.lattice, cfg, RingConfig()).matched_beam()
    assert mb["kinetic_mev"] == pytest.approx(200.0) and cfg.beta_x == mb["beta_x"]
    assert cfg.beta_x != pytest.approx(o.start["betx"])        # (the 100 MeV optics')
    # no RF: the beam coasts, the longitudinal settings stay
    assert mb["beta_z"] is None and cfg.beta_z == 819.05492
    assert "No RF at the start: the beam coasts" in tab._optics_status.text()
    # (applied with the matched beam: the 100 MeV optics went)
    assert tab._optics is None and tab._optics_text.text() == "No optics yet."


def test_the_banner_calls_a_declared_ring_declared(win, tmp_path):
    deck = tmp_path / "declared.dat"
    deck.write_text(";@LG ring=1\n" + FODO.read_text())
    win.open_lattice(deck)
    win.pump(0.1)
    banner = win.lattice_tab._ring_banner
    assert banner.isVisibleTo(win.lattice_tab)
    assert "declared a ring" in banner._text.text()
    assert "closes on itself" not in banner._text.text()


def test_ring_settings_go_through_a_saved_project_file(win, monkeypatch, tmp_path):
    from linac_gen_gui.interphase import app as app_mod
    win.open_lattice(FIXTURE)
    tab = _ring_on(win)
    tab._model.setCurrentIndex(1)
    tab._turns.setValue(12)
    target = tmp_path / "ring.lgproj"
    app_mod._settings().remove(app_mod._SETTINGS_LAST_PROJECT)
    monkeypatch.setattr(app_mod.QFileDialog, "getSaveFileName",
                        staticmethod(lambda *a, **k: (str(target), "")))
    win._save_project()
    saved = json.loads(target.read_text(encoding="utf-8"))
    assert saved["ring"] == {"enabled": True, "config": {
        "model": "curvilinear", "edge_model": "linear", "constants": "helix", "n_turns": 12,
        "source": "beam"}}
    win._toolbar._ring_action.trigger()               # (off, and other settings)
    tab._model.setCurrentIndex(0)
    tab._turns.setValue(3)
    win.state.mark_project_clean()
    monkeypatch.setattr(app_mod.QFileDialog, "getOpenFileName",
                        staticmethod(lambda *a, **k: (str(target), "")))
    win._open_project()
    win.pump(0.2)
    assert win.state.ring_mode and win.ring_tab._model.currentData() == "curvilinear"
    assert win.ring_tab._turns.value() == 12 and win.ring_tab._edge.currentData() == "linear"


def test_a_linac_lattice_never_loads_ring_mode(tmp_path):
    # the banner's first check needs no ring-mode code: a linac deck opened in
    # a fresh window imports none of it (fresh process, sandboxed settings)
    code = ("import os, sys\n"
            "from PyQt6.QtWidgets import QApplication\n"
            "app = QApplication([])\n"
            "from linac_gen_gui.interphase.app import InterphaseWindow\n"
            "InterphaseWindow._launch_update_check = lambda self: None\n"
            "w = InterphaseWindow()\n"
            f"ok = w._open_lattice_path({str(FODO)!r}) if hasattr(w, '_open_lattice_path') else None\n"
            "if ok is None:\n"
            "    from linac_gen_gui.interphase import app as appmod\n"
            f"    appmod.QFileDialog.getOpenFileName = staticmethod(lambda *a, **k: ({str(FODO)!r}, ''))\n"
            "    w._open_lattice()\n"
            "app.processEvents()\n"
            "assert w.state.lattice is not None\n"
            "bad = sorted(m for m in sys.modules if m.startswith('linac_gen.ring') or m == 'numba')\n"
            "print('BANNER:' + str(w.lattice_tab._ring_banner.isVisibleTo(w.lattice_tab)), flush=True)\n"
            "print('IMPORTED:' + ','.join(bad), flush=True)\n"
            "os._exit(0)\n")
    env = dict(os.environ)
    env.update({"PYTHONPATH": os.pathsep.join([str(ROOT), str(ROOT / "gui")]),
                "QT_QPA_PLATFORM": "offscreen", "HELIX_QSETTINGS_DIR": str(tmp_path),
                "HELIX_ASSIST_NO_PREWARM": "1", "OMP_NUM_THREADS": "1"})
    out = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True,
                         text=True, env=env, timeout=300)
    lines = [ln for ln in out.stdout.splitlines() if ln.startswith(("IMPORTED:", "BANNER:"))]
    assert len(lines) == 2, (out.returncode, out.stderr[-2000:])
    assert lines[0] == "BANNER:False"
    assert lines[1] == "IMPORTED:", f"opening a linac lattice imported ring-mode modules: {lines[1]}"


def test_the_matched_beam_refuses_to_replace_an_injected_beam_tab_beam(win):
    # with injection on and the Beam tab's distribution as the injected beam,
    # the ring's matched beam would replace the injected one (its species,
    # its Twiss at the foil): refused, the Beam tab left as it is
    win.open_lattice(FIXTURE)
    tab = _ring_on(win)
    page = tab._injection
    page._on.setCurrentIndex(1)
    assert (page.injection_config() or {}).get("source", {}).get("kind", "beam") == "beam"
    before = win.beam_tab.get_beam_config()
    tab._to_beam_btn.click()
    assert tab._matched_worker is None
    assert "The Injection page injects the Beam tab's beam" in tab._optics_status.text()
    assert win.beam_tab.get_beam_config() == before
    # (the Run page says that the start and the injection share that beam)
    tab._source.setCurrentIndex([tab._source.itemData(i) for i in
                                 range(tab._source.count())].index("beam"))
    assert "The Injection page injects the Beam tab's beam as well." in tab._beam_line.text()


def _held_matched(monkeypatch):
    """matched_beam held until released (a computation that takes a while)."""
    import threading

    from linac_gen.ring import engine as E
    started, release = threading.Event(), threading.Event()
    real = E.RingSimulation.matched_beam

    def held(self, *a, **kw):
        started.set()
        release.wait(120)
        return real(self, *a, **kw)

    monkeypatch.setattr(E.RingSimulation, "matched_beam", held)
    return started, release


def test_settings_changed_during_the_computation_leave_the_beam_tab(win, monkeypatch):
    # a matched beam computed for settings that changed meanwhile is not used,
    # and the page says so (with no optics shown as well)
    started, release = _held_matched(monkeypatch)
    win.open_lattice(FIXTURE)
    tab = _ring_on(win)
    before = win.beam_tab.get_beam_config()
    tab._to_beam_btn.click()
    try:
        assert started.wait(300)
        worker = tab._matched_worker
        tab._edge.setCurrentIndex(1)                      # (another element model)
        assert tab._matched_worker is None and tab._to_beam_btn.isEnabled()
        assert tab._optics_status.text().startswith(
            "The element model changed: the matched beam was not set")
    finally:
        release.set()
    assert worker.wait(120_000)
    win.pump(0.3)
    assert win.beam_tab.get_beam_config() == before
    # the Beam tab's frequency changed in the form (no Apply) while computing
    started.clear()
    release.clear()
    tab._to_beam_btn.click()
    try:
        assert started.wait(300)
        worker = tab._matched_worker
        win.beam_tab._freq.setValue(44.7044096)
    finally:
        release.set()
    assert worker.wait(120_000)
    win.pump(0.3)
    assert "The Beam tab changed during the computation" in tab._optics_status.text()
    assert win.beam_tab.get_beam_config().beta_z == before.beta_z


def test_a_particle_files_beam_is_reshaped_to_the_matched_twiss(win, tmp_path):
    # a beam read from a particle file takes the transverse Twiss only,
    # re-shaped (its emittances kept); its dispersion and longitudinal
    # distribution stay the file's — and the page says so
    import dataclasses

    import numpy as np

    from linac_gen.io.tracewin_dst import write_dst
    rng = np.random.default_rng(3)
    parts = np.column_stack([rng.normal(0, 2.0, 400), rng.normal(0, 0.2, 400),
                             rng.normal(0, 1.0, 400), rng.normal(0, 0.3, 400),
                             rng.normal(0, 5.0, 400), rng.normal(0, 0.1, 400)])
    dst = tmp_path / "booster.dst"
    write_dst(str(dst), parts, current_mA=0.0, frequency_MHz=44.7044096,
              mass_MeV=938.27208816, w_kin_ref=800.0)
    win.open_lattice(FIXTURE)
    tab = _ring_on(win)
    bt = win.beam_tab
    bt.set_beam_config(dataclasses.replace(bt.get_beam_config(), source="file",
                                           distribution_file=str(dst), disp_x=0.0))
    cfg = _use_matched(win, tab)
    assert cfg.source == "file" and cfg.reshape_to_twiss
    assert cfg.beta_x == pytest.approx(REF["start"]["betx"], rel=1e-4)
    assert cfg.disp_x == 0.0                              # (not the ring's: not used)
    assert "read from a particle file" in tab._optics_status.text()
    assert "distribution file booster.dst" in tab._beam_line.text()


# a matched beam as RingSimulation.matched_beam gives it (no RF at the start)
_CANNED = {"alpha_x": 0.1, "beta_x": 10.0, "alpha_y": -0.1, "beta_y": 5.0, "disp_x": 0.0,
           "disp_xp": 0.0, "disp_y": 0.0, "disp_yp": 0.0, "alpha_z": None, "beta_z": None,
           "centroid_x": 0.0, "centroid_xp": 0.0, "centroid_y": 0.0, "centroid_yp": 0.0,
           "centroid_dphi": 0.0, "centroid_dw": 0.0, "species": "proton", "kinetic_mev": 800.0,
           "beta0": 0.84, "p0c_mev": 1463.0, "frequency_mhz": 44.7044096, "rf_on": False,
           "rf_frequency_hz": None, "ramp": False, "qx": 0.1, "qy": 0.2, "qs": None,
           "notes": []}


def _stub_worker(release, made, result=None, fail=None):
    """A worker that sends ``result`` (or the failure ``fail``) at once, or
    when ``release`` is set if neither, and runs on until ``release`` is set;
    every one made is kept in ``made`` (so that a test fails, never aborts,
    when the tab drops one still running)."""
    from PyQt6.QtCore import QThread, pyqtSignal

    class Stub(QThread):
        finished_ok = pyqtSignal(object)
        failed = pyqtSignal(str)

        def __init__(self, *_a, **_k):
            super().__init__()
            made.append(self)

        def request_stop(self):
            pass

        def run(self):
            if fail is not None:
                self.failed.emit(fail)
            elif result is not None:
                self.finished_ok.emit(dict(result))
            release.wait(60)

    return Stub


@pytest.mark.parametrize("outcome", ["result", "failure"])
def test_a_matched_beam_worker_that_reported_stays_referenced_until_its_thread_ends(
        win, monkeypatch, outcome):
    # the worker reports just before its thread ends: dropped in its own
    # completion slot, a thread still running was destroyed (an abort)
    import threading

    from linac_gen_gui.interphase import app as app_mod
    from linac_gen_gui.interphase import workers
    release, made = threading.Event(), []
    monkeypatch.setattr(workers, "RingMatchedBeamWorker", _stub_worker(
        release, made, result=_CANNED if outcome == "result" else None,
        fail="no closed orbit" if outcome == "failure" else None))
    win.open_lattice(FIXTURE)
    tab = _ring_on(win)
    try:
        tab._to_beam_btn.click()
        worker = made[-1]
        assert win.pump_until(lambda: tab._matched_worker is None, 30_000)
        # (reported; its thread still running: kept until it ends)
        assert worker.isRunning() and worker in app_mod._PARKED_WORKERS
        if outcome == "result":
            assert tab._optics_status.text().startswith("Beam tab: the beam matched")
            assert win.beam_tab.get_beam_config().beta_x == 10.0
            # a matched beam that was set is not said to be "not set" by a
            # change of the settings after it
            tab._edge.setCurrentIndex(1)
            assert tab._optics_status.text().startswith("Beam tab: the beam matched")
        else:
            assert tab._optics_status.text() == "The matched beam: no closed orbit"
        assert tab._to_beam_btn.isEnabled()
    finally:
        release.set()
    assert worker.wait(30_000)
    assert win.pump_until(lambda: worker not in app_mod._PARKED_WORKERS, 10_000)


def test_many_presses_of_the_matched_beam_button_never_abort(tmp_path):
    # the race itself: each matched beam canned (the presses come fast), the
    # real worker and tab — in a fresh process, as an abort cannot be caught
    code = (
        "import os, sys, time\n"
        "from PyQt6.QtWidgets import QApplication\n"
        "app = QApplication([])\n"
        "from linac_gen.cli.common import load_lattice\n"
        "from linac_gen.core.config import BeamConfig\n"
        "from linac_gen_gui.interphase import app as app_mod\n"
        "from linac_gen_gui.interphase.state import AppState\n"
        "import linac_gen.ring.engine as E\n"
        f"canned = {_CANNED!r}\n"
        "E.RingSimulation.matched_beam = lambda self, *a, **k: dict(canned)\n"
        f"lat = load_lattice({str(FODO)!r})\n"
        "st = AppState()\n"
        f"st.set_lattice(lat, {str(FODO)!r})\n"
        "st.set_ring_mode(True)\n"
        "from linac_gen_gui.interphase.tabs.ring_tab import RingTab\n"
        "tab = RingTab(st)\n"
        "box = {'cfg': BeamConfig(species='proton', energy=800.0, frequency=44.7044096,\n"
        "                         current=0.0, emit_nx=1.5, emit_ny=1.5, emit_z=5.0,\n"
        "                         beta_z=100.0, cutoff=4.0)}\n"
        "tab.beam_config_provider = lambda: box['cfg']\n"
        "tab.beam_config_setter = lambda cfg: box.__setitem__('cfg', cfg)\n"
        "for i in range(400):\n"
        "    tab.use_matched_beam()\n"
        "    assert tab._matched_worker is not None\n"
        "    while tab._matched_worker is not None:\n"
        "        app.processEvents()\n"
        "assert tab._optics_status.text().startswith('Beam tab: the beam matched')\n"
        "t0 = time.time()\n"
        "while app_mod._PARKED_WORKERS and time.time() - t0 < 30:\n"
        "    app.processEvents()\n"
        "print('DONE parked:', len(app_mod._PARKED_WORKERS), flush=True)\n"
        "os._exit(0)\n")
    env = dict(os.environ)
    env.update({"PYTHONPATH": os.pathsep.join([str(ROOT), str(ROOT / "gui")]),
                "QT_QPA_PLATFORM": "offscreen", "HELIX_QSETTINGS_DIR": str(tmp_path),
                "HELIX_ASSIST_NO_PREWARM": "1", "OMP_NUM_THREADS": "1"})
    out = subprocess.run([sys.executable, "-c", code], cwd=tmp_path, capture_output=True,
                         text=True, env=env, timeout=300)
    assert "QThread: Destroyed" not in out.stderr, out.stderr[-2000:]
    assert out.returncode == 0, (out.returncode, out.stderr[-2000:])
    assert "DONE parked: 0" in out.stdout


def test_ring_mode_off_drops_the_computations_of_its_pages(win, monkeypatch):
    # the optics, the matched beam and the ramp preview being computed when
    # ring mode is switched off go; a matched beam that arrives after is not
    # taken (the linac's Beam tab and the project stay as they are)
    import threading

    from linac_gen_gui.interphase import workers
    release, made = threading.Event(), []
    monkeypatch.setattr(workers, "RingOpticsWorker", _stub_worker(release, made))
    monkeypatch.setattr(workers, "RingRampWorker", _stub_worker(release, made))
    held = threading.Event()

    class Matched(_stub_worker(held, made)):
        def run(self):
            held.wait(60)                        # (finishes just after the switch)
            self.finished_ok.emit(dict(_CANNED))
            release.wait(60)

    monkeypatch.setattr(workers, "RingMatchedBeamWorker", Matched)
    win.open_lattice(FIXTURE)
    tab = _ring_on(win)
    page = tab._programs
    page._e_kind.setCurrentIndex(1)                 # flat: a ramp to preview
    before = win.beam_tab.get_beam_config()
    try:
        tab._optics_btn.click()
        page._preview_btn.click()
        tab._to_beam_btn.click()
        assert len(made) == 3 and all(w.isRunning() for w in made)
        win._toolbar._ring_action.trigger()         # ring mode off
        win.pump(0.05)
        assert not win.state.ring_mode
        win.state.mark_project_clean()
        assert tab._optics_worker is None and tab._matched_worker is None
        assert page._worker is None
        assert tab._optics_status.text().startswith(
            "Ring mode was switched off: the matched beam was not set")
        assert page._status.text() == "Ring mode was switched off: press Preview again."
        assert tab._optics_btn.isEnabled() and page._preview_btn.isEnabled()
        held.set()
        win.pump(0.3)
    finally:
        held.set()
        release.set()
    assert all(w.wait(30_000) for w in made)
    win.pump(0.3)
    assert win.beam_tab.get_beam_config() == before and not win.state.project_dirty


def test_the_tabs_declared_beam_is_the_compilers():
    # (the tab takes the rule from linac_gen.ring.config, so as not to import
    # the ring compiler and numba on the GUI thread; the compiler re-exports it)
    from types import SimpleNamespace

    from linac_gen_gui.interphase.tabs.ring_tab import _declared_beam

    from linac_gen.ring import compile as ring_compile
    from linac_gen.ring import config as ring_config
    assert ring_compile.declared_beam is ring_config.declared_beam
    declared_beam = ring_config.declared_beam
    rb = {"species": "proton", "energy": 800.0, "frequency": 44.7}
    for lat in (SimpleNamespace(), SimpleNamespace(ring_reference_beam=None),
                SimpleNamespace(ring_reference_beam={}),
                SimpleNamespace(ring_reference_beam=dict(rb)),
                SimpleNamespace(ring_reference_beam=dict(rb, declared=True)),
                SimpleNamespace(ring_reference_beam=dict(rb, declared=False))):
        assert _declared_beam(lat) is declared_beam(lat)


def test_a_deck_that_declares_no_beam_runs_at_the_beam_tabs_species_and_energy(win):
    # its reference beam only scaled the magnets (an elegant deck, a MAD-X
    # BEAM without an energy): like a lattice without a beam, the ring runs at
    # the Beam tab's species and energy, which the matched beam leaves alone
    win.open_lattice(FIXTURE)
    rb = win.state.lattice.ring_reference_beam
    assert rb.get("declared", True)
    rb["declared"] = False
    bt = win.beam_tab
    bt._species.setCurrentIndex(bt._species.findText("proton"))
    bt._energy.setValue(400.0)
    tab = _ring_on(win)
    ref = tab._verdict.text().split("Reference:")[1]
    assert ref.startswith(" the Beam tab's species and energy — the deck declares no beam")
    assert "800" not in ref
    tab._optics_btn.click()
    assert tab._optics_beam == ("proton", 400.0)       # (the Beam tab's: followed)
    win.wait_worker(tab._optics_worker)
    assert "400 MeV" in tab._optics_text.text()
    cfg = _use_matched(win, tab)
    assert (cfg.species, cfg.energy) == ("proton", 400.0)
    assert "(proton 400 MeV, fixed energy" in tab._optics_status.text()
    # a preview of the ramp is the Beam tab's energy's: a Beam tab change drops it
    tab._programs._preview = {"shown": True}
    win.state.beam_config_changed.emit(bt.get_beam_config())
    assert tab._programs._preview is None
    # the same deck declaring its beam: its species and energy come with the
    # matched beam, and the preview stays through a Beam tab change
    rb["declared"] = True
    cfg = _use_matched(win, tab)
    assert cfg.species == "proton" and cfg.energy == pytest.approx(800.0, rel=1e-12)
    tab._programs._preview = {"shown": True}
    win.state.beam_config_changed.emit(bt.get_beam_config())
    assert tab._programs._preview == {"shown": True}
