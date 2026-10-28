"""The assistant's ring tools in the desktop GUI: the panel's adapter maps the
session's ring settings, ring mode and ring results to AppState and the Ring
tab (a ramp set by the assistant is the Programs page's; run_in_gui's "ring"
kind presses the Ring tab's Run button) — the real window, the adapter's own
GUI-thread hop."""
from __future__ import annotations

import types
from pathlib import Path

import pytest

pytest.importorskip("PyQt6")
pytest.importorskip("numba")

ROOT = Path(__file__).resolve().parents[2]
FIXTURE = ROOT / "tests" / "ring" / "fixtures" / "simple_booster.madx"
RAMP = {"energy": {"kind": "sine", "start_kinetic_mev": 800.0, "end_kinetic_mev": 801.0,
                   "frequency_hz": 20000.0},
        "voltage": {"kind": "constant"}, "phase": {"kind": "auto"}, "gamma_t": 5.449167323}


class _Nav:
    """The panel's side the adapter uses: the window, and run_on_gui (here
    the test's own thread is the GUI thread)."""

    def __init__(self, app):
        self._app = app

    def run_on_gui(self, fn, timeout=30.0):
        return fn()


def _context(win, tmp_path):
    from linac_gen_gui.interphase.dialogs import assistant_panel as ap
    return ap._make_context(win.state, str(tmp_path), nav=_Nav(win))


def _call(name, ctx, **kw):
    from linac_gen.assist.tools import TOOLS
    return TOOLS[name].fn(ctx, **kw)


def test_the_assistant_drives_the_ring_tab(win, tmp_path):
    from linac_gen.ring.programs import RampConfig
    win.open_lattice(FIXTURE)
    ctx = _context(win, tmp_path)
    assert ctx.ring_mode is False and ctx.ring_config == {}
    r = _call("set_ring_mode", ctx, on=True)
    assert r["status"] == "ok" and r["data"]["ring_mode"] is True
    win.pump(0.1)
    tab = win.ring_tab
    assert tab is not None and win.state.ring_mode and ctx.ring_mode
    assert win._tabs.isTabVisible(win._tabs.indexOf(tab)) and not tab.is_running()
    # a ramp set by the assistant: in the state, on the Programs page, in the
    # Run page's summary, the project dirty
    win.state.mark_project_clean()
    r = _call("set_ring_program", ctx, ramp=RAMP)
    assert r["status"] == "ok", r
    assert win.state.project_dirty
    want = RampConfig.from_dict(RAMP).to_dict()
    assert win.state.ring_config["ramp"] == want == ctx.ring_config["ramp"]
    assert tab._programs.ramp_config() == want
    assert "sine" in tab._ramp_line.text()
    assert _call("ring_status", ctx)["data"]["settings"]["ramp"] == "set"
    # the Ring tab's own Run button, pressed by run_in_gui
    win.beam_tab._npart.setValue(100)
    tab._turns.setValue(3)
    r = _call("run_in_gui", ctx, kind="ring")
    assert r["status"] == "ok" and r["data"]["state"] == "started", r
    assert tab.is_running()
    busy = _call("run_in_gui", ctx, kind="ring")
    assert busy["status"] == "refused" and "ring run" in busy["data"]["message"]
    win.wait_worker(tab._worker)
    assert not tab.is_running() and win.state.ring_results is not None
    assert win.state.ring_results.turns_done == 3
    # a headless ring run's results go to the Ring tab too
    r = _call("run_ring", ctx, turns=2, particles=50)
    assert r["status"] == "ok", r
    win.pump(0.2)
    assert win.state.ring_results is ctx.ring_results
    assert win.state.ring_results.turns_done == 2
    assert ctx.ring_results_path == r["data"]["file"]
    assert win.state.ring_results.meta["dump_path"] == r["data"]["file"]
    # clearing the ramp clears the page
    assert _call("set_ring_program", ctx, clear=True)["status"] == "ok"
    assert "ramp" not in win.state.ring_config and tab._programs.ramp_config() is None
    # ring mode off: the tab hidden
    assert _call("set_ring_mode", ctx, on=False)["data"]["ring_mode"] is False
    win.pump(0.1)
    assert not win._tabs.isTabVisible(win._tabs.indexOf(tab))
    off = _call("run_in_gui", ctx, kind="ring")
    assert off["status"] == "error" and "ring mode is off" in off["data"]["message"]


def test_a_mode_that_does_not_switch_on_is_reported(win, tmp_path, monkeypatch):
    ctx = _context(win, tmp_path)
    monkeypatch.setattr(win.state, "set_ring_mode", lambda on: None)
    r = _call("set_ring_mode", ctx, on=True)
    assert r["status"] == "error" and "could not be switched on" in r["data"]["message"]


def test_a_gui_that_does_not_answer_is_reported(win, tmp_path):
    from linac_gen_gui.interphase.dialogs import assistant_panel as ap

    class _Deaf(_Nav):
        def run_on_gui(self, fn, timeout=30.0):
            return None

    ctx = ap._make_context(win.state, str(tmp_path), nav=_Deaf(win))
    r = _call("set_ring_program", ctx, ramp=RAMP)
    assert r["status"] == "error" and "GUI" in r["data"]["message"]
    assert "ramp" not in (win.state.ring_config or {})
    assert _call("set_ring_mode", ctx, on=True)["status"] == "error"


def test_the_apps_ring_run_guards(qapp):
    from linac_gen_gui.interphase.app import InterphaseWindow

    class _Tab:
        def __init__(self):
            self.running = False
            self.starts = True

        def is_running(self):
            return self.running

        def run(self):
            if self.starts:
                self.running = True

    fake = types.SimpleNamespace(state=types.SimpleNamespace(ring_mode=False, lattice=1,
                                                             running=True),
                                 ring_tab=None)
    fake._assistant_run_ring = types.MethodType(InterphaseWindow._assistant_run_ring, fake)
    run = InterphaseWindow.assistant_run_simulation
    assert "ring mode is off" in run(fake, "ring")
    fake.state.ring_mode = True
    assert "ring mode is off" in run(fake, "ring")          # (no Ring tab yet)
    fake.ring_tab = _Tab()
    # (a linac run in progress does not hold a ring run back)
    assert run(fake, "ring") == "started"
    assert run(fake, "ring") == "busy"
    fake.ring_tab = _Tab()
    fake.ring_tab.starts = False
    assert "declined" in run(fake, "ring")
    fake.state.lattice = None
    assert run(fake, "ring") == "no lattice is loaded in the GUI"
    fake.state.lattice = 1

    def boom():
        raise RuntimeError("no")

    fake.ring_tab.run = boom
    assert run(fake, "ring") == "RuntimeError: no"


def test_the_ring_tabs_pages_are_the_sessions_settings(win, tmp_path):
    """(what the Ring tab's Run would use — its defaults too — and a ramp set
    by the assistant is saved with the project, ring mode on or not)"""
    win.open_lattice(FIXTURE)
    ctx = _context(win, tmp_path)
    assert ctx.ring_config == {}                    # (no Ring tab yet: the state's)
    assert _call("set_ring_program", ctx, ramp=RAMP)["status"] == "ok"
    assert win.state.ring_used and "ramp" in win._collect_project_dict()["ring"]["config"]
    _call("set_ring_mode", ctx, on=True)
    win.pump(0.1)
    tab = win.ring_tab
    cfg = ctx.ring_config
    # (the particles are the Beam tab's: the run settings hold the source)
    assert cfg["n_turns"] == tab._turns.value() and "n_particles" not in cfg
    assert cfg["source"] == tab._source.currentData() == "beam"
    assert cfg["snapshot_turns"] == [0, "last"] and cfg["model"] == tab._model.currentData()
    assert cfg["ramp"] == tab._programs.ramp_config() and cfg["corrector_mode"] == "follow"
    win.beam_tab._npart.setValue(123)
    win.beam_tab._apply_btn.click()
    assert _call("ring_status", ctx)["data"]["settings"]["particles"] == 123


def test_the_gui_runs_file_and_state_are_what_ring_status_reports(win, tmp_path):
    win.open_lattice(FIXTURE)
    ctx = _context(win, tmp_path)
    _call("set_ring_mode", ctx, on=True)
    win.pump(0.1)
    tab = win.ring_tab
    win.beam_tab._npart.setValue(100)
    tab._turns.setValue(3)
    assert _call("run_in_gui", ctx, kind="ring")["data"]["state"] == "started"
    st = _call("ring_status", ctx)["data"]
    assert st["run_in_progress"] is True
    r = _call("run_ring", ctx, turns=1, particles=50)
    assert r["status"] == "refused" and "Ring tab" in r["data"]["message"]
    win.wait_worker(tab._worker)
    st = _call("ring_status", ctx)["data"]
    assert st["run_in_progress"] is False
    assert st["last_run"]["file"] == win.state.ring_results.meta["dump_path"]
    assert Path(st["last_run"]["file"]).is_file()


def test_a_gui_hop_that_timed_out_never_lands_late(win, tmp_path):
    from linac_gen_gui.interphase.dialogs import assistant_panel as ap

    class _Late(_Nav):
        """run_on_gui times out (None) but the queued call still runs later."""
        def __init__(self, app):
            super().__init__(app)
            self.queued = []

        def run_on_gui(self, fn, timeout=30.0):
            self.queued.append(fn)
            return None

    nav = _Late(win)
    ctx = ap._make_context(win.state, str(tmp_path), nav=nav)
    assert _call("set_ring_program", ctx, ramp=RAMP)["status"] == "error"
    assert _call("set_ring_mode", ctx, on=True)["status"] == "error"
    for fn in nav.queued:                           # (the GUI gets to them at last)
        fn()
    assert "ramp" not in (win.state.ring_config or {}) and not win.state.ring_mode


def test_the_ring_tools_read_the_beam_tabs_form_as_the_ring_tab(win, tmp_path):
    # (one beam: an edit of the Beam tab not yet applied counts for the Ring
    # tab's Run and for a project save — and so for the assistant's ring tools)
    win.open_lattice(FIXTURE)
    ctx = _context(win, tmp_path)
    assert _call("set_ring_mode", ctx, on=True)["status"] == "ok"
    win.pump(0.1)
    applied = win.state.beam_config.n_particles
    win.beam_tab._npart.setValue(applied + 23)          # (typed, not applied)
    assert win.state.beam_config.n_particles == applied
    assert ctx.ring_beam_config().n_particles == applied + 23
    st = _call("ring_status", ctx)["data"]["settings"]
    assert (st["source"], st["particles"]) == ("beam", applied + 23)


def test_the_assistants_copy_of_the_lattice_waits_for_the_walks(win, tmp_path):
    # (as the Ring tab's workers copy it: never while a walk or a run changes
    # the elements; a stop while waiting is honoured)
    import threading

    from linac_gen_gui.interphase.workers import WALK_LOCK
    win.open_lattice(FIXTURE)
    ctx = _context(win, tmp_path)
    assert WALK_LOCK.acquire(timeout=5)
    got = {}
    stop = threading.Event()

    def _copy():
        try:
            got["lattice"] = ctx.lattice_snapshot(should_stop=stop.is_set)
        except RuntimeError as exc:
            got["error"] = str(exc)

    try:
        t = threading.Thread(target=_copy)
        t.start()
        t.join(0.6)
        assert t.is_alive() and not got                  # (waiting for the lock)
        stop.set()
        t.join(5)
        assert not t.is_alive() and "stopped" in got.get("error", "")
    finally:
        WALK_LOCK.release()
    lat = ctx.lattice_snapshot()
    assert lat is not win.state.lattice and len(lat.elements) == len(win.state.lattice.elements)
