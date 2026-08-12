"""App-level multibunch / pulse-study GUI tests (M8).

House rule: bugs live in the seams, so everything here drives the REAL
entry path — the toolbar signal / ``_open_train_study`` slot, a real
``TrainWorker`` QThread, queued signals into ``_train_done`` /
``_train_aborted`` — with only the human interactions monkeypatched.
The abort test is the load-bearing one: a mid-train Stop through the
real toolbar stop path must leave a LOADABLE partial train file
(``load_train_results`` succeeds, ``truncated`` is True).
"""
from __future__ import annotations

import json

import pytest

pytest.importorskip("PyQt6")

FREQ = 162.5


@pytest.fixture(autouse=True)
def _settings_hygiene():
    """Leave the process-wide (sandboxed) QSettings exactly as found.

    These tests run real windows that PERSIST session state
    (``state.set_beam_config`` → sessionBeamConfig, lattice paths); a
    later window's ``_restore_last_session`` would then replay it and
    emit "Restored last session's beam …" — which
    test_update_gui::test_fetch_failure_is_totally_silent rightly
    asserts never happens.  Found as a full-suite ordering failure
    (2026-08-11): this file sorts between the session-persistence tests
    (which clean up after themselves) and the update tests."""
    from linac_gen_gui.interphase.app import (
        _SETTINGS_LAST_DIR, _SETTINGS_LAST_LATTICE, _SETTINGS_LAST_PROJECT,
        _SETTINGS_SESSION_BEAM, _settings,
    )
    keys = (_SETTINGS_SESSION_BEAM, _SETTINGS_LAST_LATTICE,
            _SETTINGS_LAST_PROJECT, _SETTINGS_LAST_DIR)
    s = _settings()
    before = {k: s.value(k) for k in keys}
    yield
    for k, v in before.items():
        if v is None:
            s.remove(k)
        else:
            s.setValue(k, v)
    s.sync()


@pytest.fixture()
def win(qapp):
    from linac_gen_gui.interphase.app import InterphaseWindow
    w = InterphaseWindow()
    yield w
    w.close()
    w.deleteLater()


@pytest.fixture()
def calc_dir(tmp_path):
    """Point the auto-dump calc dir at a per-test tmp (sandboxed
    QSettings — conftest sets HELIX_QSETTINGS_DIR)."""
    from linac_gen_gui.interphase.app import _SETTINGS_CALC_DIR, _settings
    s = _settings()
    old = s.value(_SETTINGS_CALC_DIR, "")
    s.setValue(_SETTINGS_CALC_DIR, str(tmp_path))
    yield tmp_path
    s.setValue(_SETTINGS_CALC_DIR, old)


def _beam_config(n_particles=300):
    from linac_gen.core.config import BeamConfig
    return BeamConfig(species="proton", energy=3.0, frequency=FREQ,
                      current=5.0, n_particles=n_particles)


def _sidecar(tmp_path, name_pattern="G1"):
    p = tmp_path / "cav.json"
    p.write_text(json.dumps(
        {name_pattern: {"r_over_q": 200.0, "q_loaded": 5.0e6}}))
    return str(p)


def _wait_worker(qapp, worker, timeout_ms=120_000):
    assert worker is not None, "worker was never constructed"
    assert worker.wait(timeout_ms), "train worker did not finish"
    qapp.processEvents()               # deliver queued signals


# ---------------------------------------------------------------------------
# dialog validation refusals (real widgets, in-dialog error label)
# ---------------------------------------------------------------------------
def test_dialog_opt_in_and_validation_refusals(qapp, tmp_path):
    from linac_gen_gui.interphase.dialogs.train_dialog import (
        TrainConfigDialog,
    )
    dlg = TrainConfigDialog(default_freq_MHz=FREQ)
    try:
        # OFF by default: form inert, OK disabled — the opt-in contract.
        assert not dlg._enable.isChecked()
        assert not dlg._form_box.isEnabled()
        assert not dlg._ok_button().isEnabled()

        dlg._enable.setChecked(True)
        assert dlg._form_box.isEnabled()
        assert dlg._ok_button().isEnabled()

        # Live pattern preview: valid RLE → slot/duty facts; bad → error.
        dlg._pattern.setText("1*6 0*2 1*2")
        assert "10 slots" in dlg._pattern_preview.text()
        assert "8 bunches" in dlg._pattern_preview.text()
        dlg._pattern.setText("2*5")
        assert "RLE" in dlg._pattern_preview.text()

        # beam_loading without a sidecar → loud in-dialog refusal that
        # names exactly what is missing; the dialog must NOT accept.
        # (isVisibleTo: plain isVisible() is always False for children
        # of a dialog that was never shown — the offscreen trap.)
        dlg._pattern.setText("1*4")
        dlg._loading.setChecked(True)
        assert dlg._sidecar_row.isEnabled()      # picker revealed
        dlg._validate_and_accept()
        assert dlg._error.isVisibleTo(dlg)
        assert "cavity_params" in dlg._error.text()
        assert dlg.result() != dlg.DialogCode.Accepted
        assert dlg.train_config is None

        # envelope mode cannot carry the loading channel.
        dlg._sidecar.setText(_sidecar(tmp_path))
        dlg._mode.setCurrentText("envelope")
        dlg._validate_and_accept()
        assert dlg._error.isVisibleTo(dlg)
        assert "envelope" in dlg._error.text()
        assert dlg.result() != dlg.DialogCode.Accepted

        # Fix everything → accepted, validated TrainConfig built.
        dlg._mode.setCurrentText("mp")
        dlg._validate_and_accept()
        assert not dlg._error.isVisibleTo(dlg)
        assert dlg.result() == dlg.DialogCode.Accepted
        tc = dlg.train_config
        assert tc is not None and tc.mode == "mp"
        assert tc.pattern.n_bunches == 4
        assert tc.physics.beam_loading
    finally:
        dlg.deleteLater()


def test_dialog_select_bunches_hybrid_gate(qapp):
    from linac_gen_gui.interphase.dialogs.train_dialog import (
        TrainConfigDialog,
    )
    dlg = TrainConfigDialog(default_freq_MHz=FREQ)
    try:
        dlg._enable.setChecked(True)
        assert not dlg._select.isEnabled()          # mp: hybrid-only knob
        dlg._mode.setCurrentText("hybrid")
        assert dlg._select.isEnabled()
        dlg._mode.setCurrentText("envelope")
        assert not dlg._select.isEnabled()
        assert not dlg._sc_check.isEnabled()        # inert for envelope
    finally:
        dlg.deleteLater()


# ---------------------------------------------------------------------------
# toolbar action routing (the real signal path)
# ---------------------------------------------------------------------------
def test_toolbar_action_opens_dialog(qapp, win, mini_lattice, monkeypatch):
    from linac_gen_gui.interphase.dialogs import train_dialog as td
    win.state.set_lattice(mini_lattice, "/tmp/train_route.dat")
    cfg = _beam_config()
    win.beam_tab.set_beam_config(cfg)
    win.state.set_beam_config(cfg)
    seen = []

    def fake_exec(self):
        seen.append(self)
        return False                                # user cancels

    monkeypatch.setattr(td.TrainConfigDialog, "exec", fake_exec)
    win._toolbar.open_train_requested.emit()
    assert seen, "toolbar action did not open the TrainConfigDialog"
    assert win._train_worker is None
    assert not win.state.running


def test_slot_guards_without_lattice(qapp, win, monkeypatch):
    from PyQt6.QtWidgets import QMessageBox
    boxes = []
    monkeypatch.setattr(QMessageBox, "warning",
                        staticmethod(lambda *a, **k: boxes.append(a)))
    win._open_train_study()
    assert boxes, "expected the 'No lattice' warning"
    assert win._train_worker is None


# ---------------------------------------------------------------------------
# full run through the real slot: completion → auto-dump run_type "train"
# ---------------------------------------------------------------------------
def test_full_run_auto_dumps_train_file(qapp, win, mini_lattice,
                                        monkeypatch, tmp_path, calc_dir):
    from linac_gen.train import PulsePattern, TrainConfig, TrainPhysics
    from linac_gen.train.results import load_train_results
    from linac_gen_gui.interphase.dialogs import train_dialog as td

    win.state.set_lattice(mini_lattice, "/tmp/train_e2e.dat")
    cfg = _beam_config(n_particles=300)
    win.beam_tab.set_beam_config(cfg)
    win.state.set_beam_config(cfg)
    side = _sidecar(tmp_path)
    tc = TrainConfig(
        bunch_frequency_MHz=FREQ,
        pattern=PulsePattern.from_rle("1*3 0*2 1*1"), mode="fast",
        physics=TrainPhysics(beam_loading=True), cavity_params=side)

    def fake_exec(self):
        self.train_config = tc
        self.space_charge_enabled = False           # explicit off
        return True

    monkeypatch.setattr(td.TrainConfigDialog, "exec", fake_exec)
    win._open_train_study()
    _wait_worker(qapp, win._train_worker)

    assert not win.state.running
    dumps = sorted(calc_dir.glob("*_train.h5"))
    assert dumps, "completed train run did not auto-dump a *_train.h5"
    ld = load_train_results(str(dumps[-1]))
    assert ld.mode == "fast"
    assert not ld.truncated
    assert ld.fast is not None and len(ld.fast.w_exit_MeV) == 4
    # v1 results surface: the modeless summary popup came up populated.
    dlg = win._train_summary_dlg
    assert dlg is not None and dlg._series, "summary popup not populated"
    dlg.close()


# ---------------------------------------------------------------------------
# THE abort test: real toolbar Stop mid-train → loadable partial result
# ---------------------------------------------------------------------------
def test_mid_train_abort_leaves_loadable_partial(qapp, win, mini_lattice,
                                                 monkeypatch, tmp_path,
                                                 calc_dir):
    from linac_gen.train import PulsePattern, TrainConfig, TrainPhysics
    from linac_gen.train.results import load_train_results
    from linac_gen_gui.interphase.dialogs import train_dialog as td

    win.state.set_lattice(mini_lattice, "/tmp/train_abort.dat")
    # Heavy-ish bunches (≥ tens of ms each): the run must comfortably
    # outlive the stop round-trip, or the test races its own event loop.
    cfg = _beam_config(n_particles=3000)
    win.beam_tab.set_beam_config(cfg)
    win.state.set_beam_config(cfg)
    side = _sidecar(tmp_path)
    n_bunches = 150                    # far more than can finish pre-stop
    tc = TrainConfig(
        bunch_frequency_MHz=FREQ,
        pattern=PulsePattern.uniform(n_bunches), mode="mp",
        physics=TrainPhysics(beam_loading=True), cavity_params=side)

    def fake_exec(self):
        self.train_config = tc
        self.space_charge_enabled = False
        return True

    monkeypatch.setattr(td.TrainConfigDialog, "exec", fake_exec)
    # Drain the window-startup backlog (restore timers, deferred
    # deletions from earlier tests) BEFORE the run starts — otherwise
    # the first processEvents() below can swallow the whole train.
    for _ in range(50):
        qapp.processEvents()
    win._open_train_study()
    worker = win._train_worker
    assert worker is not None and worker.isRunning()
    assert win.state.running

    # Note the first completed bunch (signal fires on the worker thread;
    # list append is atomic under the GIL), then stop through the REAL
    # user path — toolbar Stop → _stop_active_worker → request_stop —
    # from the GUI thread, pumping the event loop like a live session.
    progressed = []

    def note(done, total):
        progressed.append(done)

    worker.progress_bunch.connect(note)
    stopped = []
    import time
    t0 = time.time()
    while worker.isRunning() and time.time() - t0 < 300:
        qapp.processEvents()
        if progressed and not stopped:
            stopped.append(progressed[0])
            win._toolbar.stop_requested.emit()
            # The real path is synchronous on the GUI thread — the
            # cooperative stop flag must be armed the moment the
            # toolbar signal returns.
            assert worker._stop_event.is_set(), \
                "toolbar stop did not reach TrainWorker.request_stop"
        worker.wait(10)
    _wait_worker(qapp, worker, timeout_ms=300_000)

    assert stopped, "no per-bunch progress was ever delivered"
    assert not win.state.running
    dumps = sorted(calc_dir.glob("*_train.h5"))
    assert dumps, "aborted train run did not auto-dump a partial file"
    ld = load_train_results(str(dumps[-1]))
    assert ld.truncated, "partial train file is not marked truncated"
    assert 1 <= ld.n_bunches_tracked < n_bunches
    assert len(ld.slots) == ld.n_bunches_tracked
    # Per-bunch summary of the processed prefix survived intact.
    assert len(ld.summary["ref_w_kin"]) == ld.n_bunches_tracked
    # Beam-loading state of the processed bunches persisted too.
    assert ld.applied_loading, "no cavity_state ledger in partial file"
    dlg = win._train_summary_dlg
    assert dlg is not None, "partial result did not open the summary popup"
    assert "ABORTED" in dlg._header.text()
    dlg.close()


# ---------------------------------------------------------------------------
# Live progress: the summary popup opens AT LAUNCH and shows per-bunch
# progress, then becomes the plot on completion (user feedback 2026-08-11:
# "OK closed the box and nothing showed progress").
# ---------------------------------------------------------------------------
def test_summary_popup_live_progress(qapp, win, mini_lattice, monkeypatch,
                                     tmp_path, calc_dir):
    from linac_gen.train import PulsePattern, TrainConfig, TrainPhysics
    from linac_gen_gui.interphase.dialogs import train_dialog as td

    win.state.set_lattice(mini_lattice, "/tmp/train_live.dat")
    cfg = _beam_config(n_particles=300)
    win.beam_tab.set_beam_config(cfg)
    win.state.set_beam_config(cfg)
    tc = TrainConfig(
        bunch_frequency_MHz=FREQ,
        pattern=PulsePattern.from_rle("1*3 0*2 1*1"), mode="fast",
        physics=TrainPhysics(beam_loading=True),
        cavity_params=_sidecar(tmp_path))

    def fake_exec(self):
        self.train_config = tc
        self.space_charge_enabled = False
        return True

    monkeypatch.setattr(td.TrainConfigDialog, "exec", fake_exec)
    win._open_train_study()

    # Synchronously after launch (no queued signals delivered yet): the
    # popup exists, is in live-progress mode, and announces the design
    # pass.  This is deterministic even for a millisecond fast run —
    # worker completion is a queued signal not yet processed.
    dlg = win._train_summary_dlg
    assert dlg is not None, "summary popup must open at study launch"
    assert dlg._prog.isVisibleTo(dlg)
    assert dlg._prog_label.isVisibleTo(dlg)
    assert "RUNNING" in dlg._header.text()
    assert "Design pass" in dlg._prog_label.text()
    assert dlg._prog.maximum() == tc.pattern.n_bunches

    # Widget-level progress transition (no race with the worker).
    dlg.on_progress(2, 4)
    assert dlg._prog.value() == 2
    assert "2/4" in dlg._prog_label.text()

    _wait_worker(qapp, win._train_worker)

    # Completion: progress row hidden, plot populated, run not claimed
    # to be running any more.
    assert not dlg._prog.isVisibleTo(dlg)
    assert not dlg._prog_label.isVisibleTo(dlg)
    assert dlg._series, "plot did not populate after completion"
    assert "RUNNING" not in dlg._header.text()
    # Stale progress after results must be ignored (guarded, no revive).
    dlg.on_progress(3, 4)
    assert not dlg._prog.isVisibleTo(dlg)
    dlg.close()


# ---------------------------------------------------------------------------
# Pulse visualization (user request): live pattern strip in the config
# dialog; fill-pattern series + µs time axis in the summary window.
# ---------------------------------------------------------------------------
def test_config_dialog_pulse_strip(qapp):
    from linac_gen_gui.interphase.dialogs.train_dialog import (
        TrainConfigDialog,
    )
    dlg = TrainConfigDialog(default_freq_MHz=FREQ)
    dlg._enable.setChecked(True)
    dlg._pattern.setText("1*3 0*2 1*1")
    items = dlg._pattern_plot.listDataItems()
    assert items, "pulse strip did not draw for a valid RLE"
    x, y = items[0].getData()
    assert len(x) == 6 and len(y) == 6
    assert y.tolist() == [1.0, 1.0, 1.0, 0.0, 0.0, 1.0]
    # x is time in µs: last slot at (n-1)/f
    assert abs(x[-1] - 5.0 / FREQ) < 1e-12
    # invalid RLE clears the strip instead of keeping a stale pulse
    dlg._pattern.setText("garbage")
    assert not dlg._pattern_plot.listDataItems()
    dlg.close()


def test_summary_fill_pattern_and_us_axis(qapp, win, mini_lattice,
                                          monkeypatch, tmp_path, calc_dir):
    from linac_gen.train import PulsePattern, TrainConfig, TrainPhysics
    from linac_gen_gui.interphase.dialogs import train_dialog as td

    win.state.set_lattice(mini_lattice, "/tmp/train_pulseplot.dat")
    cfg = _beam_config(n_particles=300)
    win.beam_tab.set_beam_config(cfg)
    win.state.set_beam_config(cfg)
    tc = TrainConfig(
        bunch_frequency_MHz=FREQ,
        pattern=PulsePattern.from_rle("1*3 0*2 1*1"), mode="fast",
        physics=TrainPhysics(beam_loading=True),
        cavity_params=_sidecar(tmp_path))

    def fake_exec(self):
        self.train_config = tc
        self.space_charge_enabled = False
        return True

    monkeypatch.setattr(td.TrainConfigDialog, "exec", fake_exec)
    win._open_train_study()
    _wait_worker(qapp, win._train_worker)
    dlg = win._train_summary_dlg
    assert dlg is not None
    key = "fill pattern (1 = bunch present)"
    assert key in dlg._series, "pulse structure series missing"
    idx = dlg._quantity.findText(key)
    assert idx >= 0
    dlg._quantity.setCurrentIndex(idx)
    x, y = dlg._plot.listDataItems()[0].getData()
    assert len(x) == 6 and y.tolist() == [1.0, 1.0, 1.0, 0.0, 0.0, 1.0]
    assert x[-1] == 5.0                        # slot-index axis
    # µs toggle rescales the x axis by 1/f
    dlg._us_axis.setChecked(True)
    x2, y2 = dlg._plot.listDataItems()[0].getData()
    assert abs(x2[-1] - 5.0 / FREQ) < 1e-12
    assert y2.tolist() == y.tolist()
    dlg._us_axis.setChecked(False)
    dlg.close()


# ---------------------------------------------------------------------------
# Results-tab import of a saved *_train.h5 opens the summary window
# (was: an info box pointing at Python — a finished study's file was
# unreachable from the GUI once the summary window moved on).
# ---------------------------------------------------------------------------
def test_results_tab_opens_saved_train_file(qapp, win, mini_lattice,
                                            monkeypatch, tmp_path,
                                            calc_dir):
    from linac_gen.train import PulsePattern, TrainConfig, TrainPhysics
    from linac_gen_gui.interphase.dialogs import train_dialog as td
    from linac_gen_gui.interphase.tabs import results_tab as rt

    win.state.set_lattice(mini_lattice, "/tmp/train_reopen.dat")
    cfg = _beam_config(n_particles=300)
    win.beam_tab.set_beam_config(cfg)
    win.state.set_beam_config(cfg)
    tc = TrainConfig(
        bunch_frequency_MHz=FREQ,
        pattern=PulsePattern.from_rle("1*3 0*2 1*1"), mode="fast",
        physics=TrainPhysics(beam_loading=True),
        cavity_params=_sidecar(tmp_path))

    def fake_exec(self):
        self.train_config = tc
        self.space_charge_enabled = False
        return True

    monkeypatch.setattr(td.TrainConfigDialog, "exec", fake_exec)
    win._open_train_study()
    _wait_worker(qapp, win._train_worker)
    dumps = sorted(calc_dir.glob("*_train.h5"))
    assert dumps
    saved = str(dumps[-1])

    # Re-open the saved file through the real Results-tab import path.
    monkeypatch.setattr(rt.QFileDialog, "getOpenFileName",
                        staticmethod(lambda *a, **k: (saved, "")))
    tab = win.results_tab
    tab._import_results()
    popups = getattr(tab, "_train_summary_popups", [])
    assert popups, "train file import did not open a summary window"
    dlg = popups[-1]
    assert dlg._series, "reopened summary window is empty"
    key = "fill pattern (1 = bunch present)"
    assert key in dlg._series
    assert dumps[-1].name in dlg.windowTitle()
    # loaded (not live) results: no progress row claiming to run
    assert not dlg._prog.isVisibleTo(dlg)
    dlg.close()
