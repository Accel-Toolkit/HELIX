"""Offscreen test of the GUI assistant panel driven by a MockProvider
(no network, no keys) — proves the toggle-free surface: confirmation
strip echoes the resolved call, approval runs the tool, state syncs to
AppState, and closing mid-turn does not hang."""
from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("PyQt6")


def _state_with_lattice(qapp):
    from linac_gen.core.config import BeamConfig
    from linac_gen.core.lattice import Lattice
    from linac_gen.elements.drift import Drift
    from linac_gen.elements.quadrupole import Quadrupole
    from linac_gen_gui.interphase.state import AppState
    lat = Lattice()
    lat.add(Quadrupole("QF", 50.0, gradient=5.0, aperture=20.0))
    lat.add(Drift("D1", 200.0))
    lat.add(Quadrupole("QD", 50.0, gradient=-5.0, aperture=20.0))
    lat.add(Drift("D2", 200.0))
    st = AppState()
    st.set_lattice(lat, "<test>")
    st.set_beam_config(BeamConfig(
        species="proton", energy=3.0, frequency=352.21, current=0.0,
        duty_cycle=100.0, n_particles=100, distribution="gaussian",
        cutoff=3.0, emit_nx=0.25, alpha_x=0.0, beta_x=1.0,
        emit_ny=0.25, alpha_y=0.0, beta_y=1.0,
        emit_z=0.30, alpha_z=0.0, beta_z=5.0))
    return st


def test_panel_opens_and_reports_via_mock(qapp, monkeypatch, tmp_path):
    from linac_gen.assist.config import AssistConfig
    from linac_gen.assist.testing import (
        MockProvider, turn_text, turn_tools,
    )
    from linac_gen_gui.interphase.dialogs.assistant_panel import (
        AssistantPanel,
    )

    st = _state_with_lattice(qapp)
    panel = AssistantPanel(None, st)

    # replace the (unconfigured) session with a mock-backed one
    from linac_gen.assist.agent import AgentSession
    from linac_gen.assist.ledger import Ledger
    from linac_gen_gui.interphase.dialogs.assistant_panel import (
        _make_context, _GuiApprover,
    )
    ctx = _make_context(st, str(tmp_path))
    panel._approver = _GuiApprover(panel)
    provider = MockProvider([
        turn_tools(("get_status", {})),
        turn_text("The lattice has 4 elements (source: session)."),
    ])
    panel._session = AgentSession(
        AssistConfig(provider="openai", model="mock",
                     base_url="http://blocked/v1", api_key=""),
        ctx, approver=panel._approver, provider=provider,
        ledger=Ledger(str(tmp_path)),
        on_event=panel._on_event_threadsafe)
    panel._set_enabled(True)

    try:
        panel._input.setText("what's loaded?")
        panel._send()
        # pump the worker thread to completion
        assert panel._worker is not None
        for _ in range(200):
            qapp.processEvents()
            if not panel._worker.isRunning():
                break
        panel._worker.wait(5000)
        qapp.processEvents()
        text = panel._transcript.toPlainText()
        assert "4 elements" in text
        # a read tool needs NO confirmation
        assert panel._btn_approve.isHidden()
    finally:
        panel.shutdown()


def test_panel_confirmation_echoes_and_syncs_state(qapp, tmp_path):
    from linac_gen.assist.agent import AgentSession, Decision
    from linac_gen.assist.config import AssistConfig
    from linac_gen.assist.ledger import Ledger
    from linac_gen.assist.testing import (
        MockProvider, turn_text, turn_tools,
    )
    from linac_gen_gui.interphase.dialogs.assistant_panel import (
        AssistantPanel, _GuiApprover, _make_context,
    )

    st = _state_with_lattice(qapp)
    panel = AssistantPanel(None, st)
    ctx = _make_context(st, str(tmp_path))
    panel._approver = _GuiApprover(panel)
    provider = MockProvider([
        turn_tools(("set_element_param",
                    {"element_name": "QF", "param": "gradient",
                     "value": 8.0})),
        turn_text("QF gradient set to 8.0."),
    ])
    panel._session = AgentSession(
        AssistConfig(provider="openai", model="mock",
                     base_url="http://blocked/v1", api_key=""),
        ctx, approver=panel._approver, provider=provider,
        ledger=Ledger(str(tmp_path)),
        on_event=panel._on_event_threadsafe)
    panel._set_enabled(True)

    try:
        panel._input.setText("set QF gradient to 8")
        panel._send()
        # wait for the confirmation strip to appear (mutate → always)
        for _ in range(200):
            qapp.processEvents()
            if not panel._btn_approve.isHidden():
                break
        assert not panel._btn_approve.isHidden()
        # the echoed call shows the exact resolved params + tier
        label = panel._confirm_label.text()
        assert "mutate" in label and "gradient" in label
        assert "8.0" in label and "QF" in label
        panel._on_approve()
        for _ in range(200):
            qapp.processEvents()
            if not panel._worker.isRunning():
                break
        panel._worker.wait(5000)
        # the mutation reached the real lattice via AppState
        assert st.lattice.elements[0].gradient == 8.0
    finally:
        panel.shutdown()


def test_panel_close_midturn_does_not_hang(qapp, tmp_path):
    """A confirmation left pending when the panel closes must release
    the worker (abort), never deadlock."""
    from linac_gen.assist.agent import AgentSession
    from linac_gen.assist.config import AssistConfig
    from linac_gen.assist.ledger import Ledger
    from linac_gen.assist.testing import MockProvider, turn_tools
    from linac_gen_gui.interphase.dialogs.assistant_panel import (
        AssistantPanel, _GuiApprover, _make_context,
    )

    st = _state_with_lattice(qapp)
    panel = AssistantPanel(None, st)
    ctx = _make_context(st, str(tmp_path))
    panel._approver = _GuiApprover(panel)
    provider = MockProvider([
        turn_tools(("set_element_param",
                    {"element_name": "QF", "param": "gradient",
                     "value": 9.0}))])
    panel._session = AgentSession(
        AssistConfig(provider="openai", model="mock",
                     base_url="http://blocked/v1", api_key=""),
        ctx, approver=panel._approver, provider=provider,
        ledger=Ledger(str(tmp_path)),
        on_event=panel._on_event_threadsafe)
    panel._set_enabled(True)
    panel._input.setText("set QF gradient to 9")
    panel._send()
    for _ in range(200):
        qapp.processEvents()
        if not panel._btn_approve.isHidden():
            break
    # close WITHOUT answering — must not hang
    panel.shutdown()
    qapp.processEvents()
    assert panel._worker.wait(5000) is True      # thread exited


def test_panel_offers_keyless_claude_sdk_backend(qapp, monkeypatch, tmp_path):
    """The panel offers the keyless 'Claude (subscription login)' backend
    and, when picked, starts a session with provider=claude_sdk and NO key.
    QSettings is redirected to a temp store so real prefs are untouched, and
    the login probe is stubbed so no real claude CLI is required."""
    from PyQt6.QtCore import QSettings
    from linac_gen_gui.interphase.dialogs import assistant_panel as ap
    from linac_gen_gui.interphase.state import AppState

    store = QSettings(str(tmp_path / "s.ini"), QSettings.Format.IniFormat)
    monkeypatch.setattr(ap.AssistantPanel, "_settings", lambda self: store)
    monkeypatch.setattr("linac_gen.assist.sdk_backend.sdk_available",
                        lambda: (True, "ok"))

    panel = ap.AssistantPanel(None, AppState())
    try:
        idx = panel._provider.findData("claude_sdk")
        assert idx >= 0                       # the keyless option is offered
        panel._provider.setCurrentIndex(idx)
        panel._key_edit.setText("")           # deliberately NO api key
        panel._on_connect()
        assert panel._session is not None
        assert panel._session.config.provider == "claude_sdk"
        assert panel._session.config.api_key == ""     # keyless
        assert panel._session._sdk is not None         # SDK backend wired
    finally:
        panel.shutdown()


def test_panel_keyless_backend_shows_guidance_when_login_missing(
        qapp, monkeypatch, tmp_path):
    """If the claude CLI/login is missing, picking the keyless backend must
    surface setup guidance, not start a dead session."""
    from PyQt6.QtCore import QSettings
    from linac_gen_gui.interphase.dialogs import assistant_panel as ap
    from linac_gen_gui.interphase.state import AppState

    store = QSettings(str(tmp_path / "s.ini"), QSettings.Format.IniFormat)
    monkeypatch.setattr(ap.AssistantPanel, "_settings", lambda self: store)
    monkeypatch.setattr(
        "linac_gen.assist.sdk_backend.sdk_available",
        lambda: (False, "the `claude` CLI was not found — install Claude Code"))

    panel = ap.AssistantPanel(None, AppState())
    try:
        panel._provider.setCurrentIndex(panel._provider.findData("claude_sdk"))
        panel._on_connect()
        assert panel._session is None                  # not started
        assert "claude` CLI" in panel._status.text()
    finally:
        panel.shutdown()


def test_panel_streams_deltas_into_transcript(qapp, monkeypatch, tmp_path):
    """Streaming render: incremental deltas type into the transcript without
    per-token newlines, and the finalized segment is captured for speak-back."""
    from PyQt6.QtCore import QSettings
    from linac_gen_gui.interphase.dialogs import assistant_panel as ap
    from linac_gen_gui.interphase.state import AppState

    store = QSettings(str(tmp_path / "s.ini"), QSettings.Format.IniFormat)
    monkeypatch.setattr(ap.AssistantPanel, "_settings", lambda self: store)
    panel = ap.AssistantPanel(None, AppState())
    try:
        panel._stream_buf = ""
        panel._streaming = False
        panel._last_reply = ""
        panel._stream_delta("800 MeV ")
        panel._stream_delta("H-.")
        panel._stream_done()
        text = panel._transcript.toPlainText()
        assert "800 MeV H-." in text          # single line, no token newlines
        assert panel._last_reply == "800 MeV H-."   # captured for TTS
        assert panel._streaming is False
    finally:
        panel.shutdown()


def test_backend_button_reopens_provider_row_for_key_switch(
        qapp, monkeypatch, tmp_path):
    """Regression: the provider row hides after a successful connect and
    nothing re-showed it — the saved backend auto-connects at panel open,
    so switching (e.g. subscription login -> API key) was unreachable.
    The 'backend…' button must re-open the row, and Connect must replace
    the running session with the newly chosen backend."""
    from PyQt6.QtCore import QSettings
    from linac_gen_gui.interphase.dialogs import assistant_panel as ap
    from linac_gen_gui.interphase.state import AppState

    store = QSettings(str(tmp_path / "s.ini"), QSettings.Format.IniFormat)
    monkeypatch.setattr(ap.AssistantPanel, "_settings", lambda self: store)
    monkeypatch.setattr("linac_gen.assist.sdk_backend.sdk_available",
                        lambda: (True, "ok"))

    panel = ap.AssistantPanel(None, AppState())
    try:
        assert panel._settings_box.isVisibleTo(panel)   # nothing saved yet
        idx = panel._provider.findData("claude_sdk")
        panel._provider.setCurrentIndex(idx)
        panel._on_connect()
        first = panel._session
        assert first is not None
        assert not panel._settings_box.isVisibleTo(panel)   # hidden now

        panel._backend_btn.click()                          # re-open
        assert panel._settings_box.isVisibleTo(panel)

        idx = panel._provider.findData("anthropic")
        panel._provider.setCurrentIndex(idx)
        panel._key_edit.setText("sk-test-not-a-real-key")
        panel._on_connect()                                 # switch
        assert panel._session is not None and panel._session is not first
        assert panel._session.config.provider == "anthropic"
        assert panel._session.config.api_key == "sk-test-not-a-real-key"
        assert not panel._settings_box.isVisibleTo(panel)   # hidden again
        # the choice persisted (used at next panel open)
        assert store.value("assist/provider") == "anthropic"
    finally:
        panel.shutdown()


def test_model_choice_honored_for_api_key_and_local_backends(
        qapp, monkeypatch, tmp_path):
    """Regression: the local backend hard-coded model='llama3.1' at
    connect (a typed model name was ignored until an app restart) and a
    URL in the box erased any chance to pick one.  The box must accept
    'URL', 'model', or 'URL model' — and anthropic must honor the model
    box (e.g. a litellm alias) rather than only its default."""
    from PyQt6.QtCore import QSettings
    from linac_gen_gui.interphase.dialogs import assistant_panel as ap
    from linac_gen_gui.interphase.state import AppState

    store = QSettings(str(tmp_path / "s.ini"), QSettings.Format.IniFormat)
    monkeypatch.setattr(ap.AssistantPanel, "_settings", lambda self: store)

    panel = ap.AssistantPanel(None, AppState())
    try:
        # -- local: "URL model" in one go ------------------------------
        panel._provider.setCurrentIndex(panel._provider.findData("openai"))
        panel._model_edit.setCurrentText("http://localhost:11434/v1 qwen2.5")
        panel._on_connect()
        assert panel._session.config.base_url == "http://localhost:11434/v1"
        assert panel._session.config.model == "qwen2.5"

        # -- local: bare model name reuses the saved URL ---------------
        panel._backend_btn.click()
        panel._model_edit.setCurrentText("mistral")
        panel._on_connect()
        assert panel._session.config.base_url == "http://localhost:11434/v1"
        assert panel._session.config.model == "mistral"
        assert store.value("assist/model") == "mistral"

        # -- local: bare URL keeps the saved model ---------------------
        panel._backend_btn.click()
        panel._model_edit.setCurrentText("http://127.0.0.1:8000/v1")
        panel._on_connect()
        assert panel._session.config.base_url == "http://127.0.0.1:8000/v1"
        assert panel._session.config.model == "mistral"

        # -- anthropic: model box honored (litellm-style alias) --------
        panel._backend_btn.click()
        panel._provider.setCurrentIndex(
            panel._provider.findData("anthropic"))
        panel._key_edit.setText("sk-test-not-a-real-key")
        panel._model_edit.setCurrentText("azure/claude-sonnet-4-6")
        panel._on_connect()
        assert panel._session.config.provider == "anthropic"
        assert panel._session.config.model == "azure/claude-sonnet-4-6"
    finally:
        panel.shutdown()


def test_model_dropdown_suggestions_and_local_server_fetch(
        qapp, monkeypatch, tmp_path):
    """The model box is an editable dropdown: per-backend suggestions
    (subscription aliases, Anthropic IDs), a server-fetched list for the
    local backend (through the real daemon thread + queued signal), a
    staleness guard, and typed free text still wins."""
    import json
    import time
    import types
    import io
    from PyQt6.QtCore import QSettings
    from linac_gen_gui.interphase.dialogs import assistant_panel as ap
    from linac_gen_gui.interphase.state import AppState

    store = QSettings(str(tmp_path / "s.ini"), QSettings.Format.IniFormat)
    monkeypatch.setattr(ap.AssistantPanel, "_settings", lambda self: store)

    panel = ap.AssistantPanel(None, AppState())
    try:
        def items():
            return [panel._model_edit.itemText(i)
                    for i in range(panel._model_edit.count())]

        # anthropic (default provider): IDs offered, blank first
        assert items()[0] == ""
        assert "claude-sonnet-5" in items()
        assert "claude-opus-4-8" in items()

        # subscription: SDK aliases offered (incl. the Fable tier)
        panel._provider.setCurrentIndex(
            panel._provider.findData("claude_sdk"))
        assert {"fable", "sonnet", "opus", "haiku"} <= set(items())

        # local: server list fetched via the REAL thread + queued signal
        payload = {"data": [{"id": "qwen2.5"}, {"id": "mistral"}]}

        class _Resp(io.BytesIO):
            def __enter__(self):
                return self
            def __exit__(self, *a):
                return False

        seen = {}
        def fake_urlopen(url, timeout=None):
            seen["url"] = url
            return _Resp(json.dumps(payload).encode())

        import urllib.request
        monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
        store.setValue("assist/base_url", "http://localhost:11434/v1")
        panel._provider.setCurrentIndex(panel._provider.findData("openai"))
        deadline = time.time() + 3.0
        while "mistral" not in items() and time.time() < deadline:
            qapp.processEvents()
            time.sleep(0.01)
        assert {"qwen2.5", "mistral"} <= set(items())
        assert seen["url"] == "http://localhost:11434/v1/models"

        # staleness guard: a late reply after switching backends is ignored
        panel._provider.setCurrentIndex(
            panel._provider.findData("claude_sdk"))
        panel._on_models_listed("http://localhost:11434/v1", ["late-model"])
        assert "late-model" not in items()

        # typed free text is never clobbered by repopulation
        panel._model_edit.setCurrentText("my/custom-alias")
        panel._repopulate_models("claude_sdk")
        assert panel._model_edit.currentText() == "my/custom-alias"
    finally:
        panel.shutdown()


def test_listen_failed_no_retry_stops_cleanly(qapp):
    """retry=False (missing optional voice dep): no backoff loop, wake
    button unchecks — the Windows report showed 6 stacked sounddevice
    tracebacks from wake auto-enable + 5 futile retries."""
    from linac_gen_gui.interphase.dialogs.assistant_panel import (
        AssistantPanel,
    )

    st = _state_with_lattice(qapp)
    panel = AssistantPanel(None, st)
    try:
        panel._wake_btn.blockSignals(True)
        panel._wake_btn.setChecked(True)
        panel._wake_btn.blockSignals(False)
        before = getattr(panel, "_reopen_attempt", 0)
        panel._listen_start_failed(
            panel._listen_gen, "No module named 'sounddevice'", retry=False)
        assert not panel._wake_btn.isChecked()
        assert getattr(panel, "_reopen_attempt", 0) == before  # no backoff
        assert "voice unavailable" in panel._prog.text()
    finally:
        panel.shutdown()


# ---------------------------------------------------------------------------
# Assistant edits flow through the CommandBus (undo / dirty / tab sync)
# ---------------------------------------------------------------------------
def _inspector_gradient(insp, qapp):
    """Walk the LIVE inspector layout rows for the Gradient spinbox value
    (stale rows are deleteLater'd and no longer in the layout)."""
    from PyQt6.QtWidgets import QDoubleSpinBox, QLabel
    qapp.processEvents()
    lay = insp._body_lay
    for i in range(lay.count()):
        row = lay.itemAt(i).widget()
        if row is None:
            continue
        labs = row.findChildren(QLabel)
        if labs and labs[0].text() == "Gradient":
            sb = row.findChildren(QDoubleSpinBox)
            return sb[0].value() if sb else None
    return None


def _mock_session(panel, st, tmp_path, turns):
    """Replace the panel's session with a MockProvider-backed one whose
    context proxies AppState WITH the panel as nav — the app's own wiring
    (`_make_context(self.state, calc_dir, nav=self)`)."""
    from linac_gen.assist.agent import AgentSession
    from linac_gen.assist.config import AssistConfig
    from linac_gen.assist.ledger import Ledger
    from linac_gen.assist.testing import MockProvider
    from linac_gen_gui.interphase.dialogs.assistant_panel import (
        _GuiApprover, _make_context,
    )
    ctx = _make_context(st, str(tmp_path), nav=panel)
    panel._approver = _GuiApprover(panel)
    panel._session = AgentSession(
        AssistConfig(provider="openai", model="mock",
                     base_url="http://blocked/v1", api_key=""),
        ctx, approver=panel._approver, provider=MockProvider(turns),
        ledger=Ledger(str(tmp_path)),
        on_event=panel._on_event_threadsafe)
    panel._set_enabled(True)
    return ctx


def test_assistant_param_edit_goes_through_command_bus(win, qapp, tmp_path):
    """An approved set_element_param in the REAL window (worker QThread →
    _GuiApprover → tool) pushes exactly ONE undoable command through
    state.bus: listing + inspector refresh, Undo works, dirty is True."""
    import json as _json
    import pathlib
    from linac_gen.assist.testing import turn_text, turn_tools

    repo = pathlib.Path(__file__).resolve().parents[2]
    win.open_lattice(repo / "examples" / "fodo_cell.dat")
    st = win.state
    quad = next(e for e in st.lattice.elements
                if type(e).__name__ == "Quadrupole")
    assert quad.gradient == 5.0
    st.set_selected(quad)
    qapp.processEvents()

    win._open_assistant()
    qapp.processEvents()
    panel = win._assistant_panel
    _mock_session(panel, st, tmp_path, [
        turn_tools(("set_element_param",
                    {"element_name": quad.name, "param": "gradient",
                     "value": 7.5})),
        turn_text("done."),
    ])

    fired = {"lattice_changed": 0, "bus_changed": 0, "dirty": []}
    st.lattice_changed.connect(
        lambda *_: fired.__setitem__("lattice_changed",
                                     fired["lattice_changed"] + 1))
    st.bus.changed.connect(
        lambda *_: fired.__setitem__("bus_changed",
                                     fired["bus_changed"] + 1))
    st.bus.dirty_changed.connect(lambda v: fired["dirty"].append(bool(v)))

    try:
        panel._input.setText(f"set {quad.name} gradient to 7.5")
        panel._send()
        assert win.pump_until(lambda: not panel._btn_approve.isHidden())
        panel._on_approve()
        assert win.pump_until(lambda: not panel._worker.isRunning())
        panel._worker.wait(5000)
        win.pump(0.3)

        # exactly one undoable command through the bus
        assert fired["lattice_changed"] == 1
        assert fired["bus_changed"] == 1
        assert fired["dirty"] == [True]
        assert quad.gradient == 7.5
        # listing + inspector show the new value WITHOUT reselecting
        listing = win.lattice_tab._listing
        row = listing._list.item(st.lattice.elements.index(quad)).text()
        assert "G=7.5T/m" in row
        assert _inspector_gradient(win.lattice_tab._inspector, qapp) == 7.5
        # undo machinery armed
        assert st.bus.can_undo is True
        assert len(st.bus._undo) == 1
        assert st.bus.dirty is True
        assert win.lattice_tab._btn_undo.isEnabled()
        lines = st.bus.describe_changes_since_clean()
        assert lines and lines[0].startswith("Assistant edit")
        # the ledger recorded the tool call as ok
        recs = []
        for p in (tmp_path / "assist_sessions").glob("*.jsonl"):
            recs += [_json.loads(ln)
                     for ln in p.read_text().splitlines() if ln]
        tool_recs = [r for r in recs if r.get("event") == "tool"
                     and r.get("tool") == "set_element_param"]
        assert tool_recs and tool_recs[0]["status"] == "ok"
        # one click of the Lattice-tab Undo restores the value bit-exact
        win.lattice_tab._btn_undo.click()
        qapp.processEvents()
        assert quad.gradient == 5.0
        row = listing._list.item(st.lattice.elements.index(quad)).text()
        assert "G=5T/m" in row
        assert st.bus.dirty is False
    finally:
        panel.shutdown()


@pytest.mark.parametrize(
    "scenario", ["kept", "missed", "missed_after_user_edit"])
def test_assistant_tuning_plan_rollback_composes_with_undo(
        qapp, monkeypatch, tmp_path, scenario):
    """tuning_plan in the GUI: window met → ONE undo step with all knobs;
    window missed → the plan's own step is undone (history exactly as
    before, plan on the redo stack); a user edit landing mid-run makes
    the rollback a compensating step instead."""
    import json as _json
    import threading
    import time as _time
    from linac_gen.assist.tools import TOOLS
    from linac_gen_gui.interphase.commands import ParamChangeCommand
    from linac_gen_gui.interphase.dialogs.assistant_panel import (
        AssistantPanel, _make_context,
    )

    st = _state_with_lattice(qapp)
    panel = AssistantPanel(None, st)
    ctx = _make_context(st, str(tmp_path), nav=panel)
    qf, qd = st.lattice.elements[0], st.lattice.elements[2]
    g0, gd0 = qf.gradient, qd.gradient

    if scenario == "kept":
        # prior inspector-style edit → non-trivial baseline (dirty0 True)
        st.bus.do(ParamChangeCommand(qd, "aperture", qd.aperture, 21.0))
    depth0 = len(st.bus._undo)
    dirty0 = st.bus.dirty

    fired = {"lattice_changed": 0}
    st.lattice_changed.connect(
        lambda *_: fired.__setitem__("lattice_changed",
                                     fired["lattice_changed"] + 1))

    in_verify = threading.Event()
    release = threading.Event()
    if scenario == "missed_after_user_edit":
        real_run = TOOLS["run_envelope"].fn

        def _blocking_run(c, **kw):
            in_verify.set()
            assert release.wait(30)
            return real_run(c, **kw)

        monkeypatch.setattr(TOOLS["run_envelope"], "fn", _blocking_run)

    window = 1000.0 if scenario == "kept" else 1e-9
    knobs = _json.dumps([
        {"element_name": "QF", "param": "gradient", "value": 5.2},
        {"element_name": "QD", "param": "gradient", "value": -5.2}])
    box = {}

    def _worker():
        box["env"] = TOOLS["tuning_plan"].fn(
            ctx, knobs_json=knobs, objective="sigma_x",
            success_below=window, mode="envelope", measure_before=False)

    th = threading.Thread(target=_worker, name="fake-job-thread")
    try:
        th.start()
        if scenario == "missed_after_user_edit":
            t0 = _time.time()
            while not in_verify.is_set() and _time.time() - t0 < 30:
                qapp.processEvents()
            assert in_verify.is_set()
            # knobs are applied; land a user edit on TOP of the stack
            st.bus.do(ParamChangeCommand(qd, "aperture", qd.aperture, 22.0))
            release.set()
        t0 = _time.time()
        while th.is_alive() and _time.time() - t0 < 120:
            qapp.processEvents()
        th.join(10)
        assert not th.is_alive()
        env = box["env"]
        assert env["status"] == "ok"
    finally:
        release.set()
        th.join(10)
        panel.shutdown()

    if scenario == "kept":
        assert env["data"]["kept"] is True
        assert qf.gradient == 5.2 and qd.gradient == -5.2
        assert len(st.bus._undo) == depth0 + 1     # ONE step, no coalescing
        assert st.bus.dirty is True
        d = st.bus.peek_undo().describe()
        assert d.startswith("Assistant tuning plan")
        assert "QF" in d and "QD" in d
        st.bus.undo()                              # one undo → both knobs
        assert qf.gradient == g0 and qd.gradient == gd0
    elif scenario == "missed":
        assert env["data"]["kept"] is False
        assert qf.gradient == g0 and qd.gradient == gd0   # bit-exact ==
        assert len(st.bus._undo) == depth0
        assert st.bus.dirty == dirty0
        assert len(st.bus._redo) == 1     # honest history: plan on redo
        assert fired["lattice_changed"] == 2              # apply + undo
    else:
        assert env["data"]["kept"] is False
        assert qf.gradient == g0 and qd.gradient == gd0   # bit-exact ==
        assert qd.aperture == 22.0            # the user edit survives
        # plan step + user edit + compensating rollback step
        assert len(st.bus._undo) == depth0 + 3
        assert st.bus.dirty is True
        assert "rollback" in st.bus.peek_undo().describe()


def test_assistant_edit_refused_when_element_orphaned(qapp, tmp_path):
    """The GUI hook refuses to apply to an element that is no longer in
    the loaded lattice (replaced mid-turn), or when no lattice is loaded
    — lattice, undo depth and dirty flag stay untouched."""
    from linac_gen.core.lattice import Lattice
    from linac_gen.elements.drift import Drift
    from linac_gen_gui.interphase.dialogs.assistant_panel import (
        _make_context,
    )

    st = _state_with_lattice(qapp)
    ctx = _make_context(st, str(tmp_path))       # nav=None: same thread
    elem = st.lattice.elements[0]
    other = Lattice()
    other.add(Drift("D_OTHER", 100.0))
    st.set_lattice(other, "<b>")                 # orphans `elem`

    with pytest.raises(RuntimeError,
                       match="no longer in the loaded lattice"):
        ctx.apply_param_changes([(elem, "gradient", 9.0)],
                                label="Assistant edit")
    assert elem.gradient == 5.0
    assert len(st.bus._undo) == 0
    assert st.bus.dirty is False

    st.set_lattice(None, "")
    with pytest.raises(RuntimeError, match="no lattice loaded"):
        ctx.apply_param_changes([(elem, "gradient", 9.0)],
                                label="Assistant edit")
    assert elem.gradient == 5.0


def test_assistant_edit_gui_timeout_reports_error(qapp, tmp_path):
    """run_on_gui returning None (timeout / panel closing) must surface
    as RuntimeError from the hook and status='error' from the tool —
    never ok for an edit that did not land; lattice and bus untouched."""
    from linac_gen.assist.tools import TOOLS
    from linac_gen_gui.interphase.dialogs.assistant_panel import (
        _make_context,
    )

    class _DeafNav:
        """A panel whose GUI thread never answers."""

        def run_on_gui(self, fn, timeout=3.0):
            return None

    st = _state_with_lattice(qapp)
    ctx = _make_context(st, str(tmp_path), nav=_DeafNav())
    qf = st.lattice.elements[0]

    with pytest.raises(RuntimeError, match="did not apply"):
        ctx.apply_param_changes([(qf, "gradient", 9.0)],
                                label="Assistant edit")
    assert qf.gradient == 5.0
    assert len(st.bus._undo) == 0 and st.bus.dirty is False

    env = TOOLS["set_element_param"].fn(
        ctx, element_name="QF", param="gradient", value=9.0)
    assert env["status"] == "error"
    assert "could not apply" in env["data"]["message"]
    assert qf.gradient == 5.0
    assert len(st.bus._undo) == 0 and st.bus.dirty is False


def test_assistant_revert_timeout_cannot_land_late(qapp, tmp_path):
    """A revert whose GUI hop times out must raise — and when the GUI
    thread finally runs the queued thunk, it must be a NO-OP: the
    revert used to hand ``_gui_sync`` a throwaway cancelled box its
    ``_do`` never consulted, so a timed-out revert still landed late."""
    from linac_gen_gui.interphase.dialogs.assistant_panel import (
        _make_context,
    )

    class _FlakyNav:
        """Answers the first hop (apply), goes deaf afterwards but keeps
        the queued thunk — the real panel's late-delivery shape."""

        def __init__(self):
            self.deaf = False
            self.late = []

        def run_on_gui(self, fn, timeout=3.0):
            if self.deaf:
                self.late.append(fn)
                return None          # timeout — thunk still queued
            return fn()

    nav = _FlakyNav()
    st = _state_with_lattice(qapp)
    ctx = _make_context(st, str(tmp_path), nav=nav)
    qf = st.lattice.elements[0]

    handle = ctx.apply_param_changes([(qf, "gradient", 9.0)],
                                     label="Assistant edit")
    assert qf.gradient == 9.0 and len(st.bus._undo) == 1

    nav.deaf = True
    with pytest.raises(RuntimeError, match="did not apply"):
        ctx.revert_param_changes(handle, [(qf, "gradient", 5.0)],
                                 label="Assistant edit")

    # The GUI thread runs the queued thunk AFTER the timeout was
    # reported: it must refuse (cancelled) and touch nothing.
    assert len(nav.late) == 1
    out = nav.late[0]()
    assert out == {"ok": False, "err": "cancelled"}
    assert qf.gradient == 9.0                 # edit still in place
    assert len(st.bus._undo) == 1             # no late undo landed
    assert st.bus.peek_undo() is handle


def test_assistant_revert_refused_after_lattice_swap(qapp, tmp_path):
    """Lattice replaced after the apply (bus.reset): the compensating
    branch must refuse instead of pushing a MacroCommand over orphaned
    elements of the discarded lattice — nothing mutated, nothing pushed
    onto the NEW lattice's pristine bus."""
    from linac_gen.core.lattice import Lattice
    from linac_gen.elements.drift import Drift
    from linac_gen_gui.interphase.dialogs.assistant_panel import (
        _make_context,
    )

    st = _state_with_lattice(qapp)
    ctx = _make_context(st, str(tmp_path))       # nav=None: same thread
    qf = st.lattice.elements[0]
    handle = ctx.apply_param_changes([(qf, "gradient", 9.0)],
                                     label="Assistant edit")
    assert qf.gradient == 9.0

    other = Lattice()
    other.add(Drift("D_OTHER", 100.0))
    st.set_lattice(other, "<b>")     # bus.reset(): handle gone, qf orphaned

    with pytest.raises(RuntimeError,
                       match="no longer in the loaded lattice"):
        ctx.revert_param_changes(handle, [(qf, "gradient", 5.0)],
                                 label="Assistant edit")
    assert qf.gradient == 9.0                    # orphan left untouched
    assert len(st.bus._undo) == 0                # nothing pushed
    assert st.bus.dirty is False


def test_assistant_revert_refused_when_no_lattice_loaded(qapp, tmp_path):
    """Compensating-branch guard, empty regime: lattice unloaded after
    the apply — revert refuses instead of pushing onto a bare bus."""
    from linac_gen_gui.interphase.dialogs.assistant_panel import (
        _make_context,
    )

    st = _state_with_lattice(qapp)
    ctx = _make_context(st, str(tmp_path))
    qf = st.lattice.elements[0]
    handle = ctx.apply_param_changes([(qf, "gradient", 9.0)],
                                     label="Assistant edit")
    st.set_lattice(None, "")
    with pytest.raises(RuntimeError, match="no lattice loaded"):
        ctx.revert_param_changes(handle, [(qf, "gradient", 5.0)],
                                 label="Assistant edit")
    assert qf.gradient == 9.0
    assert len(st.bus._undo) == 0
