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
