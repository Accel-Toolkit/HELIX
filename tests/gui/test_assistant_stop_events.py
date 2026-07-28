"""Phase-1 panel features: Esc-as-Stop, reusable Stop, machine-event
rendering + idle narration, progress label, confirm timeout auto-deny."""
from __future__ import annotations

import time

import pytest

pytest.importorskip("PyQt6")

from tests.gui.test_assistant_panel import _state_with_lattice  # noqa: E402


def _mock_panel(qapp, tmp_path, turns, timeout_s: float = 0.0):
    from linac_gen.assist.agent import AgentSession
    from linac_gen.assist.config import AssistConfig
    from linac_gen.assist.ledger import Ledger
    from linac_gen.assist.testing import MockProvider
    from linac_gen_gui.interphase.dialogs.assistant_panel import (
        AssistantPanel, _GuiApprover, _make_context,
    )
    st = _state_with_lattice(qapp)
    panel = AssistantPanel(None, st)
    ctx = _make_context(st, str(tmp_path))
    panel._approver = _GuiApprover(panel, timeout_s=timeout_s)
    provider = MockProvider(list(turns))
    panel._session = AgentSession(
        AssistConfig(provider="openai", model="mock",
                     base_url="http://blocked/v1", api_key=""),
        ctx, approver=panel._approver, provider=provider,
        ledger=Ledger(str(tmp_path)),
        on_event=panel._on_event_threadsafe)
    panel._set_enabled(True)
    return panel, provider


def _pump_worker(qapp, panel, timeout_s=8.0):
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        qapp.processEvents()
        w = panel._worker
        if w is not None and not w.isRunning():
            w.wait(2000)
            qapp.processEvents()
            return
        time.sleep(0.01)
    raise AssertionError("worker did not finish")


def test_esc_stops_instead_of_closing(qapp, tmp_path):
    from PyQt6.QtCore import Qt
    from PyQt6.QtGui import QKeyEvent
    from PyQt6.QtCore import QEvent
    from linac_gen.assist.testing import turn_text

    panel, _ = _mock_panel(qapp, tmp_path, [turn_text("hi")])
    panel.show()
    qapp.processEvents()
    ev = QKeyEvent(QEvent.Type.KeyPress, Qt.Key.Key_Escape,
                   Qt.KeyboardModifier.NoModifier)
    panel.keyPressEvent(ev)
    qapp.processEvents()
    assert panel.isVisible()               # QDialog default would close
    assert "stopped." in panel._transcript.toPlainText()
    panel.close()


def test_stop_between_turns_keeps_session_usable(qapp, tmp_path):
    from linac_gen.assist.testing import turn_text
    panel, _ = _mock_panel(qapp, tmp_path,
                           [turn_text("first"), turn_text("second")])
    panel._input.setText("one")
    panel._send()
    _pump_worker(qapp, panel)
    panel._on_stop()                       # Stop button between turns
    panel._input.setText("two")
    panel._send()
    _pump_worker(qapp, panel)
    text = panel._transcript.toPlainText()
    assert "first" in text and "second" in text
    panel.close()


def test_event_line_renders_without_a_worker(qapp, tmp_path):
    from linac_gen.assist.testing import turn_text
    panel, _ = _mock_panel(qapp, tmp_path, [turn_text("unused")])
    panel._session.submit_event("watch: transmission fell 2.3 pts")
    qapp.processEvents()
    assert "‣ [event] watch: transmission fell" in \
        panel._transcript.toPlainText()
    panel.close()


def test_narrate_checkbox_gives_the_model_a_turn(qapp, tmp_path):
    from linac_gen.assist.testing import turn_text
    panel, provider = _mock_panel(
        qapp, tmp_path, [turn_text("I see the run finished — looks ok.")])
    panel._narrate.setChecked(True)
    panel._session.submit_event("job finished: done")
    qapp.processEvents()                   # event line + worker start
    _pump_worker(qapp, panel)
    text = panel._transcript.toPlainText()
    assert "looks ok" in text
    assert "▶" not in text                 # no fake user line
    from linac_gen.assist.messages import SystemNote
    notes = [m for m in provider.requests[0]["transcript"]
             if isinstance(m, SystemNote)]
    assert any("job finished" in n.text for n in notes)
    panel.close()


def test_progress_event_updates_label(qapp, tmp_path):
    from linac_gen.assist.testing import turn_text
    panel, _ = _mock_panel(qapp, tmp_path, [turn_text("x")])
    panel._session._emit(type="progress", tool="run_mp",
                         text="run_mp: s = 12.3 m")
    qapp.processEvents()
    assert panel._prog.text() == "run_mp: s = 12.3 m"
    panel.close()


def test_confirm_timeout_auto_denies_and_hides_strip(qapp, tmp_path):
    from linac_gen.assist.agent import Decision
    from linac_gen.assist.testing import turn_text, turn_tools
    panel, _ = _mock_panel(
        qapp, tmp_path,
        [turn_tools(("run_envelope", {})), turn_text("denied then")],
        timeout_s=0.3)
    panel._input.setText("run it")
    panel._send()
    # pump until the strip appears, then let the timeout fire
    t0 = time.time()
    while time.time() - t0 < 5.0:
        qapp.processEvents()
        if panel._confirm_label.isVisible():
            break
        time.sleep(0.01)
    _pump_worker(qapp, panel, timeout_s=10.0)
    assert not panel._confirm_label.isVisible()
    assert "auto-denied" in panel._transcript.toPlainText()
    panel.close()


def test_stale_button_click_cannot_resolve_next_request(qapp, tmp_path):
    """Generation guard: a resolve stamped with an old generation is
    ignored once a newer confirmation is pending."""
    from linac_gen.assist.agent import Decision
    from linac_gen.assist.testing import turn_text
    from linac_gen_gui.interphase.dialogs.assistant_panel import (
        _GuiApprover,
    )
    panel, _ = _mock_panel(qapp, tmp_path, [turn_text("x")])
    ap = panel._approver
    ap._gen = 3                            # a request is pending (gen 3)
    ap.resolve(Decision.APPROVE, gen=2)    # stale click from gen 2
    assert not ap._event.is_set()
    ap.resolve(Decision.APPROVE, gen=3)    # current click
    assert ap._event.is_set()
    panel.close()


def test_results_changed_feeds_run_watch(qapp, tmp_path):
    """Phase 6: a run finishing (user- OR assistant-initiated) is
    inspected and alerts ride the event channel into the transcript."""
    from types import SimpleNamespace
    import numpy as np
    from linac_gen.assist.testing import turn_text
    panel, _ = _mock_panel(qapp, tmp_path, [turn_text("x")])
    panel._watch_chk.setChecked(True)

    def _res(tr_final):
        n = 50
        return SimpleNamespace(
            s=np.linspace(0, 5000.0, n),
            transmission=np.linspace(100.0, tr_final, n),
            sigma_x=np.full(n, 2.0), sigma_y=np.full(n, 1.5),
            emit_x=np.full(n, 0.25), emit_y=np.full(n, 0.25),
            emit_z=np.full(n, 0.30), ref_w_kin=np.full(n, 3.0))

    panel._state.set_results(_res(100.0))    # reference run
    qapp.processEvents()
    panel._state.set_results(_res(90.0))     # the bad run
    qapp.processEvents()
    text = panel._transcript.toPlainText()
    assert "‣ [event] run watch: transmission fell" in text
    panel.close()


def test_watch_checkbox_off_means_silent(qapp, tmp_path):
    from types import SimpleNamespace
    import numpy as np
    from linac_gen.assist.testing import turn_text
    panel, _ = _mock_panel(qapp, tmp_path, [turn_text("x")])
    panel._watch_chk.setChecked(False)
    n = 20
    res = SimpleNamespace(s=np.linspace(0, 100.0, n),
                          transmission=np.linspace(100, 80.0, n),
                          sigma_x=np.full(n, 2.0),
                          sigma_y=np.full(n, 30.0),
                          emit_x=np.full(n, 0.25),
                          emit_y=np.full(n, 0.25),
                          emit_z=np.full(n, 0.3),
                          ref_w_kin=np.full(n, 3.0))
    panel._state.set_results(res)
    qapp.processEvents()
    assert "[event]" not in panel._transcript.toPlainText()
    panel.close()
