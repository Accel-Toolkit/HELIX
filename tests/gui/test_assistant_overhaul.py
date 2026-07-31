"""2026-07-28 assistant-overhaul regressions (panel layer).

Pins the interaction-critical fixes from the architecture audit:
✕ hides (never a dead-session zombie), the never-drop input inbox,
Stop resolving a pending approver, the turn watchdog recovering a
dead worker, and the orb leaving "listening" after an empty capture."""
from __future__ import annotations

import time

import pytest

pytest.importorskip("PyQt6")

from tests.gui.test_assistant_stop_events import (  # noqa: E402
    _mock_panel, _pump_worker,
)


def test_close_hides_never_tears_down(qapp, tmp_path):
    """THE hang: ✕ used to kill the session while the app kept reusing
    the dead panel — every later message was silently swallowed."""
    from linac_gen.assist.testing import turn_text
    panel, _ = _mock_panel(qapp, tmp_path, [turn_text("hi"),
                                            turn_text("still alive")])
    panel.show()
    qapp.processEvents()
    panel.close()                        # the ✕ path
    qapp.processEvents()
    assert not panel.isVisible()         # hidden...
    assert panel._closing is False       # ...but NOT torn down
    assert panel._session is not None
    # and it still answers afterwards
    panel._input.setText("are you there")
    panel._send()
    _pump_worker(qapp, panel)
    assert "hi" in panel._transcript.toPlainText()
    panel.shutdown()
    assert panel._closing is True


def test_input_midturn_is_queued_not_dropped(qapp, tmp_path):
    """MIRAGE's inbox: an utterance during a running turn is HELD and
    runs when the turn ends — never silently swallowed."""
    from linac_gen.assist.testing import turn_text
    panel, _ = _mock_panel(qapp, tmp_path, [turn_text("one"),
                                            turn_text("two")])

    class _Busy:
        def isRunning(self):
            return True

    panel._worker = _Busy()                     # a turn is "running"
    panel._input.setText("second question")
    panel._send()
    panel._input.setText("third question")
    panel._send()                               # FIFO: both are held
    assert [t for t, _v in panel._inbox] == ["second question",
                                             "third question"]
    assert "queued" in panel._transcript.toPlainText()
    panel._worker = None                        # turn ends
    panel._turn_open_ui = True                  # (the real dispatch sets
    #                                             this; the fake didn't)
    panel._on_turn_done()                       # → drains FIFO head
    _pump_worker(qapp, panel)
    text = panel._transcript.toPlainText()
    assert "second question" in text
    assert "one" in text                        # the drained turn ran
    panel.shutdown()


def test_stop_resolves_pending_approver_deny(qapp, tmp_path):
    """Stop used to leave a blocked approver (and its executor thread)
    wedged forever — it must resolve DENY and hide the strip."""
    from linac_gen.assist.agent import Decision
    from linac_gen.assist.testing import turn_text
    panel, _ = _mock_panel(qapp, tmp_path, [turn_text("x")])
    ap = panel._approver
    ap._gen = 1
    ap._pending = object()
    panel._confirm_gen = 1
    panel._on_stop()
    assert ap._event.is_set()
    assert ap._decision is Decision.DENY
    assert panel._confirm_frame.isHidden()
    panel.shutdown()


def test_stop_clears_queued_inbox(qapp, tmp_path):
    from linac_gen.assist.testing import turn_text
    panel, _ = _mock_panel(qapp, tmp_path, [turn_text("x")])
    panel._inbox.append(("held utterance", True))
    panel._inbox.append(("another one", False))
    panel._on_stop()
    assert len(panel._inbox) == 0
    assert "dropped 2 queued" in panel._transcript.toPlainText()
    panel.shutdown()


def test_turn_watchdog_recovers_dead_worker(qapp, tmp_path):
    """A worker that dies without emitting turn_done used to leave the
    panel wedged in 'thinking' forever, swallowing all input."""
    from linac_gen.assist.testing import turn_text
    panel, _ = _mock_panel(qapp, tmp_path, [turn_text("x")])

    class _Dead:
        def isRunning(self):
            return False

    panel._turn_open_ui = True                  # a turn "was" running
    panel._worker = _Dead()                     # ...its thread is gone
    panel._check_turn_health()
    assert panel._turn_open_ui is False         # UI recovered
    assert "unexpectedly" in panel._transcript.toPlainText()
    assert not panel._turn_watchdog.isActive()
    panel.shutdown()


def test_turn_watchdog_notes_long_silence_without_killing(qapp, tmp_path):
    from linac_gen.assist.testing import turn_text
    panel, _ = _mock_panel(qapp, tmp_path, [turn_text("x")])

    class _Runs:
        def isRunning(self):
            return True

    panel._turn_open_ui = True
    panel._worker = _Runs()
    panel._t_activity = time.monotonic() - 400  # >5 min silent
    panel._check_turn_health()
    assert panel._turn_open_ui is True          # NOT killed (long sims!)
    assert "still working" in panel._transcript.toPlainText()
    panel._turn_watchdog.stop()
    panel._worker = None                        # drop the fake for teardown
    panel.shutdown()


def test_orb_leaves_listening_after_empty_capture(qapp, tmp_path):
    """Empty wake capture: no command fires, and the orb used to stay
    'listening' forever.  The re-armed status must pull it back."""
    from linac_gen.assist.testing import turn_text
    panel, _ = _mock_panel(qapp, tmp_path, [turn_text("x")])
    panel._set_state("listening")
    panel._on_wake_status("say 'HELIX'")
    assert panel._orb_state == "idle"
    # ...but a REAL command path (state already thinking) is untouched
    panel._set_state("thinking")
    panel._on_wake_status("say 'HELIX'")
    assert panel._orb_state == "thinking"
    panel.shutdown()


def test_barge_mutes_rest_of_turn(qapp, tmp_path):
    """After a barge-in, later streamed sentences must NOT resume
    speaking over the human."""
    from linac_gen.assist.testing import turn_text
    panel, _ = _mock_panel(qapp, tmp_path, [turn_text("x")])
    spoken = []

    class _Sp:
        backend_name = "fake"

        def say(self, t):
            spoken.append(t)

        def stop(self):
            pass

        def busy(self):
            return False

    panel._speaker = _Sp()
    panel._speak.setChecked(True)
    panel._on_barge()                           # user talked over us
    panel._speak_sentence("This must stay unspoken.")
    panel._maybe_speak("So must this.")
    assert spoken == []
    panel._begin_turn_ui()                      # next turn re-arms speech
    assert panel._barged_this_turn is False
    panel._turn_watchdog.stop()
    panel.shutdown()


def test_session_replacement_resolves_stuck_approver(qapp, tmp_path):
    """Connecting a new backend while a confirmation blocks the OLD
    session's worker must resolve it (ABORT) — the executor thread used
    to wait forever and block SDK shutdown at quit."""
    from linac_gen.assist.agent import Decision
    from linac_gen.assist.config import AssistConfig
    from linac_gen.assist.testing import MockProvider, turn_text
    panel, _ = _mock_panel(qapp, tmp_path, [turn_text("x")])
    ap = panel._approver
    ap._gen = 1
    ap._pending = object()
    cfg = AssistConfig(provider="openai", model="mock",
                       base_url="http://blocked/v1", api_key="")
    panel._start_session(cfg, provider=MockProvider([]))    # replace
    assert ap._event.is_set()
    assert ap._decision is Decision.ABORT
    panel.shutdown()


def test_close_silences_speech_and_followup(qapp, tmp_path):
    """User report: '"I closed it but still can hear it" — dismissing
    the window must cut speech, mute the rest of the turn, and cancel
    any open follow-up mic window (while the session stays alive)."""
    from linac_gen.assist.testing import turn_text
    panel, _ = _mock_panel(qapp, tmp_path, [turn_text("x")])
    stopped = []

    class _Sp:
        backend_name = "fake"

        def stop(self):
            stopped.append(1)

        def busy(self):
            return False

    cancelled = []

    class _Fl:
        active = False

        def cancel(self):
            cancelled.append(1)

    panel._speaker = _Sp()
    panel._followup = _Fl()
    panel.show()
    qapp.processEvents()
    panel.close()                                # ✕ while speaking
    qapp.processEvents()
    assert not panel.isVisible()
    assert stopped == [1]                        # speech cut NOW
    assert cancelled == [1]                      # mic window closed
    assert panel._barged_this_turn is True       # rest of turn muted
    assert panel._session is not None            # still alive (hidden)
    panel.shutdown()


def test_dismissed_panel_never_reopens_followup(qapp, tmp_path):
    """✕-dismissed → no hot mic from the drain path; reopening (or the
    wake popup) clears the dismissal.  A merely NOT-YET-SHOWN warm
    panel is unaffected — its voice turns keep the follow-up flow."""
    from linac_gen.assist.testing import turn_text
    panel, _ = _mock_panel(qapp, tmp_path, [turn_text("x")])
    opened = []

    class _Fl:
        active = False

        def open_window(self):
            opened.append(1)
            return True

        def cancel(self):
            pass

    panel._followup = _Fl()
    panel._last_turn_was_voice = True
    panel._open_followup_if_voice()              # never-shown warm panel:
    assert opened == [1]                         # follow-up still works
    panel.show()
    qapp.processEvents()
    panel.close()                                # ✕ = dismissed
    qapp.processEvents()
    panel._last_turn_was_voice = True
    panel._open_followup_if_voice()              # drain path, dismissed
    assert opened == [1]                         # NO new hot mic
    panel.show()                                 # re-engaged
    qapp.processEvents()
    panel._open_followup_if_voice()
    assert opened == [1, 1]                      # normal again
    panel.shutdown()


def test_own_echo_is_filtered_not_answered(qapp, tmp_path):
    """User report ×2: 'Helix is listening to herself and responding in
    a loop'.  Whatever audio leaks into a capture window, a transcript
    that is mostly the assistant's OWN recent words must be dropped —
    never start a turn."""
    from linac_gen.assist.testing import turn_text
    panel, _ = _mock_panel(qapp, tmp_path, [turn_text("x")])
    panel._note_spoken("sigma x at the exit is about 0.62 millimeters "
                       "and transmission stays near one hundred percent")
    # her own words come back through the mic → filtered, no turn
    panel._on_voice_text("sigma x at the exit is about zero point six "
                         "two millimeters")
    assert "ignored own echo" in panel._transcript.toPlainText()
    assert panel._worker is None                 # no turn started
    # a real user command with fresh words passes through
    panel._on_voice_text("open the results tab please")
    _pump_worker(qapp, panel)
    assert "open the results tab" in panel._transcript.toPlainText()
    panel.shutdown()


def test_own_echo_of_confirm_prompt_cannot_self_resolve(qapp, tmp_path):
    """The spoken 'Approval needed… say confirm, or cancel' echo
    contains BOTH yes- and no-words — it used to self-DENY.  The echo
    filter must drop it before confirm interpretation."""
    from linac_gen.assist.testing import turn_text
    panel, _ = _mock_panel(qapp, tmp_path, [turn_text("x")])
    ap = panel._approver
    ap._gen = 1
    ap._pending = object()
    panel._confirm_gen = 1
    panel._note_spoken("Approval needed: run_mp. Say confirm, or cancel.")
    panel._on_voice_text("approval needed run mp say confirm or cancel")
    assert not ap._event.is_set()                # NOT resolved by echo
    # the real human's short answer still resolves (too short to judge
    # as echo — deliberately passes the filter)
    panel._on_voice_text("yes go ahead")
    assert ap._event.is_set()
    panel.shutdown()


def _pump(qapp, s=0.4):
    t0 = time.time()
    while time.time() - t0 < s:
        qapp.processEvents()
        time.sleep(0.005)


def _content_bottom_indep(v) -> int:
    """Independent oracle: walk the layout ourselves — never call
    the implementation's own helper (a circular oracle silently
    redefines the test whenever the implementation changes)."""
    b = 0
    for i in range(v._col.count()):
        w = v._col.itemAt(i).widget()
        if w is not None and not w.isHidden():
            b = max(b, w.y() + w.height())
    return b + v._col.contentsMargins().bottom()


def _blank_px(v) -> int:
    """EMPTY viewport space below real content."""
    bar = v.verticalScrollBar()
    return max(0, (bar.value() + v.viewport().height())
               - _content_bottom_indep(v))


def _deficit_px(v) -> int:
    """Content BELOW the fold while pinned — the rounds-1/2
    symptom.  A one-sided blank check passes with autoscroll
    deleted entirely (adversarial review, executed)."""
    bar = v.verticalScrollBar()
    return max(0, _content_bottom_indep(v)
               - (bar.value() + v.viewport().height()))


def _assert_at_bottom(v, limit=24, where=""):
    b, d = _blank_px(v), _deficit_px(v)
    assert b <= limit, f"blank={b}px {where}"
    assert d <= limit, f"deficit={d}px {where}"


def test_transcript_never_shows_blank_space(qapp):
    """The measured bug: phantom height (+77 px/turn) accumulated in
    the trailing stretch until the pinned viewport was 100% empty after
    ~8 turns.  BLANK must stay tiny through every lifecycle event."""
    from linac_gen_gui.interphase.dialogs.assistant_chat import ChatView
    from PyQt6.QtWidgets import QWidget, QVBoxLayout
    holder = QWidget()
    holder.resize(560, 620)
    lay = QVBoxLayout(holder)
    v = ChatView(holder)
    lay.addWidget(v)
    holder.show()
    _pump(qapp, 0.2)
    LIMIT = 24
    # 15 conversational turns (the measured failure was total at ~8)
    for i in range(15):
        v.add_message("user", f"question number {i} about the beam "
                              "envelope and the matching section")
        v.add_message("assistant",
                      f"Answer {i}: sigma x at the exit is about 0.62 "
                      "millimeters and transmission stays near one "
                      "hundred percent.\n- first point\n- second point")
        v.add_message("tool", f"· run_envelope … #{i}")
    _pump(qapp)
    _assert_at_bottom(v, LIMIT, "after turns")
    # mid-stream
    for i in range(40):
        v.stream_delta(f"streamed clause {i}, with enough words to "
                       "wrap the label at this width.\n")
    _pump(qapp)
    _assert_at_bottom(v, LIMIT, "mid-stream")
    # a code block (wide <pre> used to re-inflate the phantom)
    v.stream_delta("```python\n" + "x = " + "1 + " * 60 + "1\n```\n")
    _pump(qapp)
    v.end_stream()
    _pump(qapp)
    _assert_at_bottom(v, LIMIT, "after end")
    # font rebuild
    v.set_text_size(22)
    _pump(qapp)
    _assert_at_bottom(v, LIMIT, "after A+")
    # hidden (voice-only) appends, then re-shown
    v.hide()
    for i in range(4):
        v.add_message("assistant", f"hidden append {i} with some text")
    v.show()
    _pump(qapp)
    v._scroll_bottom()
    _pump(qapp, 0.2)
    _assert_at_bottom(v, LIMIT, "after reshow")
    holder.close()


def test_rebuild_preserves_inflight_stream(qapp):
    """A+/A− mid-reply used to silently destroy the streamed prefix."""
    from linac_gen_gui.interphase.dialogs.assistant_chat import ChatView
    v = ChatView()
    v.resize(400, 300)
    v.show()
    _pump(qapp, 0.1)
    v.stream_delta("The first half of the reply. ")
    _pump(qapp, 0.1)
    v.set_text_size(24)                      # rebuild mid-stream
    _pump(qapp, 0.1)
    v.stream_delta("And the second half arrives after the rebuild.")
    _pump(qapp, 0.1)
    raw = v.end_stream()
    assert "first half" in raw and "second half" in raw
    assert "first half" in v.toPlainText()
    v.close()


def test_user_scroll_up_unpins_until_back_at_bottom(qapp):
    """Reading history during a stream must not be a fight: a user
    gesture unpins; returning to the bottom (or a new-message jump)
    re-pins."""
    from linac_gen_gui.interphase.dialogs.assistant_chat import ChatView
    v = ChatView()
    v.resize(400, 260)
    v.show()
    _pump(qapp, 0.1)
    for i in range(30):
        v.add_message("assistant", f"history line {i} with some words")
    _pump(qapp)
    assert v._pinned
    bar = v.verticalScrollBar()
    bar.setValue(0)
    bar.sliderMoved.emit(0)                  # the user drags up
    _pump(qapp, 0.2)
    assert not v._pinned
    for i in range(10):
        v.stream_delta(f"more streamed text {i}\n")
    _pump(qapp, 0.4)
    assert bar.value() < 40                  # view did NOT yank down
    v._scroll_bottom()                       # explicit jump re-pins
    _pump(qapp, 0.2)
    assert v._pinned
    assert _blank_px(v) <= 24
    v.end_stream()
    v.close()


def test_orb_reacts_to_voice_in_every_hearing_state(qapp):
    """User request: 'whenever I say something the orb should react so
    I know it is listening' — the fast-attack envelope must visibly
    kick the ring in listening AND idle the moment a level arrives."""
    from linac_gen_gui.interphase.dialogs.assistant_orb import AssistantOrb
    orb = AssistantOrb()
    orb._timer.stop()
    for state in ("listening", "idle"):
        orb.set_state(state)
        orb._env = 0.0
        orb._bars = [0.05] * len(orb._bars)
        for _ in range(10):
            orb._tick()
        flat = max(orb._bars)
        orb.set_level(0.12)                     # one spoken word
        assert orb._env > 0.5                   # instant attack
        for _ in range(6):
            orb._tick()
        assert max(orb._bars) > flat + 0.15     # the ring visibly dances
    # decay: silence lets the ring settle again
    for _ in range(60):
        orb._tick()
    assert orb._env < 0.05
    orb.stop()


def test_orb_is_larger_and_paints_all_states(qapp):
    from PyQt6.QtGui import QPainter, QPixmap
    from linac_gen_gui.interphase.dialogs.assistant_orb import AssistantOrb
    orb = AssistantOrb()
    assert orb.minimumHeight() >= 200           # "make the orb larger"
    orb.resize(320, 260)
    for st in ("idle", "thinking", "responding", "listening",
               "awaiting-confirm", "error", "starting", "off"):
        orb.set_state(st)
        orb.set_level(0.4)
        orb._tick()
        pm = QPixmap(320, 260)
        p = QPainter(pm)
        orb.render(p)                           # all paint paths incl. 3-D
        p.end()
    orb.stop()


def test_text_toggle_hides_transcript_and_persists(qapp, tmp_path):
    from linac_gen.assist.testing import turn_text
    panel, _ = _mock_panel(qapp, tmp_path, [turn_text("x")])
    try:
        assert panel._transcript.isVisibleTo(panel)   # default: shown
        panel._text_chk.setChecked(False)             # voice-only view
        qapp.processEvents()
        assert not panel._transcript.isVisibleTo(panel)
        assert str(panel._settings().value("assist/show_text")) == "0"
        # the ORB inherits the stretch — it fills the freed window space
        if panel._orb is not None:
            i_orb = panel._vbox.indexOf(panel._orb)
            i_txt = panel._vbox.indexOf(panel._transcript)
            assert panel._vbox.stretch(i_orb) == 1
            assert panel._vbox.stretch(i_txt) == 0
        # conversation keeps recording underneath
        panel._append("still recorded")
        assert "still recorded" in panel._transcript.toPlainText()
        panel._text_chk.setChecked(True)
        qapp.processEvents()
        assert panel._transcript.isVisibleTo(panel)
        assert str(panel._settings().value("assist/show_text")) == "1"
    finally:
        panel._settings().setValue("assist/show_text", "1")
        panel.shutdown()


def test_wake_ack_line_lands_in_transcript(qapp, tmp_path):
    from linac_gen.assist.testing import turn_text
    panel, _ = _mock_panel(qapp, tmp_path, [turn_text("x")])
    panel._on_wake_word()
    for _ in range(50):
        qapp.processEvents()
    assert "heard 'HELIX'" in panel._transcript.toPlainText()
    panel.shutdown()
