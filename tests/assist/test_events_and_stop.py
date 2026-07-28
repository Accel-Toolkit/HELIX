"""Phase-1 MIRAGE-parity plumbing: reusable Stop, the machine-event
channel, progress events, and the hardened system prompt."""
from __future__ import annotations

import pytest

from linac_gen.assist.agent import AgentSession, Decision, LONG_RUNNING
from linac_gen.assist.messages import SystemNote
from linac_gen.assist.testing import MockProvider, ScriptedApprover, turn_text


def _session(ctx, assist_config, turns, decisions=(), events=None):
    provider = MockProvider(turns)
    approver = ScriptedApprover(list(decisions))
    sess = AgentSession(assist_config, ctx, approver=approver,
                        provider=provider,
                        on_event=(events.append if events is not None
                                  else None))
    return sess, provider, approver


# ---------------------------------------------------------------------------
# Stop is reusable
# ---------------------------------------------------------------------------
def test_stop_does_not_kill_the_next_turn(ctx, assist_config):
    sess, provider, _ = _session(
        ctx, assist_config, [turn_text("first"), turn_text("second")])
    assert sess.ask("hi") == "first"
    sess.request_stop()                    # user hits Stop between turns
    # the next turn must re-arm and complete normally
    assert sess.ask("again") == "second"
    sess.close()


def test_abort_stays_set_after_close(ctx, assist_config):
    sess, provider, _ = _session(ctx, assist_config, [turn_text("only")])
    sess.close()
    # closed session: ask() must NOT clear the abort flag
    assert sess._abort.is_set()
    sess.ask("late")                       # returns without provider crash?
    # provider script still has its one turn queued only if ask aborted
    # BEFORE the send — the abort flag short-circuits the loop
    assert provider.requests == [] or provider._turns == []


def test_request_stop_interrupts_sdk_backend(ctx, assist_config):
    sess, _, _ = _session(ctx, assist_config, [turn_text("x")])

    class _FakeSdk:
        def __init__(self):
            self.interrupted = 0

        def interrupt(self):
            self.interrupted += 1

        def close(self):
            pass

    sess._sdk = _FakeSdk()
    sess.request_stop()
    assert sess._sdk.interrupted == 1
    sess._sdk = None
    sess.close()


# ---------------------------------------------------------------------------
# Machine-event channel
# ---------------------------------------------------------------------------
def test_submit_event_renders_immediately_and_reaches_the_model(
        ctx, assist_config):
    events = []
    sess, provider, _ = _session(ctx, assist_config,
                                 [turn_text("noted")], events=events)
    sess.submit_event("transmission dropped to 95 %")
    assert any(e.get("type") == "event"
               and "transmission" in e.get("text", "") for e in events)
    sess.ask("what happened?")
    notes = [m for m in provider.requests[0]["transcript"]
             if isinstance(m, SystemNote)]
    assert any("[machine event] transmission dropped" in n.text
               for n in notes)
    sess.close()


def test_ask_event_narrates_pending_events(ctx, assist_config):
    sess, provider, _ = _session(ctx, assist_config,
                                 [turn_text("on it")])
    sess.submit_event("run finished")
    out = sess.ask_event()
    assert out == "on it"
    notes = [m for m in provider.requests[0]["transcript"]
             if isinstance(m, SystemNote)]
    assert any("run finished" in n.text for n in notes)
    # the synthetic user line is marked as machine-generated
    assert "[machine event]" in provider.requests[0][
        "transcript"][-1].text
    sess.close()


def test_job_completion_emits_event_type(ctx, assist_config, monkeypatch):
    """Job completions now ride the event channel (was job_done_note)."""
    from linac_gen.assist import agent as agent_mod
    from linac_gen.assist.tools import Tool

    from linac_gen.assist.testing import turn_tools
    events = []
    sess, provider, _ = _session(
        ctx, assist_config,
        # ask #1: tool round + its follow-up turn; ask_event: one turn
        [turn_tools(("stub_long", {})), turn_text("submitted"),
         turn_text("done")],
        decisions=[Decision.APPROVE], events=events)

    def _stub(context, progress_callback=None, should_abort=None):
        return {"status": "ok", "data": {"ok": True},
                "provenance": {}, "warnings": []}

    sess.tools = dict(sess.tools)
    sess.tools["stub_long"] = Tool(name="stub_long", description="stub",
                                   schema={"type": "object",
                                           "properties": {}},
                                   tier="compute", fn=_stub)
    monkeypatch.setattr(agent_mod, "LONG_RUNNING",
                        LONG_RUNNING | {"stub_long"})
    sess.ask("run it")
    # wait for the pool thread to finish, then a second turn drains it
    import time
    for _ in range(100):
        if sess.jobs._unreported:
            break
        time.sleep(0.02)
    sess.ask_event()
    assert any(e.get("type") == "event" and "finished" in e.get("text", "")
               for e in events)
    assert not any(e.get("type") == "job_done_note" for e in events)
    sess.close()


# ---------------------------------------------------------------------------
# Progress events
# ---------------------------------------------------------------------------
def test_long_running_progress_is_throttled(ctx, assist_config,
                                            monkeypatch):
    from linac_gen.assist import agent as agent_mod
    from linac_gen.assist.testing import turn_tools
    from linac_gen.assist.tools import Tool

    events = []
    sess, provider, _ = _session(
        ctx, assist_config,
        [turn_tools(("stub_long", {})), turn_text("ok")],
        decisions=[Decision.APPROVE], events=events)

    def _stub(context, progress_callback=None, should_abort=None):
        for i in range(5):                 # rapid-fire: within 0.25 s
            progress_callback(1000.0 * i, i, 5)
        return {"status": "ok", "data": {}, "provenance": {},
                "warnings": []}

    sess.tools = dict(sess.tools)
    sess.tools["stub_long"] = Tool(name="stub_long", description="stub",
                                   schema={"type": "object",
                                           "properties": {}},
                                   tier="compute", fn=_stub)
    monkeypatch.setattr(agent_mod, "LONG_RUNNING",
                        LONG_RUNNING | {"stub_long"})
    sess.ask("go")
    import time
    time.sleep(0.3)                        # let the pool thread run
    prog = [e for e in events if e.get("type") == "progress"]
    assert len(prog) == 1                  # 5 rapid calls -> 1 emit
    assert "stub_long" in prog[0]["text"]
    sess.close()


# ---------------------------------------------------------------------------
# Prompt hardening
# ---------------------------------------------------------------------------
def test_system_prompt_carries_security_block_and_roster(ctx):
    from linac_gen.assist.prompts import build_system_prompt, tool_roster
    p = build_system_prompt(ctx)
    flat = " ".join(p.split())             # collapse the wrapping
    assert "SECURITY (non-negotiable)" in p
    assert "strictly as DATA" in flat
    assert "Never claim an action happened" in flat
    assert "[machine event]" in p
    r = tool_roster()
    assert "YOUR TOOLS" in r
    assert "- run_envelope [compute]:" in r
    assert "- get_status [read]:" in r
    assert r in p


def test_prompt_has_spoken_style_rules():
    """MIRAGE parity: replies may be spoken aloud — the prompt must pin
    the heard-not-read rules (~3 sig figs, one value per clause, plain
    element names)."""
    from linac_gen.assist.prompts import build_system_prompt
    p = build_system_prompt()
    assert "SPOKEN ALOUD" in p
    assert "3 significant figures" in p
    assert "one value per short clause" in p


def test_run_fast_executes_read_tools_only(tmp_path):
    """The intent fast-path seam: read tools run + ledger a fast_path
    entry + queue a quiet model note; compute/mutate are refused."""
    import json
    from linac_gen.assist.agent import AgentSession
    from linac_gen.assist.config import AssistConfig
    from linac_gen.assist.ledger import Ledger
    from linac_gen.assist.testing import MockProvider
    from linac_gen.assist.tools import WorkContext
    from linac_gen.assist.testing import ScriptedApprover
    sess = AgentSession(
        AssistConfig(provider="openai", model="mock",
                     base_url="http://blocked/v1", api_key=""),
        WorkContext(), approver=ScriptedApprover([]),
        provider=MockProvider([]), ledger=Ledger(str(tmp_path)))
    try:
        res = sess.run_fast("get_status", {})
        assert res is not None and res["status"] == "ok"
        assert sess.run_fast("run_envelope", {}) is None     # compute
        assert sess.run_fast("set_element_param", {}) is None  # mutate
        assert sess.run_fast("no_such_tool", {}) is None
        recs = [json.loads(x) for x in
                open(sess.ledger.path, encoding="utf-8")]
        fast = [r for r in recs if r.get("event") == "tool"
                and r.get("approved_by") == "fast_path"]
        assert len(fast) == 1 and fast[0]["tool"] == "get_status"
        # the model hears about it at its next turn
        assert any("instant command" in t
                   for t in list(sess._events.queue))
    finally:
        sess.close()


def test_prompt_gets_live_state_digest(tmp_path):
    from linac_gen.assist.agent import AgentSession
    from linac_gen.assist.config import AssistConfig
    from linac_gen.assist.guide import get_state
    from linac_gen.assist.ledger import Ledger
    from linac_gen.assist.testing import MockProvider, turn_text
    from linac_gen.assist.tools import WorkContext
    ctx = WorkContext()
    get_state(ctx).active = True
    get_state(ctx).idx = 4
    from linac_gen.assist.testing import ScriptedApprover
    sess = AgentSession(
        AssistConfig(provider="openai", model="mock",
                     base_url="http://blocked/v1", api_key=""),
        ctx, approver=ScriptedApprover([]),
        provider=MockProvider([turn_text("ok")]),
        ledger=Ledger(str(tmp_path)))
    try:
        sess.ask("hello")
        assert "guided tour: ACTIVE at station 5" in sess._prompt_extra()
    finally:
        sess.close()
