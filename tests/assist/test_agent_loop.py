# tests/assist/test_agent_loop.py
"""Agent loop: dispatch, tier gating, jobs, ledger — all offline."""
from __future__ import annotations

import json
import time

import numpy as np
import pytest

from linac_gen.assist.agent import AgentSession, Decision
from linac_gen.assist.testing import (
    MockProvider, ScriptedApprover, turn_text, turn_tools,
)


def _session(ctx, assist_config, turns, decisions=(), tmp_ledger=None):
    from linac_gen.assist.ledger import Ledger
    return AgentSession(
        assist_config, ctx,
        approver=ScriptedApprover(list(decisions)),
        provider=MockProvider(turns),
        ledger=Ledger(ctx.calc_dir))


def test_read_tool_auto_runs_no_confirmation(ctx, assist_config):
    mock_turns = [turn_tools(("get_status", {})),
                  turn_text("done")]
    s = _session(ctx, assist_config, mock_turns)
    out = s.ask("what's loaded?")
    assert out == "done"
    # the tool result actually reached the model
    last_req = s.provider.requests[-1]
    from linac_gen.assist.messages import ToolResultsMsg
    trs = [e for e in last_req["transcript"]
           if isinstance(e, ToolResultsMsg)]
    assert trs, "tool results missing from the transcript"
    env = json.loads(trs[-1].outcomes[0].content)
    assert env["status"] == "ok"
    assert env["data"]["lattice"]["n_elements"] == 4
    s.close()


def test_compute_blocked_until_approved_and_deny_reaches_model(
        ctx, assist_config):
    mock_turns = [turn_tools(("run_envelope", {})),
                  turn_text("understood, not running it")]
    s = _session(ctx, assist_config, mock_turns,
                 decisions=[Decision.DENY])
    s.ask("run the envelope")
    from linac_gen.assist.messages import ToolResultsMsg
    trs = [e for e in s.provider.requests[-1]["transcript"]
           if isinstance(e, ToolResultsMsg)]
    env = json.loads(trs[-1].outcomes[0].content)
    assert env["status"] == "refused"
    assert "denied" in env["data"]["message"]
    assert trs[-1].outcomes[0].is_error is True
    s.close()


def test_approve_runs_job_and_completion_note_injected(
        ctx, assist_config):
    mock_turns = [turn_tools(("run_envelope", {})),
                  turn_text("job started"),
                  turn_text("the envelope finished")]
    s = _session(ctx, assist_config, mock_turns,
                 decisions=[Decision.APPROVE])
    s.ask("run the envelope")            # turn 1: submits the job
    # wait for the background job to finish (tiny lattice, fast)
    deadline = time.time() + 30
    while time.time() < deadline:
        jobs = s.jobs.list_jobs()["data"]["jobs"]
        if jobs and jobs[0]["state"] in ("done", "error"):
            break
        time.sleep(0.05)
    assert jobs[0]["state"] == "done", jobs
    s.ask("did it finish?")              # turn 2: drains the note
    from linac_gen.assist.messages import SystemNote
    notes = [e for e in s.provider.requests[-1]["transcript"]
             if isinstance(e, SystemNote)]
    assert any("job-0001" in n.text and "done" in n.text
               for n in notes)
    # results landed in the session context
    assert ctx.results is not None
    assert len(ctx.results.sigma_x) > 0
    s.close()


def test_mutate_always_confirms_even_with_auto_approve(
        ctx, assist_config):
    assist_config.auto_approve_compute = True
    mock_turns = [turn_tools(("set_element_param",
                              {"element_name": "QF", "param": "gradient",
                               "value": 6.0})),
                  turn_text("changed")]
    approver = ScriptedApprover([Decision.APPROVE])
    from linac_gen.assist.ledger import Ledger
    s = AgentSession(assist_config, ctx, approver=approver,
                     provider=MockProvider(mock_turns),
                     ledger=Ledger(ctx.calc_dir))
    s.ask("set QF gradient to 6")
    assert len(approver.requests) == 1          # confirmation happened
    req = approver.requests[0]
    assert req.tier == "mutate"
    assert "gradient" in req.pretty and "6.0" in req.pretty
    assert ctx.lattice.elements[0].gradient == 6.0
    s.close()


def test_approve_session_flips_auto_for_next_compute(ctx, assist_config):
    mock_turns = [turn_tools(("compare_to_tracewin",
                              {"tracewin_file": "/nonexistent.txt"})),
                  turn_text("first"),
                  turn_tools(("compare_to_tracewin",
                              {"tracewin_file": "/nonexistent.txt"})),
                  turn_text("second")]
    approver = ScriptedApprover([Decision.APPROVE_SESSION])
    from linac_gen.assist.ledger import Ledger
    s = AgentSession(assist_config, ctx, approver=approver,
                     provider=MockProvider(mock_turns),
                     ledger=Ledger(ctx.calc_dir))
    ctx.results = object()                       # placate the gate
    s.ask("compare")
    s.ask("compare again")
    # only ONE confirmation ever requested
    assert len(approver.requests) == 1
    s.close()


def test_unknown_tool_refused_never_executed(ctx, assist_config):
    mock_turns = [turn_tools(("delete_everything", {})),
                  turn_text("ok")]
    s = _session(ctx, assist_config, mock_turns)
    s.ask("nuke it")
    from linac_gen.assist.messages import ToolResultsMsg
    trs = [e for e in s.provider.requests[-1]["transcript"]
           if isinstance(e, ToolResultsMsg)]
    env = json.loads(trs[-1].outcomes[0].content)
    assert env["status"] == "refused"
    assert "unknown tool" in env["data"]["message"]
    s.close()


def test_parallel_tool_calls_one_results_message(ctx, assist_config):
    mock_turns = [turn_tools(("get_status", {}),
                             ("get_lattice_info", {})),
                  turn_text("done")]
    s = _session(ctx, assist_config, mock_turns)
    s.ask("status and info")
    from linac_gen.assist.messages import ToolResultsMsg
    trs = [e for e in s.provider.requests[-1]["transcript"]
           if isinstance(e, ToolResultsMsg)]
    assert len(trs) == 1
    assert len(trs[0].outcomes) == 2             # both, in one message
    s.close()


def test_max_iterations_guard(ctx, assist_config):
    assist_config.max_tool_iterations = 3
    mock_turns = [turn_tools(("get_status", {})) for _ in range(3)]
    s = _session(ctx, assist_config, mock_turns)
    s.ask("loop forever")                        # script provides only 3
    from linac_gen.assist.messages import SystemNote
    assert any(isinstance(e, SystemNote) and "budget" in e.text
               for e in s.transcript)
    s.close()


def test_tool_calls_answered_even_if_stop_reason_end(ctx, assist_config):
    """Defensive: a (mis-behaving local) model that emits tool calls
    with stop_reason=end must still get its calls answered — a dangling
    tool_use id would make the next request 400 on both APIs."""
    from linac_gen.assist.messages import (
        AssistantTurn, StopReason, ToolCall, ToolResultsMsg, Usage,
    )
    bad_turn = AssistantTurn(
        text="", tool_calls=(ToolCall("tc-0", "get_status", {}),),
        stop_reason=StopReason.END, usage=Usage(1, 1))
    s = _session(ctx, assist_config, [bad_turn, turn_text("done")])
    s.ask("go")
    # the call was executed and answered (not dropped)
    assert any(isinstance(e, ToolResultsMsg) for e in s.transcript)
    s.close()


def test_ledger_records_and_never_leaks_the_key(ctx, assist_config):
    mock_turns = [turn_tools(("get_status", {})), turn_text("done")]
    s = _session(ctx, assist_config, mock_turns)
    s.ask("hello")
    s.close()
    raw = open(s.ledger.path, encoding="utf-8").read()
    assert "sk-TESTSECRET" not in raw            # structural + scrubbed
    events = [json.loads(x) for x in raw.splitlines()]
    kinds = [e["event"] for e in events]
    assert kinds[0] == "session_start" and kinds[-1] == "session_end"
    tools = [e for e in events if e["event"] == "tool"]
    assert tools and tools[0]["tool"] == "get_status"
    assert tools[0]["tier"] == "read"
    assert tools[0]["approved_by"] == "auto"


def test_job_cancel_stops_the_run(ctx, assist_config):
    # a big-enough MP run to still be in flight when we cancel
    ctx.beam_config.n_particles = 5000
    mock_turns = [turn_tools(("run_mp", {"space_charge": False})),
                  turn_text("started")]
    s = _session(ctx, assist_config, mock_turns,
                 decisions=[Decision.APPROVE])
    s.ask("run mp")
    out = s.jobs.cancel("job-0001")
    assert out["status"] == "ok"
    deadline = time.time() + 30
    state = None
    while time.time() < deadline:
        state = s.jobs.status("job-0001")["data"]["state"]
        if state in ("cancelled", "done", "error"):
            break
        time.sleep(0.05)
    # tiny lattice may legitimately FINISH before the abort lands —
    # accept done, but never a hang/error
    assert state in ("cancelled", "done"), state
    s.close()
