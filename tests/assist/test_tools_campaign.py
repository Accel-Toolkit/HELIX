"""Wave-2 plan tools: run_campaign (one-confirm numbered plans) and
tuning_plan (success window + bit-exact auto-rollback)."""
from __future__ import annotations

import json

import pytest

from linac_gen.assist.agent import (
    AgentSession, Decision, LONG_RUNNING,
)
from linac_gen.assist.testing import (
    MockProvider, ScriptedApprover, turn_text, turn_tools,
)
from linac_gen.assist.tools import TOOLS, render_call


def _quad(ctx, name="QF"):
    return next(e for e in ctx.lattice.elements
                if getattr(e, "name", "") == name)


def _plan(*steps):
    return json.dumps(list(steps))


# ---------------------------------------------------------------------------
# rendering & registration
# ---------------------------------------------------------------------------
def test_campaign_is_long_running_mutate():
    assert TOOLS["run_campaign"].tier == "mutate"
    assert TOOLS["tuning_plan"].tier == "mutate"
    assert {"run_campaign", "tuning_plan"} <= LONG_RUNNING


def test_render_call_shows_full_numbered_plan():
    plan = _plan({"tool": "set_element_param", "element_name": "QF",
                  "param": "gradient", "value": 5.5},
                 {"tool": "run_envelope"},
                 {"tool": "anomaly_check", "expect": "ok"})
    echo = render_call("run_campaign", {"plan_json": plan})
    assert "1. set QF.gradient = 5.5" in echo
    assert "2. run_envelope" in echo
    assert "3. anomaly check (STOP unless verdict is 'ok')" in echo
    assert "NOT reverted" in echo


def test_render_call_generic_tools_unchanged():
    echo = render_call("set_element_param",
                       {"element_name": "QF", "param": "gradient",
                        "value": 5.0})
    assert echo.startswith("set_element_param  [mutate]")
    assert "element_name = 'QF'" in echo


# ---------------------------------------------------------------------------
# pre-gate validation (never prompts on an invalid plan)
# ---------------------------------------------------------------------------
def test_invalid_plan_refused_without_prompting(ctx, assist_config):
    approver = ScriptedApprover([])          # would raise if consulted
    sess = AgentSession(
        assist_config, ctx, approver=approver,
        provider=MockProvider([
            turn_tools(("run_campaign", {"plan_json": _plan(
                {"tool": "set_element_param", "element_name": "NOPE",
                 "param": "gradient", "value": 1.0})})),
            turn_text("refused then"),
        ]))
    # the gate DOES fire for mutate before the tool body runs — so the
    # validation refusal happens inside the approved call.  To assert
    # never-prompts semantics we call the function directly:
    out = TOOLS["run_campaign"].fn(
        ctx, plan_json=_plan({"tool": "set_element_param",
                              "element_name": "NOPE",
                              "param": "gradient", "value": 1.0}))
    assert out["status"] == "refused"
    assert "not in lattice" in out["data"]["message"]
    sess.close()


@pytest.mark.parametrize("plan,frag", [
    ("[]", "non-empty"),
    (_plan({"tool": "quantum_leap"}), "unknown step tool"),
    (_plan(*[{"tool": "run_envelope"}] * 13), "too long"),
    (_plan({"tool": "set_element_param", "element_name": "QF",
            "param": "no_such", "value": 1.0}), "no parameter"),
    (_plan({"tool": "set_element_param", "element_name": "QF",
            "param": "gradient", "value": "high"}), "must be a number"),
    (_plan({"tool": "anomaly_check", "expect": "ok"}),
     "needs a baseline"),
])
def test_plan_validation_catalogue(ctx, plan, frag):
    out = TOOLS["run_campaign"].fn(ctx, plan_json=plan)
    assert out["status"] == "refused"
    assert frag in out["data"]["message"]


# ---------------------------------------------------------------------------
# execution semantics
# ---------------------------------------------------------------------------
def test_campaign_executes_in_order_and_sets_persist(ctx):
    out = TOOLS["run_campaign"].fn(ctx, plan_json=_plan(
        {"tool": "set_element_param", "element_name": "QF",
         "param": "gradient", "value": 5.5},
        {"tool": "run_envelope"},
        {"tool": "set_element_param", "element_name": "QD",
         "param": "gradient", "value": -5.5}))
    assert out["status"] == "ok"
    d = out["data"]
    assert d["completed"] == 3 and d["of"] == 3
    assert [t["tool"] for t in d["trail"]] == [
        "set_element_param", "run_envelope", "set_element_param"]
    assert _quad(ctx, "QF").gradient == 5.5
    assert _quad(ctx, "QD").gradient == -5.5
    assert ctx.results is not None           # the run really happened


def test_campaign_checkpoint_failure_stops_chain(ctx, tmp_path):
    # baseline on the healthy machine
    TOOLS["run_envelope"].fn(ctx)
    TOOLS["anomaly_baseline"].fn(ctx)
    # detune hard, run, then checkpoint expecting 'ok' — must stop
    out = TOOLS["run_campaign"].fn(ctx, plan_json=_plan(
        {"tool": "set_element_param", "element_name": "QF",
         "param": "gradient", "value": 50.0},
        {"tool": "run_envelope"},
        {"tool": "anomaly_check", "expect": "ok"},
        {"tool": "set_element_param", "element_name": "QD",
         "param": "gradient", "value": -1.0}))
    d = out["data"]
    assert d["failed_step"] == 3
    assert "checkpoint" in d["trail"][2]
    assert d["completed"] == 3               # checkpoint ran, step 4 not
    assert len(d["trail"]) == 3
    assert _quad(ctx, "QF").gradient == 50.0  # SET steps persist
    assert _quad(ctx, "QD").gradient == -5.0  # never reached


def test_campaign_via_agent_single_confirmation(ctx, assist_config):
    plan = _plan({"tool": "set_element_param", "element_name": "QF",
                  "param": "gradient", "value": 4.75})
    approver = ScriptedApprover([Decision.APPROVE])
    sess = AgentSession(
        assist_config, ctx, approver=approver,
        provider=MockProvider([
            turn_tools(("run_campaign", {"plan_json": plan})),
            turn_text("campaign started"),
        ]))
    sess.ask("do the plan")
    # exactly ONE confirmation, whose echo IS the numbered plan
    assert len(approver.requests) == 1
    req = approver.requests[0]
    assert req.tier == "mutate"
    assert "1. set QF.gradient = 4.75" in req.pretty
    # long-running: submitted as a job; wait for it, then verify the set
    import time
    for _ in range(200):
        if sess.jobs._unreported:
            break
        time.sleep(0.02)
    assert _quad(ctx, "QF").gradient == 4.75
    sess.close()


def test_campaign_denied_touches_nothing(ctx, assist_config):
    g0 = _quad(ctx, "QF").gradient
    plan = _plan({"tool": "set_element_param", "element_name": "QF",
                  "param": "gradient", "value": 9.9})
    sess = AgentSession(
        assist_config, ctx, approver=ScriptedApprover([Decision.DENY]),
        provider=MockProvider([
            turn_tools(("run_campaign", {"plan_json": plan})),
            turn_text("ok, not doing it"),
        ]))
    sess.ask("do the plan")
    assert _quad(ctx, "QF").gradient == g0
    sess.close()


# ---------------------------------------------------------------------------
# tuning_plan
# ---------------------------------------------------------------------------
def test_tuning_kept_when_window_met(ctx):
    out = TOOLS["tuning_plan"].fn(
        ctx, knobs_json=json.dumps(
            [{"element_name": "QF", "param": "gradient", "value": 5.2}]),
        objective="sigma_x", success_below=1000.0, mode="envelope")
    assert out["status"] == "ok"
    assert out["data"]["kept"] is True
    assert _quad(ctx, "QF").gradient == 5.2
    assert out["data"]["before"] is not None
    assert out["data"]["after"] > 0.0


def test_tuning_rollback_is_bit_exact(ctx):
    q = _quad(ctx, "QF")
    g0 = q.gradient                          # exact float
    out = TOOLS["tuning_plan"].fn(
        ctx, knobs_json=json.dumps(
            [{"element_name": "QF", "param": "gradient", "value": 5.7},
             {"element_name": "QD", "param": "gradient",
              "value": -5.7}]),
        objective="sigma_x", success_below=1e-9)   # impossible window
    assert out["status"] == "ok"
    assert out["data"]["kept"] is False
    assert "restored bit-exact" in out["data"]["note"]
    assert q.gradient == g0                  # == : bit-exact, not approx
    assert _quad(ctx, "QD").gradient == -5.0


def test_tuning_transmission_objective_in_envelope_mode_explains(ctx):
    out = TOOLS["tuning_plan"].fn(
        ctx, knobs_json=json.dumps(
            [{"element_name": "QF", "param": "gradient", "value": 5.2}]),
        objective="transmission", success_above=50.0, mode="envelope")
    assert out["status"] == "error"
    assert "mode='mp'" in out["data"]["message"]
    assert _quad(ctx, "QF").gradient == 5.0  # nothing was left applied


@pytest.mark.parametrize("kwargs,frag", [
    (dict(knobs_json="[]", objective="transmission",
          success_above=1.0), "1-8 knobs"),
    (dict(knobs_json=json.dumps([{"element_name": "QF",
                                  "param": "gradient", "value": 5.0}]),
          objective="charisma", success_above=1.0), "objective"),
    (dict(knobs_json=json.dumps([{"element_name": "QF",
                                  "param": "gradient", "value": 5.0}]),
          objective="transmission"), "success window"),
    (dict(knobs_json=json.dumps([{"element_name": "GHOST",
                                  "param": "gradient", "value": 5.0}]),
          objective="transmission", success_above=1.0),
     "not in lattice"),
])
def test_tuning_validation(ctx, kwargs, frag):
    out = TOOLS["tuning_plan"].fn(ctx, **kwargs)
    assert out["status"] == "refused"
    assert frag in out["data"]["message"]


def test_tuning_render_echo():
    echo = render_call("tuning_plan", {
        "knobs_json": json.dumps(
            [{"element_name": "QF", "param": "gradient", "value": 5.5}]),
        "objective": "sigma_y", "success_below": 2.0, "mode": "envelope"})
    assert "set QF.gradient = 5.5" in echo
    assert "sigma_y < 2.0" in echo
    assert "AUTO-ROLLBACK" in echo
