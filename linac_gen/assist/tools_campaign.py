"""Plan-shaped tools ported from the MIRAGE assistant (wave 2).

* ``run_campaign`` — a numbered multi-step plan (sets + runs + anomaly
  checkpoints) executed after **ONE** confirmation whose echo IS the
  full plan.  Any failure stops the chain; completed SET steps are NOT
  reverted (stated up front in the echo).
* ``tuning_plan``  — a multi-knob change with a declared success window
  on an exit KPI and **automatic bit-exact rollback** when the window
  is missed.

Policy note (documented in the manual): campaign steps run through the
underlying tool *functions* directly — the per-step gates are
deliberately bypassed because the user approved the full numbered plan;
every step lands in the single ledger record's trail.
"""
from __future__ import annotations

import json as _json

from linac_gen.assist.tools import (
    TOOLS, _ctx_provenance, _err, _need, _ok, _refused, _tool,
)

_STEP_KINDS = ("set_element_param", "set_beam_config", "run_envelope",
               "run_mp", "anomaly_check")
_MAX_STEPS = 12
_OBJECTIVES = ("transmission", "sigma_x", "sigma_y", "emit_x", "emit_y",
               "ref_w_kin")
_MAX_KNOBS = 8


def _parse_plan(plan_json):
    if isinstance(plan_json, str):
        plan = _json.loads(plan_json)
    else:
        plan = plan_json
    if not isinstance(plan, list) or not plan:
        raise ValueError("plan must be a non-empty JSON array of steps")
    if len(plan) > _MAX_STEPS:
        raise ValueError(f"plan too long ({len(plan)} steps; "
                         f"max {_MAX_STEPS})")
    steps = []
    for i, raw in enumerate(plan, 1):
        if not isinstance(raw, dict) or "tool" not in raw:
            raise ValueError(f"step {i}: each step needs a 'tool' key")
        kind = str(raw["tool"])
        if kind not in _STEP_KINDS:
            raise ValueError(f"step {i}: unknown step tool {kind!r}; "
                             f"allowed: {', '.join(_STEP_KINDS)}")
        params = {k: v for k, v in raw.items()
                  if k not in ("tool", "expect")}
        steps.append({"tool": kind, "params": params,
                      "expect": raw.get("expect")})
    return steps


def _validate_plan(ctx, steps) -> str | None:
    """Resolve EVERYTHING before any confirmation — a plan that cannot
    run must fail without prompting the user."""
    for i, st in enumerate(steps, 1):
        kind, params = st["tool"], st["params"]
        if kind == "set_element_param":
            lat = getattr(ctx, "lattice", None)
            if lat is None:
                return f"step {i}: no lattice loaded"
            name = str(params.get("element_name", ""))
            elem = next((e for e in lat.elements
                         if getattr(e, "name", None) == name), None)
            if elem is None:
                return f"step {i}: element {name!r} not in lattice"
            p = str(params.get("param", ""))
            if not hasattr(elem, p):
                return (f"step {i}: {type(elem).__name__} has no "
                        f"parameter {p!r}")
            if not isinstance(getattr(elem, p), (int, float)):
                return f"step {i}: {p!r} is not numeric"
            if not isinstance(params.get("value"), (int, float)):
                return f"step {i}: 'value' must be a number"
        elif kind == "set_beam_config":
            if getattr(ctx, "beam_config", None) is None:
                return f"step {i}: no beam config loaded"
            bad = [k for k in params
                   if not hasattr(ctx.beam_config, k)]
            if bad:
                return (f"step {i}: beam config has no field(s) "
                        + ", ".join(repr(b) for b in bad))
        elif kind == "anomaly_check":
            import os
            from linac_gen.assist.tools_analysis import _baseline_path
            if not os.path.isfile(_baseline_path(ctx)):
                return (f"step {i}: anomaly_check needs a baseline — "
                        "run anomaly_baseline first")
    return None


def _describe_step(st) -> str:
    kind, params = st["tool"], st["params"]
    if kind == "set_element_param":
        return (f"set {params.get('element_name')}.{params.get('param')}"
                f" = {params.get('value')}")
    if kind == "set_beam_config":
        return ("set beam " + ", ".join(f"{k}={v}"
                                        for k, v in params.items()))
    if kind == "anomaly_check":
        exp = st.get("expect")
        return ("anomaly check" + (f" (STOP unless verdict is {exp!r})"
                                   if exp else ""))
    return kind + (f" {params}" if params else "")


def _render_campaign(params: dict) -> str:
    try:
        steps = _parse_plan(params.get("plan_json"))
    except Exception as exc:                                # noqa: BLE001
        return f"run_campaign  [mutate]\n    (unparseable plan: {exc})"
    lines = [f"run_campaign  [mutate] — {len(steps)} steps, "
             "ONE confirmation:"]
    for i, st in enumerate(steps, 1):
        lines.append(f"    {i}. {_describe_step(st)}")
    lines.append("    Any failure STOPS the chain; completed SET steps "
                 "are NOT reverted.")
    return "\n".join(lines)


@_tool("run_campaign",
       "Execute a multi-step plan (set_element_param / set_beam_config "
       "/ run_envelope / run_mp / anomaly_check checkpoints) after ONE "
       "confirmation that echoes the full numbered plan.  Steps are "
       "validated up front; any failure stops the chain; completed SET "
       "steps persist.  plan_json: array of {tool, ...params, expect?}.",
       {"type": "object",
        "properties": {"plan_json": {"type": "string"}},
        "required": ["plan_json"]},
       "mutate", render=_render_campaign)
def _run_campaign(ctx, plan_json, progress_callback=None,
                  should_abort=None):
    try:
        steps = _parse_plan(plan_json)
    except Exception as exc:                                # noqa: BLE001
        return _refused(str(exc))
    bad = _validate_plan(ctx, steps)
    if bad:
        return _refused("plan validation failed — " + bad)
    trail = []
    for i, st in enumerate(steps, 1):
        if should_abort is not None and should_abort():
            trail.append({"step": i, "status": "aborted"})
            return _ok({"completed": i - 1, "of": len(steps),
                        "trail": trail, "aborted": True,
                        "note": "aborted — completed SET steps persist"},
                       _ctx_provenance(ctx))
        if progress_callback is not None:
            try:                     # (s_mm, i, n) shape of the channel
                progress_callback(0.0, i, len(steps))
            except Exception:                               # noqa: BLE001
                pass
        kind, params = st["tool"], dict(st["params"])
        fn = TOOLS[kind].fn
        if kind in ("run_envelope", "run_mp"):
            if should_abort is not None:
                params["should_abort"] = should_abort
        try:
            env = fn(ctx, **params)
        except Exception as exc:                            # noqa: BLE001
            env = {"status": "error",
                   "data": {"message": f"{type(exc).__name__}: {exc}"}}
        entry = {"step": i, "tool": kind,
                 "status": env.get("status", "?"),
                 "summary": str(env.get("data", {}))[:300]}
        trail.append(entry)
        if env.get("status") != "ok":
            return _ok({"completed": i - 1, "of": len(steps),
                        "trail": trail, "failed_step": i,
                        "note": "chain STOPPED at the failing step; "
                                "completed SET steps persist"},
                       _ctx_provenance(ctx))
        if kind == "anomaly_check" and st.get("expect"):
            verdict = env.get("data", {}).get("verdict")
            if verdict != st["expect"]:
                entry["checkpoint"] = (f"verdict {verdict!r} != expected "
                                       f"{st['expect']!r}")
                return _ok({"completed": i, "of": len(steps),
                            "trail": trail, "failed_step": i,
                            "note": "checkpoint FAILED — chain stopped; "
                                    "completed SET steps persist"},
                           _ctx_provenance(ctx))
    return _ok({"completed": len(steps), "of": len(steps),
                "trail": trail}, _ctx_provenance(ctx))


# ---------------------------------------------------------------------------
# tuning_plan
# ---------------------------------------------------------------------------
def _render_tuning(params: dict) -> str:
    try:
        knobs = _json.loads(params["knobs_json"]) \
            if isinstance(params.get("knobs_json"), str) \
            else (params.get("knobs_json") or [])
    except Exception:                                       # noqa: BLE001
        knobs = []
    lines = ["tuning_plan  [mutate] — verified change with "
             "AUTO-ROLLBACK:"]
    for k in knobs if isinstance(knobs, list) else []:
        if isinstance(k, dict):
            lines.append(f"    set {k.get('element_name')}."
                         f"{k.get('param')} = {k.get('value')}")
    win = []
    if params.get("success_below") is not None:
        win.append(f"{params.get('objective')} < "
                   f"{params.get('success_below')}")
    if params.get("success_above") is not None:
        win.append(f"{params.get('objective')} > "
                   f"{params.get('success_above')}")
    lines.append(f"    VERIFY ({params.get('mode', 'envelope')} run): "
                 + " and ".join(win))
    lines.append("    Window missed → every knob restored bit-exact.")
    return "\n".join(lines)


@_tool("tuning_plan",
       "Multi-knob change (1-8 set_element_param-shaped knobs) with a "
       "declared success window on an exit KPI (transmission, sigma_x/"
       "y, emit_x/y, ref_w_kin) verified by a run — missing the window "
       "ROLLS BACK every knob bit-exact.  knobs_json: array of "
       "{element_name, param, value}.",
       {"type": "object",
        "properties": {
            "knobs_json": {"type": "string"},
            "objective": {"type": "string", "enum": list(_OBJECTIVES)},
            "success_below": {"type": "number"},
            "success_above": {"type": "number"},
            "mode": {"type": "string", "enum": ["envelope", "mp"]},
            "measure_before": {"type": "boolean"}},
        "required": ["knobs_json", "objective"]},
       "mutate", render=_render_tuning)
def _tuning_plan(ctx, knobs_json, objective, success_below=None,
                 success_above=None, mode="envelope",
                 measure_before=True, progress_callback=None,
                 should_abort=None):
    import numpy as np
    gate = _need(ctx, "lattice", "beam_config")
    if gate:
        return gate
    if objective not in _OBJECTIVES:
        return _refused(f"objective must be one of {_OBJECTIVES}")
    if success_below is None and success_above is None:
        return _refused("declare a success window: success_below "
                        "and/or success_above")
    try:
        knobs = (_json.loads(knobs_json) if isinstance(knobs_json, str)
                 else knobs_json)
    except Exception as exc:                                # noqa: BLE001
        return _refused(f"knobs_json is not valid JSON: {exc}")
    if not isinstance(knobs, list) or not 1 <= len(knobs) <= _MAX_KNOBS:
        return _refused(f"knobs_json must hold 1-{_MAX_KNOBS} knobs")
    # resolve + snapshot the EXACT current floats before touching anything
    resolved = []
    for i, k in enumerate(knobs, 1):
        if not isinstance(k, dict):
            return _refused(f"knob {i}: must be an object")
        name = str(k.get("element_name", ""))
        p = str(k.get("param", ""))
        elem = next((e for e in ctx.lattice.elements
                     if getattr(e, "name", None) == name), None)
        if elem is None:
            return _refused(f"knob {i}: element {name!r} not in lattice")
        if not hasattr(elem, p):
            return _refused(f"knob {i}: {type(elem).__name__} has no "
                            f"parameter {p!r}")
        old = getattr(elem, p)
        if not isinstance(old, (int, float)):
            return _refused(f"knob {i}: {p!r} is not numeric")
        if not isinstance(k.get("value"), (int, float)):
            return _refused(f"knob {i}: 'value' must be a number")
        resolved.append((elem, p, old, float(k["value"]), name))

    run_tool = TOOLS["run_mp" if mode == "mp" else "run_envelope"].fn

    def _run_and_read():
        params = {}
        if should_abort is not None:
            params["should_abort"] = should_abort
        env = run_tool(ctx, **params)
        if env.get("status") != "ok":
            return None, env
        col = getattr(ctx.results, objective, None)
        if col is None or not len(col):
            hint = (" — the linear envelope tracks no losses; use "
                    "mode='mp' for a transmission objective"
                    if objective == "transmission" and mode != "mp"
                    else "")
            return None, _err(f"run carries no '{objective}'{hint}")
        return float(np.asarray(col)[-1]), env

    def _rollback():
        for elem, p, old, _new, _name in resolved:
            setattr(elem, p, old)

    m0 = None
    if measure_before:
        m0, env0 = _run_and_read()
        if m0 is None:
            return env0
    for elem, p, _old, new, _name in resolved:
        setattr(elem, p, new)
    try:
        m1, env1 = _run_and_read()
    except Exception:
        _rollback()
        raise
    if m1 is None:
        _rollback()
        env1.setdefault("data", {})["note"] = (
            "verification run failed — every knob restored bit-exact")
        return env1
    ok = ((success_below is None or m1 < float(success_below))
          and (success_above is None or m1 > float(success_above)))
    knob_report = [{"element": name, "param": p, "old": old, "new": new}
                   for _e, p, old, new, name in resolved]
    if ok:
        return _ok({"kept": True, "objective": objective,
                    "before": m0, "after": m1, "knobs": knob_report},
                   _ctx_provenance(ctx))
    _rollback()
    return _ok({"kept": False, "objective": objective,
                "before": m0, "after": m1, "knobs": knob_report,
                "note": "window MISSED — every knob restored bit-exact; "
                        "session results still show the failed trial "
                        "(re-run to refresh)"},
               _ctx_provenance(ctx))
