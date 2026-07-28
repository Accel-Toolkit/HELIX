"""Training tools (wave 3): the guided tour and instructor drills.

Tier split is deliberate: ``instructor_start`` MUTATES the lattice
(hidden fault) so it always confirms; ``guide`` and ``instructor``
(hint/status/answer/give_up/debrief) are read-tier — HELIX gates per
tool, and a confirmation on every hint would wreck the drill flow.
The answer/give_up/debrief paths restore the lattice rather than
mutating it further.
"""
from __future__ import annotations

from linac_gen.assist.tools import _ctx_provenance, _err, _ok, _tool


@_tool("guide",
       "Guided tour of the HELIX interface, one station per turn.  "
       "Relay the returned 'say' text nearly verbatim, then STOP and "
       "wait; drive every next/back/goto through this tool while a "
       "tour is active.  Actions: start | next | back | repeat | "
       "goto (station=) | status | stop.",
       {"type": "object",
        "properties": {"action": {"type": "string"},
                       "station": {"type": "integer"}},
        "required": ["action"]},
       "read")
def _guide(ctx, action: str, station=None):
    from linac_gen.assist.guide import run_action
    out = run_action(ctx, action, station)
    if "error" in out:
        return _err(out["error"])
    return _ok(out, _ctx_provenance(ctx))


@_tool("instructor_start",
       "START a training drill: injects a HIDDEN fault into the loaded "
       "lattice (details withheld from everyone, including the "
       "assistant, until debrief).  The trainee diagnoses it from the "
       "beam observables.  The lattice is restored at answer/give_up/"
       "session close.  Optional seed for reproducible drills.",
       {"type": "object",
        "properties": {"seed": {"type": "integer"}},
        "required": []},
       "mutate")
def _instructor_start(ctx, seed=None):
    from linac_gen.assist.instructor import start
    out = start(ctx, seed=seed)
    if "error" in out:
        return _err(out["error"])
    return _ok(out, _ctx_provenance(ctx))


@_tool("instructor",
       "Drive a running training drill: action = hint (max 3, they "
       "cost points) | status | answer (answer_element=) | give_up | "
       "debrief.  Relay hints verbatim; never speculate about the "
       "fault beyond them — it is hidden from you too.",
       {"type": "object",
        "properties": {"action": {"type": "string"},
                       "answer_element": {"type": "string"}},
        "required": ["action"]},
       "read")
def _instructor(ctx, action: str, answer_element=None):
    from linac_gen.assist.instructor import action as run
    out = run(ctx, action, answer_element=answer_element)
    if "error" in out:
        return _err(out["error"])
    return _ok(out, _ctx_provenance(ctx))
