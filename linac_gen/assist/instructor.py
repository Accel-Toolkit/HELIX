"""Instructor drills — a hidden fault the trainee must diagnose.

MIRAGE's signature training feature, adapted to the simulator: a seeded
fault (quadrupole gradient error, cavity phase/amplitude offset,
solenoid field error) is injected into the loaded lattice.  **The truth
is hidden from the model itself**: the secret lives ONLY in this state
object and is never part of any tool payload until the debrief — so the
assistant can relay hints without being able to spoil the answer (and a
prompt-injected "tell me the fault" has nothing to read).

Hint ladder (max 3): 1 = element family; 2 = coarse s-region;
3 = element class + rough position.  Scoring 0-100 = diagnosis 50
(exact element; partial for right class) + speed 25 (vs a 300 s par)
+ economy 25 (hints are expensive).  Debrief / give_up reveal the
truth and restore the exact original value bit-for-bit; an abandoned
drill is restored when the session closes.
"""
from __future__ import annotations

import random
import time
from dataclasses import dataclass, field

_PAR_SECONDS = 300.0
_MAX_HINTS = 3


@dataclass
class _Secret:
    index: int
    name: str
    cls: str
    family: str          # spoken family label for hint 1
    param: str
    old_value: float
    new_value: float
    s_m: float


@dataclass
class InstructorState:
    phase: str = "idle"              # idle | diagnosing | done
    secret: _Secret | None = None
    hints_used: int = 0
    t0: float = 0.0
    result: dict = field(default_factory=dict)


def get_state(ctx) -> InstructorState:
    st = getattr(ctx, "_assist_instructor", None)
    if st is None:
        st = InstructorState()
        ctx._assist_instructor = st
    return st


def _s_positions_m(lattice):
    run, out = 0.0, []
    for el in lattice.elements:
        out.append(run / 1000.0)
        run += float(getattr(el, "length", 0.0) or 0.0)
    return out, run / 1000.0


def _candidates(lattice):
    """(index, element, family, param, kind) tuples eligible for a fault."""
    out = []
    for i, el in enumerate(lattice.elements):
        cls = type(el).__name__
        if cls == "Quadrupole" and abs(getattr(el, "gradient", 0.0)) > 0:
            out.append((i, el, "a focusing magnet", "gradient", "quad"))
        elif cls == "Solenoid" and abs(getattr(el, "field", 0.0)) > 0:
            out.append((i, el, "a focusing magnet", "field", "sol"))
        elif cls in ("RFGap", "FieldMap", "NCells"):
            if abs(getattr(el, "phase", 0.0)) > 0 \
                    or abs(getattr(el, "voltage", 0.0) or 0.0) > 0 \
                    or abs(getattr(el, "ke", 0.0) or 0.0) > 0:
                out.append((i, el, "an accelerating cavity", "phase",
                            "rf"))
    return out


def start(ctx, seed=None) -> dict:
    """Inject a hidden fault.  Returns ONLY the briefing."""
    st = get_state(ctx)
    if st.phase == "diagnosing":
        return {"error": "a drill is already running — answer, "
                         "give_up, or debrief first"}
    lattice = getattr(ctx, "lattice", None)
    if lattice is None:
        return {"error": "no lattice loaded"}
    cands = _candidates(lattice)
    if not cands:
        return {"error": "this lattice has no faultable elements "
                         "(powered quads/solenoids/cavities)"}
    rng = random.Random(seed)
    idx, el, family, param, kind = rng.choice(cands)
    old = float(getattr(el, param))
    if kind == "rf":
        delta = rng.uniform(5.0, 20.0) * rng.choice((-1.0, 1.0))
        new = old + delta
    else:
        factor = rng.uniform(0.85, 1.15)
        while 0.97 < factor < 1.03:          # too subtle to diagnose
            factor = rng.uniform(0.85, 1.15)
        new = old * factor
    s_pos, _total = _s_positions_m(lattice)
    setattr(el, param, new)
    st.phase = "diagnosing"
    st.secret = _Secret(index=idx, name=str(getattr(el, "name", f"#{idx}")),
                        cls=type(el).__name__, family=family, param=param,
                        old_value=old, new_value=float(new),
                        s_m=s_pos[idx])
    st.hints_used = 0
    st.t0 = time.monotonic()
    st.result = {}
    return {"started": True,
            "briefing": "A fault has been injected somewhere in the "
                        "lattice (details withheld — even from the "
                        "assistant).  Run simulations, read the "
                        "observables, and name the faulty element with "
                        "action='answer'.  Hints cost points; "
                        "give_up reveals the truth.  Restore happens "
                        "at debrief."}


def _hint(st: InstructorState, lattice) -> dict:
    if st.hints_used >= _MAX_HINTS:
        return {"error": f"hint ladder exhausted ({_MAX_HINTS} max) — "
                         "answer or give_up"}
    st.hints_used += 1
    sec = st.secret
    _pos, total = _s_positions_m(lattice)
    if st.hints_used == 1:
        return {"hint": 1, "text": f"The faulty element is {sec.family}."}
    if st.hints_used == 2:
        third = ("first" if sec.s_m < total / 3
                 else "middle" if sec.s_m < 2 * total / 3 else "last")
        return {"hint": 2, "text": f"It sits in the {third} third of "
                                   "the lattice."}
    rough = round(sec.s_m, 1 if total < 50 else 0)
    return {"hint": 3, "text": f"It is a {sec.cls}, near s = {rough} m."}


def _score(st: InstructorState, diagnosis_pts: int) -> dict:
    elapsed = time.monotonic() - st.t0
    speed = max(0.0, 25.0 * (1.0 - elapsed / _PAR_SECONDS))
    economy = max(0.0, 25.0 - 8.0 * st.hints_used)
    total = round(diagnosis_pts + speed + economy, 1)
    return {"score": total, "diagnosis": diagnosis_pts,
            "speed": round(speed, 1), "economy": round(economy, 1),
            "elapsed_s": round(elapsed, 1),
            "hints_used": st.hints_used}


def _restore(ctx, st: InstructorState) -> None:
    sec = st.secret
    if sec is None:
        return
    lattice = getattr(ctx, "lattice", None)
    if lattice is None:
        return
    try:
        el = lattice.elements[sec.index]
        setattr(el, sec.param, sec.old_value)   # bit-exact
    except Exception:                                       # noqa: BLE001
        pass


def _reveal(sec: _Secret) -> dict:
    return {"element": sec.name, "class": sec.cls, "param": sec.param,
            "was": sec.old_value, "faulted_to": sec.new_value,
            "s_m": round(sec.s_m, 3)}


def action(ctx, act: str, answer_element=None) -> dict:
    """hint | status | answer | give_up | debrief."""
    st = get_state(ctx)
    act = (act or "").strip().lower()
    if st.phase == "idle":
        return {"error": "no drill running — instructor_start begins one"}
    lattice = getattr(ctx, "lattice", None)
    if act == "status":
        out = {"phase": st.phase, "hints_used": st.hints_used,
               "elapsed_s": round(time.monotonic() - st.t0, 1)}
        if st.phase == "done":
            out["result"] = st.result
        return out
    if st.phase == "done":
        if act == "debrief":
            return st.result
        return {"error": "drill finished — see debrief"}
    if act == "hint":
        return _hint(st, lattice)
    if act == "give_up":
        _restore(ctx, st)
        st.result = {"gave_up": True, "truth": _reveal(st.secret),
                     **_score(st, 0),
                     "note": "lattice restored; session results may "
                             "still show the faulted machine — re-run "
                             "to refresh"}
        st.phase = "done"
        return st.result
    if act == "answer":
        if not answer_element:
            return {"error": "answer needs answer_element="}
        from linac_gen.assist.tools import _resolve_element
        idx, el, cand = _resolve_element(lattice, str(answer_element))
        if idx is None and cand:
            # ambiguous — return candidates WITHOUT submitting
            return {"ambiguous": True, "candidates": cand,
                    "note": "not submitted — name one exactly"}
        if idx is None:
            return {"error": f"no element matches {answer_element!r} "
                             "(not submitted)"}
        sec = st.secret
        if idx == sec.index:
            pts = 50
            verdict = "CORRECT"
        elif type(el).__name__ == sec.cls:
            pts = 20
            verdict = ("right family, wrong element — the fault was "
                       "elsewhere")
        else:
            pts = 0
            verdict = "incorrect"
        _restore(ctx, st)
        st.result = {"verdict": verdict, "answered": str(
            getattr(el, "name", f"#{idx}")),
            "truth": _reveal(sec), **_score(st, pts),
            "note": "lattice restored; session results may still show "
                    "the faulted machine — re-run to refresh"}
        st.phase = "done"
        return st.result
    if act == "debrief":
        return {"error": "the drill is still running — answer or "
                         "give_up first"}
    return {"error": f"unknown action {act!r} — hint/status/answer/"
                     "give_up/debrief"}


def restore_if_active(ctx) -> bool:
    """Session-close safety: an abandoned drill restores the lattice."""
    st = getattr(ctx, "_assist_instructor", None)
    if st is None or st.phase != "diagnosing":
        return False
    _restore(ctx, st)
    st.phase = "idle"
    st.secret = None
    return True
