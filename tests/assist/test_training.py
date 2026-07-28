"""Training features: guide-tour state machine and instructor drills.
THE drill invariant: the truth never appears in any payload before the
debrief — asserted by walking every returned structure."""
from __future__ import annotations

import json

import pytest

from linac_gen.assist.tools import TOOLS


def _walk_strings(obj):
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield str(k)
            yield from _walk_strings(v)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            yield from _walk_strings(v)
    else:
        yield str(obj)


# ---------------------------------------------------------------------------
# guide
# ---------------------------------------------------------------------------
def test_guide_walks_stations_in_order(ctx):
    out = TOOLS["guide"].fn(ctx, action="start")
    assert out["status"] == "ok"
    d = out["data"]
    assert d["station"] == 1 and d["title"] == "Welcome"
    assert "Say next" in d["say"] or "say next" in d["say"].lower()
    d2 = TOOLS["guide"].fn(ctx, action="next")["data"]
    assert d2["station"] == 2 and "Lattice" in d2["title"]
    d3 = TOOLS["guide"].fn(ctx, action="back")["data"]
    assert d3["station"] == 1
    d4 = TOOLS["guide"].fn(ctx, action="repeat")["data"]
    assert d4["station"] == 1


def test_guide_goto_status_stop(ctx):
    TOOLS["guide"].fn(ctx, action="start")
    d = TOOLS["guide"].fn(ctx, action="goto", station=4)["data"]
    assert d["station"] == 4 and "Beam" in d["title"]
    st = TOOLS["guide"].fn(ctx, action="status")["data"]
    assert st["active"] and st["station"] == 4
    stop = TOOLS["guide"].fn(ctx, action="stop")["data"]
    assert stop["stopped"]
    assert not TOOLS["guide"].fn(ctx, action="status")["data"]["active"]


def test_guide_completes_at_last_station(ctx):
    from linac_gen.assist.guide import CURRICULUM
    TOOLS["guide"].fn(ctx, action="start")
    TOOLS["guide"].fn(ctx, action="goto", station=len(CURRICULUM))
    done = TOOLS["guide"].fn(ctx, action="next")["data"]
    assert done["done"]


def test_guide_actions_require_a_tour(ctx):
    out = TOOLS["guide"].fn(ctx, action="next")
    assert out["status"] == "error"
    out2 = TOOLS["guide"].fn(ctx, action="warp")
    assert out2["status"] == "error"


def test_guide_headless_is_narration_only(ctx):
    TOOLS["guide"].fn(ctx, action="start")
    d = TOOLS["guide"].fn(ctx, action="next")["data"]   # Lattice-tab nav
    assert d["navigated"] is False          # inert hooks headless


def test_guide_tab_labels_match_the_app_registry(ctx):
    """Every nav target must be a REAL tab label (state.py TABS)."""
    from linac_gen.assist.guide import CURRICULUM
    try:
        from linac_gen_gui.interphase.state import TABS
    except Exception:
        pytest.skip("GUI not importable")
    labels = {label for _id, label in TABS}
    for _t, nav, _s, _e in CURRICULUM:
        if nav and "tab" in nav:
            assert nav["tab"] in labels, nav


# ---------------------------------------------------------------------------
# instructor
# ---------------------------------------------------------------------------
def _fault_free(ctx):
    return {getattr(e, "name", i): getattr(e, "gradient", None)
            for i, e in enumerate(ctx.lattice.elements)}


def test_drill_hides_the_truth_until_debrief(ctx):
    g0 = {e.name: getattr(e, "gradient", None)
          for e in ctx.lattice.elements if hasattr(e, "gradient")}
    out = TOOLS["instructor_start"].fn(ctx, seed=7)
    assert out["status"] == "ok"
    # the fault is real: exactly one gradient changed
    changed = [n for n, g in g0.items()
               if getattr(next(e for e in ctx.lattice.elements
                               if e.name == n), "gradient", None) != g]
    assert len(changed) == 1
    secret_name = changed[0]
    secret_old = g0[secret_name]
    # THE leak check: neither briefing nor hints nor status may carry
    # the faulted element's name or its old/new values
    payloads = [out]
    for act in ("status", "hint", "hint", "hint"):
        payloads.append(TOOLS["instructor"].fn(ctx, action=act))
    for p in payloads:
        for s in _walk_strings(p):
            assert secret_name not in s, (secret_name, p)
            assert str(secret_old) not in s, p
    # 4th hint refused — and still no leak
    out4 = TOOLS["instructor"].fn(ctx, action="hint")
    assert out4["status"] == "error"
    # correct answer scores diagnosis 50 and restores bit-exact
    ans = TOOLS["instructor"].fn(ctx, action="answer",
                                 answer_element=secret_name)
    d = ans["data"]
    assert d["verdict"] == "CORRECT"
    assert d["diagnosis"] == 50
    assert d["truth"]["element"] == secret_name
    el = next(e for e in ctx.lattice.elements if e.name == secret_name)
    assert el.gradient == secret_old        # bit-exact restore


def test_drill_hint_ladder_content(ctx):
    TOOLS["instructor_start"].fn(ctx, seed=3)
    h1 = TOOLS["instructor"].fn(ctx, action="hint")["data"]
    assert h1["hint"] == 1 and "magnet" in h1["text"]
    h2 = TOOLS["instructor"].fn(ctx, action="hint")["data"]
    assert h2["hint"] == 2 and "third" in h2["text"]
    h3 = TOOLS["instructor"].fn(ctx, action="hint")["data"]
    assert h3["hint"] == 3 and "Quadrupole" in h3["text"]
    TOOLS["instructor"].fn(ctx, action="give_up")


def test_drill_ambiguous_answer_not_submitted(ctx):
    TOOLS["instructor_start"].fn(ctx, seed=1)
    out = TOOLS["instructor"].fn(ctx, action="answer",
                                 answer_element="Q")   # QF and QD match
    d = out["data"]
    assert d.get("ambiguous")
    assert set(d["candidates"]) >= {"QF", "QD"}
    # drill still running
    st = TOOLS["instructor"].fn(ctx, action="status")["data"]
    assert st["phase"] == "diagnosing"
    TOOLS["instructor"].fn(ctx, action="give_up")


def test_drill_give_up_scores_zero_and_restores(ctx):
    g0 = [getattr(e, "gradient", None) for e in ctx.lattice.elements]
    TOOLS["instructor_start"].fn(ctx, seed=11)
    out = TOOLS["instructor"].fn(ctx, action="give_up")["data"]
    assert out["gave_up"] and out["diagnosis"] == 0
    assert "truth" in out
    g1 = [getattr(e, "gradient", None) for e in ctx.lattice.elements]
    assert g0 == g1


def test_drill_wrong_class_scores_zero_right_class_partial(ctx):
    TOOLS["instructor_start"].fn(ctx, seed=7)      # faults a quad
    from linac_gen.assist.instructor import get_state
    sec = get_state(ctx).secret
    other_quad = "QD" if sec.name == "QF" else "QF"
    ans = TOOLS["instructor"].fn(ctx, action="answer",
                                 answer_element=other_quad)["data"]
    assert ans["diagnosis"] == 20              # right family, wrong element
    assert "wrong element" in ans["verdict"]


def test_drill_second_start_refused_and_close_restores(ctx,
                                                       assist_config):
    from linac_gen.assist.agent import AgentSession
    from linac_gen.assist.testing import MockProvider, ScriptedApprover
    g0 = [getattr(e, "gradient", None) for e in ctx.lattice.elements]
    TOOLS["instructor_start"].fn(ctx, seed=5)
    again = TOOLS["instructor_start"].fn(ctx, seed=5)
    assert again["status"] == "error"
    # abandoning the drill: session close restores the lattice
    sess = AgentSession(assist_config, ctx,
                        approver=ScriptedApprover([]),
                        provider=MockProvider([]))
    sess.close()
    g1 = [getattr(e, "gradient", None) for e in ctx.lattice.elements]
    assert g0 == g1


def test_prompt_carries_training_rules():
    from linac_gen.assist.prompts import build_system_prompt
    p = build_system_prompt()
    assert "hidden FROM YOU TOO" in p
    assert "one\n  station per turn" in p or "one station per turn" in \
        " ".join(p.split())
