"""Local intent fast-path: exact-match philosophy, read-tier only,
tour vocabulary gated on an ACTIVE tour."""
from __future__ import annotations

import pytest

from linac_gen.assist import intents as I
from linac_gen.assist.tools import TOOLS, WorkContext


@pytest.mark.parametrize("text,tool", [
    ("status", "get_status"),
    ("What's the status?", "get_status"),
    ("how's it looking", "get_status"),
    ("summarize the results", "result_summary"),
    ("results summary", "result_summary"),
    ("summarize the beam", "summarize_beam"),
    ("list runs", "list_runs"),
    ("list the plots", "list_plots"),
    ("read the notebook", "read_notebook"),
])
def test_simple_intents_match(text, tool):
    fi = I.match(text, WorkContext())
    assert fi is not None and fi.tool == tool


@pytest.mark.parametrize("text,tool,key,val", [
    ("show the RMS plot", "open_plot", "name", "rms"),
    ("open the phase space plot", "open_plot", "name", "phase space"),
    ("switch to the Results tab", "show_tab", "tab", "results"),
    ("go to the beam tab", "show_tab", "tab", "beam"),
])
def test_nav_intents_extract_target(text, tool, key, val):
    fi = I.match(text, WorkContext())
    assert fi is not None and fi.tool == tool
    assert fi.params[key] == val


@pytest.mark.parametrize("text", [
    "run the envelope",                    # compute — model + confirm
    "set quad 5 to 3 tesla per meter",     # mutate — never fast
    "what would happen if I raise the gradient",
    "show me how matching works",
    "why is the transmission dropping",
    "next",                                # tour word with NO active tour
    "",
])
def test_nuance_and_writes_fall_through(text):
    assert I.match(text, WorkContext()) is None


def test_tour_words_gated_on_active_tour():
    from linac_gen.assist.guide import get_state
    ctx = WorkContext()
    assert I.match("next", ctx) is None
    get_state(ctx).active = True
    fi = I.match("next", ctx)
    assert fi is not None and fi.tool == "guide"
    assert fi.params == {"action": "next"}
    assert I.match("stop the tour", ctx).params == {"action": "stop"}
    assert I.match("where were we", ctx).params == {"action": "status"}
    # nuance still falls through even mid-tour
    assert I.match("why is station three important", ctx) is None


def test_every_fast_tool_is_read_tier():
    """The registry must never let a fast intent reach compute/mutate."""
    names = {t for _, t in I._SIMPLE} | {"open_plot", "show_tab", "guide"}
    for n in names:
        assert TOOLS[n].tier == "read", n


def test_render_result_guide_and_nav():
    fi = I.FastIntent("guide", {"action": "next"}, kind="tour")
    chat, speech = I.render_result(fi, {
        "status": "ok",
        "data": {"station": 2, "of": 15, "title": "The Lattice tab",
                 "say": "This is the machine.", "exercise": "Click one."}})
    assert "2/15" in chat and "This is the machine." in chat
    assert "Try it" in chat
    assert "This is the machine." in speech and "Click one." in speech
    fi2 = I.FastIntent("open_plot", {"name": "rms"}, kind="nav")
    chat2, _ = I.render_result(fi2, {"status": "ok", "data": {}})
    assert "rms" in chat2
