"""Guided tour of the HELIX interface (MIRAGE's guide mode, adapted).

A fixed curriculum of stations, each ``(title, nav, say, exercise)``:
``nav`` drives the GUI through the WorkContext hooks (inert headless —
the tour degrades to narration-only), ``say`` is pre-written
voice-formatted narration the assistant relays nearly verbatim, and
``exercise`` is an optional hands-on step.  Cross-turn state lives on
the WorkContext (``ctx._assist_guide``) so "next / back / where were
we" survive turns.
"""
from __future__ import annotations

from dataclasses import dataclass, field

#: (title, nav, say, exercise) — nav: {"tab": label[, "subtab": label]}
#: | {"plot": key} | None.  Tab labels match the app's TABS registry
#: (state.py): Beam, Lattice, Matching, Numerics, Surrogates,
#: Error Study, Failure Study, Results.
CURRICULUM: list[tuple] = [
    ("Welcome", None,
     "Welcome to HELIX — a beam-dynamics simulator for linear "
     "accelerators, validated against TraceWin.  This tour walks the "
     "interface one station at a time.  Say next to continue, back to "
     "repeat, or stop to end the tour.",
     ""),
    ("The Lattice tab", {"tab": "Lattice"},
     "This is the machine: every element of the loaded lattice in "
     "order — drifts, quadrupoles, cavities, bends.  The table and the "
     "graphic stay in sync, and the s cursor marks a position along "
     "the line.",
     "Click any element in the listing and watch it highlight."),
    ("Selecting elements", {"tab": "Lattice"},
     "You can ask me to point: say highlight, then an element name, "
     "and I will select it, halo it, and move the s cursor to its "
     "entrance.",
     "Try: 'highlight the first quadrupole'."),
    ("The Beam tab", {"tab": "Beam"},
     "The input beam: species, energy, current, and the Twiss "
     "parameters.  Two conventions matter: transverse emittances are "
     "NORMALIZED, in pi millimeter milliradian, and the longitudinal "
     "alpha z is MINUS the TraceWin value — HELIX negates it, and so "
     "must you when copying numbers from a TraceWin deck.",
     ""),
    ("The Numerics tab", {"tab": "Numerics"},
     "Every solver setting a run uses: integration steps, the space "
     "charge grid and kernel, and the convergence scans that validate "
     "them.  When I press Run for you, THESE settings are honoured — "
     "the same as clicking the button yourself.",
     ""),
    ("Envelope versus multiparticle", None,
     "HELIX has two main engines.  The envelope solver propagates "
     "second moments — seconds fast, great for matching.  The "
     "multiparticle engine tracks thousands of macro particles with a "
     "3D space-charge solver — slower, but it carries halo and "
     "nonlinear physics.  Ask me to run either.",
     "Say: 'run the envelope' — I will ask you to confirm first."),
    ("The Results tab", {"tab": "Results"},
     "Every diagnostic of the last run: KPI tiles on top, then cards "
     "opening full plot windows — envelopes, emittances, Twiss, "
     "losses, phase space, and more.  Cards grey out when their "
     "quantity does not exist for the run type.",
     ""),
    ("Reading plots together", {"plot": "rms"},
     "I opened the RMS envelope plot.  I can SEE figures: ask me "
     "whether the beam looks matched, or what that bump near the end "
     "is, and I will look at the actual image before answering.",
     "Ask: 'look at this plot — is the beam matched?'"),
    ("The Matching tab", {"tab": "Matching"},
     "Lattice matching: ADJUST cards define the knobs, SET cards the "
     "targets, and seven algorithms drive them.  Matching REWRITES "
     "lattice parameters, so it always asks before running.",
     ""),
    ("The Error Study tab", {"tab": "Error Study"},
     "Monte-Carlo error studies: alignment, field and RF errors drawn "
     "per seed, tracked in ensembles, with orbit correction.  This is "
     "where tolerance budgets are validated.",
     ""),
    ("The Failure Study tab", {"tab": "Failure Study"},
     "What-if failures: cavities or magnets off, and the retuning "
     "workflows that recover the beam.",
     ""),
    ("The Surrogates tab", {"tab": "Surrogates"},
     "Neural surrogates that can stand in for expensive elements, with "
     "honest scope guards — out-of-range inputs fall back to physics.",
     ""),
    ("Reports, notebook, comparisons", None,
     "I keep a lab notebook across sessions, write markdown run "
     "reports with figures, compare runs against each other or "
     "against a TraceWin export, and can brief you on what a session "
     "did.  Ask for 'a report of this run' or 'what did we do "
     "yesterday'.",
     ""),
    ("Working with me safely", None,
     "Read-only questions run freely.  Anything that computes asks "
     "once; anything that CHANGES the machine always asks, showing "
     "the exact call.  You can say yes or cancel out loud, stop me "
     "any time with the Stop button or Escape, or just talk over me.",
     ""),
    ("Where to learn more", None,
     "The full manual is built in: ask me anything and I will cite "
     "the section I used.  That is the end of the tour — the machine "
     "is yours.",
     "Ask: 'search the manual for space charge convergence'."),
]


@dataclass
class GuideState:
    active: bool = False
    idx: int = 0
    visited: list = field(default_factory=list)


def get_state(ctx) -> GuideState:
    st = getattr(ctx, "_assist_guide", None)
    if st is None:
        st = GuideState()
        ctx._assist_guide = st
    return st


def _navigate(ctx, nav) -> bool:
    """Drive the GUI hooks; headless they are inert → narration-only."""
    if not nav:
        return False
    try:
        if "tab" in nav:
            return bool(ctx.show_tab(nav["tab"], nav.get("subtab")))
        if "plot" in nav:
            return bool(ctx.open_plot(nav["plot"]))
    except Exception:                                       # noqa: BLE001
        pass
    return False


def _station_payload(ctx, st: GuideState) -> dict:
    title, nav, say, exercise = CURRICULUM[st.idx]
    navigated = _navigate(ctx, nav)
    out = {"station": st.idx + 1, "of": len(CURRICULUM),
           "title": title, "say": say, "navigated": navigated}
    if exercise:
        out["exercise"] = exercise
    if st.idx == len(CURRICULUM) - 1:
        out["last_station"] = True
    return out


def run_action(ctx, action: str, station=None) -> dict:
    """The guide state machine.  Actions: start | next | back | repeat
    | goto | status | stop.  Returns a payload the assistant relays —
    the ``say`` text nearly verbatim, ONE station per turn."""
    st = get_state(ctx)
    action = (action or "").strip().lower()
    if action == "start":
        st.active = True
        st.idx = 0
        st.visited = [0]
        return _station_payload(ctx, st)
    if action == "status":
        if not st.active:
            return {"active": False,
                    "note": "no tour running — action='start' begins one"}
        title = CURRICULUM[st.idx][0]
        return {"active": True, "station": st.idx + 1,
                "of": len(CURRICULUM), "title": title}
    if not st.active:
        return {"error": "no tour is running — use action='start'"}
    if action == "stop":
        st.active = False
        return {"stopped": True,
                "say": "Tour ended.  Ask me anything, any time."}
    if action == "next":
        if st.idx >= len(CURRICULUM) - 1:
            st.active = False
            return {"done": True,
                    "say": "That was the last station — tour complete."}
        st.idx += 1
    elif action == "back":
        st.idx = max(0, st.idx - 1)
    elif action == "goto":
        try:
            st.idx = int(station) - 1
        except (TypeError, ValueError):
            return {"error": "goto needs station= (1-based index)"}
        if not 0 <= st.idx < len(CURRICULUM):
            st.idx = 0
            return {"error": f"station must be 1..{len(CURRICULUM)}"}
    elif action != "repeat":
        return {"error": f"unknown action {action!r} — use start/next/"
                         "back/repeat/goto/status/stop"}
    if st.idx not in st.visited:
        st.visited.append(st.idx)
    return _station_payload(ctx, st)
