"""Local intent fast-path — instant execution for unambiguous commands.

A full LLM round-trip (seconds) is the wrong tool for "show the RMS
plot", "status", or saying "next" during the guided tour.  This module
maps a SMALL, exact-match set of phrasings onto direct READ-tier tool
calls; anything nuanced falls through to the model untouched.

Philosophy: precision over recall.  A missed shortcut costs one model
turn; a wrong shortcut hijacks the user's actual question — so every
pattern anchors the WHOLE utterance, and only read-tier tools are
eligible (anything that computes or mutates always goes through the
model and the confirmation flow).
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field


@dataclass
class FastIntent:
    tool: str
    params: dict = field(default_factory=dict)
    kind: str = ""                    # "tour" | "nav" | "" (plain read)


_WS = re.compile(r"\s+")


def _norm(text) -> str:
    t = str(text or "").strip().lower()
    t = re.sub(r"^(?:please|hey|ok|okay)[,\s]+", "", t)
    t = re.sub(r"[.!?\s]+$", "", t)
    return _WS.sub(" ", t)


#: tour vocabulary — matched ONLY while a tour is actually active
_TOUR = {
    "next": "next", "continue": "next", "go on": "next",
    "next station": "next", "next step": "next",
    "back": "back", "previous": "back", "go back": "back",
    "repeat": "back",
    "where were we": "status", "tour status": "status",
    "stop the tour": "stop", "stop tour": "stop", "end the tour": "stop",
    "exit the tour": "stop",
}

_SIMPLE = [
    (re.compile(r"^(?:what'?s the )?status$|^how(?:'s| is) it looking$"),
     "get_status"),
    (re.compile(r"^(?:show |give me )?(?:the )?results? summary$"
                r"|^summari[sz]e (?:the )?results$"),
     "result_summary"),
    (re.compile(r"^summari[sz]e (?:the )?beam$|^beam summary$"),
     "summarize_beam"),
    (re.compile(r"^list (?:the )?runs$"), "list_runs"),
    (re.compile(r"^list (?:the )?plots$"), "list_plots"),
    (re.compile(r"^list (?:the )?tabs$"), "list_tabs"),
    (re.compile(r"^read (?:the )?notebook$"), "read_notebook"),
]

_PLOT_RX = re.compile(
    r"^(?:show|open|display|pull up)(?: me)?(?: the)? (.+?) plot$")
_TAB_RX = re.compile(
    r"^(?:show|open|switch to|go to|take me to)(?: the)? (.+?) tab$")


def match(text, ctx) -> FastIntent | None:
    """The fast intent for ``text``, or None → send to the model."""
    t = _norm(text)
    if not t:
        return None
    try:                             # tour words only while touring
        from linac_gen.assist.guide import get_state
        if get_state(ctx).active and t in _TOUR:
            return FastIntent("guide", {"action": _TOUR[t]}, kind="tour")
    except Exception:                                       # noqa: BLE001
        pass
    for rx, tool in _SIMPLE:
        if rx.match(t):
            return FastIntent(tool)
    m = _PLOT_RX.match(t)
    if m:
        return FastIntent("open_plot", {"name": m.group(1)}, kind="nav")
    m = _TAB_RX.match(t)
    if m:
        return FastIntent("show_tab", {"tab": m.group(1)}, kind="nav")
    return None


def render_result(fi: FastIntent, res: dict) -> tuple[str, str]:
    """(chat_markdown, speech) for an ok ToolResult of intent ``fi``."""
    data = (res or {}).get("data") or {}
    if fi.tool == "guide":
        say = str(data.get("say") or data.get("message") or "").strip()
        st, of = data.get("station"), data.get("of")
        title = data.get("title") or ""
        head = (f"**🎓 {st}/{of} — {title}**" if st
                else "**🎓 Guided tour**")
        ex = str(data.get("exercise") or "").strip()
        chat = head + ("\n\n" + say if say else "")
        speech = say
        if ex:
            chat += f"\n\n*Try it:* {ex}"
            speech += "  Try it: " + ex
        return chat, speech
    if fi.tool == "open_plot":
        return (f"Opened the **{fi.params.get('name')}** plot.", "Opened.")
    if fi.tool == "show_tab":
        return (f"Switched to the **{fi.params.get('tab')}** tab.", "Done.")
    if fi.tool == "get_status":
        lat = data.get("lattice")
        bits = []
        if lat:
            import os
            nm = os.path.basename(lat.get("path") or "") or "(unsaved)"
            bits.append(f"lattice **{nm}** — {lat.get('n_elements')} "
                        f"elements, {lat.get('length_m', 0):.1f} m")
        else:
            bits.append("no lattice loaded")
        bits.append("results loaded" if data.get("results_loaded")
                    else "no results yet")
        chat = "Status: " + "; ".join(bits) + "."
        return chat, chat.replace("**", "")
    if fi.tool in ("list_runs", "list_plots", "list_tabs"):
        for v in data.values():
            if isinstance(v, list):
                names = [str(x.get("name", x)) if isinstance(x, dict)
                         else str(x) for x in v][:12]
                chat = ", ".join(names) if names else "(none)"
                return chat, (f"{len(names)} items."
                              if len(names) > 3 else chat)
        # fall through to generic
    text = data.get("text")
    if isinstance(text, str) and text.strip():
        t = text.strip()
        return (t if len(t) < 1600 else t[:1600] + " …"), ""
    blob = json.dumps(data, default=str)
    return (blob if len(blob) < 400 else blob[:400] + " …"), ""
