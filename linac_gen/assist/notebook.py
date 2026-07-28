"""Persistent lab notebook — the assistant's cross-session memory.

One markdown file per calc dir (``assist_notebook.md``): each session
appends a mechanical summary (derived from the ledger — no extra LLM
turn), and the ``notebook_note`` tool appends user-dictated notes.  At
session start the most recent entries are folded into the system prompt
so the assistant remembers what was done and concluded before.

All I/O is best-effort: a missing/corrupt notebook never breaks a
session (memory is a nicety, the run is the job).
"""
from __future__ import annotations

import datetime as _dt
import json
import os

_FILENAME = "assist_notebook.md"


def notebook_path(calc_dir: str) -> str:
    return os.path.join(calc_dir or ".", _FILENAME)


def append_note(calc_dir: str, text: str) -> str:
    """Append a timestamped user note; returns the notebook path."""
    path = notebook_path(calc_dir)
    stamp = _dt.datetime.now().strftime("%Y-%m-%d %H:%M")
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(f"- [{stamp}] {text.strip()}\n")
    return path


def load_tail(calc_dir: str, k_entries: int = 3,
              max_chars: int = 2400) -> str:
    """The last ``k_entries`` session entries (## headings) plus any trailing
    loose notes, capped at ``max_chars``.  Empty string when absent."""
    path = notebook_path(calc_dir)
    try:
        # errors="replace": a corrupt/binary notebook degrades to mojibake
        # instead of raising — memory must never break a session
        text = open(path, encoding="utf-8", errors="replace").read()
    except OSError:
        return ""
    if not text.strip():
        return ""
    # split into blocks at session headings, keeping loose notes attached
    blocks, cur = [], []
    for line in text.splitlines():
        if line.startswith("## ") and cur:
            blocks.append("\n".join(cur))
            cur = [line]
        else:
            cur.append(line)
    if cur:
        blocks.append("\n".join(cur))
    tail = "\n".join(blocks[-max(1, k_entries):]).strip()
    return tail[-max_chars:]


def append_session_summary(session) -> None:
    """Mechanical end-of-session entry derived from the ledger JSONL.
    Never raises."""
    try:
        if getattr(session, "turn", 0) < 1:
            return                                  # nothing happened
        calc_dir = getattr(session.context, "calc_dir", ".") or "."
        lines = open(session.ledger.path, encoding="utf-8",
                     errors="replace").readlines()
        tools, written, last_assistant = {}, [], ""
        for raw in lines:
            try:
                rec = json.loads(raw)
            except ValueError:
                continue
            ev = rec.get("event")
            if ev == "tool":
                name = rec.get("tool", "?")
                tools[name] = tools.get(name, 0) + 1
                if name == "write_results" and rec.get("status") == "ok":
                    written.append(str(rec.get("params", {})
                                       .get("path", "?")))
            elif ev == "assistant" and rec.get("text"):
                last_assistant = str(rec["text"])
        stamp = _dt.datetime.now().strftime("%Y-%m-%d %H:%M")
        lat = getattr(session.context, "lattice_path", "") or "(none)"
        parts = [f"## Session {stamp}",
                 f"- lattice: {lat}",
                 f"- turns: {session.turn}; tools: "
                 + (", ".join(f"{n}×{c}" if c > 1 else n
                              for n, c in sorted(tools.items()))
                    or "(none)")]
        if written:
            parts.append("- results written: " + ", ".join(written))
        if last_assistant:
            parts.append("- last conclusion: "
                         + " ".join(last_assistant.split())[:240])
        path = notebook_path(calc_dir)
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            f.write("\n" + "\n".join(parts) + "\n")
    except Exception:                                       # noqa: BLE001
        pass
