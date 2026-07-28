"""Terminal chat REPL for the assistant (also the manual testbed)."""
from __future__ import annotations

import sys


class CliApprover:
    """Blocking y/N/a prompt echoing the exact resolved call."""

    def __call__(self, req):
        from linac_gen.assist.agent import Decision
        print(f"\n[confirm] {req.tier}:")
        print("  " + req.pretty.replace("\n", "\n  "))
        opts = ("[y]es / [N]o"
                + (" / [a]lways (compute, this session)"
                   if req.allow_session_auto else ""))
        try:
            ans = input(f"Approve? {opts}: ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            return Decision.DENY
        if ans in ("y", "yes"):
            return Decision.APPROVE
        if ans in ("a", "always") and req.allow_session_auto:
            return Decision.APPROVE_SESSION
        return Decision.DENY


def _print_event(event: dict) -> None:
    t = event.get("type")
    if t == "assistant_delta":                 # live streaming (SDK backend)
        print(event.get("text", ""), end="", flush=True)
    elif t == "assistant_delta_done":
        print()                                # newline to close the reply
    elif t == "assistant_text":
        print(f"\n{event['text']}\n")
    elif t == "tool_start":
        print(f"  · {event['tool']} …", file=sys.stderr)
    elif t == "job_submitted":
        print(f"  · {event['tool']} → {event['job_id']} (background)",
              file=sys.stderr)
    elif t == "event":
        print(f"  ‣ [event] {event.get('text')}", file=sys.stderr)
    elif t == "progress":
        print(f"  … {event.get('text')}", file=sys.stderr)
    elif t == "error":
        print(f"  ! {event.get('message')}", file=sys.stderr)


def run_repl(session) -> int:
    print("HELIX assistant — /help for commands, /quit to exit.")
    try:
        import readline                                     # noqa: F401
    except ImportError:
        pass
    while True:
        try:
            text = input("assist> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not text:
            continue
        if text in ("/quit", "/exit", "/q"):
            break
        if text == "/help":
            print("/jobs  /auto on|off  /usage  /tools  /quit")
            continue
        if text == "/jobs":
            for j in session.jobs.list_jobs()["data"]["jobs"]:
                print(f"  {j['job_id']}  {j['tool']:14s} {j['state']:9s}"
                      f" s={j['progress_s_mm'] / 1000:.1f} m")
            continue
        if text.startswith("/auto"):
            session.auto_approve_compute = text.endswith("on")
            print(f"  auto-approve compute: "
                  f"{session.auto_approve_compute}")
            continue
        if text == "/usage":
            u = session.usage_total
            print(f"  tokens in={u.input_tokens} out={u.output_tokens}")
            continue
        if text == "/tools":
            for t in session.tools.values():
                print(f"  {t.name:22s} [{t.tier}]")
            continue
        try:
            session.ask(text)
        except Exception as exc:                            # noqa: BLE001
            print(f"[assist error] {exc}", file=sys.stderr)
    session.close()
    print(f"session ledger: {session.ledger.path}")
    return 0
