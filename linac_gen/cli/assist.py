"""``linac_gen assist`` — the OPTIONAL AI-assistant chat REPL.

Everything here lazy-imports :mod:`linac_gen.assist`; no other CLI
path touches it, and an unconfigured environment prints setup help
instead of failing.
"""
from __future__ import annotations


def add_arguments(p) -> None:
    p.add_argument("input", nargs="?", default=None,
                   help="optional .lgproj project or lattice file to "
                        "load into the session")
    p.add_argument("--provider",
                   choices=("anthropic", "openai", "claude_sdk"),
                   help="LLM backend (claude_sdk = keyless, reuses the "
                        "Claude Code login; default: inferred from env)")
    p.add_argument("--base-url", help="OpenAI-compatible endpoint, "
                   "e.g. http://localhost:11434/v1 for local ollama")
    p.add_argument("--model", help="model name")
    p.add_argument("--calc-dir", default="runs",
                   help="directory for results + session ledgers")
    p.add_argument("--auto-approve-compute", action="store_true",
                   help="skip confirmation for compute tools "
                        "(mutate tools always confirm)")
    p.add_argument("--once", metavar="PROMPT",
                   help="ask one question and exit (no REPL)")
    p.add_argument("--replay", metavar="LEDGER",
                   help="re-execute a session ledger deterministically "
                        "(no LLM, no network) and exit")
    p.add_argument("--mcp", action="store_true",
                   help="run HELIX as an MCP server on stdio (drive it "
                        "from Claude Code / Claude Desktop with YOUR "
                        "own Claude subscription — no API key, HELIX "
                        "makes no network call)")


def run(args) -> int:
    import os
    import sys

    if args.mcp:
        from linac_gen.assist.mcp_server import serve
        try:
            return serve(initial_input=args.input,
                         calc_dir=args.calc_dir)
        except RuntimeError as exc:
            print(str(exc), file=sys.stderr)
            return 2

    if args.replay:
        from linac_gen.assist.replay import replay
        report = replay(args.replay, calc_dir=args.calc_dir)
        print(report.summary())
        for name, why in report.skipped:
            print(f"  skip {name}: {why}")
        for name, msg in report.failed:
            print(f"  FAIL {name}: {msg}", file=sys.stderr)
        return 0 if report.ok else 1

    # config: CLI flags override env
    if args.provider:
        os.environ["HELIX_ASSIST_PROVIDER"] = args.provider
    if args.base_url:
        os.environ["HELIX_ASSIST_BASE_URL"] = args.base_url
    if args.model:
        os.environ["HELIX_ASSIST_MODEL"] = args.model

    import linac_gen.assist as assist
    ok, reason = assist.available()
    if not ok:
        print(reason, file=sys.stderr)
        return 2

    from linac_gen.assist.agent import AgentSession
    from linac_gen.assist.config import resolve_config
    from linac_gen.assist.repl import CliApprover, _print_event, run_repl
    from linac_gen.assist.tools import WorkContext

    cfg = resolve_config()
    cfg.auto_approve_compute = bool(args.auto_approve_compute)
    ctx = WorkContext(calc_dir=args.calc_dir)
    if args.input:
        from linac_gen.assist.tools import TOOLS
        env = TOOLS["load_lattice"].fn(ctx, path=args.input)
        if env["status"] != "ok":
            print(f"could not load {args.input}: "
                  f"{env['data'].get('message')}", file=sys.stderr)
            return 1
        print(f"loaded {args.input} "
              f"({env['data']['n_elements']} elements)")

    session = AgentSession(cfg, ctx, approver=CliApprover(),
                           on_event=_print_event)
    print(f"provider: {cfg.provider} ({cfg.model})")
    if args.once:
        try:
            session.ask(args.once)
        finally:
            session.close()
        return 0
    return run_repl(session)
