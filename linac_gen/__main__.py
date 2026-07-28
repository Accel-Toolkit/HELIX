"""``python -m linac_gen`` — HELIX batch-mode command-line interface.

Subcommands:

* ``run``   — one headless tracking simulation (envelope / mp / matrix);
* ``scan``  — a native parameter scan, CSV summary;
* ``batch`` — a multi-run campaign from a JSON job file;
* ``twiss`` — matched Twiss: whole-lattice periodic, or a FODO-cell input
  match back-propagated to the entrance (for transfer lines);
* ``backtrack`` — backward tracking: reconstruct an upstream distribution
  from a downstream ``.dst`` (or design-mode exit Twiss);
* ``match`` — delegates to the matcher (``python -m linac_gen.matching``).

All of it is GUI-free and drives the same engines the GUI uses.
"""
from __future__ import annotations

import argparse
import sys


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)

    # The matcher keeps its own argument parser — delegate to it verbatim
    # so `python -m linac_gen match …` == `python -m linac_gen.matching …`.
    if argv and argv[0] == "match":
        from linac_gen.matching.__main__ import main as match_main
        return match_main(argv[1:])

    parser = argparse.ArgumentParser(
        prog="python -m linac_gen",
        description="HELIX batch-mode CLI — headless tracking, scans, "
                    "and multi-run campaigns.",
        epilog="`python -m linac_gen match …` delegates to the matcher "
               "(python -m linac_gen.matching).",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    from linac_gen.cli import assist as assist_cmd
    from linac_gen.cli import backtrack as backtrack_cmd
    from linac_gen.cli import batch as batch_cmd
    from linac_gen.cli import run as run_cmd
    from linac_gen.cli import scan as scan_cmd
    from linac_gen.cli import twiss as twiss_cmd
    from linac_gen.cli import multiobjective as mo_cmd
    from linac_gen.cli import failures as fail_cmd

    run_cmd.add_arguments(sub.add_parser(
        "run", help="run one headless simulation"))
    scan_cmd.add_arguments(sub.add_parser(
        "scan", help="native parameter scan → CSV summary"))
    batch_cmd.add_arguments(sub.add_parser(
        "batch", help="multi-run campaign from a JSON job file"))
    twiss_cmd.add_arguments(sub.add_parser(
        "twiss", help="matched Twiss — whole-lattice or FODO-cell input match"))
    mo_cmd.add_arguments(sub.add_parser(
        "mo", help="multi-objective design — Pareto front over ADJUST knobs"))
    fail_cmd.add_arguments(sub.add_parser(
        "failures", help="element failure impact + recovery → CSV"))
    backtrack_cmd.add_arguments(sub.add_parser(
        "backtrack", help="backward tracking — reconstruct an upstream "
                          "distribution from a downstream state"))
    assist_cmd.add_arguments(sub.add_parser(
        "assist", help="AI assistant chat (OPTIONAL — local or cloud "
                       "LLM; the rest of HELIX never needs it)"))
    sub.add_parser("match", add_help=False,
                   help="run the matcher (delegates to linac_gen.matching)")

    args = parser.parse_args(argv)
    if args.command == "run":
        return run_cmd.run(args)
    if args.command == "scan":
        return scan_cmd.run(args)
    if args.command == "batch":
        return batch_cmd.run(args)
    if args.command == "twiss":
        return twiss_cmd.run(args)
    if args.command == "mo":
        return mo_cmd.run(args)
    if args.command == "failures":
        return fail_cmd.run(args)
    if args.command == "backtrack":
        return backtrack_cmd.run(args)
    if args.command == "assist":
        return assist_cmd.run(args)
    return 2


if __name__ == "__main__":
    sys.exit(main())
