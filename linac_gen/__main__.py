"""``python -m linac_gen`` — HELIX batch-mode command-line interface.

Subcommands:

* ``run``   — one headless tracking simulation (envelope / mp / matrix);
* ``scan``  — a native parameter scan, CSV summary;
* ``batch`` — a multi-run campaign from a JSON job file;
* ``study`` — a parameter study: per-run folders, resume,
  oat/zip/grid/random/lhs strategies;
* ``twiss`` — matched Twiss: whole-lattice periodic, or a FODO-cell input
  match back-propagated to the entrance (for transfer lines);
* ``mo``    — multi-objective design: a Pareto front over ADJUST knobs;
* ``failures`` — element failure impact + recovery, CSV summary;
* ``backtrack`` — backward tracking: reconstruct an upstream distribution
  from a downstream ``.dst`` (or design-mode exit Twiss);
* ``assist`` — the AI assistant chat (optional — local or cloud LLM;
  the rest of HELIX never needs it);
* ``export`` — write the lattice in another code's format (MAD-X
  ``SEQUENCE``, the exact inverse of the MAD-X importer);
* ``twini`` — TraceWin project options file (``.ini``): the converted
  beam, a decode report, or a ``.lgproj`` for the deck;
* ``reliability`` — the Reliability Study mode: fault tolerance with
  compensation, imperfections with faults on error seeds, foil scenarios
  and availability as one resumable campaign with a report; job export /
  import and a built-in self-test;
* ``ring`` — ring mode: ``ring optics`` gives a ring's tunes, chromaticity,
  momentum compaction and periodic Twiss from the ring kernels;
* ``match`` — delegates to the matcher (``python -m linac_gen.matching``).

Every subcommand takes ``--threads N|all``: the CPU threads for the
space-charge FFT, the field maps and the charge deposition (default
``$HELIX_THREADS``, else ``$OMP_NUM_THREADS``, else every CPU; results are
the same for any number).

All of it is GUI-free and drives the same engines the GUI uses.
"""
from __future__ import annotations

import argparse
import sys


def main(argv: list[str] | None = None) -> int:
    from linac_gen.cli.common import (apply_threads_arg,
                                      make_console_encoding_safe,
                                      threads_parent,
                                      warn_if_no_cpp_kernels)
    make_console_encoding_safe()
    warn_if_no_cpp_kernels()
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
    threads = threads_parent()

    def command(name, **kw):
        # every subcommand takes --threads N|all
        return sub.add_parser(name, parents=[threads], **kw)

    from linac_gen.cli import assist as assist_cmd
    from linac_gen.cli import backtrack as backtrack_cmd
    from linac_gen.cli import batch as batch_cmd
    from linac_gen.cli import run as run_cmd
    from linac_gen.cli import study as study_cmd
    from linac_gen.cli import scan as scan_cmd
    from linac_gen.cli import twiss as twiss_cmd
    from linac_gen.cli import multiobjective as mo_cmd
    from linac_gen.cli import failures as fail_cmd
    from linac_gen.cli import export as export_cmd
    from linac_gen.cli import orm as orm_cmd
    from linac_gen.cli import tracewin_ini as twini_cmd
    from linac_gen.cli import reliability as rel_cmd
    from linac_gen.cli import ring as ring_cmd

    run_cmd.add_arguments(command(
        "run", help="run one headless simulation"))
    scan_cmd.add_arguments(command(
        "scan", help="native parameter scan → CSV summary"))
    batch_cmd.add_arguments(command(
        "batch", help="multi-run campaign from a JSON job file"))
    study_cmd.add_arguments(command(
        "study", help="parameter study — per-run folders, resume, "
                      "oat/zip/grid/random/lhs strategies"))
    twiss_cmd.add_arguments(command(
        "twiss", help="matched Twiss — whole-lattice or FODO-cell input match"))
    mo_cmd.add_arguments(command(
        "mo", help="multi-objective design — Pareto front over ADJUST knobs"))
    fail_cmd.add_arguments(command(
        "failures", help="element failure impact + recovery → CSV"))
    backtrack_cmd.add_arguments(command(
        "backtrack", help="backward tracking — reconstruct an upstream "
                          "distribution from a downstream state"))
    assist_cmd.add_arguments(command(
        "assist", help="AI assistant chat (OPTIONAL — local or cloud "
                       "LLM; the rest of HELIX never needs it)"))
    export_cmd.add_arguments(command(
        "export", help="write the lattice in another code's format "
                       "(MAD-X sequence)"))
    orm_cmd.add_arguments(command(
        "orm", help="orbit-response matrix — compare a measurement with the "
                    "model, LOCO-style quad/trim/BPM calibration, deck export"))
    twini_cmd.add_arguments(command(
        "twini", help="TraceWin project options file (.ini) — converted "
                      "beam, decode report, or a .lgproj for the deck"))
    rel_cmd.add_arguments(command(
        "reliability", help="Reliability Study mode — faults + compensation, imperfections, "
                            "foil, availability as one resumable campaign with a report; "
                            "job export/import; self-test"))
    ring_cmd.add_arguments(command(
        "ring", help="ring mode — `ring optics`: tunes, chromaticity, momentum "
                     "compaction and periodic Twiss of a ring; `ring run`: turns of a bunch, "
                     "at fixed energy or through a ramp; `ring ramp`: the reference "
                     "particle through a ramp"))
    sub.add_parser("match", add_help=False,
                   help="run the matcher (delegates to linac_gen.matching)")

    args = parser.parse_args(argv)
    apply_threads_arg(args)
    if args.command == "run":
        return run_cmd.run(args)
    if args.command == "scan":
        return scan_cmd.run(args)
    if args.command == "batch":
        return batch_cmd.run(args)
    if args.command == "study":
        return study_cmd.run(args)
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
    if args.command == "export":
        return export_cmd.run(args)
    if args.command == "orm":
        return orm_cmd.run(args)
    if args.command == "twini":
        return twini_cmd.run(args)
    if args.command == "reliability":
        return rel_cmd.run(args)
    if args.command == "ring":
        return ring_cmd.run(args)
    return 2


if __name__ == "__main__":
    sys.exit(main())
