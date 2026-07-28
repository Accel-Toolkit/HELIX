"""``python -m linac_gen twiss`` — matched Twiss of a lattice or FODO cell.

Two modes:

* ``--mode whole`` — the whole-lattice periodic Twiss (correct for a true
  periodic ring);
* ``--mode cell``  — the periodic Twiss of one FODO cell back-propagated to
  the lattice entrance — the *input matched Twiss* for a one-pass transfer
  line (default).

A bare ``.dat`` carries no beam energy; pass ``--energy`` (and usually
``--species`` / ``--freq``) or use a ``.lgproj`` project.
"""
from __future__ import annotations

import sys
from pathlib import Path

from linac_gen.cli import common


def add_arguments(p) -> None:
    """Populate the ``twiss`` sub-parser."""
    p.add_argument("input", help="a .lgproj project or a .dat/.madx lattice")
    p.add_argument("--mode", choices=("whole", "cell"), default="cell",
                   help="'whole' = whole-lattice periodic Twiss (a ring); "
                        "'cell' = FODO-cell match back-propagated to the "
                        "entrance, the input match for a transfer line "
                        "(default cell)")
    p.add_argument("--cell", type=int, default=0,
                   help="which detected FODO cell to use, 0-based "
                        "(cell mode; default 0 = first)")
    p.add_argument("--list-cells", action="store_true", dest="list_cells",
                   help="list the detected FODO cells and exit")
    p.add_argument("--cell-start", type=int, default=None, dest="cell_start",
                   help="manually specify the inclusive start index of "
                        "the periodic cell (overrides --cell auto-pick).  "
                        "Use with --cell-end when the auto-detection "
                        "doesn't find the period you want.")
    p.add_argument("--cell-end", type=int, default=None, dest="cell_end",
                   help="manually specify the inclusive end index of the "
                        "periodic cell.  Required if --cell-start is set.")
    p.add_argument("--energy", type=float, help="kinetic energy (MeV)")
    p.add_argument("--freq", type=float, help="beam frequency (MHz)")
    p.add_argument("--species", help="proton / deuteron / H-")
    p.add_argument("-q", "--quiet", action="store_true",
                   help="print only the four Twiss numbers "
                        "(alpha_x beta_x alpha_y beta_y)")
    p.add_argument("--disp", action="store_true",
                   help="with -q: append the four periodic-dispersion "
                        "numbers (disp_x disp_xp disp_y disp_yp, in mm/MeV "
                        "and mrad/MeV) after the Twiss.  Without -q this "
                        "is a no-op — dispersion is already printed when "
                        "nonzero.")


def run(args) -> int:
    """Execute the ``twiss`` subcommand; return a process exit code."""
    if not Path(args.input).is_file():
        print(f"error: input not found: {args.input}", file=sys.stderr)
        return 2
    try:
        lattice, beam_cfg, _conv = common.load_input(args.input)
        if args.energy is not None:
            beam_cfg.energy = args.energy
        if args.freq is not None:
            beam_cfg.frequency = args.freq
        if args.species is not None:
            beam_cfg.species = args.species
        ref = common.build_ref(beam_cfg)
    except (ValueError, KeyError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    from linac_gen.matching.periodic import (
        find_fodo_cells, find_matched_input_twiss, find_periodic_twiss,
    )

    cells = find_fodo_cells(lattice)

    if args.list_cells:
        if not cells:
            print("no FODO cells detected (need at least 3 quadrupoles)")
            return 0
        print(f"{len(cells)} FODO-cell candidate(s):")
        for k, (cs, ce) in enumerate(cells):
            print(f"  [{k:3d}]  elements {cs}..{ce}")
        return 0

    try:
        if args.mode == "whole":
            tw = find_periodic_twiss(lattice, ref)
            label = "whole-lattice periodic Twiss (ring)"
        else:
            # Resolve cell bounds: manual override wins if both set,
            # else fall back to the n-th auto-detected cell.
            if args.cell_start is not None or args.cell_end is not None:
                if args.cell_start is None or args.cell_end is None:
                    print("error: --cell-start and --cell-end must be "
                          "specified together", file=sys.stderr)
                    return 2
                cs, ce = int(args.cell_start), int(args.cell_end)
                label_prefix = (f"matched input Twiss — manual cell "
                                f"(elements {cs}..{ce})")
            else:
                if not cells:
                    print("error: no FODO cell detected — use --mode whole, "
                          "specify bounds via --cell-start/--cell-end, or "
                          "use a lattice with at least 3 focusing elements "
                          "(quads, solenoids, or solenoid FieldMaps)",
                          file=sys.stderr)
                    return 2
                if not (0 <= args.cell < len(cells)):
                    print(f"error: --cell {args.cell} out of range "
                          f"(0..{len(cells) - 1}); see --list-cells",
                          file=sys.stderr)
                    return 2
                cs, ce = cells[args.cell]
                label_prefix = (f"matched input Twiss — FODO cell "
                                f"{args.cell} (elements {cs}..{ce})")
            tw = find_matched_input_twiss(lattice, ref, cs, ce)
            label = f"{label_prefix} back-propagated to s=0"
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    import math
    dvals = [float(tw.get(k, 0.0))
             for k in ("disp_x", "disp_xp", "disp_y", "disp_yp")]

    if args.quiet:
        line = (f"{tw['alpha_x']:.8e} {tw['beta_x']:.8e} "
                f"{tw['alpha_y']:.8e} {tw['beta_y']:.8e}")
        if args.disp:
            line += " " + " ".join(f"{v:.8e}" for v in dvals)
        print(line)
    else:
        def _mu_str(plane: str) -> str:
            """Principal [0,180°] value; append the oriented branch when
            it differs (μ > 180° cells) so neither convention is lost."""
            mu = float(tw[f"mu_{plane}"])
            folded = float(tw.get(f"mu_{plane}_folded", mu))
            if abs(folded - mu) > 1e-9:
                return f"mu_{plane} = {folded:.4f} deg (oriented {mu:.4f})"
            return f"mu_{plane} = {mu:.4f} deg"

        print(f"[twiss] {label}")
        print(f"  alpha_x = {tw['alpha_x']:+.6f}    "
              f"beta_x = {tw['beta_x']:.6f} m    {_mu_str('x')}")
        print(f"  alpha_y = {tw['alpha_y']:+.6f}    "
              f"beta_y = {tw['beta_y']:.6f} m    {_mu_str('y')}")
        # Periodic dispersion — printed only when it exists (bending
        # lattice) or is undefined, so straight-lattice output is
        # byte-identical to the pre-dispersion CLI.
        if any(v != 0.0 for v in dvals) or any(math.isnan(v) for v in dvals):
            if any(math.isnan(v) for v in dvals):
                print("  dispersion: undefined (near-integer tune)")
            else:
                # η[m] = D[mm/MeV] × β²γ·mc²[MeV] × 1e-3 (entrance ref).
                conv = ref.beta ** 2 * ref.gamma * ref.species.mass * 1e-3
                print(f"  D_x  = {dvals[0]:+.6f} mm/MeV   "
                      f"D_x' = {dvals[1]:+.6f} mrad/MeV   "
                      f"(eta_x = {dvals[0] * conv:+.4f} m)")
                print(f"  D_y  = {dvals[2]:+.6f} mm/MeV   "
                      f"D_y' = {dvals[3]:+.6f} mrad/MeV   "
                      f"(eta_y = {dvals[2] * conv:+.4f} m)")
        if tw.get("coupled"):
            print(f"  COUPLED (eigenvector method) — "
                  f"normal-mode tunes mu_1 = {tw.get('mu_1', tw['mu_x']):.4f} deg, "
                  f"mu_2 = {tw.get('mu_2', tw['mu_y']):.4f} deg")
            print(f"  per-plane alpha/beta above are projections of the "
                  f"matched 4×4 Σ")
    return 0
