"""``python -m linac_gen twiss`` — matched Twiss of a lattice or FODO cell.

Two modes:

* ``--mode whole`` — the whole-lattice periodic Twiss (correct for a true
  periodic ring);
* ``--mode cell``  — the periodic Twiss of one FODO cell back-propagated to
  the lattice entrance — the *input matched Twiss* for a one-pass transfer
  line (default).  A deck with an RFQ is matched to the RFQ's first regular
  period instead, at the RFQ entrance (``find_rfq_matched_input_twiss``).

``--current`` matches with space charge (an RFQ for a DC input beam).

A bare ``.dat`` carries no beam energy; pass ``--energy`` (and usually
``--species`` / ``--freq``) or use a ``.lgproj`` project.
"""
from __future__ import annotations

import sys
from pathlib import Path

from linac_gen.cli import common


def add_arguments(p) -> None:
    """Populate the ``twiss`` sub-parser."""
    p.add_argument("input", help="a .lgproj project or a lattice file (.dat, .madx, .lat, .lte; .bmad/.jl/.pals.yaml via lattix)")
    common.add_tracewin_ini_argument(p)
    p.add_argument("--mode", choices=("whole", "cell"), default="cell",
                   help="'whole' = whole-lattice periodic Twiss (a ring); "
                        "'cell' = FODO-cell match back-propagated to the "
                        "entrance, the input match for a transfer line "
                        "(default cell)")
    p.add_argument("--cell", type=int, default=0,
                   help="which detected cell to use, 0-based: RFQ periods "
                        "first, then FODO cells (cell mode; default 0 = "
                        "first)")
    p.add_argument("--list-cells", action="store_true", dest="list_cells",
                   help="list the detected cells (RFQ periods and FODO "
                        "cells) and exit")
    p.add_argument("--cell-start", type=int, default=None, dest="cell_start",
                   help="manually specify the inclusive start index of "
                        "the periodic cell (overrides --cell auto-pick).  "
                        "Use with --cell-end when the auto-detection "
                        "doesn't find the period you want.")
    p.add_argument("--cell-end", type=int, default=None, dest="cell_end",
                   help="manually specify the inclusive end index of the "
                        "periodic cell.  Required if --cell-start is set.")
    p.add_argument("--current", type=float, default=None, metavar="MA",
                   help="match with space charge at this beam current (mA); "
                        "the emittances come from the project, "
                        "--tracewin-ini or --emit-nx/--emit-ny.  An RFQ is "
                        "matched for a DC input beam")
    p.add_argument("--emit-nx", type=float, default=None, dest="emit_nx",
                   help="normalised rms emittance in x (mm mrad) for "
                        "--current; overrides the input's beam")
    p.add_argument("--emit-ny", type=float, default=None, dest="emit_ny",
                   help="normalised rms emittance in y (mm mrad) for "
                        "--current; overrides the input's beam")
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
        lattice, beam_cfg, _conv = common.load_input(
            args.input, tracewin_ini=args.tracewin_ini)
        if args.energy is not None:
            beam_cfg.energy = args.energy
        if args.freq is not None:
            beam_cfg.frequency = args.freq
        if args.species is not None:
            beam_cfg.species = args.species
        common.note_tracewin_ini_overrides(
            args.tracewin_ini, {"energy": args.energy, "freq": args.freq,
                                "species": args.species})
        ref = common.build_ref(beam_cfg)
    except (ValueError, KeyError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    from linac_gen.matching.periodic import (
        find_fodo_cells, find_matched_input_twiss, find_periodic_twiss,
        find_rfq_matched_input_twiss, find_rfq_periods,
        find_sc_matched_input_twiss,
    )

    # Candidate cells: RFQ periods first (an RFQ is the first periodic
    # structure its beam meets; the solenoids of a LEBT ahead of it are no
    # period), then FODO cells.  Without an RFQ the list, its numbering
    # and every output line are exactly the FODO-cell ones.
    rfq_periods = find_rfq_periods(lattice)
    cells = ([("rfq", p) for p in rfq_periods]
             + [("fodo", c) for c in find_fodo_cells(lattice)])

    def _rfq_names(period):
        els = lattice.elements
        return (f"{els[period.start].name} + {els[period.end].name}",
                els[period.rfq_start].name)

    if args.list_cells:
        if not cells:
            print("no cell detected (needs an RFQ, or at least 3 focusing "
                  "elements: quads, solenoids or solenoid FieldMaps)")
            return 0
        if not rfq_periods:
            print(f"{len(cells)} FODO-cell candidate(s):")
            for k, (_kind, (cs, ce)) in enumerate(cells):
                print(f"  [{k:3d}]  elements {cs}..{ce}")
            return 0
        print(f"{len(cells)} cell candidate(s):")
        for k, (kind, c) in enumerate(cells):
            if kind == "rfq":
                pair, entry = _rfq_names(c)
                print(f"  [{k:3d}]  RFQ period {pair}, elements "
                      f"{c.start}..{c.end} (matched at the RFQ entrance, "
                      f"{entry}, element {c.rfq_start})")
            else:
                print(f"  [{k:3d}]  FODO cell, elements {c[0]}..{c[1]}")
        return 0

    # Space charge: --current, with the beam's emittances.
    current = args.current
    if current is not None and current < 0:
        print("error: --current must be >= 0", file=sys.stderr)
        return 2
    sc = current is not None and current > 0
    geo = None
    bare = (Path(args.input).suffix.lower() != ".lgproj"
            and not args.tracewin_ini)
    if sc:
        if args.mode == "whole":
            print("error: --current applies to cell mode (input matching)",
                  file=sys.stderr)
            return 2
        if args.emit_nx is not None:
            beam_cfg.emit_nx = args.emit_nx
        if args.emit_ny is not None:
            beam_cfg.emit_ny = args.emit_ny
        if bare and (args.emit_nx is None or args.emit_ny is None):
            print("error: space-charge matching needs the beam emittances: "
                  "pass a project (.lgproj), --tracewin-ini, or --emit-nx "
                  "and --emit-ny", file=sys.stderr)
            return 2
        from linac_gen.distributions.factory import geometric_emittances
        geo = geometric_emittances(beam_cfg, ref.bg)

    def _bunched_on_bare_deck() -> bool:
        """A bunched beam's space charge depends on its bunch length: a
        bare deck has none to give (BeamConfig's default would be used)."""
        if bare:
            print("error: space-charge matching of a bunched beam needs its "
                  "longitudinal emittance: pass a project (.lgproj) or "
                  "--tracewin-ini", file=sys.stderr)
        return bare

    def _seed(dc: bool) -> dict:
        """Fixed envelope inputs of find_sc_matched_input_twiss: a bunched
        beam as the GUI's cell matching builds it, or a DC beam."""
        if dc:
            return {"emit_x": geo[0], "emit_y": geo[1], "emit_z": 0.0,
                    "alpha_z": 0.0, "beta_z": 1.0, "continuous": True,
                    "dc_energy_spread_keV": beam_cfg.dc_energy_spread_keV}
        return {"alpha_z": 0.0, "beta_z": beam_cfg.beta_z,
                "emit_x": geo[0], "emit_y": geo[1], "emit_z": geo[2]}

    sc_tag = f"space-charge ({current:g} mA) " if sc else ""
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
                label_prefix = (f"{sc_tag}matched input Twiss — manual cell "
                                f"(elements {cs}..{ce})")
                if sc:
                    from linac_gen.rfq_td.chain import find_chains
                    # a range inside an RFQ sees a DC input beam (as the
                    # RFQ periods do), not the bunched seed
                    dc = any(a <= cs <= b
                             for a, b in find_chains(lattice.elements))
                    if not dc and _bunched_on_bare_deck():
                        return 2
                    # an RFQ period's Twiss sit at α ≈ 0, where the
                    # relative convergence test is strict
                    tw = find_sc_matched_input_twiss(
                        lattice, ref, cs, ce, current, _seed(dc),
                        max_iter=200 if dc else 50)
                else:
                    tw = find_matched_input_twiss(lattice, ref, cs, ce)
                label = f"{label_prefix} back-propagated to s=0"
            else:
                if not cells:
                    print("error: no cell detected — use --mode whole, "
                          "specify bounds via --cell-start/--cell-end, or "
                          "use a lattice with an RFQ or at least 3 focusing "
                          "elements (quads, solenoids, or solenoid "
                          "FieldMaps)", file=sys.stderr)
                    return 2
                if not (0 <= args.cell < len(cells)):
                    print(f"error: --cell {args.cell} out of range "
                          f"(0..{len(cells) - 1}); see --list-cells",
                          file=sys.stderr)
                    return 2
                kind, cell = cells[args.cell]
                if kind == "rfq":
                    tw = find_rfq_matched_input_twiss(
                        lattice, ref, cell, current=current if sc else 0.0,
                        emit_x=geo[0] if sc else None,
                        emit_y=geo[1] if sc else None,
                        dc_energy_spread_keV=beam_cfg.dc_energy_spread_keV)
                    pair, entry = _rfq_names(cell)
                    label = (f"{sc_tag}matched input Twiss at the RFQ "
                             f"entrance ({entry}, element {cell.rfq_start})"
                             f" — RFQ period {pair} (elements "
                             f"{cell.start}..{cell.end})"
                             + (", DC input beam" if sc else ""))
                else:
                    cs, ce = cell
                    label_prefix = (f"{sc_tag}matched input Twiss — FODO "
                                    f"cell {args.cell} (elements {cs}..{ce})")
                    if sc:
                        if _bunched_on_bare_deck():
                            return 2
                        tw = find_sc_matched_input_twiss(
                            lattice, ref, cs, ce, current, _seed(False))
                    else:
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
        if sc:
            print("  the phase advances are the zero-current values")
        if "rfq_start" in tw and not tw["at_lattice_entrance"]:
            print(f"  note: the RFQ starts {tw['upstream_mm'] / 1e3:.4f} m "
                  "into the lattice; these are the Twiss the line upstream "
                  "must deliver at the RFQ entrance, not the beam at s=0")
        if tw.get("caution"):
            print(f"  caution: {tw['caution']}")
    if args.quiet and tw.get("caution"):
        print(f"warning: {tw['caution']}", file=sys.stderr)
    if sc and not tw.get("converged", True):
        print("warning: the space-charge matching did not converge; the "
              "values are its last iterate", file=sys.stderr)
    return 0
