"""``python -m linac_gen failures`` — element failure impact + recovery → CSV.

Sweeps element failures (single / pairs / custom sets) of a chosen type and
mode (off / cavity detune / magnet partial), ranks them by beam impact, and
optionally re-tunes neighbouring elements to recover the beam (compensation).

    python -m linac_gen failures lattice.dat \
        --types cavity,quad --mode off --combination single \
        --forward mp --workers 8 --energy 2.5 --current 5 --freq 162.5 \
        --compensate --strategy k_out_of_n --k 2 --out failures.csv
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

from linac_gen.cli import common


def add_arguments(p) -> None:
    p.add_argument("input", nargs="?", default=None,
                   help="a .lgproj project or a .dat/.madx lattice")
    p.add_argument("--types", default="cavity,quad,solenoid,dipole",
                   help="comma list of element types to fail "
                        "(cavity,quad,solenoid,dipole)")
    p.add_argument("--list-elements", action="store_true", dest="list_elements",
                   help="print the failable elements and exit")
    p.add_argument("--mode", choices=("off", "detune", "partial"),
                   default="off", help="failure mode applied to each element")
    p.add_argument("--mode-amp", type=float, default=1.0, dest="mode_amp",
                   help="amplitude fraction for detune/partial (e.g. 0.9)")
    p.add_argument("--mode-phase", type=float, default=0.0, dest="mode_phase",
                   help="phase offset [deg] for cavity detune")
    p.add_argument("--combination", choices=("single", "pairs", "custom"),
                   default="single", help="failure-set enumeration")
    p.add_argument("--custom-set", action="append", dest="custom_sets",
                   default=None, metavar="N1,N2,...",
                   help="comma-list of element names failing together "
                        "(repeatable; with --combination custom)")
    p.add_argument("--forward", choices=("envelope", "mp"), default="mp",
                   dest="forward", help="forward-pass model for the sweep")
    p.add_argument("--env-solver", choices=("matrix", "sacherer"),
                   default="matrix", dest="env_solver")
    p.add_argument("--workers", type=int, default=None,
                   help="process-pool size (default: all cores; 1 = serial)")
    # compensation
    p.add_argument("--compensate", action="store_true",
                   help="re-tune neighbours to recover the worst scenarios")
    p.add_argument("--strategy", default="k_out_of_n",
                   choices=("k_out_of_n", "l_neighboring_lattices", "manual"))
    p.add_argument("--k", type=int, default=2)
    p.add_argument("--l", type=int, default=1)
    p.add_argument("--comp-names", default=None, dest="comp_names",
                   help="manual compensators (comma list, --strategy manual)")
    p.add_argument("--comp-algorithm", default="cmaes", dest="comp_algorithm")
    p.add_argument("--comp-cost-solver", choices=("envelope", "mp"),
                   default="envelope", dest="comp_cost_solver")
    p.add_argument("--top", type=int, default=5,
                   help="compensate the top-N scenarios by criticality")
    p.add_argument("--out", default="failures.csv", help="output CSV path")
    # beam overrides
    p.add_argument("--energy", type=float)
    p.add_argument("--current", type=float)
    p.add_argument("--freq", type=float)
    p.add_argument("--species")


def run(args) -> int:
    from linac_gen.core.config import BeamConfig
    from linac_gen.failures import (CompensationConfig, FailureKind,
                                    FailureStudy, compensate, enumerate_scenarios,
                                    failable_elements)

    if not args.input or not Path(args.input).is_file():
        print(f"error: input not found: {args.input}", file=sys.stderr)
        return 2

    try:
        lattice, beam_cfg, _conv = common.load_input(args.input)
    except (ValueError, KeyError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    types = {t.strip() for t in args.types.split(",") if t.strip()}

    if args.list_elements:
        for n, lbl, cls in failable_elements(lattice, types):
            print(f"  {n:<20s} {lbl:<10s} {cls}")
        return 0

    beam_overrides = {}
    if args.energy is not None:
        beam_overrides["energy"] = args.energy
    if args.freq is not None:
        beam_overrides["frequency"] = args.freq
    if args.current is not None:
        beam_overrides["current"] = args.current
    if args.species is not None:
        beam_overrides["species"] = args.species

    kind = FailureKind(args.mode)
    custom = ([s.split(",") for s in args.custom_sets]
              if args.custom_sets else None)
    scenarios, name_to_class, names = enumerate_scenarios(
        lattice, types=types, kind=kind, combination=args.combination,
        amp_scale=args.mode_amp, phase_deg=args.mode_phase, custom_sets=custom)
    if not scenarios:
        print("error: no failable elements match the selection",
              file=sys.stderr)
        return 1

    print(f"[failures] {len(scenarios)} scenario(s) over {len(names)} "
          f"element(s); mode={args.mode}; forward={args.forward}")
    study = FailureStudy(args.input, beam_overrides=beam_overrides,
                         mode=args.forward, env_solver=args.env_solver)
    results = study.run(scenarios, names, name_to_class,
                        combination=args.combination,
                        max_workers=(1 if args.workers == 1 else args.workers),
                        serial=(args.workers == 1))
    print(f"[failures] baseline transmission="
          f"{results.baseline.get('transmission')}, "
          f"ref_w_kin={results.baseline.get('ref_w_kin')}")

    # optional compensation of the worst scenarios
    comp_by_idx: dict[int, object] = {}
    if args.compensate:
        cfg = CompensationConfig(
            strategy=args.strategy, k=args.k, l=args.l,
            manual_names=(args.comp_names.split(",") if args.comp_names else None),
            algorithm=args.comp_algorithm, cost_solver=args.comp_cost_solver)
        cbeam = BeamConfig()
        common.apply_beam_overrides(cbeam, beam_overrides)
        for rank_i in results.ranking[:max(0, args.top)]:
            im = results.impacts[rank_i]
            print(f"[failures] compensating: {im.label} …")
            cr = compensate(lattice, cbeam, im.scenario, name_to_class,
                            results.baseline, cfg)
            comp_by_idx[rank_i] = cr
            print(f"           recovered={cr.recovered} "
                  f"compensators={cr.compensator_names}")

    # write CSV
    rows = results.to_rows()
    extra = ["recovered", "compensators", "settings"] if args.compensate else []
    with open(args.out, "w", newline="") as fh:
        w = csv.writer(fh)
        header = list(rows[0].keys()) + extra if rows else []
        w.writerow(header)
        for i, row in enumerate(rows):
            vals = list(row.values())
            if args.compensate:
                cr = comp_by_idx.get(i)
                if cr is not None:
                    vals += [cr.recovered, "|".join(cr.compensator_names),
                             ";".join(f"{k}={v:.4g}" for k, v in cr.settings.items())]
                else:
                    vals += ["", "", ""]
            w.writerow(vals)
    print(f"[failures] wrote {args.out}")
    return 0
