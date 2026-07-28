"""``python -m linac_gen scan`` — a native parameter scan.

Sweeps one or more variables (a beam field, a PIC grid / step parameter,
or an element parameter) and writes a CSV summary — one row per point.
Reuses HELIX's process-pool scan engine (``linac_gen/parallel``).
"""
from __future__ import annotations

import csv
import itertools
import sys
from pathlib import Path

from linac_gen.cli import common

_STRUCTURAL = ("nx", "grid_extent", "step1", "step2")
_METRICS = ("sigma_x", "sigma_y", "sigma_phi", "sigma_w", "emit_x", "emit_y",
            "emit_z", "transmission", "ref_w_kin", "x_max", "y_max",
            "elapsed")


def add_arguments(p) -> None:
    """Populate the ``scan`` sub-parser."""
    p.add_argument("input", help="a .lgproj project or a .dat/.madx lattice")
    p.add_argument("--vary", action="append", default=[], required=True,
                   metavar="VAR=start:stop:step",
                   help="variable to sweep (repeatable → Cartesian product); "
                        "VAR is a beam field, nx/grid_extent/step1/step2, or "
                        "an element param ELEM.attr.  Also VAR=v1,v2,v3")
    p.add_argument("--out", default="scan.csv", help="CSV output (default scan.csv)")
    p.add_argument("--mode", choices=("envelope", "mp"), default="envelope",
                   help="solver mode (default envelope)")
    p.add_argument("--parallel", type=int, default=1,
                   help="worker processes (>1 → process pool)")
    # fixed overrides applied to every point
    p.add_argument("--energy", type=float, help="kinetic energy (MeV)")
    p.add_argument("--current", type=float, help="beam current (mA)")
    p.add_argument("--freq", type=float, help="beam frequency (MHz)")
    p.add_argument("--n-particles", type=int, dest="n_particles",
                   help="number of macroparticles")
    p.add_argument("--species", help="proton / deuteron / H-")
    p.add_argument("--beam", action="append", default=[], metavar="NAME=VALUE",
                   help="fixed beam-config override (repeatable)")
    p.add_argument("--set", action="append", default=[], dest="set_",
                   metavar="ELEM.attr=VALUE",
                   help="fixed element-parameter override (repeatable)")
    p.add_argument("--nx", type=int, help="fixed PIC grid size")
    p.add_argument("--grid-extent", type=float, dest="grid_extent",
                   help="fixed PIC grid extent (sigma)")
    p.add_argument("--step1", type=float, help="fixed integration steps/m")
    p.add_argument("--step2", type=float, help="fixed space-charge kicks/m")
    p.add_argument("--backend", help="compute backend (auto / cpu / gpu)")
    p.add_argument("--sc", action="append", default=[], metavar="NAME=VALUE",
                   help="fixed SpaceChargeConfig override (repeatable)")
    p.add_argument("--env-solver", choices=("matrix", "sacherer"),
                   default="matrix", dest="env_solver")
    p.add_argument("--seed", type=int, default=42, help="RNG seed")
    p.add_argument("-q", "--quiet", action="store_true")


def _parse_vary(spec: str):
    """Parse one ``VAR=range`` spec → ``(var, [values])``."""
    if "=" not in spec:
        raise ValueError(f"--vary expects VAR=range, got '{spec}'")
    var, rng = spec.split("=", 1)
    var, rng = var.strip(), rng.strip()
    if ":" in rng:
        parts = rng.split(":")
        if len(parts) != 3:
            raise ValueError(f"--vary range '{rng}' must be start:stop:step")
        a, b, c = (float(x) for x in parts)
        if c == 0:
            raise ValueError("--vary step cannot be zero")
        n = int(round((b - a) / c)) + 1
        vals = [a + i * c for i in range(max(n, 1))]
    else:
        vals = [float(x) for x in rng.split(",") if x.strip()]
    if not vals:
        raise ValueError(f"--vary '{spec}' produced no values")
    return var, vals


def _kind(var: str) -> str:
    if "." in var or var.startswith("@"):
        return "element"
    if var in _STRUCTURAL:
        return "structural"
    return "beam"


def _fixed_beam(args) -> dict:
    d: dict = {}
    for name, val in (("energy", args.energy), ("current", args.current),
                      ("frequency", args.freq),
                      ("n_particles", args.n_particles),
                      ("species", args.species)):
        if val is not None:
            d[name] = val
    d.update(common.parse_assignments(args.beam))
    return d


def run(args) -> int:
    """Execute the ``scan`` subcommand; return a process exit code."""
    if not Path(args.input).is_file():
        print(f"error: input not found: {args.input}", file=sys.stderr)
        return 2
    try:
        varied = [_parse_vary(s) for s in args.vary]
        var_kinds = [(v, _kind(v)) for v, _ in varied]
        fixed_beam = _fixed_beam(args)
        fixed_elem = []
        for item in args.set_:
            sel, _, val = item.partition("=")
            if not val:
                raise ValueError(f"--set expects ELEM.attr=VALUE, got '{item}'")
            fixed_elem.append((sel.strip(), val.strip()))
        fixed_sc = common.parse_assignments(args.sc)
        fixed_cli = {"nx": args.nx, "grid_extent": args.grid_extent,
                     "step1": args.step1, "step2": args.step2,
                     "backend": args.backend}
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    # Apply project-level field-map settings once in the parent.  The
    # sampling choice reaches spawned workers via the env var set inside
    # apply_fieldmap_settings; integrator/interp remain parent-only class
    # attributes (pre-existing limitation of the spawn pool).
    try:
        _lat, _beam, conv = common.load_input(args.input)
        common.apply_fieldmap_settings(conv, {})
    except Exception:                                  # noqa: BLE001
        pass                       # per-point load reports real errors

    combos = list(itertools.product(*[vals for _, vals in varied]))
    points = []
    for combo in combos:
        beam_ov = dict(fixed_beam)
        elem_ov = list(fixed_elem)
        cli_ov = dict(fixed_cli)
        for (var, kind), val in zip(var_kinds, combo):
            if kind == "beam":
                beam_ov[var] = val
            elif kind == "structural":
                cli_ov[var] = val
            else:
                elem_ov.append((var, val))
        try:
            points.append(common.build_scan_point(
                args.input, beam_overrides=beam_ov, element_overrides=elem_ov,
                sc_overrides=fixed_sc, mode=args.mode,
                env_solver=args.env_solver, seed=args.seed, cli=cli_ov))
        except (ValueError, KeyError) as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2

    if not args.quiet:
        print(f"[scan] {len(points)} point(s), mode={args.mode}, "
              f"parallel={args.parallel}")
    try:
        from linac_gen.parallel.scan_pool import (
            run_scan_points, run_scan_points_serial,
        )
        if args.parallel > 1:
            results = run_scan_points(points, max_workers=args.parallel)
        else:
            results = run_scan_points_serial(points)
    except Exception as exc:                                # noqa: BLE001
        print(f"error: scan failed: {exc}", file=sys.stderr)
        return 1

    vary_names = [v for v, _ in varied]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow([*vary_names, *_METRICS])
        for combo, res in zip(combos, results):
            row = list(combo)
            for m in _METRICS:
                v = res.get(m)
                row.append("" if v is None else f"{v:.8g}")
            w.writerow(row)
    if not args.quiet:
        print(f"[scan] wrote {out}  ({len(results)} rows)")
    return 0
