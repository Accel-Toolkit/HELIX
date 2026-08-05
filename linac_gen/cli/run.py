"""``python -m linac_gen run`` — one headless tracking simulation.

Loads a ``.lgproj`` project or a bare lattice file, applies scalar and
element-parameter overrides, runs an envelope / multi-particle / matrix
simulation, and writes the results.
"""
from __future__ import annotations

import sys
from pathlib import Path

from linac_gen.cli import common

_EXT = {"hdf5": ".h5", "openpmd": ".opmd.h5", "partran": ".txt"}


def add_arguments(p) -> None:
    """Populate the ``run`` sub-parser."""
    p.add_argument("input", help="a .lgproj project or a .dat/.madx lattice")
    p.add_argument("--mode", choices=("envelope", "mp", "matrix"),
                   default="envelope", help="solver mode (default envelope)")
    p.add_argument("--out", default=".", help="output directory (default .)")
    p.add_argument("--format", choices=("hdf5", "openpmd", "partran"),
                   default="hdf5", help="results format (default hdf5)")
    p.add_argument("--write-dst", action="store_true",
                   help="also write the final distribution as a .dst (mp only)")
    # common beam overrides
    p.add_argument("--energy", type=float, help="kinetic energy (MeV)")
    p.add_argument("--current", type=float, help="beam current (mA)")
    p.add_argument("--freq", type=float, help="beam frequency (MHz)")
    p.add_argument("--n-particles", type=int, dest="n_particles",
                   help="number of macroparticles")
    p.add_argument("--species", help="proton / deuteron / H-")
    p.add_argument("--beam", action="append", default=[], metavar="NAME=VALUE",
                   help="generic beam-config override (repeatable)")
    # element + convergence overrides
    p.add_argument("--set", action="append", default=[], dest="set_",
                   metavar="ELEM.attr=VALUE",
                   help="element-parameter override (repeatable)")
    p.add_argument("--nx", type=int, help="PIC grid size (mp)")
    p.add_argument("--grid-extent", type=float, dest="grid_extent",
                   help="PIC grid extent in sigma (mp)")
    p.add_argument("--step1", type=float, help="integration steps / metre")
    p.add_argument("--step2", type=float, help="space-charge kicks / metre")
    p.add_argument("--kernel", help="PIC deposit kernel (cic / tsc)")
    p.add_argument("--backend", help="compute backend (auto / cpu / gpu)")
    p.add_argument("--fieldmap-sampling", choices=("kernel", "scipy"),
                   default=None, dest="fieldmap_sampling",
                   help="3-D field-map sampling path: fused C++ kernel "
                        "(default when built) or the legacy scipy "
                        "interpolator — bitwise-identical results, "
                        "scipy is slower (verification/debugging)")
    p.add_argument("--sc", action="append", default=[], metavar="NAME=VALUE",
                   help="generic SpaceChargeConfig override (repeatable)")
    p.add_argument("--env-solver", choices=("matrix", "sacherer"),
                   default="matrix", dest="env_solver",
                   help="envelope solver kind (default matrix)")
    p.add_argument("--seed", type=int, default=42, help="RNG seed")
    p.add_argument("--fail-under-transmission", type=float, default=None,
                   dest="fail_under",
                   help="exit non-zero if final transmission < this %%")
    p.add_argument("-q", "--quiet", action="store_true",
                   help="suppress the run report")


def _beam_overrides(args) -> dict:
    d: dict = {}
    for name, val in (("energy", args.energy), ("current", args.current),
                      ("frequency", args.freq),
                      ("n_particles", args.n_particles),
                      ("species", args.species)):
        if val is not None:
            d[name] = val
    d.update(common.parse_assignments(args.beam))
    return d


def _cli_overrides(args) -> dict:
    return {
        "nx": args.nx, "grid_extent": args.grid_extent,
        "step1": args.step1, "step2": args.step2,
        "kernel": args.kernel, "backend": args.backend,
        "fieldmap_sampling": args.fieldmap_sampling,
        "sc": common.parse_assignments(args.sc),
    }


def run(args) -> int:
    """Execute the ``run`` subcommand; return a process exit code."""
    if not Path(args.input).is_file():
        print(f"error: input not found: {args.input}", file=sys.stderr)
        return 2
    try:
        lattice, beam_cfg, conv = common.load_input(args.input)
        common.apply_beam_overrides(beam_cfg, _beam_overrides(args))
        for item in args.set_:
            if "=" not in item:
                raise ValueError(f"--set expects ELEM.attr=VALUE, got '{item}'")
            sel, val = item.split("=", 1)
            common.apply_element_override(lattice, sel.strip(), val.strip())
        cli = _cli_overrides(args)
        common.apply_fieldmap_settings(conv, cli)
    except (ValueError, KeyError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    out_dir = Path(args.out)
    stem = Path(args.input).stem
    sc = None                       # bound only on the mp branch below
    try:
        if args.mode == "matrix":
            return _run_matrix(args, lattice, beam_cfg, out_dir, stem)
        if args.mode == "envelope":
            results = common.run_envelope_sim(lattice, beam_cfg,
                                              env_solver=args.env_solver)
            final_beam = None
        else:  # mp
            step_cfg = common.make_step_config(conv, cli)
            sc = common.make_sc_config(beam_cfg, conv, cli)
            results, final_beam = common.run_mp_sim(
                lattice, beam_cfg, sc, step_cfg, seed=args.seed)
    except Exception as exc:                                # noqa: BLE001
        print(f"error: simulation failed: {exc}", file=sys.stderr)
        return 1

    out_path = out_dir / f"{stem}_results{_EXT[args.format]}"
    written = [common.write_results(results, out_path, args.format,
                                    beam_cfg, lattice,
                                    lattice_path=args.input,
                                    seed=args.seed,
                                    sc_config=sc)]
    if args.write_dst:
        if final_beam is None:
            print("warning: --write-dst ignored (no particles in "
                  f"{args.mode} mode)", file=sys.stderr)
        else:
            written.append(common.write_final_dst(
                final_beam, out_dir / f"{stem}_final.dst"))

    summary = common.result_summary(results)
    if not args.quiet:
        _report(args, summary, written)

    trans = summary.get("transmission")
    if args.fail_under is not None and trans is not None \
            and trans < args.fail_under:
        print(f"FAIL: final transmission {trans:.2f}% < "
              f"{args.fail_under:.2f}%", file=sys.stderr)
        return 1
    return 0


def _run_matrix(args, lattice, beam_cfg, out_dir, stem) -> int:
    import numpy as np
    M, twiss = common.run_matrix(lattice, beam_cfg)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{stem}_matrix.txt"
    lines = ["# HELIX linear transfer matrix (6x6)"]
    for row in np.asarray(M):
        lines.append("  ".join(f"{v: .16e}" for v in row))
    lines.append("")
    for plane in ("x", "y"):
        tw = twiss[plane]
        if "error" in tw:
            lines.append(f"# Twiss {plane}: {tw['error']}")
        else:
            lines.append(f"# Twiss {plane}: alpha={tw['alpha']:.6f}  "
                         f"beta={tw['beta']:.6f}  mu={tw['mu']:.4f} deg")
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    if not args.quiet:
        print("\n".join(lines))
        print(f"\n[run] wrote {out_path}")
    return 0


def _report(args, summary: dict, written: list) -> None:
    print(f"[run] mode={args.mode}  input={args.input}")
    for k in ("transmission", "sigma_x", "sigma_y", "emit_x", "emit_y",
              "emit_z", "ref_w_kin"):
        v = summary.get(k)
        if v is not None:
            print(f"  {k:<14}= {v:.6g}")
    for w in written:
        print(f"[run] wrote {w}")
