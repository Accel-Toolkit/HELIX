"""``python -m linac_gen twini`` — TraceWin project options file (``.ini``) → HELIX beam / ``.lgproj``.

    python -m linac_gen twini PROJECT.ini                      # the converted beam + warnings
    python -m linac_gen twini PROJECT.ini --report             # every decoded slot: applied / recorded / unknown / not decoded
    python -m linac_gen twini PROJECT.ini --json               # machine-readable
    python -m linac_gen twini PROJECT.ini --lgproj [OUT.lgproj] [--lattice DECK.dat] [--beam 2] [--species H-] [--force]

The deck is ``<project>.dat`` next to the ``.ini`` unless ``--lattice`` says
otherwise; it is parsed for the FREQ check and recorded in the project file.
Manual: docs/manual/06_running/02_tracewin_dat.md#importing-tracewin-project-settings-ini
"""
from __future__ import annotations

import json
import sys
from dataclasses import asdict
from pathlib import Path

from linac_gen.cli import common


def add_arguments(p) -> None:
    p.add_argument("ini", help="TraceWin project options file (<project>.ini)")
    p.add_argument("--beam", type=int, choices=(1, 2), default=1,
                   help="which of TraceWin's two input beams (default 1)")
    p.add_argument("--species", choices=("proton", "deuteron", "H-"),
                   help="override the particle-table match (needed for a "
                        "user-defined particle row)")
    p.add_argument("--lattice", metavar="DECK",
                   help="the deck the .ini belongs to (default: <project>.dat "
                        "next to the .ini); parsed for the FREQ check and "
                        "recorded in the project file")
    p.add_argument("--lgproj", nargs="?", const="auto", metavar="OUT",
                   help="write a HELIX project file (default <project>.lgproj "
                        "next to the .ini)")
    p.add_argument("--force", action="store_true",
                   help="overwrite an existing --lgproj file")
    p.add_argument("--report", action="store_true",
                   help="print the full decode report (applied, identified "
                        "but not applied, unidentified slots, not decoded)")
    p.add_argument("--json", action="store_true",
                   help="print the converted beam, warnings and recorded "
                        "settings as JSON")
    p.add_argument("-q", "--quiet", action="store_true",
                   help="print only warnings and errors")


def run(args) -> int:
    try:
        return _run(args)
    except (ValueError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


def _summary_lines(ini, cfg, beam: int) -> list[str]:
    d = ini.identified_not_applied()
    lines = [
        f"{ini.name} ({ini.version}) — beam {beam}: {cfg.species}, "
        f"{cfg.energy:g} MeV, {cfg.frequency:g} MHz, {cfg.current:g} mA, "
        f"{cfg.n_particles} particles",
        f"  emit_nx {cfg.emit_nx:.6g} pi mm mrad  alpha_x {cfg.alpha_x:.6g}  "
        f"beta_x {cfg.beta_x:.6g} m",
        f"  emit_ny {cfg.emit_ny:.6g} pi mm mrad  alpha_y {cfg.alpha_y:.6g}  "
        f"beta_y {cfg.beta_y:.6g} m",
    ]
    if cfg.continuous:
        lines.append("  longitudinal: DC beam (continuous = True)")
    else:
        lines.append(
            f"  emit_z {cfg.emit_z:.6g} pi deg MeV  alpha_z {cfg.alpha_z:.6g} "
            f"(HELIX sign)  beta_z {cfg.beta_z:.6g} deg/MeV")
    lines.append(
        f"  recorded, not applied: nbr_thread {d['nbr_thread']}, PICNIC r/z "
        f"{d['picnic_r_mesh']}x{d['picnic_z_mesh']}, xy "
        f"{d['picnic_xy_mesh_x']}x{d['picnic_xy_mesh_y']}")
    return lines


def _run(args) -> int:
    from linac_gen.io.tracewin_ini import (load_tracewin_ini, report,
                                           to_beam_config, to_project_extras)
    ini_path = Path(args.ini)
    if not ini_path.is_file():
        print(f"error: input not found: {args.ini}", file=sys.stderr)
        return 2
    ini = load_tracewin_ini(ini_path)

    lattice_path = None
    if args.lattice:
        lattice_path = Path(args.lattice)
        if not lattice_path.is_file():
            print(f"error: lattice not found: {args.lattice}", file=sys.stderr)
            return 2
    else:
        cand = ini_path.with_suffix(".dat")
        if cand.is_file():
            lattice_path = cand
    lat = None
    if lattice_path is not None:
        try:
            lat = common.load_lattice(str(lattice_path))
        except Exception as exc:   # the project is still convertible
            print(f"note: {lattice_path.name} could not be parsed for the "
                  f"FREQ check: {exc}", file=sys.stderr)

    cfg, warns = to_beam_config(ini, beam=args.beam, species=args.species,
                                lattice=lat)
    all_warns = list(ini.warnings) + list(warns)

    if args.json:
        print(json.dumps({
            "file": str(ini_path), "layout": ini.version, "beam_index": args.beam,
            "beam": asdict(cfg), "warnings": all_warns,
            "recorded": to_project_extras(ini)["tracewin_ini"],
            "unknown_slots": ini.unknown,
        }, indent=2))
    elif args.report:
        print(report(ini, beam=args.beam, species=args.species, lattice=lat))
    elif not args.quiet:
        print("\n".join(_summary_lines(ini, cfg, args.beam)))
    if not args.report:
        for w in all_warns:
            print(f"warning: {w}", file=sys.stderr)

    if args.lgproj is not None:
        if lattice_path is None:
            print(f"error: no {ini_path.stem}.dat next to {ini_path.name} — "
                  f"pass --lattice DECK to write a project file", file=sys.stderr)
            return 2
        out = ini_path.with_suffix(".lgproj") if args.lgproj == "auto" else Path(args.lgproj)
        if out.is_dir():
            print(f"error: {out} is a directory — pass the project file name",
                  file=sys.stderr)
            return 2
        if out.exists() and not args.force:
            print(f"error: {out} exists (use --force to overwrite)", file=sys.stderr)
            return 2
        from linac_gen.io.project import write_project
        write_project(out, lattice_path=lattice_path, beam=cfg,
                      convergence=to_project_extras(ini), overwrite=args.force)
        if not args.quiet:
            print(f"wrote {out} (lattice {lattice_path.name}, beam {args.beam} "
                  f"of {ini_path.name})")
    return 0
