"""``python -m linac_gen export`` — write a HELIX lattice in another code's
format (currently MAD-X).

    python -m linac_gen export deck.dat out.madx --energy 2.1 --species H-

The exporter is the exact inverse of the MAD-X importer
(:mod:`linac_gen.io.madx_writer`).  A bare ``.dat`` carries no beam energy
and the magnetic rigidity fixes every ``k1``/``ks`` in the output, so
``--energy`` is required unless the input is a ``.lgproj`` project (whose
Beam settings are used; ``--energy`` / ``--species`` / ``--freq`` still
override them).

Elements MAD-X cannot represent (field maps, RFQ cells, multi-gap
cavities, foils, explicit matrices, electric steerers) become a ``MARKER``
plus a body ``DRIFT`` of the same length and a warning on stderr;
``--strict`` refuses the export instead.  Exit codes: 0 written,
1 export refused (``--strict``) or failed, 2 bad arguments / missing input.
"""
from __future__ import annotations

import sys
from pathlib import Path

from linac_gen.cli import common

FORMATS = ("madx",)
_SUFFIX_FORMAT = {".madx": "madx", ".seq": "madx"}


def add_arguments(p) -> None:
    """Populate the ``export`` sub-parser."""
    p.add_argument("input", help="a .lgproj project or a .dat/.madx/.lat/"
                                 ".lte/.bmad/.jl/.pals.yaml lattice")
    p.add_argument("output", help="output file (.madx / .seq)")
    p.add_argument("--format", choices=FORMATS, default=None,
                   help="output format (default: from the output suffix)")
    p.add_argument("--energy", type=float,
                   help="kinetic energy (MeV) — required for a bare lattice")
    p.add_argument("--freq", type=float, help="beam frequency (MHz)")
    p.add_argument("--species", help="proton / deuteron / H-")
    p.add_argument("--sequence", default=None,
                   help="MAD-X sequence name (default: the input stem)")
    p.add_argument("--title", default=None,
                   help="MAD-X TITLE text (default: the input file name)")
    p.add_argument("--strict", action="store_true",
                   help="refuse the export if any element has no MAD-X "
                        "equivalent (default: MARKER + body DRIFT + warning)")
    p.add_argument("--linearize", action="store_true",
                   help="export field maps / multi-gap cavities / RFQ cells "
                        "/ thin lenses / explicit matrices as MAD-X MATRIX "
                        "elements built from their first-order map (an "
                        "energy gain becomes a kick6 pt shift; warned)")
    p.add_argument("-q", "--quiet", action="store_true",
                   help="print nothing on success (warnings still go to "
                        "stderr)")


def run(args) -> int:
    """Execute the ``export`` subcommand; return a process exit code."""
    src = Path(args.input)
    if not src.is_file():
        print(f"error: input not found: {args.input}", file=sys.stderr)
        return 2
    out = Path(args.output)
    fmt = args.format or _SUFFIX_FORMAT.get(out.suffix.lower())
    if fmt is None:
        print(f"error: cannot infer the output format from {out.name!r}; "
              f"pass --format {'/'.join(FORMATS)}", file=sys.stderr)
        return 2
    if out.exists() and out.resolve() == src.resolve():
        print("error: output is the input file — the source is never "
              "overwritten; choose another name", file=sys.stderr)
        return 2
    is_project = src.suffix.lower() == ".lgproj"
    if not is_project and args.energy is None:
        print("error: a bare lattice carries no beam energy and the "
              "rigidity fixes every k1/ks — pass --energy (MeV)",
              file=sys.stderr)
        return 2
    try:
        lattice, beam_cfg, _conv = common.load_input(str(src))
        if args.energy is not None:
            beam_cfg.energy = args.energy
        if args.freq is not None:
            beam_cfg.frequency = args.freq
        if args.species is not None:
            beam_cfg.species = args.species
        if beam_cfg.species not in ("proton", "deuteron", "H-"):
            raise ValueError(f"unknown species {beam_cfg.species!r} "
                             "(proton / deuteron / H-)")
        if not beam_cfg.energy > 0:
            raise ValueError(f"energy must be > 0 MeV (got {beam_cfg.energy})")
        ref = common.build_ref(beam_cfg)
    except (ValueError, KeyError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    from linac_gen.io.madx_writer import write_madx
    try:
        warnings = write_madx(
            lattice, out, ref,
            sequence_name=args.sequence,
            on_unsupported="error" if args.strict else "marker",
            linearize=args.linearize,
            title=args.title or src.name)
    except (ValueError, NotImplementedError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    for w in warnings:
        print(f"export warning: {w}", file=sys.stderr)
    if not args.quiet:
        n = len(lattice.elements)
        print(f"wrote {out} — {n} lattice entries, {len(warnings)} "
              f"warning(s); reference {beam_cfg.species} "
              f"{beam_cfg.energy:g} MeV")
    return 0
