"""Lattice import formats — the one place that knows which file suffix
goes to which parser.

HELIX reads TraceWin ``.dat`` natively and has in-house subset parsers for
MAD-X, MAD8 flat files and Elegant.  Every other lattice format (Bmad,
SciBmad, PALS, lattix JSON) comes in through the optional ``lattix``
translator (:mod:`linac_gen.io.lattix_bridge`), which turns the deck into
a HELIX :class:`~linac_gen.core.lattice.Lattice` with a fidelity report.

:func:`parse_lattice_file` is the shared dispatcher behind the CLI
(``linac_gen.cli.common.load_lattice``), the GUI open/restore paths and
the New Project wizard, so a new format is wired once.
"""
from __future__ import annotations

from pathlib import Path

#: suffix → native parser
NATIVE_SUFFIXES: dict[str, str] = {
    ".dat": "tracewin",
    ".madx": "madx", ".seq": "madx",
    ".lat": "mad8", ".flat": "mad8",
    ".lte": "elegant",
}
#: suffixes delegated to lattix (double suffixes are matched with ``endswith``)
LATTIX_SUFFIXES: tuple[str, ...] = (
    ".bmad", ".jl", ".scibmad",
    ".pals.yaml", ".pals.yml", ".pals.json",
    ".lattix.json",
)
IMPORT_SUFFIXES: tuple[str, ...] = tuple(NATIVE_SUFFIXES) + LATTIX_SUFFIXES

#: dialog filter, one entry per family (used by the GUI open dialog and the wizard)
DIALOG_FILTER = (
    "Lattice files (*.dat *.madx *.seq *.lat *.flat *.lte *.bmad *.jl *.scibmad "
    "*.pals.yaml *.pals.yml *.pals.json *.lattix.json);;"
    "TraceWin (*.dat);;MAD-X (*.madx *.seq);;"
    "MAD8 (*.lat *.flat);;Elegant (*.lte);;"
    "Bmad (*.bmad);;SciBmad (*.jl *.scibmad);;"
    "PALS (*.pals.yaml *.pals.yml *.pals.json);;lattix JSON (*.lattix.json);;"
    "All Files (*)"
)


def import_format(path) -> str | None:
    """``"tracewin"`` / ``"madx"`` / ``"mad8"`` / ``"elegant"`` / ``"lattix"``
    for a recognised lattice suffix, else ``None``."""
    name = Path(path).name.lower()
    for suf in LATTIX_SUFFIXES:
        if name.endswith(suf):
            return "lattix"
    return NATIVE_SUFFIXES.get(Path(name).suffix)


def is_foreign_source(path) -> bool:
    """True when *path* is a lattice HELIX must never overwrite with TraceWin
    text (every import format except TraceWin itself; unknown suffixes are
    parsed as TraceWin and stay writable)."""
    return import_format(path) not in (None, "tracewin")


def parse_lattice_file(path, *, warn_unknown: bool = True):
    """Extension-dispatched parse → ``(lattice, metadata)``.

    ``metadata["warnings"]`` is a list of strings on every path.  A suffix
    nobody claims is parsed as TraceWin — with a warning in the list, since
    a MAD/Bmad deck fed to the TraceWin parser mis-parses silently.
    """
    fp = str(path)
    fmt = import_format(fp)
    if fmt == "madx":
        from linac_gen.io.madx_parser import parse_madx
        lat, meta = parse_madx(fp)[:2]
    elif fmt == "mad8":
        from linac_gen.io.mad8_parser import parse_mad8
        lat, meta = parse_mad8(fp)[:2]
    elif fmt == "elegant":
        from linac_gen.io.elegant_parser import parse_elegant
        lat, meta = parse_elegant(fp)[:2]
    elif fmt == "lattix":
        from linac_gen.io.lattix_bridge import parse_with_lattix
        lat, meta = parse_with_lattix(fp)
    else:
        from linac_gen.io.tracewin_parser import parse_tracewin
        lat, meta = parse_tracewin(fp)
        if fmt is None and warn_unknown:
            suf = Path(fp).suffix or "(none)"
            if not isinstance(meta, dict):
                meta = {"warnings": []}
            meta.setdefault("warnings", [])
            meta["warnings"] = list(meta["warnings"]) + [
                f"unknown lattice suffix {suf!r} — parsed as TraceWin .dat "
                f"(known: {' '.join(IMPORT_SUFFIXES)})"]
    if not isinstance(meta, dict):
        meta = {"warnings": []}
    return lat, meta


__all__ = ["DIALOG_FILTER", "IMPORT_SUFFIXES", "LATTIX_SUFFIXES", "NATIVE_SUFFIXES",
           "import_format", "is_foreign_source", "parse_lattice_file"]
