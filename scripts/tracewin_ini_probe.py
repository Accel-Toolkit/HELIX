#!/usr/bin/env python3
"""Map fields of TraceWin's binary project options file (``.ini``).

Read-only helper for extending ``linac_gen/io/tracewin_ini.py``'s field
table (docs/manual/appendices/G_tracewin_ini_format.md, "Extending the
map").  Three modes:

    python3 scripts/tracewin_ini_probe.py dump FILE.ini [--raw]
        the reader's report, the particle table and the string slots;
        --raw adds every non-zero int32/f64 slot of the beam window.

    python3 scripts/tracewin_ini_probe.py find VALUE FILE.ini [FILE2.ini ...]
        [--types f64,f32,i32] [--scale 1,1e3,1e6,1e-3,1e-6] [--tol 1e-9]
        every offset whose int32 / float32 / float64 reading equals VALUE
        (times one of --scale) in every file given — e.g. the energy you
        typed into TraceWin in eV.  A hit inside a known slot is named.

    python3 scripts/tracewin_ini_probe.py diff BASE.ini VARIANT.ini[:label] ...
        [--max-runs 4]
        byte-diff BASE against each VARIANT (a copy of the same project
        saved after changing ONE setting in TraceWin); every run of
        differing bytes is printed with its int32 / float64 readings at
        natural alignment in both files, named when the run falls inside
        a known slot, otherwise with a ready-to-paste FieldSpec line.
        Exit 3 when more than --max-runs runs differ: you changed more
        than one thing (TraceWin also rewrites the last-directory
        strings and window state — keep the project in one folder).

Exit codes: 0 found / diffed, 1 nothing found, 2 bad arguments or
unreadable file, 3 too many differing runs.
"""
from __future__ import annotations

import argparse
import struct
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from linac_gen.io.tracewin_ini import (  # noqa: E402
    FIELDS, KNOWN_SIZES, MAGIC, RAW_F64_WINDOW, RAW_INT_WINDOW,
    load_tracewin_ini, report,
)

_SLOTS = {}
for _f in FIELDS:
    if _f.fmt != "s":
        _SLOTS[(_f.fmt, _f.offset)] = _f


def _known(fmt: str, offset: int):
    """The FieldSpec whose slot starts exactly at ``offset`` with ``fmt``."""
    return _SLOTS.get((fmt, offset))


def _inside_known(offset: int, size: int) -> bool:
    """True when [offset, offset+size) overlaps any known slot — a
    misaligned or partial reading of a field that is already mapped."""
    for (f, o), _spec in _SLOTS.items():
        if o < offset + size and offset < o + struct.calcsize(f):
            return True
    return False


def _read_bytes(path: str) -> bytes:
    p = Path(path)
    if not p.is_file():
        raise SystemExit(f"error: {path} not found")
    b = p.read_bytes()
    if not b.startswith(MAGIC):
        raise SystemExit(f"error: {path} is not a TraceWin options file")
    return b


# ---------------------------------------------------------------------------
# dump
# ---------------------------------------------------------------------------
def cmd_dump(args) -> int:
    try:
        ini = load_tracewin_ini(args.file)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(report(ini))
    print()
    print("Particle table (row: mass eV, charge, name)")
    for r in ini.particles:
        print(f"  {r.index:2d}: {r.mass_eV:.10g}  {r.charge:+d}  {r.name!r}")
    print()
    print("String slots")
    for f in FIELDS:
        if f.fmt == "s":
            print(f"  0x{f.offset:04x} {f.name:<14} {ini.fields[f.name]!r}")
    if args.raw:
        print()
        print(f"Raw non-zero slots (int32 0x{RAW_INT_WINDOW[0]:04x}-"
              f"0x{RAW_INT_WINDOW[1]:04x}, f64 0x{RAW_F64_WINDOW[0]:04x}-"
              f"0x{RAW_F64_WINDOW[1]:04x})")
        for k, v in ini.raw.items():
            fmt = "<i" if k.startswith("i32") else "<d"
            off = int(k.split("@")[1], 16)
            spec = _known(fmt, off)
            tag = f"{spec.name} ({spec.status})" if spec else "unknown"
            print(f"  {k} = {v!r:<24} {tag}")
    return 0


# ---------------------------------------------------------------------------
# find
# ---------------------------------------------------------------------------
def _matches(reading: float, target: float, tol: float) -> bool:
    if target == 0:
        return reading == 0
    return abs(reading - target) <= tol * abs(target)


def cmd_find(args) -> int:
    try:
        value = float(args.value)
    except ValueError:
        print(f"error: VALUE must be a number, got {args.value!r}", file=sys.stderr)
        return 2
    types = [t.strip() for t in args.types.split(",") if t.strip()]
    scales = [float(s) for s in args.scale.split(",") if s.strip()]
    # The options struct is packed on 4 bytes: its doubles sit at offsets
    # = 4 (mod 8) (freq1 at 0x2f24), so every type is scanned on a 4-byte
    # grid — an 8-byte grid misses the whole beam block.
    fmts = {"f64": ("<d", 8, 4), "f32": ("<f", 4, 4), "i32": ("<i", 4, 4)}
    bad = [t for t in types if t not in fmts]
    if bad:
        print(f"error: unknown --types {bad}", file=sys.stderr)
        return 2
    bufs = [(f, _read_bytes(f)) for f in args.files]
    hits: dict = {}
    for name, buf in bufs:
        for t in types:
            fmt, size, step = fmts[t]
            for off in range(0, len(buf) - size + 1, step):
                r = struct.unpack_from(fmt, buf, off)[0]
                if t == "i32":
                    ok = any(abs(r - value * s) < 0.5 for s in scales
                             if float(value * s).is_integer())
                else:
                    ok = any(_matches(r, value * s, args.tol) for s in scales)
                if ok:
                    hits.setdefault((t, off), set()).add(name)
    n_files = len(bufs)
    common = sorted(k for k, names in hits.items() if len(names) == n_files)
    if not common:
        print(f"no {'/'.join(types)} slot equals {value:g} (x {args.scale}) in "
              f"{'every one of ' if n_files > 1 else ''}"
              f"{', '.join(Path(f).name for f, _ in bufs)}")
        return 1
    for t, off in common:
        fmt = fmts[t][0]
        readings = [struct.unpack_from(fmt, buf, off)[0] for _, buf in bufs]
        spec = _known("<i" if t == "i32" else "<d", off) if t != "f32" else None
        tag = (f"known: {spec.name} ({spec.status}, {spec.unit or 'no unit'})"
               if spec else "unknown slot")
        print(f"0x{off:04x} {t:<3} = {', '.join(f'{r:.10g}' for r in readings):<32} {tag}")
    return 0


# ---------------------------------------------------------------------------
# diff
# ---------------------------------------------------------------------------
def _runs(a: bytes, b: bytes):
    """Maximal runs [start, end) of differing bytes (gaps < 4 merged)."""
    n = min(len(a), len(b))
    runs = []
    i = 0
    while i < n:
        if a[i] != b[i]:
            j = i
            while j < n and (a[j] != b[j] or (j + 3 < n and any(
                    a[k] != b[k] for k in range(j, min(j + 4, n))))):
                j += 1
            runs.append((i, j))
            i = j
        else:
            i += 1
    if len(a) != len(b):
        runs.append((n, max(len(a), len(b))))
    return runs


def _readings(buf: bytes, start: int, end: int):
    """(fmt, offset, value) for the int32 and f64 slots on the struct's
    4-byte grid that could hold the change [start, end): every slot that
    CONTAINS the run when the run fits in one slot, else every slot that
    overlaps it."""
    out = []
    for fmt, size in (("<i", 4), ("<d", 8)):
        first = max(0, ((start - size + 1 + 3) // 4) * 4)   # first 4-aligned slot overlapping
        for off in range(first, end, 4):
            if off + size > len(buf) or off + size <= start:
                continue
            contains = off <= start and off + size >= end
            if end - start <= size and not contains:
                continue
            out.append((fmt, off, struct.unpack_from(fmt, buf, off)[0]))
    return out


def cmd_diff(args) -> int:
    base = _read_bytes(args.base)
    if len(base) not in KNOWN_SIZES:
        print(f"note: {Path(args.base).name} is {len(base)} bytes — not one of "
              f"the known layouts {sorted(KNOWN_SIZES)}; offsets are raw",
              file=sys.stderr)
    too_many = False
    for item in args.variants:
        path, _, label = item.partition(":")
        var = _read_bytes(path)
        runs = _runs(base, var)
        print(f"== {Path(args.base).name} -> {Path(path).name}"
              f"{f' ({label})' if label else ''}: {len(runs)} differing run(s)")
        if len(runs) > args.max_runs:
            too_many = True
            print(f"   more than --max-runs {args.max_runs}: more than one "
                  f"setting changed (or the project moved folders)")
        for start, end in runs:
            print(f"  bytes 0x{start:04x}-0x{end:04x} ({end - start} B)")
            for fmt, off, vb in _readings(base, start, end):
                vv = (struct.unpack_from(fmt, var, off)[0]
                      if off + struct.calcsize(fmt) <= len(var) else None)
                if vb == vv:
                    continue
                kind = "i32" if fmt == "<i" else "f64"
                spec = _known(fmt, off)
                if spec:
                    tag = f"known: {spec.name} ({spec.status})"
                elif _inside_known(off, struct.calcsize(fmt)):
                    continue            # partial/misaligned view of a mapped slot
                else:
                    name = (label or "new_field").replace(" ", "_")
                    tag = (f'FieldSpec("{name}", 0x{off:04X}, "{fmt}", "", 1, '
                           f'"probable", "changed {vb!r} -> {vv!r} in '
                           f'{Path(path).name}")')
                print(f"    0x{off:04x} {kind} {vb!r} -> {vv!r}   {tag}")
    return 3 if too_many else 0


# ---------------------------------------------------------------------------
def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        prog="tracewin_ini_probe",
        description="map fields of TraceWin's binary .ini options file")
    sub = p.add_subparsers(dest="mode", required=True)
    d = sub.add_parser("dump", help="report + particle table + strings")
    d.add_argument("file")
    d.add_argument("--raw", action="store_true", help="also every non-zero slot")
    f = sub.add_parser("find", help="offsets whose reading equals VALUE")
    f.add_argument("value")
    f.add_argument("files", nargs="+")
    f.add_argument("--types", default="f64,f32,i32")
    f.add_argument("--scale", default="1,1e3,1e6,1e-3,1e-6")
    f.add_argument("--tol", type=float, default=1e-9)
    g = sub.add_parser("diff", help="byte-diff BASE against VARIANT[:label] …")
    g.add_argument("base")
    g.add_argument("variants", nargs="+")
    g.add_argument("--max-runs", type=int, default=4)
    args = p.parse_args(argv)
    try:
        return {"dump": cmd_dump, "find": cmd_find, "diff": cmd_diff}[args.mode](args)
    except SystemExit as exc:
        if isinstance(exc.code, str):
            print(exc.code, file=sys.stderr)
            return 2
        raise


if __name__ == "__main__":
    sys.exit(main())
