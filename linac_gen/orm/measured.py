"""Measured orbit response matrices: FORMA export folders and the HELIX ORM CSV.

A FORMA v8 scan folder holds one *driven plane* (all H trims or all V trims)
and two *read planes*::

    response_matrix_<stamp>_horizontal.csv         rows = horizontal BPM devices
    response_matrix_<stamp>_vertical.csv           rows = vertical BPM devices
    response_matrix_<stamp>_horizontal_error.csv   1σ errors (optional)
    response_matrix_<stamp>_vertical_error.csv
    response_matrix_<stamp>_corrector_gains.csv    commanded/achieved amplitude per trim (optional)
    scan_setup.json  scan_info.json  baseline_rms.json                      (optional metadata)

Header ``Reading Device,<trim device>,…``; one row per BPM device; values in
**mm per ampere** of trim current (the FORMA convention — the file carries no
unit).  The kick plane is inferred from the trim device names (``…TMH`` →
``"x"``, ``…TMV`` → ``"y"``).

The HELIX ORM CSV is one matrix per file (one kick plane, one read plane),
keyed by lattice labels::

    # helix_orm_csv 1
    # kick_plane: x
    # read_plane: x
    # units: mm/A
    bpm,D01T,D02T,…
    D02BPM,0.502,-0.004,…

with an optional companion ``<stem>_error.csv`` of identical layout.
"""
from __future__ import annotations

import csv
import glob
import json
import os
import re
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

_PLANE_OF_SUFFIX = {"H": "x", "V": "y"}
_KIND_OF_PLANE = {"x": "horizontal", "y": "vertical"}


@dataclass
class MeasuredOrm:
    """One driven plane of a measured ORM (see the module docstring)."""

    kick_plane: str                                   # "x" | "y"
    trims: list                                       # column device names (as in the files)
    bpms: dict = field(default_factory=dict)          # read plane -> row device names
    values: dict = field(default_factory=dict)        # read plane -> (n_bpm, n_trim) array
    errors: dict = field(default_factory=dict)        # read plane -> same shape, NaN when absent
    units: str = "mm/A"
    meta: dict = field(default_factory=dict)
    source: str = ""

    def read_planes(self) -> list:
        return sorted(self.values)

    def has_errors(self, read_plane: str) -> bool:
        e = self.errors.get(read_plane)
        return e is not None and bool(np.isfinite(e).any())

    def summary(self) -> str:
        parts = [f"{self.source or 'measured'}: kick plane {self.kick_plane}, {len(self.trims)} trims"]
        for p in self.read_planes():
            parts.append(f"read {p}: {len(self.bpms[p])} BPMs" + ("" if self.has_errors(p) else " (no errors)"))
        return "; ".join(parts)


# ---------------------------------------------------------------------------
# CSV helpers
# ---------------------------------------------------------------------------
def _read_matrix_csv(path):
    """``(columns, rows, values)`` of a ``Reading Device,…`` / ``bpm,…`` matrix file."""
    text = _read_text(path)
    lines = [ln for ln in text.splitlines() if ln.strip() and not ln.lstrip().startswith("#")]
    if not lines:
        raise ValueError(f"{path}: empty matrix file")
    reader = list(csv.reader(lines))
    header = [h.strip() for h in reader[0]]
    cols = header[1:]
    rows, vals = [], []
    for k, r in enumerate(reader[1:], start=2):
        if not r or not r[0].strip():
            continue
        if len(r) != len(header):
            raise ValueError(f"{path}: line {k} has {len(r)} fields, header has {len(header)}")
        rows.append(r[0].strip())
        try:
            vals.append([float(v) if v.strip() not in ("", "nan", "NaN", "None") else np.nan for v in r[1:]])
        except ValueError as exc:
            raise ValueError(f"{path}: line {k}: {exc}") from None
    return cols, rows, np.asarray(vals, dtype=float).reshape(len(rows), len(cols))


def _read_text(path) -> str:
    """FORMA writes UTF-8 (sometimes with a BOM) on Windows; fall back to latin-1."""
    raw = Path(path).read_bytes()
    try:
        return raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        return raw.decode("latin-1")


def _csv_comments(path) -> dict:
    """``# key: value`` header comments of a HELIX ORM CSV."""
    out = {}
    for ln in _read_text(path).splitlines():
        if not ln.startswith("#"):
            break
        body = ln[1:].strip()
        if ":" in body:
            k, _, v = body.partition(":")
            out[k.strip()] = v.strip()
        else:
            out.setdefault("_tags", []).append(body)
    return out


# ---------------------------------------------------------------------------
# FORMA folder
# ---------------------------------------------------------------------------
def _kick_plane_of(trims) -> str:
    planes = {_PLANE_OF_SUFFIX.get(t.strip()[-1:].upper()) for t in trims}
    if len(planes) != 1 or None in planes:
        raise ValueError("cannot infer the driven plane from the trim devices "
                         f"{list(trims)[:4]}…: every column must end in H or V; pass plane= explicitly")
    return planes.pop()


def read_forma_folder(path, *, plane: str | None = None) -> MeasuredOrm:
    """Read one FORMA scan folder (one driven plane, both read planes).

    ``plane`` overrides the kick-plane inference from the trim names.  Value
    and error files must list the same devices in the same order (a
    mismatch is a hard error — silently re-aligning would misassign rows).
    Missing error files give NaN errors and a warning.
    """
    folder = Path(path)
    if not folder.is_dir():
        raise ValueError(f"{folder}: not a FORMA scan folder (directory expected)")
    out: dict = {}
    stamp = None
    trims = None
    bpms, values, errors = {}, {}, {}
    for read_plane, kind in _KIND_OF_PLANE.items():
        files = sorted(glob.glob(str(folder / f"response_matrix_*_{kind}.csv")))
        if not files:
            continue
        if len(files) > 1:
            raise ValueError(f"{folder}: {len(files)} '{kind}' matrices; a scan folder holds exactly one")
        m = re.match(r"response_matrix_(.+)_" + kind + r"\.csv$", Path(files[0]).name)
        stamp = stamp or (m.group(1) if m else None)
        cols, rows, A = _read_matrix_csv(files[0])
        if trims is None:
            trims = cols
        elif cols != trims:
            raise ValueError(f"{folder}: trim columns of the {kind} matrix differ from the other plane")
        efile = folder / f"response_matrix_{stamp}_{kind}_error.csv"
        if efile.exists():
            ecols, erows, E = _read_matrix_csv(efile)
            if ecols != cols or erows != rows:
                raise ValueError(f"{efile.name}: device order differs from {Path(files[0]).name}; refusing to re-align")
        else:
            warnings.warn(f"{folder.name}: no {kind} error file — errors set to NaN (fit weights use the systematic floor only)")
            E = np.full_like(A, np.nan)
        bpms[read_plane], values[read_plane], errors[read_plane] = rows, A, E
    if trims is None:
        raise ValueError(f"{folder}: no response_matrix_*_horizontal.csv / _vertical.csv found")
    meta = {"stamp": stamp, "folder": str(folder)}
    for name in ("scan_setup.json", "scan_info.json", "baseline_rms.json"):
        f = folder / name
        if f.exists():
            try:
                meta[name[:-5]] = json.loads(_read_text(f))
            except (ValueError, OSError) as exc:            # metadata is optional; never fatal
                warnings.warn(f"{f.name}: unreadable ({exc})")
    kick = plane
    if kick is None:
        try:
            kick = _kick_plane_of(trims)
        except ValueError:
            # synthetic / non-Fermilab exports: an explicit "kick_plane" in scan_info.json
            info = meta.get("scan_info")
            kick = info.get("kick_plane") if isinstance(info, dict) else None
            if kick is None:
                raise
    if kick not in ("x", "y"):
        raise ValueError(f"plane must be 'x' or 'y', got {kick!r}")
    gains = folder / f"response_matrix_{stamp}_corrector_gains.csv"
    if gains.exists():
        rowsg = list(csv.DictReader(_read_text(gains).splitlines()))
        meta["corrector_gains"] = {r.get("device", "").strip(): r for r in rowsg if r.get("device")}
    dead = []
    info = meta.get("scan_info") or {}
    if isinstance(info, dict):
        dead = list((info.get("baseline") or {}).get("dead_bpms", []) or [])
    if not dead and isinstance(meta.get("baseline_rms"), dict):
        dead = list(meta["baseline_rms"].get("dead_bpms", []) or [])
    meta["dead_devices"] = dead
    return MeasuredOrm(kick_plane=kick, trims=list(trims), bpms=bpms, values=values, errors=errors,
                       units="mm/A", meta=meta, source=f"forma:{folder.name}")


# ---------------------------------------------------------------------------
# HELIX ORM CSV (one matrix per file, keyed by lattice labels)
# ---------------------------------------------------------------------------
def write_orm_csv(path, values, bpm_labels, trim_labels, *, kick_plane: str, read_plane: str,
                  units: str = "mm/Tm", errors=None, header_lines=()) -> None:
    """Write one matrix block (and ``<stem>_error.csv`` when ``errors`` is given)."""
    values = np.asarray(values, dtype=float)
    if values.shape != (len(bpm_labels), len(trim_labels)):
        raise ValueError(f"values shape {values.shape} != ({len(bpm_labels)}, {len(trim_labels)})")
    path = Path(path)

    def _dump(p, A):
        with open(p, "w", encoding="utf-8", newline="") as fh:
            fh.write("# helix_orm_csv 1\n")
            fh.write(f"# kick_plane: {kick_plane}\n# read_plane: {read_plane}\n# units: {units}\n")
            for ln in header_lines:
                fh.write(f"# {ln}\n")
            w = csv.writer(fh, lineterminator="\n")
            w.writerow(["bpm"] + list(trim_labels))
            for lab, row in zip(bpm_labels, A):
                w.writerow([lab] + [f"{v:.17g}" if np.isfinite(v) else "nan" for v in row])

    _dump(path, values)
    if errors is not None:
        _dump(path.with_name(path.stem + "_error" + path.suffix), np.asarray(errors, dtype=float))


def read_orm_csv(path) -> MeasuredOrm:
    """Read one HELIX ORM CSV block (plus its ``_error`` companion when present)."""
    path = Path(path)
    hdr = _csv_comments(path)
    if "helix_orm_csv 1" not in hdr.get("_tags", []):
        raise ValueError(f"{path}: not a HELIX ORM CSV (missing '# helix_orm_csv 1' header)")
    kick, read = hdr.get("kick_plane"), hdr.get("read_plane")
    if kick not in ("x", "y") or read not in ("x", "y"):
        raise ValueError(f"{path}: header needs '# kick_plane: x|y' and '# read_plane: x|y'")
    cols, rows, A = _read_matrix_csv(path)
    efile = path.with_name(path.stem + "_error" + path.suffix)
    if efile.exists():
        ecols, erows, E = _read_matrix_csv(efile)
        if ecols != cols or erows != rows:
            raise ValueError(f"{efile.name}: layout differs from {path.name}")
    else:
        E = np.full_like(A, np.nan)
    return MeasuredOrm(kick_plane=kick, trims=cols, bpms={read: rows}, values={read: A}, errors={read: E},
                       units=hdr.get("units", "mm/A"), meta={"file": str(path)}, source=f"csv:{path.name}")


# ---------------------------------------------------------------------------
def load_measured(paths) -> dict:
    """Load folders (FORMA) and/or CSV files into ``{"x": MeasuredOrm, "y": MeasuredOrm}``.

    CSV blocks of the same kick plane are merged into one ``MeasuredOrm``;
    two sources for the same (kick, read) block are refused.
    """
    out: dict = {}
    for p in paths:
        m = read_forma_folder(p) if os.path.isdir(p) else read_orm_csv(p)
        cur = out.get(m.kick_plane)
        if cur is None:
            out[m.kick_plane] = m
            continue
        if cur.trims != m.trims:
            raise ValueError(f"{m.source}: trim columns differ from {cur.source} (same kick plane {m.kick_plane})")
        for rp in m.read_planes():
            if rp in cur.values:
                raise ValueError(f"{m.source}: block kick={m.kick_plane} read={rp} already loaded from {cur.source}")
            cur.bpms[rp], cur.values[rp], cur.errors[rp] = m.bpms[rp], m.values[rp], m.errors[rp]
        cur.meta.setdefault("merged_from", []).append(m.source)
    return out
