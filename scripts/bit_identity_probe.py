#!/usr/bin/env python3
"""Bit-identity probe: an opt-in feature (ring mode) must leave every linac result unchanged.

Two kinds of snapshot, both taken with whichever HELIX tree is imported:

* **runs** -- every case of ``scripts/physics_fix_report.py`` (envelope, matrix and
  multi-particle arrays, compared bit for bit) plus the importer cases added here;
* **parses** -- every lattice file found (the tracked examples plus any extra paths)
  parsed with :func:`linac_gen.io.formats.parse_lattice_file` and dumped element by
  element (class, every public attribute and every plain private one, floats by
  ``repr``) together with the parse warnings and the reference particle;
* **refruns** -- for every such file whose importer builds a reference particle
  (MAD-X, MAD8, elegant), the one-pass transfer matrix computed with THAT reference
  (the run cases go through a BeamConfig and never see it) and the reference
  after the line, so an importer change to energies or RF frequencies shows up.

Take the BEFORE snapshot on the unmodified tree, make the change, take the AFTER
snapshot with the same compiled kernels, then compare::

    PYTHONPATH=. python3 scripts/bit_identity_probe.py snapshot --tree . --out DIR \\
        [--extra-lattices PATH ...] [--cases GLOB ...] [--no-runs] [--no-parses]
    python3 scripts/bit_identity_probe.py compare --before DIR1 --after DIR2 [--out report.md]

``compare`` exits 1 when anything differs (every saved array, whatever its dtype),
when a case that ran before fails now or fails with another message, or when a
BEFORE entry is missing from AFTER.  An attribute that is new in AFTER and holds
``None`` (a new default-off field) is reported separately as benign.  Paths outside
the repository are recorded by name and a short hash, never in full.
"""
from __future__ import annotations

import argparse
import contextlib
import glob
import hashlib
import importlib.util
import io
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
LATTICE_SUFFIXES = (".dat", ".madx", ".seq", ".lat", ".flat", ".lte", ".mad", ".mad8")


def _load_pfr():
    spec = importlib.util.spec_from_file_location("physics_fix_report", HERE / "physics_fix_report.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _quiet():
    return contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO())


# ---------------------------------------------------------------------------
# extra run cases (importers touched by the ring-mode prerequisites)
# ---------------------------------------------------------------------------
def _extra_run_cases(pfr) -> None:
    fodo = dict(species="proton", energy=800.0, frequency=352.21)
    pfr._add("transport_madx", "examples/madx/transport.madx", ("env", "matrix", "mp"), n=2000, **fodo)


# ---------------------------------------------------------------------------
# parse dumps
# ---------------------------------------------------------------------------
def _canon(v, depth=0):
    """JSON-able, exact rendering of an attribute value."""
    if depth > 10:
        return f"<depth:{type(v).__name__}>"
    if v is None or isinstance(v, (bool, str)):
        return v
    if isinstance(v, (int, np.integer)):
        return int(v)
    if isinstance(v, (float, np.floating)):
        f = float(v)
        return repr(f) if (math.isfinite(f)) else str(f)
    if isinstance(v, complex):
        return f"complex({v.real!r},{v.imag!r})"
    if isinstance(v, np.ndarray):
        if v.size > 64 or v.dtype.kind not in "biuf":
            # field maps and other bulk arrays: an exact digest of the raw bytes
            # (dumping every value as text once took 15 GB on a 3-D field map)
            raw = np.ascontiguousarray(v)
            digest = hashlib.sha256(raw.tobytes() if raw.dtype.kind != "O" else repr(raw.tolist()).encode())
            return {"__ndarray__": str(v.dtype), "shape": list(v.shape), "sha256": digest.hexdigest()}
        return {"__ndarray__": str(v.dtype), "shape": list(v.shape),
                "data": [_canon(x, depth + 1) for x in v.ravel().tolist()]}
    if isinstance(v, (list, tuple)):
        items = [_canon(x, depth + 1) for x in v]
        if len(items) > 256:
            return {"__seq__": type(v).__name__, "len": len(items),
                    "sha256": hashlib.sha256(json.dumps(items, sort_keys=True).encode()).hexdigest()}
        return items
    if isinstance(v, dict):
        return {str(k): _canon(x, depth + 1) for k, x in sorted(v.items(), key=lambda kv: str(kv[0]))}
    if isinstance(v, Path):
        return f"<path:{v.name}>"
    if callable(v):
        return f"<callable:{getattr(v, '__name__', type(v).__name__)}>"
    attrs = getattr(v, "__dict__", None)
    if attrs is not None:
        return {"__class__": type(v).__name__,
                **{k: _canon(x, depth + 1) for k, x in sorted(attrs.items()) if not k.startswith("_")}}
    return f"<{type(v).__name__}:{v!r}>"


_PLAIN = (type(None), bool, int, float, str, np.integer, np.floating, np.ndarray)


def _plain(v) -> bool:
    return isinstance(v, _PLAIN) or (isinstance(v, (list, tuple))
                                     and all(isinstance(x, _PLAIN) for x in v))


def _element_dump(e) -> dict:
    d = {"__class__": type(e).__name__}
    for k, v in sorted(vars(e).items()):
        # private state: plain values only (caches, random generators and
        # helper objects are skipped — their repr can carry memory addresses)
        if k.startswith("_") and not _plain(v):
            continue
        d[k] = _canon(v)
    return d


def _parse_dump(path: Path) -> dict:
    from linac_gen.core.config import BeamConfig
    from linac_gen.io.formats import parse_lattice_file

    fb = BeamConfig(species="H-", energy=800.0, frequency=162.5) \
        if path.suffix.lower() in (".lat", ".flat", ".mad", ".mad8") else None
    o1, o2 = _quiet()
    with o1, o2:
        lat, meta = parse_lattice_file(str(path), fallback_beam=fb)
    meta = meta if isinstance(meta, dict) else {}
    ref = meta.get("reference")
    return {
        "n_elements": len(lat.elements),
        "elements": [_element_dump(e) for e in lat.elements],
        "lattice": {k: _canon(v) for k, v in sorted(vars(lat).items())
                    if not k.startswith("_") and k != "elements"},
        "warnings": [str(w) for w in meta.get("warnings", [])],
        "reference": _canon(ref) if ref is not None else None,
        "meta_keys": sorted(str(k) for k in meta),
        "meta": {str(k): _canon(v) for k, v in sorted(meta.items(), key=lambda kv: str(kv[0]))
                 if k not in ("warnings", "reference")},
    }


def _lattice_files(repo: Path, extra: list[str]) -> list[Path]:
    out = []
    try:
        tracked = subprocess.run(["git", "ls-files"], cwd=repo, capture_output=True, text=True,
                                 check=True).stdout.split()
    except Exception:  # noqa: BLE001 - no git: walk examples/
        tracked = [str(p.relative_to(repo)) for p in (repo / "examples").rglob("*")]
    for rel in tracked:
        p = repo / rel
        if p.suffix.lower() in LATTICE_SUFFIXES and rel.startswith(("examples/", "tests/")):
            out.append(p)
    for item in extra:
        for g in sorted(glob.glob(os.path.expanduser(item), recursive=True)):
            p = Path(g)
            if p.is_dir():
                out += sorted(q for q in p.rglob("*") if q.suffix.lower() in LATTICE_SUFFIXES)
            elif p.is_file():
                out.append(p)
    seen, uniq = set(), []
    for p in out:
        r = p.resolve()
        if r not in seen:
            seen.add(r)
            uniq.append(p)
    return uniq


def _key(path: Path, repo: Path) -> str:
    try:
        return "repo:" + str(path.resolve().relative_to(repo.resolve()))
    except ValueError:
        # outside the repository: the name and a short hash, never the full path
        h = hashlib.sha256(str(path.resolve()).encode()).hexdigest()[:12]
        return f"ext:{path.name}#{h}"


def _ref_run(path: Path) -> dict:
    """The one-pass transfer matrix computed with the reference particle the
    deck's own importer built, and that reference after the line."""
    from linac_gen.core.config import BeamConfig
    from linac_gen.io.formats import parse_lattice_file
    from linac_gen.tracking.matrix_tracking import compute_transfer_matrix

    fb = BeamConfig(species="H-", energy=800.0, frequency=162.5) \
        if path.suffix.lower() in (".lat", ".flat", ".mad", ".mad8") else None
    o1, o2 = _quiet()
    with o1, o2:
        lat, meta = parse_lattice_file(str(path), fallback_beam=fb)
        ref = (meta or {}).get("reference") if isinstance(meta, dict) else None
        if ref is None:
            return {}
        r = ref.copy()
        M = np.asarray(compute_transfer_matrix(lat, r), dtype=float)
    scalars = lambda q: np.array([q.w_kin, q.frequency, q.beta, q.gamma, q.species.mass,  # noqa: E731
                                  float(q.species.charge)])
    return {"M": M, "ref_in": scalars(ref), "ref_out": scalars(r)}


# ---------------------------------------------------------------------------
# snapshot / compare
# ---------------------------------------------------------------------------
def cmd_snapshot(a) -> int:
    pfr = _load_pfr()
    tree = Path(a.tree)
    pfr._assert_tree(tree)
    repo = Path(a.repo) if a.repo else tree
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    if not a.no_runs:
        _extra_run_cases(pfr)
        ns = argparse.Namespace(tree=str(tree), repo=str(repo), out=str(out / "runs"), cases=a.cases)
        cwd = os.getcwd()
        try:
            pfr.cmd_snapshot(ns)
        finally:
            os.chdir(cwd)
    if not a.no_parses:
        pdir = out / "parses"
        pdir.mkdir(exist_ok=True)
        index = {}
        for i, p in enumerate(_lattice_files(repo, a.extra_lattices or [])):
            k = _key(p, repo)
            t0 = time.time()
            try:
                d = _parse_dump(p)
                fn = f"p{i:04d}.json"
                (pdir / fn).write_text(json.dumps(d, sort_keys=True), encoding="utf-8")
                index[k] = {"file": fn, "n_elements": d["n_elements"], "n_warnings": len(d["warnings"])}
                print(f"parse {k}: {d['n_elements']} elements, {len(d['warnings'])} warnings, "
                      f"{time.time() - t0:.1f} s", flush=True)
            except Exception as exc:  # noqa: BLE001 - record, keep going
                index[k] = {"error": f"{type(exc).__name__}: {exc}"[:500]}
                print(f"parse {k}: ERROR {index[k]['error'][:120]}", flush=True)
        (pdir / "index.json").write_text(json.dumps(index, indent=1, sort_keys=True), encoding="utf-8")
    if not a.no_parses and not a.no_refruns:
        rdir = out / "refruns"
        rdir.mkdir(exist_ok=True)
        index = {}
        for i, p in enumerate(_lattice_files(repo, a.extra_lattices or [])):
            k = _key(p, repo)
            try:
                arrays = _ref_run(p)
                if not arrays:
                    continue                      # (no reference from this importer)
                fn = f"r{i:04d}.npz"
                np.savez_compressed(rdir / fn, **arrays)
                index[k] = {"file": fn}
            except Exception as exc:  # noqa: BLE001 - record, keep going
                index[k] = {"error": f"{type(exc).__name__}: {exc}"[:500]}
            print(f"refrun {k}: {index.get(k, {}).get('error', 'ok')}"[:160], flush=True)
        (rdir / "index.json").write_text(json.dumps(index, indent=1, sort_keys=True), encoding="utf-8")
    return 0


def _compare_runs(before: Path, after: Path, lines: list[str]) -> int:
    pfr = _load_pfr()
    bad = 0
    mb = json.loads((before / "manifest.json").read_text()) if (before / "manifest.json").exists() else {}
    ma = json.loads((after / "manifest.json").read_text()) if (after / "manifest.json").exists() else {}
    for name, rb in sorted(mb.items()):
        ra = ma.get(name)
        if ra is None:
            lines.append(f"| run {name} | MISSING in AFTER |")
            bad += 1
        elif ("error" in rb) != ("error" in ra) or ("skipped" in rb) != ("skipped" in ra):
            lines.append(f"| run {name} | STATUS CHANGED: before {rb} after {ra} |")
            bad += 1
        elif rb.get("error") != ra.get("error"):
            lines.append(f"| run {name} | fails before and after, with another message: "
                         f"{rb.get('error')} -> {ra.get('error')} |")
            bad += 1
    # every saved array, whatever its dtype (the run report compares float and
    # bool arrays only; integer and text arrays such as element names count too)
    for f in sorted(after.glob("*.npz")):
        g = before / f.name
        if not g.exists():
            continue
        with np.load(f, allow_pickle=False) as A, np.load(g, allow_pickle=False) as B:
            for k in sorted((set(A.files) & set(B.files)) - {"_wall_s"}):
                x, y = A[k], B[k]
                same = x.shape == y.shape and x.dtype == y.dtype and (
                    np.array_equal(x, y, equal_nan=True) if x.dtype.kind in "fc"
                    else np.array_equal(x, y))
                if not same:
                    lines.append(f"| run {f.stem} | array {k!r} differs ({y.dtype}{list(y.shape)} -> "
                                 f"{x.dtype}{list(x.shape)}) |")
                    bad += 1
    for name in sorted(set(ma) - set(mb)):
        lines.append(f"| run {name} | new in AFTER (not compared) |")
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        rc = pfr.cmd_compare(argparse.Namespace(before=str(before), after=str(after), out=None))
    text = buf.getvalue()
    lines.append("")
    lines.append(text.strip())
    if rc != 0 or " 0 changed" not in text:
        bad += 1
    return bad


def _diff_elements(eb: list, ea: list, limit=12) -> tuple[list[str], list[str]]:
    real, benign = [], []
    if len(eb) != len(ea):
        real.append(f"element count {len(eb)} -> {len(ea)}")
    for i, (b, c) in enumerate(zip(eb, ea, strict=False)):     # (counts compared above)
        if b == c:
            continue
        for k in sorted(set(b) | set(c)):
            if k not in b:
                (benign if c[k] is None else real).append(
                    f"#{i} {c.get('name', b.get('name'))} {c['__class__']}: +{k}={c[k]!r}")
            elif k not in c:
                real.append(f"#{i} {b.get('name')} {b['__class__']}: -{k}")
            elif b[k] != c[k]:
                real.append(f"#{i} {b.get('name')} {b['__class__']}.{k}: {str(b[k])[:80]} -> {str(c[k])[:80]}")
        if len(real) > limit:
            real.append("…")
            break
    return real, benign


def _compare_parses(before: Path, after: Path, lines: list[str]) -> int:
    bad = 0
    ib = json.loads((before / "index.json").read_text())
    ia = json.loads((after / "index.json").read_text())
    for k in sorted(ib):
        b, a = ib[k], ia.get(k)
        if a is None:
            lines.append(f"| parse {k} | MISSING in AFTER |")
            bad += 1
            continue
        if "error" in b or "error" in a:
            if b.get("error") != a.get("error"):
                lines.append(f"| parse {k} | before: {b.get('error', 'ok')} | after: {a.get('error', 'ok')} |")
                bad += 1
            continue
        db = json.loads((before / b["file"]).read_text())
        da = json.loads((after / a["file"]).read_text())
        if db == da:
            continue
        real, benign = _diff_elements(db["elements"], da["elements"])
        for part in ("lattice", "warnings", "reference", "meta_keys", "meta"):
            if db.get(part) != da.get(part):
                real.append(f"{part}: {str(db[part])[:200]} -> {str(da[part])[:200]}")
        if real:
            bad += 1
            lines.append(f"| parse {k} | CHANGED | " + "; ".join(real[:14]) + " |")
        elif benign:
            lines.append(f"| parse {k} | benign (new default-None attributes: "
                         + ", ".join(sorted({x.split('+')[1].split('=')[0] for x in benign})) + ") |")
    for k in sorted(set(ia) - set(ib)):
        lines.append(f"| parse {k} | new in AFTER (not compared) |")
    return bad


def _compare_refruns(before: Path, after: Path, lines: list[str]) -> int:
    bad = 0
    ib = json.loads((before / "index.json").read_text())
    ia = json.loads((after / "index.json").read_text())
    for k in sorted(ib):
        b, a = ib[k], ia.get(k)
        if a is None:
            lines.append(f"| refrun {k} | MISSING in AFTER |")
            bad += 1
            continue
        if "error" in b or "error" in a:
            if b.get("error") != a.get("error"):
                lines.append(f"| refrun {k} | before: {b.get('error', 'ok')} | after: "
                             f"{a.get('error', 'ok')} |")
                bad += 1
            continue
        with np.load(before / b["file"]) as B, np.load(after / a["file"]) as A:
            diff = [n for n in sorted(set(A.files) | set(B.files))
                    if n not in A.files or n not in B.files
                    or not np.array_equal(A[n], B[n], equal_nan=True)]
            if diff:
                bad += 1
                detail = []
                for n in diff:
                    if n in A.files and n in B.files and A[n].shape == B[n].shape:
                        d = np.max(np.abs(A[n] - B[n]))
                        detail.append(f"{n} max|Δ| {d:.3g}")
                    else:
                        detail.append(f"{n} (present in one only or reshaped)")
                lines.append(f"| refrun {k} | CHANGED | " + "; ".join(detail) + " |")
    for k in sorted(set(ia) - set(ib)):
        lines.append(f"| refrun {k} | new in AFTER (not compared) |")
    return bad


def cmd_compare(a) -> int:
    before, after = Path(a.before), Path(a.after)
    lines = ["# Bit-identity probe", "", f"before: {before}", f"after:  {after}", "",
             "| item | status | detail |", "|---|---|---|"]
    bad = 0
    if (before / "runs").exists():
        bad += _compare_runs(before / "runs", after / "runs", lines)
    if (before / "parses").exists():
        bad += _compare_parses(before / "parses", after / "parses", lines)
    if (before / "refruns").exists():
        bad += _compare_refruns(before / "refruns", after / "refruns", lines)
    lines.append("")
    lines.append("RESULT: " + ("IDENTICAL (apart from benign rows above)" if bad == 0
                               else f"{bad} problem group(s) — see rows above"))
    text = "\n".join(lines)
    print(text)
    if a.out:
        Path(a.out).write_text(text + "\n", encoding="utf-8")
    return 0 if bad == 0 else 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("snapshot")
    s.add_argument("--tree", required=True)
    s.add_argument("--repo", default=None)
    s.add_argument("--out", required=True)
    s.add_argument("--cases", nargs="*", default=None, help="glob patterns on run-case names")
    s.add_argument("--extra-lattices", nargs="*", default=None, help="files, directories or globs")
    s.add_argument("--no-runs", action="store_true")
    s.add_argument("--no-parses", action="store_true")
    s.add_argument("--no-refruns", action="store_true")
    s.set_defaults(func=cmd_snapshot)
    c = sub.add_parser("compare")
    c.add_argument("--before", required=True)
    c.add_argument("--after", required=True)
    c.add_argument("--out", default=None)
    c.set_defaults(func=cmd_compare)
    a = ap.parse_args(argv)
    return a.func(a)


if __name__ == "__main__":
    sys.exit(main())
