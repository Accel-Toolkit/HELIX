"""Parse/write snapshot of every tracked example deck — the before/after proof for parser and writer changes.

    # reference tree (a git worktree of the commit before the change)
    PYTHONPATH=/path/to/before python3 scripts/parse_snapshot.py snapshot --tree /path/to/before --out /tmp/ps_before
    # working tree
    PYTHONPATH=. python3 scripts/parse_snapshot.py snapshot --tree . --out /tmp/ps_after
    # compare (``--ignore label`` skips the attribute the change is allowed to add)
    python3 scripts/parse_snapshot.py compare --before /tmp/ps_before --after /tmp/ps_after --ignore label

For every deck listed by ``git ls-files`` under ``examples/`` and ``tests/io/fixtures/`` the snapshot records, per
element, the class name, the ``name`` and every scalar attribute (floats as ``repr``), the parser warnings, and the
text ``write_tracewin`` produces with any ``LABEL: `` prefix stripped from each line — so a change that only adds
deck labels must leave every snapshot identical except for the ignored attribute.
"""
from __future__ import annotations
import argparse, contextlib, io, json, os, re, subprocess, sys, tempfile, warnings
from pathlib import Path

_LABEL_PREFIX = re.compile(r"^([A-Za-z][^\s:]*)\s*:\s*(?=\S)")


def _assert_tree(tree: Path) -> None:
    import linac_gen
    got = Path(linac_gen.__file__).resolve().parent.parent
    if got != tree.resolve():
        sys.exit(f"REFUSED: linac_gen imported from {got}, not from --tree {tree.resolve()} (set PYTHONPATH)")


def _decks(tree: Path) -> list[str]:
    out = subprocess.run(["git", "-C", str(tree), "ls-files", "--", "examples/*.dat", "examples/**/*.dat",
                          "tests/io/fixtures/*.dat"], capture_output=True, text=True, check=True).stdout.split()
    return sorted(set(out))


def _scalars(e, ignore: set[str]) -> dict:
    out = {}
    for k, v in vars(e).items():
        if k in ignore or k.startswith("_"):
            continue
        if v is None or isinstance(v, (bool, int, str)):
            out[k] = v
        elif isinstance(v, float):
            out[k] = repr(v)
    return out


def snapshot_deck(tree: Path, rel: str, ignore: set[str]) -> dict:
    from linac_gen.io.tracewin_parser import parse_tracewin
    from linac_gen.io.tracewin_writer import write_tracewin
    path = tree / rel
    rec: dict = {"deck": rel}
    cwd = os.getcwd()
    try:
        os.chdir(path.parent)      # relative FIELD_MAP_PATHs resolve against the deck's directory
        with warnings.catch_warnings(record=True) as w, contextlib.redirect_stdout(io.StringIO()), \
                contextlib.redirect_stderr(io.StringIO()):
            warnings.simplefilter("always")
            try:
                lat, meta = parse_tracewin(str(path))
            except Exception as exc:            # noqa: BLE001
                rec["parse_error"] = f"{type(exc).__name__}: {exc}"
                return rec
            rec["parse_warnings"] = list((meta or {}).get("warnings", []) or [])
            rec["py_warnings"] = sorted({str(x.message) for x in w})
            rec["elements"] = [{"type": type(e).__name__, "name": getattr(e, "name", None),
                                "attrs": _scalars(e, ignore)} for e in lat.elements]
            try:
                with tempfile.TemporaryDirectory() as td:
                    out = Path(td) / "w.dat"
                    write_tracewin(lat, str(out))
                    text = out.read_text(encoding="latin-1")
                rec["written"] = [_LABEL_PREFIX.sub("", ln) for ln in text.splitlines()]
            except Exception as exc:            # noqa: BLE001
                rec["write_error"] = f"{type(exc).__name__}: {exc}"
    finally:
        os.chdir(cwd)
    return rec


def cmd_snapshot(a) -> int:
    tree = Path(a.tree).resolve(); _assert_tree(tree)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    ignore = set(a.ignore or [])
    decks = _decks(tree)
    for i, rel in enumerate(decks, 1):
        rec = snapshot_deck(tree, rel, ignore)
        text = json.dumps(rec, indent=0, sort_keys=True).replace(str(tree), "<TREE>")   # absolute paths differ per checkout
        (out / (rel.replace("/", "__") + ".json")).write_text(text, encoding="utf-8")
        print(f"[{i:3d}/{len(decks)}] {rel}: " + ("ERROR " + rec["parse_error"] if "parse_error" in rec else
              f"{len(rec['elements'])} elements, {len(rec.get('parse_warnings', []))} warnings"), flush=True)
    (out / "_index.json").write_text(json.dumps(decks, indent=0), encoding="utf-8")
    return 0


def cmd_compare(a) -> int:
    before, after = Path(a.before), Path(a.after)
    decks = json.loads((before / "_index.json").read_text(encoding="utf-8"))
    ignore = set(a.ignore or []); n_same = n_diff = n_missing = 0
    for rel in decks:
        fb = before / (rel.replace("/", "__") + ".json"); fa = after / (rel.replace("/", "__") + ".json")
        if not fa.exists():
            n_missing += 1; print(f"MISSING after: {rel}"); continue
        b = json.loads(fb.read_text(encoding="utf-8")); c = json.loads(fa.read_text(encoding="utf-8"))
        for rec in (b, c):
            for e in rec.get("elements", []):
                for k in ignore: e["attrs"].pop(k, None)
        if b == c:
            n_same += 1; continue
        n_diff += 1; print(f"DIFF: {rel}")
        for key in sorted(set(b) | set(c)):
            if b.get(key) != c.get(key):
                if key == "elements":
                    for j, (eb, ec) in enumerate(zip(b[key], c[key])):
                        if eb != ec:
                            print(f"   element {j}: {eb} != {ec}"); break
                    if len(b[key]) != len(c[key]): print(f"   element count {len(b[key])} != {len(c[key])}")
                elif key == "written":
                    for j, (lb, lc) in enumerate(zip(b[key], c[key])):
                        if lb != lc:
                            print(f"   written line {j}: {lb!r} != {lc!r}"); break
                    if len(b[key]) != len(c[key]): print(f"   written line count {len(b[key])} != {len(c[key])}")
                else:
                    print(f"   {key}: {str(b.get(key))[:200]} != {str(c.get(key))[:200]}")
    print(f"\n{n_same} identical, {n_diff} different, {n_missing} missing (ignored attrs: {sorted(ignore)})")
    return 0 if (n_diff == 0 and n_missing == 0) else 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("snapshot"); s.add_argument("--tree", required=True); s.add_argument("--out", required=True)
    s.add_argument("--ignore", nargs="*", default=[]); s.set_defaults(func=cmd_snapshot)
    c = sub.add_parser("compare"); c.add_argument("--before", required=True); c.add_argument("--after", required=True)
    c.add_argument("--ignore", nargs="*", default=[]); c.set_defaults(func=cmd_compare)
    a = ap.parse_args(argv); return a.func(a)


if __name__ == "__main__":
    sys.exit(main())
