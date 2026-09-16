#!/usr/bin/env python3
"""Before/after harness for core-physics changes.

Runs a fixed set of example lattices in envelope, matrix and multi-particle
mode with whichever HELIX tree is on ``PYTHONPATH`` and stores every
comparable array, so two trees (e.g. a git worktree at the previous commit
and the working tree) can be compared bit for bit and the physically
changed cases quantified.

    # snapshot with the reference tree
    PYTHONPATH=/path/to/before python3 scripts/physics_fix_report.py snapshot \
        --tree /path/to/before --out /tmp/snap_before

    # snapshot with the working tree
    PYTHONPATH=. python3 scripts/physics_fix_report.py snapshot --tree . --out /tmp/snap_after

    # compare
    python3 scripts/physics_fix_report.py compare --before /tmp/snap_before --after /tmp/snap_after

``snapshot`` refuses to run if ``linac_gen`` was imported from outside
``--tree``.  Cases are selected with ``--cases`` (glob on the case name);
``--repo`` points at the checkout whose ``examples/`` are run (defaults to
``--tree``), so the same decks can be run through two code trees, or two
deck sets through one code tree.
"""
from __future__ import annotations

import argparse
import contextlib
import fnmatch
import io
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Case table: name -> (input, mode, overrides)
#   input: a .lgproj (beam from the project) or a .dat/.madx (beam from
#          overrides); mode: env | matrix | mp
#   overrides: BeamConfig fields; "n" = MP particle count; "seed"
# ---------------------------------------------------------------------------
H800 = dict(species="H-", energy=800.0, frequency=162.5)
MEBT = dict(species="H-", energy=2.1226695, frequency=162.5)
FODO = dict(species="proton", energy=800.0, frequency=352.21)

CASES: dict[str, tuple[str, str, dict]] = {}


def _add(name, path, modes, **ov):
    for m in modes:
        CASES[f"{name}:{m}"] = (path, m, dict(ov))


# --- affected families ------------------------------------------------------
_add("bal_branch", "examples/pipii/bal/bal_branch.lgproj", ("env", "matrix", "mp"), n=5000)
_add("bal_branch_0mA", "examples/pipii/bal/bal_branch.lgproj", ("env",), current=0.0)
_add("mebt_to_bal", "examples/pipii/bal/mebt_to_bal.lgproj", ("env", "matrix"))
_add("btl", "examples/pipii/btl/btl.lgproj", ("env", "matrix", "mp"), n=5000)
_add("btl_0mA", "examples/pipii/btl/btl.lgproj", ("env", "mp"), current=0.0, n=5000)
_add("btl_with_foil", "examples/pipii/btl/btl_with_foil.lgproj", ("env", "matrix"))
_add("btl_2025v0703", "examples/pipii/btl/btl_2025v0703.lgproj", ("env", "matrix"))
_add("btl_2025v0703_L2340", "examples/pipii/btl/btl_2025v0703_L2340.lgproj", ("env", "matrix"))
_add("mebt_to_foil", "examples/MEBT_To_Foil/mebt_to_foil.lgproj", ("env", "matrix"))
_add("stage6_btl_booster", "examples/commissioning/stage6_btl_booster/stage6_btl_booster.lgproj", ("env", "matrix"))
# --- must-not-change controls ---------------------------------------------------
_add("fodo_cell", "examples/fodo_cell.dat", ("env", "matrix", "mp"), n=2000, **MEBT)
_add("halo_fodo", "examples/halo_fodo.dat", ("env", "mp"), n=2000, **MEBT)
_add("bend_line", "examples/bend_line.dat", ("env", "matrix", "mp"), n=2000, **MEBT)
_add("csr_chicane", "examples/csr_chicane.lgproj", ("env", "matrix", "mp"), n=2000)
_add("chicane_batch", "examples/batch_mode/chicane.lgproj", ("env",))
_add("coupled_fodo", "examples/impactx_features/coupled_fodo.dat", ("env", "matrix", "mp"), n=2000, **MEBT)
_add("dtl_section", "examples/dtl_section.dat", ("env", "matrix", "mp"), n=2000, species="proton", energy=3.0, frequency=352.21)
_add("solenoid_channel", "examples/solenoid_channel.dat", ("env", "matrix", "mp"), n=2000, **MEBT)
_add("matching_demo", "examples/matching_demo.dat", ("env", "matrix"), **MEBT)
_add("correction_demo", "examples/correction_demo/correction_demo.dat", ("env", "matrix", "mp"), n=2000, **MEBT)
_add("lebt_scc_demo", "examples/lebt_scc_demo.lgproj", ("env",))
_add("hofmann_demo", "examples/hofmann_stability/hofmann_demo_linac.dat", ("env", "matrix"), species="proton", energy=3.0, frequency=352.21)
_add("fodo_madx", "examples/madx/fodo.madx", ("env", "matrix", "mp"), n=2000, **FODO)
_add("fnalscl", "examples/piplattice/fnalscl.lgproj", ("env", "matrix"))
_add("mebt_pipii", "examples/pipii/mebt/mebt.lgproj", ("env", "matrix"))
_add("stage5_btl_absorber", "examples/commissioning/stage5_btl_absorber/stage5_btl_absorber.lgproj", ("env",))

RECORD_FIELDS = ("s", "sigma_x", "sigma_y", "sigma_phi", "sigma_w", "emit_x", "emit_y",
                 "emit_z", "alpha_x", "beta_x", "alpha_y", "beta_y", "transmission",
                 "ref_w_kin", "centroid_x", "centroid_y", "centroid_xp", "centroid_yp",
                 # the full 6x6 per record: a bit change confined to an off-diagonal
                 # (xy-coupling) entry is invisible in every projected scalar above
                 "sigma_matrix",
                 # per-element tangent maps, present only when the run carried the
                 # phase probe; an empty list is skipped so both trees agree on the
                 # file set
                 "element_maps_dep")


def _quiet():
    return contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO())


def _assert_tree(tree: Path) -> None:
    import linac_gen
    got = Path(linac_gen.__file__).resolve()
    if tree.resolve() not in got.parents:
        sys.exit(f"linac_gen imported from {got}, not from --tree {tree}: fix PYTHONPATH")


def _load(repo: Path, rel: str):
    from linac_gen.cli import common
    p = repo / rel
    if not p.exists():
        return None, None
    o1, o2 = _quiet()
    with o1, o2:
        lat, cfg, conv = common.load_input(str(p))
    return lat, cfg


def _apply(cfg, ov: dict):
    for k, v in ov.items():
        if k in ("n", "seed"):
            continue
        setattr(cfg, k, v)


def _dispersion_at_end(M6, ref):
    """eta_x, eta_x', eta_y, eta_y' [m, rad] from the 6x6 in MAD-X units."""
    from linac_gen.tracking.longitudinal_coords import matrix_to_madx
    R = matrix_to_madx(M6, ref)
    return np.array([R[0, 5], R[1, 5], R[2, 5], R[3, 5]])


def run_case(repo: Path, name: str, spec) -> dict:
    from linac_gen.cli import common
    from linac_gen.core.config import SpaceChargeConfig
    from linac_gen.core.step_config import StepConfig

    rel, mode, ov = spec
    lat, cfg = _load(repo, rel)
    if lat is None:
        return {"skipped": f"missing {rel}"}
    _apply(cfg, ov)
    out: dict = {}
    t0 = time.time()
    o1, o2 = _quiet()
    if mode == "env":
        with o1, o2:
            res = common.run_envelope_sim(lat, cfg)
        for f in RECORD_FIELDS:
            a = getattr(res, f, None)
            if a is None or (hasattr(a, "__len__") and len(a) == 0):
                continue
            out[f] = np.asarray(a, dtype=float)
    elif mode == "matrix":
        with o1, o2:
            M, twiss = common.run_matrix(lat, cfg)
        ref = common.build_ref(cfg)
        out["M6"] = np.asarray(M, dtype=float)
        out["eta_end"] = _dispersion_at_end(out["M6"], ref)
        for plane in ("x", "y"):
            tw = twiss.get(plane, {})
            out[f"twiss_{plane}"] = np.array([tw.get("beta", np.nan), tw.get("alpha", np.nan),
                                              tw.get("phase_advance", np.nan)], dtype=float)
        # per-element 6x6 of every bend (element name -> matrix)
        from linac_gen.elements.dipole import Dipole
        from linac_gen.elements.edge import Edge
        from linac_gen.tracking.matrix_tracking import get_element_matrix
        rc = ref.copy()
        names = []
        mats = []
        for el in lat.elements:
            if isinstance(el, (Dipole, Edge)):
                names.append(el.name)
                mats.append(np.asarray(get_element_matrix(el, rc.copy()), dtype=float))
            rc.s += el.length
        out["bend_names"] = np.array(names)
        out["bend_mats"] = np.array(mats) if mats else np.zeros((0, 6, 6))
    elif mode == "mp":
        n = int(ov.get("n", 2000))
        cfg.n_particles = n
        sc = SpaceChargeConfig(nx=24, ny=24, nz=24)
        with o1, o2:
            rec, beam = common.run_mp_sim(lat, cfg, sc, StepConfig(), seed=int(ov.get("seed", 42)))
        for f in RECORD_FIELDS:
            a = getattr(rec, f, None)
            if a is not None:
                out[f] = np.asarray(a, dtype=float)
        out["particles"] = np.asarray(beam.particles, dtype=float)
        out["lost"] = np.asarray(beam.lost, dtype=bool)
    out["_wall_s"] = np.array([time.time() - t0])
    return out


def cmd_snapshot(a) -> int:
    tree = Path(a.tree)
    _assert_tree(tree)
    repo = Path(a.repo) if a.repo else tree
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    os.chdir(repo)
    manifest = {}
    for name, spec in CASES.items():
        if a.cases and not any(fnmatch.fnmatch(name, pat) for pat in a.cases):
            continue
        try:
            res = run_case(repo, name, spec)
        except Exception as exc:  # noqa: BLE001 - record, keep going
            res = {"error": f"{type(exc).__name__}: {exc}"}
        if "skipped" in res or "error" in res:
            manifest[name] = res
            print(f"{name:>32}: {res}")
            continue
        np.savez_compressed(out / (name.replace(":", "__") + ".npz"), **res)
        manifest[name] = {"wall_s": float(res["_wall_s"][0])}
        print(f"{name:>32}: {res['_wall_s'][0]:6.1f} s")
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    return 0


def _last(a):
    a = np.asarray(a, dtype=float)
    return float(a[-1]) if a.size else float("nan")


def summarise(d: dict) -> dict:
    s = {}
    for f in ("sigma_x", "sigma_y", "sigma_phi", "sigma_w", "emit_x", "emit_y", "emit_z",
              "alpha_x", "beta_x", "alpha_y", "beta_y", "transmission"):
        if f in d:
            s[f"{f}_end"] = _last(d[f])
    if "eta_end" in d:
        for i, k in enumerate(("eta_x", "eta_xp", "eta_y", "eta_yp")):
            s[k + "_end"] = float(d["eta_end"][i])
    if "twiss_x" in d:
        s["beta_x_periodic"] = float(d["twiss_x"][0])
        s["beta_y_periodic"] = float(d["twiss_y"][0])
    if "lost" in d:
        s["n_lost"] = int(np.asarray(d["lost"]).sum())
    return s


def cmd_compare(a) -> int:
    before, after = Path(a.before), Path(a.after)
    rows = []
    for f in sorted(after.glob("*.npz")):
        name = f.stem.replace("__", ":")
        g = before / f.name
        if not g.exists():
            rows.append((name, "no BEFORE snapshot", {}, {}))
            continue
        A, B = np.load(f, allow_pickle=False), np.load(g, allow_pickle=False)
        fa, fb = set(A.files) - {"_wall_s"}, set(B.files) - {"_wall_s"}
        if fa != fb:
            # intersecting the key sets would let a field that disappeared in
            # AFTER pass silently; report it as a change with the delta named
            rows.append((name, "CHANGED (field set differs: -%s +%s)" % (
                ",".join(sorted(fb - fa)) or "none", ",".join(sorted(fa - fb)) or "none"),
                summarise(B), summarise(A)))
            continue
        keys = [k for k in A.files if k in B.files and k != "_wall_s" and A[k].dtype.kind in "fb"]
        ident = all(A[k].shape == B[k].shape and np.array_equal(A[k], B[k], equal_nan=True)
                    for k in keys)
        rows.append((name, "bit-identical" if ident else "CHANGED", summarise(B), summarise(A)))
    lines = ["| case | status | quantity | before | after | Δ |", "|---|---|---|---|---|---|"]
    n_changed = 0
    for name, status, sb, sa in rows:
        if not status.startswith("CHANGED"):
            lines.append(f"| {name} | {status} | | | | |")
            continue
        n_changed += 1
        if status != "CHANGED":
            lines.append(f"| {name} | {status} | | | | |")
            continue
        first = True
        for k in sa:
            vb, va = sb.get(k, float("nan")), sa[k]
            if isinstance(va, float) and isinstance(vb, float) and (np.isnan(va) and np.isnan(vb)):
                continue
            if va == vb:
                continue
            lines.append(f"| {name if first else ''} | {'CHANGED' if first else ''} | {k} | {vb:.6g} | {va:.6g} | {va - vb:+.3g} |")
            first = False
        if first:
            lines.append(f"| {name} | CHANGED (arrays differ; end-of-line scalars equal) | | | | |")
    text = "\n".join(lines)
    print(text)
    n_missing = sum(1 for _n, s, _b, _a in rows if s == "no BEFORE snapshot")
    n_ident = len(rows) - n_changed - n_missing
    print(f"\n{n_ident} bit-identical, {n_changed} changed, {n_missing} not compared")
    if n_missing:
        # counting an uncompared case as identical once reported "22 bit-identical,
        # 0 changed" for a run that had compared nothing at all
        print(f"REFUSED: {n_missing} case(s) have no BEFORE snapshot — the comparison is incomplete",
              file=sys.stderr)
    if a.out:
        Path(a.out).write_text(text + "\n", encoding="utf-8")
    return 1 if n_missing else 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("snapshot")
    s.add_argument("--tree", required=True, help="checkout whose linac_gen must be the one imported")
    s.add_argument("--repo", default=None, help="checkout whose examples/ are run (default: --tree)")
    s.add_argument("--out", required=True)
    s.add_argument("--cases", nargs="*", default=None, help="glob patterns on case names")
    s.set_defaults(func=cmd_snapshot)
    c = sub.add_parser("compare")
    c.add_argument("--before", required=True)
    c.add_argument("--after", required=True)
    c.add_argument("--out", default=None, help="write the Markdown table here")
    c.set_defaults(func=cmd_compare)
    a = ap.parse_args(argv)
    return a.func(a)


if __name__ == "__main__":
    sys.exit(main())
