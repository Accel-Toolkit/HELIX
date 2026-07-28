"""HALO-PIC M2: coarse-PIC error surface on the mismatched-FODO testbed.

Sweeps (N_c, grid, sc cadence) against a fine reference and reports, for
each operating point, the error of core (sigma, eps_rms) and tail
(eps_99, eps_99.9, r_99.9, kurtosis halo) observables plus wall-clock —
the honest no-ML baseline that defines what the corrector must fix.

Error decomposition: for each axis, points that vary ONLY that axis from
the finest setting isolate grid vs cadence vs sampling contributions.

Usage (repo root; runs serially, each point seconds-to-minutes):

    PYTHONPATH=. python3 scripts/halo_coarse_sweep.py \
        --ref docs/halo_pic/baselines/ref_m1.4_seed0.npz \
        --out docs/halo_pic/coarse_sweep.csv

With --ref-config N,GRID,STEP2 the reference is (re)computed in-process
instead of loaded (used before the big baseline ensemble exists).
"""
from __future__ import annotations

import argparse
import csv
import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from scripts.halo_testbed import matched_twiss, run_testbed  # noqa: E402

# The sweep grid (kept modest for CPU; extend on the CUDA box)
N_LIST = (2_000, 5_000, 10_000, 20_000, 50_000)
GRID_LIST = (24, 32, 48)
STEP2_LIST = (25, 50, 100)
SEEDS = (0, 1, 2)
MISMATCH = 1.4


def _metrics(out: dict) -> dict:
    """End-of-line observables of one run."""
    return {
        "sigma_x": float(out["sigma_x"][-1]),
        "emit_x": float(out["emit_x"][-1]),
        "emit_x_q99": float(out["tail_emit_x_q99"][-1]),
        "emit_x_q999": float(out["tail_emit_x_q999"][-1]),
        "r_q999": float(out["tail_r_q999"][-1]),
        "halo_x": float(out["halo_x"][-1]),
        "wall": float(out["wall_seconds"]),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ref", type=str, default="",
                    help=".npz reference run (from halo_testbed.py)")
    ap.add_argument("--ref-config", type=str, default="100000,96,100",
                    help="N,GRID,STEP2 to compute the reference in-process")
    ap.add_argument("--out", type=str,
                    default=str(REPO / "docs" / "halo_pic" /
                                "coarse_sweep.csv"))
    ap.add_argument("--fast", action="store_true",
                    help="reduced sweep (single seed, no 50k) for smoke")
    args = ap.parse_args()

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)

    # Matched Twiss once (deterministic, shared by every run)
    from linac_gen.core.particle import H_MINUS
    from linac_gen.core.reference import ReferenceParticle
    from linac_gen.io.tracewin_parser import parse_tracewin
    from scripts.halo_testbed import CURRENT, EMIT_N, FREQ, LATTICE, W_KIN
    lattice, _ = parse_tracewin(str(LATTICE))
    ref_p = ReferenceParticle(species=H_MINUS, w_kin=W_KIN, frequency=FREQ)
    twiss = matched_twiss(lattice, ref_p, CURRENT, EMIT_N / ref_p.bg)
    print(f"matched twiss: {twiss}")

    # Reference metrics (per seed)
    ref_metrics: dict[int, dict] = {}
    if args.ref:
        data = np.load(args.ref, allow_pickle=False)
        meta = json.loads(str(data["meta"]))
        out = {k: data[k] for k in data.files if k != "meta"}
        out["wall_seconds"] = float(data["wall_seconds"])
        ref_metrics[int(meta["seed"])] = _metrics(out)
        ref_cfg = (meta["n"], meta["grid"], meta["step2"])
    else:
        n_r, g_r, s_r = (int(x) for x in args.ref_config.split(","))
        ref_cfg = (n_r, g_r, s_r)
        for seed in (SEEDS[:1] if args.fast else SEEDS):
            t0 = time.perf_counter()
            out = run_testbed(n=n_r, grid=g_r, step2=s_r, mismatch=MISMATCH,
                              seed=seed, twiss=twiss)
            ref_metrics[seed] = _metrics(out)
            print(f"[ref seed {seed}] {time.perf_counter()-t0:7.1f}s "
                  f"eps999={ref_metrics[seed]['emit_x_q999']:.4f}")
    print(f"reference config: N={ref_cfg[0]} grid={ref_cfg[1]} "
          f"step2={ref_cfg[2]}")

    n_list = N_LIST[:-1] if args.fast else N_LIST
    seeds = SEEDS[:1] if args.fast else SEEDS
    rows = []
    for n, grid, step2 in itertools.product(n_list, GRID_LIST, STEP2_LIST):
        for seed in seeds:
            if seed not in ref_metrics:
                continue
            t0 = time.perf_counter()
            out = run_testbed(n=n, grid=grid, step2=step2,
                              mismatch=MISMATCH, seed=seed, twiss=twiss)
            m = _metrics(out)
            r = ref_metrics[seed]
            row = {"n": n, "grid": grid, "step2": step2, "seed": seed,
                   "wall": m["wall"]}
            for k in ("sigma_x", "emit_x", "emit_x_q99", "emit_x_q999",
                      "r_q999", "halo_x"):
                row[f"{k}"] = m[k]
                row[f"err_{k}"] = abs(m[k] - r[k]) / max(abs(r[k]), 1e-30)
            rows.append(row)
            print(f"N={n:6d} grid={grid:2d} step2={step2:3d} seed={seed} "
                  f"wall={m['wall']:6.1f}s  "
                  f"err(eps)={row['err_emit_x']*100:5.2f}%  "
                  f"err(eps999)={row['err_emit_x_q999']*100:5.2f}%")

    with open(args.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {args.out} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
