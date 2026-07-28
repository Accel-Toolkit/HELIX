"""HALO-PIC M2b: production-N error/cost surface (the go/no-go sweep).

M3's verdict (docs/halo_pic/m3_gate_status.md): at N=20k the cost-matched
plain-coarse control beats every corrector configuration.  This sweep
re-measures the frontier at production N (1e5-2e5) where the sampling
floor drops ~1/sqrt(N) and deposit/gather (N-scaled) dominates the cost:

  arms per N in {1e5, 2e5}:
    * plain coarse at 24/32/48/64^3, step2=50  (the error/cost frontier)
    * 96^3 converged arm                        (empirical sampling floor)
    * K=1 anchored 32^3 (fine field at coarse nodes every kick)
                                                (corrector CEILING)
    * K=32 anchored 32^3, no net                (realistic anchor overhead)

All errors are along-s integrated vs the 10-seed N=200k/96^3/100-per-m
reference ensemble mean (same protocol as scripts/halo_m3_eval.py).

Decision rule: a corrector window exists iff the K=1 ceiling at 32^3
beats the plain-coarse configuration of EQUAL OR GREATER wall-clock on
tail metrics AND passes the 0.1 deg/cell tune gate with margin.

Usage:
    PYTHONPATH=. python3 scripts/halo_m2b_sweep.py \
        --out docs/halo_pic/m2b_production_n.json
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from scripts.halo_testbed import (CURRENT, EMIT_N, FREQ, LATTICE,  # noqa: E402
                                  W_KIN, matched_twiss, run_testbed)
from scripts.halo_m3_eval import integ_err, traj_metrics  # noqa: E402

METRICS = ("emit_x", "q999", "halo_x")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=str,
                    default="docs/halo_pic/m2b_production_n.json")
    ap.add_argument("--seeds-frontier", type=int, default=6)
    ap.add_argument("--seeds-floor", type=int, default=4)
    ap.add_argument("--seeds-ceiling", type=int, default=2)
    args = ap.parse_args()

    refs = sorted(glob.glob(
        str(REPO / "docs/halo_pic/baselines/ref_m1.4_seed*.npz")))
    assert len(refs) >= 8
    R = []
    for f in refs:
        d = np.load(f, allow_pickle=False)
        R.append(traj_metrics({k: d[k] for k in d.files if k != "meta"}
                              | {"wall_seconds": float(d["wall_seconds"])}))
    s_ref = R[0]["s"]
    ref_mean = {k: np.mean([r[k] for r in R], axis=0) for k in METRICS}
    ref_tune = float(np.mean([r["tune"] for r in R]))
    print(f"reference: {len(R)} seeds, tune {ref_tune:.3f}, "
          f"wall {np.mean([r['wall'] for r in R]):.1f}s")

    from linac_gen.io.tracewin_parser import parse_tracewin
    from linac_gen.core.particle import H_MINUS
    from linac_gen.core.reference import ReferenceParticle
    lat, _ = parse_tracewin(str(LATTICE))
    refp = ReferenceParticle(species=H_MINUS, w_kin=W_KIN, frequency=FREQ)
    tw = matched_twiss(lat, refp, CURRENT, EMIT_N / refp.bg)

    def run_arm(tag, n, grid, n_seeds, **kw):
        errs = {k: [] for k in METRICS}
        tunes, walls = [], []
        for seed in range(100, 100 + n_seeds):
            t0 = time.perf_counter()
            out = run_testbed(n=n, grid=grid, step2=50, mismatch=1.4,
                              seed=seed, twiss=tw, **kw)
            walls.append(time.perf_counter() - t0)
            m = traj_metrics(out)
            tunes.append(m["tune"])
            for k in METRICS:
                errs[k].append(integ_err(np.interp(s_ref, m["s"], m[k]),
                                         ref_mean[k]))
        res = {"n": n, "grid": grid,
               "wall": float(np.mean(walls)),
               "dtune": float(np.mean(tunes) - ref_tune),
               "dtune_std": float(np.std(tunes)),
               "seeds": n_seeds}
        res.update({f"err_{k}": float(np.mean(v)) for k, v in errs.items()})
        res.update({f"err_{k}_std": float(np.std(v))
                    for k, v in errs.items()})
        print(f"[{tag:22s}] wall {res['wall']:6.1f}s  "
              f"dtune {res['dtune']:+.3f}  "
              + "  ".join(f"{k} {res[f'err_{k}']*100:5.2f}%"
                          for k in METRICS), flush=True)
        return res

    report = {"reference_tune": ref_tune, "arms": {}}
    for n in (100_000, 200_000):
        nk = f"{n // 1000}k"
        for g in (24, 32, 48, 64):
            report["arms"][f"coarse_{nk}_{g}"] = run_arm(
                f"coarse {nk} {g}^3", n, g, args.seeds_frontier)
        report["arms"][f"floor_{nk}"] = run_arm(
            f"floor {nk} 96^3", n, 96, args.seeds_floor)
        report["arms"][f"ceiling_{nk}_32"] = run_arm(
            f"CEILING {nk} 32^3 K=1", n, 32, args.seeds_ceiling,
            sc_backend="halo",
            halo=dict(anchors=True, collect=False, k_init=1, k_min=1,
                      k_max=1, fine_factor=3, basis_degree=4))
        report["arms"][f"anchored_{nk}_32_K32"] = run_arm(
            f"anchored {nk} 32^3 K=32", n, 32, 1,
            sc_backend="halo",
            halo=dict(anchors=True, collect=False, k_init=32, k_min=32,
                      k_max=32, fine_factor=3, basis_degree=4))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=1))
    print(f"-> {out}")


if __name__ == "__main__":
    main()
