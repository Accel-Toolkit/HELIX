"""HALO-PIC M3 acceptance evaluation.

Compares, over >= 8 evaluation seeds on the mismatched-FODO testbed:

    fine reference (docs/halo_pic/baselines/, N=200k 96^3)
    raw coarse     (operating point N=20k 24^3 25/m)
    HALO-PIC       (same operating point + trained corrector + anchors)

Metrics (per the M2 amendments — along-s INTEGRATED errors, not 3-seed
endpoints):

  * rms-eps trajectory error:   mean_s |eps(s) - eps_ref(s)| / eps_ref(s)
  * eps_99.9 trajectory error:  same on tail_emit_x_q999
  * halo kurtosis trajectory error
  * envelope tune gate: dominant sigma_x(s) oscillation frequency
    (Hann-windowed, zero-padded, parabolic-interpolated FFT) vs the
    reference, in deg/cell — target |delta| < 0.1 deg/cell
  * wall-clock per arm

The reference curve at each s is the MEAN over the reference ensemble;
tolerances are judged against the ensemble's own seed scatter.

Usage:
    PYTHONPATH=. python3 scripts/halo_m3_eval.py --net /tmp/halo_m3/net
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

CELL_MM = 400.0


def tune_deg_per_cell(s_mm: np.ndarray, sig: np.ndarray) -> float:
    """Dominant oscillation frequency of sig(s) in deg per FODO cell."""
    s = np.asarray(s_mm, float)
    y = np.asarray(sig, float)
    y = y - y.mean()
    # uniform resample (records are per element, nearly uniform already)
    su = np.linspace(s[0], s[-1], 4 * len(s))
    yu = np.interp(su, s, y)
    yu *= np.hanning(len(yu))
    pad = 64 * len(yu)
    f = np.fft.rfft(yu, n=pad)
    freqs = np.fft.rfftfreq(pad, d=(su[1] - su[0]))     # cycles / mm
    k = np.argmax(np.abs(f[1:])) + 1
    # parabolic peak interpolation
    if 1 <= k < len(f) - 1:
        a, b, c = np.abs(f[k - 1]), np.abs(f[k]), np.abs(f[k + 1])
        dk = 0.5 * (a - c) / (a - 2 * b + c + 1e-300)
    else:
        dk = 0.0
    f_peak = freqs[k] + dk * (freqs[1] - freqs[0])
    return float(f_peak * CELL_MM * 360.0)              # deg / cell


def traj_metrics(out: dict) -> dict:
    return {
        "s": out["s"],
        "emit_x": out["emit_x"],
        "q999": out["tail_emit_x_q999"],
        "halo_x": out["halo_x"],
        "sigma_x": out["sigma_x"],
        "wall": out["wall_seconds"],
        "tune": tune_deg_per_cell(out["s"], out["sigma_x"]),
    }


def integ_err(y: np.ndarray, ref: np.ndarray) -> float:
    """Mean relative trajectory error (skip the first 10% — transient)."""
    n0 = len(y) // 10
    return float(np.mean(np.abs(y[n0:] - ref[n0:])
                         / np.maximum(np.abs(ref[n0:]), 1e-30)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--net", type=str, required=True)
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--step2", type=int, default=50)
    ap.add_argument("--fine-factor", type=int, default=2)
    ap.add_argument("--out", type=str, default="docs/halo_pic/m3_eval.json")
    args = ap.parse_args()

    # ---- reference ensemble --------------------------------------------
    refs = sorted(glob.glob(str(REPO / "docs/halo_pic/baselines/ref_m1.4_seed*.npz")))
    assert len(refs) >= 8, f"need >=8 reference seeds, found {len(refs)}"
    R = []
    for f in refs:
        d = np.load(f, allow_pickle=False)
        R.append(traj_metrics({k: d[k] for k in d.files if k != "meta"}
                              | {"wall_seconds": float(d["wall_seconds"])}))
    s_ref = R[0]["s"]
    ref_mean = {k: np.mean([r[k] for r in R], axis=0)
                for k in ("emit_x", "q999", "halo_x", "sigma_x")}
    ref_tunes = np.array([r["tune"] for r in R])
    ref_wall = float(np.mean([r["wall"] for r in R]))
    # reference self-scatter (the tolerance floor): per-seed integrated
    # deviation from the ensemble mean
    scatter = {k: np.mean([integ_err(r[k], ref_mean[k]) for r in R])
               for k in ("emit_x", "q999", "halo_x")}
    print(f"reference: {len(R)} seeds, wall {ref_wall:.1f}s, "
          f"tune {ref_tunes.mean():.3f} ± {ref_tunes.std():.3f} deg/cell")
    print(f"reference self-scatter: " +
          "  ".join(f"{k}={v*100:.2f}%" for k, v in scatter.items()))

    # ---- evaluation arms -------------------------------------------------
    from linac_gen.io.tracewin_parser import parse_tracewin
    from linac_gen.core.particle import H_MINUS
    from linac_gen.core.reference import ReferenceParticle
    lat, _ = parse_tracewin(str(LATTICE))
    refp = ReferenceParticle(species=H_MINUS, w_kin=W_KIN, frequency=FREQ)
    tw = matched_twiss(lat, refp, CURRENT, EMIT_N / refp.bg)

    def run_arm(tag: str, **kw) -> dict:
        errs = {"emit_x": [], "q999": [], "halo_x": []}
        tunes, walls = [], []
        for seed in range(100, 100 + args.seeds):        # disjoint from refs
            out = run_testbed(n=20_000, grid=24, step2=args.step2,
                              mismatch=1.4, seed=seed, twiss=tw, **kw)
            m = traj_metrics(out)
            y = {k: np.interp(s_ref, m["s"], m[k])
                 for k in ("emit_x", "q999", "halo_x")}
            for k in errs:
                errs[k].append(integ_err(y[k], ref_mean[k]))
            tunes.append(m["tune"])
            walls.append(m["wall"])
        res = {f"err_{k}": float(np.mean(v)) for k, v in errs.items()}
        res["tune"] = float(np.mean(tunes))
        res["dtune"] = float(np.mean(tunes) - ref_tunes.mean())
        res["wall"] = float(np.mean(walls))
        print(f"[{tag:9s}] wall {res['wall']:6.1f}s  "
              f"err eps={res['err_emit_x']*100:5.2f}%  "
              f"q999={res['err_q999']*100:5.2f}%  "
              f"halo={res['err_halo_x']*100:5.2f}%  "
              f"dtune={res['dtune']:+.4f} deg/cell")
        return res

    t0 = time.perf_counter()
    coarse = run_arm("coarse", sc_backend="numpy")
    halo_cfg = dict(anchors=True, collect=False, k_init=8, k_min=2,
                    k_max=64, fine_factor=args.fine_factor, basis_degree=4,
                    corrector_dir=args.net)
    halo = run_arm("halo-pic", sc_backend="halo", halo=halo_cfg)

    report = {"reference": {"wall": ref_wall,
                            "tune": float(ref_tunes.mean()),
                            "tune_scatter": float(ref_tunes.std()),
                            "self_scatter": {k: float(v)
                                             for k, v in scatter.items()}},
              "coarse": coarse, "halo": halo,
              "speedup_vs_fine": ref_wall / halo["wall"],
              "overhead_vs_coarse": halo["wall"] / coarse["wall"],
              "gates": {
                  "tail_improvement": coarse["err_q999"] / max(
                      halo["err_q999"], 1e-12),
                  "tune_gate_deg_per_cell": abs(halo["dtune"]),
              }}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=1))
    print(f"\nspeedup vs fine: {report['speedup_vs_fine']:.1f}x   "
          f"overhead vs coarse: {report['overhead_vs_coarse']:.2f}x")
    print(f"GATES: tail improvement x{report['gates']['tail_improvement']:.2f}"
          f" (need >=2)   |dtune| = "
          f"{report['gates']['tune_gate_deg_per_cell']:.4f} deg/cell "
          f"(need < 0.1)")
    print(f"eval wall total {time.perf_counter()-t0:.0f}s -> {args.out}")


if __name__ == "__main__":
    main()
