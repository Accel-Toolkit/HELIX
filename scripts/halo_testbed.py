"""HALO-PIC M1 benchmark driver: mismatched-FODO halo testbed.

The classic Gluckstern/Wangler setup — a space-charge-depressed FODO
channel with an rms-mismatched Gaussian beam drives the 2:1 parametric
(breathing) resonance and grows a halo.  This driver:

  1. finds the SC-matched input Twiss on the first declared cell of
     examples/halo_fodo.dat (envelope fixed point),
  2. applies a breathing-mode mismatch (beta -> m^2*beta, sizes x m),
  3. runs the MP PIC simulation with tail diagnostics enabled,
  4. dumps recorder arrays + the final particle distribution to .npz.

Usage (repo root):

    PYTHONPATH=. python3 scripts/halo_testbed.py \
        --n 20000 --grid 48 --mismatch 1.4 --seed 0 --out run.npz

    # fine-PIC reference ensemble (>= 10 seeds):
    PYTHONPATH=. python3 scripts/halo_testbed.py --reference

Every parameter that defines fidelity (n, grid, step2) is explicit so
the M2 sweep can drive this module programmatically via run_testbed().
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from linac_gen.core.beam import Beam
from linac_gen.core.config import SpaceChargeConfig
from linac_gen.core.particle import H_MINUS
from linac_gen.core.reference import ReferenceParticle
from linac_gen.core.simulation import Simulation
from linac_gen.io.tracewin_parser import parse_tracewin

REPO = Path(__file__).resolve().parents[1]
LATTICE = REPO / "examples" / "halo_fodo.dat"

# Beam definition (H- injector energy, MEBT-like)
W_KIN = 2.1226695          # MeV
FREQ = 162.5               # MHz
CURRENT = 5.0              # mA (peak)
EMIT_N = 0.21              # normalised rms emittance, pi mm mrad, x and y
EMIT_Z = 0.06231832        # pi deg MeV
ALPHA_Z, BETA_Z = 0.0, 819.05492

TAIL_FRACTIONS = (0.99, 0.999)


def matched_twiss(lattice, ref, current_mA: float, emit_geo: float) -> dict:
    """SC-matched input Twiss on the first declared cell."""
    from linac_gen.analysis.period_detect import detect_periods
    from linac_gen.matching.periodic import (
        find_matched_input_twiss, find_sc_matched_input_twiss)
    period = next(p for p in detect_periods(lattice)
                  if p.source == "lattice_card")
    c0, c1 = period.spans()[0]        # (start, end-exclusive)
    base = dict(emit_x=emit_geo, emit_y=emit_geo, emit_z=EMIT_Z,
                alpha_z=ALPHA_Z, beta_z=BETA_Z)
    if current_mA > 0:
        m = find_sc_matched_input_twiss(lattice, ref, c0, c1,
                                        current_mA, base)
    else:
        m = find_matched_input_twiss(lattice, ref, c0, c1)
    return {k: float(m[k]) for k in
            ("alpha_x", "beta_x", "alpha_y", "beta_y")}


def make_beam(n: int, seed: int, twiss: dict, mismatch: float,
              emit_geo: float) -> Beam:
    """Gaussian bunch at the (possibly mismatched) Twiss.

    Breathing-mode mismatch: beta -> mismatch^2 * beta in BOTH planes at
    fixed emittance and alpha, i.e. rms sizes scale by ``mismatch`` —
    the classic 2:1 halo drive.
    """
    rng = np.random.default_rng(seed)
    p = np.zeros((n, 6))
    for (i, j), a, b, e in (((0, 1), twiss["alpha_x"],
                             twiss["beta_x"] * mismatch**2, emit_geo),
                            ((2, 3), twiss["alpha_y"],
                             twiss["beta_y"] * mismatch**2, emit_geo),
                            ((4, 5), ALPHA_Z, BETA_Z, EMIT_Z)):
        g = (1 + a * a) / b
        cov = e * np.array([[b, -a], [-a, g]])
        L = np.linalg.cholesky(cov)
        p[:, (i, j)] = rng.standard_normal((n, 2)) @ L.T
    ref = ReferenceParticle(species=H_MINUS, w_kin=W_KIN, frequency=FREQ)
    beam = Beam(ref=ref, n_particles=n, current=CURRENT)
    beam.particles[:] = p
    return beam


def run_testbed(n: int = 20_000, grid: int = 48, step1: int = 100,
                step2: int = 50, mismatch: float = 1.4, seed: int = 0,
                current_mA: float = CURRENT,
                lattice_path: str | Path = LATTICE,
                twiss: dict | None = None,
                sc_backend: str = "numpy",
                halo: dict | None = None,
                halo_log: str | None = None) -> dict:
    """One testbed run; returns a dict of arrays + metadata.

    ``twiss`` can be passed in to skip the (deterministic) envelope
    matching when sweeping many fidelities of the same physics case.
    """
    lattice, _ = parse_tracewin(str(lattice_path))
    ref = ReferenceParticle(species=H_MINUS, w_kin=W_KIN, frequency=FREQ)
    emit_geo = EMIT_N / ref.bg
    if twiss is None:
        twiss = matched_twiss(lattice, ref, current_mA, emit_geo)
    beam = make_beam(n, seed, twiss, mismatch, emit_geo)
    beam.current = current_mA
    from linac_gen.core.step_config import StepConfig
    lattice.step_config = StepConfig(integration_steps_per_metre=step1,
                                     sc_steps_per_metre=step2)
    sc = SpaceChargeConfig(nx=grid, ny=grid, nz=grid,
                           sc_backend=sc_backend, halo=halo)
    sim = Simulation(lattice, beam, space_charge=sc,
                     tail_fractions=TAIL_FRACTIONS)
    t0 = time.perf_counter()
    res = sim.run()
    wall = time.perf_counter() - t0
    if halo_log and sim._pic_solver is not None and hasattr(
            sim._pic_solver, "save_log"):
        sim._pic_solver.save_log(halo_log)

    out = {
        "s": np.asarray(res.s, float),
        "sigma_x": np.asarray(res.sigma_x, float),
        "sigma_y": np.asarray(res.sigma_y, float),
        "emit_x": np.asarray(res.emit_x, float),
        "emit_y": np.asarray(res.emit_y, float),
        "halo_x": np.asarray(res.halo_x, float),
        "halo_y": np.asarray(res.halo_y, float),
        "x_max": np.asarray(res.x_max, float),
        "y_max": np.asarray(res.y_max, float),
        "transmission": np.asarray(res.transmission, float),
        "final_particles": beam.particles.copy(),
        "final_lost": beam.lost.copy(),
        "wall_seconds": wall,
    }
    for k, v in res.tail.items():
        out[f"tail_{k}"] = np.asarray(v, float)
    out["meta"] = json.dumps({
        "n": n, "grid": grid, "step1": step1, "step2": step2,
        "mismatch": mismatch, "seed": seed, "current_mA": current_mA,
        "w_kin": W_KIN, "freq": FREQ, "emit_n": EMIT_N,
        "twiss": twiss, "lattice": str(lattice_path),
        "tail_fractions": TAIL_FRACTIONS,
    })
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=20_000)
    ap.add_argument("--grid", type=int, default=48)
    ap.add_argument("--step1", type=int, default=100)
    ap.add_argument("--step2", type=int, default=50)
    ap.add_argument("--mismatch", type=float, default=1.4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--current", type=float, default=CURRENT)
    ap.add_argument("--out", type=str, default="halo_run.npz")
    ap.add_argument("--reference", action="store_true",
                    help="fine-PIC reference ensemble: N=200k, 96^3, "
                         "step2=100, seeds 0..9, into docs/halo_pic/baselines/")
    args = ap.parse_args()

    if args.reference:
        outdir = REPO / "docs" / "halo_pic" / "baselines"
        outdir.mkdir(parents=True, exist_ok=True)
        lattice, _ = parse_tracewin(str(LATTICE))
        ref = ReferenceParticle(species=H_MINUS, w_kin=W_KIN, frequency=FREQ)
        tw = matched_twiss(lattice, ref, args.current, EMIT_N / ref.bg)
        for seed in range(10):
            t0 = time.perf_counter()
            out = run_testbed(n=200_000, grid=96, step1=100, step2=100,
                              mismatch=args.mismatch, seed=seed,
                              current_mA=args.current, twiss=tw)
            f = outdir / f"ref_m{args.mismatch:g}_seed{seed}.npz"
            np.savez_compressed(f, **out)
            print(f"[ref seed {seed}] {time.perf_counter()-t0:8.1f}s -> {f}")
        return

    out = run_testbed(n=args.n, grid=args.grid, step1=args.step1,
                      step2=args.step2, mismatch=args.mismatch,
                      seed=args.seed, current_mA=args.current)
    np.savez_compressed(args.out, **out)
    ex_growth = out["emit_x"][-1] / out["emit_x"][0]
    q999 = out.get("tail_emit_x_q999")
    print(f"wall {out['wall_seconds']:.1f}s  eps_x growth x{ex_growth:.3f}  "
          f"eps_x_99.9 growth x{q999[-1]/q999[0]:.3f}  "
          f"transmission {out['transmission'][-1]:.2f}%")


if __name__ == "__main__":
    main()
