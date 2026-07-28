"""Auto-adjust demo: TraceWin SET / ADJUST cards driving the matcher.

Loads ``examples/matching_demo.dat`` (a short FODO-style cell with two
ADJUST cards on linked quad gradients plus one SET_SIZE constraint at
the exit), runs the matcher, prints a report, and saves a plot of the
σ_x(s) envelope before vs after matching to
``examples/matching_demo_compare.png``.

Run::

    PYTHONPATH=. python3 examples/matching_demo.py
"""
from __future__ import annotations

import copy
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from linac_gen.core.config import BeamConfig
from linac_gen.distributions.factory import create_beam
from linac_gen.io.tracewin_parser import parse_tracewin
from linac_gen.matching import match
from linac_gen.tracking.envelope import EnvelopeSolver


HERE = Path(__file__).parent
LATTICE = HERE / "matching_demo.dat"
OUT_PNG = HERE / "matching_demo_compare.png"


def _beam_cfg() -> BeamConfig:
    return BeamConfig(
        species="proton", energy=3.0, frequency=352.21,
        current=0.0, duty_cycle=100.0,
        n_particles=10, distribution="waterbag", cutoff=3.0,
        emit_nx=0.25, alpha_x=0.0, beta_x=2.0,
        emit_ny=0.25, alpha_y=0.0, beta_y=2.0,
        emit_z=0.3,    alpha_z=0.0, beta_z=10.0,
    )


def _envelope_initial(cfg: BeamConfig, ref) -> dict:
    bg = max(ref.bg, 1e-9)
    return dict(
        alpha_x=cfg.alpha_x, beta_x=cfg.beta_x, emit_x=cfg.emit_nx / bg,
        alpha_y=cfg.alpha_y, beta_y=cfg.beta_y, emit_y=cfg.emit_ny / bg,
        alpha_z=cfg.alpha_z, beta_z=cfg.beta_z, emit_z=cfg.emit_z,
    )


def _run_envelope(lattice, cfg: BeamConfig):
    beam = create_beam(cfg, seed=42)
    solver = EnvelopeSolver(
        lattice, beam.ref, _envelope_initial(cfg, beam.ref),
        current=cfg.current,
    )
    return solver.run()


def main() -> None:
    if not LATTICE.exists():
        raise SystemExit(f"missing fixture: {LATTICE}")

    print(f"Loading {LATTICE.name} …")
    lat_orig, meta = parse_tracewin(str(LATTICE))
    if meta.get("warnings"):
        print(f"  warnings: {meta['warnings']}")
    cfg = _beam_cfg()

    # --- baseline (pre-match) envelope --------------------------------
    res_before = _run_envelope(copy.deepcopy(lat_orig), cfg)
    sigma_x_before = float(res_before.sigma_x[-1])
    print(f"  σ_x at exit (before): {sigma_x_before:.4f} mm")

    # --- run the matcher ---------------------------------------------
    print("Running matcher …")
    lat_after = copy.deepcopy(lat_orig)
    cfg_after = copy.deepcopy(cfg)
    result = match(lat_after, cfg_after, max_iter=200)
    print(f"  status   : {'OK' if result.success else 'FAILED'}")
    print(f"  iters    : {result.n_iter}")
    print(f"  cost     : {result.cost:.4e}")
    print(f"  message  : {result.message}")
    for var, x0, xf in zip(result.variables, result.x0, result.x_final):
        print(f"    {var.label:<30s}  {x0:>10.4g}  →  {xf:<10.4g}")

    # --- post-match envelope -----------------------------------------
    res_after = _run_envelope(lat_after, cfg_after)
    sigma_x_after = float(res_after.sigma_x[-1])
    print(f"  σ_x at exit (after):  {sigma_x_after:.4f} mm  (target = 4.000)")

    # --- plot σ_x(s) before / after ----------------------------------
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(np.asarray(res_before.s) * 1e-3, res_before.sigma_x,
            "C1--", label=f"before  (σ_x_end={sigma_x_before:.3f} mm)")
    ax.plot(np.asarray(res_after.s) * 1e-3, res_after.sigma_x,
            "C0-",  label=f"after   (σ_x_end={sigma_x_after:.3f} mm)",
            lw=1.4)
    ax.axhline(4.0, color="k", ls=":", lw=0.8, label="target = 4 mm")
    ax.set_xlabel("s [m]")
    ax.set_ylabel("σ_x [mm]")
    ax.set_title("Auto-adjust demo: SET_SIZE constraint via ADJUST on QUAD gradients")
    ax.grid(alpha=0.3); ax.legend()
    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  plot     : {OUT_PNG}")


if __name__ == "__main__":
    main()
