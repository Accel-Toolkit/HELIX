"""Twiss matching demo — invert a transport line.

Loads ``examples/twiss_matching_demo.dat``, which carries:

* one ``ADJUST_BEAM_TWISS`` declaring α_x, β_x, α_y, β_y as variables, and
* one ``SET_TWISS`` constraint pinning the exit Twiss to (α=0, β=1.5)
  in both planes.

The matcher solves the inverse problem: what input Twiss produces the
desired exit Twiss?  Output is a side-by-side plot of β_x(s) and β_y(s)
before vs after matching, plus a stdout report.

Run::

    PYTHONPATH=. python3 examples/twiss_matching_demo.py
"""
from __future__ import annotations

import copy
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
LATTICE = HERE / "twiss_matching_demo.dat"
OUT_PNG = HERE / "twiss_matching_demo_compare.png"


def _baseline_cfg() -> BeamConfig:
    return BeamConfig(
        species="proton", energy=3.0, frequency=352.21,
        current=0.0, duty_cycle=100.0,
        n_particles=10, distribution="waterbag", cutoff=3.0,
        emit_nx=0.25, alpha_x=0.0, beta_x=1.0,
        emit_ny=0.25, alpha_y=0.0, beta_y=1.0,
        emit_z=0.3,    alpha_z=0.0, beta_z=10.0,
    )


def _envelope_initial(cfg, ref):
    bg = max(ref.bg, 1e-9)
    return dict(
        alpha_x=cfg.alpha_x, beta_x=cfg.beta_x, emit_x=cfg.emit_nx / bg,
        alpha_y=cfg.alpha_y, beta_y=cfg.beta_y, emit_y=cfg.emit_ny / bg,
        alpha_z=cfg.alpha_z, beta_z=cfg.beta_z, emit_z=cfg.emit_z,
    )


def _run_envelope(lattice, cfg):
    beam = create_beam(cfg, seed=42)
    solver = EnvelopeSolver(
        lattice, beam.ref, _envelope_initial(cfg, beam.ref),
        current=cfg.current,
    )
    return solver.run()


def main():
    if not LATTICE.exists():
        raise SystemExit(f"missing fixture: {LATTICE}")

    print(f"Loading {LATTICE.name} …")
    lat_orig, meta = parse_tracewin(str(LATTICE))
    if meta.get("warnings"):
        print(f"  warnings: {meta['warnings']}")

    cfg_before = _baseline_cfg()
    res_before = _run_envelope(copy.deepcopy(lat_orig), cfg_before)
    print(f"  baseline input    α_x={cfg_before.alpha_x:.3f}  "
          f"β_x={cfg_before.beta_x:.3f}  α_y={cfg_before.alpha_y:.3f}  "
          f"β_y={cfg_before.beta_y:.3f}")
    print(f"  baseline exit     α_x={res_before.alpha_x[-1]:.3f}  "
          f"β_x={res_before.beta_x[-1]:.3f}  α_y={res_before.alpha_y[-1]:.3f}  "
          f"β_y={res_before.beta_y[-1]:.3f}")

    print("Running matcher …")
    lat_after = copy.deepcopy(lat_orig)
    cfg_after = copy.deepcopy(cfg_before)
    result = match(lat_after, cfg_after, max_iter=400)

    print(f"  status   : {'OK' if result.success else 'FAILED'}")
    print(f"  iters    : {result.n_iter}")
    print(f"  cost     : {result.cost:.3e}")
    for var, x0, xf in zip(result.variables, result.x0, result.x_final):
        print(f"    {var.label:<22s}  {x0:>10.4g}  →  {xf:<10.4g}")

    res_after = _run_envelope(lat_after, cfg_after)
    print(f"  matched input     α_x={cfg_after.alpha_x:.3f}  "
          f"β_x={cfg_after.beta_x:.3f}  α_y={cfg_after.alpha_y:.3f}  "
          f"β_y={cfg_after.beta_y:.3f}")
    print(f"  matched exit      α_x={res_after.alpha_x[-1]:.3f}  "
          f"β_x={res_after.beta_x[-1]:.3f}  α_y={res_after.alpha_y[-1]:.3f}  "
          f"β_y={res_after.beta_y[-1]:.3f}    (target  α=0  β=1.5)")

    s_b = np.asarray(res_before.s) * 1e-3
    s_a = np.asarray(res_after.s)  * 1e-3
    fig, ax = plt.subplots(2, 1, figsize=(8, 7), sharex=True)
    ax[0].plot(s_b, res_before.beta_x, "C1--", label="β_x before")
    ax[0].plot(s_a, res_after.beta_x,  "C0-",  label="β_x after",  lw=1.4)
    ax[0].axhline(1.5, color="k", ls=":", lw=0.8, label="target β=1.5")
    ax[0].set_ylabel("β_x [mm/mrad]")
    ax[0].grid(alpha=0.3); ax[0].legend()

    ax[1].plot(s_b, res_before.beta_y, "C1--", label="β_y before")
    ax[1].plot(s_a, res_after.beta_y,  "C0-",  label="β_y after",  lw=1.4)
    ax[1].axhline(1.5, color="k", ls=":", lw=0.8, label="target β=1.5")
    ax[1].set_xlabel("s [m]"); ax[1].set_ylabel("β_y [mm/mrad]")
    ax[1].grid(alpha=0.3); ax[1].legend()
    fig.suptitle("Inverse Twiss match: input α/β tuned so exit α=0, β=1.5",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  plot     : {OUT_PNG}")


if __name__ == "__main__":
    main()
