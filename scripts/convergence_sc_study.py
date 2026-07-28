"""Space-charge-specific convergence comparison.

For each of the SC-sensitive knobs (grid_extent, n_grid, integration
step density, SC step density), run the FODO + H- beam at three
currents:

    I = 0 mA    (SC off  -- isolates pure optics / tracking)
    I = 5 mA    (default operating point)
    I = 50 mA   (strong SC)

If a knob only diverges when I > 0, we know its convergence requirement
is SC-driven.  If it diverges even at I = 0, something else in the
tracker is under-resolved.

Run:
    PYTHONPATH=. python3 scripts/convergence_sc_study.py
"""
from __future__ import annotations

import os
import time
from dataclasses import replace

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scripts.convergence_study import Knobs, make_beam, simulate

FODO_PATH = "examples/fodo_cell.dat"
PLOT_DIR = "docs/convergence"

# Reduced sweeps -- this file runs 4 sweeps × 3 currents = 12 campaigns
GRID_EXT_VALS   = [2.5, 3.0, 4.0, 5.0, 6.0]
N_GRID_VALS     = [16, 32, 48, 64, 96]
STEP1_VALS      = [20, 50, 100, 200, 500]
STEP2_VALS      = [10, 25, 50, 100, 200]
CURRENTS_MA     = [0.0, 5.0, 50.0]
COLOURS         = {0.0: "#268bd2", 5.0: "#859900", 50.0: "#dc322f"}
N_PARTICLES     = 8_000     # fast


def simulate_at_current(base: Knobs, current_mA: float) -> Knobs:
    """Create a Knobs variant that forces the beam to have `current` mA."""
    # `simulate()` re-creates the beam from Knobs + module constants, so we
    # temporarily override REF_CURRENT via the beam factory.  Easier: patch
    # the module-level REF_CURRENT for the scope of the call.
    return replace(base, seed=42)  # current override done below by monkeypatch


def _run_one(base, current_mA, value, override):
    import scripts.convergence_study as cs
    cs.REF_CURRENT = current_mA
    return simulate(override(base, value))


def sweep_at_current(base: Knobs, values, currents, override, label):
    """Run `values × currents` and return a dict {current_mA: [(value, result), ...]}."""
    out = {}
    for I_mA in currents:
        out[I_mA] = []
        for v in values:
            r = _run_one(base, I_mA, v, override)
            out[I_mA].append((v, r))
            print(f"    {label}={v!r:<8s}  I={I_mA:>5.1f} mA   "
                  f"sigma_x={r.sigma_x_final:.3f}  emit_x={r.emit_x_final:.3f}")
    return out


def plot_sc_sweep(name, unit, by_current, out_path, default_value):
    """2×2 figure of σ_x, σ_y, ε_x, ε_y vs the knob, one line per current."""
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True)
    fig.suptitle(f"Convergence vs. {name} [{unit}] at three beam currents  "
                 f"(FODO + H- defaults)", fontsize=13)

    for panel_idx, (ax, metric_key, ylabel) in enumerate([
        (axes[0][0], "sigma_x_final", "sigma_x (mm)"),
        (axes[0][1], "sigma_y_final", "sigma_y (mm)"),
        (axes[1][0], "emit_x_final",  "emit_x (mm.mrad)"),
        (axes[1][1], "emit_y_final",  "emit_y (mm.mrad)"),
    ]):
        for I_mA, rows in by_current.items():
            values = [r[0] for r in rows]
            ys = [getattr(r[1], metric_key) for r in rows]
            ax.plot(values, ys, "o-", lw=2.0,
                    color=COLOURS[I_mA], label=f"I = {I_mA:g} mA")
        ax.set_xscale("log")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
        ax.axvline(default_value, color="#d33682", ls="--", lw=1.3, alpha=0.7)
        if panel_idx == 0:
            ax.legend(loc="best", fontsize=10)

    for ax in axes[1]:
        ax.set_xlabel(f"{name} [{unit}]")

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    print(f"  saved: {out_path}")


def main() -> None:
    os.makedirs(PLOT_DIR, exist_ok=True)
    base = Knobs(n_particles=N_PARTICLES)

    print("=" * 78)
    print("Space-charge-specific convergence study")
    print(f"  currents compared: {CURRENTS_MA} mA")
    print(f"  n_particles      : {N_PARTICLES}")
    print("=" * 78)

    print("\nSweep: grid_extent  (most SC-sensitive knob)")
    data = sweep_at_current(base, GRID_EXT_VALS, CURRENTS_MA,
                            lambda b, v: replace(b, grid_extent=v),
                            label="grid_ext")
    plot_sc_sweep("grid_ext", "sigma", data,
                  f"{PLOT_DIR}/sc_convergence_grid_ext.png",
                  default_value=3.0)

    print("\nSweep: n_grid")
    data = sweep_at_current(base, N_GRID_VALS, CURRENTS_MA,
                            lambda b, v: replace(b, n_grid=v),
                            label="n_grid")
    plot_sc_sweep("n_grid", "cells", data,
                  f"{PLOT_DIR}/sc_convergence_n_grid.png",
                  default_value=64)

    print("\nSweep: step1 (integration)")
    data = sweep_at_current(base, STEP1_VALS, CURRENTS_MA,
                            lambda b, v: replace(b, integration_per_m=v),
                            label="step1")
    plot_sc_sweep("step1", "1/m", data,
                  f"{PLOT_DIR}/sc_convergence_step1.png",
                  default_value=100)

    print("\nSweep: step2 (SC kick density)")
    data = sweep_at_current(base, STEP2_VALS, CURRENTS_MA,
                            lambda b, v: replace(b, sc_per_m=v),
                            label="step2")
    plot_sc_sweep("step2", "1/m", data,
                  f"{PLOT_DIR}/sc_convergence_step2.png",
                  default_value=50)

    print(f"\nAll SC-comparison plots saved to {PLOT_DIR}/")


if __name__ == "__main__":
    main()
