"""Numerical convergence study for the FODO example with the H- injector
beam defaults.

Runs the simulation repeatedly while varying one knob at a time:

    1. integration_steps_per_metre  (PARTRAN_STEP step1)
    2. sc_steps_per_metre           (PARTRAN_STEP step2)
    3. PIC grid resolution (nx = ny = nz)
    4. PIC grid_extent (sigma multipliers)
    5. Number of macro-particles

After each sweep we compare the final RMS sizes and geometric emittances
to a high-resolution reference run and print a table plus a short
"converged at <value>" recommendation.

Run from the repo root:

    PYTHONPATH=. python3 scripts/convergence_study.py

Fast mode (fewer sweep points, 10 k particles):

    PYTHONPATH=. python3 scripts/convergence_study.py --fast
"""
from __future__ import annotations

import argparse
import os
import time
from dataclasses import dataclass, replace
from typing import Callable

import numpy as np
import matplotlib
matplotlib.use("Agg")           # headless (save to PNG, no window)
import matplotlib.pyplot as plt

from linac_gen.core.beam import Beam
from linac_gen.core.config import SpaceChargeConfig
from linac_gen.core.particle import H_MINUS
from linac_gen.core.reference import ReferenceParticle
from linac_gen.core.simulation import Simulation
from linac_gen.core.step_config import StepConfig
from linac_gen.io.tracewin_parser import parse_tracewin

FODO_PATH = "examples/fodo_cell.dat"

# ------------------------------------------------------------------------
# H- injector defaults (match gui/linac_gen_gui/widgets/beam_config.py)
# ------------------------------------------------------------------------
REF_SPECIES = H_MINUS
REF_ENERGY = 2.1226695    # MeV
REF_FREQUENCY = 162.5     # MHz
REF_CURRENT = 5.0         # mA

# Normalised emittance and Twiss per plane (as in the H- config)
EMIT_NX = 0.21                 # pi.mm.mrad
EMIT_NY = 0.21
EMIT_Z = 0.06231832            # pi.deg.MeV (already geometric for z)
ALPHA_X = 1.228
BETA_X = 0.316
ALPHA_Y = -0.095394323
BETA_Y = 0.113
ALPHA_Z = 0.0
BETA_Z = 819.05492


# ------------------------------------------------------------------------
# Beam generator
# ------------------------------------------------------------------------
def _twiss_to_sigma(alpha, beta, emit):
    gamma = (1 + alpha ** 2) / beta
    return np.array([[beta * emit, -alpha * emit],
                     [-alpha * emit, gamma * emit]])


def make_beam(n_particles: int, rng_seed: int = 42) -> Beam:
    """Create the H- injector reference beam (waterbag with the given Twiss)."""
    ref = ReferenceParticle(species=REF_SPECIES,
                            w_kin=REF_ENERGY,
                            frequency=REF_FREQUENCY)
    bg = ref.bg
    emit_x = EMIT_NX / bg          # geometric
    emit_y = EMIT_NY / bg
    emit_z = EMIT_Z

    rng = np.random.default_rng(rng_seed)
    particles = np.zeros((n_particles, 6))

    # Uniform (waterbag-ish) sampling in each plane using Cholesky of sigma
    for (i, j), alpha, beta, emit in (
        ((0, 1), ALPHA_X, BETA_X, emit_x),
        ((2, 3), ALPHA_Y, BETA_Y, emit_y),
        ((4, 5), ALPHA_Z, BETA_Z, emit_z),
    ):
        sigma = _twiss_to_sigma(alpha, beta, emit)
        L = np.linalg.cholesky(sigma)
        z = rng.normal(size=(n_particles, 2))
        xy = z @ L.T
        particles[:, i] = xy[:, 0]
        particles[:, j] = xy[:, 1]

    beam = Beam(ref=ref, n_particles=n_particles, current=REF_CURRENT)
    beam.particles[:] = particles
    return beam


# ------------------------------------------------------------------------
# Simulation runner
# ------------------------------------------------------------------------
@dataclass
class Knobs:
    """Everything we might vary in one run."""
    n_particles: int = 10_000
    integration_per_m: float = 100.0
    sc_per_m: float = 50.0
    n_grid: int = 64
    grid_extent: float = 3.0
    seed: int = 42


@dataclass
class RunResult:
    knobs: Knobs
    sigma_x_final: float
    sigma_y_final: float
    sigma_phi_final: float
    emit_x_final: float      # geometric mm.mrad
    emit_y_final: float
    emit_z_final: float      # deg.MeV (native)
    emit_x_growth: float     # final/initial - 1
    emit_y_growth: float
    emit_z_growth: float
    transmission: float
    runtime_s: float


def simulate(knobs: Knobs) -> RunResult:
    """Run the FODO simulation with the given knobs, return summary stats."""
    lat, _ = parse_tracewin(FODO_PATH)
    lat.step_config = StepConfig(
        integration_steps_per_metre=knobs.integration_per_m,
        sc_steps_per_metre=knobs.sc_per_m,
    )
    beam = make_beam(knobs.n_particles, rng_seed=knobs.seed)
    sc = SpaceChargeConfig(
        nx=knobs.n_grid, ny=knobs.n_grid, nz=knobs.n_grid,
        grid_extent=knobs.grid_extent,
    )
    sim = Simulation(lat, beam, space_charge=sc)
    t0 = time.time()
    res = sim.run()
    dt = time.time() - t0

    emit_x0, emit_y0, emit_z0 = res.emit_x[0], res.emit_y[0], res.emit_z[0]
    return RunResult(
        knobs=knobs,
        sigma_x_final=res.sigma_x[-1],
        sigma_y_final=res.sigma_y[-1],
        sigma_phi_final=res.sigma_phi[-1],
        emit_x_final=res.emit_x[-1],
        emit_y_final=res.emit_y[-1],
        emit_z_final=res.emit_z[-1],
        emit_x_growth=res.emit_x[-1] / emit_x0 - 1.0 if emit_x0 > 0 else 0.0,
        emit_y_growth=res.emit_y[-1] / emit_y0 - 1.0 if emit_y0 > 0 else 0.0,
        emit_z_growth=res.emit_z[-1] / emit_z0 - 1.0 if emit_z0 > 0 else 0.0,
        transmission=res.transmission[-1],
        runtime_s=dt,
    )


# ------------------------------------------------------------------------
# Sweep + convergence-detect helpers
# ------------------------------------------------------------------------
def _rel_delta(a: float, b: float) -> float:
    """Relative difference |a - b| / max(|b|, epsilon)."""
    return abs(a - b) / max(abs(b), 1e-30)


def find_converged(values, metric, tol=0.01):
    """Return the smallest `values[i]` for which |metric[i] - metric[-1]|/|metric[-1]| <= tol.

    `metric[-1]` is taken as the reference (finest setting).
    """
    ref = metric[-1]
    for v, m in zip(values, metric):
        if _rel_delta(m, ref) <= tol:
            return v
    return values[-1]


def print_sweep(name: str, unit: str, values, results, columns):
    """Pretty-print a sweep table."""
    print(f"\n=== Sweep: {name} [{unit}] ===")
    header = f"{name:>14s}  " + "  ".join(f"{col:>12s}" for col in columns) + "    runtime"
    print(header)
    print("-" * len(header))
    ref = results[-1]
    for v, r in zip(values, results):
        fields = [
            f"{r.sigma_x_final:12.4f}",
            f"{r.sigma_y_final:12.4f}",
            f"{r.emit_x_final:12.4f}",
            f"{r.emit_y_final:12.4f}",
            f"{r.emit_x_growth * 100:+12.2f}",
            f"{r.emit_y_growth * 100:+12.2f}",
        ]
        row = f"{v!s:>14s}  " + "  ".join(fields[:len(columns)])
        row += f"    {r.runtime_s:6.2f} s"
        print(row)


def sweep_and_report(name, unit, values, make_knobs,
                     tol_sigma=0.01, tol_emit=0.01):
    """Run a 1-D sweep, print the table, print the convergence recommendation."""
    results = []
    for v in values:
        knobs = make_knobs(v)
        r = simulate(knobs)
        results.append(r)
        print(f"  ... {name} = {v!r}  sigma_x={r.sigma_x_final:.4f} mm  "
              f"emit_x={r.emit_x_final:.4f} mm.mrad  t={r.runtime_s:.2f}s")
    print_sweep(name, unit, values, results, columns=[
        "sigma_x", "sigma_y", "emit_x", "emit_y",
        "dEx %", "dEy %",
    ])
    sx = [r.sigma_x_final for r in results]
    ex = [r.emit_x_final for r in results]
    print(f"  converged sigma_x (|δ|≤{tol_sigma*100:.0f}%): "
          f"{name} >= {find_converged(values, sx, tol_sigma)!r}")
    print(f"  converged emit_x  (|δ|≤{tol_emit*100:.0f}%):  "
          f"{name} >= {find_converged(values, ex, tol_emit)!r}")
    return results


# ------------------------------------------------------------------------
# Plot helpers
# ------------------------------------------------------------------------

def plot_sweep(name: str, unit: str, values, results, default_value,
               out_path: str) -> None:
    """Save a 2x2 convergence figure for one sweep.

    Panels: sigma_x, sigma_y, emit_x, emit_y  (each vs the sweep knob)
    """
    sig_x = [r.sigma_x_final for r in results]
    sig_y = [r.sigma_y_final for r in results]
    emt_x = [r.emit_x_final for r in results]
    emt_y = [r.emit_y_final for r in results]

    fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharex=True)
    fig.suptitle(f"Convergence vs. {name} [{unit}]  "
                 f"(FODO + H- defaults, 5 mA)", fontsize=13)

    def _panel(ax, ys, ylabel, ref_colour="#2aa198"):
        ax.plot(values, ys, "o-", lw=2.0, color=ref_colour)
        ax.set_xscale("log")
        ax.grid(True, alpha=0.3)
        ax.set_ylabel(ylabel)
        ax.axvline(default_value, color="#d33682", ls="--",
                   lw=1.5, label=f"default = {default_value}")
        # Mark the final (reference) value as a faint horizontal guide
        ax.axhline(ys[-1], color="#586e75", ls=":", lw=1.0, alpha=0.6,
                   label=f"ref = {ys[-1]:.3f}")
        ax.legend(fontsize=9, loc="best")

    _panel(axes[0][0], sig_x, "sigma_x (mm)")
    _panel(axes[0][1], sig_y, "sigma_y (mm)")
    _panel(axes[1][0], emt_x, "emit_x (mm.mrad)")
    _panel(axes[1][1], emt_y, "emit_y (mm.mrad)")

    for ax in axes[1]:
        ax.set_xlabel(f"{name} [{unit}]")

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    print(f"  plot saved: {out_path}")


def plot_overview(summaries, default_values, out_path) -> None:
    """One combined figure with relative-change vs knob for all five sweeps."""
    fig, ax = plt.subplots(figsize=(11, 6.5))
    colors = plt.cm.tab10.colors

    for i, (name, unit, values, results, default) in enumerate(summaries):
        emt_x = np.asarray([r.emit_x_final for r in results])
        ref = emt_x[-1]
        rel_delta = (emt_x - ref) / max(abs(ref), 1e-30) * 100.0
        x_frac = np.asarray(values, dtype=float) / float(default)
        ax.plot(x_frac, rel_delta, "o-", lw=2.0, color=colors[i],
                label=f"{name} (default={default})")

    ax.axhline(0.0, color="#586e75", ls=":", lw=1.0)
    ax.axhline(+1.0, color="#859900", ls="--", lw=0.8, alpha=0.6)
    ax.axhline(-1.0, color="#859900", ls="--", lw=0.8, alpha=0.6)
    ax.axvline(1.0, color="#d33682", ls="--", lw=1.2, alpha=0.7,
               label="current default")
    ax.set_xscale("log")
    ax.set_xlabel("knob / default  (log)")
    ax.set_ylabel("Δ emit_x relative to finest run [%]")
    ax.set_title("Convergence overview — emit_x vs. each knob, normalised")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10, loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    print(f"  plot saved: {out_path}")


# ------------------------------------------------------------------------
# Main
# ------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fast", action="store_true",
                        help="fewer sweep points, 10k particles (quick demo)")
    args = parser.parse_args()

    if args.fast:
        n_part_for_sweeps = 10_000
        int_sweep = [20, 50, 100, 200, 500]
        sc_sweep = [10, 25, 50, 100, 200]
        grid_sweep = [16, 32, 64, 96]
        extent_sweep = [2.5, 3.0, 4.0, 5.0]
        part_sweep = [2_000, 5_000, 10_000, 25_000]
    else:
        n_part_for_sweeps = 20_000
        int_sweep = [10, 25, 50, 100, 200, 500, 1000]
        sc_sweep = [5, 10, 25, 50, 100, 200, 500]
        grid_sweep = [16, 24, 32, 48, 64, 96, 128]
        extent_sweep = [2.0, 2.5, 3.0, 4.0, 5.0, 6.0]
        part_sweep = [2_000, 5_000, 10_000, 25_000, 50_000, 100_000]

    print("=" * 78)
    print("FODO + H- default-beam convergence study")
    print(f"  lattice      : {FODO_PATH}")
    print(f"  beam         : H-  {REF_ENERGY} MeV  {REF_CURRENT} mA  "
          f"n={n_part_for_sweeps}")
    print(f"  baseline     : PARTRAN_STEP 100 50, grid 64^3, extent 3σ")
    print("=" * 78)

    base = Knobs(n_particles=n_part_for_sweeps)

    # Baseline run for reference
    print("\nReference run (current defaults):")
    ref = simulate(base)
    print(f"  sigma_x = {ref.sigma_x_final:.4f} mm   emit_x = {ref.emit_x_final:.4f} mm.mrad")
    print(f"  sigma_y = {ref.sigma_y_final:.4f} mm   emit_y = {ref.emit_y_final:.4f} mm.mrad")
    print(f"  sigma_phi = {ref.sigma_phi_final:.4f} deg  emit_z = {ref.emit_z_final:.4f} deg.MeV")
    print(f"  transmission = {ref.transmission:.2f} %")
    print(f"  runtime = {ref.runtime_s:.2f} s")

    plot_dir = "docs/convergence"
    os.makedirs(plot_dir, exist_ok=True)

    sweeps = [
        ("step1",    "1/m",   int_sweep,
         lambda v: replace(base, integration_per_m=v), 100),
        ("step2",    "1/m",   sc_sweep,
         lambda v: replace(base, sc_per_m=v), 50),
        ("n_grid",   "cells", grid_sweep,
         lambda v: replace(base, n_grid=v), 64),
        ("grid_ext", "sigma", extent_sweep,
         lambda v: replace(base, grid_extent=v), 3.0),
        ("n_part",   "-",     part_sweep,
         lambda v: replace(base, n_particles=v), n_part_for_sweeps),
    ]

    summaries = []
    for title, unit, values, make_knobs, default_value in sweeps:
        print(f"\n----- Sweep: {title} -----")
        results = sweep_and_report(title, unit, values, make_knobs)
        summaries.append((title, unit, values, results, default_value))
        plot_sweep(title, unit, values, results, default_value,
                   out_path=f"{plot_dir}/convergence_{title}.png")

    plot_overview(summaries, default_values=None,
                  out_path=f"{plot_dir}/convergence_overview.png")

    print(f"\nAll plots saved to {plot_dir}/")


if __name__ == "__main__":
    main()
