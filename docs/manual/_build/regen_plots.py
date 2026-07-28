"""Regenerate every figure used in the HELIX manual.

Each figure is a function decorated with @figure(key) — the decorator
registers it so this script can find and call all of them.  Each
function returns a matplotlib Figure (or saves directly).  The output
lives in ``docs/manual/_build/figures/<key>.png``.

Usage:
    python docs/manual/_build/regen_plots.py            # regenerate all
    python docs/manual/_build/regen_plots.py fig_03_*   # regenerate a glob

Adding a new figure:
    @figure("fig_07_05_matching_recipe")
    def plot_matching_recipe():
        '''Caption for the figure (used in the index).'''
        fig, ax = plt.subplots(...)
        ...
        return fig
"""
from __future__ import annotations

import argparse
import fnmatch
import sys
from pathlib import Path
from typing import Callable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------
_FIGURES: dict[str, tuple[Callable[[], plt.Figure], str]] = {}


def figure(key: str):
    """Decorator: register a plotting function under a string key."""
    def deco(fn: Callable[[], plt.Figure]):
        caption = (fn.__doc__ or "").strip().split("\n", 1)[0] or "(no caption)"
        _FIGURES[key] = (fn, caption)
        return fn
    return deco


# ---------------------------------------------------------------------------
# Figure definitions
# ---------------------------------------------------------------------------
# Phase 0 ships an empty registry.  Each part-N chapter populates this file
# with its own @figure(...) functions as the manual is written.

# Concepts ------------------------------------------------------------------

@figure("fig_02_01_coordinate_system")
def plot_coordinate_system():
    """Figure 2.1 — HELIX/TraceWin reduced phase space (x, x', y, y', Δφ, ΔW)."""
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.set_axis_off()
    ax.text(0.5, 0.95, "HELIX 6-D phase space", ha="center", fontsize=13,
            weight="bold", transform=ax.transAxes)
    rows = [
        ("x", "transverse position", "mm"),
        ("x'", "transverse divergence dx/dz", "mrad"),
        ("y", "transverse position", "mm"),
        ("y'", "transverse divergence dy/dz", "mrad"),
        ("Δφ", "phase relative to ref. particle", "deg"),
        ("ΔW", "kinetic-energy deviation", "MeV"),
    ]
    for i, (sym, desc, unit) in enumerate(rows):
        y = 0.80 - i * 0.12
        ax.text(0.10, y, sym, fontsize=14, family="serif",
                transform=ax.transAxes)
        ax.text(0.25, y, desc, fontsize=11, transform=ax.transAxes)
        ax.text(0.85, y, f"[{unit}]", fontsize=11, family="monospace",
                color="#555", transform=ax.transAxes)
    return fig


@figure("fig_02_03_tracking_modes")
def plot_tracking_modes():
    """Figure 2.3 — Decision tree: envelope vs MP vs DC modes."""
    import matplotlib.patches as mpatches
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_axis_off()
    ax.set_xlim(0, 10); ax.set_ylim(0, 6)

    def box(x, y, w, h, text, color="#e3eafc"):
        ax.add_patch(mpatches.FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0.1",
            facecolor=color, edgecolor="#444"))
        ax.text(x + w/2, y + h/2, text, ha="center", va="center",
                fontsize=10)

    def arrow(x1, y1, x2, y2, label=""):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="->", color="#444"))
        if label:
            ax.text((x1+x2)/2 + 0.1, (y1+y2)/2, label,
                    fontsize=9, color="#666")

    # Root
    box(3.5, 5, 3, 0.7, "Is the beam bunched?", "#fff3cd")
    # Bunched branch
    box(0.5, 3.5, 3, 0.7, "RMS only? → EnvelopeSolver", "#d4edda")
    box(0.5, 2.3, 3, 0.7, "MP? → Tracker + 3-D PIC", "#d4edda")
    # DC branch
    box(6.5, 3.5, 3, 0.7, "RMS only? → Sacherer ODE", "#cce5ff")
    box(6.5, 2.3, 3, 0.7, "MP? → Tracker + 2-D DC kick", "#cce5ff")

    arrow(4.5, 5, 2, 4.2, "yes (bunched)")
    arrow(5.5, 5, 8, 4.2, "no (DC)")
    arrow(2, 3.5, 2, 3.0)
    arrow(8, 3.5, 8, 3.0)

    ax.text(5, 0.7, "Use envelope first for matching/sweeps; MP for production answer.",
            ha="center", fontsize=10, style="italic", color="#555")
    return fig


# Beam ----------------------------------------------------------------------

@figure("fig_04_01_distributions_gallery")
def plot_distributions_gallery():
    """Figure 4.1 — x–x' projections of the six built-in distributions."""
    import sys, os
    sys.path.insert(0, os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..", "..")))
    from linac_gen.core.config import BeamConfig
    from linac_gen.distributions.factory import create_beam

    common = dict(
        species="proton", energy=3.0, frequency=352.21,
        current=0.0, n_particles=2000, cutoff=4.0,
        emit_nx=0.25, emit_ny=0.25, emit_z=0.30,
        alpha_x=0.0, beta_x=2.0,
        alpha_y=0.0, beta_y=2.0,
        alpha_z=0.0, beta_z=1.0,
    )

    distros = [
        ("gaussian",  "gaussian"),
        ("waterbag",  "waterbag"),
        ("kv",        "KV"),
        ("parabolic", "parabolic"),
        ("uniform",   "uniform"),
        ("thermal",   "thermal\n(halo_frac=0.05)"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(11, 7), sharex=True, sharey=True)
    for ax, (key, label) in zip(axes.flat, distros):
        cfg = BeamConfig(distribution=key, halo_fraction=0.05, halo_ratio=5.0,
                         **common)
        try:
            beam = create_beam(cfg, seed=0)
            xs = beam.particles[:, 0]
            xps = beam.particles[:, 1]
            ax.scatter(xs, xps, s=1, alpha=0.4, color="C0")
        except Exception as e:
            ax.text(0.5, 0.5, f"(error: {e})", transform=ax.transAxes,
                    ha="center", va="center")
        ax.set_title(label, fontsize=10)
        ax.set_xlabel("x [mm]"); ax.set_ylabel("x' [mrad]")
        ax.grid(True, alpha=0.3)
    fig.suptitle("HELIX distribution generators (x–x' projection)",
                 fontsize=12, y=0.99)
    plt.tight_layout()
    return fig


# Space charge --------------------------------------------------------------

@figure("fig_05_02_pic_cycle")
def plot_pic_cycle():
    """Figure 5.2 — The 8-step PIC cycle."""
    import matplotlib.patches as mpatches
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_axis_off()
    ax.set_xlim(0, 10); ax.set_ylim(0, 8)

    steps = [
        "1. Lorentz boost on z (γ on z)",
        "2. Set up grid (nx × ny × nz, ±5σ box)",
        "3. Charge deposition (CIC or TSC)",
        "4. Doubled-grid FFT convolution with IGF",
        "5. E-field gather to particles",
        "6. Transverse momentum kick",
        "7. Longitudinal momentum kick",
        "8. Inverse Lorentz boost",
    ]
    for i, txt in enumerate(steps):
        y = 7.2 - i * 0.85
        ax.add_patch(mpatches.FancyBboxPatch(
            (0.5, y - 0.3), 9, 0.6, boxstyle="round,pad=0.1",
            facecolor="#e3eafc", edgecolor="#444"))
        ax.text(5, y, txt, ha="center", va="center", fontsize=10)
        if i < len(steps) - 1:
            ax.annotate("", xy=(5, y - 0.55), xytext=(5, y - 0.3),
                        arrowprops=dict(arrowstyle="->", color="#444"))
    ax.text(5, 0.2, "Reference: Qiang et al., PRSTAB 9, 044204 (2006)",
            ha="center", fontsize=9, style="italic", color="#555")
    return fig


# Validation ----------------------------------------------------------------

def _load_pipii_cache():
    """Helper: load the PIP-II validation cache, returning None if missing."""
    import os, numpy as np
    path = os.path.abspath(os.path.join(
        os.path.dirname(__file__), "..", "..", "..",
        "examples", "comparisons", "full_pipii", "data_cache.npz"))
    if not os.path.exists(path):
        return None
    return np.load(path)


@figure("fig_12_02_pipii_env_overlay")
def plot_pipii_env_overlay():
    """Figure 12.2a — HELIX env vs TraceWin partran (5 mA SC, 256m)."""
    c = _load_pipii_cache()
    if c is None:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.text(0.5, 0.5, "Cache examples/comparisons/full_pipii/data_cache.npz\n"
                          "not present — figure skipped.",
                ha="center", va="center", transform=ax.transAxes)
        return fig

    fig, axes = plt.subplots(2, 2, figsize=(13, 8), sharex=True)
    fig.suptitle("PIP-II 256m, 5 mA SC: HELIX env vs TraceWin partran",
                 fontsize=14, y=0.995)
    panels = [
        (axes[0, 0], "sx",   "σ_x [mm]"),
        (axes[0, 1], "sy",   "σ_y [mm]"),
        (axes[1, 0], "sphi", "σ_φ [deg]"),
        (axes[1, 1], "W",    "W [MeV]"),
    ]
    for a, key, ylabel in panels:
        a.plot(c["tw_sc_s"], c[f"tw_sc_{key}"], color="C3", lw=1.0,
               label="TraceWin partran")
        a.plot(c["lg_sc_s"], c[f"lg_sc_{key}"], color="C0", lw=1.0,
               label="HELIX env")
        a.set_ylabel(ylabel); a.grid(True, alpha=0.3)
    axes[0, 0].legend(loc="upper left", fontsize=10)
    axes[1, 0].set_xlabel("s [m]"); axes[1, 1].set_xlabel("s [m]")
    plt.tight_layout()
    return fig


@figure("fig_12_02_pipii_residuals")
def plot_pipii_residuals():
    """Figure 12.2b — Relative residuals (HELIX − TW)/TW for σ_x, σ_y."""
    import numpy as np
    c = _load_pipii_cache()
    if c is None:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.text(0.5, 0.5, "(cache not present — figure skipped)",
                ha="center", va="center", transform=ax.transAxes)
        return fig

    fig, axes = plt.subplots(2, 2, figsize=(13, 7), sharex=True)
    fig.suptitle("Relative residuals: (HELIX − TraceWin) / TraceWin",
                 fontsize=14, y=0.995)
    cols = [("nosc", "no SC"), ("sc", "5 mA, with SC")]
    rows = [("sx", "σ_x rel %"), ("sy", "σ_y rel %")]
    for j, (mode, mtitle) in enumerate(cols):
        for i, (key, ylabel) in enumerate(rows):
            ax = axes[i, j]
            s_tw = c[f"tw_{mode}_s"]
            s_lg = c[f"lg_{mode}_s"]
            hx_on_tw = np.interp(s_tw, s_lg, c[f"lg_{mode}_{key}"])
            tw = c[f"tw_{mode}_{key}"]
            rel = (hx_on_tw - tw) / np.maximum(np.abs(tw), 1e-9) * 100
            color = "C2" if mode == "sc" else "C1"
            ax.plot(s_tw, rel, color=color, lw=0.8)
            ax.axhline(0, color="k", lw=0.4, alpha=0.5)
            ax.axhline(3, color="k", lw=0.4, ls=":", alpha=0.4)
            ax.axhline(-3, color="k", lw=0.4, ls=":", alpha=0.4)
            if i == 0:
                ax.set_title(mtitle, fontsize=11)
            ax.set_ylabel(ylabel, fontsize=10)
            ax.grid(True, alpha=0.3)
    for ax in axes[-1, :]:
        ax.set_xlabel("s [m]")
    plt.tight_layout()
    return fig


# Worked examples -----------------------------------------------------------

@figure("fig_11_01_basic_fodo")
def plot_basic_fodo():
    """Figure 11.1 — σ_x and σ_y through the 2-cell FODO of basic_tracking.py."""
    import sys, os
    import numpy as np
    sys.path.insert(0, os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..", "..")))
    from linac_gen.core.lattice import Lattice
    from linac_gen.core.simulation import Simulation
    from linac_gen.core.config import BeamConfig, SpaceChargeConfig
    from linac_gen.distributions.factory import create_beam
    from linac_gen.elements.drift import Drift
    from linac_gen.elements.quadrupole import Quadrupole
    from linac_gen.elements.rf_gap import RFGap

    lat = Lattice()
    for cell in range(2):
        lat.add(Quadrupole(name=f"QF_{cell}", length=100.0, gradient=+10.0))
        lat.add(Drift(name=f"D1_{cell}", length=200.0))
        lat.add(Quadrupole(name=f"QD_{cell}", length=100.0, gradient=-10.0))
        lat.add(Drift(name=f"D2_{cell}", length=200.0))
        lat.add(RFGap(name=f"GAP_{cell}", voltage=1.0e6, phase=-30.0,
                       frequency=352.21))

    cfg = BeamConfig(species="proton", energy=3.0, frequency=352.21,
                     current=5.0, n_particles=5000,
                     distribution="gaussian", cutoff=4.0,
                     emit_nx=0.25, emit_ny=0.25, emit_z=0.30,
                     alpha_x=0.0, beta_x=2.0,
                     alpha_y=0.0, beta_y=2.0,
                     alpha_z=0.0, beta_z=1.0)
    beam = create_beam(cfg, seed=42)
    sc = SpaceChargeConfig(nx=32, ny=32, nz=32, kernel="cic")
    res = Simulation(lat, beam, space_charge=sc).run()

    s_m = np.asarray(res.s) / 1e3
    fig, ax = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    ax[0].plot(s_m, res.sigma_x, label="σ_x", color="C0")
    ax[0].plot(s_m, res.sigma_y, label="σ_y", color="C1")
    ax[0].set_ylabel("σ [mm]"); ax[0].legend()
    ax[0].set_title("Basic FODO: 2 cells, 5 mA proton, MP+SC", fontsize=12)
    ax[1].plot(s_m, res.transmission, color="C2")
    ax[1].set_ylabel("transmission [%]"); ax[1].set_xlabel("s [m]")
    ax[1].set_ylim(95, 100.5)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("patterns", nargs="*", default=["*"],
                        help="figure-key glob(s) to regenerate (default: all)")
    parser.add_argument("--list", action="store_true",
                        help="list registered figures and exit")
    args = parser.parse_args()

    out_dir = Path(__file__).parent / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    keys = sorted(_FIGURES)
    if args.list:
        for k in keys:
            _, caption = _FIGURES[k]
            print(f"{k:<50} {caption}")
        return 0

    selected = [k for k in keys
                if any(fnmatch.fnmatch(k, p) for p in args.patterns)]
    if not selected:
        print(f"No figures match {args.patterns}", file=sys.stderr)
        return 1

    for k in selected:
        fn, caption = _FIGURES[k]
        out = out_dir / f"{k}.png"
        print(f"  generating {k}  ({caption})")
        fig = fn()
        if fig is None:
            print(f"    -> {k} returned None; skipped")
            continue
        fig.savefig(out, dpi=130, bbox_inches="tight")
        plt.close(fig)
    print(f"\n{len(selected)} figure(s) written to {out_dir}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
