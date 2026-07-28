"""Demo: DC (continuous) beam through a simple LEBT → RFQ-stub lattice.

Shows:

1. Beam generation in continuous mode — transverse distribution plus a
   uniform phase spread over one RF period and Gaussian energy spread.
2. 4-D tracking through drift + solenoid + drift sections with the 2-D
   analytic space-charge kick active.
3. Automatic DC → bunched transition at the first RF bunching element
   (here represented by a thin ``RFGap`` acting as the RFQ entry gap).
4. Continued 6-D tracking afterwards.
"""
from __future__ import annotations

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from linac_gen.core.config import BeamConfig
from linac_gen.core.simulation import Simulation
from linac_gen.core.lattice import Lattice
from linac_gen.distributions.factory import create_beam
from linac_gen.elements.drift import Drift
from linac_gen.elements.solenoid import Solenoid
from linac_gen.elements.rf_gap import RFGap


def main() -> None:
    # H- at 60 keV from an ion source; 5 mA DC, 2 keV 1-σ energy spread.
    cfg = BeamConfig(
        species="H-",
        energy=0.060,                     # MeV
        frequency=162.5,                  # MHz — RF frequency of the downstream RFQ
        current=5.0,                      # mA, peak
        n_particles=10000,
        distribution="gaussian", cutoff=4.0,
        emit_nx=0.2, alpha_x=0.0, beta_x=0.1,
        emit_ny=0.2, alpha_y=0.0, beta_y=0.1,
        # Continuous beam mode: uniform phase, explicit energy spread.
        # ε_z / α_z / β_z are ignored in this mode.
        continuous=True,
        dc_energy_spread_keV=2.0,
    )
    beam = create_beam(cfg, seed=42)
    # Snapshot the input distribution — ``Simulation.run`` mutates the
    # beam's particles in place, so plotting ``beam`` at the end would
    # show the post-LEBT state, not the initial one.
    particles_in = beam.particles.copy()
    print(f"Input (pre-LEBT):")
    print(f"  continuous       = {beam.continuous}")
    print(f"  σ_x              = {particles_in[:,0].std():.3f} mm")
    print(f"  σ_φ              = {particles_in[:,4].std():.2f} deg  "
          f"(uniform → ~103.9)")
    print(f"  σ_W              = {particles_in[:,5].std()*1000:.2f} keV  "
          f"(1 σ set = 2.0)")

    # Minimal LEBT: 500 mm drift → 200 mm solenoid → 300 mm drift → thin
    # RFGap (acts as the RFQ bunching gap) → 200 mm drift.
    lat = Lattice()
    lat.add(Drift(name="d1", length=500.0))
    lat.add(Solenoid(name="sol1", length=200.0, field=0.3))
    lat.add(Drift(name="d2", length=300.0))
    lat.add(RFGap(name="rfq_entry", voltage=0.05, phase=-90.0,
                  frequency=162.5))
    lat.add(Drift(name="d3", length=200.0))

    results = Simulation(lat, beam, space_charge=None).run()
    print(f"\nExit (post-RFQ gap):")
    print(f"  continuous       = {beam.continuous}   "
          f"(flipped by rfq_entry)")
    print(f"  σ_x              = {results.sigma_x[-1]:.3f} mm")
    print(f"  σ_y              = {results.sigma_y[-1]:.3f} mm")
    print(f"  σ_φ              = {results.sigma_phi[-1]:.2f} deg")
    print(f"  σ_W              = {results.sigma_w[-1]*1000:.2f} keV")
    print(f"  transmission     = {results.transmission[-1]:.2f} %")

    # The ``continuous_at`` array flags each record as DC (True) or bunched
    # (False) so downstream plots / popups can mark the LEBT segment.
    flags = results.continuous_at
    print(f"\nPer-record DC/bunched flag: "
          f"{sum(flags)} DC points, {len(flags) - sum(flags)} bunched points "
          f"(transition at record {flags.index(False) if False in flags else '-'})")

    _plot_run(results, particles_in, beam.particles,
              "lebt_dc_demo.png", "lebt_dc_demo_phasespace.png")


def _plot_run(results, particles_in, particles_out,
              path_timeline: str, path_phase: str) -> None:
    """Summary plot: σ_x, σ_y, σ_φ, σ_W, transmission, W_ref along s, plus
    a red shaded band for the DC segment so the pre/post transition is
    obvious at a glance."""
    s = np.asarray(results.s) * 1e-3                       # mm → m
    sigma_x = np.asarray(results.sigma_x)
    sigma_y = np.asarray(results.sigma_y)
    sigma_phi = np.asarray(results.sigma_phi)
    sigma_w = np.asarray(results.sigma_w) * 1e3            # MeV → keV
    trans = np.asarray(results.transmission)
    w_ref = np.asarray(results.ref_w_kin)
    dc = np.asarray(results.continuous_at, dtype=bool)

    # Shade span where the beam was still DC.
    dc_s_range = None
    if dc.any():
        s_dc = s[dc]
        dc_s_range = (float(s_dc.min()), float(s_dc.max()))

    fig, axes = plt.subplots(3, 2, figsize=(12, 8), sharex=True)
    panels = [
        (axes[0, 0], s, sigma_x, "σ_x", "mm"),
        (axes[0, 1], s, sigma_y, "σ_y", "mm"),
        (axes[1, 0], s, sigma_phi, "σ_φ", "deg"),
        (axes[1, 1], s, sigma_w, "σ_W", "keV"),
        (axes[2, 0], s, trans, "Transmission", "%"),
        (axes[2, 1], s, w_ref, "W_ref", "MeV"),
    ]
    for ax, x, y, label, unit in panels:
        ax.plot(x, y, "o-", color="C0", lw=1.5)
        ax.set_ylabel(f"{label} [{unit}]")
        ax.grid(alpha=0.3)
        if dc_s_range is not None:
            ax.axvspan(*dc_s_range, alpha=0.12, color="C3",
                       label="continuous (DC)")
            ax.legend(loc="upper right", fontsize=8)
    axes[2, 0].set_xlabel("s [m]"); axes[2, 1].set_xlabel("s [m]")
    fig.suptitle(
        "DC LEBT demo · H⁻ 60 keV · 5 mA · auto DC → bunched transition",
        fontsize=12, fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(path_timeline, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"\nWrote {path_timeline}")

    # ---- Phase space: input vs output --------------------------------
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    panels = [
        (0, 1, "x [mm]", "x' [mrad]"),
        (4, 5, "Δφ [deg]", "ΔW [MeV]"),
        (0, 2, "x [mm]", "y [mm]"),
    ]
    for row, (tag, p, colour) in enumerate([
        ("INPUT (DC, pre-LEBT)",  particles_in,  "C0"),
        ("EXIT (bunched, post-RFQ gap)", particles_out, "C3"),
    ]):
        for ax, (xi, yi, xl, yl) in zip(axes[row], panels):
            ax.plot(p[:, xi], p[:, yi], ".", ms=1.2, alpha=0.4, color=colour)
            ax.set_xlabel(xl); ax.set_ylabel(yl); ax.grid(alpha=0.3)
        axes[row, 0].set_ylabel(f"{tag}\n{axes[row, 0].get_ylabel()}")
    axes[0, 1].set_title("uniform phase + Gaussian ΔW (DC signature)")
    axes[1, 1].set_title("post-gap bunched (cavity has wrapped phase)")
    fig.suptitle(
        "DC LEBT demo · phase-space before and after auto-transition",
        fontsize=12, fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(path_phase, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {path_phase}")


if __name__ == "__main__":
    main()
