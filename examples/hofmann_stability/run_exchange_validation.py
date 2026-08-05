"""PIC tracking cross-validation of the Hofmann chart (the slow one).

Runs full 3-D-PIC multiparticle tracking through BOTH committed exchange
projects (``exchange_resonant.lgproj`` / ``exchange_control.lgproj`` —
the decks and SC-matched beams are read from those files, the single
source of truth) and plots the transverse↔longitudinal emittance
exchange that appears only on the channel whose depressed working point
the chart flags.

Outputs (next to this script):
* ``exchange_validation.npz``       — emittance histories + chart verdicts
* ``fig_exchange_validation.png``   — the three-panel comparison

Runtime ~10 min (2 × 150 cells × 20 000 particles, 32³ PIC grid).
``--plot-only`` re-renders the figure from an existing npz.

Run:  PYTHONPATH=<repo root> python run_exchange_validation.py
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
NPZ = HERE / "exchange_validation.npz"
PNG = HERE / "fig_exchange_validation.png"
CELL_MM = 200.0            # 4 x 30 mm drifts + 2 x 40 mm quads + thin gap

C_X, C_Y, C_Z = "#0284c7", "#38bdf8", "#f59e0b"
C_RES, C_CTL = "#0284c7", "#6b7280"


def _load_project(name):
    from linac_gen.core.config import BeamConfig
    from linac_gen.io.tracewin_parser import parse_tracewin

    proj = json.loads((HERE / f"{name}.lgproj").read_text())
    lat, meta = parse_tracewin(str(HERE / proj["lattice_path"]))
    assert not meta["warnings"], meta["warnings"]
    fields = {f.name for f in BeamConfig.__dataclass_fields__.values()}
    cfg = BeamConfig(**{k: v for k, v in proj["beam"].items()
                        if k in fields})
    return lat, cfg


def _chart_verdict(lat, cfg):
    from linac_gen.analysis.hofmann_stability import hofmann_stability
    from linac_gen.analysis.period_detect import detect_periods
    from linac_gen.analysis.phase_advance import run_phase_probe
    from linac_gen.cli.common import build_ref, _envelope_initial

    ref = build_ref(cfg)
    res = run_phase_probe(lat, ref, _envelope_initial(cfg, ref),
                          current=cfg.current)
    per = next(p for p in detect_periods(lat) if p.source != "fallback")
    tab = hofmann_stability(res, per)
    fin = np.isfinite(tab["R"])
    return {"R": float(np.nanmedian(tab["R"][fin])),
            "Y": float(np.nanmedian(tab["Y"][fin])),
            "gamma": float(tab["worst_growth"]),
            "n_flagged": int(tab["n_flagged"]),
            "n_cells": int(tab["n_cells"])}


def _run_one(name):
    from linac_gen.core.config import SpaceChargeConfig
    from linac_gen.core.simulation import Simulation
    from linac_gen.distributions.factory import create_beam

    lat, cfg = _load_project(name)
    verdict = _chart_verdict(lat, cfg)
    print(f"[{name}] chart: (R={verdict['R']:.2f}, Y={verdict['Y']:.2f}) "
          f"flags {verdict['n_flagged']}/{verdict['n_cells']} "
          f"gamma={verdict['gamma']:.4f}", flush=True)
    beam = create_beam(cfg, seed=42)
    sc = SpaceChargeConfig(nx=32, ny=32, nz=32, grid_extent=5.0,
                           use_gpu="cpu")
    print(f"[{name}] tracking {len(lat.elements)} elements, "
          f"{cfg.n_particles} particles ...", flush=True)
    rec = Simulation(lat, beam, space_charge=sc).run()
    print(f"[{name}] done: eps_x {rec.emit_x[0]:.3f}->{rec.emit_x[-1]:.3f}, "
          f"eps_z {rec.emit_z_mmmrad[0]:.3f}->{rec.emit_z_mmmrad[-1]:.3f} "
          "mm.mrad", flush=True)
    return {"s": np.asarray(rec.s, float),
            "ex": np.asarray(rec.emit_x, float),
            "ey": np.asarray(rec.emit_y, float),
            "ez": np.asarray(rec.emit_z_mmmrad, float),
            "verdict": verdict}


def make_figure(data):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.3), dpi=160)
    for ax, tag in ((axes[0], "resonant"), (axes[1], "control")):
        v = data[f"{tag}_verdict"]
        cells = data[f"{tag}_s"] / CELL_MM
        ax.plot(cells, data[f"{tag}_ex"], lw=1.6, color=C_X,
                label=r"$\epsilon_x$")
        ax.plot(cells, data[f"{tag}_ey"], lw=1.6, color=C_Y,
                label=r"$\epsilon_y$")
        ax.plot(cells, data[f"{tag}_ez"], lw=1.6, color=C_Z,
                label=r"$\epsilon_z$ (mm·mrad equiv.)")
        ax.set_xlabel("cell number")
        ax.set_ylabel("rms emittance (mm·mrad)")
        ax.set_title(f"{tag.upper()} channel — chart: "
                     f"(R={v['R']:.2f}, Y={v['Y']:.2f}), "
                     f"{v['n_flagged']}/{v['n_cells']} flagged, "
                     f"γ/ν₀ₓ={v['gamma']:.3f}", fontsize=9.5)
        ax.set_ylim(1.6, 6.4)
        ax.set_xlim(0, 150)
        ax.legend(fontsize=8, loc="center right")
        ax.grid(alpha=0.25)
    ax = axes[2]
    for tag, col, lab in (("resonant", C_RES, "resonant (on l=2 band)"),
                          ("control", C_CTL, "control (off band)")):
        cells = data[f"{tag}_s"] / CELL_MM
        ax.plot(cells, data[f"{tag}_ez"] / data[f"{tag}_ex"], lw=1.8,
                color=col, label=lab)
    ax.axhline(1.0, color=C_CTL, lw=1.0, ls="--")
    ax.annotate("equipartition", xy=(4, 1.04), fontsize=8, color=C_CTL)
    ax.set_xlabel("cell number")
    ax.set_ylabel(r"anisotropy  $\epsilon_z/\epsilon_x$")
    ax.set_title("Emittance-exchange progress\n(anisotropy relaxing "
                 "toward equipartition)", fontsize=9.5)
    ax.set_xlim(0, 150)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(PNG)
    print(f"wrote {PNG.name}")


def main():
    if "--plot-only" in sys.argv:
        data = dict(np.load(NPZ, allow_pickle=True))
        data = {k: (v.item() if v.dtype == object else v)
                for k, v in data.items()}
        make_figure(data)
        return
    warnings.simplefilter("ignore")
    data = {}
    for tag, name in (("resonant", "exchange_resonant"),
                      ("control", "exchange_control")):
        out = _run_one(name)
        for k, v in out.items():
            data[f"{tag}_{k}"] = v
    np.savez(NPZ, **data)
    print(f"wrote {NPZ.name}")
    make_figure(data)


if __name__ == "__main__":
    main()
