"""Validation figures for the corrected Hofmann solver (fast, no tracking).

Writes next to this script:

* ``fig_validation_fidelity.png`` — (a) growth-rate sweeps with the source
  manuscript package's solver overlaid point-by-point (bitwise identical;
  panel drawn HELIX-only when the external package is absent), and
  (b) the isotropic-limit reduction of every branch to Hofmann's printed
  Eqs. (37)/(42)/(46), at machine precision.
* ``fig_validation_physics.png`` — (a) per-branch anatomy of the two
  flags found by the 46-period demo at 5 mA (the l=3-odd flag an
  l=2-only chart misses), and (b) an anisotropy margin scan showing a
  quiet cell's higher-order onset.  Needs ``hofmann_demo_results.json``
  (run ``run_stability_demo.py`` first).

Run:  PYTHONPATH=<repo root> python make_validation_figures.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from linac_gen.analysis.hofmann_dispersion import (
    DispersionSolver,
    l3_even_D,
    l4_even_D,
    l4_odd_D,
    map_to_pre_grid,
)

HERE = Path(__file__).resolve().parent
#: source manuscript package (external; panel degrades gracefully without it)
PAPER_PKG = Path.home() / "Desktop/Projects/PIP_II/Paper/PRAB_Hofmann"

C_HELIX, C_PAPER, C_GREEN = "#0284c7", "#f59e0b", "#10b981"
C_MUTED, C_FLAG = "#6b7280", "#dc2626"


def _paper_solver():
    if not (PAPER_PKG / "analysis" / "dispersion.py").is_file():
        return None
    sys.path.insert(0, str(PAPER_PKG))
    from analysis.dispersion import DispersionSolver as PaperSolver
    return PaperSolver()


# ---------------------------------------------------------------------------
# printed isotropic forms (transcribed from Hofmann 1998 / manuscript App. A)
# ---------------------------------------------------------------------------
def _D3e_iso_eq37(s, sp2):
    return (8.0 + 12.0 * sp2 / (9.0 - s)
            - 4.0 * sp2 ** 2 * s * (3.0 - s)
            / ((9.0 - s) ** 2 * (1.0 - s) ** 2))


def _D4e_iso_eq42(s, sp2):
    g16, g4 = 16.0 - s, 4.0 - s
    return (16.0 + sp2 * (44.0 / g16 - 4.0 / g4)
            + sp2 ** 2 * (-34.0 / g16 ** 2 + 2.0 / g4 ** 2)
            + sp2 ** 3 * (6.0 / g16 ** 3 - 2.0 / (g16 * g4 ** 2)
                          + 4.0 / (g16 ** 2 * g4)))


def _D4o_iso_eq46(s, sp2):
    g16, g4 = 16.0 - s, 4.0 - s
    return (16.0 + sp2 * (4.0 / g4 + 20.0 / g16)
            + sp2 ** 2 * (4.0 / g16 ** 2 + 4.0 / (g16 * g4)))


def fig_fidelity():
    hx, pp = DispersionSolver(), _paper_solver()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2), dpi=160)

    eps_arr = np.linspace(1.0, 9.0, 161)
    sweeps = [("(R=2.10, Y=0.30)", 2.10, 0.30),
              ("(R=1.10, Y=0.40)", 1.10, 0.40)]
    max_diff = 0.0
    for k, (label, R, Y) in enumerate(sweeps):
        g_h = np.array([hx.growth_rate(R=R, Y=Y, eps_ratio=float(e))
                        for e in eps_arr])
        ax1.plot(eps_arr, g_h, lw=2.0, color=[C_HELIX, C_PAPER][k],
                 label=f"HELIX  l=2+3+4 @ {label}")
        if pp is not None:
            sub = eps_arr[::16]
            g_p = np.array([pp.growth_rate(R=R, Y=Y, eps_ratio=float(e))
                            for e in sub])
            max_diff = max(max_diff, float(np.max(np.abs(g_p - g_h[::16]))))
            ax1.plot(sub, g_p, "o", ms=7, mfc="none", mew=1.6,
                     color=[C_HELIX, C_PAPER][k],
                     label="paper solver (same points)" if k == 0 else None)
    note = (f"max |Δ| over open circles = {max_diff:.1e}" if pp is not None
            else "source package not found — HELIX curves only")
    ax1.annotate(note, xy=(0.03, 0.93), xycoords="axes fraction",
                 fontsize=9, color=C_MUTED)
    ax1.set_xlabel(r"emittance ratio  $\epsilon_z/\epsilon_x$")
    ax1.set_ylabel(r"max coherent growth  $\gamma/\nu_{0x}$")
    ax1.set_title("Port fidelity: HELIX curve vs source-package points")
    ax1.legend(fontsize=8, loc="center right")
    ax1.grid(alpha=0.25)

    rng = np.random.default_rng(0)
    s_arr = -np.logspace(-2, 2.2, 300)
    sp2 = 3.0
    for fn, ref, lab, col in (
            (l3_even_D, _D3e_iso_eq37, "l=3 even vs Eq. (37)", C_HELIX),
            (l4_even_D, _D4e_iso_eq42, "l=4 even vs Eq. (42)", C_PAPER),
            (l4_odd_D, _D4o_iso_eq46, "l=4 odd vs Eq. (46)", C_GREEN)):
        got = np.array([fn(s, 1.0, 1.0, sp2) for s in s_arr])
        want = np.array([ref(s, sp2) for s in s_arr])
        rel = np.abs(got - want) / np.maximum(np.abs(want), 1e-300)
        ax2.plot(-s_arr, np.maximum(rel, 1e-18), lw=1.6, color=col,
                 label=lab)
    ax2.axhline(2.2e-16, color=C_MUTED, lw=1.0, ls="--")
    ax2.annotate("double-precision epsilon", xy=(0.03, 3e-16), fontsize=8,
                 color=C_MUTED)
    ax2.set_xscale("log")
    ax2.set_yscale("log")
    ax2.set_ylim(1e-18, 1e-12)
    ax2.set_xlabel(r"$-s$  (mode-frequency variable)")
    ax2.set_ylabel("relative deviation from printed form")
    ax2.set_title(r"Isotropic limit $\alpha=\hat\eta=1$: reduction to"
                  "\nHofmann's published equations ($S^2$ = 3)")
    ax2.legend(fontsize=8, loc="upper right")
    ax2.grid(alpha=0.25, which="both")
    fig.tight_layout()
    out = HERE / "fig_validation_fidelity.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"wrote {out.name}")


def fig_physics():
    results = HERE / "hofmann_demo_results.json"
    if not results.is_file():
        print(f"SKIP {results.name} missing — run run_stability_demo.py "
              "first for fig_validation_physics.png")
        return
    rep = json.loads(results.read_text())
    cells5 = rep["scenarios"]["bunched_5mA"]
    cellA = next(c for c in cells5["A"]["cells"] if c["flagged"])
    cellB = next(c for c in cells5["B"]["cells"] if c["flagged"])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2), dpi=160)
    branches = ["g_l2", "g_l3_even", "g_l3_odd", "g_l4_even"]
    blabels = ["l=2", "l=3 even", "l=3 odd", "l=4 even"]
    x = np.arange(len(branches))
    w = 0.38
    for off, cell, name, col in (
            (-w / 2, cellA,
             f"section A cell {cellA['cell']} (R={cellA['R']:.2f})", C_HELIX),
            (+w / 2, cellB,
             f"section B cell {cellB['cell']} (R={cellB['R']:.2f})", C_PAPER)):
        vals = [cell[b] or 0.0 for b in branches]
        ax1.bar(x + off, vals, w, color=col, label=name,
                edgecolor="white", linewidth=0.6)
        for xb, v in zip(x + off, vals):
            if v > 0:
                ax1.annotate(f"{v:.3f}", (xb, v), ha="center", va="bottom",
                             fontsize=8, color="#111827")
    ax1.axhline(0.01, color=C_FLAG, lw=1.4, ls="--")
    ax1.annotate("flag threshold γ/ν₀ₓ = 0.01", xy=(2.0, 0.0115),
                 fontsize=9, color=C_FLAG)
    ax1.set_xticks(x, blabels)
    ax1.set_ylabel(r"growth rate  $\gamma/\nu_{0x}$")
    ax1.set_title("Demo linac, 5 mA — per-branch anatomy of the two flags\n"
                  "(section A is driven by l=3 odd: invisible to an "
                  "l=2-only chart)")
    ax1.legend(fontsize=8)
    ax1.grid(alpha=0.25, axis="y")

    c15 = rep["scenarios"]["bunched_15mA"]["C"]["cells"]
    cell = min((c for c in c15 if c["margin"] is not None),
               key=lambda c: c["margin"])
    R, Y = cell["R"], cell["Y"]
    eps_scan = np.arange(1.0, 9.001, 0.05)
    s2 = map_to_pre_grid(np.full_like(eps_scan, R),
                         np.full_like(eps_scan, Y), eps_scan)[2]
    dom = s2 <= 10.0
    g_ho = DispersionSolver().growth_rate_batch(
        np.full(int(dom.sum()), R), np.full(int(dom.sum()), Y),
        eps_scan[dom], l_modes=(3, 4))
    ax2.plot(eps_scan[dom], g_ho, lw=2.0, color=C_HELIX,
             label="higher-order growth (l=3, l=4e)")
    ax2.axhline(0.01, color=C_FLAG, lw=1.4, ls="--", label="onset threshold")
    ax2.axvline(cell["eps_ratio"], color=C_MUTED, lw=1.4, ls=":",
                label=f"design ε ratio = {cell['eps_ratio']:.2f}")
    ax2.axvline(cell["onset_eps"], color=C_GREEN, lw=1.6,
                label=f"onset = {cell['onset_eps']:.2f} "
                      f"(margin ×{cell['margin']:.2f})")
    ax2.set_xlabel(r"emittance ratio  $\epsilon_z/\epsilon_x$")
    ax2.set_ylabel(r"growth rate  $\gamma/\nu_{0x}$")
    ax2.set_title(f"Anisotropy margin scan — demo section C cell "
                  f"{cell['cell']} at 15 mA\n(R={R:.2f}, Y={Y:.2f}; "
                  "quiet at design, onset found upslope)")
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.25)
    fig.tight_layout()
    out = HERE / "fig_validation_physics.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"wrote {out.name}")


if __name__ == "__main__":
    fig_fidelity()
    fig_physics()
