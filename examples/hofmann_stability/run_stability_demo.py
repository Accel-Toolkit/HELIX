"""Scenario-matrix demo of the corrected Hofmann stability analysis.

Drives the 46-period demo linac (``make_lattice.py``) through the full
Hofmann stack in every regime the analysis distinguishes:

* bunched beam at I = 0, 5, 15, 30 mA — tune depression sweeping the
  chart ordinate; per-section flags, anisotropy margins, Monte-Carlo
  instability probabilities;
* a DC (continuous) beam — must refuse with the no-longitudinal-tune
  reason;
* the solenoid section D — must refuse with the x-y-coupled reason in
  every scenario.

Outputs (written next to this script):

* ``hofmann_demo_results.json`` — the full per-scenario, per-section,
  per-cell table;
* ``hofmann_demo_chart.png``    — growth-rate chart with the three quad
  sections' trajectories at the highest current;
* a console summary table.

Run:  PYTHONPATH=<repo root> python run_stability_demo.py
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np

from linac_gen.analysis.hofmann_dispersion import DispersionSolver
from linac_gen.analysis.hofmann_probabilistic import instability_probability
from linac_gen.analysis.hofmann_stability import (
    anisotropy_margin,
    hofmann_stability,
)
from linac_gen.analysis.period_detect import detect_periods
from linac_gen.core.particle import PROTON
from linac_gen.core.reference import ReferenceParticle
from linac_gen.io.tracewin_parser import parse_tracewin
from linac_gen.tracking.envelope import EnvelopeSolver
from linac_gen.tracking.matrix_tracking import (
    compute_transfer_matrix,
    compute_twiss,
)

HERE = Path(__file__).resolve().parent
DECK = HERE / "hofmann_demo_linac.dat"
OUT_JSON = HERE / "hofmann_demo_results.json"
OUT_PNG = HERE / "hofmann_demo_chart.png"

W_KIN_MEV = 2.5
FREQ_MHZ = 162.5
EMIT_T = 2.0        # geometric transverse emittance, mm.mrad
EMIT_Z = 0.035      # native longitudinal emittance, deg.MeV (geometric eps_z/eps_x ~ 1.3 at injection)
CURRENTS = (0.0, 5.0, 15.0, 30.0)
N_MC = 200

SECTION_NAMES = ("A", "B", "C", "D")


def _matched_init(lat, ref, period):
    """Injection matched to the FIRST cell of ``period`` (all 3 planes)."""
    a, b = period.spans()[0]
    M = compute_transfer_matrix(lat, ref, start=a, end=b - 1)
    twx = compute_twiss(M, "x")
    twy = compute_twiss(M, "y")
    twz = compute_twiss(M, "z")
    return dict(alpha_x=twx["alpha"], beta_x=twx["beta"], emit_x=EMIT_T,
                alpha_y=twy["alpha"], beta_y=twy["beta"], emit_y=EMIT_T,
                alpha_z=twz["alpha"], beta_z=twz["beta"], emit_z=EMIT_Z)


def _f(x):
    x = float(x)
    return x if np.isfinite(x) else None


def _section_report(tab, mg, prob):
    fin = np.isfinite(tab["g_combined"])
    rows = []
    for k in range(int(tab["n_cells"])):
        rows.append({
            "cell": int(tab["cells"][k]),
            "R": _f(tab["R"][k]), "Y": _f(tab["Y"][k]),
            "eps_ratio": _f(tab["eps_ratio"][k]), "S2": _f(tab["S2"][k]),
            "g_l2": _f(tab["g_l2"][k]),
            "g_l3_even": _f(tab["g_l3_even"][k]),
            "g_l3_odd": _f(tab["g_l3_odd"][k]),
            "g_l4_even": _f(tab["g_l4_even"][k]),
            "g_combined": _f(tab["g_combined"][k]),
            "valid": bool(tab["valid"][k]),
            "flagged": bool(tab["flagged"][k]),
            "flagged_extrap": bool(tab["flagged_extrap"][k]),
            "onset_eps": _f(mg["onset_eps"][k]) if mg else None,
            "margin": _f(mg["margin"][k]) if mg else None,
            "p_unstable": _f(prob[k]) if prob is not None else None,
        })
    return {
        "reason": tab["reason"],
        "n_cells": int(tab["n_cells"]),
        "n_valid": int(tab["n_valid"]),
        "n_flagged": int(tab["n_flagged"]),
        "worst_cell": _f(tab["worst_cell"]),
        "worst_growth": _f(tab["worst_growth"]),
        "n_finite": int(fin.sum()),
        "margin_smallest_smooth":
            _f(mg["smallest_smooth_margin"]) if mg else None,
        "p_max": _f(np.nanmax(prob)) if prob is not None
        and np.isfinite(prob).any() else None,
        "cells": rows,
    }


def run_scenarios():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        lat, meta = parse_tracewin(str(DECK))
    assert not meta["warnings"], meta["warnings"]
    periods = [p for p in detect_periods(lat) if p.source != "fallback"]
    assert len(periods) == 4, [p.label for p in periods]
    sections = dict(zip(SECTION_NAMES, periods))

    ref0 = ReferenceParticle(species=PROTON, w_kin=W_KIN_MEV,
                             frequency=FREQ_MHZ)
    init = _matched_init(lat, ref0, sections["A"])

    report = {"deck": DECK.name, "n_periods_total":
              sum(p.n_repeats for p in periods), "scenarios": {}}
    keep = {}                                  # (label) -> (tab, results)

    scenarios = [(f"bunched_{int(cur)}mA", cur, False) for cur in CURRENTS]
    scenarios.append(("dc_15mA", 15.0, True))

    for label, current, dc in scenarios:
        scen_init = dict(init)
        if dc:
            scen_init.update(alpha_z=0.0, beta_z=1.0, emit_z=0.0,
                             continuous=True)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = EnvelopeSolver(lat, ref0.copy(), scen_init,
                                 current=current, phase_probe=True).run()
        scen = {}
        for name, period in sections.items():
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                tab = hofmann_stability(res, period)
                mg = (anisotropy_margin(res, period, coords=tab)
                      if tab["reason"] is None else None)
                prob = (instability_probability(tab, N_mc=N_MC)
                        if tab["reason"] is None and current > 0 else None)
            scen[name] = _section_report(tab, mg, prob)
            keep[(label, name)] = (tab, res)
        report["scenarios"][label] = scen
    return report, keep, sections


def print_summary(report):
    print(f"\ndeck: {report['deck']}  "
          f"({report['n_periods_total']} periods in 4 sections)")
    hdr = (f"{'scenario':>13s} {'sec':>3s} {'cells':>5s} {'valid':>5s} "
           f"{'flags':>5s} {'worst γ/ν0x':>11s} {'min margin':>10s} "
           f"{'max P':>6s}  outcome")
    print(hdr)
    print("-" * len(hdr))
    for label, scen in report["scenarios"].items():
        for name, s in scen.items():
            if s["reason"]:
                outcome = ("refused: coupled" if "coupled" in s["reason"]
                           else "refused: " + s["reason"].split(":")[0])
                print(f"{label:>13s} {name:>3s} {s['n_cells']:>5d} "
                      f"{'—':>5s} {'—':>5s} {'—':>11s} {'—':>10s} "
                      f"{'—':>6s}  {outcome}")
                continue
            wg = "—" if s["worst_growth"] is None else f"{s['worst_growth']:.4f}"
            mm = ("—" if s["margin_smallest_smooth"] is None
                  else f"x{s['margin_smallest_smooth']:.2f}")
            pm = "—" if s["p_max"] is None else f"{s['p_max']:.2f}"
            print(f"{label:>13s} {name:>3s} {s['n_cells']:>5d} "
                  f"{s['n_valid']:>5d} {s['n_flagged']:>5d} "
                  f"{wg:>11s} {mm:>10s} {pm:>6s}  ok")


def make_figure(keep, scenario_label):
    """Growth chart + section trajectories for one bunched scenario."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    tabs = {n: keep[(scenario_label, n)][0] for n in ("A", "B", "C")}
    eps_all = np.concatenate([t["eps_ratio"] for t in tabs.values()])
    eps_used = float(np.nanmedian(eps_all))
    r_arr, y_arr, G = DispersionSolver().chart(
        eps_used, r_max=2.6, steps=140)

    fig, ax = plt.subplots(figsize=(8.2, 5.2), dpi=160)
    im = ax.pcolormesh(r_arr, y_arr, G, cmap="viridis", shading="auto",
                       rasterized=True)
    cb = fig.colorbar(im, ax=ax, pad=0.015)
    cb.set_label(r"coherent growth  Im$\,\omega/\nu_{0x}$")

    # Fixed categorical order for the three sections; identity is carried
    # by color + direct label, never color alone.
    colors = {"A": "#7dd3fc", "B": "#fbbf24", "C": "#f9a8d4"}
    r_lim = 2.6
    for name, tab in tabs.items():
        fin = np.isfinite(tab["R"]) & np.isfinite(tab["Y"])
        in_range = fin & (tab["R"] <= r_lim)
        if not in_range.any():
            # e.g. a weakly-focused section whose transverse plane is far
            # more depressed than z pushes R beyond the plotted chart.
            r_hi = np.nanmax(tab["R"][fin]) if fin.any() else float("nan")
            ax.plot([], [], "o-", color=colors[name],
                    label=f"section {name} — off scale (R up to {r_hi:.1f})")
            continue
        ax.plot(tab["R"][in_range], tab["Y"][in_range], "o-", ms=4.5,
                lw=1.1, color=colors[name], mec="white", mew=0.4,
                label=f"section {name}")
        ax.annotate(name, (tab["R"][in_range][-1], tab["Y"][in_range][-1]),
                    textcoords="offset points", xytext=(7, 4),
                    color="white", fontsize=10, fontweight="bold")
        flag = fin & tab["flagged"]
        if flag.any():
            ax.plot(tab["R"][flag], tab["Y"][flag], "o", ms=10, mfc="none",
                    mec="#ef4444", mew=1.8, label="_nolegend_")
        ext = fin & ~tab["valid"]
        if ext.any():
            ax.plot(tab["R"][ext], tab["Y"][ext], "s", ms=9, mfc="none",
                    mec="#9ca3af", mew=1.2, label="_nolegend_")
    ax.plot([], [], "o", mfc="none", mec="#ef4444", mew=1.8,
            label="flagged (γ/ν₀ₓ>0.01, S²≤10)")
    ax.plot([], [], "s", mfc="none", mec="#9ca3af", mew=1.2,
            label="S²>10 (extrapolation)")
    ax.set_xlabel(r"tune ratio  $\nu_z/\nu_x$  (depressed)")
    ax.set_ylabel(r"tune depression  $\nu_x/\nu_{0x}$")
    ax.set_title(f"Corrected Hofmann chart at "
                 f"$\\epsilon_z/\\epsilon_x$ = {eps_used:.2f} — "
                 f"{scenario_label.replace('_', ' ')}")
    ax.set_xlim(0.0, 2.6)
    ax.set_ylim(y_arr[0], 1.05)     # headroom: mismatch can push cells to eta ~ 1
    ax.legend(loc="lower right", fontsize=8, framealpha=0.85)
    fig.tight_layout()
    fig.savefig(OUT_PNG)
    plt.close(fig)
    print(f"wrote {OUT_PNG}")


def main():
    report, keep, _sections = run_scenarios()
    print_summary(report)
    OUT_JSON.write_text(json.dumps(report, indent=1))
    print(f"\nwrote {OUT_JSON}")
    # The 5 mA point is the instructive one: it carries the in-domain
    # l=3-odd flag (section A) and the R=1 l=2 flag (section B).
    make_figure(keep, "bunched_5mA")


if __name__ == "__main__":
    main()
