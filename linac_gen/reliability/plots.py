"""Campaign figures (PNG, headless matplotlib; never imported by Qt code)."""
from __future__ import annotations

from pathlib import Path

import numpy as np

INK, RED, BLUE, GREY, GREEN = "#22223b", "#c1121f", "#023e8a", "#6c757d", "#2a9d8f"


def _plt():
    import matplotlib
    if matplotlib.get_backend().lower() not in ("agg", "module://matplotlib_inline.backend_inline"):
        try:
            matplotlib.use("Agg")
        except Exception:               # noqa: BLE001
            pass
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 9, "axes.edgecolor": INK, "axes.spines.top": False,
                         "axes.spines.right": False, "figure.facecolor": "white"})
    return plt


def _save(fig, path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=110, bbox_inches="tight")
    import matplotlib.pyplot as plt
    plt.close(fig)
    return str(path)


def write_figures(campaign, faults, comps, seeds, fos, foil, avail) -> list[str]:
    plt = _plt()
    out: list[str] = []
    fd = campaign.figures_dir
    rows = [r for r in faults if r.get("criticality") is not None and r["bracket"] == "rephased"]
    if rows:
        fig, ax = plt.subplots(figsize=(max(5.0, 0.28 * len(rows) + 2), 3.2))
        labels = [r["case_id"] for r in rows]
        vals = [r["criticality"] for r in rows]
        cols = [RED if r.get("critical") else BLUE for r in rows]
        ax.bar(range(len(rows)), vals, color=cols)
        ax.set_xticks(range(len(rows)))
        ax.set_xticklabels(labels, rotation=90, fontsize=7)
        ax.set_ylabel("criticality score")
        ax.set_title("Fault leg — criticality per case (red = critical by the rule)")
        out.append(_save(fig, fd / "criticality_by_case.png"))
        fr = {r["case_id"].replace("S10_", "S1_").replace("_fr", "_re"): r for r in faults
              if r["bracket"] == "frozen" and r.get("criticality") is not None}
        pairs = [(r, fr.get(r["case_id"])) for r in rows if r["class"] == "S1"]
        pairs = [(a, b) for a, b in pairs if b is not None]
        if pairs:
            fig, ax = plt.subplots(figsize=(5.5, 3.2))
            x = np.arange(len(pairs))
            ax.bar(x - 0.2, [a.get("d_energy_mev") or 0.0 for a, _b in pairs], 0.4, color=BLUE, label="re-phased")
            ax.bar(x + 0.2, [b.get("d_energy_mev") or 0.0 for _a, b in pairs], 0.4, color=RED, label="frozen phases")
            ax.set_xticks(x); ax.set_xticklabels([a["elements"][0] if a["elements"] else a["case_id"] for a, _b in pairs],
                                                 rotation=90, fontsize=7)
            ax.set_ylabel("exit-energy deficit (MeV)"); ax.legend(frameon=False)
            ax.set_title("Single cavity off — both RF brackets")
            out.append(_save(fig, fd / "bracket_frozen_vs_rephased.png"))
    if comps:
        fig, ax = plt.subplots(figsize=(5.5, 3.0))
        labels = [f"{c['case_id']}\n{c['strategy']}" for c in comps]
        vals = [1.0 if c.get("recovered_by_rule") else 0.0 for c in comps]
        ax.bar(range(len(comps)), vals, color=[GREEN if v else RED for v in vals])
        ax.set_xticks(range(len(comps))); ax.set_xticklabels(labels, rotation=90, fontsize=6)
        ax.set_yticks([0, 1]); ax.set_yticklabels(["not recovered", "recovered"])
        ax.set_title("Compensation outcome per (case, strategy) — by the study rule")
        out.append(_save(fig, fd / "compensation_recovery.png"))
    sd = [s for s in seeds if s["kind"] == "seed" and s["status"] == "ok" and s.get("transmission") is not None]
    if sd:
        fig, ax = plt.subplots(figsize=(5.0, 3.0))
        ax.hist([s["transmission"] for s in sd], bins=min(20, max(3, len(sd))), color=BLUE)
        ax.set_xlabel("transmission (%)"); ax.set_ylabel("seeds")
        ax.set_title("Imperfection seeds — transmission")
        out.append(_save(fig, fd / "seeds_transmission.png"))
    sd_e = [s for s in seeds if s["kind"] == "seed" and s["status"] == "ok" and s.get("emit_nx") is not None]
    if sd_e:
        fig, ax = plt.subplots(figsize=(5.0, 3.0))
        ax.plot([s["seed"] for s in sd_e], [s["emit_nx"] for s in sd_e], "o", color=RED, ms=3, label="εn,x")
        ax.plot([s["seed"] for s in sd_e], [s["emit_ny"] for s in sd_e], "s", color=BLUE, ms=3, label="εn,y")
        ax.set_xlabel("seed"); ax.set_ylabel("exit normalised emittance (mm mrad)"); ax.legend(frameon=False)
        ax.set_title("Imperfection seeds — exit emittances")
        out.append(_save(fig, fd / "seeds_emittance.png"))
    if fos:
        by = {}
        for f in fos:
            if f.get("critical") is None:
                continue
            k = (f["case_id"], f["variant"])
            by.setdefault(k, []).append(0.0 if f["critical"] else 1.0)
        if by:
            fig, ax = plt.subplots(figsize=(5.5, 3.0))
            keys = sorted(by)
            ax.bar(range(len(keys)), [np.mean(by[k]) for k in keys],
                   color=[GREEN if k[1] == "after" else GREY for k in keys])
            ax.set_xticks(range(len(keys))); ax.set_xticklabels([f"{k[0]}\n{k[1]}" for k in keys], rotation=90, fontsize=6)
            ax.set_ylabel("fraction of seeds not critical"); ax.set_ylim(0, 1.05)
            ax.set_title("Faults on error seeds — robustness (grey before, green after compensation)")
            out.append(_save(fig, fd / "faults_on_seeds_robustness.png"))
    fl = [r for r in foil if r["status"] == "ok"]
    if fl:
        fig, ax = plt.subplots(figsize=(5.5, 3.0))
        ax.bar(range(len(fl)), [r.get("foil_sigma_x_mm") or 0.0 for r in fl], 0.4, color=RED, label="σx")
        ax.bar([i + 0.4 for i in range(len(fl))], [r.get("foil_sigma_y_mm") or 0.0 for r in fl], 0.4, color=BLUE, label="σy")
        ax.set_xticks([i + 0.2 for i in range(len(fl))]); ax.set_xticklabels([r["case_id"] for r in fl], rotation=90, fontsize=7)
        ax.set_ylabel("rms size at the foil (mm)"); ax.legend(frameon=False); ax.set_title("Foil scenarios — spot size")
        out.append(_save(fig, fd / "foil_spot.png"))
        th = [(float(r["case_id"].split("_")[1]), r) for r in fl if r["case_id"].startswith("thick_")]
        nom = [r for r in fl if r["case_id"] == "nominal"]
        curve = [(t, r.get("strip_eff_analytic")) for t, r in th if r.get("strip_eff_analytic") is not None]
        if curve:
            fig, ax = plt.subplots(figsize=(5.0, 3.0))
            curve.sort()
            ax.plot([t for t, _ in curve], [100.0 * e for _, e in curve], "o-", color=BLUE, label="two-step model")
            mp = [(t, r["strip_eff"]) for t, r in th if r.get("strip_eff") is not None]
            if mp:
                ax.plot([t for t, _ in mp], [100.0 * e for _, e in mp], "s", color=RED, label="tracked")
            ax.set_xlabel("thickness (µg/cm²)"); ax.set_ylabel("stripped (%)"); ax.legend(frameon=False)
            ax.set_title("Stripping efficiency vs thickness (calibrated model)")
            out.append(_save(fig, fd / "foil_efficiency_vs_thickness.png"))
        _ = nom
    if avail and avail.get("variants"):
        fig, ax = plt.subplots(figsize=(5.0, 3.0))
        names = list(avail["variants"])
        means = [avail["variants"][v]["availability_mean"] * 100 for v in names]
        lo = [means[i] - avail["variants"][v]["availability_p05"] * 100 for i, v in enumerate(names)]
        hi = [avail["variants"][v]["availability_p95"] * 100 - means[i] for i, v in enumerate(names)]
        ax.bar(range(len(names)), means, color=BLUE, yerr=[lo, hi], capsize=4)
        ax.set_xticks(range(len(names))); ax.set_xticklabels(names)
        ax.set_ylabel("availability (%)"); ax.set_ylim(max(0.0, min(means) - 5.0), 100.0)
        ax.set_title("Availability per SRF-trip variant (mean, p05–p95)")
        out.append(_save(fig, fd / "availability_hist.png"))
        first = avail["variants"][names[0]]
        bins = list(first["bins_s"])
        fig, ax = plt.subplots(figsize=(5.5, 3.0))
        w = 0.8 / max(1, len(names))
        for i, v in enumerate(names):
            h = avail["variants"][v]["trip_histogram"]
            ax.bar([j + i * w for j in range(len(h))], h, w, label=v)
        ax.set_xticks([j + 0.4 for j in range(len(bins) + 1)])
        ax.set_xticklabels([f"≤{b:g}s" for b in bins] + [">"], rotation=45, fontsize=7)
        ax.set_ylabel("trips per year"); ax.legend(frameon=False); ax.set_title("Beam-trip budget by duration")
        out.append(_save(fig, fd / "trip_budget.png"))
    return out
