"""Derived views of a campaign: per-leg CSV files, ``summary.json`` with the
verdicts against the study rule, the figures and the HTML report."""
from __future__ import annotations

import csv
import json
import logging
import time
from pathlib import Path

import numpy as np

_log = logging.getLogger(__name__)

FAULT_COLS = ("id", "case_id", "class", "label", "elements", "bracket", "mode", "verification", "status",
              "transmission", "ref_w_kin", "d_energy_mev", "energy_dev_pct", "d_phi_end_deg",
              "emit_nx", "emit_ny", "emit_nz", "emit_nx_growth_pct", "emit_ny_growth_pct",
              "emit_nz_growth_pct", "loss_frac", "loss_w_total", "loss_w_per_m_peak",
              "loss_w_per_m_peak_s_m", "treaty_w_kin", "treaty_phi_deg", "treaty_sigma_x",
              "treaty_sigma_y", "foil_sigma_x_mm", "foil_sigma_y_mm", "criticality", "critical",
              "beam_lost", "recovered_by", "error", "results_path")
COMP_COLS = ("case_id", "class", "label", "bracket", "strategy", "recovered_by_rule", "recovered",
             "after_critical", "match_success", "compensators", "n_knobs", "ref_w_kin_after",
             "transmission_after", "emit_nx_after", "emit_ny_after", "emit_nz_after", "residual_cost",
             "message", "elapsed", "error", "settings")
SEED_COLS = ("id", "seed", "kind", "status", "transmission", "ref_w_kin", "emit_nx", "emit_ny",
             "emit_nz", "sigma_x", "sigma_y", "loss_w_total", "loss_w_per_m_peak",
             "correction_status", "orbit_rms_before_mm", "orbit_rms_after_mm", "correction_passes",
             "n_draws", "error", "results_path")
FOS_COLS = ("id", "case_id", "class", "label", "seed", "variant", "status", "transmission",
            "ref_w_kin", "d_energy_vs_seed_mev", "emit_nx_growth_vs_seed_pct",
            "loss_w_total", "critical", "error")
FOIL_COLS = ("id", "case_id", "label", "mode", "status", "ref_w_kin", "foil_sigma_x_mm",
             "foil_sigma_y_mm", "foil_w_cm2_peak", "strip_eff", "h0_frac", "missed_frac",
             "n_at_foil", "strip_eff_analytic", "h0_frac_analytic", "transmission", "error")


def _write_csv(path: Path, cols, rows, header_note: str | None = None) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        if header_note:
            for line in header_note.splitlines():
                fh.write(f"# {line}\n")
        w = csv.DictWriter(fh, fieldnames=list(cols), extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: _fmt(r.get(k)) for k in cols})
    return path


def _fmt(v):
    if v is None:
        return ""
    if isinstance(v, float):
        return f"{v:.10g}"
    if isinstance(v, (list, tuple)):
        return ";".join(str(x) for x in v)
    if isinstance(v, dict):
        return json.dumps(v, sort_keys=True)
    return v


def read_csv(path) -> list[dict]:
    with open(path, encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(ln for ln in fh if not ln.startswith("#")))


# ---------------------------------------------------------------------------
def fault_table(campaign) -> list[dict]:
    rows = campaign.fault_rows()
    comps = {}
    for c in campaign.compensation_rows():
        res = c["result"]
        if res.get("error"):
            continue
        comps.setdefault(c["item"]["case_id"], []).append((c["item"]["strategy"], bool(res.get("recovered_by_rule"))))
    out = []
    for r in rows:
        it, m, crit = r["item"], r["row"], r.get("criticality")
        rec = {"id": it["id"], "case_id": it["case_id"], "class": it["cls"], "label": it["label"],
               "elements": it.get("elements", []), "bracket": it["bracket"], "mode": it["mode"],
               "verification": it.get("strategy") or "",
               "status": r["status"].get("status"), "error": r["status"].get("error"),
               "results_path": str(campaign.item_dir(it) / "results.h5")}
        for k in ("transmission", "ref_w_kin", "emit_nx", "emit_ny", "emit_nz", "loss_w_total",
                  "loss_w_per_m_peak", "loss_w_per_m_peak_s_m", "treaty_w_kin", "treaty_phi_deg",
                  "treaty_sigma_x", "treaty_sigma_y", "foil_sigma_x_mm", "foil_sigma_y_mm"):
            rec[k] = m.get(k)
        for k in m:
            if k.startswith("loss_w_per_m_peak_"):
                rec[k] = m[k]
        if crit:
            rec.update({"criticality": crit["score"], "critical": crit["critical"],
                        "beam_lost": crit["beam_lost"], "d_energy_mev": crit["d_energy_mev"],
                        "d_phi_end_deg": crit["d_phi_end_deg"]})
            rec.update({k: v for k, v in crit["rule_terms"].items()})
            if it.get("strategy"):
                # a multiparticle re-run of a compensated case: not a scenario
                rec["recovered_by"] = f"verified:{it['strategy']}"
            else:
                rec["recovered_by"] = _recovered_by(crit, comps.get(it["case_id"]))
        else:
            rec.update({"criticality": None, "critical": None, "beam_lost": None,
                        "recovered_by": None if it["case_id"] != "baseline" else "baseline"})
        out.append(rec)
    return out


def _recovered_by(crit: dict, comps) -> str:
    if not crit["critical"]:
        return "auto_rephase"
    if comps:
        ok = [s for s, rec in comps if rec]
        if ok:
            return "operator_retune:" + "+".join(ok)
        return "unrecoverable"
    return "not_compensated"


def compensation_table(campaign) -> list[dict]:
    out = []
    for c in campaign.compensation_rows():
        it, res = c["item"], c["result"]
        after = res.get("metrics_after") or {}
        out.append({"case_id": it["case_id"], "class": it["cls"], "label": it["label"],
                    "bracket": it["bracket"], "strategy": it["strategy"],
                    "recovered_by_rule": res.get("recovered_by_rule"),
                    "recovered": res.get("recovered"), "after_critical": res.get("after_critical"),
                    "match_success": res.get("match_success"),
                    "compensators": res.get("compensator_names") or [],
                    "n_knobs": len(res.get("settings") or {}),
                    "ref_w_kin_after": after.get("ref_w_kin"),
                    "transmission_after": after.get("transmission"),
                    "emit_nx_after": after.get("emit_nx"), "emit_ny_after": after.get("emit_ny"),
                    "emit_nz_after": after.get("emit_nz"),
                    "residual_cost": res.get("residual_cost"), "message": res.get("message"),
                    "elapsed": res.get("elapsed"), "error": res.get("error"),
                    "settings": res.get("settings") or {}})
    return out


def seed_tables(campaign) -> tuple[list[dict], list[dict]]:
    items = campaign.plan().get("imperfections", [])
    draws = {}
    seeds, fos = [], []
    for it in items:
        st = campaign.item_status(it)
        if st is None:
            continue
        if it.get("kind") == "draw":
            draws[it["case_id"]] = st
            continue
        if it.get("kind") != "point":
            continue
        m = dict(st.get("metrics") or {})
        m.update({k: v for k, v in (st.get("extras") or {}).items() if not k.startswith("_")})
        if it.get("cls") in ("seed", "control"):
            d = draws.get(it["case_id"], {})
            corr = d.get("correction") or {}
            hist = corr.get("history") or []
            seeds.append({"id": it["id"], "seed": it.get("seed"), "kind": it["cls"],
                          "status": st.get("status"), "error": st.get("error"),
                          "transmission": m.get("transmission"), "ref_w_kin": m.get("ref_w_kin"),
                          "emit_nx": m.get("emit_nx"), "emit_ny": m.get("emit_ny"), "emit_nz": m.get("emit_nz"),
                          "sigma_x": m.get("sigma_x"), "sigma_y": m.get("sigma_y"),
                          "loss_w_total": m.get("loss_w_total"), "loss_w_per_m_peak": m.get("loss_w_per_m_peak"),
                          "correction_status": corr.get("status"),
                          "orbit_rms_before_mm": corr.get("rms_before_mm"),
                          "orbit_rms_after_mm": (hist[-1].get("rms_orbit_mm") if hist else None),
                          "correction_passes": (len(hist) or None),
                          "n_draws": len(d.get("overrides") or []),
                          "results_path": str(campaign.item_dir(it) / "results.h5"), "_row": m})
        elif it.get("cls") == "faults_on_seeds":
            fos.append({"id": it["id"], "case_id": it["case_id"], "class": it["cls"], "label": it["label"],
                        "seed": it.get("seed"), "variant": it.get("variant"), "status": st.get("status"),
                        "error": st.get("error"), "transmission": m.get("transmission"),
                        "ref_w_kin": m.get("ref_w_kin"), "loss_w_total": m.get("loss_w_total"),
                        "_row": m, "_seed_item": it.get("draw")})
    by_seed = {s["seed"]: s for s in seeds if s["kind"] == "seed"}
    rule = (campaign.spec.criticality or {}).get("rule") or {}
    from linac_gen.reliability.observables import criticality_rule
    for f in fos:
        base = by_seed.get(f["seed"])
        if base and f["status"] == "ok" and base["status"] == "ok":
            b, m = base["_row"], f["_row"]
            crit, terms = criticality_rule(b, m, rule, campaign._sections())
            f["critical"] = crit
            f["d_energy_vs_seed_mev"] = (abs(m["ref_w_kin"] - b["ref_w_kin"])
                                         if m.get("ref_w_kin") is not None and b.get("ref_w_kin") is not None else None)
            f["emit_nx_growth_vs_seed_pct"] = terms.get("emit_nx_growth_pct")
        else:
            f["critical"] = None
    return seeds, fos


def foil_table(campaign) -> list[dict]:
    items = campaign.plan().get("foil", [])
    out = []
    lat, cfg, _v = campaign._load()
    foil_name = campaign.landmarks.get("foil_element")
    foil_el = lat.elements[campaign._elem_index[foil_name]] if foil_name in campaign._elem_index else None
    for it in items:
        st = campaign.item_status(it)
        if st is None:
            continue
        m = dict(st.get("metrics") or {})
        m.update({k: v for k, v in (st.get("extras") or {}).items() if not k.startswith("_")})
        rec = {"id": it["id"], "case_id": it["case_id"], "label": it["label"], "mode": it["mode"],
               "status": st.get("status"), "error": st.get("error"),
               "ref_w_kin": m.get("ref_w_kin"), "transmission": m.get("transmission")}
        for k in ("foil_sigma_x_mm", "foil_sigma_y_mm", "foil_w_cm2_peak", "strip_eff", "h0_frac",
                  "missed_frac", "n_at_foil"):
            rec[k] = m.get(k)
        rec["strip_eff_analytic"] = rec["h0_frac_analytic"] = None
        if foil_el is not None and m.get("ref_w_kin") is not None:
            try:
                from linac_gen.cli.common import build_ref
                thick = float(getattr(foil_el, "thickness_ug_cm2", 0.0))
                for sel, val in it.get("overrides", []):
                    if sel == f"{foil_name}.thickness_ug_cm2":
                        thick = float(val)
                ref = build_ref(cfg)
                ref.w_kin = float(m.get("ref_w_kin"))
                f_p, f_h0, _f = foil_el.stripping_fractions(ref, thick)
                rec["strip_eff_analytic"], rec["h0_frac_analytic"] = f_p, f_h0
            except Exception as exc:              # noqa: BLE001
                _log.debug("analytic stripping failed: %s", exc)
        out.append(rec)
    return out


# ---------------------------------------------------------------------------
def _availability_error(campaign):
    it = next((x for x in campaign.plan().get("availability", [])), None)
    st = campaign.item_status(it) if it else None
    return (st or {}).get("error")


def write_summary(campaign) -> Path:
    d = campaign.dir
    legs = campaign.legs_dir
    # stale figures of a leg that no longer runs would survive a rebuild
    if campaign.figures_dir.exists():
        for old in campaign.figures_dir.glob("*.png"):
            old.unlink()
    spec = campaign.spec
    rule = (spec.criticality or {}).get("rule") or {}
    faults = fault_table(campaign)
    _write_csv(legs / "faults" / "faults.csv", FAULT_COLS
               + tuple(k for k in (faults[0].keys() if faults else ()) if k.startswith("loss_w_per_m_peak_")),
               faults, "fault leg: one row per (case, bracket, mode); deltas and criticality vs the "
               "matching baseline")
    comps = compensation_table(campaign)
    _write_csv(legs / "faults" / "compensation.csv", COMP_COLS, comps,
               "compensation attempts per (case, strategy)")
    crit_map = [{"case_id": r["case_id"], "class": r["class"], "bracket": r["bracket"],
                 "elements": r["elements"], "criticality": r.get("criticality"),
                 "critical": r.get("critical"), "recovered_by": r.get("recovered_by")}
                for r in faults if r["case_id"] != "baseline"]
    _write_csv(legs / "faults" / "criticality_map.csv",
               ("case_id", "class", "bracket", "elements", "criticality", "critical", "recovered_by"), crit_map)
    unrec = [r["case_id"] for r in faults if r.get("recovered_by") == "unrecoverable"]
    (legs / "faults").mkdir(parents=True, exist_ok=True)
    (legs / "faults" / "unrecoverable.json").write_text(json.dumps(unrec, indent=1) + "\n", encoding="utf-8")
    seeds, fos = seed_tables(campaign)
    _write_csv(legs / "imperfections" / "seeds.csv", SEED_COLS, seeds, "one row per error seed (+ the control)")
    fos_rows = [{k: v for k, v in f.items() if not k.startswith("_")} for f in fos]
    _write_csv(legs / "imperfections" / "faults_on_seeds.csv", FOS_COLS, fos_rows,
               "top fault cases replayed on error seeds, before/after compensation; critical vs that seed")
    foil = foil_table(campaign)
    _write_csv(legs / "foil" / "foil.csv", FOIL_COLS, foil, "foil scenarios")
    avail = None
    ap = legs / "availability" / "availability.json"
    if ap.exists():
        try:
            avail = json.loads(ap.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            avail = None
    if avail and avail.get("variants"):
        rows = []
        for vname, s in avail["variants"].items():
            rows.append({"variant": vname, "availability_mean": s["availability_mean"],
                         "availability_p05": s["availability_p05"], "availability_p95": s["availability_p95"],
                         "availability_analytic": s["availability_analytic"],
                         "trips_per_year": sum(s["trips_per_year"].values()),
                         **{f"trips_{k}": v for k, v in s["trips_per_year"].items()}})
        _write_csv(legs / "availability" / "availability.csv", list(rows[0].keys()), rows)
        first = next(iter(avail["variants"].values()))
        bins = list(first["bins_s"]) + ["overflow"]
        hrows = [{"bin_upper_s": b, **{v: avail["variants"][v]["trip_histogram"][i] for v in avail["variants"]}}
                 for i, b in enumerate(bins)]
        _write_csv(legs / "availability" / "trip_histogram.csv", list(hrows[0].keys()), hrows)
        srows = [{"variant": v, "block": name, "dA_per_efold_mtbf": a, "dA_per_efold_mttr": b}
                 for v, s in avail["variants"].items() for name, a, b in s["sensitivity"]]
        if srows:
            _write_csv(legs / "availability" / "sensitivity.csv", list(srows[0].keys()), srows)

    # verdicts
    scen = [r for r in faults if r["case_id"] != "baseline" and not r.get("verification")]
    n_cases = sum(1 for r in scen if r["status"] == "ok")
    n_crit = sum(1 for r in scen if r.get("critical"))
    counts = {}
    for r in scen:
        c = counts.setdefault(r["class"], {"cases": 0, "critical": 0, "failed": 0})
        c["cases"] += 1
        c["critical"] += 1 if r.get("critical") else 0
        c["failed"] += 1 if r["status"] != "ok" else 0
    rec_by = {}
    for r in scen:
        if r.get("recovered_by"):
            rec_by[r["recovered_by"].split(":")[0]] = rec_by.get(r["recovered_by"].split(":")[0], 0) + 1
    robust = {}
    for f in fos:
        if f.get("critical") is None:
            continue
        key = (f["case_id"], f["variant"])
        r = robust.setdefault(key, [0, 0])
        r[0] += 0 if f["critical"] else 1
        r[1] += 1
    robust_rows = [{"case_id": k[0], "variant": k[1], "robust_fraction": v[0] / v[1], "n": v[1]}
                   for k, v in sorted(robust.items())]
    if robust_rows:
        _write_csv(legs / "imperfections" / "robustness.csv", ("case_id", "variant", "robust_fraction", "n"),
                   robust_rows)
    seed_ok = [s for s in seeds if s["kind"] == "seed" and s["status"] == "ok"]
    seed_T = [s["transmission"] for s in seed_ok if s.get("transmission") is not None]
    summary = {
        "name": spec.name, "preset": spec.preset, "input": str(campaign.input_path),
        "lattice_sha256": spec.lattice_sha256, "spec_sha256": spec.spec_sha256,
        "written": time.strftime("%Y-%m-%dT%H:%M:%S"), "status": campaign.status(),
        "rule": rule,
        "faults": {"n_cases": n_cases, "n_critical": n_crit, "by_class": counts,
                   "recovered_by": rec_by, "unrecoverable": unrec,
                   "top": [{"case_id": r["case_id"], "label": r["label"], "bracket": r["bracket"],
                            "criticality": r.get("criticality"), "critical": r.get("critical"),
                            "d_energy_mev": r.get("d_energy_mev")}
                           for r in sorted([x for x in scen if x.get("criticality") is not None],
                                           key=lambda x: -x["criticality"])[:10]],
                   "verification": [{"case_id": r["case_id"], "strategy": r["verification"],
                                     "critical": r.get("critical"), "d_energy_mev": r.get("d_energy_mev")}
                                    for r in faults if r.get("verification")]},
        "compensation": {"n": len(comps), "recovered": sum(1 for c in comps if c.get("recovered_by_rule")),
                         "energy_only": sum(1 for c in comps if c.get("recovered") and not c.get("recovered_by_rule")),
                         "failed": sum(1 for c in comps if c.get("error"))},
        "imperfections": {"n_seeds": len(seed_ok), "transmission_mean": (float(np.mean(seed_T)) if seed_T else None),
                          "transmission_min": (float(np.min(seed_T)) if seed_T else None),
                          "correction_converged": sum(1 for s in seed_ok if s.get("correction_status") == "converged"),
                          "faults_on_seeds": robust_rows},
        "foil": [{k: r.get(k) for k in ("case_id", "label", "foil_sigma_x_mm", "foil_sigma_y_mm", "strip_eff",
                                       "strip_eff_analytic", "h0_frac_analytic", "missed_frac", "foil_w_cm2_peak")}
                 for r in foil],
        "availability": ({v: {k: s[k] for k in ("availability_mean", "availability_p05", "availability_p95",
                                                "availability_analytic", "trips_per_year")}
                          for v, s in avail["variants"].items()} if avail and avail.get("variants") else None),
        "availability_source": (avail or {}).get("source"),
        "availability_error": _availability_error(campaign),
        "class_split_applied": (avail or {}).get("class_split_applied"),
    }
    (d / "summary.json").write_text(json.dumps(summary, indent=1, default=str) + "\n", encoding="utf-8")
    try:
        from linac_gen.reliability import plots
        figs = plots.write_figures(campaign, faults, comps, seeds, fos_rows, foil, avail)
    except Exception as exc:                       # noqa: BLE001
        _log.warning("figures failed: %s", exc)
        figs = []
    try:
        from linac_gen.reliability.report import write_report
        write_report(campaign, summary=summary, faults=faults, comps=comps, seeds=seeds,
                     fos=fos_rows, foil=foil, avail=avail, figures=figs)
    except Exception as exc:                       # noqa: BLE001
        _log.warning("report failed: %s", exc)
    return d / "summary.json"
