"""One self-contained HTML report (stdlib only, data-URI figures, an HTML
subset ``QTextDocument`` renders: attribute-styled tables, no CSS grid,
no scripts)."""
from __future__ import annotations

import base64
import json
import time
from html import escape
from pathlib import Path

__all__ = ["write_report"]


def _img(path) -> str:
    p = Path(path)
    if not p.exists():
        return ""
    data = base64.b64encode(p.read_bytes()).decode("ascii")
    return (f'<p><img src="data:image/png;base64,{data}" alt="{escape(p.stem)}" '
            f'width="640"></p>')


def _table(rows, cols, limit=25) -> str:
    if not rows:
        return "<p><i>no rows</i></p>"
    cols = [c for c in cols if any(r.get(c) not in (None, "", []) for r in rows)]
    h = ['<table border="1" cellpadding="3" cellspacing="0">', "<tr>"]
    h += [f"<th>{escape(str(c))}</th>" for c in cols]
    h.append("</tr>")
    for r in rows[:limit]:
        h.append("<tr>")
        for c in cols:
            v = r.get(c)
            if isinstance(v, float):
                v = f"{v:.5g}"
            elif isinstance(v, (list, tuple)):
                v = ", ".join(str(x) for x in v)
            elif isinstance(v, dict):
                v = json.dumps(v)
            h.append(f"<td>{escape('' if v is None else str(v))}</td>")
        h.append("</tr>")
    h.append("</table>")
    if len(rows) > limit:
        h.append(f"<p><i>{len(rows) - limit} more row(s) in the CSV.</i></p>")
    return "\n".join(h)


def _pct(v):
    return "n/a" if v is None else f"{100.0 * v:.3f} %"


def write_report(campaign, *, summary, faults, comps, seeds, fos, foil, avail, figures) -> Path:
    spec = campaign.spec
    figs = {Path(f).stem: f for f in figures}
    rule = summary.get("rule") or {}
    try:
        from linac_gen import __version__
    except Exception:                       # noqa: BLE001
        __version__ = "?"
    parts = [
        "<html><head><meta charset='utf-8'><title>Reliability Study — "
        f"{escape(spec.name)}</title></head><body style='font-family: sans-serif; color:#22223b'>",
        f"<h1>Reliability Study — {escape(spec.name)}</h1>",
        "<p><b>Provenance.</b> deck: <code>" + escape(str(campaign.input_path)) + "</code> "
        f"(sha256 {escape(str(spec.lattice_sha256)[:16])}) · spec {escape(str(spec.spec_sha256)[:16])} · "
        f"preset <b>{escape(spec.preset)}</b> · HELIX {escape(str(__version__))} · written "
        f"{escape(summary.get('written', ''))}</p>",
        "<p><b>Status.</b> " + escape(json.dumps(summary.get("status"))) + "</p>",
        "<p><b>Criticality rule.</b> normalised-emittance growth &gt; "
        f"{rule.get('emit_growth_pct', 5)} %, loss fraction &gt; {rule.get('loss_frac', 1e-4):g}, "
        f"exit-energy deviation &gt; {rule.get('energy_pct', 0.5)} %, or a section's peak loss density "
        f"above its W/m limit ({escape(json.dumps(rule.get('w_per_m')))}).</p>",
    ]
    # ---- leg A
    fa = summary.get("faults") or {}
    parts.append("<h2>Leg A — fault tolerance and compensation</h2>")
    parts.append(f"<p>{fa.get('n_cases', 0)} scenario runs, <b>{fa.get('n_critical', 0)} critical</b> by the rule. "
                 f"Recovered-by counts: {escape(json.dumps(fa.get('recovered_by')))}. "
                 f"Unrecoverable: {escape(', '.join(fa.get('unrecoverable') or []) or 'none')}.</p>")
    by = fa.get("by_class") or {}
    if by:
        parts.append(_table([{"class": k, **v} for k, v in by.items()], ["class", "cases", "critical", "failed"]))
    for key in ("criticality_by_case", "bracket_frozen_vs_rephased", "compensation_recovery"):
        if key in figs:
            parts.append(_img(figs[key]))
    parts.append("<h3>Top cases</h3>")
    parts.append(_table(fa.get("top") or [], ["case_id", "label", "bracket", "criticality", "critical", "d_energy_mev"]))
    parts.append("<h3>All cases (first 25; <code>legs/faults/faults.csv</code>)</h3>")
    parts.append(_table(faults, ["case_id", "class", "bracket", "mode", "status", "transmission", "ref_w_kin",
                                 "d_energy_mev", "d_phi_end_deg", "emit_nx_growth_pct", "emit_ny_growth_pct",
                                 "loss_w_per_m_peak", "criticality", "critical", "recovered_by"]))
    if fa.get("verification"):
        parts.append("<h3>Multiparticle verification of compensated cases</h3>")
        parts.append(_table(fa["verification"], ["case_id", "strategy", "critical", "d_energy_mev"]))
    parts.append("<h3>Compensation (<code>legs/faults/compensation.csv</code>)</h3>")
    parts.append("<p><i>recovered_by_rule</i>: the matcher met its energy / transmission objectives AND the "
                 "compensated beam is not critical by the study rule; <i>recovered</i> alone is the energy verdict, "
                 "trivially met by a magnet fault.</p>")
    parts.append(_table(comps, ["case_id", "strategy", "recovered_by_rule", "recovered", "after_critical",
                                "compensators", "ref_w_kin_after", "transmission_after", "emit_nx_after",
                                "emit_ny_after", "residual_cost", "error"]))
    # ---- leg C
    im = summary.get("imperfections") or {}
    parts.append("<h2>Leg C — imperfections and faults on error seeds</h2>")
    parts.append(f"<p>{im.get('n_seeds', 0)} seeds; transmission mean {im.get('transmission_mean')} %, "
                 f"min {im.get('transmission_min')} %; orbit correction converged on "
                 f"{im.get('correction_converged', 0)} seed(s).</p>")
    for key in ("seeds_transmission", "seeds_emittance", "faults_on_seeds_robustness"):
        if key in figs:
            parts.append(_img(figs[key]))
    parts.append(_table(seeds, ["id", "seed", "kind", "status", "transmission", "ref_w_kin", "emit_nx", "emit_ny",
                                "sigma_x", "sigma_y", "correction_status", "n_draws"]))
    if im.get("faults_on_seeds"):
        parts.append("<h3>Robustness of the top faults on error seeds</h3>")
        parts.append(_table(im["faults_on_seeds"], ["case_id", "variant", "robust_fraction", "n"]))
    # ---- leg D
    parts.append("<h2>Leg D — foil scenarios</h2>")
    for key in ("foil_spot", "foil_efficiency_vs_thickness"):
        if key in figs:
            parts.append(_img(figs[key]))
    parts.append(_table(foil, ["case_id", "label", "mode", "status", "foil_sigma_x_mm", "foil_sigma_y_mm",
                               "foil_w_cm2_peak", "strip_eff", "strip_eff_analytic", "h0_frac_analytic",
                               "missed_frac"]))
    parts.append("<p><i>Stripping fractions come from the two-step model calibrated to 99.956 % at 600 µg/cm² "
                 "and 99.1 % at 380 µg/cm² (800 MeV H⁻ on carbon) — calibration points, not measurements.</i></p>")
    # ---- leg B
    parts.append("<h2>Leg B — availability and beam-trip budget</h2>")
    av = summary.get("availability")
    if av:
        parts.append(f"<p>Block source: {escape(str(summary.get('availability_source')))}.  Recovered-by split "
                     f"applied to blocks: {escape(json.dumps(summary.get('class_split_applied') or {}))}</p>")
        parts.append(_table([{"variant": v, **{k: (_pct(s[k]) if k.startswith('availability') else s[k])
                                             for k in s}} for v, s in av.items()],
                            ["variant", "availability_mean", "availability_p05", "availability_p95",
                             "availability_analytic", "trips_per_year"]))
        for key in ("availability_hist", "trip_budget"):
            if key in figs:
                parts.append(_img(figs[key]))
    elif summary.get("availability_error"):
        parts.append(f"<p><b>FAILED:</b> {escape(str(summary['availability_error']))}</p>")
    else:
        parts.append("<p><i>not run</i></p>")
    # ---- appendix
    parts.append("<h2>Appendix — campaign spec</h2>")
    parts.append("<pre>" + escape(json.dumps({k: v for k, v in spec.__dict__.items()
                                              if k not in ("error_budget",)}, indent=1, default=str)) + "</pre>")
    parts.append(f"<p><i>Generated {escape(time.strftime('%Y-%m-%d %H:%M:%S'))}.</i></p></body></html>")
    out = campaign.dir / "report.html"
    out.write_text("\n".join(parts), encoding="utf-8")
    return out
