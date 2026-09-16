"""Analysis tools ported from the MIRAGE assistant (wave 1).

Registered via the same ``_tool`` decorator as the core registry —
importing this module (done at the bottom of ``tools.py``) makes the
tools appear on all three transports and the MCP server automatically.

* ``run_python``       (compute) — sandboxed analysis code with plots
* ``shift_data``       (read)    — session/ledger digest ("briefing")
* ``diagnose``         (read)    — ordered differential over the results
* ``anomaly_baseline`` (mutate)  — fit + store the reference fingerprint
* ``anomaly_check``    (read)    — score current results vs the baseline
"""
from __future__ import annotations

import glob as _glob
import json as _json
import os as _os
import time as _time

from linac_gen.assist.tools import (
    _capture, _ctx_provenance, _err, _local_path, _need, _ok, _refused,
    _save_capture, _tool,
)

#: results columns run_python may request via ``include``
_ARRAY_QUANTITIES = (
    "s", "sigma_x", "sigma_y", "sigma_phi", "sigma_w",
    "emit_x", "emit_y", "emit_z", "alpha_x", "beta_x", "alpha_y",
    "beta_y", "alpha_z", "beta_z", "transmission", "ref_w_kin",
)

_BASELINE_NAME = "assist_baseline.json"
_PROFILE_POINTS = 64
#: per-feature tolerance = max(rel · |baseline|, floor) — the floors are
#: physical noise scales (mm / % / MeV), not statistical guesses
_TOLERANCES = {
    "sigma_x": (0.05, 0.02), "sigma_y": (0.05, 0.02),
    "transmission": (0.002, 0.1), "ref_w_kin": (0.001, 0.01),
    "emit_x": (0.10, 0.005), "emit_y": (0.10, 0.005),
    "emit_z": (0.15, 0.005),
}


# ---------------------------------------------------------------------------
# run_python
# ---------------------------------------------------------------------------
@_tool("run_python",
       "Run short Python ANALYSIS code in an isolated sandbox "
       "(numpy/scipy/matplotlib; no linac_gen, no network, CPU/file "
       "limits).  Pass results columns via include= (loaded as "
       "arrays['name']); small JSON via data=.  Save a PNG (plot() "
       "helper) and you SEE the figure.  For fits, FFTs, custom plots "
       "the built-in tools don't cover.",
       {"type": "object",
        "properties": {
            "code": {"type": "string"},
            "data": {"type": "object"},
            "include": {"type": "array", "items": {"type": "string"}},
            "timeout": {"type": "number"}},
        "required": ["code"]},
       "compute")
def _run_python(ctx, code: str, data=None, include=None, timeout=30.0):
    from linac_gen.assist.sandbox import run_python_sandbox
    arrays = None
    warnings: list[str] = []
    if include:
        import numpy as np
        gate = _need(ctx, "results")
        if gate:
            return gate
        arrays = {}
        for name in include:
            if name not in _ARRAY_QUANTITIES:
                return _refused(
                    f"unknown results column {name!r}; available: "
                    + ", ".join(_ARRAY_QUANTITIES))
            col = getattr(ctx.results, name, None)
            if col is None or not len(col):
                return _err(f"results carry no '{name}'")
            arr = np.asarray(col, dtype=float)
            if arr.size > 200_000:
                step = arr.size // 200_000 + 1
                arr = arr[::step]
                warnings.append(f"{name} downsampled ×{step} "
                                f"(was {len(col)} points)")
            arrays[name] = arr
    out = run_python_sandbox(code, data=data, arrays=arrays,
                             timeout=timeout)
    if out.get("error"):
        return _err(out["error"], warnings=warnings)
    if out.get("img_b64"):
        out = _save_capture(ctx, out, "python")
    return _ok(out, _ctx_provenance(ctx), warnings=warnings)


# ---------------------------------------------------------------------------
# ledger helpers (shift_data, diagnose)
# ---------------------------------------------------------------------------
def _ledger_files(ctx, newest: int = 4) -> list[str]:
    d = _os.path.join(getattr(ctx, "calc_dir", ".") or ".",
                      "assist_sessions")
    files = sorted(_glob.glob(_os.path.join(d, "*.jsonl")),
                   key=_os.path.getmtime, reverse=True)
    return files[:newest]


def _ledger_records(paths) -> list[dict]:
    out: list[dict] = []
    for p in paths:
        try:
            with open(p, encoding="utf-8", errors="replace") as f:
                for line in f:
                    try:
                        out.append(_json.loads(line))
                    except Exception:                       # noqa: BLE001
                        continue
        except OSError:
            continue
    return out


def _parse_ts(rec) -> float:
    try:
        import datetime as _dt
        return _dt.datetime.fromisoformat(rec["ts"]).timestamp()
    except Exception:                                       # noqa: BLE001
        return 0.0


# ---------------------------------------------------------------------------
# shift_data
# ---------------------------------------------------------------------------
@_tool("shift_data",
       "Session briefing: what happened over the last N hours — tool "
       "usage and mutate/denied counts from the session ledgers, result "
       "files written in the calc dir, and the current exit KPIs if "
       "results are loaded.  Use for 'what did we do' / handoff "
       "questions.",
       {"type": "object",
        "properties": {"hours": {"type": "number"}},
        "required": []},
       "read")
def _shift_data(ctx, hours: float = 8.0):
    horizon = _time.time() - float(hours) * 3600.0
    recs = [r for r in _ledger_records(_ledger_files(ctx))
            if _parse_ts(r) >= horizon]
    kinds: dict[str, int] = {}
    tools_used: dict[str, int] = {}
    mutates = denied = 0
    recent: list[dict] = []
    for r in recs:
        kinds[r.get("event", "?")] = kinds.get(r.get("event", "?"), 0) + 1
        if r.get("event") == "tool":
            tools_used[r.get("tool", "?")] = (
                tools_used.get(r.get("tool", "?"), 0) + 1)
            if r.get("tier") == "mutate":
                mutates += 1
            if r.get("approved_by") == "denied":
                denied += 1
            recent.append({"ts": r.get("ts"), "tool": r.get("tool"),
                           "tier": r.get("tier"),
                           "status": r.get("status")})
    calc = getattr(ctx, "calc_dir", ".") or "."
    runs = []
    for p in _glob.glob(_os.path.join(calc, "**", "*.h5"),
                        recursive=True):
        try:
            m = _os.path.getmtime(p)
        except OSError:
            continue
        if m >= horizon:
            runs.append({"file": _os.path.relpath(p, calc),
                         "age_min": round((_time.time() - m) / 60, 1),
                         "size_kb": round(_os.path.getsize(p) / 1024, 1)})
    runs.sort(key=lambda r: r["age_min"])
    data = {"hours": float(hours),
            "ledger_events": kinds, "tool_usage": tools_used,
            "mutate_calls": mutates, "denied_calls": denied,
            "recent_tool_calls": recent[-40:],
            "result_files_written": runs[:30]}
    if not recs:
        data["note"] = ("no session ledgers in the window — nothing "
                        "recorded, or a different calc dir")
    if getattr(ctx, "results", None) is not None:
        from linac_gen.assist.tools import TOOLS
        summ = TOOLS["result_summary"].fn(ctx)
        if summ.get("status") == "ok":
            data["current_exit_kpis"] = summ["data"]
    return _ok(data, _ctx_provenance(ctx))


# ---------------------------------------------------------------------------
# diagnose
# ---------------------------------------------------------------------------
def _element_at_s(ctx, s_mm: float):
    """(index, name, class) of the element containing s_mm, else None."""
    lat = getattr(ctx, "lattice", None)
    if lat is None:
        return None
    run = 0.0
    for i, el in enumerate(lat.elements):
        run += float(getattr(el, "length", 0.0) or 0.0)
        if run >= s_mm:
            return {"index": i,
                    "name": str(getattr(el, "name", "?")),
                    "class": type(el).__name__}
    return None


def first_drop_onset(s_mm, tr, drop_pts: float = 0.5):
    """First s where transmission fell ``drop_pts`` below its running
    maximum (None if it never does).  Shared with RunWatch."""
    import numpy as np
    tr = np.asarray(tr, dtype=float)
    s_mm = np.asarray(s_mm, dtype=float)
    if tr.size == 0 or s_mm.size != tr.size:
        return None
    peak = np.maximum.accumulate(tr)
    idx = np.where(tr <= peak - drop_pts)[0]
    return float(s_mm[idx[0]]) if idx.size else None


@_tool("diagnose",
       "One-shot differential diagnosis of the loaded results: "
       "transmission-loss onset (mapped to the element), sigma blow-up "
       "onset, emittance growth, energy deviation, recent parameter "
       "changes from the session ledger — ordered by likelihood, with "
       "a suggested next tool.  Start here for 'why does this look "
       "wrong'.",
       {"type": "object", "properties": {}, "required": []},
       "read")
def _diagnose(ctx):
    import numpy as np
    if getattr(ctx, "results", None) is None:
        return _ok({"verdict": "no results loaded — run a simulation "
                               "first",
                    "suggested_tool": "run_envelope"})
    res = ctx.results
    findings: list[dict] = []
    s = np.asarray(getattr(res, "s", []), dtype=float)

    # 1. transmission-drop onset (skip the very first samples: an
    #    inherited aperture scrape at injection is normal MEBT physics)
    tr = getattr(res, "transmission", None)
    if tr is not None and len(tr) and s.size == len(tr):
        onset = first_drop_onset(s, tr)
        if onset is not None:
            el = _element_at_s(ctx, onset)
            findings.append({
                "kind": "transmission_drop",
                "s_m": round(onset / 1000.0, 3),
                "element": el,
                "total_loss_pts": round(float(np.max(tr) - tr[-1]), 3),
                "note": "loss starts here — the CAUSE is at or upstream "
                        "of this element"})

    # 2. sigma blow-up onset (vs the early-lattice scale)
    for plane in ("sigma_x", "sigma_y"):
        sig = getattr(res, plane, None)
        if sig is None or not len(sig) or s.size != len(sig):
            continue
        sig = np.asarray(sig, dtype=float)
        ref = float(np.median(sig[:max(5, sig.size // 20)]))
        if ref > 0:
            idx = np.where(sig > 5.0 * ref)[0]
            if idx.size:
                el = _element_at_s(ctx, float(s[idx[0]]))
                findings.append({
                    "kind": f"{plane}_blowup",
                    "s_m": round(float(s[idx[0]]) / 1000.0, 3),
                    "element": el,
                    "value_mm": round(float(sig[idx[0]]), 3),
                    "reference_mm": round(ref, 3)})

    # 3. emittance growth end/start
    for em in ("emit_x", "emit_y", "emit_z"):
        e = getattr(res, em, None)
        if e is None or len(e) < 2:
            continue
        e0, e1 = float(e[0]), float(e[-1])
        if e0 > 0 and e1 / e0 > 1.5:
            findings.append({"kind": f"{em}_growth",
                             "ratio": round(e1 / e0, 2),
                             "note": "≥1.5× growth start→end"})

    # 4. energy deviation
    w = getattr(res, "ref_w_kin", None)
    if w is not None and len(w) >= 2:
        findings.append({"kind": "energy",
                         "w_in_mev": round(float(w[0]), 4),
                         "w_out_mev": round(float(w[-1]), 4)})

    # 5. recent mutations from the ledgers (last 30 min)
    horizon = _time.time() - 1800.0
    changes = [
        {"ts": r.get("ts"), "tool": r.get("tool"),
         "params": r.get("params"), "status": r.get("status")}
        for r in _ledger_records(_ledger_files(ctx, newest=2))
        if (r.get("event") == "tool" and r.get("tier") == "mutate"
            and r.get("approved_by") not in ("denied",)
            and _parse_ts(r) >= horizon)]
    if changes:
        findings.append({"kind": "recent_mutations",
                         "calls": changes[-10:],
                         "note": "check these first — they changed the "
                                 "machine within the last 30 min"})

    # verdict = the first (most upstream-causal) finding
    interesting = [f for f in findings
                   if f["kind"] not in ("energy",)]
    if not interesting:
        verdict = "nothing anomalous found in the loaded results"
        suggested = "anomaly_check"
    else:
        top = interesting[0]
        verdict = top["kind"].replace("_", " ")
        if top.get("element"):
            verdict += (f" near s = {top['s_m']} m "
                        f"({top['element']['name']})")
        suggested = {"transmission_drop": "look_at_plot",
                     "recent_mutations": "shift_data"}.get(
            top["kind"], "parameter_scan")
    return _ok({"verdict": verdict, "findings": findings,
                "suggested_tool": suggested}, _ctx_provenance(ctx))


# ---------------------------------------------------------------------------
# anomaly baseline / check
# ---------------------------------------------------------------------------
def _baseline_path(ctx) -> str:
    return _os.path.join(getattr(ctx, "calc_dir", ".") or ".",
                         _BASELINE_NAME)


def _identity(ctx) -> dict:
    bc = getattr(ctx, "beam_config", None)
    lat = getattr(ctx, "lattice", None)
    return {"lattice_path": str(getattr(ctx, "lattice_path", "") or ""),
            "n_elements": (len(lat.elements) if lat is not None else 0),
            "species": str(getattr(bc, "species", "?")),
            "energy_mev": float(getattr(bc, "energy", 0.0) or 0.0)}


def _fingerprint(res) -> dict:
    import numpy as np
    s = np.asarray(res.s, dtype=float)
    grid = np.linspace(s[0], s[-1], _PROFILE_POINTS)
    fp: dict = {"s_grid_mm": [round(float(x), 1) for x in grid]}
    for key in ("sigma_x", "sigma_y", "transmission", "ref_w_kin"):
        col = getattr(res, key, None)
        if col is not None and len(col) == s.size:
            prof = np.interp(grid, s, np.asarray(col, dtype=float))
            fp[key] = [float(v) for v in prof]
    for key in _TOLERANCES:
        col = getattr(res, key, None)
        if col is not None and len(col):
            fp[f"exit_{key}"] = float(np.asarray(col)[-1])
    return fp


@_tool("anomaly_baseline",
       "Record the CURRENT loaded results as the healthy reference "
       "fingerprint (exit KPIs + sigma/transmission/energy profiles) "
       "for anomaly_check.  Writes <calc_dir>/assist_baseline.json.",
       {"type": "object", "properties": {}, "required": []},
       "mutate")
def _anomaly_baseline(ctx):
    gate = _need(ctx, "results")
    if gate:
        return gate
    if not len(getattr(ctx.results, "s", [])):
        return _err("results carry no s axis")
    payload = {"identity": _identity(ctx),
               "fingerprint": _fingerprint(ctx.results),
               "created": _time.strftime("%Y-%m-%d %H:%M:%S")}
    path = _baseline_path(ctx)
    with open(path, "w", encoding="utf-8") as f:
        _json.dump(payload, f, indent=1)
    return _ok({"written": path,
                "features": sorted(payload["fingerprint"].keys())},
               _ctx_provenance(ctx))


@_tool("anomaly_check",
       "Score the loaded results against the stored healthy baseline "
       "(anomaly_baseline): per-feature z-scores, top offenders, "
       "verdict.  Refuses honestly when the baseline belongs to a "
       "different lattice.",
       {"type": "object", "properties": {}, "required": []},
       "read")
def _anomaly_check(ctx):
    import numpy as np
    gate = _need(ctx, "results")
    if gate:
        return gate
    path = _baseline_path(ctx)
    if not _os.path.isfile(path):
        return _refused("no baseline recorded — run anomaly_baseline on "
                        "a known-good result first")
    with open(path, encoding="utf-8") as f:
        base = _json.load(f)
    ident, now = base.get("identity", {}), _identity(ctx)
    if (ident.get("lattice_path") != now["lattice_path"]
            or ident.get("n_elements") != now["n_elements"]):
        return _refused(
            "the stored baseline was made for a DIFFERENT lattice "
            f"({ident.get('lattice_path', '?')}, "
            f"{ident.get('n_elements')} elements) — re-run "
            "anomaly_baseline; refusing to score against it")
    fp_base = base.get("fingerprint", {})
    fp_now = _fingerprint(ctx.results)
    scores: list[dict] = []
    for key, (rel, floor) in _TOLERANCES.items():
        b, c = fp_base.get(f"exit_{key}"), fp_now.get(f"exit_{key}")
        if b is None or c is None:
            continue
        tol = max(rel * abs(b), floor)
        scores.append({"feature": f"exit_{key}",
                       "baseline": round(float(b), 5),
                       "current": round(float(c), 5),
                       "z": round(abs(float(c) - float(b)) / tol, 2)})
    for key in ("sigma_x", "sigma_y", "transmission", "ref_w_kin"):
        b, c = fp_base.get(key), fp_now.get(key)
        if not b or not c or len(b) != len(c):
            continue
        rel, floor = _TOLERANCES.get(key, (0.05, 0.02))
        b_arr, c_arr = np.asarray(b), np.asarray(c)
        tol = np.maximum(rel * np.abs(b_arr), floor)
        z = np.max(np.abs(c_arr - b_arr) / tol)
        i = int(np.argmax(np.abs(c_arr - b_arr) / tol))
        scores.append({"feature": f"profile_{key}",
                       "z": round(float(z), 2),
                       "worst_at_s_m": round(
                           fp_base["s_grid_mm"][i] / 1000.0, 2)})
    if not scores:
        return _err("baseline and current results share no features")
    scores.sort(key=lambda d: -d["z"])
    overall = scores[0]["z"]
    try:
        age_s = _time.time() - _os.path.getmtime(path)
    except OSError:
        age_s = -1.0
    return _ok({"verdict": ("anomalous" if overall > 1.0 else "ok"),
                "overall_z": overall,
                "top_offenders": scores[:5],
                "baseline_created": base.get("created"),
                "baseline_age_s": round(age_s, 0)},
               _ctx_provenance(ctx))


# ---------------------------------------------------------------------------
# hofmann_stability
# ---------------------------------------------------------------------------
@_tool("hofmann_stability",
       "Corrected anisotropic Hofmann (PRE 57, 4713) coherent-instability "
       "screen per lattice-period cell: per-branch growth rates "
       "gamma/nu_0x for l=2, 3(even/odd), 4(even) from the depressed-tune "
       "chart coordinates (R = nu_z/nu_x, Y = eta_x, geometric "
       "eps_z/eps_x), the S^2<=10 perturbative-validity gate, instability "
       "flags, optional anisotropy margins to higher-order onset, and "
       "optional Monte-Carlo instability probabilities under the "
       "engineering jitter budget.  Reports an in-band data.reason on x-y "
       "coupled or DC lattices.  Runs its own probe-bearing envelope when "
       "the session results lack phase-probe maps.  Long-running: "
       "executes as a background job.",
       {"type": "object",
        "properties": {
            "period_index": {"type": "integer", "minimum": 0,
                             "description": "index into detected periods "
                             "(default: non-fallback period with >= 2 "
                             "repeats and the most repeats)"},
            "margin": {"type": "boolean", "default": False},
            "probability": {"type": "boolean", "default": False},
            "n_mc": {"type": "integer", "default": 200},
            "threshold": {"type": "number", "default": 0.01}},
        "required": []},
       "compute")
def _hofmann_stability(ctx, period_index=None, margin=False,
                       probability=False, n_mc=200, threshold=0.01,
                       progress_callback=None, should_abort=None,
                       _assist_prov=None):
    gate = _need(ctx, "lattice", "beam_config")
    if gate:
        return gate
    import numpy as np

    from linac_gen.analysis import hofmann_stability as _hs
    from linac_gen.analysis.period_detect import detect_periods

    try:
        n_mc = max(10, min(int(n_mc), 1000))
        threshold = float(threshold)
    except (TypeError, ValueError):
        return _refused(f"n_mc/threshold must be numeric, got "
                        f"{n_mc!r} / {threshold!r}")
    if not np.isfinite(threshold) or threshold <= 0.0:
        return _refused(f"threshold must be a positive finite growth rate "
                        f"(gamma/nu_0x), got {threshold!r}")
    warns: list[str] = []

    results = ctx.results
    probe_src = "session results (phase-probe maps present)"
    maps = getattr(results, "element_maps_dep", None) if results else None
    if maps and len(maps) != len(ctx.lattice.elements):
        # Probe maps are one-per-element: a length mismatch means the
        # results came from a different lattice — run a fresh probe
        # rather than silently combining the two.
        results = None
    if results is None or not getattr(results, "element_maps_dep", None):
        from linac_gen.analysis.phase_advance import run_phase_probe
        from linac_gen.cli.common import _envelope_initial, build_ref
        ref = build_ref(ctx.beam_config)
        results, refusal, w0 = _capture(
            run_phase_probe, ctx.lattice, ref,
            _envelope_initial(ctx.beam_config, ref),
            current=getattr(ctx.beam_config, "current", 0.0),
            progress_callback=progress_callback,
            should_abort=should_abort)
        if refusal:
            return refusal
        warns += w0
        probe_src = "fresh envelope phase probe"

    periods = detect_periods(ctx.lattice)
    if not periods:
        return _refused("no periodic structure detected in the session "
                        "lattice")
    if period_index is not None:
        if not 0 <= int(period_index) < len(periods):
            return _refused(
                f"period_index out of range 0..{len(periods) - 1}: "
                + "; ".join(f"{i}: {p.label}"
                            for i, p in enumerate(periods)))
        period = periods[int(period_index)]
    else:
        cands = [p for p in periods
                 if p.source != "fallback" and p.n_repeats >= 2]
        period = (max(cands, key=lambda p: p.n_repeats) if cands
                  else periods[0])

    tab, refusal, w1 = _capture(
        _hs.hofmann_stability, results, period, threshold=float(threshold),
        should_stop=should_abort)
    if refusal:
        return refusal
    warns += w1

    margin_out = None
    if margin and tab["reason"] is None:
        margin_out, refusal, w2 = _capture(
            _hs.anisotropy_margin, None, None, coords=tab,
            gamma_th=float(threshold))
        if refusal:
            return refusal
        warns += w2

    prob = None
    if probability and tab["reason"] is None:
        from linac_gen.analysis.hofmann_probabilistic import (
            instability_probability,
        )
        prob, refusal, w3 = _capture(
            instability_probability, tab, N_mc=n_mc,
            threshold=float(threshold),
            should_stop=should_abort)
        if refusal:
            return refusal
        warns += w3

    def _f(x):
        x = float(x)
        return x if np.isfinite(x) else None

    rows = []
    for k in range(int(tab["n_cells"])):
        row = {
            "cell": int(tab["cells"][k]),
            "R": _f(tab["R"][k]), "Y": _f(tab["Y"][k]),
            "eps_ratio": _f(tab["eps_ratio"][k]),
            "S2": _f(tab["S2"][k]),
            "g_l2": _f(tab["g_l2"][k]),
            "g_l3_even": _f(tab["g_l3_even"][k]),
            "g_l3_odd": _f(tab["g_l3_odd"][k]),
            "g_l4_even": _f(tab["g_l4_even"][k]),
            "g_combined": _f(tab["g_combined"][k]),
            "valid": bool(tab["valid"][k]),
            "flagged": bool(tab["flagged"][k]),
            "flagged_extrap": bool(tab["flagged_extrap"][k]),
            "fold_risk": bool(tab["fold_risk"][k]),
        }
        if margin_out is not None:
            row["onset_eps"] = _f(margin_out["onset_eps"][k])
            row["margin"] = _f(margin_out["margin"][k])
            row["is_seam"] = bool(margin_out["is_seam"][k])
        if prob is not None:
            row["p_unstable"] = _f(prob[k])
        rows.append(row)

    data = {
        "period": period.label,
        "probe": probe_src,
        "reason": tab["reason"],
        "n_cells": int(tab["n_cells"]),
        "n_valid": int(tab["n_valid"]),
        "n_flagged": int(tab["n_flagged"]),
        "worst_cell": _f(tab["worst_cell"]),
        "worst_growth": _f(tab["worst_growth"]),
        "threshold": _f(tab["threshold"]),
        "s2_gate": _f(tab["s2_gate"]),
        "solver_fingerprint": tab["solver_fingerprint"],
        "cells": rows,
    }
    if margin_out is not None:
        data["margin_summary"] = {
            "n_onsets": int(margin_out["n_onsets"]),
            "n_smooth": int(margin_out["n_smooth"]),
            "n_seam": int(margin_out["n_seam"]),
            "smallest_smooth_margin":
                _f(margin_out["smallest_smooth_margin"]),
            "smallest_smooth_margin_cell":
                _f(margin_out["smallest_smooth_margin_cell"]),
            "earliest_smooth_onset_eps":
                _f(margin_out["earliest_smooth_onset_eps"]),
        }
    return _ok(data, _ctx_provenance(ctx), warns)


# ---------------------------------------------------------------------------
# lebt_scc
# ---------------------------------------------------------------------------
@_tool("lebt_scc",
       "LEBT space-charge-compensation analysis of a DC/continuous run: "
       "residual-gas neutralisation (tau_scc, self-consistent "
       "Poisson-Boltzmann steady state or assumed eta, exact f_c "
       "build-up), on-axis beam potential, gas stripping/capture loss, "
       "and suggested SPACE_CHARGE_COMP card factors.  Reports an "
       "in-band data.reason on bunched sessions.  Runs its own DC "
       "envelope when the session lacks continuous-beam results.  "
       "Long-running: executes as a background job.",
       {"type": "object",
        "properties": {
            "gas": {"type": "string", "default": "H2",
                    "enum": ["H2", "He", "N2", "Ar", "Kr", "Xe"]},
            "pressure_mbar": {"type": "number", "default": 8.0e-6},
            "mode": {"type": "string", "default": "computed",
                     "enum": ["computed", "assumed"]},
            "eta_assumed": {"type": "number", "default": 0.92},
            "trapped_temp_eV": {"type": "number", "default": 3.0},
            "taper": {"type": "boolean", "default": True},
            "build_up_us": {"type": "number",
                            "description": "f_c build-up time; omit for "
                            "steady state"},
            "n_cards": {"type": "integer", "default": 8, "minimum": 1},
            "cleared_regions": {
                "type": "array",
                "items": {"type": "array",
                          "items": {"type": "integer"},
                          "minItems": 2, "maxItems": 2},
                "description": "element ranges [start, end] (inclusive) "
                "where compensating ions are actively cleared (e.g. a "
                "biased chopper — the PXIE un-neutralised section); f_c "
                "is forced toward cleared_residual there"},
            "cleared_residual": {"type": "number", "default": 0.0,
                                 "minimum": 0.0, "maximum": 1.0},
            "self_consistent": {
                "type": "boolean", "default": False,
                "description": "iterate envelope <-> cards on a working "
                "copy until the factors stop moving (the one-shot "
                "analysis uses beam sizes tracked at the run's own "
                "space-charge state — a first iteration, not the fixed "
                "point); the loaded lattice is not modified"}},
        "required": []},
       "compute")
def _lebt_scc(ctx, gas="H2", pressure_mbar=8.0e-6, mode="computed",
              eta_assumed=0.92, trapped_temp_eV=3.0, taper=True,
              build_up_us=None, n_cards=8, cleared_regions=None,
              cleared_residual=0.0, self_consistent=False,
              progress_callback=None, should_abort=None, _assist_prov=None):
    gate = _need(ctx, "lattice", "beam_config")
    if gate:
        return gate
    import numpy as np

    if cleared_regions is not None:
        n_el = len(ctx.lattice.elements)
        try:
            pairs = [[float(r[0]), float(r[1])] for r in cleared_regions]
            cleared_residual = float(cleared_residual)
        except (TypeError, ValueError, IndexError, KeyError):
            return _refused("cleared_regions must be [start, end] element "
                            "pairs and cleared_residual a number")
        if any(v != int(v) for pr in pairs for v in pr):
            return _refused("cleared region bounds must be whole element "
                            "indices (schema: integers)")
        cleared_regions = [[int(a), int(b)] for a, b in pairs]
        for e0, e1 in cleared_regions:
            if not (0 <= e0 <= e1 < n_el):
                return _refused(f"cleared region [{e0}, {e1}] outside the "
                                f"lattice (0..{n_el - 1})")
        if not (0.0 <= cleared_residual <= 1.0):
            return _refused(f"cleared_residual must be in [0, 1], "
                            f"got {cleared_residual!r}")

    from linac_gen.analysis.scc.driver import (
        scc_analysis, suggested_scc_deck_lines)

    try:
        pressure_mbar = float(pressure_mbar)
        eta_assumed = float(eta_assumed)
        trapped_temp_eV = float(trapped_temp_eV)
        n_cards = max(1, min(int(n_cards), 64))
        if build_up_us is not None:
            build_up_us = float(build_up_us)
            if not np.isfinite(build_up_us) or build_up_us < 0:
                return _refused(f"build_up_us must be a non-negative "
                                f"finite time, got {build_up_us!r}")
    except (TypeError, ValueError):
        return _refused("numeric parameters must be numbers")
    if not (0.0 < pressure_mbar <= 1.0):
        return _refused(f"pressure_mbar out of range (0, 1]: {pressure_mbar!r}")
    if not (0.0 <= eta_assumed <= 1.0):
        return _refused(f"eta_assumed must be in [0, 1]: {eta_assumed!r}")

    from linac_gen.analysis.scc.driver import is_continuous_results

    warns: list[str] = []
    results = ctx.results
    probe_src = "session results"
    if not self_consistent and (results is None
                                or not is_continuous_results(results)):
        cfg = ctx.beam_config
        # A fresh envelope is only useful if the CONFIG is DC — refuse
        # in-band for bunched configs whether or not results exist.
        if not bool(getattr(cfg, "continuous", False)):
            return _ok({"reason": (
                "bunched-beam session: the SCC gas-neutralisation "
                "analysis applies to DC/continuous (LEBT) transport — "
                "for bunched beams use hofmann_stability instead.")},
                _ctx_provenance(ctx))
        from linac_gen.cli.common import _envelope_initial, build_ref
        from linac_gen.tracking.envelope import EnvelopeSolver
        ref = build_ref(cfg)
        results, refusal, w0 = _capture(
            lambda: EnvelopeSolver(
                ctx.lattice, ref, _envelope_initial(cfg, ref),
                current=getattr(cfg, "current", 0.0),
                progress_callback=progress_callback,
                should_abort=should_abort).run())
        if refusal:
            return refusal
        warns += w0
        probe_src = "fresh DC envelope run"

    species = str(getattr(ctx.beam_config, "species", "H-") or "H-")
    if self_consistent:
        # The iterator runs its own envelopes on a deepcopy — session
        # results are irrelevant, and current comes from the config.
        if not bool(getattr(ctx.beam_config, "continuous", False)):
            return _ok({"reason": (
                "bunched-beam session: the SCC gas-neutralisation "
                "analysis applies to DC/continuous (LEBT) transport — "
                "for bunched beams use hofmann_stability instead.")},
                _ctx_provenance(ctx))
        from linac_gen.analysis.scc.iterate import scc_self_consistent
        prog = (None if progress_callback is None else
                (lambda k, _d: progress_callback(min(0.95, k / 8.0))))
        a, refusal, w1 = _capture(
            scc_self_consistent, ctx.lattice, ctx.beam_config,
            should_stop=should_abort, progress=prog, species=species,
            gas=gas, pressure_mbar=pressure_mbar, mode=mode,
            eta_assumed=eta_assumed, trapped_temp_eV=trapped_temp_eV,
            taper=bool(taper), build_up_us=build_up_us, n_cards=n_cards,
            cleared_regions=cleared_regions,
            cleared_residual=cleared_residual)
        probe_src = "self-consistent iterate"
    else:
        a, refusal, w1 = _capture(
            scc_analysis, results, ctx.lattice, species=species, gas=gas,
            current_mA=getattr(ctx.beam_config, "current", None),
            pressure_mbar=pressure_mbar, mode=mode, eta_assumed=eta_assumed,
            trapped_temp_eV=trapped_temp_eV, taper=bool(taper),
            build_up_us=build_up_us, n_cards=n_cards,
            cleared_regions=cleared_regions,
            cleared_residual=cleared_residual,
            should_stop=should_abort)
    if refusal:
        return refusal
    warns += w1

    def _f(x):
        x = float(x)
        return x if np.isfinite(x) else None

    data = {"reason": a["reason"], "probe": probe_src}
    if a["reason"] is None:
        data.update({
            "fc_source": a["fc_source"],
            "species": a["species"],
            "gas_mix_mbar": a["gas_mix_mbar"],
            "tau_scc_global_us": _f(a["tau_scc_global_us"]),
            "mean_fc": _f(a["mean_fc"]),
            "phi_min_V": _f(a["phi_min_V"]),
            "phi_max_V": _f(a["phi_max_V"]),
            "strip_mfp_m": _f(a["strip_mfp_m"]),
            "mean_ion_mass_amu": _f(a["mean_ion_mass_amu"]),
            "transmission_gas_pct": _f(a["transmission_gas_pct"]),
            "build_up_us": (None if a["build_up_us"] is None
                            else _f(a["build_up_us"])),
            "notes": a["notes"],
            "n_slices_converged": int(a["slice_converged"].sum()),
            "n_slices": int(a["slice_converged"].size),
            "profiles": {
                "z_m": [_f(v) for v in a["z_m"]],
                "fc": [_f(v) for v in a["fc"]],
                "eta_ss": [_f(v) for v in a["eta_ss"]],
                "phi_V": [_f(v) for v in a["phi_V"]],
                "survival_gas": [_f(v) for v in a["survival_gas"]],
            },
            "scc_cards": [{"z_m": _f(c["z_m"]), "factor": _f(c["factor"])}
                          for c in a["scc_cards"]],
            "deck_lines": suggested_scc_deck_lines(a),
        })
        if a.get("cleared_regions"):
            data["cleared_regions"] = a["cleared_regions"]
            data["cleared_residual"] = _f(a["cleared_residual"])
        it = a.get("iterate")
        if it:
            data["iterate"] = {
                "converged": bool(it["converged"]),
                "n_iter": int(it["n_iter"]),
                "omega": _f(it["omega"]), "tol": _f(it["tol"]),
                "history": [{"max_delta": _f(h["max_delta"]),
                             "mean_fc": _f(h["mean_fc"])}
                            for h in it["history"]],
            }
    return _ok(data, _ctx_provenance(ctx), warns)


# ---------------------------------------------------------------------------
# orm_calibrate / orm_apply — orbit-response comparison, LOCO-style calibration
# ---------------------------------------------------------------------------
_ORM_SCHEMA = {
    "type": "object",
    "properties": {
        "measured": {"type": "array", "items": {"type": "string"}, "minItems": 1,
                     "description": "FORMA scan folders (one driven plane each; give both) "
                                    "and/or HELIX ORM CSV files"},
        "mode": {"type": "string", "enum": ["compare", "fit", "validate"],
                 "default": "compare",
                 "description": "compare = map + model + metrics; fit = LOCO-style "
                                "calibration (kept on the session for orm_apply); "
                                "validate = synthetic closed loop"},
        "mapping_file": {"type": "string",
                         "description": "device-map JSON (default: built-in naming rules)"},
        "stage": {"type": "string",
                  "enum": ["trims", "trims+quads", "trims+quads+bpms"],
                  "default": "trims+quads"},
        "prior_g": {"type": "number", "default": 0.10,
                    "description": "Gaussian prior width on the quad scale factors"},
        "prior_G": {"type": "number", "default": 0.05,
                    "description": "prior width on the BPM gains"},
        "sys_floor": {"type": "number", "default": 0.05,
                      "description": "systematic floor, fraction of each column's maximum"},
        "max_iter": {"type": "integer", "default": 12},
        "fix_quads": {"type": "array", "items": {"type": "string"}, "default": [],
                      "description": "quad labels held at scale 1"},
        "exclude_bpms": {"type": "array", "items": {"type": "string"}, "default": []},
        "exclude_trims": {"type": "array", "items": {"type": "string"}, "default": []},
        "check_tracking": {"type": "boolean", "default": False,
                           "description": "compare: also build the kick-and-read model"},
        "calibration_out": {"type": "string",
                            "description": "fit: write the calibration JSON here"},
        "export_deck": {"type": "string",
                        "description": "fit: write the recalibrated deck here (text "
                                       "surgery on the session's deck file)"},
        "seed": {"type": "integer", "default": 7},
        "g_sigma": {"type": "number", "default": 0.03,
                    "description": "validate: rms of the injected quad errors"},
        "G_sigma": {"type": "number", "default": 0.03,
                    "description": "validate: rms of the injected BPM-gain errors"},
    },
    "required": ["measured"],
}


def _orm_f(x):
    try:
        x = float(x)
    except (TypeError, ValueError):
        return None
    return x if x == x and abs(x) != float("inf") else None


def _orm_selection_json(sel) -> dict:
    return {"n_bpm": int(sel.n_bpm), "n_trim": int(sel.n_trim),
            "bpms": list(sel.bpm_labels), "trims": list(sel.trim_labels),
            "unmatched_devices": {k: list(v) for k, v in sel.unmatched_devices.items()},
            "lattice_devices_without_data": {k: list(v) for k, v in sel.unmatched_elements.items()},
            "dead_devices": list(sel.dead_devices), "notes": list(sel.notes)}


def _orm_compare_json(cmp) -> dict:
    planes = {}
    for p, d in cmp["planes"].items():
        planes[p] = {
            "n_entries": int(d["n_entries"]),
            "k_global_Tm_per_A": _orm_f(d["k_global_Tm_per_A"]),
            "r_global": _orm_f(d["r_global"]),
            "nrms_global": _orm_f(d["nrms_global"]),
            "tank_noise_mm_per_A": _orm_f(d["tank_noise_mm_per_A"]),
            "polarity_flipped": list(d["polarity_flipped"]),
            "per_trim": [{"trim": r["trim"], "device": r["device"],
                          "k_Tm_per_A": _orm_f(r["k_Tm_per_A"]), "dk": _orm_f(r["dk"]),
                          "kick_mrad_per_A": _orm_f(r["kick_mrad_per_A"]),
                          "r": _orm_f(r["r"]), "nrms_resid": _orm_f(r["nrms_resid"]),
                          "n_down": int(r["n_down"]), "included": bool(r["included"])}
                         for r in d["per_trim"]],
        }
    return {"reason": cmp["reason"], "sys_floor": _orm_f(cmp["sys_floor"]), "planes": planes}


def _orm_session_deck(ctx):
    """The deck file behind the session lattice (a .lgproj is resolved), or None."""
    p = str(ctx.lattice_path or "")
    if not p or not _os.path.isfile(p):
        return None
    if p.lower().endswith(".lgproj"):
        from linac_gen.io.project import load_project
        return str(load_project(p).lattice_path)
    return p


@_tool("orm_calibrate",
       "Orbit-response matrix (ORM) against the session lattice: map the "
       "measured BPM/trim devices onto the deck (Fermilab L:DnnBPH / "
       "L:DnnTMH rules built in, or a mapping file), build the model "
       "response from the envelope phase probe and compare (per-trim "
       "calibration k in T.m per ampere, correlation, residual, noise "
       "floor, polarity flips); mode='fit' runs the LOCO-style "
       "Levenberg-Marquardt fit of quad gradient scale factors, trim "
       "calibrations and optional BPM gains with Gaussian priors, keeps "
       "the calibration on the session for orm_apply, and can write the "
       "calibration JSON and a recalibrated deck; mode='validate' injects "
       "random errors into the model and reports how well the fit "
       "recovers them.  The session lattice is never modified (the fit "
       "runs on a copy).  Long-running: executes as a background job.",
       _ORM_SCHEMA, "compute")
def _orm_calibrate(ctx, measured, mode="compare", mapping_file=None,
                   stage="trims+quads", prior_g=0.10, prior_G=0.05,
                   sys_floor=0.05, max_iter=12, fix_quads=(),
                   exclude_bpms=(), exclude_trims=(), check_tracking=False,
                   calibration_out=None, export_deck=None, seed=7,
                   g_sigma=0.03, G_sigma=0.03,
                   progress_callback=None, should_abort=None,
                   _assist_prov=None):
    gate = _need(ctx, "lattice", "beam_config")
    if gate:
        return gate
    import copy

    from linac_gen.orm import (DeviceMap, OrmFitOptions, OrmModel,
                               calibration_from_fit, compare_orm,
                               default_device_map, export_recalibrated_deck,
                               fit_orm, load_measured, resolve_devices,
                               save_calibration, synthetic_validation)
    from linac_gen.orm.fit import STAGES

    if isinstance(measured, str):
        measured = [measured]
    try:
        paths = [_local_path(p) for p in (measured or [])]
        for opt in (mapping_file, calibration_out, export_deck):
            if opt:
                _local_path(opt)
    except ValueError as exc:
        return _refused(exc)
    if not paths:
        return _refused("measured: give at least one FORMA folder or ORM CSV")
    if mode not in ("compare", "fit", "validate"):
        return _refused(f"mode must be compare / fit / validate, got {mode!r}")
    if stage not in STAGES:
        return _refused(f"stage must be one of {', '.join(STAGES)}, got {stage!r}")
    missing = [p for p in paths if not _os.path.exists(p)]
    if missing:
        return _err(f"measured input not found: {', '.join(missing)}")
    try:
        prior_g, prior_G, sys_floor = float(prior_g), float(prior_G), float(sys_floor)
        max_iter, seed = int(max_iter), int(seed)
        g_sigma, G_sigma = float(g_sigma), float(G_sigma)
    except (TypeError, ValueError) as exc:
        return _refused(f"numeric option: {exc}")
    warns: list[str] = []

    meas, refusal, w0 = _capture(load_measured, paths)
    if refusal:
        return refusal
    warns += w0
    lat = copy.deepcopy(ctx.lattice)      # the fit sets gradient_rel on the copy only

    def _map():
        dm = (DeviceMap.from_json(mapping_file) if mapping_file
              else default_device_map(lat, meas))
        dm.exclude_bpms |= set(exclude_bpms or ())
        dm.exclude_trims |= set(exclude_trims or ())
        return dm, resolve_devices(lat, meas, dm)
    out, refusal, w1 = _capture(_map)
    if refusal:
        return refusal
    dm, sel = out
    warns += w1
    if sel.n_trim == 0 or sel.n_bpm == 0:
        return _refused("no measured device could be matched to the session "
                        "lattice — " + sel.summary()
                        + "; unmatched BPMs: " + ", ".join(sel.unmatched_devices["bpm"][:8])
                        + "; unmatched trims: " + ", ".join(sel.unmatched_devices["trim"][:8])
                        + " (give a mapping_file)")

    def _model():
        model = OrmModel(lat, ctx.beam_config, sel)
        return model, model.response(should_stop=should_abort)
    out, refusal, w2 = _capture(_model)
    if refusal:
        return refusal
    model, R = out
    warns += w2
    cmp, refusal, w3 = _capture(compare_orm, meas, R, sel, model.brho_trim,
                                model.w_trim_MeV, sys_floor=sys_floor)
    if refusal:
        return refusal
    warns += w3
    data = {"mode": mode, "selection": _orm_selection_json(sel),
            "compare": _orm_compare_json(cmp)}
    prov = _ctx_provenance(ctx)
    prov["measured"] = [str(p) for p in paths]

    if mode == "compare":
        if check_tracking:
            T, refusal, w4 = _capture(model.tracked_response, should_stop=should_abort)
            if refusal:
                return refusal
            warns += w4
            data["tracking_max_rel_dev"] = _orm_f(model.selfcheck(R, T))
        if cmp["reason"]:
            warns.append("compare refused: " + str(cmp["reason"]))
        return _ok(data, prov, warns)

    opts = OrmFitOptions(stage=stage, prior_g=prior_g, prior_G=prior_G,
                         sys_floor=sys_floor, fix_quads=tuple(fix_quads or ()),
                         max_iter=max(1, max_iter))
    prog = (None if progress_callback is None else
            (lambda it, _chi2: progress_callback(min(0.95, it / max(1, max_iter)))))

    if mode == "validate":
        v, refusal, w4 = _capture(synthetic_validation, model, meas, opts,
                                  seed=seed, g_sigma=g_sigma, G_sigma=G_sigma,
                                  should_stop=should_abort, progress=prog)
        if refusal:
            return refusal
        warns += w4
        data["validation"] = {
            "seed": int(v["seed"]), "stage": stage,
            "g_rms_injected": _orm_f(v["g_rms_injected"]),
            "g_rms_recovered": _orm_f(v["g_rms_recovered"]),
            "g_max_recovered": _orm_f(v["g_max_recovered"]),
            "k_rel_rms": _orm_f(v["k_rel_rms"]),
            "G_rms_recovered": _orm_f(v["G_rms_recovered"]),
            "worst_quads": [{"quad": lab, "injected": _orm_f(t), "recovered": _orm_f(f)}
                            for lab, t, f in v["worst_quads"]],
            "converged": bool(v["fit"].converged),
        }
        return _ok(data, prov, warns)

    fit, refusal, w4 = _capture(fit_orm, model, meas, opts,
                                should_stop=should_abort, progress=prog)
    if refusal:
        return refusal
    warns += w4
    cal = calibration_from_fit(fit, model, meas,
                               lattice_path=_orm_session_deck(ctx),
                               beam_cfg=ctx.beam_config, device_map=dm)
    ctx.orm_calibration = cal
    quads = sorted(cal["quads"], key=lambda q: -abs(q["scale"] - 1.0))
    data["fit"] = {
        "stage": fit.stage, "converged": bool(fit.converged),
        "n_iter": int(fit.n_iter), "n_data": int(fit.n_data),
        "n_params": int(fit.n_params), "dof": int(fit.dof),
        "chi2_per_point": _orm_f(fit.chi2_history[-1] / max(fit.n_data, 1)) if fit.chi2_history else None,
        "metrics": cal["metrics"],
        "quads": [{"label": q["label"], "scale": _orm_f(q["scale"]),
                   "scale_err": _orm_f(q["scale_err"]), "fixed": bool(q["fixed"]),
                   "reason": q.get("reason")} for q in quads],
        "held_fixed": [q["label"] for q in cal["quads"] if q["fixed"]],
        "trims": cal["trims"],
        "bpm_gains": cal.get("bpms") if stage == "trims+quads+bpms" else None,
        "summary": fit.summary_lines(),
    }
    files = {}
    if calibration_out:
        _, refusal, w5 = _capture(save_calibration, calibration_out, cal)
        if refusal:
            return refusal
        warns += w5
        files["calibration"] = str(calibration_out)
    if export_deck:
        src = _orm_session_deck(ctx)
        if src is None:
            warns.append("export_deck skipped: the session lattice has no deck file "
                         "on disk (load it from a .dat / .lgproj first)")
        else:
            rep, refusal, w6 = _capture(export_recalibrated_deck, src, export_deck, cal)
            if refusal:
                return refusal
            warns += w6
            files["deck"] = rep["dst"]
            data["fit"]["deck_quads_rescaled"] = len(rep["changed"])
    data["files"] = files
    if fit.reason:
        warns.append(str(fit.reason))
    return _ok(data, prov, warns)


@_tool("orm_apply",
       "Apply an ORM calibration to the session lattice: the fitted quad "
       "scale factors become Quadrupole.gradient_rel (mode='rel', the "
       "design gradients stay; ErrorStudy overwrites gradient_rel per "
       "seed and the deck writer does not save it) or are baked into the "
       "gradients (mode='bake').  Uses the last orm_calibrate fit of the "
       "session unless a calibration_file is given.  Refuses atomically "
       "when a quad's design gradient no longer matches the calibration.  "
       "In the GUI the change is one undoable command.",
       {"type": "object",
        "properties": {
            "calibration_file": {"type": "string",
                                 "description": "calibration JSON (default: the "
                                                "session's last fit)"},
            "mode": {"type": "string", "enum": ["rel", "bake"], "default": "rel"}},
        "required": []},
       "mutate")
def _orm_apply(ctx, calibration_file=None, mode="rel"):
    gate = _need(ctx, "lattice")
    if gate:
        return gate
    from linac_gen.orm import apply_calibration, load_calibration
    if mode not in ("rel", "bake"):
        return _refused(f"mode must be 'rel' or 'bake', got {mode!r}")
    if calibration_file:
        try:
            _local_path(calibration_file)
        except ValueError as exc:
            return _refused(exc)
        if not _os.path.isfile(calibration_file):
            return _err(f"{calibration_file} not found")
        cal, refusal, _w = _capture(load_calibration, calibration_file)
        if refusal:
            return refusal
    else:
        cal = getattr(ctx, "orm_calibration", None)
        if cal is None:
            return _refused("no calibration in the session — run orm_calibrate "
                            "with mode='fit' first, or give calibration_file")
    changes, refusal, warns = _capture(apply_calibration, ctx.lattice, cal, mode=mode)
    if refusal:
        return refusal
    if not changes:
        warns.append("nothing to apply: the lattice already carries this calibration (or every scale factor is 1)")
    rows = [{"element": getattr(q, "name", None), "label": getattr(q, "label", None),
             "attr": attr, "old": _orm_f(old), "new": _orm_f(new)}
            for q, attr, old, new in changes]
    if changes:
        try:
            ctx.apply_param_changes([(q, attr, new) for q, attr, _old, new in changes],
                                    label="ORM calibration")
        except Exception as exc:                                # noqa: BLE001
            return _err(f"could not apply the calibration: {exc}")
    return _ok({"mode": mode, "n_applied": len(changes), "changes": rows,
                "source": str(calibration_file) if calibration_file else "session"},
               _ctx_provenance(ctx), warns)
