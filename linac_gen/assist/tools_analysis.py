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
    _ctx_provenance, _err, _need, _ok, _refused, _save_capture, _tool,
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
