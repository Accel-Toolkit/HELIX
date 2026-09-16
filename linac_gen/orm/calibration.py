"""Calibration files, applying a fit to a lattice, and exporting a recalibrated deck.

* :func:`calibration_from_fit` — the JSON-serialisable record of a fit
  (``__kind__ = "helix_orm_calibration"``, version 1): quad scale factors with
  errors, trim calibrations, BPM gains, exclusions, metrics, provenance.
* :func:`apply_calibration` — resolves every quad by label (then name), checks
  the design gradient, and returns ``[(element, attr, old, new)]`` so the
  caller applies the change its own way (CLI: ``setattr``; GUI: one undoable
  command; assistant: ``ctx.apply_param_changes``).  ``mode="rel"`` writes
  ``gradient_rel`` (design gradient untouched), ``mode="bake"`` writes
  ``gradient`` itself.
* :func:`export_recalibrated_deck` — copies the ORIGINAL deck text, rescales
  only the gradient token of the fitted quads (label-keyed; ordinal fallback
  for unlabeled decks), appends an inline audit comment, prepends a provenance
  header, and re-parses the result to verify it.  ``write_tracewin`` is never
  used: it would drop comments and labels.
"""
from __future__ import annotations

import datetime as _dt
import hashlib
import json
import re
from pathlib import Path

import numpy as np

from linac_gen.elements.quadrupole import Quadrupole
from linac_gen.orm.devices import element_label
from linac_gen.orm.fit import OrmFitResult

KIND = "helix_orm_calibration"
VERSION = 1
_QUAD_LINE = re.compile(r"^(?P<lead>\s*)(?:(?P<label>[A-Za-z][^\s:]*)\s*:\s*)?(?P<kw>QUAD)(?P<ws1>\s+)(?P<len>\S+)(?P<ws2>\s+)(?P<grad>\S+)(?P<rest>.*)$", re.IGNORECASE)


def _f(x):
    x = float(x)
    return x if np.isfinite(x) else None


def _sha256(path) -> str | None:
    try:
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()
    except OSError:
        return None


def _git_commit() -> str | None:
    try:
        from linac_gen.io.hdf5_output import _git_commit as gc
        return gc()
    except Exception:              # noqa: BLE001 — provenance only
        return None


def calibration_from_fit(fit: OrmFitResult, model, measured: dict, *, lattice_path=None, beam_cfg=None,
                         device_map=None, validation: dict | None = None) -> dict:
    """Build the calibration record of ``fit`` (a plain JSON-ready dict)."""
    from linac_gen import __version__
    sel = model.sel
    quads = []
    for n, (q, lab) in enumerate(zip(model.quads, model.quad_labels)):
        quads.append({"label": lab, "name": q.name, "index": int(model.quad_idx[n]), "gradient_design": float(model.g0[n]),
                      "scale": float(fit.g[n]), "scale_err": _f(fit.g_err[n]), "gradient_rel": float((1.0 + model.rel0[n]) * fit.g[n] - 1.0),
                      "fixed": bool(fit.g_fixed[n]), "reason": fit.fixed_reason.get(lab)})
    trims = []
    for j, lab in enumerate(sel.trim_labels):
        row = {"label": lab, "name": model.lattice.elements[int(sel.trim_idx[j])].name, "index": int(sel.trim_idx[j]),
               "s_m": float(sel.s_trim_m[j]), "W_MeV": _f(model.w_trim_MeV[j]), "brho_Tm": _f(model.brho_trim[j])}
        for p in ("x", "y"):
            row[f"device_{p}"] = sel.trim_devices[p][j]
            k = fit.k.get(p); ke = fit.k_err.get(p)
            row[f"k_{p}_Tm_per_A"] = _f(k[j]) if k is not None else None
            row[f"k_{p}_err"] = _f(ke[j]) if ke is not None else None
            row[f"kick_{p}_mrad_per_A"] = _f(k[j] / model.brho_trim[j] * 1e3) if k is not None and np.isfinite(k[j]) else None
        trims.append(row)
    bpms = []
    for i, lab in enumerate(sel.bpm_labels):
        row = {"label": lab, "name": model.lattice.elements[int(sel.bpm_idx[i])].name, "index": int(sel.bpm_idx[i]), "s_m": float(sel.s_bpm_m[i])}
        for p in ("x", "y"):
            row[f"device_{p}"] = sel.bpm_devices[p][i]; row[f"included_{p}"] = bool(sel.bpm_include[p][i])
            row[f"sign_{p}"] = int(sel.bpm_sign[p][i])
            row[f"gain_{p}"] = _f(fit.G[p][i]) if p in fit.G else None
            row[f"gain_{p}_err"] = _f(fit.G_err[p][i]) if p in fit.G_err else None
        bpms.append(row)
    meas_prov = [{"kick_plane": p, "source": m.source, "stamp": m.meta.get("stamp"), "folder": m.meta.get("folder") or m.meta.get("file"),
                  "units": m.units, "dead_devices": list(m.meta.get("dead_devices", []))} for p, m in measured.items()]
    beam = None
    if beam_cfg is not None:
        beam = {k: getattr(beam_cfg, k, None) for k in ("species", "energy", "frequency", "current")}
    return {"__kind__": KIND, "__version__": VERSION, "created": _dt.datetime.now().astimezone().isoformat(timespec="seconds"),
            "helix_version": __version__, "git_commit": _git_commit(),
            "lattice": {"path": str(lattice_path) if lattice_path else None, "sha256": _sha256(lattice_path) if lattice_path else None,
                        "n_elements": len(model.lattice.elements)},
            "beam": beam, "measured": meas_prov,
            "device_map": device_map.to_dict() if device_map is not None else None,
            "selection": {"notes": list(sel.notes), "unmatched_devices": sel.unmatched_devices, "unmatched_elements": sel.unmatched_elements},
            "fit": {"stage": fit.stage, "prior_g": fit.options.prior_g, "prior_G": fit.options.prior_G, "sys_floor": fit.options.sys_floor,
                    "fix_quads": list(fit.options.fix_quads), "bounds_g": list(fit.options.bounds_g), "chi2_history": [float(c) for c in fit.chi2_history],
                    "n_iter": fit.n_iter, "converged": fit.converged, "n_data": fit.n_data, "n_params": fit.n_params, "dof": fit.dof,
                    "notes": list(fit.notes), "reason": fit.reason},
            "quads": quads, "trims": trims, "bpms": bpms,
            "metrics": {"before": _metrics_json(fit.metrics_before), "after": _metrics_json(fit.metrics_after)},
            "validation": validation,
            "provenance": {"model": "envelope phase-probe bare maps at zero current; unit kick at the trim exit chained to each BPM",
                           "kick_convention": "dx' = sign(q)*by_l/Brho, dy' = sign(q)*bx_l/Brho (Steerer._kick_mrad)",
                           "units": {"measured": measured[next(iter(measured))].units if measured else "mm/A", "model": "mm/(T·m)",
                                     "k": "T·m/A (signed)", "gradient": "T/m"},
                           "applied_as": "gradient_rel = (1 + gradient_rel_before) * scale - 1"}}


def _metrics_json(m: dict) -> dict:
    return {p: {"nrms_all": _f(d["nrms_all"]), "r_all": _f(d["r_all"]), "r_median": _f(d["r_median"]),
                "nrms_per_trim": [_f(v) for v in d["nrms_per_trim"]], "r_per_trim": [_f(v) for v in d["r_per_trim"]]} for p, d in m.items()}


def save_calibration(path, cal: dict) -> None:
    if cal.get("__kind__") != KIND:
        raise ValueError("not a HELIX ORM calibration record")
    Path(path).write_text(json.dumps(cal, indent=1), encoding="utf-8")


def load_calibration(path) -> dict:
    cal = json.loads(Path(path).read_text(encoding="utf-8"))
    if cal.get("__kind__") != KIND:
        raise ValueError(f"{path}: not a HELIX ORM calibration file (missing __kind__ = {KIND!r})")
    if int(cal.get("__version__", 0)) > VERSION:
        raise ValueError(f"{path}: calibration version {cal.get('__version__')} is newer than this HELIX ({VERSION})")
    validate_calibration(cal, where=str(path))
    return cal


def validate_calibration(cal: dict, *, where: str = "calibration") -> None:
    """Refuse a record whose quad entries cannot be applied (missing / non-numeric fields)."""
    quads = cal.get("quads")
    if not isinstance(quads, list):
        raise ValueError(f"{where}: 'quads' must be a list")
    for n, q in enumerate(quads):
        if not isinstance(q, dict) or not isinstance(q.get("label"), (str, type(None))):
            raise ValueError(f"{where}: quads[{n}] is not a calibration entry")
        for key in ("scale", "gradient_design", "gradient_rel"):
            v = q.get(key)
            if isinstance(v, bool) or not isinstance(v, (int, float)) or not np.isfinite(v):
                raise ValueError(f"{where}: quads[{n}] ({q.get('label')}): {key!r} must be a finite number, got {v!r}")


# ---------------------------------------------------------------------------
def _resolve_quad(lattice, entry: dict):
    """The Quadrupole an entry refers to: by label (unique), else by name, else by index + design gradient."""
    els = lattice.elements
    lab, name, idx = entry.get("label"), entry.get("name"), entry.get("index")
    cand = [e for e in els if isinstance(e, Quadrupole) and lab and getattr(e, "label", None) == lab]
    if len(cand) > 1 and name:
        cand = [e for e in cand if e.name == name] or cand
    if not cand and name:
        cand = [e for e in els if isinstance(e, Quadrupole) and e.name == name]
    if not cand and idx is not None and 0 <= int(idx) < len(els) and isinstance(els[int(idx)], Quadrupole):
        cand = [els[int(idx)]]
    if len(cand) != 1:
        raise ValueError(f"calibration entry {lab or name!r}: {'no' if not cand else len(cand)} matching quadrupole in the lattice")
    q = cand[0]
    g0 = float(entry["gradient_design"])
    if not np.isclose(float(q.gradient), g0, rtol=1e-9, atol=0.0):
        raise ValueError(f"quad {element_label(q)}: design gradient {q.gradient!r} differs from the calibration's {g0!r} — different deck?")
    return q


def apply_calibration(lattice, cal: dict, *, mode: str = "rel") -> list:
    """Return the changes ``[(quad, attr, old, new)]`` that apply ``cal`` (nothing is mutated here).

    Every quad is resolved before anything is returned, so a mismatch leaves
    the lattice untouched (atomic by construction).  ``mode="rel"`` sets the
    calibration's absolute ``gradient_rel`` target, so applying twice is a
    no-op (quads already at the target are skipped); a quad whose
    ``gradient_rel`` is neither the value the fit started from nor the
    target (another calibration applied, an error study active) is refused,
    and so is baking on top of an applied calibration.
    """
    if mode not in ("rel", "bake"):
        raise ValueError("mode must be 'rel' or 'bake'")
    validate_calibration(cal)
    changes = []
    for entry in cal.get("quads", []):
        q = _resolve_quad(lattice, entry)
        scale = float(entry["scale"])
        if entry.get("fixed") and abs(scale - 1.0) < 1e-15:
            continue
        target = float(entry["gradient_rel"])                 # (1 + rel_start) * scale - 1
        start = (1.0 + target) / scale - 1.0                  # what the fitted lattice carried
        cur = float(getattr(q, "gradient_rel", 0.0))
        at_target = np.isclose(cur, target, rtol=1e-9, atol=1e-12)
        at_start = np.isclose(cur, start, rtol=1e-9, atol=1e-12)
        if mode == "rel":
            if at_target:
                continue                                      # already applied — idempotent
            if not at_start:
                raise ValueError(f"quad {element_label(q)}: gradient_rel {cur!r} is neither the calibration's starting value "
                                 f"{start!r} nor its target {target!r} — another calibration or an error study is active")
            changes.append((q, "gradient_rel", cur, target))
        else:
            if not at_start:
                raise ValueError(f"quad {element_label(q)}: gradient_rel {cur!r} differs from the calibration's starting value "
                                 f"{start!r} — undo the applied calibration before baking (or export the deck instead)")
            old = float(q.gradient); changes.append((q, "gradient", old, float(old * scale)))
    return changes


def apply_changes(changes: list) -> None:
    for q, attr, _old, new in changes:
        setattr(q, attr, new)


def revert_changes(changes: list) -> None:
    for q, attr, old, _new in changes:
        setattr(q, attr, old)


# ---------------------------------------------------------------------------
def export_recalibrated_deck(src_dat, dst_dat, cal: dict, *, stamp: str | None = None, lgproj_src=None, comment_tag: str = "ORM fit") -> dict:
    """Write ``dst_dat`` = ``src_dat`` with the fitted quads' gradient tokens rescaled; verify by re-parsing.

    Text is handled as latin-1 bytes with line endings preserved.  A quad is
    located by its label (``Q13: QUAD …``); unlabeled decks fall back to the
    k-th ``QUAD`` card ↔ the calibration's k-th quad by lattice order.  The
    token must equal the calibration's design gradient (rtol 1e-9), otherwise
    the deck is not the one that was fitted and the export is refused.
    """
    src, dst = Path(src_dat).resolve(), Path(dst_dat).resolve()      # absolute: the verifier chdirs into the deck folder
    raw = src.read_bytes()
    nl = "\r\n" if b"\r\n" in raw else "\n"
    text = raw.decode("latin-1")
    lines = text.split(nl)
    entries = [e for e in cal.get("quads", []) if not (e.get("fixed") and abs(float(e["scale"]) - 1.0) < 1e-15)]
    quad_lines = [(n, m) for n, ln in enumerate(lines) for m in [_QUAD_LINE.match(ln)] if m]
    by_label: dict = {}
    for n, m in quad_lines:
        if m.group("label"):
            by_label.setdefault(m.group("label"), []).append(n)
    all_cal_quads = cal.get("quads", [])
    stamp = stamp or _dt.datetime.now().strftime("%Y-%m-%d")
    changed = []
    for entry in entries:
        lab = entry.get("label"); scale = float(entry["scale"]); g0 = float(entry["gradient_design"])
        if lab and lab in by_label:
            hits = by_label[lab]
            if len(hits) != 1:
                raise ValueError(f"{src.name}: label {lab!r} appears on {len(hits)} QUAD lines")
            n = hits[0]
        else:
            # ordinal fallback: the calibration's position among all its quads ↔ the deck's QUAD cards
            k_all = [i for i, e in enumerate(all_cal_quads) if e is entry][0]
            ordinal = [n for n, _ in quad_lines]
            if not by_label and k_all < len(ordinal):
                n = ordinal[k_all]
            else:
                raise ValueError(f"{src.name}: no QUAD line labelled {lab!r} (deck labels differ from the fitted lattice)")
        m = _QUAD_LINE.match(lines[n])
        try:
            g_text = float(m.group("grad"))
        except ValueError:
            raise ValueError(f"{src.name} line {n + 1}: gradient token {m.group('grad')!r} is not a number") from None
        if not np.isclose(g_text, g0, rtol=1e-9, atol=0.0):
            raise ValueError(f"{src.name} line {n + 1} ({lab}): gradient {g_text!r} differs from the fitted design value {g0!r}; "
                             "the deck on disk is not the one that was fitted")
        if abs(scale - 1.0) < 1e-15:
            continue
        new_tok = f"{g_text * scale:.10g}"
        err = entry.get("scale_err")
        tail = f"\t; {comment_tag} {stamp}: x{scale:.5f}" + (f" ± {err:.4f}" if err is not None else "") + f" (was {m.group('grad')})"
        lines[n] = (m.group("lead") + (f"{m.group('label')}: " if m.group("label") else "") + m.group("kw") + m.group("ws1")
                    + m.group("len") + m.group("ws2") + new_tok + m.group("rest") + tail)
        changed.append((lab or f"QUAD #{k_all + 1}", g_text, g_text * scale, scale))
    hdr = [f"; DERIVED DECK - {comment_tag} {stamp}: {len(changed)} quadrupole gradient(s) rescaled by a LOCO-style fit of the HELIX orbit-response model",
           f";   source deck: {src.name}" + (f" (sha256 {_sha256(src)[:16]}...)" if _sha256(src) else ""),
           ";   measurement: " + "; ".join(f"{m.get('source')} ({m.get('stamp')})" for m in cal.get("measured", []) or []),
           ";   fit stage: " + str((cal.get("fit") or {}).get("stage")) + "; residual (rms/signal) before → after: "
           + ", ".join(f"{p}: {100 * (cal['metrics']['before'][p]['nrms_all'] or float('nan')):.0f} % → {100 * (cal['metrics']['after'][p]['nrms_all'] or float('nan')):.1f} %"
                       for p in (cal.get("metrics") or {}).get("after", {})),
           ";   changes (%): " + ", ".join(f"{lab} {100 * (s - 1):+.1f}" for lab, _a, _b, s in changed),
           ";   the design gradients are those of the source deck; HELIX applies the same factors as Quadrupole.gradient_rel"]
    out = nl.join(hdr + lines)
    dst.write_bytes(out.encode("latin-1", errors="replace"))
    report = verify_exported_deck(src, dst, cal)
    report.update({"changed": changed, "dst": str(dst), "header_lines": len(hdr)})
    if lgproj_src is not None:
        proj = json.loads(Path(lgproj_src).read_text(encoding="utf-8")); proj["lattice_path"] = dst.name
        lp = dst.with_suffix(".lgproj"); lp.write_text(json.dumps(proj, indent=2) + "\n", encoding="utf-8"); report["lgproj"] = str(lp)
    return report


def verify_exported_deck(src_dat, dst_dat, cal: dict) -> dict:
    """Re-parse both decks: same element sequence, fitted gradients = design × scale, everything else equal."""
    import contextlib, io, os
    from linac_gen.io.tracewin_parser import parse_tracewin
    src, dst = Path(src_dat).resolve(), Path(dst_dat).resolve()

    def _parse(p):
        cwd = os.getcwd(); os.chdir(p.parent)
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                return parse_tracewin(str(p))[0]
        finally:
            os.chdir(cwd)
    a, b = _parse(src), _parse(dst)
    if [type(e).__name__ for e in a.elements] != [type(e).__name__ for e in b.elements] or [e.name for e in a.elements] != [e.name for e in b.elements]:
        raise ValueError("exported deck parses to a different element sequence than the source")
    expected = {}
    for entry in cal.get("quads", []):
        expected[(entry.get("label"), entry.get("name"))] = float(entry["gradient_design"]) * float(entry["scale"])
    n_checked = 0; max_rel = 0.0
    for ea, eb in zip(a.elements, b.elements):
        if isinstance(ea, Quadrupole):
            key = (getattr(ea, "label", None), ea.name)
            want = expected.get(key)
            if want is None:
                want = next((v for (lab, nm), v in expected.items() if lab and lab == getattr(ea, "label", None)), None)
            if want is not None:
                if not np.isclose(float(eb.gradient), want, rtol=1e-9, atol=0.0):
                    raise ValueError(f"quad {element_label(ea)}: exported gradient {eb.gradient} != design × scale {want}")
                n_checked += 1; max_rel = max(max_rel, abs(float(eb.gradient) / want - 1.0))
            elif float(eb.gradient) != float(ea.gradient):
                raise ValueError(f"quad {element_label(ea)}: gradient changed although it was not fitted")
        for k, v in vars(ea).items():
            if k in ("gradient",) or k.startswith("_"):
                continue
            if isinstance(v, (int, float, str, bool)) or v is None:
                if getattr(eb, k, None) != v and not (isinstance(v, float) and np.isnan(v)):
                    raise ValueError(f"{element_label(ea)}.{k}: {v!r} → {getattr(eb, k, None)!r} changed on export")
    return {"n_elements": len(a.elements), "n_quads_checked": n_checked, "max_rel_dev": max_rel}
