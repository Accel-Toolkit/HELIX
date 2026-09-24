"""Per-run observables read back from ``results.h5`` and the criticality
rule applied against a baseline row."""
from __future__ import annotations

import math

import numpy as np

__all__ = ["evaluate_run", "criticality_rule", "EXTRA_KEYS"]

EXTRA_KEYS = ("w_kin_end", "ref_phi_s_end", "treaty_w_kin", "treaty_phi_deg",
              "treaty_sigma_x", "treaty_sigma_y", "treaty_alpha_x", "treaty_beta_x",
              "treaty_alpha_y", "treaty_beta_y", "loss_w_total", "loss_w_per_m_peak",
              "loss_w_per_m_peak_s_m", "n_lost", "n_macro", "foil_sigma_x_mm",
              "foil_sigma_y_mm", "foil_w_cm2_peak", "strip_eff", "h0_frac",
              "missed_frac", "n_at_foil")


def _row_at(res: dict, key: str, idx):
    arr = res.get(key)
    if arr is None or idx is None:
        return None
    try:
        arr = np.asarray(arr, dtype=float)
        if arr.ndim != 1 or idx >= arr.size:
            return None
        return float(arr[int(idx)])
    except (TypeError, ValueError):
        return None


def _last(res: dict, key: str):
    arr = res.get(key)
    try:
        return float(np.asarray(arr, dtype=float)[-1]) if arr is not None and len(arr) else None
    except (TypeError, ValueError):
        return None


def evaluate_run(results_h5: str, *, element_index: dict, landmarks: dict,
                 sections, current_mA: float, duty_pct: float,
                 hits_per_particle: float = 1.0) -> dict:
    """``{EXTRA_KEYS...}`` for one run.  ``element_index`` maps element
    name → 0-based lattice index; ``landmarks`` = {treaty_element,
    foil_element}; ``sections`` = [(name, s_start_m, s_end_m)] (loss power
    per section lands under ``loss_w_per_m_peak_<section>``)."""
    from linac_gen.io.hdf5_output import load_results_hdf5
    res = load_results_hdf5(str(results_h5))
    out: dict = {k: None for k in EXTRA_KEYS}
    out["w_kin_end"] = _last(res, "ref_w_kin")
    out["ref_phi_s_end"] = _last(res, "ref_phi_s")
    exit_idx = res.get("element_exit_idx")
    exit_idx = np.asarray(exit_idx, dtype=int) if exit_idx is not None else None

    def _exit_row(name):
        i = element_index.get(name)
        if i is None or exit_idx is None or i >= exit_idx.size:
            return None
        return int(exit_idx[i])

    tr = landmarks.get("treaty_element")
    if tr:
        j = _exit_row(tr)
        out["treaty_w_kin"] = _row_at(res, "ref_w_kin", j)
        out["treaty_phi_deg"] = _row_at(res, "ref_phi_s", j)
        out["treaty_sigma_x"] = _row_at(res, "sigma_x", j)
        out["treaty_sigma_y"] = _row_at(res, "sigma_y", j)
        for k in ("alpha_x", "beta_x", "alpha_y", "beta_y"):
            out[f"treaty_{k}"] = _row_at(res, k, j)
    # loss power
    lt = res.get("loss_table")
    n_macro = int(res.get("n_macro") or 0)
    out["n_macro"] = n_macro or None
    if lt is not None and np.asarray(lt).size and n_macro:
        from linac_gen.analysis.loss_power import loss_power_profile, loss_power_summary
        lt = np.asarray(lt)
        s_end = _last(res, "s") or 1.0
        summ = loss_power_summary(lt, current_mA=current_mA, duty_pct=duty_pct, n_macro=n_macro)
        centers, wpm = loss_power_profile(lt, current_mA=current_mA, duty_pct=duty_pct,
                                          n_macro=n_macro, s_end_mm=max(s_end, 1.0))
        j = int(np.argmax(wpm)) if len(wpm) else 0
        out["loss_w_total"] = float(summ["lost_w"])
        out["n_lost"] = int(summ["n_lost"])
        out["loss_w_per_m_peak"] = float(wpm[j]) if len(wpm) else 0.0
        out["loss_w_per_m_peak_s_m"] = float(centers[j]) * 1e-3 if len(wpm) else None
        for name, s0, s1 in sections:
            m = (centers * 1e-3 >= s0) & (centers * 1e-3 < s1)
            out[f"loss_w_per_m_peak_{name}"] = float(wpm[m].max()) if m.any() else 0.0
    # foil plane
    foil = landmarks.get("foil_element")
    if foil:
        j = _exit_row(foil)
        out["foil_sigma_x_mm"] = _row_at(res, "sigma_x", j)
        out["foil_sigma_y_mm"] = _row_at(res, "sigma_y", j)
        snap = _foil_snapshot(results_h5, element_index.get(foil), res)
        if snap is not None:
            x, y, w = snap                      # ALIVE particles at the foil
            out["n_at_foil"] = int(x.size)
            if x.size:
                out["foil_sigma_x_mm"] = float(np.std(x))
                out["foil_sigma_y_mm"] = float(np.std(y))
                if n_macro and current_mA > 0:
                    from linac_gen.analysis.loss_power import plane_power_density
                    H, _xe, _ye = plane_power_density(
                        x, y, energy_mev=w, current_mA=current_mA, duty_pct=duty_pct,
                        n_macro=n_macro)
                    out["foil_w_cm2_peak"] = float(H.max()) * float(hits_per_particle)
        ut = res.get("unstripped_table")
        n_foil = out["n_at_foil"]
        if ut is not None and n_foil:
            ut = np.asarray(ut)
            states = ut["state"]
            n_h0 = int(np.count_nonzero(states == "H0"))
            n_hm = int(np.count_nonzero(states == "H-"))
            n_miss = int(np.count_nonzero(states == "missed"))
            out["h0_frac"] = n_h0 / n_foil
            out["missed_frac"] = n_miss / n_foil
            out["strip_eff"] = max(0.0, 1.0 - (n_h0 + n_hm + n_miss) / n_foil)
        elif n_foil:
            out["h0_frac"], out["missed_frac"], out["strip_eff"] = 0.0, 0.0, None
    return out


def _foil_snapshot(results_h5, foil_index, res):
    """(x, y, energy) of the ALIVE particles snapshotted at the foil, if
    any: the snapshot's own ``lost`` mask when the file carries it, else
    the loss record's particle ids (a snapshot stores every launched row,
    dead ones included)."""
    if foil_index is None:
        return None
    try:
        import h5py
    except ImportError:            # pragma: no cover
        return None
    exit_idx = res.get("element_exit_idx")
    s_arr = res.get("s")
    if exit_idx is None or s_arr is None:
        return None
    exit_idx = np.asarray(exit_idx, dtype=int)
    if foil_index >= exit_idx.size:
        return None
    s_foil = float(np.asarray(s_arr, dtype=float)[int(exit_idx[foil_index])])
    with h5py.File(str(results_h5), "r") as f:
        parts = f.get("particles")
        if parts is None:
            return None
        best = None
        for key in parts:
            g = parts[key]
            if abs(float(g.attrs.get("s", -1e30)) - s_foil) < 1e-6:
                best = g
                break
        if best is None:
            return None
        data = np.asarray(best["data"])
        w_kin = float(best.attrs.get("w_kin", 0.0))
        lost = np.asarray(best["lost"], dtype=bool) if "lost" in best else None
    if data.size == 0:
        return np.zeros(0), np.zeros(0), np.zeros(0)
    if lost is None:
        lt = res.get("loss_table")
        lost = np.zeros(data.shape[0], dtype=bool)
        if lt is not None and np.asarray(lt).size:
            ids = np.asarray(np.asarray(lt)["particle_id"], dtype=int)
            lost[ids[(ids >= 0) & (ids < data.shape[0])]] = True
    alive = data[~lost]
    return alive[:, 0], alive[:, 2], w_kin + alive[:, 5]


def _threshold(rule: dict, key: str, default: float):
    """A rule threshold; ``None`` (JSON ``null``) switches the term off."""
    if key in rule and rule[key] is None:
        return None
    return float(rule.get(key, default))


def criticality_rule(baseline: dict, row: dict, rule: dict,
                     sections=()) -> tuple[bool, dict]:
    """The study plan's criticality rule against the baseline row:
    normalised emittance growth > ``emit_growth_pct`` in any plane, loss
    fraction > ``loss_frac`` (when transmission is known), exit-energy
    deviation > ``energy_pct``, or a section's peak loss density above
    its ``w_per_m`` limit.  A threshold set to ``None`` switches its term
    off (the growth / deviation is still reported in ``terms``); a run
    whose exit energy is unknown stays critical whatever the rule.
    Returns ``(critical, terms)``."""
    terms: dict = {}
    crit = False
    eg = _threshold(rule, "emit_growth_pct", 5.0)
    for k in ("emit_nx", "emit_ny", "emit_nz"):
        b, v = baseline.get(k), row.get(k)
        if b and v is not None and b > 0:
            g = 100.0 * (float(v) / float(b) - 1.0)
            terms[f"{k}_growth_pct"] = g
            if eg is not None:
                crit |= g > eg
    t0, t = baseline.get("transmission"), row.get("transmission")
    if t0 is not None and t is not None:
        lf = max(0.0, (float(t0) - float(t)) / 100.0)
        terms["loss_frac"] = lf
        lim = _threshold(rule, "loss_frac", 1e-4)
        if lim is not None:
            crit |= lf > lim
    e0, e = baseline.get("ref_w_kin"), row.get("ref_w_kin")
    if e0 and e is not None and math.isfinite(float(e)):
        dp = 100.0 * abs(float(e) - float(e0)) / abs(float(e0))
        terms["energy_dev_pct"] = dp
        lim = _threshold(rule, "energy_pct", 0.5)
        if lim is not None:
            crit |= dp > lim
    elif e0 and e is None:
        terms["energy_dev_pct"] = None
        crit = True
    limits = rule.get("w_per_m") or {}
    for name, _s0, _s1 in sections:
        v = row.get(f"loss_w_per_m_peak_{name}")
        if v is None:
            continue
        lim = float(limits.get(name, limits.get("default", 1.0)))
        terms[f"loss_w_per_m_{name}"] = float(v)
        crit |= float(v) > lim
    return bool(crit), terms
