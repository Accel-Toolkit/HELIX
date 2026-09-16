"""Measured-vs-model comparison of an orbit response matrix.

Per trim column: the largest downstream value normalises the shape, a
weighted least-squares scale ``k = measured ÷ model`` (T·m per ampere, signed)
is the trim's effective calibration, the Pearson correlation and the residual
rms over the downstream BPMs measure the shape agreement, the entries upstream
of the trim give the measured noise floor, and the cross-plane block gives the
coupling level.  Result dicts follow the analysis convention: ``reason`` is
``None`` when usable, NaN marks undefined per-row entries.
"""
from __future__ import annotations

import numpy as np

from linac_gen.orm.devices import OrmSelection, align_measured, upstream_rows
from linac_gen.orm.model import ModelOrm


def column_normalize(A: np.ndarray, down: np.ndarray) -> np.ndarray:
    """Every column divided by its largest |value| over the downstream rows (NaN-safe)."""
    out = np.full_like(A, np.nan, dtype=float)
    for j in range(A.shape[1]):
        col = A[:, j]; m = down[:, j] & np.isfinite(col)
        scale = np.max(np.abs(col[m])) if m.any() else 0.0
        out[:, j] = col / scale if scale > 0 else np.nan
    return out


def sigma_eff(meas: np.ndarray, err: np.ndarray, down: np.ndarray, sys_floor: float) -> np.ndarray:
    """``sqrt(err² + (sys_floor · column max)²)`` — the systematic floor keeps trims comparable."""
    out = np.full_like(meas, np.nan, dtype=float)
    for j in range(meas.shape[1]):
        m = down[:, j] & np.isfinite(meas[:, j])
        cm = np.max(np.abs(meas[m, j])) if m.any() else 0.0
        e = np.where(np.isfinite(err[:, j]), err[:, j], 0.0)
        out[:, j] = np.sqrt(e ** 2 + (sys_floor * cm) ** 2)
    out[out <= 0] = np.nan
    return out


def calibrate_column(meas, err, model, mask):
    """Weighted LSQ scale ``k`` (and its error) of ``model`` onto ``meas`` over ``mask``."""
    m = mask & np.isfinite(meas) & np.isfinite(model) & np.isfinite(err) & (err > 0)
    if m.sum() == 0 or not np.any(model[m] != 0):
        return float("nan"), float("nan")
    w = 1.0 / err[m] ** 2; x, y = model[m], meas[m]
    sxx = float(np.sum(w * x * x))
    return float(np.sum(w * x * y) / sxx), float(np.sqrt(1.0 / sxx))


def compare_orm(measured: dict, model: ModelOrm, sel: OrmSelection, brho_trim, w_trim=None, *, sys_floor: float = 0.05) -> dict:
    """Per-plane comparison of the in-plane blocks (``kick == read``)."""
    out = {"reason": None, "sys_floor": float(sys_floor), "planes": {}}
    brho_trim = np.asarray(brho_trim, dtype=float)
    for p in ("x", "y"):
        if p not in measured:
            continue
        A, E = align_measured(measured, sel, p, p)
        R = model.per_Tm[(p, p)]
        other = "y" if p == "x" else "x"
        Ax, _ = align_measured(measured, sel, p, other) if other in measured.get(p).values else (np.full_like(A, np.nan), None)
        sig = sigma_eff(A, E, sel.down, sys_floor)
        rows = []
        kg_num = kg_den = 0.0; ys, xs = [], []
        for j in range(sel.n_trim):
            dn = sel.down[:, j] & np.isfinite(A[:, j]); up = (~sel.down[:, j]) & np.isfinite(A[:, j])
            k, dk = calibrate_column(A[:, j], sig[:, j], R[:, j], dn)
            n_down = int(dn.sum())
            if n_down > 2 and np.isfinite(k) and np.std(R[dn, j]) > 0 and np.std(A[dn, j]) > 0:
                r = float(np.corrcoef(R[dn, j], A[dn, j])[0, 1])
            else:
                r = float("nan")
            resid = A[dn, j] - k * R[dn, j] if np.isfinite(k) else np.array([])
            nrms = float(np.sqrt(np.mean(resid ** 2)) / np.sqrt(np.mean(A[dn, j] ** 2))) if resid.size and np.any(A[dn, j] != 0) else float("nan")
            noise_up = float(np.sqrt(np.mean(A[up, j] ** 2))) if up.any() else float("nan")
            cross = Ax[dn, j] if Ax is not None else np.array([])
            cross = cross[np.isfinite(cross)]
            cross_ratio = float(np.sqrt(np.mean(cross ** 2)) / np.sqrt(np.mean(A[dn, j] ** 2))) if cross.size and n_down else float("nan")
            brho = float(brho_trim[j]) if j < brho_trim.size else float("nan")
            rows.append({"trim": sel.trim_labels[j], "device": sel.trim_devices[p][j], "k_Tm_per_A": k, "dk": dk,
                         "kick_mrad_per_A": k / brho * 1e3 if np.isfinite(k) and brho > 0 else float("nan"),
                         "r": r, "nrms_resid": nrms, "n_down": n_down, "noise_upstream_mm_per_A": noise_up,
                         "cross_over_inplane": cross_ratio, "brho_Tm": brho,
                         "W_MeV": float(w_trim[j]) if w_trim is not None and j < len(w_trim) else float("nan"),
                         "included": bool(sel.trim_include[p][j])})
            if np.isfinite(k) and n_down:
                w = 1.0 / sig[dn, j] ** 2; kg_num += float(np.sum(w * R[dn, j] * A[dn, j])); kg_den += float(np.sum(w * R[dn, j] ** 2))
                ys.append(A[dn, j]); xs.append(R[dn, j])
        k_global = kg_num / kg_den if kg_den > 0 else float("nan")
        if xs:
            x = np.concatenate(xs); y = np.concatenate(ys)
            r_global = float(np.corrcoef(x, y)[0, 1]) if x.std() > 0 and y.std() > 0 else float("nan")
            nrms_global = float(np.sqrt(np.mean((y - k_global * x) ** 2)) / np.sqrt(np.mean(y ** 2))) if np.isfinite(k_global) else float("nan")
        else:
            r_global = nrms_global = float("nan")
        tank = upstream_rows(measured, sel, p, p)
        ks = np.array([row["k_Tm_per_A"] for row in rows]); ok = np.isfinite(ks) & (np.array([row["n_down"] for row in rows]) > 2)
        flipped = [row["trim"] for row, o in zip(rows, ok) if o and np.isfinite(k_global) and np.sign(row["k_Tm_per_A"]) != np.sign(k_global)]
        out["planes"][p] = {"per_trim": rows, "k_global_Tm_per_A": k_global, "r_global": r_global, "nrms_global": nrms_global,
                            "tank_noise_mm_per_A": float(np.sqrt(np.nanmean(tank ** 2))) if tank.size else float("nan"),
                            "polarity_flipped": flipped, "n_entries": int(sum(len(v) for v in ys)),
                            "aligned": A, "aligned_err": E, "model": R, "sigma_eff": sig,
                            "measured_norm": column_normalize(A, sel.down),
                            "model_norm": column_normalize(R * (np.sign(k_global) if np.isfinite(k_global) and k_global != 0 else 1.0), sel.down)}
        if not xs:
            out["reason"] = (out["reason"] or "") + f"plane {p}: no downstream measured entries; "
    if not out["planes"]:
        out["reason"] = "no measured plane loaded"
    return out


def summary_lines(cmp: dict) -> list:
    """Human-readable one-liners for the CLI / GUI status."""
    lines = []
    for p, d in cmp["planes"].items():
        rs = [row["r"] for row in d["per_trim"] if np.isfinite(row["r"])]
        lines.append(f"plane {p}: {d['n_entries']} entries, k_global {d['k_global_Tm_per_A']*1e3:+.3f} mT·m/A, "
                     f"|r| median {np.median(np.abs(rs)) if rs else float('nan'):.3f}, residual {100*d['nrms_global']:.0f} % of signal, "
                     f"noise floor {d['tank_noise_mm_per_A']:.3f} mm/A" + (f", polarity flipped: {', '.join(d['polarity_flipped'])}" if d["polarity_flipped"] else ""))
    if cmp.get("reason"):
        lines.append("refused: " + cmp["reason"])
    return lines
