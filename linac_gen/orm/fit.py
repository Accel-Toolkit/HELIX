"""LOCO-style fit of the model ORM to the measurement.

Parameters: quadrupole gradient scale factors ``g`` (bound to
``Quadrupole.gradient_rel`` through :class:`OrmModel`), signed trim
calibrations ``k`` in T·m per ampere (one per trim and kick plane) and,
optionally, BPM gains ``G`` (one per BPM and read plane).  Model entry
``R_ij = G_i · k_j · M_ij(g)`` for the two in-plane blocks.

Residual rows: ``(measured − R) / σ_eff`` over the included entries downstream
of each trim, both planes together, plus Gaussian priors ``(g − 1)/prior_g``
and ``(G − 1)/prior_G``.  ``σ_eff = sqrt(σ_meas² + (sys_floor · column max)²)``.
Solver: Levenberg–Marquardt (the prototype that produced 50 % → 4 % on the
SCL), Jacobian analytic in ``k``/``G`` and forward-difference in ``g`` (one
probe per free quad).  Quads with fewer than ``min_bpms_per_quad`` included
downstream BPMs (in the union of both planes) are held at 1 and reported;
trims with no included downstream entry get ``k = NaN``.  Cancellation
raises :class:`OperationCancelled`; the lattice is always restored.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from linac_gen.core.cancelled import OperationCancelled
from linac_gen.orm.compare import calibrate_column, sigma_eff
from linac_gen.orm.devices import OrmSelection, align_measured
from linac_gen.orm.model import ModelOrm, OrmModel

STAGES = ("trims", "trims+quads", "trims+quads+bpms")
PLANES = ("x", "y")


@dataclass
class OrmFitOptions:
    stage: str = "trims+quads"
    prior_g: float = 0.10                 # Gaussian prior width on the quad scale factors (fraction)
    prior_G: float = 0.05                 # on the BPM gains
    sys_floor: float = 0.05               # systematic floor as a fraction of the column maximum
    fix_quads: tuple = ()                 # quad labels or names held at scale 1
    max_iter: int = 12
    lam0: float = 1e-2                    # initial LM damping
    rel_tol: float = 1e-3                 # stop when χ² improves by less than this fraction
    h_g: float = 1e-3                     # forward-difference step on g
    min_bpms_per_quad: int = 2            # degeneracy rule
    bounds_g: tuple = (0.5, 1.5)

    def __post_init__(self):
        if self.stage not in STAGES:
            raise ValueError(f"stage must be one of {STAGES}, got {self.stage!r}")


@dataclass
class OrmFitResult:
    options: OrmFitOptions
    stage: str
    quads: list                           # labels
    g: np.ndarray                         # scale factors (1 = design)
    g_err: np.ndarray                     # NaN for fixed quads
    g_fixed: np.ndarray                   # bool
    fixed_reason: dict                    # label -> reason
    trims: list
    k: dict                               # kick plane -> (n_trim,) T·m per A (NaN when unconstrained)
    k_err: dict
    bpms: list
    G: dict                               # read plane -> (n_bpm,) gains (1 when not fitted)
    G_err: dict
    chi2_history: list
    n_iter: int
    converged: bool
    n_data: int
    n_params: int
    dof: int
    metrics_before: dict                  # plane -> {nrms_all, r_all, nrms_per_trim, r_per_trim}
    metrics_after: dict
    model_before: ModelOrm
    model_after: ModelOrm
    included: dict                        # plane -> bool mask of fitted entries
    reason: str | None = None
    notes: list = field(default_factory=list)
    n_probes: int = 0

    def fitted_block(self, plane: str) -> np.ndarray:
        """``G·k·M`` in mm/A for the in-plane block of ``plane`` (NaN where k is undefined)."""
        return self.G[plane][:, None] * self.k[plane][None, :] * self.model_after.per_Tm[(plane, plane)]

    def summary_lines(self) -> list:
        lines = [f"stage {self.stage}: {self.n_data} entries, {self.n_params} parameters, {self.n_iter} iterations, "
                 f"χ²/N {self.chi2_history[-1] / max(self.n_data, 1):.3f}" + ("" if self.converged else " (not converged)")]
        for p in PLANES:
            if p in self.metrics_after:
                lines.append(f"plane {p}: residual {100 * self.metrics_before[p]['nrms_all']:.0f} % → {100 * self.metrics_after[p]['nrms_all']:.1f} % of signal, "
                             f"|r| median {self.metrics_after[p]['r_median']:.3f}")
        free = ~self.g_fixed
        if free.any():
            j = int(np.argmax(np.where(free, np.abs(self.g - 1), -1)))
            lines.append(f"quads: rms change {100 * np.std(self.g[free]):.1f} %, largest {self.quads[j]} {100 * (self.g[j] - 1):+.1f} % ± {100 * self.g_err[j]:.1f} %"
                         + (f"; held fixed: {', '.join(sorted(self.fixed_reason))}" if self.fixed_reason else ""))
        if self.reason:
            lines.append("refused: " + self.reason)
        return lines


# ---------------------------------------------------------------------------
def _fit_masks(measured, sel: OrmSelection, sys_floor):
    """Aligned data, σ_eff and the fitted-entry mask for both in-plane blocks."""
    A, S, M = {}, {}, {}
    for p in PLANES:
        if p not in measured:
            continue
        a, e = align_measured(measured, sel, p, p)
        s = sigma_eff(a, e, sel.down, sys_floor)
        m = sel.down & np.isfinite(a) & np.isfinite(s)
        A[p], S[p], M[p] = a, s, m
    return A, S, M


def degeneracy_report(model: OrmModel, masks: dict, opts: OrmFitOptions) -> dict:
    """Which quads are held fixed (and why) and which trims have no fitted entry."""
    sel = model.sel
    fixed = {}
    names = {q.name for q in model.quads}
    for lab, q, i_q in zip(model.quad_labels, model.quads, model.quad_idx):
        if lab in opts.fix_quads or q.name in opts.fix_quads:
            fixed[lab] = "held by the user"
            continue
        seen = set()                                   # distinct BPMs (union over the planes)
        for p, m in masks.items():
            for n, i_b in enumerate(sel.bpm_idx):
                if i_b > i_q and m[n].any() and np.any(sel.trim_idx[m[n]] < i_q):
                    seen.add(int(i_b))
        n_bpm = len(seen)
        if n_bpm < opts.min_bpms_per_quad:
            fixed[lab] = f"only {n_bpm} fitted BPM(s) downstream (need {opts.min_bpms_per_quad}): degenerate with the BPM gains"
    unknown = [f for f in opts.fix_quads if f not in model.quad_labels and f not in names]
    dead_trims = {p: [sel.trim_labels[j] for j in range(sel.n_trim) if not m[:, j].any()] for p, m in masks.items()}
    return {"fixed": fixed, "unknown_fix_quads": unknown, "trims_without_data": dead_trims}


def solve_trims_linear(A: dict, S: dict, R: ModelOrm, G: dict, masks: dict) -> dict:
    """Closed-form per-column calibration (the ``trims`` stage and the LM start)."""
    k = {}
    for p in A:
        n_t = A[p].shape[1]
        k[p] = np.array([calibrate_column(A[p][:, j], S[p][:, j], G[p] * R.per_Tm[(p, p)][:, j], masks[p][:, j])[0] for j in range(n_t)])
    return k


def _metrics(A: dict, masks: dict, fitted: dict, sel: OrmSelection) -> dict:
    out = {}
    for p in A:
        m = masks[p]; per_n, per_r = [], []
        for j in range(sel.n_trim):
            mj = m[:, j] & np.isfinite(fitted[p][:, j])
            y, x = A[p][mj, j], fitted[p][mj, j]
            per_n.append(float(np.sqrt(np.mean((y - x) ** 2)) / np.sqrt(np.mean(y ** 2))) if mj.sum() and np.any(y != 0) else float("nan"))
            per_r.append(float(np.corrcoef(x, y)[0, 1]) if mj.sum() > 2 and x.std() > 0 and y.std() > 0 else float("nan"))
        ma = m & np.isfinite(fitted[p]); y, x = A[p][ma], fitted[p][ma]
        out[p] = {"nrms_per_trim": per_n, "r_per_trim": per_r,
                  "nrms_all": float(np.sqrt(np.mean((y - x) ** 2)) / np.sqrt(np.mean(y ** 2))) if y.size and np.any(y != 0) else float("nan"),
                  "r_all": float(np.corrcoef(x, y)[0, 1]) if y.size > 2 and x.std() > 0 and y.std() > 0 else float("nan"),
                  "r_median": float(np.nanmedian(np.abs(per_r))) if np.isfinite(per_r).any() else float("nan")}
    return out


class _Problem:
    """Parameter packing, residuals and Jacobian for one fit."""

    def __init__(self, model: OrmModel, A, S, masks, opts: OrmFitOptions, free_quads: np.ndarray, use_G: bool, planes):
        self.model, self.A, self.S, self.masks, self.opts = model, A, S, masks, opts
        self.free = free_quads; self.use_G = use_G; self.planes = planes
        self.nq = int(free_quads.sum()); self.nt = model.sel.n_trim; self.nb = model.sel.n_bpm
        self.n_k = self.nt * len(planes); self.n_G = self.nb * len(planes) if use_G else 0
        self.n_data = int(sum(masks[p].sum() for p in planes))

    def unpack(self, pv):
        g = np.ones(len(self.model.quads)); g[self.free] = 1.0 + pv[:self.nq]
        k = {p: pv[self.nq + i * self.nt: self.nq + (i + 1) * self.nt] for i, p in enumerate(self.planes)}
        off = self.nq + self.n_k
        G = {p: (1.0 + pv[off + i * self.nb: off + (i + 1) * self.nb]) if self.use_G else np.ones(self.nb) for i, p in enumerate(self.planes)}
        return g, k, G

    def residuals(self, pv, R: ModelOrm):
        g, k, G = self.unpack(pv)
        parts = []
        for p in self.planes:
            fitted = G[p][:, None] * k[p][None, :] * R.per_Tm[(p, p)]
            parts.append(((self.A[p] - fitted) / self.S[p])[self.masks[p]])
        parts.append((g[self.free] - 1.0) / self.opts.prior_g)
        if self.use_G:
            for p in self.planes:
                parts.append((G[p] - 1.0) / self.opts.prior_G)
        return np.concatenate(parts)

    def jacobian(self, pv, R: ModelOrm, should_stop):
        """Numeric in g (one probe per free quad), analytic in k and G."""
        r0 = self.residuals(pv, R); n = pv.size; J = np.zeros((r0.size, n))
        g, k, G = self.unpack(pv)
        free_idx = np.flatnonzero(self.free)
        for c, iq in enumerate(free_idx):
            gq = g.copy(); gq[iq] += self.opts.h_g
            Rq = self.model.response(gq, should_stop=should_stop)
            pq = pv.copy(); pq[c] += self.opts.h_g
            J[:, c] = (self.residuals(pq, Rq) - r0) / self.opts.h_g
        # analytic columns: d(resid)/dk_j = -G_i M_ij / σ_ij on the data rows of that plane
        row = 0
        offs = {p: None for p in self.planes}
        for p in self.planes:
            m = self.masks[p]; cnt = int(m.sum()); offs[p] = (row, row + cnt); row += cnt
        for i, p in enumerate(self.planes):
            r0_, r1_ = offs[p]; m = self.masks[p]; M = R.per_Tm[(p, p)]
            ii, jj = np.nonzero(m)                       # row-major order == the mask flattening order
            base_k = self.nq + i * self.nt
            J[r0_ + np.arange(ii.size), base_k + jj] = -(G[p][ii] * M[ii, jj]) / self.S[p][ii, jj]
            if self.use_G:
                base_G = self.nq + self.n_k + i * self.nb
                J[r0_ + np.arange(ii.size), base_G + ii] = -(k[p][jj] * M[ii, jj]) / self.S[p][ii, jj]
        # prior rows
        J[row + np.arange(self.nq), np.arange(self.nq)] = 1.0 / self.opts.prior_g; row += self.nq
        if self.use_G:
            base_G = self.nq + self.n_k
            J[row + np.arange(self.n_G), base_G + np.arange(self.n_G)] = 1.0 / self.opts.prior_G
        return J, r0


def fit_orm(model: OrmModel, measured: dict, opts: OrmFitOptions | None = None, *, should_stop=None, progress=None,
            warm_start: OrmFitResult | None = None) -> OrmFitResult:
    """Fit the model to ``measured`` (``{"x": MeasuredOrm, "y": MeasuredOrm}``); see the module docstring."""
    opts = opts or OrmFitOptions()
    sel = model.sel
    planes = tuple(p for p in PLANES if p in measured)
    if not planes:
        raise ValueError("no measured plane loaded")
    A, S, masks = _fit_masks(measured, sel, opts.sys_floor)
    for p in planes:
        if not masks[p].any():
            raise ValueError(f"plane {p}: no measured entry downstream of a trim can be fitted (check the device map / exclusions)")
    R0 = model.response(should_stop=should_stop)
    deg = degeneracy_report(model, masks, opts)
    if deg["unknown_fix_quads"]:
        raise ValueError(f"fix_quads: unknown quad label(s) {deg['unknown_fix_quads']}; known: {model.quad_labels}")
    use_quads = opts.stage in ("trims+quads", "trims+quads+bpms"); use_G = opts.stage == "trims+quads+bpms"
    fixed = np.array([lab in deg["fixed"] for lab in model.quad_labels], dtype=bool)
    free = (~fixed) if use_quads else np.zeros(len(model.quads), dtype=bool)
    prob = _Problem(model, A, S, masks, opts, free, use_G, planes)
    # start: closed-form k at g = 1, G = 1 (or the warm start)
    G1 = {p: np.ones(sel.n_bpm) for p in planes}
    k_start = solve_trims_linear(A, S, R0, G1, masks)
    metrics_before = _metrics(A, masks, {p: G1[p][:, None] * k_start[p][None, :] * R0.per_Tm[(p, p)] for p in planes}, sel)
    if warm_start is not None:
        g_ws = np.asarray(warm_start.g, dtype=float)
        pv = np.concatenate([g_ws[free] - 1.0] + [np.where(np.isfinite(warm_start.k[p]), warm_start.k[p], k_start[p]) for p in planes]
                            + ([warm_start.G[p] - 1.0 for p in planes] if use_G else []))
    else:
        pv = np.concatenate([np.zeros(prob.nq)] + [np.where(np.isfinite(k_start[p]), k_start[p], 0.0) for p in planes]
                            + ([np.zeros(sel.n_bpm) for _ in planes] if use_G else []))
    R = model.response(prob.unpack(pv)[0], should_stop=should_stop) if warm_start is not None else R0
    r = prob.residuals(pv, R); chi2 = float(r @ r); hist = [chi2]; lam = opts.lam0; converged = False; n_iter = 0
    notes = []
    if prob.nq == 0 and not use_G:
        # linear problem in k: the closed form is exact
        converged = True
    else:
        for it in range(opts.max_iter):
            if should_stop is not None and should_stop():
                raise OperationCancelled("ORM fit cancelled")
            J, r = prob.jacobian(pv, R, should_stop)
            Amat = J.T @ J; b = -J.T @ r
            accepted = False
            while lam <= 1e8:
                step = np.linalg.solve(Amat + lam * np.diag(np.diag(Amat) + 1e-12), b)
                pn = pv + step
                gq = prob.unpack(pn)[0]
                if np.any(gq < opts.bounds_g[0]) or np.any(gq > opts.bounds_g[1]):
                    lam *= 5; continue
                Rn = model.response(gq, should_stop=should_stop) if prob.nq else R
                rn = prob.residuals(pn, Rn); chi2n = float(rn @ rn)
                if chi2n < chi2:
                    pv, R, r, chi2, lam, accepted = pn, Rn, rn, chi2n, max(lam / 3, 1e-9), True
                    break
                lam *= 5
            n_iter = it + 1; hist.append(chi2)
            if progress is not None:
                progress(n_iter, chi2 / max(prob.n_data, 1))
            if not accepted:
                notes.append(f"iteration {n_iter}: no χ² decrease found (λ > 1e8)")
                break
            if (hist[-2] - hist[-1]) < opts.rel_tol * hist[-2]:
                converged = True
                break
    # covariance at the solution
    g, k, G = prob.unpack(pv)
    n_params = pv.size
    if prob.nq or use_G:
        J, r = prob.jacobian(pv, R, should_stop)
        dof = max(r.size - n_params, 1)
        cov = np.linalg.pinv(J.T @ J) * (float(r @ r) / dof)
        err = np.sqrt(np.maximum(np.diag(cov), 0.0))
    else:
        dof = max(r.size - n_params, 1)
        err = np.full(n_params, np.nan)
        for i, p in enumerate(planes):
            for j in range(sel.n_trim):
                err[prob.nq + i * sel.n_trim + j] = calibrate_column(A[p][:, j], S[p][:, j], R.per_Tm[(p, p)][:, j], masks[p][:, j])[1]
    g_err = np.full(len(model.quads), np.nan); g_err[free] = err[:prob.nq]
    k_err = {p: err[prob.nq + i * sel.n_trim: prob.nq + (i + 1) * sel.n_trim].copy() for i, p in enumerate(planes)}
    off = prob.nq + prob.n_k
    G_err = {p: (err[off + i * sel.n_bpm: off + (i + 1) * sel.n_bpm].copy() if use_G else np.full(sel.n_bpm, np.nan)) for i, p in enumerate(planes)}
    # trims without any fitted entry: k is meaningless
    for p in planes:
        dead = ~masks[p].any(axis=0)
        k[p] = k[p].copy(); k[p][dead] = np.nan; k_err[p][dead] = np.nan
    fitted = {p: G[p][:, None] * k[p][None, :] * R.per_Tm[(p, p)] for p in planes}
    metrics_after = _metrics(A, masks, fitted, sel)
    return OrmFitResult(options=opts, stage=opts.stage, quads=list(model.quad_labels), g=g, g_err=g_err,
                        g_fixed=fixed if use_quads else np.ones(len(model.quads), dtype=bool),
                        fixed_reason=deg["fixed"] if use_quads else {lab: "quads not fitted in this stage" for lab in model.quad_labels},
                        trims=list(sel.trim_labels), k=k, k_err=k_err, bpms=list(sel.bpm_labels), G=G, G_err=G_err,
                        chi2_history=hist, n_iter=n_iter, converged=converged, n_data=prob.n_data, n_params=n_params, dof=dof,
                        metrics_before=metrics_before, metrics_after=metrics_after, model_before=R0, model_after=R, included=masks,
                        notes=notes + [f"trims without fitted data ({p}): {', '.join(v)}" for p, v in deg["trims_without_data"].items() if v],
                        n_probes=model.n_probe)


def run_stages(model: OrmModel, measured: dict, opts: OrmFitOptions | None = None, *, stages=STAGES, should_stop=None, progress=None) -> dict:
    """Chain the stages with warm starts: ``{"trims": …, "trims+quads": …, "trims+quads+bpms": …}``."""
    opts = opts or OrmFitOptions()
    out = {}; prev = None
    for st in stages:
        o = OrmFitOptions(**{**opts.__dict__, "stage": st})
        prev = fit_orm(model, measured, o, should_stop=should_stop, progress=progress, warm_start=prev if st != "trims" else None)
        out[st] = prev
    return out


def synthetic_validation(model: OrmModel, measured_template: dict, opts: OrmFitOptions | None = None, *, seed: int = 7,
                         g_sigma: float = 0.03, G_sigma: float = 0.03, k_range=(3e-4, 8e-4), noise: str = "measured",
                         should_stop=None, progress=None) -> dict:
    """Closed loop: inject random quad errors, trim calibrations and BPM gains, refit, report the recovery.

    The synthetic measurement keeps the devices, exclusions and error bars of
    ``measured_template`` (``noise="measured"`` draws Gaussian noise with the
    measured σ, or 1 % of the column maximum where σ is missing).
    """
    from linac_gen.orm.measured import MeasuredOrm
    opts = opts or OrmFitOptions()
    rng = np.random.default_rng(seed)
    sel = model.sel; nq = len(model.quads)
    g_true = 1.0 + rng.normal(0.0, g_sigma, nq)
    R_true = model.response(g_true, should_stop=should_stop)
    sign = model.kick_sign
    synth = {}
    k_true, G_true = {}, {}
    for p, m in measured_template.items():
        k_true[p] = sign * rng.uniform(k_range[0], k_range[1], sel.n_trim)
        G_true[p] = 1.0 + (rng.normal(0.0, G_sigma, sel.n_bpm) if opts.stage == "trims+quads+bpms" else np.zeros(sel.n_bpm))
        A_model = G_true[p][:, None] * k_true[p][None, :] * R_true.per_Tm[(p, p)]
        # scatter the synthetic block back onto the template's devices
        vals = {rp: m.values[rp].copy() for rp in m.read_planes()}; errs = {rp: m.errors[rp].copy() for rp in m.read_planes()}
        col = {d: j for j, d in enumerate(m.trims)}
        for rp in m.read_planes():
            row = {d: i for i, d in enumerate(m.bpms[rp])}
            for j, dev in enumerate(sel.trim_devices[p]):
                if dev is None or dev not in col:
                    continue
                for i, bdev in enumerate(sel.bpm_devices[rp]):
                    if bdev is None or bdev not in row:
                        continue
                    v = A_model[i, j] if (rp == p and sel.down[i, j]) else 0.0
                    e = errs[rp][row[bdev], col[dev]]
                    if not np.isfinite(e):
                        cm = np.nanmax(np.abs(A_model[:, j])) if np.isfinite(A_model[:, j]).any() else 1.0
                        e = 0.01 * cm; errs[rp][row[bdev], col[dev]] = e
                    vals[rp][row[bdev], col[dev]] = sel.bpm_sign[rp][i] * (v + (rng.normal(0.0, e) if noise == "measured" else 0.0))
        synth[p] = MeasuredOrm(kick_plane=p, trims=list(m.trims), bpms={rp: list(m.bpms[rp]) for rp in m.read_planes()},
                               values=vals, errors=errs, units=m.units, meta=dict(m.meta), source=f"synthetic({m.source})")
    fit = fit_orm(model, synth, opts, should_stop=should_stop, progress=progress)
    free = ~fit.g_fixed
    k_rel = np.concatenate([(fit.k[p] / k_true[p] - 1.0)[np.isfinite(fit.k[p])] for p in synth])
    G_res = np.concatenate([(fit.G[p] - G_true[p]) for p in synth]) if opts.stage == "trims+quads+bpms" else np.zeros(0)
    worst = sorted(zip(fit.quads, g_true - 1.0, fit.g - 1.0, fit.g_fixed), key=lambda t: -abs(t[2] - t[1]) if not t[3] else 0.0)[:5]
    return {"seed": seed, "g_sigma_injected": g_sigma, "g_rms_injected": float(np.std(g_true[free] - 1.0)) if free.any() else float("nan"),
            "g_rms_recovered": float(np.sqrt(np.mean((fit.g - g_true)[free] ** 2))) if free.any() else float("nan"),
            "g_max_recovered": float(np.max(np.abs((fit.g - g_true)[free]))) if free.any() else float("nan"),
            "k_rel_rms": float(np.sqrt(np.mean(k_rel ** 2))) if k_rel.size else float("nan"),
            "G_rms_recovered": float(np.sqrt(np.mean(G_res ** 2))) if G_res.size else float("nan"),
            "worst_quads": [(lab, float(t), float(f)) for lab, t, f, fx in worst if not fx],
            "fit": fit, "g_true": g_true, "k_true": k_true, "G_true": G_true}
