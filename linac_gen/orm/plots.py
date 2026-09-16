"""Matplotlib figures for the ORM comparison and fit (CLI ``--plots``; the GUI draws its own pyqtgraph views)."""
from __future__ import annotations

import numpy as np

RED, BLUE, INK, GRID, GREY = "#c1121f", "#023e8a", "#22223b", "#e5e5ea", "#6c757d"
PLANE = {"x": ("horizontal", "H", RED), "y": ("vertical", "V", BLUE)}


def _plt():
    import matplotlib
    if matplotlib.get_backend().lower() not in ("agg", "module://matplotlib_inline.backend_inline"):
        try:
            matplotlib.use("Agg")
        except Exception:               # noqa: BLE001 — a GUI backend is already active
            pass
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 10, "axes.edgecolor": INK, "axes.spines.top": False, "axes.spines.right": False, "figure.facecolor": "white"})
    return plt


def _first_downstream(sel):
    return [int(np.argmax(sel.down[:, j])) if sel.down[:, j].any() else sel.n_bpm for j in range(sel.n_trim)]


def fig_heatmaps(cmp: dict, sel):
    """measured | model | difference per plane, columns normalised to their largest downstream value."""
    plt = _plt()
    planes = [p for p in ("x", "y") if p in cmp["planes"]]
    fig, axes = plt.subplots(len(planes), 3, figsize=(13.6, 3.0 * len(planes) + 0.5), squeeze=False, gridspec_kw=dict(wspace=0.08, hspace=0.4))
    fd = _first_downstream(sel)
    for r, p in enumerate(planes):
        d = cmp["planes"][p]; name, L, col = PLANE[p]
        Mm, Ms = d["measured_norm"], d["model_norm"]
        for c, (A, t) in enumerate(((Mm, f"measured, {L} trims → {L} BPMs (column ÷ max)"), (Ms, "model (column ÷ max, measured polarity)"), (Mm - Ms, "measured − model"))):
            ax = axes[r, c]; im = ax.imshow(A, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto", interpolation="nearest")
            ax.set_title(t, loc="left", fontsize=9.5, fontweight="bold", color=INK)
            ax.set_xticks(range(sel.n_trim)); ax.set_xticklabels(sel.trim_labels, rotation=90, fontsize=7)
            if c == 0:
                ax.set_yticks(range(sel.n_bpm)); ax.set_yticklabels(sel.bpm_labels, fontsize=6.5); ax.set_ylabel(f"{name}: BPM")
            else:
                ax.set_yticks([])
            for j, k in enumerate(fd):
                ax.plot([j - 0.5, j + 0.5], [k - 0.5, k - 0.5], color="k", lw=0.8)
            ax.spines["top"].set_visible(True); ax.spines["right"].set_visible(True)
    cb = fig.colorbar(im, ax=axes, fraction=0.02, pad=0.01); cb.set_label("normalised response (per trim column)")
    return fig


def fig_trends(cmp: dict, sel, plane: str, fitted: np.ndarray | None = None, *, r_per_trim=None, model_label: str = "model"):
    """One panel per trim: measured (÷ column max, ±1σ) vs model.

    ``fitted`` (mm/A, e.g. ``OrmFitResult.fitted_block``) replaces the per-trim
    k-scaled model; pass the matching ``r_per_trim`` so the panel titles show
    the correlation of the curve that is drawn.
    """
    plt = _plt()
    d = cmp["planes"][plane]; name, L, col = PLANE[plane]
    A, E, R = d["aligned"], d["aligned_err"], d["model"]
    n = sel.n_trim; ncol = 5; nrow = int(np.ceil((n + 1) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(13, 1.75 * nrow + 0.5), sharex=True, sharey=True, squeeze=False, gridspec_kw=dict(wspace=0.08, hspace=0.45))
    x = np.arange(1, sel.n_bpm + 1); fd = _first_downstream(sel)
    for j in range(n):
        ax = axes.flat[j]; dn = sel.down[:, j] & np.isfinite(A[:, j])
        nm = np.nanmax(np.abs(A[dn, j])) if dn.any() else 1.0; nm = nm if nm > 0 else 1.0
        row = d["per_trim"][j]; k = row["k_Tm_per_A"]
        ax.axhline(0, color="#bbb", lw=0.7); ax.axvspan(0, fd[j] + 0.5, color="#f1f1f4", lw=0, zorder=0)
        ax.errorbar(x, A[:, j] / nm, yerr=np.where(np.isfinite(E[:, j]), E[:, j], 0) / nm, fmt="o", color=col, ms=3.2, elinewidth=0.8, capsize=0, zorder=3)
        mod = fitted[:, j] if fitted is not None else (k * R[:, j] if np.isfinite(k) else np.full(sel.n_bpm, np.nan))
        ax.plot(x[dn], mod[dn] / nm, "-", color=INK, lw=1.3, marker="s", ms=2.5, zorder=2)
        rj = row["r"] if r_per_trim is None else r_per_trim[j]
        rtxt = f"r = {abs(rj):.2f}" if np.isfinite(rj) else f"{row['n_down']} BPM"
        ax.set_title(f"{sel.trim_labels[j]} · {rtxt}", fontsize=8.5, loc="left", pad=2); ax.set_ylim(-1.25, 1.25); ax.set_xlim(0.3, sel.n_bpm + 0.7); ax.tick_params(labelsize=7)
        if np.isfinite(row["kick_mrad_per_A"]):
            ax.text(0.98, 0.04, f"{abs(row['kick_mrad_per_A']):.2f} mrad/A", transform=ax.transAxes, ha="right", va="bottom", fontsize=7, color=GREY)
    for ax in axes.flat[n:]:
        ax.axis("off")
    axes.flat[n].text(0.5, 0.5, f"points: measured ÷ column max (±1σ)\nline: {'fitted model' if fitted is not None else 'model'}, measured polarity\ngrey: BPMs upstream of the trim", ha="center", va="center", fontsize=8, color=GREY, transform=axes.flat[n].transAxes)
    fig.suptitle(f"{name} plane: measured orbit response vs {model_label}, one panel per trim", x=0.01, ha="left", fontsize=11, fontweight="bold", color=INK)
    return fig


def fig_calibration(cmp: dict, sel):
    """|k| per trim, kick per ampere, |r| per trim, and measured vs global-k model."""
    plt = _plt()
    fig, axes = plt.subplots(2, 2, figsize=(13, 6.3), gridspec_kw=dict(wspace=0.22, hspace=0.5)); xi = np.arange(sel.n_trim)
    for p in ("x", "y"):
        if p not in cmp["planes"]:
            continue
        d = cmp["planes"][p]; name, L, col = PLANE[p]; off = -0.18 if p == "x" else 0.18
        rows = d["per_trim"]; k = np.array([r["k_Tm_per_A"] for r in rows]) * 1e3; dk = np.array([r["dk"] for r in rows]) * 1e3
        kick = np.array([r["kick_mrad_per_A"] for r in rows]); rr = np.array([r["r"] for r in rows]); nd = np.array([r["n_down"] for r in rows])
        axes[0, 0].errorbar(xi + off, np.abs(k), yerr=dk, fmt="o", color=col, ms=4.5, label=f"{L} trims (global {abs(d['k_global_Tm_per_A'])*1e3:.2f})")
        axes[0, 0].axhline(abs(d["k_global_Tm_per_A"]) * 1e3, color=col, lw=1, ls="--")
        axes[0, 1].plot(xi + off, np.abs(kick), "o", color=col, ms=4.5, label=f"{L} trims")
        ok = nd > 2; axes[1, 0].bar(xi[ok] + off, np.abs(rr[ok]), width=0.34, color=col, label=f"{L} (median {np.nanmedian(np.abs(rr[ok])) if ok.any() else float('nan'):.2f})")
        m = sel.down & np.isfinite(d["aligned"]); axes[1, 1].plot((d["model"] * d["k_global_Tm_per_A"])[m], d["aligned"][m], ".", color=col, ms=3.5, alpha=0.7,
                                                                   label=f"{L}: r = {abs(d['r_global']):.2f}, residual {d['nrms_global']:.0%} rms")
    for ax, ttl, yl in ((axes[0, 0], "fitted calibration |k| = measured ÷ model, per trim", "|k| (mT·m per A)"), (axes[0, 1], "kick per ampere = |k| / Bρ at the trim", "|Δθ| (mrad per A)"),
                        (axes[1, 0], "|correlation| measured vs model, downstream BPMs", "|r|")):
        ax.set_title(ttl, loc="left", fontsize=10, fontweight="bold", color=INK); ax.set_ylabel(yl); ax.set_xticks(xi); ax.set_xticklabels(sel.trim_labels, rotation=90, fontsize=8); ax.grid(axis="y", color=GRID, lw=0.7); ax.legend(frameon=False, fontsize=8)
    axes[1, 0].set_ylim(0, 1.05)
    lim = max([abs(v) for a in axes[1, 1].get_lines() for v in np.concatenate([a.get_xdata(), a.get_ydata()])] + [1e-9])
    axes[1, 1].plot([-lim, lim], [-lim, lim], color="#999", lw=0.8, ls="--"); axes[1, 1].set_title("all downstream entries, measured vs model × global k", loc="left", fontsize=10, fontweight="bold", color=INK)
    axes[1, 1].set_xlabel("model × k_global (mm/A)"); axes[1, 1].set_ylabel("measured (mm/A)"); axes[1, 1].legend(frameon=False, fontsize=8); axes[1, 1].grid(color=GRID, lw=0.7)
    return fig


def fig_fit_quads(fit, model):
    """Fitted quad corrections with errors (fixed quads greyed)."""
    plt = _plt()
    fig, ax = plt.subplots(figsize=(13, 4.2)); g = (fit.g - 1) * 100; e = np.where(np.isfinite(fit.g_err), fit.g_err, 0) * 100; xi = np.arange(len(fit.quads))
    cols = ["#bbb" if f else RED for f in fit.g_fixed]
    ax.bar(xi, g, yerr=e, color=cols, width=0.72, error_kw=dict(elinewidth=1, ecolor=INK, capsize=2)); ax.axhline(0, color=INK, lw=0.8)
    ax.set_xticks(xi); ax.set_xticklabels(fit.quads, rotation=90, fontsize=8); ax.set_ylabel("gradient change (%)"); ax.grid(axis="y", color=GRID, lw=0.7)
    free = ~fit.g_fixed
    ax.set_title(f"Fitted quadrupole gradient corrections ({fit.stage}): rms {np.std(g[free]) if free.any() else 0:.1f} %, held fixed: {', '.join(sorted(fit.fixed_reason)) or 'none'}", loc="left", fontsize=10, fontweight="bold", color=INK)
    return fig


def fig_fit_residuals(fit, sel, brho):
    """Residual per trim before/after and the fitted kick per ampere for both planes."""
    plt = _plt()
    planes = [p for p in ("x", "y") if p in fit.metrics_after]
    fig, axes = plt.subplots(2, len(planes), figsize=(6.5 * len(planes), 6.0), squeeze=False, gridspec_kw=dict(wspace=0.2, hspace=0.55)); xi = np.arange(sel.n_trim)
    for c, p in enumerate(planes):
        name, L, col = PLANE[p]
        nA = np.array(fit.metrics_before[p]["nrms_per_trim"]) * 100; nD = np.array(fit.metrics_after[p]["nrms_per_trim"]) * 100
        ax = axes[0, c]; ax.bar(xi - 0.19, nA, width=0.36, color="#c9c9d1", label="before (trims only)"); ax.bar(xi + 0.19, nD, width=0.36, color=col, label=f"after ({fit.stage})")
        ax.set_xticks(xi); ax.set_xticklabels(sel.trim_labels, rotation=90, fontsize=8); ax.set_ylabel("residual rms / signal rms (%)"); ax.grid(axis="y", color=GRID, lw=0.7); ax.legend(frameon=False, fontsize=8)
        ax.set_title(f"{name} plane: residual per trim", loc="left", fontsize=10, fontweight="bold", color=INK)
        ax = axes[1, c]; kk = np.abs(fit.k[p]) / brho * 1e3
        ax.plot(xi, kk, "o", color=col, ms=5, label=f"{L} winding")
        other = "y" if p == "x" else "x"
        if other in fit.k:
            ax.plot(xi, np.abs(fit.k[other]) / brho * 1e3, "x", color=INK, ms=6, mew=1.4, label=f"{PLANE[other][1]} winding of the same trim")
        ax.set_xticks(xi); ax.set_xticklabels(sel.trim_labels, rotation=90, fontsize=8); ax.set_ylabel("|kick| (mrad per A)"); ax.grid(axis="y", color=GRID, lw=0.7); ax.legend(frameon=False, fontsize=8)
        ax.set_title(f"{L} trims: fitted kick per ampere", loc="left", fontsize=10, fontweight="bold", color=INK)
    return fig


def fig_residual_maps(fit, sel, cmp: dict):
    """(measured − model) ÷ column max, before (per-trim calibration only, from ``compare_orm``) and after the fit."""
    plt = _plt()
    planes = [p for p in ("x", "y") if p in fit.metrics_after and p in cmp["planes"]]
    fig, axes = plt.subplots(len(planes), 2, figsize=(9.6, 3.6 * len(planes) + 0.4), squeeze=False, gridspec_kw=dict(wspace=0.06, hspace=0.3))
    for r, p in enumerate(planes):
        d = cmp["planes"][p]; A = d["aligned"]; name, L, col = PLANE[p]
        k0 = np.array([row["k_Tm_per_A"] for row in d["per_trim"]])
        before = np.where(np.isfinite(k0)[None, :], k0[None, :] * d["model"], np.nan)
        for c, (Rm, t) in enumerate(((before, "before: trims only"), (fit.fitted_block(p), f"after: {fit.stage}"))):
            out = np.full_like(A, np.nan)
            for j in range(sel.n_trim):
                dn = sel.down[:, j] & np.isfinite(A[:, j]); m = np.nanmax(np.abs(A[dn, j])) if dn.any() else 1.0
                out[dn, j] = (A[dn, j] - Rm[dn, j]) / (m if m > 0 else 1.0)
            ax = axes[r, c]; im = ax.imshow(out, cmap="RdBu_r", vmin=-0.6, vmax=0.6, aspect="auto", interpolation="nearest")
            ax.set_title(f"{name}: {t}", loc="left", fontsize=9.5, fontweight="bold", color=INK); ax.set_xticks(range(sel.n_trim)); ax.set_xticklabels(sel.trim_labels, rotation=90, fontsize=7)
            if c == 0:
                ax.set_yticks(range(sel.n_bpm)); ax.set_yticklabels(sel.bpm_labels, fontsize=6.5)
            else:
                ax.set_yticks([])
            ax.spines["top"].set_visible(True); ax.spines["right"].set_visible(True)
    cb = fig.colorbar(im, ax=axes, fraction=0.025, pad=0.02); cb.set_label("(measured − model) ÷ column max")
    return fig
