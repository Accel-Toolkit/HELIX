"""Per-cell corrected Hofmann stability analysis driven by HELIX runs.

Bridges the phase-probe channel model to the corrected anisotropic
dispersion solver (:mod:`linac_gen.analysis.hofmann_dispersion`):

* :func:`hofmann_coordinates` — per-cell chart coordinates
  ``(R, Y, eps_z/eps_x)`` from probe-bearing envelope results.  ``R`` is the
  DEPRESSED tune ratio ``nu_z/nu_x = mu_z_dep/mu_x_dep`` — algebraically
  identical to ``(k_z0 eta_z)/(k_x0 eta_x)``, the form used by the source
  manuscript's TraceWin loader; ``Y = eta_x`` is the transverse tune
  depression; the emittance ratio is GEOMETRIC, sampled per cell at each
  repeat-span entry (``emit_z_mmmrad`` is converted per record with the
  local reference frequency, so the sampling is FREQ-jump safe).
* :func:`hofmann_stability` — per-cell, per-branch growth rates
  ``gamma/nu_0x`` with the manuscript's perturbative-validity gate
  ``S^2 <= 10`` and instability flags at ``gamma/nu_0x > 0.01``.
* :func:`anisotropy_margin` — how much MORE ``eps_z/eps_x`` each cell could
  absorb before a higher-order (l=3, l=4 even) channel first activates.

Guards (refuse-with-``reason``, never raise, so the GUI/MCP surfaces can
render the explanation):

* x-y coupled lattices — the channel tunes are normal modes I/II, not the
  x/z planes of Hofmann's dispersion relations; relabelling a normal mode
  as "x" would be a different physics claim, so the analysis refuses.
* DC/continuous beams — no longitudinal tune, chart undefined.
* ``eta_x >= 1`` cells (no net tune depression) are kept but warned about:
  the solver treats them as trivially stable, mirroring the source package.

``fold_risk`` marks cells whose bare advance approaches the 180 deg
principal-value fold of ``mu_folded`` — a true per-cell advance above
180 deg would be silently reflected, corrupting both ``R`` and ``Y``.
"""
from __future__ import annotations

import warnings

import numpy as np

from linac_gen.analysis.hofmann_dispersion import (
    ChartAborted,
    DispersionSolver,
    map_to_pre_grid,
    solver_fingerprint,
)
from linac_gen.analysis.phase_advance import (
    channel_phase_advance,
    element_record_span,
)

__all__ = [
    "S2_GATE",
    "FLAG_THRESHOLD",
    "FOLD_RISK_DEG",
    "hofmann_coordinates",
    "hofmann_stability",
    "anisotropy_margin",
    "per_branch_growth",
]

#: Perturbative-validity gate on the space-charge parameter S^2 (manuscript
#: Sec. "validity"): growth rates at S^2 above this are extrapolations.
S2_GATE = 10.0

#: A cell is flagged unstable when the combined growth gamma/nu_0x exceeds
#: this (matches the source package's screening threshold).
FLAG_THRESHOLD = 0.01

#: Bare per-cell advance (deg) above which the mu_folded principal value
#: [0, 180] may have silently reflected a true advance > 180 deg.
FOLD_RISK_DEG = 170.0


def _nan(n: int) -> np.ndarray:
    return np.full(n, np.nan)


def hofmann_coordinates(results, period, *, channel: dict | None = None) -> dict:
    """Per-cell Hofmann chart coordinates from a probe-bearing run.

    Parameters
    ----------
    results : EnvelopeResults
        Probe-bearing results (``EnvelopeSolver(..., phase_probe=True)`` or
        the ``run_phase_probe`` helper).
    period : PeriodicStructure
        The lattice period whose repeats define the cells.
    channel : dict, optional
        Precomputed output of :func:`channel_phase_advance` for the same
        ``(results, period)`` — avoids recomputing the per-cell monodromies.

    Returns
    -------
    dict
        ``cells``; per-cell arrays ``R``, ``Y``, ``eps_ratio``,
        ``kx0_deg``, ``kz0_deg``, ``eta_x``, ``eta_z``, ``fold_risk``
        (bool); ``transverse_label``; ``reason`` — ``None`` when usable,
        else a human-readable refusal string (arrays are NaN then).
    """
    if channel is None:
        channel = channel_phase_advance(results, period)

    cells = np.asarray(channel.get("cells", []), dtype=float)
    n = cells.size
    base = {
        "cells": cells,
        "R": _nan(n), "Y": _nan(n), "eps_ratio": _nan(n),
        "kx0_deg": _nan(n), "kz0_deg": _nan(n),
        "eta_x": _nan(n), "eta_z": _nan(n),
        "fold_risk": np.zeros(n, dtype=bool),
        "transverse_label": "x",
        "reason": None,
    }

    if bool(channel.get("coupled_xy", False)):
        base["reason"] = (
            "x-y coupled lattice: channel tunes are normal modes I/II, not "
            "the x/z planes of Hofmann's dispersion relations — refusing "
            "rather than mislabeling a normal mode as \"x\"."
        )
        return base
    if bool(getattr(results, "continuous", False)):
        base["reason"] = (
            "DC/continuous beam: no longitudinal tune — the Hofmann chart "
            "coordinate R = nu_z/nu_x is undefined."
        )
        return base

    kx_bare = np.asarray(channel.get("mu_x_bare_deg", _nan(n)), float)
    kx_dep = np.asarray(channel.get("mu_x_dep_deg", _nan(n)), float)
    kz_bare = np.asarray(channel.get("mu_z_bare_deg", _nan(n)), float)
    kz_dep = np.asarray(channel.get("mu_z_dep_deg", _nan(n)), float)
    eta_x = np.asarray(channel.get("eta_x", _nan(n)), float)
    eta_z = np.asarray(channel.get("eta_z", _nan(n)), float)

    if not np.isfinite(kz_dep).any():
        base["reason"] = (
            "no finite longitudinal channel tune in any cell — the chart "
            "coordinate R = nu_z/nu_x is undefined."
        )
        return base

    # R = nu_z/nu_x from the DEPRESSED per-cell tunes (== kz0*eta_z /
    # (kx0*eta_x) identically, since mu_dep = mu_bare * eta per plane).
    with np.errstate(divide="ignore", invalid="ignore"):
        R = np.where((kx_dep > 0) & np.isfinite(kz_dep), kz_dep / kx_dep,
                     np.nan)

    n_eta_ge1 = int(np.sum(eta_x[np.isfinite(eta_x)] >= 1.0))
    if n_eta_ge1:
        warnings.warn(
            f"{n_eta_ge1}/{n} cells have eta_x >= 1 (no space-charge tune "
            "depression); the dispersion solver treats them as trivially "
            "stable.",
            stacklevel=2,
        )

    # Per-cell GEOMETRIC emittance ratio eps_z/eps_x at each repeat-span
    # entry.  Spans are filtered exactly like channel_phase_advance filters
    # its cell list, so index k here is the same cell as channel index k.
    eps_ratio = _nan(n)
    emit_z = np.asarray(getattr(results, "emit_z_mmmrad", None) or [], float)
    emit_x = np.asarray(getattr(results, "emit_x", None) or [], float)
    maps_dep = getattr(results, "element_maps_dep", None) or []
    spans = list(period.spans())
    if maps_dep:
        spans = [sp for sp in spans if sp[1] <= len(maps_dep)]
    if emit_z.size and emit_x.size:
        for k, (a, b) in enumerate(spans[:n]):
            r0, _ = element_record_span(results, a, b)
            if r0 < emit_z.size and r0 < emit_x.size and emit_x[r0] > 0:
                eps_ratio[k] = emit_z[r0] / emit_x[r0]

    base.update({
        "R": R,
        "Y": eta_x,
        "eps_ratio": eps_ratio,
        "kx0_deg": kx_bare,
        "kz0_deg": kz_bare,
        "eta_x": eta_x,
        "eta_z": eta_z,
        "fold_risk": (kx_bare > FOLD_RISK_DEG) | (kz_bare > FOLD_RISK_DEG),
    })
    return base


def per_branch_growth(
    solver: DispersionSolver,
    R: float,
    Y: float,
    eps_ratio: float,
    include_l4_odd: bool = False,
) -> tuple[dict, float]:
    """Per-branch scaled growth ``gamma/nu_0x`` at one chart point.

    Returns ``({'l2', 'l3_even', 'l3_odd', 'l4_even'[, 'l4_odd']}, S2)`` —
    a faithful port of the source package's per-period driver: higher-order
    branches are masked to 0 at ``Y >= 1`` or ``S2 ~ 0`` exactly as there.
    """
    a, e, sp2, _, gs = map_to_pre_grid(np.array([R]), np.array([Y]),
                                       eps_ratio)
    a, e, sp2, gs = a.item(), e.item(), sp2.item(), gs.item()
    out = {"l2": float(solver.growth_rate(R=R, Y=Y, eps_ratio=eps_ratio,
                                          l_modes=(2,)))}
    branches = ["l3_even", "l3_odd", "l4_even"]
    if include_l4_odd:
        branches.append("l4_odd")
    if Y >= 1.0 or sp2 < 1e-15:
        out.update({b: 0.0 for b in branches})
    else:
        for b in branches:
            out[b] = float(solver.solve_branch_raw(b, a, e, sp2) * gs)
    return out, sp2


def hofmann_stability(
    results, period, *,
    l_modes=(2, 3, 4),
    include_l4_odd: bool = False,
    threshold: float = FLAG_THRESHOLD,
    s2_gate: float = S2_GATE,
    channel: dict | None = None,
    should_stop=None,
) -> dict:
    """Per-cell corrected Hofmann analysis of a probe-bearing run.

    Returns the :func:`hofmann_coordinates` keys plus per-cell arrays
    ``g_l2``, ``g_l3_even``, ``g_l3_odd``, ``g_l4_even``, ``g_l4_odd``
    (NaN unless ``include_l4_odd``), ``g_combined`` (max over the branches
    selected by ``l_modes``), ``S2``, ``valid`` (``S2 <= s2_gate``),
    ``flagged`` (combined > threshold AND valid), ``flagged_extrap``
    (combined > threshold, S2 above the gate — an extrapolated flag), and
    scalars ``n_cells``, ``n_valid``, ``n_flagged``, ``worst_cell``,
    ``worst_growth``, ``threshold``, ``s2_gate``, ``solver_fingerprint``,
    ``reason``.

    Flag-count convention: ``n_flagged`` counts VALIDITY-GATED flags only,
    which is stricter than the source manuscript driver's raw
    ``combined > threshold`` count — that ungated count is
    ``(flagged | flagged_extrap).sum()`` here.
    """
    if channel is None:
        channel = channel_phase_advance(results, period)
    coords = hofmann_coordinates(results, period, channel=channel)
    n = coords["cells"].size

    out = dict(coords)
    for key in ("g_l2", "g_l3_even", "g_l3_odd", "g_l4_even", "g_l4_odd",
                "g_combined", "S2"):
        out[key] = _nan(n)
    out["valid"] = np.zeros(n, dtype=bool)
    out["flagged"] = np.zeros(n, dtype=bool)
    out["flagged_extrap"] = np.zeros(n, dtype=bool)
    out.update({
        "n_cells": n, "n_valid": 0, "n_flagged": 0,
        "worst_cell": float("nan"), "worst_growth": float("nan"),
        "threshold": float(threshold), "s2_gate": float(s2_gate),
        "solver_fingerprint": solver_fingerprint(),
    })
    if coords["reason"] is not None:
        return out

    solver = DispersionSolver()
    R, Y, EPS = coords["R"], coords["Y"], coords["eps_ratio"]
    for k in range(n):
        if should_stop is not None and should_stop():
            raise ChartAborted("per-cell Hofmann analysis aborted")
        if not (np.isfinite(R[k]) and np.isfinite(Y[k])
                and np.isfinite(EPS[k]) and EPS[k] > 0):
            continue
        g, s2 = per_branch_growth(solver, float(R[k]), float(Y[k]),
                                  float(EPS[k]),
                                  include_l4_odd=include_l4_odd)
        out["g_l2"][k] = g["l2"]
        out["g_l3_even"][k] = g["l3_even"]
        out["g_l3_odd"][k] = g["l3_odd"]
        out["g_l4_even"][k] = g["l4_even"]
        if include_l4_odd:
            out["g_l4_odd"][k] = g["l4_odd"]
        parts = []
        if 2 in l_modes:
            parts.append(g["l2"])
        if 3 in l_modes:
            parts += [g["l3_even"], g["l3_odd"]]
        if 4 in l_modes:
            parts.append(g["l4_even"])
            if include_l4_odd:
                parts.append(g["l4_odd"])
        out["g_combined"][k] = max(parts) if parts else 0.0
        out["S2"][k] = s2
        out["valid"][k] = s2 <= s2_gate
        hot = out["g_combined"][k] > threshold
        out["flagged"][k] = hot and out["valid"][k]
        out["flagged_extrap"][k] = hot and not out["valid"][k]

    out["n_valid"] = int(out["valid"].sum())
    out["n_flagged"] = int(out["flagged"].sum())
    gv = np.where(out["valid"], out["g_combined"], np.nan)
    if np.isfinite(gv).any():
        k_worst = int(np.nanargmax(gv))
        out["worst_cell"] = float(coords["cells"][k_worst])
        out["worst_growth"] = float(gv[k_worst])
    return out


def anisotropy_margin(
    results, period, *,
    gamma_th: float = FLAG_THRESHOLD,
    s2_max: float = S2_GATE,
    eps_lo: float = 1.0,
    eps_hi: float = 9.0,
    eps_step: float = 0.05,
    channel: dict | None = None,
    coords: dict | None = None,
) -> dict:
    """Per-cell anisotropy margin to higher-order onset.

    For every cell with ``Y < 1`` and design ``S2 <= s2_max``, holds
    ``(R, Y)`` fixed, raises ``eps_z/eps_x`` over
    ``[eps_lo, eps_hi]`` (staying inside the perturbative domain), and
    records the smallest ratio at which a higher-order (l=3 or l=4 even)
    growth first exceeds ``gamma_th`` — the source package's margin scan,
    evaluated with the batched solver.  The batch root-finder's scan grid
    differs from the source package's scalar one at the last ulp
    (~1e-11 relative on the growth); every probed onset/seam output is
    nonetheless bitwise identical to the paper loop — a divergence would
    need a scan-point growth within ~1e-12 of ``gamma_th``.

    Returns per-cell arrays ``onset_eps`` (NaN = no onset in domain),
    ``margin`` (= onset / design ratio), ``is_seam`` (onset caused by the
    eta-hat = 1 coordinate flip, S2 jumping > 2x across one step),
    ``s2_at_onset``, ``s2_jump``, plus summary scalars
    ``n_onsets``, ``n_smooth``, ``n_seam``, ``smallest_smooth_margin``,
    ``smallest_smooth_margin_cell``, ``earliest_smooth_onset_eps``,
    ``earliest_smooth_onset_cell``, ``gamma_th``, ``s2_max``, ``reason``.
    """
    if coords is None:
        if channel is None:
            channel = channel_phase_advance(results, period)
        coords = hofmann_coordinates(results, period, channel=channel)
    n = coords["cells"].size

    out = {
        "cells": coords["cells"],
        "onset_eps": _nan(n), "margin": _nan(n),
        "is_seam": np.zeros(n, dtype=bool),
        "s2_at_onset": _nan(n), "s2_jump": _nan(n),
        "gamma_th": float(gamma_th), "s2_max": float(s2_max),
        "reason": coords["reason"],
    }
    summary_nan = {
        "n_onsets": 0, "n_smooth": 0, "n_seam": 0,
        "smallest_smooth_margin": float("nan"),
        "smallest_smooth_margin_cell": float("nan"),
        "earliest_smooth_onset_eps": float("nan"),
        "earliest_smooth_onset_cell": float("nan"),
    }
    out.update(summary_nan)
    if coords["reason"] is not None:
        return out

    solver = DispersionSolver()
    eps_scan = np.arange(eps_lo, eps_hi + eps_step / 2.0, eps_step)
    R, Y, EPS = coords["R"], coords["Y"], coords["eps_ratio"]

    for k in range(n):
        if not (np.isfinite(R[k]) and np.isfinite(Y[k])
                and np.isfinite(EPS[k]) and EPS[k] > 0):
            continue
        Rk, Yk, eps0 = float(R[k]), float(Y[k]), float(EPS[k])
        if Yk >= 1.0:
            continue
        s2_design = map_to_pre_grid(
            np.array([Rk]), np.array([Yk]), eps0)[2].item()
        if s2_design > s2_max:
            continue                      # non-perturbative at design
        s2_scan = map_to_pre_grid(
            np.full_like(eps_scan, Rk), np.full_like(eps_scan, Yk),
            eps_scan)[2]
        m = int(np.argmax(s2_scan > s2_max)) if (s2_scan > s2_max).any() \
            else eps_scan.size            # stay inside the domain
        if m == 0:
            continue
        ho = solver.growth_rate_batch(
            np.full(m, Rk), np.full(m, Yk), eps_scan[:m], l_modes=(3, 4))
        hot = ho > gamma_th
        if not hot.any():
            continue
        j = int(np.argmax(hot))
        s2_prev = s2_scan[j - 1] if j > 0 else s2_scan[0]
        s2_jump = float(s2_scan[j] / max(float(s2_prev), 1e-9))
        out["onset_eps"][k] = float(eps_scan[j])
        out["margin"][k] = float(eps_scan[j]) / eps0
        out["is_seam"][k] = s2_jump > 2.0
        out["s2_at_onset"][k] = float(s2_scan[j])
        out["s2_jump"][k] = s2_jump

    have = np.isfinite(out["onset_eps"])
    smooth = have & ~out["is_seam"]
    out["n_onsets"] = int(have.sum())
    out["n_smooth"] = int(smooth.sum())
    out["n_seam"] = int((have & out["is_seam"]).sum())
    if smooth.any():
        margins = np.where(smooth, out["margin"], np.nan)
        onsets = np.where(smooth, out["onset_eps"], np.nan)
        k_small = int(np.nanargmin(margins))
        k_early = int(np.nanargmin(onsets))
        out["smallest_smooth_margin"] = float(margins[k_small])
        out["smallest_smooth_margin_cell"] = float(coords["cells"][k_small])
        out["earliest_smooth_onset_eps"] = float(onsets[k_early])
        out["earliest_smooth_onset_cell"] = float(coords["cells"][k_early])
    return out
