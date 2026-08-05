"""Hofmann stability-chart overlay for the channel-tune trajectory.

Conventions (pinned to Hofmann, Franchetti, Boine-Frankenheim, Qiang,
Ryne, PRST-AB 6, 024202 (2003)):

* abscissa — depressed TUNE RATIO k_z/k_x (σ_l/σ_t);
* ordinate — TRANSVERSE tune depression k_x/k_0x (1 = zero current);
* the resonance-band structure is a FAMILY of charts parameterized by
  the emittance ratio ε_z/ε_x — the anisotropy resonances peak at
  rational tune ratios k_z/k_x = m/n (1/3, 1/2, 2/3, 1, 3/2, 2 …) with
  widths/growth rates that depend on (ε_z/ε_x, depression).

What is exact here and what is indicative
-----------------------------------------
The TRAJECTORY points (one per cell) are exact: they come from the
phase-probe channel tunes (``channel_phase_advance``), i.e. the
det-normalized monodromy of the solver's own tangent map.  The
resonance LINES at k_z/k_x = m/n are exact conditions.  The shaded
band WIDTHS are indicative only — the published growth-rate maps are
numerical solutions of the anisotropic-KV dispersion relations that
this module does not reproduce; the width heuristic below scales with
(1 − depression) and vanishes at zero current, which reproduces the
qualitative chart topology (bands grow toward strong space charge and
disappear at η → 1).  For quantitative stop-band edges consult the
published charts for the nearest ε_z/ε_x family (the ratio is
reported so the right family can be picked).

For SOLVED stop-band edges and per-cell growth rates, use the corrected
anisotropic dispersion solver instead:
:mod:`linac_gen.analysis.hofmann_dispersion` (branch functions + chart)
and :mod:`linac_gen.analysis.hofmann_stability` (per-cell driver).
"""
from __future__ import annotations

import math

import numpy as np

__all__ = ["hofmann_trajectory", "resonance_bands", "RESONANCE_RATIOS"]

#: The principal anisotropy resonance conditions k_z/k_x = m/n shown on
#: the published charts.
RESONANCE_RATIOS: tuple = (
    (1, 3), (1, 2), (2, 3), (1, 1), (3, 2), (2, 1),
)


def hofmann_trajectory(channel_out: dict, results=None) -> dict:
    """Per-cell (k_z/k_x, k_x/k_0x) trajectory from channel tunes.

    Parameters
    ----------
    channel_out : dict
        Output of :func:`analysis.phase_advance.channel_phase_advance`.
        Uncoupled: the x plane is the transverse reference (documented
        convention).  Coupled: the paired normal mode with the LARGER
        bare tune plays the transverse role (the solenoid-channel
        analogue of k_x) — mode identity is maintained by the channel
        pairing, and the choice is reported in ``transverse_label``.
    results : optional
        Probe-bearing results — used only to report the emittance ratio
        ε_z/ε_x (chart-family selector) from the RUN-ENTRANCE record
        (index 0; under acceleration this differs from the ratio at the
        period entry — indicative family selection only).

    Returns
    -------
    dict
        ``ratio`` — k_z/k_x per cell; ``depression`` — k_x/k_0x per
        cell; ``cells``; ``transverse_label`` ("x" or "mode II" …);
        ``emit_ratio`` — ε_z/ε_x at the run entrance (NaN when not
        derivable); NaN entries wherever a tune was unavailable.
    """
    cells = np.asarray(channel_out.get("cells", []), dtype=float)
    coupled = bool(channel_out.get("coupled_xy", False))

    if not coupled:
        kx_dep = np.asarray(channel_out.get("mu_x_dep_deg", []), float)
        kx_bare = np.asarray(channel_out.get("mu_x_bare_deg", []), float)
        label = "x"
    else:
        # Pick the mode with the larger (median) bare tune as the
        # transverse reference.
        mI = np.asarray(channel_out.get("mu_I_bare_deg", []), float)
        mII = np.asarray(channel_out.get("mu_II_bare_deg", []), float)
        medI = np.nanmedian(mI) if np.isfinite(mI).any() else -np.inf
        medII = np.nanmedian(mII) if np.isfinite(mII).any() else -np.inf
        if medII >= medI:
            kx_dep = np.asarray(channel_out.get("mu_II_dep_deg", []), float)
            kx_bare = mII
            label = "mode II"
        else:
            kx_dep = np.asarray(channel_out.get("mu_I_dep_deg", []), float)
            kx_bare = mI
            label = "mode I"

    kz_dep = np.asarray(channel_out.get("mu_z_dep_deg", []), float)

    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where((kx_dep > 0) & np.isfinite(kz_dep),
                         kz_dep / kx_dep, np.nan)
        depression = np.where((kx_bare > 0) & np.isfinite(kx_dep),
                              kx_dep / kx_bare, np.nan)

    # Chart-family selector ε_z/ε_x, from the run-entrance record —
    # both series are in mm·mrad (emit_z_mmmrad is the recorder's
    # (z, z′)-converted longitudinal emittance).
    emit_ratio = float("nan")
    if results is not None:
        try:
            ez_raw = getattr(results, "emit_z_mmmrad", None)
            ex_raw = getattr(results, "emit_x", None)
            # list() before truthiness — `arr or []` raises on ndarrays.
            ez_list = list(ez_raw) if ez_raw is not None else []
            ex_list = list(ex_raw) if ex_raw is not None else []
            if ez_list and ex_list and ex_list[0] > 0:
                emit_ratio = float(ez_list[0] / ex_list[0])
        except Exception:                                    # noqa: BLE001
            pass

    return {
        "cells": cells,
        "ratio": ratio,
        "depression": depression,
        "transverse_label": label,
        "emit_ratio": emit_ratio,
    }


def resonance_bands(emit_ratio: float = 1.0) -> list:
    """The principal resonance lines with INDICATIVE widths.

    Returns a list of dicts ``{ratio, label, width(depression)}`` where
    ``width`` is a callable giving the indicative half-width of the
    band at a given transverse depression (0..1).  The heuristic
    w = w₀·(1 − depression) with w₀ growing toward equal tunes
    reproduces the qualitative topology of the published charts (bands
    close at η → 1, widest structure near k_z/k_x = 1); quantitative
    edges must come from the published family for this ε_z/ε_x.
    """
    out = []
    er = emit_ratio if (emit_ratio and math.isfinite(emit_ratio)) else 1.0
    for m, n in RESONANCE_RATIOS:
        r = m / n
        # Main mixed mode (1:1) carries the widest band; higher-order
        # rationals get progressively narrower.  ε ratio skews the
        # width mildly (larger ε_z/ε_x widens the low-ratio side).
        order = m + n
        w0 = 0.35 / order * (1.0 + 0.25 * math.log(max(er, 1e-3)))
        w0 = max(w0, 0.02)

        def _width(depression, _w0=w0):
            d = min(max(depression, 0.0), 1.0)
            return _w0 * (1.0 - d)

        out.append({"ratio": r, "label": f"{m}/{n}", "width": _width})
    return out
