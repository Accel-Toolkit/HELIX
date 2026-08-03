"""Attach a vane-geometry gradient profile to a lattice's RFQ cells.

Usage::

    from linac_gen.io.rfq_geometry_helper import apply_rfq_geometry
    lat, meta = parse_tracewin("linac.dat")
    apply_rfq_geometry(lat, "pxie-rfq.vane")     # opt-in refinement
    Simulation(lat, beam).run()

Without this call nothing changes — the RFQ cells keep the classic
two-term card kick, bit-identical to previous releases.  With it, the
transverse quad strength follows the measured real-vane gradient
profile (see rfq_geometry_profile.py for the physics and validation).
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Union

import numpy as np

from linac_gen.core.lattice import Lattice
from linac_gen.elements.rfq_cell import RfqCell
from linac_gen.elements.rfq_geometry_profile import (
    RfqGeometryProfile, build_rfq_geometry_profile)

_log = logging.getLogger(__name__)

# numpy >= 2.0 removed np.trapz (house pattern: analysis/phase_advance.py)
_trapz = getattr(np, "trapezoid", None) or np.trapz

#: modes for the y-plane gradient (2026-08-02 identical-beam study,
#: revised 2026-08-03 loss-location campaign):
#:  "antisym"   — gy = −gx: best envelope fidelity in BOTH planes and
#:                the most Toutatis-like loss profile (default);
#:  "per_plane" — gy from the y-plane solve (= −G+D signed): best total
#:                transmission on the PXIE benchmark, but σ_y runs
#:                8–10 % low.  The two bracket Toutatis.
#:  "per_plane_dflip" — EXPERIMENTAL (2026-08-03): per-plane split with
#:                the axisymmetric defocus component D = (gx+gy)/2 sign-
#:                corrected and scaled by ``d_scale``:
#:                    gx' =  G − d_scale·D,   gy' = −G − d_scale·D,
#:                G = (gx−gy)/2.  Rationale: under the per-cell phase
#:                reset the armed D channel accumulates RF *focusing*
#:                where the analytic defocus (card C2 term, cos-clocked)
#:                requires *defocusing* — measured on PXIE: wrong sign
#:                in 100 % of cells at 0.81× magnitude, making
#:                "per_plane" transmission agreement compensatory
#:                (right totals, wrong loss locations).  With
#:                d_scale = 1/0.81 ≈ 1.23 the 5 mA identical-beam loss
#:                centroid lands exactly on Toutatis (1.115 m) and the
#:                σ_y offset vanishes; total transmission is then
#:                honest-but-low pending the entrance-gap treatment.
#:                See the manual's RFQ "Known issues" section.
_MODES = ("antisym", "per_plane", "per_plane_dflip")


def apply_rfq_geometry(lattice: Lattice,
                       vane: Union[str, Path, RfqGeometryProfile],
                       *,
                       mode: str = "antisym",
                       d_scale: float = 1.0,
                       margin_mm: float = 2.0,
                       boundary_losses: bool = True,
                       wall_mm: float = 6.0,
                       **build_kwargs) -> int:
    """Arm every contiguous RfqCell chain with the geometry profile.

    Parameters
    ----------
    vane : path to a TraceWin ``.vane`` file, or a prebuilt
        :class:`RfqGeometryProfile` (e.g. from a previous call — the
        disk cache makes repeat calls cheap either way).
    mode : "antisym" (default), "per_plane" or "per_plane_dflip" — see
        ``_MODES``.
    d_scale : defocus-channel scale for ``mode="per_plane_dflip"``
        only (ignored, with a warning, for the other modes).  1.0
        reproduces the pure sign flip; the arm-time audit's reported
        |impulse| ratio suggests the analytic value (PXIE: ≈1.23).
    margin_mm : profile overhang kept on each side of a cell so the
        per-substep interpolation never clamps mid-cell.
    boundary_losses : also switch the cells' loss model from the
        two-term box to the real quadrant boundary (tip arcs +
        electrode bodies + chamber wall, open corners) — what Toutatis
        kills on.  Default True.
    wall_mm : chamber wall radius for the boundary model.  TraceWin
        carries this in the PROJECT settings, not the deck ("Wall
        radius aperture"); 6.0 mm is the PXIE value.

    Returns the number of RfqCells armed (0 if the lattice has none).
    """
    if mode not in _MODES:
        raise ValueError(f"mode must be one of {_MODES}, got {mode!r}")
    if not (np.isfinite(d_scale) and d_scale > 0.0):
        raise ValueError(f"d_scale must be finite and > 0, "
                         f"got {d_scale!r}")
    if d_scale != 1.0 and mode != "per_plane_dflip":
        _log.warning("apply_rfq_geometry: d_scale=%.3g is only used by "
                     "mode='per_plane_dflip' -- ignored for mode=%r",
                     d_scale, mode)

    cells = [el for el in lattice.elements if isinstance(el, RfqCell)]
    if not cells:
        _log.warning("apply_rfq_geometry: lattice has no RfqCell "
                     "elements -- nothing to do")
        return 0

    # the profile's z axis assumes ONE contiguous cell chain; a finite-
    # length element between RFQ cells would silently shift every
    # downstream cell off its true vane position -- refuse instead
    # (adversarial review 2026-08-02, finding 5)
    idx = [i for i, el in enumerate(lattice.elements)
           if isinstance(el, RfqCell)]
    for el in lattice.elements[idx[0]:idx[-1] + 1]:
        if not isinstance(el, RfqCell) and getattr(el, "length", 0.0) > 1e-6:
            _log.warning(
                "apply_rfq_geometry: RFQ cell chain is interrupted by "
                "'%s' (%.3g mm) -- the vane profile cannot be aligned; "
                "NOT arming.  Arm contiguous segments separately.",
                getattr(el, "name", el), el.length)
            return 0

    # cell edges in the RFQ-local coordinate (vane files share it:
    # z = 0 at the first cell's entrance)
    lengths = np.array([c.length for c in cells])
    edges = np.concatenate([[0.0], np.cumsum(lengths)])

    if isinstance(vane, RfqGeometryProfile):
        prof = vane
    else:
        # plateau for normalisation: unmodulated cells (m ≈ 1) away
        # from the radial matcher and the exit cells
        m_arr = np.array([c.modulation for c in cells])
        flat = np.where(np.abs(m_arr - 1.0) < 0.02)[0]
        flat = flat[(flat >= 4) & (flat <= len(cells) - 5)]
        plateau = None
        if flat.size >= 10:
            plateau = (float(edges[flat[0]]),
                       float(edges[flat[min(flat.size, 80) - 1] + 1]))
        prof = build_rfq_geometry_profile(vane, plateau_z_mm=plateau,
                                          **build_kwargs)

    # wall sanity: the chamber-wall radius is a TW PROJECT setting, not
    # a deck quantity -- if it does not clear this machine's own median
    # vane aperture, it is the wrong machine's wall: disable it loudly
    # rather than kill in-aperture particles (adversarial review
    # 2026-08-02, finding 2).
    if boundary_losses and wall_mm > 0 \
            and prof.median_aperture_mm is not None \
            and wall_mm < 1.02 * prof.median_aperture_mm:
        _log.warning(
            "apply_rfq_geometry: wall_mm=%.2f does not clear the vane "
            "median aperture %.2f mm -- wall test DISABLED; pass "
            "wall_mm= explicitly for this machine.",
            wall_mm, prof.median_aperture_mm)
        wall_mm = 0.0

    gy_glob = -prof.gx if mode == "antisym" else prof.gy
    vane_end = (prof.vane_z_mm[1] if prof.vane_z_mm is not None
                else float(prof.z_mm[-1]))
    if abs(edges[-1] - vane_end) > 50.0:
        _log.warning(
            "apply_rfq_geometry: RFQ card length %.1f mm vs vane table "
            "%.1f mm -- check that this vane file belongs to this deck",
            edges[-1], vane_end)

    # the kick prefactor uses each CARD's V/r0^2 while gx is normalised
    # once globally -- compensate per cell so decks with varying r0 get
    # G_true = gx * G_plateau, not gx * G_plateau * (r0_ref/r0_cell)^2
    # (adversarial review 2026-08-02, finding 4; inert when r0 constant)
    r0s = np.array([c.r0_mm for c in cells if c.r0_mm > 0])
    r0_ref = float(np.median(r0s)) if r0s.size else 0.0

    for k, cell in enumerate(cells):
        z0, z1 = edges[k] - margin_mm, edges[k + 1] + margin_mm
        sel = (prof.z_mm >= z0) & (prof.z_mm <= z1)
        if sel.sum() < 2:
            # cell outside the solved span (shouldn't happen for a
            # matching vane file) -- leave it on the card model
            _log.warning("apply_rfq_geometry: no profile coverage for "
                         "cell %d (z %.1f-%.1f mm) -- card kick kept",
                         k, edges[k], edges[k + 1])
            continue
        comp = ((cell.r0_mm / r0_ref) ** 2
                if (r0_ref > 0 and cell.r0_mm > 0) else 1.0)
        cell._geom_z = (prof.z_mm[sel] - edges[k]).copy()
        if mode == "per_plane_dflip":
            g_quad = 0.5 * (prof.gx[sel] - prof.gy[sel])
            d_axi = 0.5 * (prof.gx[sel] + prof.gy[sel])
            cell._geom_gx = (g_quad - d_scale * d_axi) * comp
            cell._geom_gy = (-g_quad - d_scale * d_axi) * comp
        else:
            cell._geom_gx = prof.gx[sel] * comp
            cell._geom_gy = gy_glob[sel] * comp
        if boundary_losses:
            cell._geom_boundary = True
            cell._wall_mm = float(wall_mm)
    armed = sum(1 for c in cells if c._geom_z is not None)
    _log.info("apply_rfq_geometry: %d/%d RFQ cells armed with the "
              "vane-geometry profile (mode=%s)", armed, len(cells), mode)
    # Audit once per in-memory lattice AND mode (the auto path re-arms
    # on every MP run via the scoped disarm; repeated identical
    # warnings would be noise).  A diagnostic must never be able to
    # change arming semantics: any audit failure is reported and
    # swallowed.
    if armed and getattr(lattice, "_rfq_dchannel_audited", None) != mode:
        lattice._rfq_dchannel_audited = mode
        try:
            _warn_defocus_channel(cells, mode)
        except Exception:
            _log.warning("apply_rfq_geometry: defocus-channel audit "
                         "failed (arming is unaffected)", exc_info=True)
    return armed


def _audit_defocus_channel(cells):
    """Score the armed profile's axisymmetric (D) channel per cell.

    For every armed accelerating cell, accumulate what the tracking
    clock actually applies from the axisymmetric component,
    ``I_sin = ∫ D(z)·S(z)·sin(φs + 180°·z/L) dz`` with
    D = (gx+gy)/2, against the analytic RF-defocus impulse of the card
    model (C2 term, cos-clocked) in the same normalised units.  A
    correct channel agrees in SIGN with the analytic defocus; the
    shipped "per_plane" registration measured 0 % agreement at 0.81×
    magnitude on PXIE (2026-08-03 loss-location campaign).

    Returns ``(n_scored, frac_sign_agree, median_ratio)`` —
    ``(n, None, None)`` when fewer than 10 cells could be scored, and
    ``median_ratio`` may be ~0 for D-free (ideal-quad or antisym)
    profiles.
    """
    from linac_gen.elements.rfq_coefficients import (type_coeffs,
                                                     tw_calibration)
    i_sin, i_card = [], []
    for c in cells:
        if c._geom_z is None or abs(int(c.cell_type)) != 2:
            continue
        if c.length <= 0 or c.r0_mm <= 0 or c.voltage_V == 0:
            continue
        z = np.linspace(0.0, c.length, 101)
        gx = np.interp(z, c._geom_z, c._geom_gx)
        gy = np.interp(z, c._geom_z, c._geom_gy)
        d_axi = 0.5 * (gx + gy)
        phi = np.deg2rad(c.phi_s_deg + 180.0 * z / c.length)
        quad_c, accel_c = tw_calibration(c.A10)
        basefac = 2.0 * quad_c * (c.voltage_V / (c.r0_mm * 1e-3) ** 2)
        c2_arr = np.empty_like(z)
        s_arr = np.empty_like(z)
        for j, zz in enumerate(z):
            _c1, c2, sv, _c3 = type_coeffs(c.cell_type, c.type_prev,
                                           c.type_next, zz / c.length)
            c2_arr[j], s_arr[j] = c2, sv
        defoc_norm = ((np.pi / (c.length * 1e-3)) ** 2
                      * (accel_c * c.A10 * c.voltage_V / 2.0)
                      * c2_arr / basefac)
        card = -_trapz(defoc_norm * np.cos(phi), z)
        if abs(card) < 1e-9:
            continue
        i_sin.append(_trapz(d_axi * s_arr * np.sin(phi), z))
        i_card.append(card)
    if len(i_sin) < 10:
        return len(i_sin), None, None
    i_sin = np.array(i_sin)
    i_card = np.array(i_card)
    frac = float(np.mean(np.sign(i_sin) == np.sign(i_card)))
    med = float(np.median(np.abs(i_sin) / np.abs(i_card)))
    return len(i_sin), frac, med


def _warn_defocus_channel(cells, mode):
    """Arm-time honesty check — see _audit_defocus_channel."""
    n, frac, med = _audit_defocus_channel(cells)
    if frac is None:
        return
    if med < 0.05:
        _log.info(
            "apply_rfq_geometry: the armed profile carries essentially "
            "no net axisymmetric RF-defocus channel (median |D|/card "
            "impulse ratio %.3f over %d cells%s).",
            med, n,
            " -- mode='antisym' drops D by construction"
            if mode == "antisym" else "")
    elif mode == "per_plane_dflip":
        if frac < 0.5:
            _log.warning(
                "apply_rfq_geometry: per_plane_dflip audit UNEXPECTED "
                "-- the flipped defocus channel still opposes the "
                "analytic sign in %.0f %% of %d cells.  This deck's "
                "profile registration may already be correct; the flip "
                "would then double-invert.  Verify against a reference "
                "before trusting results.", 100 * (1 - frac), n)
        else:
            _log.info(
                "apply_rfq_geometry: per_plane_dflip defocus-channel "
                "sign verified against the analytic RF defocus "
                "(%.0f %% of %d cells agree; |impulse| ratio %.2f).",
                100 * frac, n, med)
    elif frac < 0.5:
        _log.warning(
            "apply_rfq_geometry: KNOWN ISSUE -- the armed profile's "
            "axisymmetric defocus channel accumulates the OPPOSITE "
            "sign vs the analytic RF defocus in %.0f %% of %d scored "
            "cells (|impulse| ratio ~%.2f): net RF *focusing* where "
            "physics requires *defocusing*.  Transmission agreement in "
            "mode=%r can be compensatory (right totals, wrong loss "
            "locations -- measured vs Toutatis on PXIE, 2026-08-03).  "
            "mode='per_plane_dflip' with d_scale~%.2f carries the "
            "corrected sign (EXPERIMENTAL); see the manual's RFQ "
            "'Known issues' section.",
            100 * (1 - frac), n, med, mode,
            1.0 / med if med > 0 else float("nan"))
    else:
        _log.info(
            "apply_rfq_geometry: defocus-channel audit clean for "
            "mode=%r (%.0f %% of %d cells agree with the analytic "
            "sign; |impulse| ratio %.2f).", mode, 100 * frac, n, med)
