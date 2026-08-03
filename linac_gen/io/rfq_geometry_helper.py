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

#: modes for the y-plane gradient (2026-08-02 identical-beam study):
#:  "antisym"   — gy = −gx: best envelope fidelity in BOTH planes and
#:                the most Toutatis-like loss profile (default);
#:  "per_plane" — gy from the y-plane solve (= −G+D signed): best total
#:                transmission on the PXIE benchmark, but σ_y runs
#:                8–10 % low.  The two bracket Toutatis.
_MODES = ("antisym", "per_plane")


def apply_rfq_geometry(lattice: Lattice,
                       vane: Union[str, Path, RfqGeometryProfile],
                       *,
                       mode: str = "antisym",
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
    mode : "antisym" (default) or "per_plane" — see ``_MODES``.
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
        cell._geom_gx = prof.gx[sel] * comp
        cell._geom_gy = gy_glob[sel] * comp
        if boundary_losses:
            cell._geom_boundary = True
            cell._wall_mm = float(wall_mm)
    armed = sum(1 for c in cells if c._geom_z is not None)
    _log.info("apply_rfq_geometry: %d/%d RFQ cells armed with the "
              "vane-geometry profile (mode=%s)", armed, len(cells), mode)
    return armed
