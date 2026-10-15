"""Electrodes of an RFQ built from its ``RFQ_CELL`` cards (no vane file).

For a deck without an ``RFQ_GEOM`` vane file the time-domain solver needs
the vane tips from somewhere; this module builds them from the cards
with the same model the per-cell solver uses:

* accelerating (±2) and transition (±4) cells: the two-term equipotential
  tip condition of :func:`~linac_gen.elements.rfq_coefficients.vane_apertures`
  (x against the +x pole, validated against the PXIE vane table to
  0.03-0.25 % per cell);
* front-end (+3) and exit (−3) cells: the vanes flare as TraceWin's
  quadrupole ramp of those cells, C1 = sin³(πz/2L) (+3) or cos³(πz/2L)
  (−3), i.e. a tip distance R = r0/√C1 (on PXIE within 10 % of the real
  entrance flare and 3 % of the exit flare, except the last half
  millimetre of vane-end rounding);
* tip radius ρ: the card's ``Tc``, else 0.75·r0 (PXIE: ρ/r0 = 0.75);
* vanes from the first ``RFQ_GAP_RMS_FFS`` gap to the chain length less
  the second; without the card, from where the entrance flare has
  narrowed to ``R_MAX_FACTOR``·r0 to where the exit flare reaches it.

The end plates' holes follow r0 (PXIE: 6.0 mm for r0 = 5.576 mm;
:func:`~linac_gen.rfq_td.geometry.end_plate_hole`).
"""
from __future__ import annotations

import math

import numpy as np

#: widest tip distance of a flaring cell, in units of its r0 (PXIE's
#: radial matching section opens to 3.47·r0 at the vanes' start)
R_MAX_FACTOR = 3.5
#: table spacing along z, mm (the PXIE vane file: 0.25 mm); the steep
#: flares of the +-3 cells get a finer one
DZ_MM = 0.25
DZ_FLARE_MM = 0.05


def _cell_tips(cell, zl):
    """(R_x, R_y) at cell-local positions ``zl`` (mm)."""
    from linac_gen.elements.rfq_coefficients import vane_apertures
    L = float(cell.length)
    r0 = float(cell.r0_mm)
    t = int(cell.cell_type)
    if abs(t) == 3:
        arg = 0.5 * np.pi * np.asarray(zl) / L
        c1 = np.sin(arg) ** 3 if t > 0 else np.cos(arg) ** 3
        r = r0 / np.sqrt(np.maximum(c1, R_MAX_FACTOR ** -2))
        return r, r.copy()
    xy = np.array([vane_apertures(r0, float(cell.A10), L, t, float(z))
                   for z in np.asarray(zl)])
    return xy[:, 0], xy[:, 1]


def electrodes_from_cards(cells, *, gaps=None, source="RFQ_CELL cards",
                          voltage=None):
    """:class:`~linac_gen.rfq_td.geometry.RfqElectrodes` of a chain of
    :class:`~linac_gen.elements.rfq_cell.RfqCell` from their cards, in
    chain coordinates (z = 0 at the first cell's entrance).  ``gaps`` =
    (entrance, exit) of ``RFQ_GAP_RMS_FFS`` in mm, or None.  ``voltage``:
    the inter-vane voltage to give the poles (default the cards'; the
    time-domain solver uses 1 V and scales the field, so one map serves
    every voltage)."""
    from linac_gen.rfq_td.geometry import EndPlate, RfqElectrodes, end_plate_hole
    if not cells:
        raise ValueError("no RFQ cells")
    volts = {float(c.voltage_V) for c in cells}
    if len(volts) != 1:
        raise ValueError("the RFQ cells carry different vane voltages")
    V = volts.pop() if voltage is None else float(voltage)
    edges = np.concatenate([[0.0], np.cumsum([float(c.length)
                                              for c in cells])])
    chain_len = float(edges[-1])
    zs, rx, ry, rho, r0s = [], [], [], [], []
    for k, c in enumerate(cells):
        L = float(c.length)
        if abs(int(c.cell_type)) == 3:
            n = max(64, int(math.ceil(L / DZ_FLARE_MM)))
        else:
            n = max(8, int(math.ceil(L / DZ_MM)))
        zl = np.linspace(0.0, L, n + 1)
        if k < len(cells) - 1:
            zl = zl[:-1]                     # the next cell starts there
        a, b = _cell_tips(c, zl)
        tc = float(getattr(c, "Tc_mm", 0.0) or 0.0)
        r_tip = tc if tc > 0 else 0.75 * float(c.r0_mm)
        zs.append(edges[k] + zl)
        rx.append(a)
        ry.append(b)
        rho.append(np.full(zl.size, r_tip))
        r0s.append(np.full(zl.size, float(c.r0_mm)))
    z = np.concatenate(zs)
    R_x = np.concatenate(rx)
    R_y = np.concatenate(ry)
    rho = np.concatenate(rho)
    r0s = np.concatenate(r0s)
    # where the vanes are
    if gaps is not None:
        z_lo, z_hi = float(gaps[0]), chain_len - float(gaps[1])
    else:
        # from the last fully flared table point at the entrance to the
        # first one at the exit
        open_ = np.minimum(R_x, R_y) >= R_MAX_FACTOR * r0s * (1.0 - 1e-12)
        inner = np.flatnonzero(~open_)
        if inner.size == 0:
            raise ValueError("the RFQ cards leave no vane inside the "
                             "flare limit")
        z_lo = float(z[max(inner[0] - 1, 0)])
        z_hi = float(z[min(inner[-1] + 1, z.size - 1)])
    if not 0.0 <= z_lo < z_hi <= chain_len:
        raise ValueError(f"vanes {z_lo:.3f}..{z_hi:.3f} mm do not fit the "
                         f"RFQ cell chain 0..{chain_len:.3f} mm")
    keep = (z > z_lo) & (z < z_hi)
    zt = np.concatenate([[z_lo], z[keep], [z_hi]])

    def at(arr):
        return np.interp(zt, z, arr)

    plates = [EndPlate(z_face=0.0, side=-1,
                       hole_radius=end_plate_hole(cells[0].r0_mm)),
              EndPlate(z_face=chain_len, side=+1,
                       hole_radius=end_plate_hole(cells[-1].r0_mm))]
    return RfqElectrodes(z=zt, R_x=at(R_x), R_y=at(R_y), rho_x=at(rho),
                         rho_y=at(rho), V_x=-0.5 * V, V_y=+0.5 * V,
                         plates=plates, source=source)


def two_term_field_map(cells, *, h=0.2, x_store=7.8, slices_per_cell=32,
                       pad_mm=20.0, voltage=1.0):
    """The cards' own analytic two-term field as an
    :class:`~linac_gen.rfq_td.fieldmap.RfqFieldMap` -- the time-domain
    solver's regression mode, which drives its integrator and clock with
    the per-cell model's field instead of the vanes' (separating integrator
    and clock effects from field effects).

    With +y vanes at +V/2 (the vane-file convention; the tips are those of
    :func:`electrodes_from_cards`):
      ±2, ±4 cells: U = V/2 [(y² − x²)/r0² − sign(type)·A10·I0(πr/L)·cos(πz/L)]
      ±3 cells:     U = V/2 (y² − x²)/r0² · C1(z), C1 = sin³ / cos³(πz/2L)
    and no field outside the chain.  Stored to ``x_store`` mm at ``h``.
    ``voltage``: the inter-vane voltage (V; None = the cards').  The
    default 1 V is what the solver expects in place of its map for
    card-built vanes, which it scales by the cards' voltage."""
    from scipy.special import i0

    from linac_gen.rfq_td.fieldmap import RfqFieldMap
    V = float(cells[0].voltage_V) if voltage is None else float(voltage)
    edges = np.concatenate([[0.0], np.cumsum([float(c.length)
                                              for c in cells])])
    nx = int(round(x_store / h)) + 3
    xs = h * np.arange(nx)
    X, Y = np.meshgrid(xs, xs, indexing="ij")
    R = np.hypot(X, Y)

    def potential(zs):
        out = np.zeros((nx, nx, zs.size))
        k = np.searchsorted(edges, zs, side="right") - 1
        for j, (z, kk) in enumerate(zip(zs, k)):
            if kk < 0 or kk >= len(cells):
                continue
            c = cells[kk]
            L = float(c.length)
            zl = z - edges[kk]
            t = int(c.cell_type)
            q = (Y * Y - X * X) / float(c.r0_mm) ** 2
            if abs(t) == 3:
                a = 0.5 * np.pi * zl / L
                u = q * (np.sin(a) ** 3 if t > 0 else np.cos(a) ** 3)
            else:
                u = q - np.sign(t) * float(c.A10) * i0(np.pi * R / L) \
                    * np.cos(np.pi * zl / L)
            out[:, :, j] = 0.5 * V * u
        return out

    spans = ([(-pad_mm, 0.0, 0.5)]
             + [(edges[k], edges[k + 1], float(cells[k].length)
                 / slices_per_cell) for k in range(len(cells))]
             + [(edges[-1], edges[-1] + pad_mm, 0.5)])
    cores, blocks = [], []
    for z0, z1, hz in spans:
        zs = np.arange(z0 - 2 * hz, z1 + 3 * hz - 1e-9, hz)
        blocks.append((float(zs[0]), float(hz),
                       potential(zs).astype(np.float32)))
        cores.append((z0, z1))
    return RfqFieldMap.from_blocks(h, cores, blocks, key="two-term-cards")
