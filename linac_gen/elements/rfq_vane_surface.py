"""True RFQ electrode surface — the tip arc the two-term model ignores.

The real four-vane electrode tip is (to machining intent) a circular
arc of transverse radius ``Tc`` tangent to the tip point at distance
``a_v(z)`` from the axis, swept along z with the modulation.  The PXIE
deck/vane file carry Tc = 4.183 mm against r0 = 5.576 mm — strongly
non-proportional vanes, which is precisely what generates the higher
multipole content (A12, A03, A30 …) absent from the two-term model.
Nothing in HELIX built this surface before (2026-07-30 audit); the
vane-TIP table alone was proven insufficient (tips are two-term to
noise — the information lives in the shape *between* tips).

Geometry per vane at fixed z (x-vane at +x shown; others by symmetry):

    arc centre  C = (a_x(z) + Tc, 0)
    arc point   P(α) = (a_x(z) + Tc·(1 − cos α),  Tc·sin α),
                α ∈ [−α_max, +α_max]

so P(0) is the tip and the surface curves AWAY from the axis — the
correct convex electrode.  Electrode voltages follow the vane file
convention (vane 1/3 on ±x at +V/2; vane 2/4 on ±y at −V/2), i.e. the
returned normalised potential is +1 on the x-pair and −1 on the
y-pair.

z-registration (resolved 2026-07-30): the ``.vane`` table's own z is
element-local with NO offset against the RFQ_CELL chain cumulative z
(validated <1 % in tests/rfq/test_rfq_losses.py); its first ~11 mm
(the RFQ_GAP_RMS_FFS front gap) and last ~16.6 mm are simply
untabulated — card-driven apertures cover those spans.
"""
from __future__ import annotations

import numpy as np

from linac_gen.elements.rfq_coefficients import vane_apertures


def sample_cell_surface(a_x_of_z, a_y_of_z, length_mm: float,
                        Tc_mm: float,
                        n_z: int = 24, n_theta: int = 17,
                        alpha_max_deg: float = 50.0):
    """Boundary points on the four electrode arcs across one cell.

    Parameters
    ----------
    a_x_of_z, a_y_of_z : callables
        Tip distance (mm) of the ±x / ±y vane pair at element-local z
        (mm, in [0, L]).
    length_mm : float
        Cell length L.
    Tc_mm : float
        Transverse tip-arc radius.
    n_z, n_theta : int
        z-slices per cell and points per arc (odd n_theta puts a point
        exactly on the tip).
    alpha_max_deg : float
        Arc half-opening sampled around each tip.

    Returns
    -------
    pts : (N, 3) ndarray  — (x, y, z_local) in mm
    v_norm : (N,) ndarray — +1 on the x-pair, −1 on the y-pair
    """
    z = (np.arange(n_z) + 0.5) / n_z * length_mm
    alpha = np.deg2rad(np.linspace(-alpha_max_deg, alpha_max_deg,
                                   n_theta))
    ca, sa = np.cos(alpha), np.sin(alpha)
    pts = []
    v = []
    for zi in z:
        ax = float(a_x_of_z(zi))
        ay = float(a_y_of_z(zi))
        # +x vane: P = (ax + Tc(1−cosα), Tc sinα)
        px = ax + Tc_mm * (1.0 - ca)
        py = Tc_mm * sa
        for sx in (+1.0, -1.0):                      # +x and −x vanes
            pts.append(np.column_stack([sx * px, py,
                                        np.full_like(px, zi)]))
            v.append(np.full(n_theta, +1.0))
        qx = Tc_mm * sa                              # y-pair (swap roles)
        qy = ay + Tc_mm * (1.0 - ca)
        for sy in (+1.0, -1.0):
            pts.append(np.column_stack([qx, sy * qy,
                                        np.full_like(qx, zi)]))
            v.append(np.full(n_theta, -1.0))
    return np.vstack(pts), np.concatenate(v)


def card_aperture_functions(r0_mm: float, A10: float, length_mm: float,
                            cell_type: int):
    """(a_x(z), a_y(z)) callables from card parameters — the two-term
    tip condition validated to 0.03-0.14 % against ``pxie-rfq.vane``."""
    def ax(z):
        return vane_apertures(r0_mm, A10, length_mm, cell_type, z)[0]

    def ay(z):
        return vane_apertures(r0_mm, A10, length_mm, cell_type, z)[1]

    return ax, ay


def vane_aperture_functions(vane, z_start_mm: float):
    """(a_x(z), a_y(z)) callables interpolating a parsed VaneGeometry.

    ``z_start_mm`` is the cell's start in the RFQ-chain cumulative z —
    the same frame as ``vane.z`` (metres; no offset, see module doc).
    Outside the tabulated span the edge value is held (np.interp
    default), which matches the untabulated front-gap/tail spans.
    """
    z_tab_mm = np.asarray(vane.z) * 1e3
    a1 = np.asarray(vane.aperture_v1) * 1e3
    a2 = np.asarray(vane.aperture_v2) * 1e3

    def ax(z):
        return float(np.interp(z_start_mm + z, z_tab_mm, a1))

    def ay(z):
        return float(np.interp(z_start_mm + z, z_tab_mm, a2))

    return ax, ay
