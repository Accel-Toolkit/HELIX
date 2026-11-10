"""The beam edge over the aperture: ``(|centroid| + 3 sigma)`` over an element's half-aperture, at
the elements' exits, in each plane.  A fault's criticality and its compensation's verdict read it
(``linac_gen.reliability.observables.criticality_rule``)."""
from __future__ import annotations

import math

import numpy as np

__all__ = ["EDGE_FLOOR", "half_apertures", "edge_over_aperture", "edge_metrics"]

#: an element's ratio is kept, element by element, from this value up (a rule threshold below
#: it would not see the elements under it)
EDGE_FLOOR = 0.5

# APERTURE card shapes the tracker does not simulate (a pepperpot, a ring): no obstruction
_NOT_SIMULATED = (2, 6)


def half_apertures(lattice) -> dict:
    """``{element index: (half-aperture x, y) in mm}`` of the elements that have one, as the
    tracker scrapes: an aperture card's own (a circular card's radius in both planes), or an
    element's bore (its vertical half-width where it has a rectangular one)."""
    out = {}
    for i, e in enumerate(lattice.elements):
        if type(e).__name__ == "Aperture":
            kind = int(getattr(e, "aperture_type", 0) or 0)
            if kind in _NOT_SIMULATED:
                continue
            dx = float(getattr(e, "dx", 0.0) or 0.0)
            dy = dx if kind == 1 else float(getattr(e, "dy", 0.0) or 0.0)
            if dx > 0.0 and dy > 0.0:
                out[i] = (dx, dy)
            continue
        a = float(getattr(e, "aperture", 0.0) or 0.0)
        if a > 0.0:
            ay = float(getattr(e, "aperture_y", 0.0) or 0.0)
            out[i] = (a, ay if ay > 0.0 else a)
    return out


def edge_over_aperture(centroid, sigma_x, sigma_y, exit_idx, names: dict, apertures: dict,
                       floor: float = EDGE_FLOOR):
    """``(largest ratio, its element, {element: ratio, from floor up})``, or ``(None, None,
    None)`` without centroids, sizes, exit rows or apertures.  ``names`` maps an element's index
    to its name; ``exit_idx[i]`` is the row of element ``i``'s exit."""
    if not apertures or centroid is None or sigma_x is None or sigma_y is None \
            or exit_idx is None:
        return None, None, None
    cen = np.asarray(centroid, dtype=float)
    sx, sy = np.asarray(sigma_x, dtype=float), np.asarray(sigma_y, dtype=float)
    exit_idx = np.asarray(exit_idx, dtype=int)
    if cen.ndim != 2 or cen.shape[0] != sx.size or sy.size != sx.size or cen.shape[0] == 0:
        return None, None, None
    worst, where, by = None, None, {}
    for i, (ax, ay) in apertures.items():
        if i >= exit_idx.size or not 0 <= int(exit_idx[i]) < sx.size:
            continue
        j = int(exit_idx[i])
        r = max((abs(cen[j, 0]) + 3.0 * sx[j]) / ax, (abs(cen[j, 2]) + 3.0 * sy[j]) / ay)
        if not math.isfinite(r):
            continue
        name = names.get(i, str(i))
        if r >= floor:
            by[name] = max(float(r), by.get(name, 0.0))
        if worst is None or r > worst:
            worst, where = float(r), name
    return worst, where, by


def edge_metrics(results, lattice) -> dict:
    """``{edge_over_aperture, edge_element, edge_by_element}`` of a run held in memory (an
    envelope or multiparticle results object of ``lattice``)."""
    names = {i: getattr(e, "name", None) or str(i) for i, e in enumerate(lattice.elements)}
    worst, where, by = edge_over_aperture(
        getattr(results, "centroid", None), getattr(results, "sigma_x", None),
        getattr(results, "sigma_y", None), getattr(results, "element_exit_idx", None),
        names, half_apertures(lattice))
    return {"edge_over_aperture": worst, "edge_element": where, "edge_by_element": by}
