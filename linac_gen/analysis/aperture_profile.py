"""Aperture profile along a lattice — for envelope plot overlays.

Walks the lattice, accumulates the longitudinal position, and emits a
piecewise step function ``(s_mm, rx_mm, ry_mm)`` that defines the beam
pipe half-width(s) at every transition.  Two sources are honoured:

1. ``element.aperture`` (mm) and optional ``element.aperture_y`` for
   rectangular pipes.  Used for elements with a constant aperture along
   their length (drifts, quadrupoles, solenoids, etc.).

2. ``FieldMap.field_data.pipe_radius_profile`` — set by the parser when
   the FIELD_MAP card has ``Ka=1`` and a ``<prefix>.ouv`` file is found.
   Format: ``(z_mm, rx_mm, ry_mm)`` along the element's local z axis.
   Inserted at the element's correct longitudinal slot, with each ``z``
   shifted by the element's start position.

Zero-length elements (markers, lattice commands, set/adjust cards, …)
are skipped.  Elements with ``aperture <= 0`` are also skipped — they
would otherwise pull the boundary down to zero and obscure the plot.

The returned arrays are suitable for direct ``setData`` into a pyqtgraph
curve or matplotlib step plot.
"""
from __future__ import annotations

from typing import Iterable, Tuple

import numpy as np


def _safe_aperture(elem) -> Tuple[float, float] | None:
    """Return ``(rx_mm, ry_mm)`` or ``None`` if the element has no usable
    aperture (zero, negative, or NaN)."""
    rx = getattr(elem, "aperture", 0.0) or 0.0
    if rx <= 0.0 or not np.isfinite(rx):
        return None
    ry = getattr(elem, "aperture_y", None)
    if ry is None or ry <= 0.0 or not np.isfinite(ry):
        ry = rx
    return float(rx), float(ry)


def aperture_profile(
    lattice,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build a piecewise aperture profile along the lattice.

    Parameters
    ----------
    lattice
        Anything iterable as ``lattice.elements`` whose entries expose
        ``length`` (mm) and ``aperture`` (mm).  ``FieldMap`` elements
        whose ``field_data`` carries a ``pipe_radius_profile`` are
        unpacked into per-z samples.

    Returns
    -------
    (s_mm, rx_mm, ry_mm) : tuple of 1-D ``np.ndarray``

        * ``s_mm`` — longitudinal positions in millimetres, monotonic
          but not strictly increasing (transitions are encoded by two
          adjacent samples at the same s with different aperture values).
        * ``rx_mm`` — horizontal half-width at each ``s``.
        * ``ry_mm`` — vertical half-width.

        For a circular pipe ``rx == ry``.  Empty arrays are returned if
        the lattice has no aperture-bearing elements.
    """
    elements = list(getattr(lattice, "elements", []) or [])
    if not elements:
        return np.array([]), np.array([]), np.array([])

    s_mm:  list[float] = []
    rx_mm: list[float] = []
    ry_mm: list[float] = []
    cursor_mm = 0.0
    for elem in elements:
        length_mm = float(getattr(elem, "length", 0.0) or 0.0)
        # FieldMap with a per-z aperture profile (.ouv loaded via Ka=1).
        fd = getattr(elem, "field_data", None)
        prof = getattr(fd, "pipe_radius_profile", None) if fd is not None else None
        if (length_mm > 0.0 and prof is not None
                and isinstance(prof, tuple) and len(prof) >= 3):
            z_arr, rx_arr, ry_arr = prof[0], prof[1], prof[2]
            try:
                z_arr  = np.asarray(z_arr,  dtype=float)
                rx_arr = np.asarray(rx_arr, dtype=float)
                ry_arr = np.asarray(ry_arr, dtype=float)
            except Exception:
                z_arr = rx_arr = ry_arr = None
            if (z_arr is not None and z_arr.size >= 2
                    and z_arr.shape == rx_arr.shape == ry_arr.shape):
                s_mm.extend((cursor_mm + z_arr).tolist())
                rx_mm.extend(rx_arr.tolist())
                ry_mm.extend(ry_arr.tolist())
                cursor_mm += length_mm
                continue

        # Constant-aperture element: emit two endpoints (entry + exit).
        ap = _safe_aperture(elem)
        if ap is None:
            cursor_mm += length_mm
            continue
        rx, ry = ap
        if length_mm == 0.0:
            # Zero-length aperture marker — keep a single transition
            # spike so the plot can still mark it.
            s_mm.append(cursor_mm)
            rx_mm.append(rx); ry_mm.append(ry)
        else:
            s_mm.append(cursor_mm)
            rx_mm.append(rx); ry_mm.append(ry)
            s_mm.append(cursor_mm + length_mm)
            rx_mm.append(rx); ry_mm.append(ry)
        cursor_mm += length_mm

    return (np.asarray(s_mm,  dtype=float),
            np.asarray(rx_mm, dtype=float),
            np.asarray(ry_mm, dtype=float))
