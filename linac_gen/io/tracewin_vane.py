"""TraceWin / Toutatis ``.vane`` file reader.

A ``.vane`` file is a flat ASCII table with one row per longitudinal
slice and 17 columns describing the four-vane RFQ geometry at that z:

  col 0    : z          [m]                   longitudinal position
  col 1    : a_v1       [m]                   vane-1 tip distance from axis
  col 2    : Tc_v1      [m]                   vane-1 transverse curvature
  col 3    : V_v1       [V]                   vane-1 voltage (typically ±V/2)
  col 4    : flag_v1                          unused / status flag
  cols 5-8 : same four columns for vane 2 (90° around)
  cols 9-12: same for vane 3 (180°)
  cols 13-16: same for vane 4 (270°)

Vane indexing convention (used throughout this module):

  ``vane 1`` lies along +x (θ = 0)
  ``vane 2`` lies along +y (θ = π/2)
  ``vane 3`` lies along −x (θ = π)
  ``vane 4`` lies along −y (θ = 3π/2)

Pairs (1,3) and (2,4) sit at orthogonal angles.  Their voltages always
have opposite signs in a four-vane RFQ — usually V₁=V₃=+V/2 and
V₂=V₄=−V/2 — but this module does not assume that and reports the
voltages it reads.

This reader is intentionally lightweight: it returns a dataclass
holding the per-slice arrays.  Higher-level code (the VaneRFQ element)
turns that geometry into fields and matrices.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Union

import numpy as np


@dataclass
class VaneGeometry:
    """Per-slice four-vane RFQ geometry read from a TraceWin ``.vane`` file.

    All length units are SI (metres) and voltages are volts, as in the
    file.  Arrays have shape ``(N,)`` with ``N`` = number of z-slices.
    """

    z: np.ndarray             # (N,)  m       longitudinal position
    aperture_v1: np.ndarray   # (N,)  m       tip distance from axis, vane 1 (along +x)
    aperture_v2: np.ndarray   # (N,)  m       vane 2 (along +y)
    aperture_v3: np.ndarray   # (N,)  m       vane 3 (along −x)
    aperture_v4: np.ndarray   # (N,)  m       vane 4 (along −y)
    Tc_v1: np.ndarray         # (N,)  m       transverse curvature, vane 1
    Tc_v2: np.ndarray
    Tc_v3: np.ndarray
    Tc_v4: np.ndarray
    voltage_v1: np.ndarray    # (N,)  V       vane voltage, vane 1
    voltage_v2: np.ndarray
    voltage_v3: np.ndarray
    voltage_v4: np.ndarray
    flag_v1: np.ndarray       # (N,)  unused
    flag_v2: np.ndarray
    flag_v3: np.ndarray
    flag_v4: np.ndarray

    @property
    def n_slices(self) -> int:
        return int(self.z.size)

    @property
    def length_m(self) -> float:
        return float(self.z[-1] - self.z[0])

    def inter_vane_voltage(self) -> float:
        """Inter-vane voltage = ``V₁ − V₂`` (typically the user's
        ``V`` parameter, e.g. 60 000 V for PXIE).

        Returns the *first-slice* value; if the .vane file modulates
        voltages along z, callers should access ``voltage_v*`` arrays
        directly.
        """
        return float(self.voltage_v1[0] - self.voltage_v2[0])

    def r0_axis(self) -> np.ndarray:
        """Effective transverse aperture ``r₀(z) = √(a₁ a₂)`` (m).

        For symmetric four-vane RFQs ``r₀`` is the geometric mean of
        the orthogonal-pair tip distances and the standard reference
        length used in the Crandall potential expansion.
        """
        return np.sqrt(self.aperture_v1 * self.aperture_v2)

    def modulation_axis(self) -> np.ndarray:
        """Local modulation ``m(z) = max(a₁,a₂)/min(a₁,a₂)``.

        ``m → 1`` in front-end / transition cells; ``m > 1`` in
        accelerating cells.  Pure transverse focusing (no acceleration)
        when ``m = 1`` everywhere.
        """
        a1 = self.aperture_v1
        a2 = self.aperture_v2
        amax = np.maximum(a1, a2)
        amin = np.minimum(a1, a2)
        # Avoid div-by-zero pathologies; amin should always be > 0 for a
        # physical RFQ.
        return np.where(amin > 0, amax / amin, 1.0)


def parse_vane_file(path: Union[str, Path]) -> VaneGeometry:
    """Parse a 17-column TraceWin / Toutatis ``.vane`` file.

    Whitespace-separated; one row per z-slice; trailing newlines OK.
    Raises ``ValueError`` if the column count is not exactly 17 on any
    row, or if the z column is non-monotonic.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"vane file not found: {path}")
    data = np.loadtxt(str(path))
    if data.ndim != 2 or data.shape[1] != 17:
        raise ValueError(
            f"vane file {path} has shape {data.shape}; expected (N, 17). "
            "First row should be z plus 4×(aperture, Tc, V, flag) = 17 cols."
        )
    z = data[:, 0]
    if np.any(np.diff(z) <= 0):
        raise ValueError(
            f"vane file {path} has non-monotonic z column; first non-monotonic "
            f"index = {int(np.argmax(np.diff(z) <= 0)) + 1}."
        )
    return VaneGeometry(
        z=z,
        aperture_v1=data[:, 1], Tc_v1=data[:, 2],
        voltage_v1=data[:, 3], flag_v1=data[:, 4],
        aperture_v2=data[:, 5], Tc_v2=data[:, 6],
        voltage_v2=data[:, 7], flag_v2=data[:, 8],
        aperture_v3=data[:, 9], Tc_v3=data[:, 10],
        voltage_v3=data[:, 11], flag_v3=data[:, 12],
        aperture_v4=data[:, 13], Tc_v4=data[:, 14],
        voltage_v4=data[:, 15], flag_v4=data[:, 16],
    )
