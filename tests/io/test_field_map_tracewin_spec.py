"""Regression tests for TraceWin-canonical ``.edz`` file layouts.

The readers must handle BOTH the manual's documented layout AND the
legacy custom layout the existing fixtures use.  Verifies:

1. Canonical 1-D: ``Nz Zmax[m] / Norm / (Nz+1)·Fz``   → correct z, Ez,
   norm_factor, grid length in mm (all spatial dims in METRES per manual).
2. Canonical 2-D: ``Nz Zmax / Nr Rmax / Norm / Ez block / Er block``
   with r-fastest ordering (inner loop over r) → correct grid shapes
   and values.
3. Legacy 1-D still reads unchanged (no regression).
4. Legacy 2-D still reads unchanged (no regression).
"""
from __future__ import annotations

import os
import tempfile

import numpy as np
import pytest

from linac_gen.io.field_map_reader import read_edz_1d, read_edz_2d


# ---------------------------------------------------------------------
def _write_canonical_1d(path: str, Nz: int, Zmax_m: float,
                        norm: float, ez: np.ndarray) -> None:
    """Emit a TraceWin-canonical 1-D .edz file (Nz+1 field values).

    Header per TraceWin manual: ``Nz  Zmax`` with Zmax in metres.
    """
    assert len(ez) == Nz + 1
    with open(path, "w") as f:
        f.write(f"{Nz} {Zmax_m:.6e}\n")
        f.write(f"{norm:.6e}\n")
        for v in ez:
            f.write(f"{v:.6e}\n")


def test_canonical_1d_roundtrip(tmp_path):
    Nz = 20
    Zmax_m = 0.1                                        # 10 cm in metres
    z_mm_expected = np.linspace(0.0, Zmax_m * 1000.0, Nz + 1)
    ez_phys = np.cos(np.pi * z_mm_expected / z_mm_expected[-1]) * 1e6   # V/m
    path = str(tmp_path / "canon.edz")
    _write_canonical_1d(path, Nz, Zmax_m, norm=2.5, ez=ez_phys)

    data = read_edz_1d(path)
    assert data.symmetry == "1d"
    np.testing.assert_allclose(data.z, z_mm_expected, rtol=1e-12)
    np.testing.assert_allclose(data.Ez, ez_phys, rtol=1e-5, atol=1e-6)
    assert data.norm_factor == pytest.approx(2.5)


def test_canonical_1d_Nz_semantics(tmp_path):
    """File with Nz=3 has 4 values (Nz+1), not 3."""
    path = str(tmp_path / "Nz3.edz")
    ez = np.array([0.0, 0.5, 1.0, 0.5])       # 4 values = Nz+1
    _write_canonical_1d(path, Nz=3, Zmax_m=0.01, norm=1.0, ez=ez)   # 1 cm
    data = read_edz_1d(path)
    assert len(data.z) == 4
    np.testing.assert_allclose(data.z, [0.0, 10.0/3, 20.0/3, 10.0], atol=1e-10)


# ---------------------------------------------------------------------
def _write_canonical_2d(path: str, Nz: int, Zmax_m: float,
                        Nr: int, Rmax_m: float, norm: float,
                        Ez: np.ndarray, Er: np.ndarray) -> None:
    """Emit a TraceWin-canonical 2-D .edz file (fm_type=2).

    ``Ez`` / ``Er`` must have shape ``(Nz+1, Nr+1)`` (z outer, r inner,
    matching the manual loop ``for k=0..Nz { for i=0..Nr }``).  Header
    has both extents in METRES.
    """
    assert Ez.shape == (Nz + 1, Nr + 1)
    assert Er.shape == Ez.shape
    with open(path, "w") as f:
        f.write(f"{Nz} {Zmax_m:.6e}\n")
        f.write(f"{Nr} {Rmax_m:.6e}\n")
        f.write(f"{norm:.6e}\n")
        for iz in range(Nz + 1):
            for ir in range(Nr + 1):
                f.write(f"{Ez[iz, ir]:.6e}\n")
        for iz in range(Nz + 1):
            for ir in range(Nr + 1):
                f.write(f"{Er[iz, ir]:.6e}\n")


def test_canonical_2d_roundtrip(tmp_path):
    Nz = 10; Zmax_m = 0.05          # 5 cm in metres
    Nr = 4;  Rmax_m = 0.02          # 2 cm
    z_mm = np.linspace(0, Zmax_m * 1000.0, Nz + 1)
    r_mm = np.linspace(0, Rmax_m * 1000.0, Nr + 1)
    # Ez(z, r) — shape (Nz+1, Nr+1) per manual loop order (r inner).
    Ez = np.broadcast_to(np.cos(np.pi * z_mm / z_mm[-1])[:, None],
                         (Nz + 1, Nr + 1)).copy() * 1e6
    # Er(z, r) = -(r/2) * dEz/dz (no r dependence in Ez keeps dEz/dz 1-D in z).
    dEz = np.gradient(Ez[:, 0], z_mm)
    Er = -0.5 * r_mm[None, :] * dEz[:, None]
    path = str(tmp_path / "canon2d.edz")
    _write_canonical_2d(path, Nz, Zmax_m, Nr, Rmax_m, norm=1.0, Ez=Ez, Er=Er)

    data = read_edz_2d(path, fm_type=2)
    assert data.symmetry == "cylindrical"
    np.testing.assert_allclose(data.z, z_mm, rtol=1e-12)
    np.testing.assert_allclose(data.r, r_mm, rtol=1e-12)
    assert data.Ez.shape == (Nz + 1, Nr + 1)
    np.testing.assert_allclose(data.Ez, Ez, rtol=1e-5)
    np.testing.assert_allclose(data.Er, Er, rtol=1e-5, atol=1e-8)


# ---------------------------------------------------------------------
# Legacy formats: auto-detect must still route to the old parser.
# ---------------------------------------------------------------------
def test_legacy_1d_still_works():
    """The existing 1-D fixture uses ``Npts / zmin zmax / values``."""
    path = os.path.join(os.path.dirname(__file__), "fixtures", "test_1d.edz")
    data = read_edz_1d(path)
    assert data.symmetry == "1d"
    assert len(data.Ez) > 0
    # Sanity: z increases monotonically
    assert np.all(np.diff(data.z) > 0)


def test_legacy_2d_still_works():
    path = os.path.join(os.path.dirname(__file__), "fixtures", "test_2d.edz")
    data = read_edz_2d(path, fm_type=2)
    assert data.symmetry == "cylindrical"
    assert data.Ez is not None
    assert data.Er is not None
