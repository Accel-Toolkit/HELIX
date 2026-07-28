"""Unit tests for the 3-D Cartesian field-map readers (ftype 70/71/74).

Uses a synthetic ``E_z(x, y, z) = cos(π·z/L) · exp(-r²/σ²)`` map
written to three sibling files, then verifies grid extents, units,
norm-factor handling, and shape ordering.
"""
from __future__ import annotations

import math
import os
import tempfile

import numpy as np
import pytest

from linac_gen.io.field_map_reader import (
    read_3d_cart_E, read_3d_cart_B, read_3d_cart_EB,
)


def _write_component_file(path: str, x, y, z,
                          values: np.ndarray, norm: float = 1.0) -> None:
    """Write a 3-D TraceWin component file at ``path``.

    ``x``, ``y``, ``z`` arrays are in metres (TraceWin manual convention).
    ``values`` has shape ``(nx, ny, nz)``.  Per the manual
    ("Dimension 3") the header is ``Nz Zmax / Nx Xmin Xmax /
    Ny Ymin Ymax / Norm`` with axis counts = intervals (len - 1),
    and the data loop runs ``for k=0..Nz { for j=0..Ny { for i=0..Nx }}``
    so **x is the fastest axis**.
    """
    nx, ny, nz = len(x), len(y), len(z)
    with open(path, "w") as f:
        f.write(f"{nz - 1}  {z[-1]:.6e}\n")                 # Nz intervals, Zmax
        f.write(f"{nx - 1}  {x[0]:.6e}  {x[-1]:.6e}\n")     # Nx, Xmin, Xmax
        f.write(f"{ny - 1}  {y[0]:.6e}  {y[-1]:.6e}\n")     # Ny, Ymin, Ymax
        f.write(f"{norm:.6e}\n")
        for iz in range(nz):
            for iy in range(ny):
                for ix in range(nx):
                    f.write(f"{values[ix, iy, iz] * norm:.6e}\n")


@pytest.fixture
def tmp_3d_E_cavity(tmp_path):
    """Write a synthetic 3-D pillbox cavity into a tmpdir.

    Returns (prefix, Ex_expected, Ey_expected, Ez_expected, x_mm, y_mm, z_mm).
    """
    # Grid in metres
    nx, ny, nz = 6, 6, 21
    x = np.linspace(-0.020, 0.020, nx)   # ±20 mm
    y = np.linspace(-0.020, 0.020, ny)
    z = np.linspace(0.000, 0.100, nz)    # 0 … 100 mm
    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
    sigma = 0.025
    r2 = X**2 + Y**2
    Ez = 1e6 * np.cos(math.pi * Z / 0.100) * np.exp(-r2 / sigma**2)   # V/m
    # Synthetic Ex, Ey: -∂Φ/∂x with Φ chosen so Ex = x/(2σ²) · Ez, etc.
    # For a cavity test we only need non-trivial non-zero fields.
    Ex = -X * np.cos(math.pi * Z / 0.100) * 1e5
    Ey = -Y * np.cos(math.pi * Z / 0.100) * 1e5

    prefix = str(tmp_path / "cavity")
    _write_component_file(prefix + ".edx", x, y, z, Ex, norm=1.0)
    _write_component_file(prefix + ".edy", x, y, z, Ey, norm=1.0)
    _write_component_file(prefix + ".edz", x, y, z, Ez, norm=1.0)
    return prefix, Ex, Ey, Ez, x * 1000, y * 1000, z * 1000


def test_read_3d_E_returns_correct_shape_and_units(tmp_3d_E_cavity):
    prefix, Ex_ref, Ey_ref, Ez_ref, x_mm, y_mm, z_mm = tmp_3d_E_cavity
    data = read_3d_cart_E(prefix)
    assert data.symmetry == "3d"
    assert data.Ex.shape == (6, 6, 21)
    assert data.Ey.shape == (6, 6, 21)
    assert data.Ez.shape == (6, 6, 21)
    # Grid axes converted from m → mm
    np.testing.assert_allclose(data.x, x_mm, rtol=1e-10)
    np.testing.assert_allclose(data.y, y_mm, rtol=1e-10)
    np.testing.assert_allclose(data.z, z_mm, rtol=1e-10)
    # Values unchanged (norm_factor = 1).  We write files with "%.6e"
    # so round-trip precision is ~1e-6 relative.
    np.testing.assert_allclose(data.Ex, Ex_ref, rtol=1e-5, atol=1e-8)
    np.testing.assert_allclose(data.Ey, Ey_ref, rtol=1e-5, atol=1e-8)
    np.testing.assert_allclose(data.Ez, Ez_ref, rtol=1e-5, atol=1e-8)


def test_read_3d_E_preserves_raw_values_and_norm(tmp_path):
    """Reader keeps values exactly as stored and exposes Norm as metadata.

    Different TraceWin versions disagree on whether to multiply or divide
    by Norm when exporting, so the reader stays neutral — the physical
    amplitude is resolved by the FIELD_MAP ``amp`` parameter (or by the
    caller explicitly, if using the direct reader API).
    """
    x = np.linspace(0, 0.001, 2)
    y = np.linspace(0, 0.001, 2)
    z = np.linspace(0, 0.001, 3)
    stored = np.arange(2 * 2 * 3, dtype=float).reshape((2, 2, 3))  # as written
    prefix = str(tmp_path / "norm")
    # The helper already scales by norm when writing (see its `values*norm`
    # pattern) — but we pass norm=1.0 so stored == values.
    _write_component_file(prefix + ".edx", x, y, z, np.zeros_like(stored), norm=2.0)
    _write_component_file(prefix + ".edy", x, y, z, np.zeros_like(stored), norm=2.0)
    _write_component_file(prefix + ".edz", x, y, z, stored, norm=2.0)
    data = read_3d_cart_E(prefix)
    # norm_factor is stored as metadata
    assert data.norm_factor == pytest.approx(2.0)
    # Values read back = values * norm (as the helper wrote them)
    expected = stored * 2.0
    np.testing.assert_allclose(data.Ez, expected, rtol=1e-5, atol=1e-8)


def test_read_3d_E_missing_file_raises(tmp_path):
    prefix = str(tmp_path / "missing")
    # Only write one of the three
    x = np.linspace(0, 0.001, 2); y = x; z = x
    _write_component_file(prefix + ".edx", x, y, z, np.zeros((2, 2, 2)))
    with pytest.raises(FileNotFoundError):
        read_3d_cart_E(prefix)


def test_read_3d_EB_merges_E_and_B(tmp_path):
    # Write a minimal 6-file EB map
    x = np.linspace(0, 0.001, 2); y = x; z = x
    zeros = np.zeros((2, 2, 2))
    prefix = str(tmp_path / "eb")
    for suf in ("edx", "edy", "edz", "bsx", "bsy", "bsz"):
        _write_component_file(f"{prefix}.{suf}", x, y, z, zeros + (1.0 if suf[1] == "d" else 0.5))
    data = read_3d_cart_EB(prefix)
    assert data.Ex is not None and data.Bx is not None
    np.testing.assert_allclose(data.Ex[0, 0, 0], 1.0)
    np.testing.assert_allclose(data.Bx[0, 0, 0], 0.5)
