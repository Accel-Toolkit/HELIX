"""Integration tests for the per-channel TraceWin field-map readers."""
from __future__ import annotations

import numpy as np
import pytest

from linac_gen.io.tracewin_fieldmap_reader import (
    read_1d_component,
    read_2d_cyl_component,
    read_2d_cart_component,
    read_3d_cart_component,
)


# ---- write-side fixture generators (manual-spec layout) --------------

def _w_1d(path, Nz, Zmax_m, norm, values):
    assert len(values) == Nz + 1
    with open(path, "w") as f:
        f.write(f"{Nz} {Zmax_m:.6e}\n")
        f.write(f"{norm:.6e}\n")
        for v in values:
            f.write(f"{v:.6e}\n")


def _w_2d_cyl(path, Nz, Zmax_m, Nr, Rmax_m, norm, values):
    """values.shape == (Nz+1, Nr+1) — z outer, r inner (r-fastest)."""
    assert values.shape == (Nz + 1, Nr + 1)
    with open(path, "w") as f:
        f.write(f"{Nz} {Zmax_m:.6e}\n")
        f.write(f"{Nr} {Rmax_m:.6e}\n")
        f.write(f"{norm:.6e}\n")
        for iz in range(Nz + 1):
            for ir in range(Nr + 1):
                f.write(f"{values[iz, ir]:.6e}\n")


def _w_2d_cart(path, Nx, Xmin_m, Xmax_m, Ny, Ymin_m, Ymax_m, norm, values):
    """values.shape == (Ny+1, Nx+1) — y outer, x inner (x-fastest)."""
    assert values.shape == (Ny + 1, Nx + 1)
    with open(path, "w") as f:
        f.write(f"{Nx} {Xmin_m:.6e} {Xmax_m:.6e}\n")
        f.write(f"{Ny} {Ymin_m:.6e} {Ymax_m:.6e}\n")
        f.write(f"{norm:.6e}\n")
        for iy in range(Ny + 1):
            for ix in range(Nx + 1):
                f.write(f"{values[iy, ix]:.6e}\n")


def _w_3d_cart(path, Nz, Zmax_m, Nx, Xmin_m, Xmax_m,
               Ny, Ymin_m, Ymax_m, norm, values):
    """values.shape == (Nz+1, Ny+1, Nx+1) — z outer, y middle, x inner."""
    assert values.shape == (Nz + 1, Ny + 1, Nx + 1)
    with open(path, "w") as f:
        f.write(f"{Nz} {Zmax_m:.6e}\n")
        f.write(f"{Nx} {Xmin_m:.6e} {Xmax_m:.6e}\n")
        f.write(f"{Ny} {Ymin_m:.6e} {Ymax_m:.6e}\n")
        f.write(f"{norm:.6e}\n")
        for iz in range(Nz + 1):
            for iy in range(Ny + 1):
                for ix in range(Nx + 1):
                    f.write(f"{values[iz, iy, ix]:.6e}\n")


# ---- 1-D --------------------------------------------------------------

class TestRead1dComponent:
    def test_roundtrip(self, tmp_path):
        p = tmp_path / "test.esz"
        Nz = 20; Zmax_m = 0.1
        vals = np.cos(np.pi * np.linspace(0, 1, Nz + 1)) * 5.0
        _w_1d(str(p), Nz, Zmax_m, norm=2.5, values=vals)
        out = read_1d_component(str(p))
        assert out.geometry == 1
        assert out.norm_factor == pytest.approx(2.5)
        np.testing.assert_allclose(out.z, np.linspace(0, 100.0, Nz + 1),
                                   rtol=1e-12)
        np.testing.assert_allclose(out.Fz, vals, rtol=1e-6)

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            read_1d_component(str(tmp_path / "nope.esz"))


# ---- 2-D cyl ---------------------------------------------------------

class TestRead2dCylComponent:
    def test_roundtrip(self, tmp_path):
        p = tmp_path / "test.edz"
        Nz, Nr = 10, 4
        Zmax_m, Rmax_m = 0.05, 0.02
        # Fz(z, r) with explicit r-dependence so the reshape is verified
        z_mm = np.linspace(0, Zmax_m * 1000.0, Nz + 1)
        r_mm = np.linspace(0, Rmax_m * 1000.0, Nr + 1)
        Fz = (np.cos(np.pi * z_mm / z_mm[-1])[:, None]
              * (1.0 + 0.01 * r_mm[None, :])) * 1.5
        _w_2d_cyl(str(p), Nz, Zmax_m, Nr, Rmax_m, norm=1.0, values=Fz)
        out = read_2d_cyl_component(str(p))
        # geometry is provisional (caller decides 4 or 5) — reader picks 4.
        assert out.geometry in (4, 5)
        assert out.Fz.shape == (Nz + 1, Nr + 1)
        np.testing.assert_allclose(out.z, z_mm, rtol=1e-12)
        np.testing.assert_allclose(out.r, r_mm, rtol=1e-12)
        np.testing.assert_allclose(out.Fz, Fz, rtol=1e-6)


# ---- 2-D Cart --------------------------------------------------------

class TestRead2dCartComponent:
    def test_roundtrip(self, tmp_path):
        p = tmp_path / "test.esx"
        Nx, Ny = 5, 4
        Xmin_m, Xmax_m = -0.01, 0.01
        Ymin_m, Ymax_m = -0.005, 0.005
        # Distinct test field so transpose ordering is verified
        vals_ny_nx = np.arange((Ny + 1) * (Nx + 1), dtype=float).reshape(
            (Ny + 1, Nx + 1))
        _w_2d_cart(str(p), Nx, Xmin_m, Xmax_m, Ny, Ymin_m, Ymax_m,
                   norm=1.0, values=vals_ny_nx)
        out = read_2d_cart_component(str(p))
        assert out.geometry == 6
        # Axes in mm, with correct extents
        np.testing.assert_allclose(
            out.x, np.linspace(Xmin_m * 1000.0, Xmax_m * 1000.0, Nx + 1),
            rtol=1e-12)
        np.testing.assert_allclose(
            out.y, np.linspace(Ymin_m * 1000.0, Ymax_m * 1000.0, Ny + 1),
            rtol=1e-12)
        # Values should be returned with shape (Nx+1, Ny+1) (x axis first)
        # to match the downstream RegularGridInterpolator(axes=(x, y))
        # convention.  I.e. the reader transposes the file's (Ny, Nx)
        # order.
        assert out.Fz.shape == (Nx + 1, Ny + 1)
        # Verify a specific value: file loop wrote vals_ny_nx[iy, ix] at
        # position (iy=1, ix=2).  After transpose, out.Fz[2, 1] should
        # equal that value.
        np.testing.assert_allclose(out.Fz[2, 1], vals_ny_nx[1, 2])


# ---- 3-D Cart --------------------------------------------------------

class TestRead3dCartComponent:
    def test_roundtrip_structure(self, tmp_path):
        p = tmp_path / "test.bsz"
        Nz, Nx, Ny = 8, 3, 2
        Zmax_m = 0.1
        Xmin_m, Xmax_m = -0.02, 0.02
        Ymin_m, Ymax_m = -0.01, 0.01
        # file-order shape (Nz+1, Ny+1, Nx+1) — that's what we write
        vals_file = np.arange((Nz + 1) * (Ny + 1) * (Nx + 1),
                              dtype=float).reshape(
            (Nz + 1, Ny + 1, Nx + 1))
        _w_3d_cart(str(p), Nz, Zmax_m,
                   Nx, Xmin_m, Xmax_m, Ny, Ymin_m, Ymax_m,
                   norm=1.0, values=vals_file)
        out = read_3d_cart_component(str(p))
        assert out.geometry == 7
        # Axes in mm
        np.testing.assert_allclose(
            out.x, np.linspace(Xmin_m * 1000.0, Xmax_m * 1000.0, Nx + 1),
            rtol=1e-12)
        np.testing.assert_allclose(
            out.y, np.linspace(Ymin_m * 1000.0, Ymax_m * 1000.0, Ny + 1),
            rtol=1e-12)
        np.testing.assert_allclose(
            out.z, np.linspace(0, Zmax_m * 1000.0, Nz + 1), rtol=1e-12)
        # Reader reshapes to (Nz, Ny, Nx) [from the file's natural C-order
        # of z-outer/y-middle/x-inner], then transposes to (Nx, Ny, Nz)
        # so downstream scipy RegularGridInterpolator with axes=(x, y, z)
        # gets a matching shape.
        assert out.Fz.shape == (Nx + 1, Ny + 1, Nz + 1)
        # Verify element-wise correspondence at a distinct cell:
        # file wrote vals_file[iz, iy, ix] at (2, 1, 2); that should now
        # appear at out.Fz[2, 1, 2] (transpose(2, 1, 0) applied).
        np.testing.assert_allclose(out.Fz[2, 1, 2], vals_file[2, 1, 2])


# ---------------------------------------------------------------------
# Top-level orchestrator: read_tracewin_fieldmap(geom, prefix, ...)
# ---------------------------------------------------------------------

from linac_gen.io.tracewin_fieldmap_reader import read_tracewin_fieldmap
from linac_gen.io.tracewin_geom import Channel


def _make_3d_stat_B(tmp_path, prefix="sol", norm=1.0, value=0.0):
    """Write a three-file .bsx/.bsy/.bsz triplet (geom=70 case)."""
    Nz, Nx, Ny = 5, 2, 2
    vals = np.full((Nz + 1, Ny + 1, Nx + 1), value, dtype=float)
    for suf in (".bsx", ".bsy", ".bsz"):
        _w_3d_cart(str(tmp_path / f"{prefix}{suf}"),
                   Nz, 0.1,               # Zmax = 0.1 m
                   Nx, -0.01, 0.01,
                   Ny, -0.01, 0.01,
                   norm=norm, values=vals)
    return str(tmp_path / prefix)


class TestReadTracewinFieldmap:
    def test_geom70_single_stat_B(self, tmp_path):
        prefix = _make_3d_stat_B(tmp_path, norm=2.0, value=0.5)
        fd = read_tracewin_fieldmap(geom=70, prefix=prefix)
        assert Channel.STAT_B in fd.channels
        assert Channel.RF_E not in fd.channels
        ch = fd.channels[Channel.STAT_B]
        assert ch.geometry == 7
        assert ch.Fx is not None and ch.Fy is not None and ch.Fz is not None
        assert ch.norm_factor == pytest.approx(2.0)
        # All three component files had value=0.5 so every entry is 0.5
        np.testing.assert_allclose(ch.Fz, 0.5)

    def test_geom7700_rf_E_and_rf_B(self, tmp_path):
        Nz, Nx, Ny = 3, 2, 2
        vals = np.ones((Nz + 1, Ny + 1, Nx + 1))
        prefix = str(tmp_path / "cav")
        for suf in (".edx", ".edy", ".edz", ".bdx", ".bdy", ".bdz"):
            _w_3d_cart(f"{prefix}{suf}",
                       Nz, 0.1, Nx, -0.01, 0.01, Ny, -0.01, 0.01,
                       norm=1.0, values=vals)
        fd = read_tracewin_fieldmap(geom=7700, prefix=prefix)
        assert set(fd.channels) == {Channel.RF_E, Channel.RF_B}
        for ch in fd.channels.values():
            assert ch.Fx is not None and ch.Fy is not None and ch.Fz is not None
            assert ch.geometry == 7

    def test_geom100_1d_rf_E(self, tmp_path):
        prefix = str(tmp_path / "cav")
        _w_1d(f"{prefix}.edz",
              Nz=10, Zmax_m=0.1, norm=1.0,
              values=np.cos(np.linspace(0, np.pi, 11)))
        fd = read_tracewin_fieldmap(geom=100, prefix=prefix)
        assert list(fd.channels) == [Channel.RF_E]
        ch = fd.channels[Channel.RF_E]
        assert ch.geometry == 1
        assert ch.Fz.shape == (11,)

    def test_geom400_2d_cyl_TM_with_Btheta(self, tmp_path):
        prefix = str(tmp_path / "tm")
        Nz, Nr = 8, 3
        zeros = np.zeros((Nz + 1, Nr + 1))
        for suf in (".edz", ".edr", ".bdq"):
            _w_2d_cyl(f"{prefix}{suf}", Nz, 0.1, Nr, 0.02,
                      norm=1.0, values=zeros)
        fd = read_tracewin_fieldmap(geom=400, prefix=prefix)
        ch = fd.channels[Channel.RF_E]
        assert ch.geometry == 4
        assert ch.Fz is not None and ch.Fr is not None and ch.Fq is not None

    def test_missing_file_raises_filenotfound(self, tmp_path):
        with pytest.raises(FileNotFoundError, match=r"\.bsz"):
            read_tracewin_fieldmap(geom=70, prefix=str(tmp_path / "missing"))

    def test_strip_known_suffix_from_prefix(self, tmp_path):
        """User accidentally passes `sol.bsz` as the prefix — reader strips it."""
        prefix = _make_3d_stat_B(tmp_path, prefix="sol", value=0.0)
        fd = read_tracewin_fieldmap(geom=70, prefix=prefix + ".bsz")
        assert Channel.STAT_B in fd.channels

    def test_base_dir_resolution(self, tmp_path):
        sub = tmp_path / "maps"
        sub.mkdir()
        prefix = _make_3d_stat_B(sub, prefix="sol", value=0.0)
        # pass only the bare name with base_dir set
        fd = read_tracewin_fieldmap(geom=70, prefix="sol",
                                    base_dir=str(sub))
        assert Channel.STAT_B in fd.channels

    def test_frequency_is_stored(self, tmp_path):
        prefix = _make_3d_stat_B(tmp_path, value=0.0)
        fd = read_tracewin_fieldmap(geom=70, prefix=prefix, frequency=352.21)
        assert fd.frequency == pytest.approx(352.21)

    def test_z_axis_propagated(self, tmp_path):
        prefix = _make_3d_stat_B(tmp_path, value=0.0)
        fd = read_tracewin_fieldmap(geom=70, prefix=prefix)
        assert len(fd.z) > 0
        assert fd.z[0] == pytest.approx(0.0)
        assert fd.z[-1] == pytest.approx(100.0)   # Zmax=0.1 m → 100 mm
