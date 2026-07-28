"""Tests for Ki/.scc loading and Ka aperture-flag handling in
read_tracewin_fieldmap.  See plan §Tasks 2c and 2d.
"""
from __future__ import annotations
import os
import numpy as np
import pytest

from linac_gen.io.tracewin_fieldmap_reader import read_tracewin_fieldmap
from linac_gen.io.tracewin_geom import Channel

# reuse the helpers from the existing test file
from tests.io.test_tracewin_fieldmap_reader import _w_1d, _w_3d_cart


def _make_3d_stat_B(tmp_path, prefix="sol"):
    Nz, Nx, Ny = 3, 2, 2
    zeros = np.zeros((Nz + 1, Ny + 1, Nx + 1))
    for suf in (".bsx", ".bsy", ".bsz"):
        _w_3d_cart(str(tmp_path / f"{prefix}{suf}"),
                   Nz, 0.1, Nx, -0.01, 0.01, Ny, -0.01, 0.01,
                   norm=1.0, values=zeros)
    return str(tmp_path / prefix)


# ----- .scc / Ki ------------------------------------------------------

class TestSccLoading:
    def test_scc_loaded_when_Ki_nonzero(self, tmp_path):
        prefix = _make_3d_stat_B(tmp_path)
        # Manual format: header "<mode> <N>" + N data rows "z_m  scc_or_I"
        with open(prefix + ".scc", "w") as f:
            f.write("0 3\n0.0  0.5\n0.05 0.8\n0.1  0.5\n")
        fd = read_tracewin_fieldmap(geom=70, prefix=prefix, Ki=0.7)
        assert fd.scc_profile is not None
        assert fd.scc_profile.shape == (3, 2)
        # z column converted to mm
        np.testing.assert_allclose(fd.scc_profile[:, 0], [0.0, 50.0, 100.0])
        # value column unchanged
        np.testing.assert_allclose(fd.scc_profile[:, 1], [0.5, 0.8, 0.5])
        assert fd.scc_scale == pytest.approx(0.7)
        assert fd.scc_mode == 0

    def test_scc_skipped_when_Ki_zero(self, tmp_path):
        prefix = _make_3d_stat_B(tmp_path)
        # No .scc written, Ki=0 → fine
        fd = read_tracewin_fieldmap(geom=70, prefix=prefix, Ki=0.0)
        assert fd.scc_profile is None
        assert fd.scc_scale == 0.0

    def test_scc_missing_when_Ki_nonzero_raises(self, tmp_path):
        prefix = _make_3d_stat_B(tmp_path)
        # No .scc written but Ki=0.5 → FileNotFoundError
        with pytest.raises(FileNotFoundError, match=r"\.scc"):
            read_tracewin_fieldmap(geom=70, prefix=prefix, Ki=0.5)


# ----- .ouv / Ka ------------------------------------------------------

class TestKaFlag:
    def test_ka0_is_default(self, tmp_path):
        prefix = _make_3d_stat_B(tmp_path)
        fd = read_tracewin_fieldmap(geom=70, prefix=prefix, Ka=0)
        assert fd.ka == 0
        assert fd.pipe_radius_profile is None

    def test_ka1_loads_ouv(self, tmp_path):
        prefix = _make_3d_stat_B(tmp_path)
        with open(prefix + ".ouv", "w") as f:
            f.write("0.0  0.020\n0.05 0.022\n0.1 0.020\n")
        fd = read_tracewin_fieldmap(geom=70, prefix=prefix, Ka=1)
        assert fd.ka == 1
        assert fd.pipe_radius_profile is not None
        # Profile now stored as (z_mm, rx_mm, ry_mm) — circular .ouv
        # files (2 columns) put the same value in rx and ry.
        z_mm, rx_mm, ry_mm = fd.pipe_radius_profile
        np.testing.assert_allclose(z_mm, [0.0, 50.0, 100.0])
        np.testing.assert_allclose(rx_mm, [20.0, 22.0, 20.0])
        np.testing.assert_allclose(ry_mm, [20.0, 22.0, 20.0])

    def test_ka1_missing_ouv_raises(self, tmp_path):
        prefix = _make_3d_stat_B(tmp_path)
        with pytest.raises(FileNotFoundError, match=r"\.ouv"):
            read_tracewin_fieldmap(geom=70, prefix=prefix, Ka=1)

    def test_ka2_records_flag_no_file_required(self, tmp_path):
        prefix = _make_3d_stat_B(tmp_path)
        fd = read_tracewin_fieldmap(geom=70, prefix=prefix, Ka=2)
        assert fd.ka == 2
        assert fd.pipe_radius_profile is None
