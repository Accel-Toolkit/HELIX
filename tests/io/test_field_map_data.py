"""Tests for the multi-channel FieldMapData structure."""
import numpy as np
import pytest

from linac_gen.io.field_map_data import FieldMapData, FieldChannel
from linac_gen.io.tracewin_geom import Channel


def _dummy_3d_channel():
    return FieldChannel(
        geometry=7,
        z=np.linspace(0.0, 100.0, 11),
        x=np.linspace(-10.0, 10.0, 5),
        y=np.linspace(-10.0, 10.0, 5),
        Fx=np.zeros((5, 5, 11)),
        Fy=np.zeros((5, 5, 11)),
        Fz=np.zeros((5, 5, 11)),
    )


class TestFieldMapData:
    def test_default_no_channels(self):
        fd = FieldMapData(z=np.linspace(0, 10, 11))
        assert fd.channels == {}

    def test_add_channel(self):
        fd = FieldMapData(z=np.linspace(0, 100, 11))
        ch = _dummy_3d_channel()
        fd.channels[Channel.STAT_B] = ch
        assert Channel.STAT_B in fd.channels
        assert fd.channels[Channel.STAT_B] is ch

    def test_has_helpers(self):
        fd = FieldMapData(z=np.linspace(0, 100, 11))
        fd.channels[Channel.STAT_B] = _dummy_3d_channel()
        assert fd.has_static() is True
        assert fd.has_rf() is False
        assert fd.has_magnetic() is True
        assert fd.has_electric() is False
        fd.channels[Channel.RF_E] = _dummy_3d_channel()
        assert fd.has_rf() is True
        assert fd.has_electric() is True

    def test_axis_length(self):
        fd = FieldMapData(z=np.linspace(0.0, 250.0, 26))
        assert fd.axis_length_mm() == pytest.approx(250.0)


class TestLegacyShims:
    def test_from_legacy_1d_creates_rf_e_channel(self):
        z = np.linspace(0, 100, 11)
        Ez = np.cos(np.pi * z / 100)
        fd = FieldMapData.from_legacy_1d(z=z, Ez=Ez, norm_factor=2.5)
        assert list(fd.channels) == [Channel.RF_E]
        ch = fd.channels[Channel.RF_E]
        assert ch.geometry == 1
        np.testing.assert_allclose(ch.Fz, Ez)
        assert ch.norm_factor == 2.5

    def test_from_legacy_2d_cyl_E_only(self):
        z = np.linspace(0, 100, 11)
        r = np.linspace(0, 20, 5)
        Ez = np.zeros((5, 11));  Er = np.zeros((5, 11))
        fd = FieldMapData.from_legacy_2d_cyl(z=z, r=r, Ez=Ez, Er=Er)
        assert list(fd.channels) == [Channel.RF_E]

    def test_from_legacy_2d_cyl_EB_both(self):
        z = np.linspace(0, 100, 11)
        r = np.linspace(0, 20, 5)
        zero = np.zeros((5, 11))
        fd = FieldMapData.from_legacy_2d_cyl(
            z=z, r=r, Ez=zero, Er=zero, Bz=zero, Br=zero,
        )
        assert set(fd.channels) == {Channel.RF_E, Channel.RF_B}


class TestBackCompatProperties:
    def test_Ez_from_rf_e(self):
        z = np.linspace(0, 100, 11);  Ez = np.ones(11)
        fd = FieldMapData.from_legacy_1d(z=z, Ez=Ez)
        np.testing.assert_allclose(fd.Ez, Ez)

    def test_Bz_from_stat_b(self):
        z = np.linspace(0, 100, 11);  Bz = np.full(11, 0.5)
        fd = FieldMapData(z=z)
        fd.channels[Channel.STAT_B] = FieldChannel(geometry=1, z=z, Fz=Bz)
        np.testing.assert_allclose(fd.Bz, Bz)

    def test_returns_none_when_channel_absent(self):
        fd = FieldMapData(z=np.linspace(0, 100, 11))
        assert fd.Ez is None and fd.Bz is None

    def test_x_y_from_3d_channel(self):
        fd = FieldMapData(z=np.linspace(0, 100, 11))
        fd.channels[Channel.STAT_B] = _dummy_3d_channel()
        assert fd.x is not None and fd.y is not None
        assert len(fd.x) == 5 and len(fd.y) == 5
