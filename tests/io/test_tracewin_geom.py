"""Unit tests for the TraceWin ``geom`` 5-digit decoder.

Manual reference (lines 18052-18113): geom = aper·10⁴ + rf_B·10³ + rf_E·10²
+ stat_B·10 + stat_E, with per-digit geometry codes 0..9.  Negative geom
means "use 2nd-order off-axis expansion".
"""
from linac_gen.io.tracewin_geom import decode_geom, GeomCode


def test_geom_zero_is_all_zeros():
    g = decode_geom(0)
    assert g == GeomCode(stat_E=0, stat_B=0, rf_E=0, rf_B=0, aper=0,
                         second_order=False)


def test_geom_70_is_3d_static_magnetic():
    """Manual example: FIELD_MAP 70 … qpole — quadrupole/solenoid."""
    g = decode_geom(70)
    assert g.stat_B == 7
    assert g.stat_E == 0 and g.rf_E == 0 and g.rf_B == 0 and g.aper == 0


def test_geom_0070_parsed_same_as_70():
    assert decode_geom(70) == decode_geom(int("0070"))


def test_geom_7700_is_3d_rf_both():
    """Manual example: FIELD_MAP 7700 … carte_3gap_2b — 3D RF cavity."""
    g = decode_geom(7700)
    assert g.rf_B == 7 and g.rf_E == 7
    assert g.stat_B == 0 and g.stat_E == 0


def test_geom_100_is_1d_rf_electric():
    """Most common RF cavity: 1D Ez(z)."""
    g = decode_geom(100)
    assert g.rf_E == 1
    assert g.stat_E == 0 and g.stat_B == 0 and g.rf_B == 0


def test_geom_400_is_2d_cyl_rf_electric_TM():
    g = decode_geom(400)
    assert g.rf_E == 4


def test_geom_90_is_1d_quad_gradient():
    g = decode_geom(90)
    assert g.stat_B == 9


def test_negative_geom_sets_second_order_flag():
    g_pos = decode_geom(70)
    g_neg = decode_geom(-70)
    assert g_neg.second_order is True
    assert g_pos.second_order is False
    assert g_neg.stat_B == g_pos.stat_B == 7


def test_full_house():
    """aper=1, rf_B=7, rf_E=7, stat_B=7, stat_E=1 → 17771."""
    g = decode_geom(17771)
    assert g.aper == 1 and g.rf_B == 7 and g.rf_E == 7
    assert g.stat_B == 7 and g.stat_E == 1


import pytest
from linac_gen.io.tracewin_geom import (
    Channel, component_files, enabled_channels,
)


class TestChannel:
    def test_kind_letters(self):
        assert Channel.STAT_E.field_letter == "e"
        assert Channel.STAT_E.type_letter  == "s"
        assert Channel.STAT_B.field_letter == "b"
        assert Channel.STAT_B.type_letter  == "s"
        assert Channel.RF_E.field_letter   == "e"
        assert Channel.RF_E.type_letter    == "d"
        assert Channel.RF_B.field_letter   == "b"
        assert Channel.RF_B.type_letter    == "d"

    def test_is_static_is_rf(self):
        assert Channel.STAT_E.is_static and not Channel.STAT_E.is_rf
        assert Channel.RF_E.is_rf and not Channel.RF_E.is_static

    def test_is_electric_is_magnetic(self):
        assert Channel.STAT_E.is_electric and not Channel.STAT_E.is_magnetic
        assert Channel.STAT_B.is_magnetic and not Channel.STAT_B.is_electric


class TestComponentFiles:
    def test_1d_stat_E(self):
        assert component_files(Channel.STAT_E, digit=1) == [".esz"]

    def test_1d_stat_B(self):
        assert component_files(Channel.STAT_B, digit=1) == [".bsz"]

    def test_1d_rf_E(self):
        assert component_files(Channel.RF_E, digit=1) == [".edz"]

    def test_1d_rf_B(self):
        assert component_files(Channel.RF_B, digit=1) == [".bdz"]

    def test_2d_cyl_E_type_static(self):
        assert component_files(Channel.STAT_E, digit=4) == [".esr", ".esz"]

    def test_2d_cyl_E_type_rf_TM(self):
        assert component_files(Channel.RF_E, digit=4) == [".edr", ".edz", ".bdq"]

    def test_2d_cyl_B_type_static(self):
        assert component_files(Channel.STAT_B, digit=5) == [".bsr", ".bsz"]

    def test_2d_cyl_B_type_rf_TE(self):
        assert component_files(Channel.RF_B, digit=5) == [".bdr", ".bdz", ".edq"]

    def test_2d_cart_stat_B(self):
        assert component_files(Channel.STAT_B, digit=6) == [".bsx", ".bsy"]

    def test_3d_cart_stat_B(self):
        assert component_files(Channel.STAT_B, digit=7) == [
            ".bsx", ".bsy", ".bsz"]

    def test_3d_cart_rf_E(self):
        assert component_files(Channel.RF_E, digit=7) == [
            ".edx", ".edy", ".edz"]

    def test_1d_quad_gradient(self):
        assert component_files(Channel.STAT_B, digit=9) == [".bsz"]

    def test_digit_9_invalid_in_stat_E(self):
        with pytest.raises(ValueError, match=r"digit 9"):
            component_files(Channel.STAT_E, digit=9)

    def test_digit_5_invalid_in_stat_E(self):
        with pytest.raises(ValueError, match=r"digit 5"):
            component_files(Channel.STAT_E, digit=5)

    def test_digit_4_invalid_in_stat_B(self):
        with pytest.raises(ValueError, match=r"digit 4"):
            component_files(Channel.STAT_B, digit=4)

    def test_digit_8_raises_not_implemented(self):
        with pytest.raises(NotImplementedError, match=r"3-D cyl"):
            component_files(Channel.STAT_B, digit=8)

    def test_digit_0_returns_empty(self):
        assert component_files(Channel.STAT_B, digit=0) == []


class TestEnabledChannels:
    def test_geom_70(self):
        from linac_gen.io.tracewin_geom import decode_geom
        code = decode_geom(70)
        assert enabled_channels(code) == [(Channel.STAT_B, 7)]

    def test_geom_7700(self):
        from linac_gen.io.tracewin_geom import decode_geom
        code = decode_geom(7700)
        assert enabled_channels(code) == [(Channel.RF_E, 7), (Channel.RF_B, 7)]

    def test_geom_90(self):
        from linac_gen.io.tracewin_geom import decode_geom
        code = decode_geom(90)
        assert enabled_channels(code) == [(Channel.STAT_B, 9)]

    def test_geom_0_empty(self):
        from linac_gen.io.tracewin_geom import decode_geom
        assert enabled_channels(decode_geom(0)) == []
