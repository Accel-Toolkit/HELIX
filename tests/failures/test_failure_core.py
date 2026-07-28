"""Unit tests for the failure-analysis core (modes, filter, scenarios, score)."""
from __future__ import annotations

import math

import pytest

from linac_gen.cli.common import apply_element_override
from linac_gen.core.lattice import Lattice
from linac_gen.elements.drift import Drift
from linac_gen.elements.quadrupole import Quadrupole
from linac_gen.elements.rf_gap import RFGap
from linac_gen.elements.solenoid import Solenoid
from linac_gen.elements.steerer import Steerer
from linac_gen.failures.criticality import criticality_score
from linac_gen.failures.element_filter import (
    classify, failable_elements, valid_kinds)
from linac_gen.failures.failure_mode import FailureKind, FailureMode
from linac_gen.failures.scenario import enumerate_scenarios


def _lattice():
    """Drift + 2 quads + 1 solenoid + 2 cavities + 1 steerer."""
    lat = Lattice()
    lat.add(Drift("D1", length=100.0, aperture=20.0))
    lat.add(Quadrupole("Q1", length=50.0, gradient=5.0, aperture=20.0))
    lat.add(Solenoid("S1", length=100.0, field=0.3, aperture=20.0))
    lat.add(RFGap("CAV1", voltage=1.0, phase=-25.0, frequency=162.5, aperture=20.0))
    lat.add(Quadrupole("Q2", length=50.0, gradient=-5.0, aperture=20.0))
    lat.add(RFGap("CAV2", voltage=1.5, phase=-20.0, frequency=162.5, aperture=20.0))
    lat.add(Steerer("ST1", bx_l=0.001, by_l=0.0))
    return lat


# ---- failure_mode: sign conventions -----------------------------------
def test_off_is_minus_one():
    assert FailureMode(FailureKind.OFF).element_overrides("Q1", "Quadrupole") == \
        (("Q1.gradient_rel", -1.0),)


def test_partial_90pct_is_minus_010():
    ov = FailureMode(FailureKind.PARTIAL, amp_scale=0.90).element_overrides(
        "Q1", "Quadrupole")
    assert ov == (("Q1.gradient_rel", pytest.approx(-0.10)),)  # NOT 0.90


def test_detune_amp_and_phase():
    ov = FailureMode(FailureKind.DETUNE, amp_scale=0.8, phase_deg=10.0
                     ).element_overrides("CAV1", "RFGap")
    assert ("CAV1.voltage_rel", pytest.approx(-0.2)) in ov
    assert ("CAV1.phase_offset", 10.0) in ov


def test_detune_phase_only():
    ov = FailureMode(FailureKind.DETUNE, amp_scale=1.0, phase_deg=10.0
                     ).element_overrides("CAV1", "RFGap")
    assert ov == (("CAV1.phase_offset", 10.0),)


def test_detune_on_magnet_raises():
    with pytest.raises(ValueError):
        FailureMode(FailureKind.DETUNE, amp_scale=0.8).element_overrides(
            "Q1", "Quadrupole")


def test_steerer_off_absolute_and_partial_raises():
    assert FailureMode(FailureKind.OFF).element_overrides("ST1", "Steerer") == \
        (("ST1.bx_l", 0.0), ("ST1.by_l", 0.0))
    with pytest.raises(ValueError):
        FailureMode(FailureKind.PARTIAL, amp_scale=0.9).element_overrides(
            "ST1", "Steerer")


# ---- injection correctness via the REAL override path -----------------
def test_injection_zeroes_and_scales_effective_values():
    lat = _lattice()
    q1 = next(e for e in lat.elements if e.name == "Q1")
    cav = next(e for e in lat.elements if e.name == "CAV1")
    v0, ph0 = cav.voltage, cav.phase

    apply_element_override(lat, "Q1.gradient_rel", -1.0)        # OFF
    assert q1.effective_gradient == pytest.approx(0.0)

    apply_element_override(lat, "CAV1.voltage_rel", -0.2)       # DETUNE amp 80%
    apply_element_override(lat, "CAV1.phase_offset", 10.0)      # DETUNE phase +10
    assert cav.effective_voltage == pytest.approx(0.8 * v0)
    assert cav.effective_phase == pytest.approx(ph0 + 10.0)


# ---- element_filter ---------------------------------------------------
def test_failable_excludes_drift_and_includes_active():
    lat = _lattice()
    names = {n for (n, _l, _c) in failable_elements(lat)}
    assert "D1" not in names                      # drift can't fail
    assert "ST1" not in names                     # steerer excluded by default
    assert {"Q1", "Q2", "S1", "CAV1", "CAV2"} <= names
    assert classify(next(e for e in lat.elements if e.name == "D1")) is None
    assert classify(next(e for e in lat.elements if e.name == "ST1")) is None
    assert FailureKind.DETUNE in valid_kinds("cavity")
    assert FailureKind.DETUNE not in valid_kinds("quad")


def test_type_filter():
    lat = _lattice()
    cav = {n for (n, _l, _c) in failable_elements(lat, types={"cavity"})}
    assert cav == {"CAV1", "CAV2"}


# ---- scenario enumeration --------------------------------------------
def test_single_count():
    lat = _lattice()
    scn, _, names = enumerate_scenarios(lat, kind=FailureKind.OFF,
                                        combination="single")
    assert len(scn) == len(names)                 # one per failable element
    assert all(len(s.failures) == 1 for s in scn)


def test_pairs_count_includes_singles_for_diagonal():
    lat = _lattice()
    scn, _, names = enumerate_scenarios(lat, kind=FailureKind.OFF,
                                        combination="pairs")
    n = len(names)
    assert len(scn) == n + n * (n - 1) // 2       # singles + unordered pairs


def test_detune_filters_to_cavities():
    lat = _lattice()
    scn, n2c, names = enumerate_scenarios(lat, kind=FailureKind.DETUNE,
                                          amp_scale=0.8, combination="single")
    assert set(names) == {"CAV1", "CAV2"}         # magnets dropped


def test_custom_sets():
    lat = _lattice()
    scn, _, _ = enumerate_scenarios(
        lat, kind=FailureKind.OFF, combination="custom",
        custom_sets=[["CAV1", "Q2"], ["S1"]])
    assert len(scn) == 2
    assert scn[0].element_names == ("CAV1", "Q2")


# ---- criticality ------------------------------------------------------
def test_criticality_monotonic_and_none_safe():
    base = {"transmission": 100.0, "emit_nx": 1.0, "emit_ny": 1.0,
            "emit_nz": 1.0, "ref_w_kin": 10.0}
    mild = {"transmission": 99.0, "emit_nx": 1.1, "emit_ny": 1.0,
            "emit_nz": 1.0, "ref_w_kin": 9.9}
    severe = {"transmission": 50.0, "emit_nx": 2.0, "emit_ny": 1.0,
              "emit_nz": 1.0, "ref_w_kin": 5.0}
    s_mild, _, lost_m = criticality_score(base, mild)
    s_sev, _, lost_s = criticality_score(base, severe)
    assert 0 < s_mild < s_sev
    assert not lost_m and not lost_s

    # envelope mode: transmission None -> that term is 0, not an error
    env_base = {"transmission": None, "emit_nx": 1.0, "emit_ny": 1.0,
                "emit_nz": None, "ref_w_kin": 10.0}
    env_scn = {"transmission": None, "emit_nx": 1.5, "emit_ny": 1.0,
               "emit_nz": None, "ref_w_kin": 8.0}
    s, terms, lost = criticality_score(env_base, env_scn)
    assert terms["transmission"] == 0.0 and terms["emit_z"] == 0.0
    assert s > 0 and not lost

    # NaN / missing exit energy -> beam_lost
    _, _, lost_nan = criticality_score(
        base, {"transmission": 0.0, "ref_w_kin": float("nan")})
    assert lost_nan
