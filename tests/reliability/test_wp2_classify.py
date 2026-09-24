"""WP2 — the reliability vocabulary of the failure engine: spare and steerer
classes (opt-in, default classification untouched), group scenarios with
per-member modes, spare candidates and knob-less exclusion in select_zone,
family headroom bounds, the objectives hook, and a demo-deck compensation."""
from __future__ import annotations

import contextlib
import copy
import io
from pathlib import Path
from types import SimpleNamespace

import pytest

from linac_gen.cli import common
from linac_gen.elements.lattice_commands import MinTransmission, SetKeOutMin, SetSizeMax
from linac_gen.elements.ncells import NCells
from linac_gen.elements.rf_gap import RFGap
from linac_gen.elements.steerer import Steerer
from linac_gen.failures import (ALL_TYPES, EXTRA_TYPES, CompensationConfig,
                                FailureKind, FailureMode, FailureScenario,
                                classify, compensate, enumerate_scenarios,
                                failable_elements, select_zone, valid_kinds)
from linac_gen.failures.compensation import (_forward_metrics, amp_bounds,
                                             family_nominals, objective_cards)
from linac_gen.io.tracewin_parser import parse_tracewin

REPO = Path(__file__).resolve().parents[2]
DEMO = REPO / "examples" / "reliability_demo"
LEBT = REPO / "examples" / "lebt_pxie" / "lebt_pxie.dat"


@pytest.fixture(scope="module")
def demo():
    with contextlib.redirect_stdout(io.StringIO()):
        lat, cfg, conv = common.load_input(str(DEMO / "reliability_demo.lgproj"))
    return lat, cfg


def _nc(name="c"):
    return NCells(name, mode=1, n_cells=2, beta_g=0.4, eot_v_per_m=1e6,
                  theta_s_deg=-30.0, aperture_mm=15.0, sync_phase=True,
                  frequency_mhz=325.0)


# --------------------------------------------------------------------------
# classification
# --------------------------------------------------------------------------
def test_default_classification_is_untouched(demo):
    lat, _ = demo
    assert ALL_TYPES == ("cavity", "quad", "solenoid", "dipole")
    assert EXTRA_TYPES == ("steerer", "spare")
    names = {n for (n, _l, _c) in failable_elements(lat)}
    assert names == {"GAP_001", "GAP_002", "GAP_003", "GAP_004", "SOL_001", "SOL_002", "SOL_003"}
    st = next(e for e in lat.elements if isinstance(e, Steerer))
    assert classify(st) is None and classify(st, extended=True) == "steerer"
    assert classify(_nc()) is None and classify(_nc(), extended=True) == "cavity"
    assert valid_kinds("steerer") == {FailureKind.OFF} and valid_kinds("spare") == set()
    assert valid_kinds("cavity") == {FailureKind.OFF, FailureKind.DETUNE}


@pytest.mark.skipif(not LEBT.exists(),
                    reason="examples/lebt_pxie is dev-only (not in the public tree)")
def test_spare_is_an_unpowered_field_map_only_in_extended_mode():
    lat, meta = parse_tracewin(str(LEBT))
    assert meta.get("warnings") == []
    default = failable_elements(lat)
    assert [(n, l) for (n, l, _c) in default] == [
        ("FMAP_001", "solenoid"), ("FMAP_002", "solenoid"), ("FMAP_003", "solenoid")]
    assert failable_elements(lat, types={"spare"}) == []           # all powered
    spare = copy.deepcopy(lat)
    fm = next(e for e in spare.elements if e.name == "FMAP_002")
    fm.kb = 0.0
    assert classify(fm) == "solenoid"                              # historical rule
    assert classify(fm, extended=True) == "spare"
    assert [(n, l) for (n, l, _c) in failable_elements(spare, types={"spare"})] == [("FMAP_002", "spare")]
    # extended mode re-labels the unpowered map for the whole selection
    ext = failable_elements(spare, types={"solenoid", "spare"})
    assert [(n, l) for (n, l, _c) in ext] == [
        ("FMAP_001", "solenoid"), ("FMAP_002", "spare"), ("FMAP_003", "solenoid")]
    assert [(n, l) for (n, l, _c) in failable_elements(spare)] == [
        ("FMAP_001", "solenoid"), ("FMAP_002", "solenoid"), ("FMAP_003", "solenoid")]
    # a spare never enumerates as a failure
    scen, _n2c, names = enumerate_scenarios(spare, types={"spare"})
    assert scen == [] and names == []


def test_steerer_scenarios_are_absolute_off(demo):
    lat, _ = demo
    scen, n2c, names = enumerate_scenarios(lat, types={"steerer"})
    assert names == ["STEER_001", "STEER_002", "STEER_003"]
    assert scen[0].element_overrides(n2c) == (("STEER_001.bx_l", 0.0), ("STEER_001.by_l", 0.0))
    partial, _, _ = enumerate_scenarios(lat, types={"steerer"}, kind=FailureKind.PARTIAL, amp_scale=0.9)
    assert partial == []                                            # filtered, not raised


def test_group_scenario_with_per_member_modes(demo):
    lat, _ = demo
    _s, n2c, _n = enumerate_scenarios(lat)
    s = FailureScenario(failures=(("GAP_001", FailureMode(FailureKind.OFF)),
                                  ("SOL_001", FailureMode(FailureKind.PARTIAL, amp_scale=0.9))),
                        label="CM1 trip")
    ov = s.element_overrides(n2c)
    assert ov == (("GAP_001.voltage_rel", -1.0), ("SOL_001.field_rel", pytest.approx(-0.1)))
    assert s.element_names == ("GAP_001", "SOL_001") and s.label == "CM1 trip"


# --------------------------------------------------------------------------
# zone selection, bounds, objectives
# --------------------------------------------------------------------------
class FieldMap:                                   # classify keys on the class NAME
    def __init__(self, name, ke, kb):
        self.name, self.ke, self.kb = name, ke, kb


def _fake_line():
    return SimpleNamespace(elements=[
        RFGap("C1", 1.0, -20.0, 325.0), FieldMap("SP", 0.0, 0.0),
        RFGap("C2", 1.0, -20.0, 325.0), _nc("NC"), Steerer("ST"),
        RFGap("C3", 1.0, -20.0, 325.0)])


def test_select_zone_spares_opt_in_and_knobless_excluded():
    lat = _fake_line()
    assert select_zone(lat, ["C2"], CompensationConfig(k=1)) == ["C1", "C3"]
    assert select_zone(lat, ["C2"], CompensationConfig(k=1, extra_types=("spare",))) == ["SP", "C1"]
    assert select_zone(lat, ["C2"], CompensationConfig(k=2, extra_types=("spare", "steerer"))) == ["SP", "C1", "C3"]
    assert select_zone(lat, ["C2"], CompensationConfig(strategy="manual",
                       manual_names=["SP", "NC", "ST", "C3"])) == ["C3"]
    assert select_zone(lat, ["C2"], CompensationConfig(strategy="manual", extra_types=("spare",),
                       manual_names=["SP", "NC", "ST", "C3"])) == ["SP", "C3"]


def test_family_nominals_and_headroom_bounds():
    lat = _fake_line()
    lat.elements[0].voltage = 1.2; lat.elements[2].voltage = 1.0; lat.elements[5].voltage = 0.8
    cfg = CompensationConfig(family_of={"C1": "F", "C2": "F", "C3": "F", "SP": "F"},
                             amp_bounds_by_family={"F": (0.0, 1.5)})
    nom = family_nominals(lat, cfg)
    assert nom == {"F": 1.0}                                    # signed median of the powered
    assert amp_bounds(cfg, "SP", 0.0, nom) == (0.0, 1.5)         # a spare gets the family window
    assert amp_bounds(cfg, "C3", 0.8, nom) == (0.0, 1.5)
    neg = CompensationConfig(family_of={"X": "G"}, amp_bounds_by_family={"G": (0.5, 1.72)})
    assert amp_bounds(neg, "X", -2.0, {"G": -2.0}) == (-3.44, -1.0)
    plain = CompensationConfig()                                 # the historical rule
    assert amp_bounds(plain, "C1", 2.0, {}) == (1.0, 3.0)
    assert amp_bounds(plain, "SP", 0.0, {}) == (-1.0, 1.0)      # cur == 0 guard
    assert family_nominals(lat, plain) == {}
    # a family with no powered member falls back to the current value
    lone = CompensationConfig(family_of={"SP": "Z"}, amp_bounds_by_family={"Z": (0.0, 2.0)})
    assert amp_bounds(lone, "SP", 0.0, {}) == (-1.0, 1.0)


def test_objective_cards_defaults_and_size_ceiling():
    base = {"ref_w_kin": 23.7, "transmission": 99.5}
    lead, trail = objective_cards(CompensationConfig(cost_solver="envelope"), base, 30)
    assert lead == [] and [type(c) for c in trail] == [SetKeOutMin]
    assert trail[0].energy_mev == 23.7 and trail[0].weight == 10.0
    lead, trail = objective_cards(CompensationConfig(cost_solver="mp"), base, 30)
    assert [type(c) for c in trail] == [SetKeOutMin, MinTransmission]
    assert trail[1].threshold_pct == pytest.approx(99.0)
    lead, trail = objective_cards(CompensationConfig(
        objectives={"size_max": {"x_mm": 3.0, "y_mm": 2.5, "weight": 2.0}, "ke_weight": 4.0}), base, 30)
    assert [type(c) for c in lead] == [SetSizeMax]
    assert (lead[0].n_elems, lead[0].x_mm, lead[0].y_mm, lead[0].k) == (30, 3.0, 2.5, 2.0)
    assert trail[0].weight == 4.0
    lead, trail = objective_cards(CompensationConfig(objectives={"ke_out_min": False}), base, 30)
    assert lead == [] and trail == []
    from linac_gen.elements.lattice_commands import MinEmitGrowth
    _l, trail = objective_cards(CompensationConfig(objectives={"emit_growth": True, "emit_weight": 2.0}), base, 30)
    assert [type(c) for c in trail] == [SetKeOutMin, MinEmitGrowth, MinEmitGrowth, MinEmitGrowth]
    assert [c.plane for c in trail[1:]] == ["X", "Y", "Z"] and trail[1].weight == 2.0
    assert objective_cards(CompensationConfig(), {}, 30) == ([], [])


def test_compensate_demo_last_gap_with_two_neighbours(demo):
    lat, cfg = demo
    scen, n2c, _ = enumerate_scenarios(lat, types={"cavity"})
    s = next(x for x in scen if x.element_names == ("GAP_004",))
    baseline = _forward_metrics(copy.deepcopy(lat), cfg, "envelope")
    comp = CompensationConfig(strategy="k_out_of_n", k=1, algorithm="least_squares",
                              cost_solver="envelope", max_iter=80,
                              family_of={g: "GAP" for g in n2c},
                              amp_bounds_by_family={"GAP": (0.0, 1.5)},
                              objectives={"size_max": {"x_mm": 4.0, "y_mm": 4.0, "weight": 1.0}})
    res = compensate(copy.deepcopy(lat), cfg, s, n2c, baseline, comp)
    assert res.compensator_names == ["GAP_003", "GAP_002"]
    assert res.match_success and res.recovered
    assert abs(res.metrics_after["ref_w_kin"] - baseline["ref_w_kin"]) <= comp.recover_tol_energy_mev
    assert set(res.settings) == {"GAP_003.voltage", "GAP_003.phase", "GAP_002.voltage", "GAP_002.phase"}
    assert all(0.0 <= res.settings[k] <= 1.5 * 1.0 + 1e-9 for k in ("GAP_003.voltage", "GAP_002.voltage"))
