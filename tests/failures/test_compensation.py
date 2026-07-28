"""Tests for fault compensation: zone selection + matcher-driven recovery."""
from __future__ import annotations

import copy

from linac_gen.cli.common import apply_beam_overrides, apply_element_override
from linac_gen.core.config import BeamConfig
from linac_gen.failures.compensation import (
    CompensationConfig, compensate, select_zone, _forward_metrics)
from linac_gen.failures.failure_mode import FailureKind
from linac_gen.failures.scenario import enumerate_scenarios
from linac_gen.io.tracewin_parser import parse_tracewin

LAT = "examples/ml_bayesopt/bo_demo.dat"
BEAM = {"energy": 2.5, "frequency": 162.5, "current": 5.0,
        "emit_nx": 0.30, "emit_ny": 0.30, "emit_z": 0.40,
        "alpha_x": -1.2, "beta_x": 0.32,
        "alpha_y": 2.0, "beta_y": 0.05, "beta_z": 10.0}


def _beam():
    cfg = BeamConfig()
    apply_beam_overrides(cfg, BEAM)
    return cfg


def test_select_zone_k_out_of_n_same_category():
    lat, _ = parse_tracewin(LAT)
    # failing a cavity selects the other cavity (same category), not solenoids
    comp = select_zone(lat, ["GAP_001"], CompensationConfig(strategy="k_out_of_n", k=2))
    assert comp == ["GAP_002"]
    # failing a solenoid selects the other solenoid
    comp = select_zone(lat, ["SOL_001"], CompensationConfig(strategy="k_out_of_n", k=2))
    assert comp == ["SOL_002"]


def test_select_zone_manual():
    lat, _ = parse_tracewin(LAT)
    comp = select_zone(lat, ["GAP_001"],
                       CompensationConfig(strategy="manual",
                                          manual_names=["GAP_002", "GAP_001"]))
    assert comp == ["GAP_002"]            # failed element excluded


def test_compensate_cavity_improves_energy_deficit():
    lat, _ = parse_tracewin(LAT)
    beam = _beam()
    _scn, n2c, _names = enumerate_scenarios(lat, kind=FailureKind.OFF,
                                            combination="single")
    scenario = next(s for s, in [(s,) for s in _scn]
                    if s.element_names == ("GAP_001",))

    baseline = _forward_metrics(lat, beam, "envelope")
    # energy with GAP_001 failed and NO compensation
    failed = copy.deepcopy(lat)
    apply_element_override(failed, "GAP_001.voltage_rel", -1.0)
    e_failed = _forward_metrics(failed, beam, "envelope")["ref_w_kin"]

    cfg = CompensationConfig(strategy="k_out_of_n", k=2,
                             algorithm="least_squares", cost_solver="envelope",
                             max_iter=80)
    res = compensate(lat, beam, scenario, n2c, baseline, cfg)

    assert res.compensator_names == ["GAP_002"]
    assert res.match_success
    e_after = res.metrics_after["ref_w_kin"]
    # the compensator ramped up -> exit energy recovered toward the baseline
    assert e_after > e_failed + 1e-6
    assert any(k.endswith(".voltage") for k in res.settings)
