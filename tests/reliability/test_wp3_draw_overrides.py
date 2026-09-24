"""WP3 — ErrorStudy.draw_overrides: the transported draws reproduce the
engine's own errored copy attribute for attribute with its RNG streams
and logs untouched; untransportable draws are reported; the budget →
study → audit policy; the pool worker with orbit correction."""
from __future__ import annotations

import contextlib
import copy
import io
import json
import warnings
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest

from linac_gen.cli import common
from linac_gen.errors.error_model import ErrorStudy, lattice_overrides
from linac_gen.reliability import (ErrorBudget, audit_or_raise, build_error_study,
                                   draw_seed, steerer_kick_overrides)
from tests.dataguard import needs

REPO = Path(__file__).resolve().parents[2]
DEMO = REPO / "examples" / "reliability_demo"

BUDGET = {
    "element": [
        {"pattern": "SOL_*", "parameter": "dx", "sigma": 0.2},
        {"pattern": "SOL_*", "parameter": "dy", "sigma": 0.2},
        {"pattern": "SOL_*", "parameter": "field_rel", "sigma": 0.005, "kind": "Solenoid"},
        {"pattern": "GAP_*", "parameter": "voltage_rel", "sigma": 0.01},
        {"pattern": "GAP_*", "parameter": "phase_offset", "sigma": 1.0},
        {"pattern": "GAP_00[12]", "parameter": "dx", "distribution": "uniform", "half_width": 0.1},
    ],
    "beam": [{"parameter": "centroid_x", "sigma": 0.1},
             {"parameter": "emit_nx_rel", "sigma": 0.05}],
}


@pytest.fixture(scope="module")
def demo():
    with contextlib.redirect_stdout(io.StringIO()):
        lat, cfg, conv = common.load_input(str(DEMO / "reliability_demo.lgproj"))
    return lat, cfg


def _public_numbers(el):
    return {k: v for k, v in vars(el).items()
            if not k.startswith("_") and isinstance(v, (int, float)) and not isinstance(v, bool)}


def test_draw_overrides_reproduce_apply_errors_and_leave_logs_identical(demo):
    lat, cfg = demo
    budget = ErrorBudget.from_dict(BUDGET)
    a = build_error_study(copy.deepcopy(lat), cfg, budget)
    b = build_error_study(copy.deepcopy(lat), cfg, budget)
    ref_lat = a._apply_errors(7)
    ref_cfg = a._apply_beam_errors(7)
    draw = b.draw_overrides(7)
    assert draw.seed == 7 and draw.untransportable == () and len(draw.overrides) == 19
    fresh = copy.deepcopy(lat)
    for sel, val in draw.overrides:
        common.apply_element_override(fresh, sel, val)
    for x, y in zip(fresh.elements, ref_lat.elements):
        assert _public_numbers(x) == _public_numbers(y), getattr(x, "name", x)
    assert asdict(draw.beam_config) == asdict(ref_cfg)
    assert draw.applied == a.applied_log[7] and draw.beam_applied == a.beam_applied_log[7]
    assert b.applied_log[7] == a.applied_log[7]                    # same streams
    # the errored copy is the same object the engine would track
    for x, y in zip(draw.lattice.elements, ref_lat.elements):
        assert _public_numbers(x) == _public_numbers(y)


def test_draws_are_reproducible_and_faults_consume_no_entropy(demo):
    lat, cfg = demo
    budget = ErrorBudget.from_dict(BUDGET)
    s = build_error_study(copy.deepcopy(lat), cfg, budget)
    d1 = s.draw_overrides(3)
    d2 = s.draw_overrides(3)
    assert d1.overrides == d2.overrides and asdict(d1.beam_config) == asdict(d2.beam_config)
    fault = (("GAP_002.voltage_rel", -1.0),)
    with_fault = copy.deepcopy(lat)
    for sel, val in d1.overrides + fault:
        common.apply_element_override(with_fault, sel, val)
    g2 = next(e for e in with_fault.elements if e.name == "GAP_002")
    assert g2.voltage_rel == -1.0 and g2.effective_voltage == 0.0
    # errors-only vs faults+errors: the draw is the same object either way
    assert s.draw_overrides(3).overrides == d1.overrides
    assert s.draw_overrides(4).overrides != d1.overrides


def test_untransportable_draws_are_reported_once(demo):
    lat, cfg = demo
    budget = ErrorBudget.from_dict({"element": [{"pattern": "BPM_*", "parameter": "dx", "sigma": 0.1}]})
    s = build_error_study(copy.deepcopy(lat), cfg, budget)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        d = s.draw_overrides(1)
        s.draw_overrides(2)
    assert d.overrides == () and [n for n, _p in d.untransportable] == ["BPM_001", "BPM_002", "BPM_003"]
    msgs = [str(x.message) for x in w if "cannot be transported" in str(x.message)]
    assert len(msgs) == 1 and "BPM_001.dx" in msgs[0]
    with pytest.raises(ValueError, match="element counts differ"):
        lattice_overrides(lat, copy.deepcopy(lat).__class__())


def test_budget_build_and_audit_policy(demo):
    lat, cfg = demo
    budget = ErrorBudget.from_dict(BUDGET)
    assert budget.to_dict() == {"element": BUDGET["element"], "beam": BUDGET["beam"]}
    s = build_error_study(copy.deepcopy(lat), cfg, budget, n_seeds=5, base_seed=100,
                          correction={"enabled": True, "n_iter": 3, "reading_backend": "envelope"})
    assert len(s._errors) == 6 and len(s._beam_errors) == 2 and s.base_seed == 100
    assert s._correction_enabled and s._correction_kwargs["n_iter"] == 3
    audit = audit_or_raise(s)
    assert {r["parameter"] for r in audit["defs"]} == {"dx", "dy", "field_rel", "voltage_rel", "phase_offset"}
    for bad, msg in ((
        {"element": [{"pattern": "NOPE_*", "parameter": "dx", "sigma": 0.1}]}, "matches no element"),
        ({"element": [{"pattern": "SOL_*", "parameter": "dz", "sigma": 0.1}]}, "inert"),
        ({"element": [{"pattern": "SOL_*", "parameter": "dx", "sigma": 0.1},
                      {"pattern": "SOL_00*", "parameter": "dx", "sigma": 0.1}]}, "overlapping"),
        ({"element": [{"pattern": "GAP_*", "parameter": "gradient_rel", "sigma": 0.1}]}, "warn-skipped"),
        ({"beam": [{"parameter": "nothing_here", "sigma": 0.1}]}, "does not exist"),
    ):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")            # add_error warns on dz itself
            st = build_error_study(copy.deepcopy(lat), cfg, ErrorBudget.from_dict(bad))
        with pytest.raises(ValueError, match=msg):
            audit_or_raise(st)
    with pytest.raises(ValueError, match="needs pattern"):
        ErrorBudget.from_dict({"element": [{"parameter": "dx"}]})


def test_draw_seed_worker_with_correction_is_serialisable_and_reproducible(demo):
    lat, cfg = demo
    job = {"lattice_path": str(DEMO / "reliability_demo.dat"), "beam_config": asdict(cfg),
           "budget": BUDGET, "seed": 5,
           "correction": {"enabled": True, "n_iter": 6, "tol_mm": 1e-3, "reading_backend": "envelope"}}
    r = draw_seed(job)
    assert r["error"] is None, r.get("traceback")
    json.dumps(r)
    assert r["correction"]["status"] == "converged" and r["correction"]["n_pairs"] == 3
    assert r["kick_overrides"] and all(sel.split(".")[1] in ("bx_l", "by_l") and sel.startswith("@")
                                       for sel, _v in r["kick_overrides"])
    r2 = draw_seed(job)
    for k in ("overrides", "beam_config", "kick_overrides", "applied"):
        assert r2[k] == r[k], k
    # the transported seed: errors + kicks on a fresh copy steer the BPM orbit down
    def bpm_rms(overrides):
        work = copy.deepcopy(lat)
        for sel, val in overrides:
            common.apply_element_override(work, sel, val)
        from linac_gen.core.config import BeamConfig
        res = common.run_envelope_sim(work, BeamConfig(**r["beam_config"]))
        vals = []
        for i, e in enumerate(work.elements):
            if getattr(e, "is_bpm", False):
                c = res.centroid[res.element_exit_idx[i]] if hasattr(res, "element_exit_idx") else None
                if c is not None:
                    vals += [float(c[0]), float(c[2])]
        return float(np.sqrt(np.mean(np.square(vals)))) if vals else None
    before = bpm_rms([tuple(o) for o in r["overrides"]])
    after = bpm_rms([tuple(o) for o in r["overrides"]] + [tuple(o) for o in r["kick_overrides"]])
    if before is not None and after is not None:
        assert after < before
    bad = draw_seed({**job, "budget": {"element": [{"pattern": 3}]}})
    assert bad["error"] and "traceback" in bad


@needs("examples/pip2_misalignment_study/run_study.py")
def test_dev_budget_converts_through_the_reliability_shape():
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "run_study", REPO / "examples" / "pip2_misalignment_study" / "run_study.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    b = ErrorBudget.from_study_budget(mod.BUDGET, mod.BEAM_BUDGET)
    n_expected = sum(len(s.get("patterns") or [s["pattern"]]) * sum(1 for _p, sg, _u in s["errors"] if sg > 0)
                     for s in mod.BUDGET.values())
    assert len(b.element) == n_expected and all(r["distribution"] == "gaussian" for r in b.element)
    assert len(b.beam) == sum(1 for _p, sg, _u in mod.BEAM_BUDGET if sg > 0)
