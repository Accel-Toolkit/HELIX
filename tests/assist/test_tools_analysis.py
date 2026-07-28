"""Wave-1 analysis tools: run_python (registry), shift_data, diagnose,
anomaly baseline/check — synthetic results + fabricated ledgers."""
from __future__ import annotations

import json
import os
import time
from types import SimpleNamespace

import numpy as np
import pytest

from linac_gen.assist.tools import TOOLS


def _results(n=200, drop_at=None, blow_y_at=None, emit_growth=1.0):
    """Synthetic recorder-shaped results over a 5 m line (s in mm)."""
    s = np.linspace(0.0, 5000.0, n)
    tr = np.full(n, 100.0)
    if drop_at is not None:
        tr[s >= drop_at] = 97.0
    sy = np.full(n, 1.5)
    if blow_y_at is not None:
        sy[s >= blow_y_at] = 12.0
    return SimpleNamespace(
        s=s, transmission=tr,
        sigma_x=np.full(n, 2.0), sigma_y=sy,
        sigma_phi=np.full(n, 5.0), sigma_w=np.full(n, 0.01),
        emit_x=np.linspace(0.25, 0.25 * emit_growth, n),
        emit_y=np.full(n, 0.25), emit_z=np.full(n, 0.30),
        alpha_x=np.zeros(n), beta_x=np.ones(n),
        alpha_y=np.zeros(n), beta_y=np.ones(n),
        alpha_z=np.zeros(n), beta_z=np.ones(n),
        ref_w_kin=np.linspace(3.0, 3.0, n))


def _write_ledger(ctx, records):
    d = os.path.join(ctx.calc_dir, "assist_sessions")
    os.makedirs(d, exist_ok=True)
    p = os.path.join(d, "20990101_000000_1.jsonl")
    with open(p, "w", encoding="utf-8") as f:
        for r in records:
            r.setdefault("ts", time.strftime(
                "%Y-%m-%dT%H:%M:%S%z", time.localtime()))
            f.write(json.dumps(r) + "\n")
    return p


# ---------------------------------------------------------------------------
# run_python through the registry
# ---------------------------------------------------------------------------
def test_run_python_registered_as_compute():
    t = TOOLS["run_python"]
    assert t.tier == "compute"


def test_run_python_include_requires_results(ctx):
    out = TOOLS["run_python"].fn(ctx, code="print(1)", include=["s"])
    assert out["status"] == "error"          # no results loaded


def test_run_python_include_unknown_column_refused(ctx):
    ctx.set_results(_results())
    out = TOOLS["run_python"].fn(ctx, code="print(1)",
                                 include=["bogus_column"])
    assert out["status"] == "refused"


def test_run_python_arrays_reach_the_child(ctx):
    ctx.set_results(_results(n=50))
    out = TOOLS["run_python"].fn(
        ctx, code="print(int(arrays['transmission'].size))",
        include=["transmission"])
    assert out["status"] == "ok"
    assert out["data"]["stdout"].strip() == "50"


# ---------------------------------------------------------------------------
# shift_data
# ---------------------------------------------------------------------------
def test_shift_data_counts_ledger_activity(ctx):
    _write_ledger(ctx, [
        {"event": "tool", "tool": "run_mp", "tier": "compute",
         "approved_by": "user", "status": "ok"},
        {"event": "tool", "tool": "set_element_param", "tier": "mutate",
         "approved_by": "user", "status": "ok"},
        {"event": "tool", "tool": "run_mp", "tier": "compute",
         "approved_by": "denied", "status": "denied"},
        {"event": "user", "turn": 1, "text": "hi"},
    ])
    out = TOOLS["shift_data"].fn(ctx, hours=8.0)
    assert out["status"] == "ok"
    d = out["data"]
    assert d["tool_usage"]["run_mp"] == 2
    assert d["mutate_calls"] == 1
    assert d["denied_calls"] == 1
    assert "note" not in d


def test_shift_data_honest_when_empty(ctx):
    out = TOOLS["shift_data"].fn(ctx, hours=1.0)
    assert out["status"] == "ok"
    assert "no session ledgers" in out["data"]["note"]


# ---------------------------------------------------------------------------
# diagnose
# ---------------------------------------------------------------------------
def test_diagnose_without_results_suggests_running(ctx):
    out = TOOLS["diagnose"].fn(ctx)
    assert out["status"] == "ok"
    assert "no results" in out["data"]["verdict"]
    assert out["data"]["suggested_tool"] == "run_envelope"


def test_diagnose_finds_transmission_onset_and_element(ctx):
    # ctx lattice: QF(50) D1(200) QD(50) D2(200) mm — drop inside D1
    ctx.set_results(_results(drop_at=150.0))
    out = TOOLS["diagnose"].fn(ctx)
    d = out["data"]
    assert "transmission drop" in d["verdict"]
    f = d["findings"][0]
    assert f["kind"] == "transmission_drop"
    assert f["element"]["name"] == "D1"
    assert abs(f["s_m"] - 0.150) < 0.03
    assert "upstream" in f["note"]


def test_diagnose_sigma_blowup_and_emit_growth(ctx):
    ctx.set_results(_results(blow_y_at=2000.0, emit_growth=2.0))
    out = TOOLS["diagnose"].fn(ctx)
    kinds = [f["kind"] for f in out["data"]["findings"]]
    assert "sigma_y_blowup" in kinds
    assert "emit_x_growth" in kinds


def test_diagnose_reports_recent_mutations(ctx):
    _write_ledger(ctx, [
        {"event": "tool", "tool": "set_element_param", "tier": "mutate",
         "approved_by": "user", "status": "ok",
         "params": {"element": "QF", "param": "gradient"}},
    ])
    ctx.set_results(_results())
    out = TOOLS["diagnose"].fn(ctx)
    kinds = [f["kind"] for f in out["data"]["findings"]]
    assert "recent_mutations" in kinds


def test_first_drop_onset_helper():
    from linac_gen.assist.tools_analysis import first_drop_onset
    s = np.array([0.0, 10.0, 20.0, 30.0])
    assert first_drop_onset(s, np.array([100, 100, 99.0, 99.0])) == 20.0
    assert first_drop_onset(s, np.array([100, 100, 100, 100.0])) is None


# ---------------------------------------------------------------------------
# anomaly baseline / check
# ---------------------------------------------------------------------------
def test_anomaly_baseline_then_ok_check(ctx):
    ctx.set_results(_results())
    out = TOOLS["anomaly_baseline"].fn(ctx)
    assert out["status"] == "ok"
    assert os.path.isfile(out["data"]["written"])
    chk = TOOLS["anomaly_check"].fn(ctx)
    assert chk["status"] == "ok"
    assert chk["data"]["verdict"] == "ok"


def test_anomaly_check_flags_perturbation(ctx):
    ctx.set_results(_results())
    TOOLS["anomaly_baseline"].fn(ctx)
    bad = _results(drop_at=2500.0)           # 3-pt transmission drop
    ctx.set_results(bad)
    chk = TOOLS["anomaly_check"].fn(ctx)
    assert chk["data"]["verdict"] == "anomalous"
    top = chk["data"]["top_offenders"][0]["feature"]
    assert "transmission" in top


def test_anomaly_check_refuses_wrong_lattice(ctx):
    ctx.set_results(_results())
    out = TOOLS["anomaly_baseline"].fn(ctx)
    # corrupt the identity: pretend the baseline came from elsewhere
    with open(out["data"]["written"], "r+", encoding="utf-8") as f:
        payload = json.load(f)
        payload["identity"]["lattice_path"] = "/other/machine.dat"
        f.seek(0)
        json.dump(payload, f)
        f.truncate()
    chk = TOOLS["anomaly_check"].fn(ctx)
    assert chk["status"] == "refused"
    assert "DIFFERENT lattice" in chk["data"]["message"]


def test_anomaly_check_without_baseline_refuses(ctx):
    ctx.set_results(_results())
    chk = TOOLS["anomaly_check"].fn(ctx)
    assert chk["status"] == "refused"
    assert "anomaly_baseline" in chk["data"]["message"]


def test_new_tools_visible_in_roster_and_mcp():
    from linac_gen.assist.prompts import tool_roster
    r = tool_roster()
    for name in ("run_python", "shift_data", "diagnose",
                 "anomaly_baseline", "anomaly_check"):
        assert f"- {name} [" in r
    from linac_gen.assist.mcp_server import tool_specs
    names = {s["name"] for s in tool_specs()}
    assert {"run_python", "diagnose"} <= names
