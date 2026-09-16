"""orm_calibrate / orm_apply assistant tools (tools_analysis.py).

Runs the ORM comparison, the LOCO-style fit and the synthetic validation
off the session context on the orm_demo FODO with a synthetic FORMA
export; checks the JSON envelope, that the session lattice is never
touched by the fit, the file outputs, and that orm_apply goes through
``ctx.apply_param_changes`` (one undoable command in the GUI)."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from linac_gen.assist.tools import TOOLS, WorkContext

REPO = Path(__file__).resolve().parents[2]
DEMO = REPO / "examples" / "orm_demo"


@pytest.fixture(scope="module")
def synthetic(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("orm_tool")
    spec = importlib.util.spec_from_file_location("make_synthetic_forma", DEMO / "make_synthetic_forma.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert mod.main(["--out", str(tmp / "synthetic"), "--g-sigma", "0.02", "--noise", "0.005", "--seed", "3"]) == 0
    return [str(tmp / "synthetic/20260101_000000_H"), str(tmp / "synthetic/20260101_000000_V")]


def _ctx(tmp_path):
    from linac_gen.cli.common import load_input
    lat, cfg, _ = load_input(str(DEMO / "fodo_orm.lgproj"))
    c = WorkContext(calc_dir=str(tmp_path))
    c.set_lattice(lat, str(DEMO / "fodo_orm.dat"))
    c.set_beam_config(cfg)
    return c


def _quads(lat):
    from linac_gen.elements.quadrupole import Quadrupole
    return [q for q in lat.elements if isinstance(q, Quadrupole)]


def test_compare_reports_selection_and_metrics(tmp_path, synthetic):
    ctx = _ctx(tmp_path)
    res = TOOLS["orm_calibrate"].fn(ctx, measured=synthetic, check_tracking=True)
    assert res["status"] == "ok", res
    d = res["data"]
    assert d["mode"] == "compare" and d["selection"]["n_bpm"] == 12 and d["selection"]["n_trim"] == 6
    for p in ("x", "y"):
        pl = d["compare"]["planes"][p]
        assert pl["n_entries"] == 42 and len(pl["per_trim"]) == 6
        assert np.median([abs(r["r"]) for r in pl["per_trim"] if r["r"] is not None]) > 0.98
    assert d["compare"]["reason"] is None and d["tracking_max_rel_dev"] < 1e-9
    assert res["provenance"]["measured"] == synthetic
    json.dumps(res, allow_nan=False)                 # strictly JSON-serialisable (NaN -> null)


def test_fit_keeps_session_lattice_untouched_and_writes_files(tmp_path, synthetic):
    ctx = _ctx(tmp_path)
    before = [(q.gradient, q.gradient_rel) for q in _quads(ctx.lattice)]
    cal_path = tmp_path / "cal.json"; deck = tmp_path / "fodo_fit.dat"
    progress = []
    res = TOOLS["orm_calibrate"].fn(ctx, measured=synthetic, mode="fit", calibration_out=str(cal_path),
                                    export_deck=str(deck), progress_callback=progress.append)
    assert res["status"] == "ok", res
    f = res["data"]["fit"]
    assert f["stage"] == "trims+quads" and f["converged"] and len(f["quads"]) == 12 and f["summary"]
    for p in ("x", "y"):
        assert f["metrics"]["after"][p]["nrms_all"] < 0.5 * f["metrics"]["before"][p]["nrms_all"]
    assert abs(f["quads"][0]["scale"] - 1) >= abs(f["quads"][-1]["scale"] - 1)      # largest change first
    assert res["data"]["files"] == {"calibration": str(cal_path), "deck": str(deck)}
    assert cal_path.exists() and deck.exists() and f["deck_quads_rescaled"] >= 8
    assert ctx.orm_calibration["__kind__"] == "helix_orm_calibration"
    assert [(q.gradient, q.gradient_rel) for q in _quads(ctx.lattice)] == before      # the fit ran on a copy
    assert progress and all(0.0 <= v <= 0.95 for v in progress)
    json.dumps(res, allow_nan=False)


def test_apply_session_calibration_rel_then_bake_from_file(tmp_path, synthetic):
    ctx = _ctx(tmp_path)
    assert TOOLS["orm_apply"].fn(ctx)["status"] == "refused"                 # nothing fitted yet
    assert TOOLS["orm_calibrate"].fn(ctx, measured=synthetic, mode="fit")["status"] == "ok"
    seen = []
    orig = ctx.apply_param_changes

    def spy(changes, label=""):
        seen.append((list(changes), label))
        return orig(changes, label)
    ctx.apply_param_changes = spy
    res = TOOLS["orm_apply"].fn(ctx)
    assert res["status"] == "ok", res
    assert res["data"]["n_applied"] >= 8 and res["data"]["source"] == "session"
    assert len(seen) == 1 and seen[0][1] == "ORM calibration"                # ONE undoable command in the GUI
    assert all(attr == "gradient_rel" for _q, attr, _v in seen[0][0])
    by_label = {q.label: q for q in _quads(ctx.lattice)}
    for row in res["data"]["changes"]:
        assert row["attr"] == "gradient_rel" and by_label[row["label"]].gradient_rel == row["new"]
    # bake from a file into a fresh session: gradients change, gradient_rel stays 0
    from linac_gen.orm import save_calibration
    cal_path = tmp_path / "cal2.json"
    save_calibration(cal_path, ctx.orm_calibration)
    ctx2 = _ctx(tmp_path)
    res = TOOLS["orm_apply"].fn(ctx2, calibration_file=str(cal_path), mode="bake")
    assert res["status"] == "ok" and res["data"]["source"] == str(cal_path)
    q2 = {q.label: q for q in _quads(ctx2.lattice)}
    for row in res["data"]["changes"]:
        assert row["attr"] == "gradient" and q2[row["label"]].gradient == row["new"] and q2[row["label"]].gradient_rel == 0.0
    # design-gradient drift -> atomic refusal, nothing applied
    ctx3 = _ctx(tmp_path)
    _quads(ctx3.lattice)[0].gradient *= 1.01
    res = TOOLS["orm_apply"].fn(ctx3, calibration_file=str(cal_path))
    assert res["status"] == "refused"
    assert all(q.gradient_rel == 0.0 for q in _quads(ctx3.lattice))
    json.dumps(res, allow_nan=False)


def test_validate_mode(tmp_path, synthetic):
    ctx = _ctx(tmp_path)
    res = TOOLS["orm_calibrate"].fn(ctx, measured=synthetic, mode="validate", seed=2, g_sigma=0.02)
    assert res["status"] == "ok", res
    v = res["data"]["validation"]
    assert v["g_rms_recovered"] < v["g_rms_injected"] and len(v["worst_quads"]) <= 5 and v["converged"]
    json.dumps(res, allow_nan=False)


def test_refusals(tmp_path, synthetic):
    ctx = _ctx(tmp_path)
    assert TOOLS["orm_calibrate"].fn(WorkContext(calc_dir=str(tmp_path)), measured=synthetic)["status"] == "error"
    assert TOOLS["orm_calibrate"].fn(ctx, measured=["https://example.invalid/orm"])["status"] == "refused"
    assert TOOLS["orm_calibrate"].fn(ctx, measured=[str(tmp_path / "nope")])["status"] == "error"
    assert TOOLS["orm_calibrate"].fn(ctx, measured=synthetic, mode="bogus")["status"] == "refused"
    assert TOOLS["orm_calibrate"].fn(ctx, measured=synthetic, mode="fit", stage="quads")["status"] == "refused"
    from tests.orm.conftest import make_forma_folder
    f = make_forma_folder(tmp_path, stamp="20260102_000000", kick_plane="x", trims=["L:D99TMH"],
                          bpms={"x": ["L:BPH9ZZ"], "y": ["L:BPV9ZZ"]}, values={"x": np.ones((1, 1)), "y": np.zeros((1, 1))})
    res = TOOLS["orm_calibrate"].fn(ctx, measured=[str(f)])
    assert res["status"] == "refused" and "L:D99TMH" in res["data"]["message"]
    assert TOOLS["orm_apply"].fn(ctx, calibration_file="https://x/cal.json")["status"] == "refused"
    assert TOOLS["orm_apply"].fn(ctx, calibration_file=str(tmp_path / "none.json"))["status"] == "error"
    assert TOOLS["orm_apply"].fn(WorkContext(calc_dir=str(tmp_path)))["status"] == "error"


def test_registry_and_mcp():
    from linac_gen.assist.agent import LONG_RUNNING
    from linac_gen.assist.mcp_server import tool_specs
    assert "orm_calibrate" in LONG_RUNNING and "orm_apply" not in LONG_RUNNING
    assert TOOLS["orm_calibrate"].tier == "compute" and TOOLS["orm_apply"].tier == "mutate"
    names = [t["name"] for t in tool_specs()]
    assert "orm_calibrate" in names and "orm_apply" in names
