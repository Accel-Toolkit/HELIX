"""run_train assistant tool (M8) + train-file routing in the
load_results / result_summary tooling.

Contract under test (plan §3b): the tool inherits TrainConfig's loud
validation VERBATIM — missing physics inputs are refused, never
defaulted — and a saved train file round-trips through the assistant's
own loaders without choking the single-bunch paths.
"""
from __future__ import annotations

import json
import os

from linac_gen.assist.tools import TOOLS, WorkContext

F = 162.5


def _train_ctx(tmp_path):
    from linac_gen.core.config import BeamConfig
    from linac_gen.core.lattice import Lattice
    from linac_gen.elements.drift import Drift
    from linac_gen.elements.rf_gap import RFGap

    lat = Lattice()
    lat.add(Drift("D0", 50.0, aperture=30.0))
    lat.add(RFGap(name="CAV1", voltage=0.8, phase=0.0, frequency=F))
    lat.add(Drift("D1", 50.0, aperture=30.0))
    cfg = BeamConfig(species="proton", energy=3.0, frequency=F,
                     current=5.0, n_particles=300,
                     distribution="waterbag",
                     emit_nx=0.25, alpha_x=0.0, beta_x=0.4,
                     emit_ny=0.25, alpha_y=0.0, beta_y=0.4,
                     emit_z=0.15, alpha_z=0.0, beta_z=1.2)
    c = WorkContext(calc_dir=str(tmp_path))
    c.set_lattice(lat, "<train>")
    c.set_beam_config(cfg)
    return c


def _sidecar(tmp_path):
    p = tmp_path / "cav.json"
    p.write_text(json.dumps(
        {"CAV*": {"r_over_q": 200.0, "q_loaded": 5.0e6}}))
    return str(p)


def test_registered_long_running_and_visible_to_mcp():
    from linac_gen.assist.agent import LONG_RUNNING
    assert "run_train" in LONG_RUNNING
    assert TOOLS["run_train"].tier == "compute"
    from linac_gen.assist.mcp_server import tool_specs
    assert any(s["name"] == "run_train" for s in tool_specs())


def test_missing_cavity_params_refused_verbatim(tmp_path):
    ctx = _train_ctx(tmp_path)
    res = TOOLS["run_train"].fn(
        ctx, bunch_frequency_MHz=F, mode="fast", pattern="1*4",
        beam_loading=True)                       # no cavity_params
    assert res["status"] == "refused"
    msg = res["data"]["message"]
    # TrainConfig's own message, verbatim — it names what is missing.
    assert "cavity_params" in msg and "R/Q" in msg


def test_inert_inputs_refused(tmp_path):
    """Every accepted-but-unconsumed input is a refusal, not a no-op."""
    ctx = _train_ctx(tmp_path)
    side = _sidecar(tmp_path)
    # sidecar with no channel enabled would be silently unused
    r = TOOLS["run_train"].fn(ctx, bunch_frequency_MHz=F, mode="fast",
                              n_bunches=3, cavity_params=side)
    assert r["status"] == "refused"
    assert "silently unused" in r["data"]["message"]
    # hybrid-only knobs outside their modes
    r = TOOLS["run_train"].fn(ctx, bunch_frequency_MHz=F, mode="mp",
                              n_bunches=3, replay_parallel=True)
    assert r["status"] == "refused" and "hybrid" in r["data"]["message"]
    r = TOOLS["run_train"].fn(ctx, bunch_frequency_MHz=F, mode="mp",
                              n_bunches=3, history_stride=4)
    assert r["status"] == "refused"
    r = TOOLS["run_train"].fn(ctx, bunch_frequency_MHz=F, mode="mp",
                              n_bunches=3, select_bunches=[0])
    assert r["status"] == "refused" and "hybrid" in r["data"]["message"]


def test_pattern_inputs_validated(tmp_path):
    ctx = _train_ctx(tmp_path)
    r = TOOLS["run_train"].fn(ctx, bunch_frequency_MHz=F, mode="fast")
    assert r["status"] == "refused"
    assert "pattern" in r["data"]["message"]
    r = TOOLS["run_train"].fn(ctx, bunch_frequency_MHz=F, mode="fast",
                              pattern="1*4", n_bunches=4)
    assert r["status"] == "refused"
    r = TOOLS["run_train"].fn(ctx, bunch_frequency_MHz=F, mode="fast",
                              pattern="1*4", duty_keep=2)
    assert r["status"] == "refused"
    r = TOOLS["run_train"].fn(ctx, bunch_frequency_MHz=F, mode="fast",
                              pattern="2*4")
    assert r["status"] == "refused"
    assert "RLE" in r["data"]["message"]


def test_happy_path_fast_with_droop_and_output(tmp_path):
    ctx = _train_ctx(tmp_path)
    side = _sidecar(tmp_path)
    out = str(tmp_path / "train.h5")
    res = TOOLS["run_train"].fn(
        ctx, bunch_frequency_MHz=F, mode="fast",
        pattern="1*5 0*3 1*2", beam_loading=True, cavity_params=side,
        space_charge=False, out_path=out)
    assert res["status"] == "ok", res
    d = res["data"]
    assert d["run_type"] == "train"
    assert d["mode"] == "fast"
    assert d["n_slots"] == 10 and d["n_bunches_pattern"] == 7
    assert d["fast_n_bunches"] == 7
    assert not d["truncated"]
    w = d["fast_w_exit_MeV"]
    # Beam loading decelerates: droop is strictly negative and monotone
    # in aggregate (last bunch below the first for a loaded uniform head).
    assert w["dw_min_keV"] < 0.0
    assert w["w_design_MeV"] > 0.0
    assert d["output_path"] == out and os.path.isfile(out)
    assert res["provenance"]["results_path"] == out
    # The session results are the train container now.
    from linac_gen.train.results import TrainResults
    assert isinstance(ctx.results, TrainResults)


def test_default_out_path_lands_in_calc_dir(tmp_path):
    ctx = _train_ctx(tmp_path)
    res = TOOLS["run_train"].fn(
        ctx, bunch_frequency_MHz=F, mode="fast", n_bunches=3,
        space_charge=False)
    assert res["status"] == "ok", res
    out = res["data"]["output_path"]
    assert os.path.dirname(out) == str(tmp_path)
    assert os.path.basename(out).startswith("train_")


def test_load_results_routes_train_files(tmp_path):
    ctx = _train_ctx(tmp_path)
    side = _sidecar(tmp_path)
    out = str(tmp_path / "train_route.h5")
    res = TOOLS["run_train"].fn(
        ctx, bunch_frequency_MHz=F, mode="fast", pattern="1*3 0*2 1*1",
        beam_loading=True, cavity_params=side, space_charge=False,
        out_path=out)
    assert res["status"] == "ok", res

    # Fresh context — as if a new session loads the file.
    ctx2 = WorkContext(calc_dir=str(tmp_path))
    r = TOOLS["load_results"].fn(ctx2, path=out)
    assert r["status"] == "ok", r
    d = r["data"]
    assert d["run_type"] == "train"
    assert "load_train_results" in d["note"]
    assert d["n_bunches_pattern"] == 4
    from linac_gen.train.results import LoadedTrainResults
    assert isinstance(ctx2.results, LoadedTrainResults)

    # result_summary on the loaded train reports the train summary
    # instead of choking on a recorder-shaped reader.
    r2 = TOOLS["result_summary"].fn(ctx2)
    assert r2["status"] == "ok", r2
    assert r2["data"]["run_type"] == "train"
    assert r2["data"]["fast_w_exit_MeV"]["w_design_MeV"] > 0.0


def test_single_bunch_files_unaffected(tmp_path):
    """The routing must not touch normal results files."""
    ctx = _train_ctx(tmp_path)
    r = TOOLS["run_mp"].fn(ctx, space_charge=False, n_particles=200)
    assert r["status"] == "ok", r
    out = str(tmp_path / "single.h5")
    w = TOOLS["write_results"].fn(ctx, path=out)
    assert w["status"] == "ok", w
    ctx2 = WorkContext(calc_dir=str(tmp_path))
    r2 = TOOLS["load_results"].fn(ctx2, path=out)
    assert r2["status"] == "ok", r2
    assert "arrays" in r2["data"]            # the classic payload shape
    assert "run_type" not in r2["data"]
