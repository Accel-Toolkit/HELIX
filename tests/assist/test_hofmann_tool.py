"""hofmann_stability assistant tool (tools_analysis.py).

Runs the corrected anisotropic Hofmann screen off the session context:
its own envelope phase probe when the session has no probe-bearing
results, per-cell JSON rows, optional margin + probability layers, and
the in-band refusal reasons (no z tune) vs envelope-level refusals
(missing session state, bad period index)."""
from __future__ import annotations

import json

import pytest

from linac_gen.assist.tools import TOOLS, WorkContext


def _bunched_ctx(tmp_path):
    """WorkContext with the stable FODO+buncher cell + matched beam."""
    from linac_gen.core.config import BeamConfig
    from linac_gen.core.lattice import Lattice
    from linac_gen.core.particle import PROTON
    from linac_gen.core.reference import ReferenceParticle
    from linac_gen.elements.drift import Drift
    from linac_gen.elements.quadrupole import Quadrupole
    from linac_gen.elements.rf_gap import RFGap
    from linac_gen.matching.periodic import find_periodic_twiss
    from linac_gen.tracking.matrix_tracking import (
        compute_transfer_matrix, compute_twiss)

    lat = Lattice()
    for _ in range(3):
        lat.add(Drift(name="D", length=30.0, aperture=20.0))
        lat.add(Quadrupole(name="QF", length=40.0, gradient=+50.0,
                           aperture=20.0))
        lat.add(Drift(name="D", length=30.0, aperture=20.0))
        lat.add(RFGap(name="G", voltage=0.10, phase=-90.0, frequency=162.5))
        lat.add(Drift(name="D", length=30.0, aperture=20.0))
        lat.add(Quadrupole(name="QD", length=40.0, gradient=-50.0,
                           aperture=20.0))
        lat.add(Drift(name="D", length=30.0, aperture=20.0))
    ref = ReferenceParticle(species=PROTON, w_kin=2.5, frequency=162.5)
    tw = find_periodic_twiss(lat, ref)
    M = compute_transfer_matrix(lat, ref, start=0, end=6)
    twz = compute_twiss(M, "z")
    bg = ref.beta * ref.gamma
    cfg = BeamConfig(species="proton", energy=2.5, frequency=162.5,
                     current=5.0, duty_cycle=100.0, n_particles=200,
                     distribution="gaussian", cutoff=3.0,
                     emit_nx=2.0 * bg, alpha_x=tw["alpha_x"],
                     beta_x=tw["beta_x"],
                     emit_ny=2.0 * bg, alpha_y=tw["alpha_y"],
                     beta_y=tw["beta_y"],
                     emit_z=0.1, alpha_z=twz["alpha"], beta_z=twz["beta"])
    c = WorkContext(calc_dir=str(tmp_path))
    c.set_lattice(lat, "<hofmann-test-lattice>")
    c.set_beam_config(cfg)
    return c


def test_tool_runs_probe_and_reports_cells(tmp_path):
    ctx = _bunched_ctx(tmp_path)
    res = TOOLS["hofmann_stability"].fn(
        ctx, margin=True, probability=True, n_mc=50)
    assert res["status"] == "ok", res
    data = res["data"]
    assert data["reason"] is None
    assert data["probe"].startswith("fresh envelope")
    assert data["n_cells"] >= 1
    assert len(data["cells"]) == data["n_cells"]
    row = data["cells"][0]
    for key in ("cell", "R", "Y", "eps_ratio", "S2", "g_l2", "g_l3_even",
                "g_l3_odd", "g_l4_even", "g_combined", "valid", "flagged",
                "flagged_extrap", "fold_risk", "onset_eps", "margin",
                "is_seam", "p_unstable"):
        assert key in row, key
    assert row["R"] is not None and row["R"] > 0
    assert row["Y"] is not None and 0 < row["Y"] < 1     # depressed at 5 mA
    assert "margin_summary" in data
    assert len(data["solver_fingerprint"]) == 10
    json.dumps(res, allow_nan=False)      # strictly JSON-serializable


def test_tool_reuses_probe_bearing_session_results(tmp_path):
    from linac_gen.cli.common import _envelope_initial, build_ref
    from linac_gen.tracking.envelope import EnvelopeSolver

    ctx = _bunched_ctx(tmp_path)
    ref = build_ref(ctx.beam_config)
    res_env = EnvelopeSolver(
        ctx.lattice, ref, _envelope_initial(ctx.beam_config, ref),
        current=ctx.beam_config.current, phase_probe=True).run()
    ctx.set_results(res_env, "")
    res = TOOLS["hofmann_stability"].fn(ctx)
    assert res["status"] == "ok", res
    assert res["data"]["probe"].startswith("session results")


def test_tool_reports_no_z_tune_reason(ctx):
    """The conftest quad-only lattice has no longitudinal focusing: the
    analysis must return an in-band reason, not crash or refuse."""
    res = TOOLS["hofmann_stability"].fn(ctx)
    assert res["status"] == "ok", res
    assert res["data"]["reason"] is not None
    assert "longitudinal" in res["data"]["reason"]
    assert res["data"]["n_valid"] == 0
    json.dumps(res, allow_nan=False)


def test_tool_needs_session_state(tmp_path):
    res = TOOLS["hofmann_stability"].fn(WorkContext(calc_dir=str(tmp_path)))
    assert res["status"] == "error"
    assert "lattice" in res["data"]["message"]


def test_tool_period_index_out_of_range(tmp_path):
    ctx = _bunched_ctx(tmp_path)
    res = TOOLS["hofmann_stability"].fn(ctx, period_index=99)
    assert res["status"] == "refused"
    assert "period_index" in res["data"]["message"]


def test_tool_visible_to_mcp_server():
    from linac_gen.assist.mcp_server import tool_specs
    names = [t["name"] for t in tool_specs()]
    assert "hofmann_stability" in names
