"""lebt_scc assistant tool (tools_analysis.py)."""
from __future__ import annotations

import json

import pytest

from linac_gen.assist.tools import TOOLS, WorkContext


def _dc_ctx(tmp_path):
    from linac_gen.core.config import BeamConfig
    from linac_gen.core.lattice import Lattice
    from linac_gen.elements.drift import Drift
    from linac_gen.elements.solenoid import Solenoid

    lat = Lattice()
    lat.add(Drift(name="D1", length=300.0, aperture=60.0))
    lat.add(Solenoid(name="S1", length=250.0, field=0.16, aperture=60.0))
    lat.add(Drift(name="D2", length=900.0, aperture=60.0))
    lat.add(Solenoid(name="S2", length=250.0, field=0.16, aperture=60.0))
    lat.add(Drift(name="D3", length=300.0, aperture=60.0))
    cfg = BeamConfig(species="H-", energy=0.030, frequency=162.5,
                     current=15.0, duty_cycle=100.0, n_particles=500,
                     distribution="gaussian", cutoff=3.0,
                     emit_nx=0.2, alpha_x=0.0, beta_x=0.6,
                     emit_ny=0.2, alpha_y=0.0, beta_y=0.6,
                     emit_z=0.0, alpha_z=0.0, beta_z=1.0,
                     continuous=True)
    c = WorkContext(calc_dir=str(tmp_path))
    c.set_lattice(lat, "<lebt>")
    c.set_beam_config(cfg)
    return c


def test_tool_runs_dc_envelope_and_reports(tmp_path):
    ctx = _dc_ctx(tmp_path)
    res = TOOLS["lebt_scc"].fn(ctx, gas="N2", pressure_mbar=2e-5)
    assert res["status"] == "ok", res
    d = res["data"]
    assert d["reason"] is None
    assert d["probe"] == "fresh DC envelope run"
    assert d["fc_source"].startswith("computed balance")
    assert d["tau_scc_global_us"] > 0
    assert 0.0 <= d["mean_fc"] <= 1.0
    assert d["phi_min_V"] < 0.0
    assert d["scc_cards"] and all(0.0 <= c["factor"] <= 1.0
                                  for c in d["scc_cards"])
    assert any(ln.startswith("SPACE_CHARGE_COMP") for ln in d["deck_lines"])
    assert len(d["profiles"]["fc"]) == len(d["profiles"]["z_m"])
    json.dumps(res, allow_nan=False)
    assert ctx.results is None                     # no session mutation


def test_tool_param_validation_and_gates(tmp_path):
    ctx = _dc_ctx(tmp_path)
    assert TOOLS["lebt_scc"].fn(
        ctx, pressure_mbar="lots")["status"] == "refused"
    assert TOOLS["lebt_scc"].fn(
        ctx, pressure_mbar=5.0)["status"] == "refused"
    assert TOOLS["lebt_scc"].fn(
        ctx, eta_assumed=1.5)["status"] == "refused"
    assert TOOLS["lebt_scc"].fn(
        WorkContext(calc_dir=str(tmp_path)))["status"] == "error"


def test_tool_bunched_session_in_band_reason(ctx):
    """conftest ctx: bunched quad lattice + bunched config."""
    res = TOOLS["lebt_scc"].fn(ctx)
    assert res["status"] == "ok", res
    assert res["data"]["reason"] is not None
    assert "bunched" in res["data"]["reason"]
    json.dumps(res, allow_nan=False)


def test_tool_visible_to_mcp_and_background():
    from linac_gen.assist.agent import LONG_RUNNING
    from linac_gen.assist.mcp_server import tool_specs
    assert "lebt_scc" in [t["name"] for t in tool_specs()]
    assert "lebt_scc" in LONG_RUNNING


def test_tool_self_consistent_iterates(tmp_path):
    ctx = _dc_ctx(tmp_path)
    res = TOOLS["lebt_scc"].fn(ctx, gas="N2", pressure_mbar=2e-5,
                               self_consistent=True)
    assert res["status"] == "ok", res
    d = res["data"]
    assert d["reason"] is None
    assert d["probe"] == "self-consistent iterate"
    assert d["fc_source"].startswith("self-consistent")
    it = d["iterate"]
    assert it["converged"] and 1 <= it["n_iter"] <= 8
    assert len(it["history"]) == it["n_iter"]
    assert all(0.0 <= c["factor"] <= 1.0 for c in d["scc_cards"])
    json.dumps(res, allow_nan=False)
    assert ctx.results is None                     # no session mutation


def test_tool_cleared_region_validation_and_pass_through(tmp_path):
    ctx = _dc_ctx(tmp_path)
    assert TOOLS["lebt_scc"].fn(
        ctx, cleared_regions=[[2, 99]])["status"] == "refused"
    assert TOOLS["lebt_scc"].fn(
        ctx, cleared_regions=[[3, 2]])["status"] == "refused"
    assert TOOLS["lebt_scc"].fn(
        ctx, cleared_regions=[[0, 1]],
        cleared_residual=7.0)["status"] == "refused"
    res = TOOLS["lebt_scc"].fn(ctx, gas="N2", pressure_mbar=2e-5,
                               cleared_regions=[[2, 2]])
    assert res["status"] == "ok", res
    d = res["data"]
    assert d["reason"] is None
    assert d["cleared_regions"] == [[2, 2]]
    assert d["cleared_residual"] == 0.0
    json.dumps(res, allow_nan=False)


def test_tool_rejects_fractional_cleared_indices(tmp_path):
    ctx = _dc_ctx(tmp_path)
    assert TOOLS["lebt_scc"].fn(
        ctx, cleared_regions=[[1.9, 2]])["status"] == "refused"
