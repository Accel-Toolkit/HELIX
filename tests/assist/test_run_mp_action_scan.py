"""Assistant ``run_mp`` tool: the optional ``action_scan`` switch records
the halo action scan (off by default, like the GUI checkbox)."""
from __future__ import annotations

from linac_gen.assist.tools import TOOLS, WorkContext
from linac_gen.core.config import BeamConfig
from linac_gen.core.lattice import Lattice
from linac_gen.elements.drift import Drift
from linac_gen.elements.quadrupole import Quadrupole


def _ctx():
    lat = Lattice()
    lat.add(Drift("D1", length=200.0))
    lat.add(Quadrupole("QF", length=100.0, gradient=8.0))
    lat.add(Drift("D2", length=200.0))
    ctx = WorkContext()
    ctx.lattice = lat
    ctx.beam_config = BeamConfig(species="proton", energy=3.0,
                                 frequency=352.21, current=0.0,
                                 n_particles=400)
    return ctx


def test_schema_offers_the_switch():
    props = TOOLS["run_mp"].schema["properties"]
    assert props["action_scan"]["type"] == "boolean"
    assert props["action_scan"]["default"] is False


def test_run_mp_records_the_scan_only_when_asked():
    ctx = _ctx()
    res = TOOLS["run_mp"].fn(ctx, space_charge=False, action_scan=True)
    assert res["status"] == "ok", res
    scan = ctx.results.action_scan
    assert len(scan["count_x"]) == len(ctx.results.s)
    assert scan["n_alive"][0] == 400
    ctx2 = _ctx()
    assert TOOLS["run_mp"].fn(ctx2, space_charge=False)["status"] == "ok"
    assert not hasattr(ctx2.results, "action_scan")
