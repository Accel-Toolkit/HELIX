"""The RFQ solver choice (StepConfig.rfq_solver) through every run path:
CLI precedence and flags, scan points and their worker, studies, the
provenance record and the assistant's run tool."""
from __future__ import annotations

import argparse
import dataclasses
from pathlib import Path

import h5py
import pytest

from linac_gen.__main__ import main
from linac_gen.cli import common

_REPO = Path(__file__).resolve().parents[2]
_DAT = _REPO / "examples" / "fodo_cell.dat"


@pytest.mark.parametrize("conv, cli, expected", [
    ({}, {}, ("auto", 48, 4)),
    ({"rfq_solver": "cells"}, {}, ("cells", 48, 4)),
    ({"rfq_solver": "cells"}, {"rfq_solver": "auto"}, ("auto", 48, 4)),
    ({"rfq_solver": "time_domain"}, {}, ("time_domain", 48, 4)),
    ({"rfq_solver": "time_domain", "rfq_td_steps_per_period": 96,
      "rfq_td_sc_every": 2}, {}, ("time_domain", 96, 2)),
    ({"rfq_solver": "time_domain"}, {"rfq_solver": "cells"}, ("cells", 48, 4)),
    ({}, {"rfq_solver": "time_domain", "rfq_td_steps": 64},
     ("time_domain", 64, 4)),
    ({"rfq_solver": "time_domain"}, {"rfq_solver": None},
     ("time_domain", 48, 4)),
    ({"rfq_solver": "Time-Domain"}, {}, ("time_domain", 48, 4)),
])
def test_make_step_config_precedence(conv, cli, expected):
    cfg = common.make_step_config(conv, cli)
    assert (cfg.rfq_solver, cfg.rfq_td_steps_per_period,
            cfg.rfq_td_sc_every) == expected
    assert cfg.drift_single_push is True


@pytest.mark.parametrize("conv, cli", [
    ({"rfq_solver": "z"}, {}),
    ({}, {"rfq_td_steps": 4}),
    ({"rfq_td_sc_every": 0}, {}),
])
def test_bad_values_fail_before_any_run(conv, cli):
    with pytest.raises(ValueError):
        common.rfq_settings(conv, cli)


def test_build_scan_point_carries_the_solver():
    p = common.build_scan_point(str(_DAT))
    assert (p.rfq_solver, p.rfq_td_steps_per_period, p.rfq_td_sc_every) \
        == ("auto", 48, 4)
    p = common.build_scan_point(str(_DAT), cli={"rfq_solver": "cells"})
    assert p.rfq_solver == "cells"
    p = common.build_scan_point(str(_DAT), cli={"rfq_solver": "time_domain",
                                                "rfq_td_steps": 72})
    assert (p.rfq_solver, p.rfq_td_steps_per_period) == ("time_domain", 72)


def test_scan_worker_builds_the_step_config(monkeypatch):
    """The worker's StepConfig carries the point's RFQ fields."""
    from linac_gen.core import simulation
    from linac_gen.parallel import scan_pool
    seen = {}

    class _Stub:
        def __init__(self, lattice, beam, **kw):
            seen["cfg"] = lattice.step_config

        def run(self):
            raise RuntimeError("stub stops here")

    monkeypatch.setattr(simulation, "Simulation", _Stub)
    p = common.build_scan_point(str(_DAT), cli={"rfq_solver": "time_domain",
                                                "rfq_td_steps": 60})
    out = scan_pool._run_one_point_worker(
        dataclasses.replace(p, capture_errors=True))
    assert "stub stops here" in out["error"]
    assert seen["cfg"].rfq_solver == "time_domain"
    assert seen["cfg"].rfq_td_steps_per_period == 60
    assert seen["cfg"].rfq_td_sc_every == 4


@pytest.mark.parametrize("module", ["run", "scan"])
def test_parsers_accept_the_flags(module):
    import importlib
    mod = importlib.import_module(f"linac_gen.cli.{module}")
    p = argparse.ArgumentParser()
    mod.add_arguments(p)
    base = ["deck.dat"] + (["--vary", "q.gradient=1:2:1"]
                           if module == "scan" else [])
    a = p.parse_args(base)
    assert a.rfq_solver is None and a.rfq_td_steps is None
    a = p.parse_args(base + ["--rfq-solver", "time_domain", "--rfq-td-steps",
                             "96"])
    assert a.rfq_solver == "time_domain" and a.rfq_td_steps == 96
    for value in ("auto", "cells"):
        assert p.parse_args(base + ["--rfq-solver", value]).rfq_solver \
            == value
    with pytest.raises(SystemExit):
        p.parse_args(base + ["--rfq-solver", "z"])
    if module == "run":
        ov = mod._cli_overrides(a)
        assert ov["rfq_solver"] == "time_domain" and ov["rfq_td_steps"] == 96


def test_scan_and_study_can_sweep_the_time_steps():
    """rfq_td_steps is a structural sweep variable (a step-convergence
    scan); the float a range produces reaches the point as an int."""
    from linac_gen.cli import scan as scan_cli
    from linac_gen.study.spec import ParamSpec
    var, vals = scan_cli._parse_vary("rfq_td_steps=24:96:24")
    assert var == "rfq_td_steps" and vals == [24.0, 48.0, 72.0, 96.0]
    assert scan_cli._kind("rfq_td_steps") == "structural"
    assert ParamSpec("rfq_td_steps").resolved_kind() == "structural"
    p = common.build_scan_point(str(_DAT), cli={"rfq_solver": "time_domain",
                                                "rfq_td_steps": 72.0})
    assert p.rfq_td_steps_per_period == 72
    assert isinstance(p.rfq_td_steps_per_period, int)


def test_run_records_the_solver_in_provenance(tmp_path):
    """A lattice without an RFQ runs the same in either mode; the choice is
    recorded."""
    rc = main(["run", str(_DAT), "--mode", "mp", "--energy", "2.1226695",
               "--freq", "162.5", "--species", "H-", "--n-particles", "300",
               "--rfq-solver", "time_domain", "--rfq-td-steps", "64",
               "--out", str(tmp_path), "-q"])
    assert rc == 0
    h5 = list(tmp_path.glob("*_results.h5"))
    assert len(h5) == 1
    with h5py.File(h5[0], "r") as f:
        prov = f["provenance"].attrs
        assert prov["rfq_solver"] == "time_domain"
        assert int(prov["rfq_td_steps_per_period"]) == 64
        assert int(prov["rfq_td_sc_every"]) == 4
        assert "rfq_td" not in prov            # no RFQ in this lattice
        assert "rfq_solver_used" not in prov


def test_run_rejects_a_bad_step_count(tmp_path, capsys):
    rc = main(["run", str(_DAT), "--mode", "mp", "--energy", "2.1226695",
               "--freq", "162.5", "--species", "H-", "--n-particles", "50",
               "--rfq-td-steps", "4", "--out", str(tmp_path), "-q"])
    assert rc == 2
    assert "rfq_td_steps_per_period must be >= 8" in capsys.readouterr().err
    assert not list(tmp_path.glob("*_results.h5"))


def test_assistant_run_tool_offers_the_choice_for_one_run(monkeypatch):
    from linac_gen.assist.tools import TOOLS, WorkContext
    from linac_gen.core import simulation
    from linac_gen.core.config import BeamConfig
    from linac_gen.core.lattice import Lattice
    from linac_gen.elements.drift import Drift

    prop = TOOLS["run_mp"].schema["properties"]["rfq_solver"]
    assert prop["enum"] == ["auto", "cells", "time_domain"]

    seen = []
    real = simulation.Simulation

    class _Spy(real):
        def __init__(self, lattice, beam, **kw):
            seen.append(lattice.step_config.rfq_solver)
            super().__init__(lattice, beam, **kw)

    monkeypatch.setattr(simulation, "Simulation", _Spy)
    lat = Lattice()
    lat.add(Drift("D1", length=100.0))
    ctx = WorkContext()
    ctx.lattice = lat
    ctx.beam_config = BeamConfig(species="proton", energy=3.0,
                                 frequency=352.21, current=0.0,
                                 n_particles=50)
    before = lat.step_config
    res = TOOLS["run_mp"].fn(ctx, space_charge=False,
                             rfq_solver="time_domain")
    assert res["status"] == "ok", res
    assert seen == ["time_domain"]
    assert lat.step_config is before           # the session's setting back
    assert TOOLS["run_mp"].fn(ctx, space_charge=False)["status"] == "ok"
    assert seen == ["time_domain", "auto"]


def test_study_numerics_reach_the_point(tmp_path):
    from linac_gen.study.engine import StudyManager
    from linac_gen.study.spec import ParamSpec, StudySpec
    from linac_gen.study.strategies import expand_runs
    spec = StudySpec(name="rfq", input=str(_DAT), mode="mp",
                     strategy="grid",
                     parameters=[ParamSpec("rfq_td_steps",
                                           values=[24.0, 96.0])],
                     numerics={"rfq_solver": "time_domain"})
    mgr = StudyManager.create(tmp_path / "study", spec)
    points = [mgr._point_for(r) for r in expand_runs(spec)]
    assert [(p.rfq_solver, p.rfq_td_steps_per_period) for p in points] \
        == [("time_domain", 24), ("time_domain", 96)]
