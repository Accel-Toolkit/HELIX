"""run_reliability — the assist/MCP door into the Reliability Study mode."""
from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from linac_gen.assist.agent import LONG_RUNNING
from linac_gen.assist.tools import TOOLS, WorkContext

REPO = Path(__file__).resolve().parents[2]
DEMO = REPO / "examples" / "reliability_demo"


@pytest.fixture()
def ctx(tmp_path):
    for name in ("reliability_demo.dat", "reliability_demo.lgproj", "circuits.json"):
        shutil.copy2(DEMO / name, tmp_path / name)
    c = WorkContext()
    c.calc_dir = str(tmp_path)
    c.deck = str(tmp_path / "reliability_demo.lgproj")
    return c


def _fn():
    return TOOLS["run_reliability"].fn


def test_registered_compute_and_long_running():
    t = TOOLS["run_reliability"]
    assert t.tier == "compute" and "run_reliability" in LONG_RUNNING
    assert "RESUME" in t.description


def test_plan_previews_without_creating(ctx, tmp_path):
    r = _fn()(ctx, action="plan", lattice_path=ctx.deck, preset="quick")
    assert r["status"] == "ok"
    assert r["data"]["items_per_leg"]["faults"] == 16 and r["data"]["classes"] == ["S1", "S2", "S5", "S10"]
    assert not (tmp_path / "reliability").exists()


def test_run_resume_summarize_and_refusals(ctx, tmp_path):
    ticks = []
    cdir = tmp_path / "camp"
    r = _fn()(ctx, action="run", lattice_path=ctx.deck, campaign_dir=str(cdir), legs=["faults", "availability"],
              progress_callback=lambda f, d, t: ticks.append((d, t)))
    assert r["status"] == "ok", r
    assert r["data"]["faults"]["n_cases"] == 14 and (cdir / "report.html").exists() and ticks
    r2 = _fn()(ctx, action="run", campaign_dir=str(cdir), legs=["faults"])
    assert r2["status"] == "ok" and r2["data"]["status"]["faults"]["done"] == 16 + 5
    r3 = _fn()(ctx, action="summarize", campaign_dir=str(cdir))
    assert r3["status"] == "ok" and r3["data"]["report_html"].endswith("report.html")
    assert _fn()(ctx, action="summarize")["status"] == "refused"
    assert _fn()(ctx, action="run", lattice_path=ctx.deck, legs=["nope"])["status"] == "refused"
    assert _fn()(ctx, action="export", campaign_dir=str(cdir))["status"] == "refused"
    bad = _fn()(ctx, action="run", lattice_path=str(tmp_path / "missing.dat"))
    assert bad["status"] == "error"
    no_lat = _fn()(WorkContext(), action="run")
    assert no_lat["status"] in ("refused", "error")


def test_name_with_separators_is_refused(ctx, tmp_path):
    r = _fn()(ctx, action="plan", lattice_path=ctx.deck, name="../../escaped")
    assert r["status"] == "refused" and "plain folder name" in r["data"]["message"]
    assert not (tmp_path.parent / "escaped").exists()
