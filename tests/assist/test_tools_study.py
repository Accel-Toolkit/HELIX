"""run_study — the assist/MCP door into the Parameter Study Manager.

Covers registration + LONG_RUNNING membership, the plan/run/resume/
summarize actions against a real (tiny, envelope-mode) study, and the
refusal paths.  The runs execute the actual linac_gen.study engine —
seconds, not minutes, for a 3-point envelope study on a 3-element deck.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from linac_gen.assist.agent import LONG_RUNNING
from linac_gen.assist.tools import TOOLS, WorkContext

DECK = ("FREQ 162.5\n"
        "DRIFT 100 15 0\n"
        "QUAD 80 12 15 0\n"
        "DRIFT 100 15 0\n"
        "END\n")

PARAMS = [{"selector": "@3.gradient", "start": 8.0, "stop": 16.0,
           "n": 3}]


@pytest.fixture()
def ctx(tmp_path):
    deck = tmp_path / "cell.dat"
    deck.write_text(DECK, encoding="utf-8")
    c = WorkContext()
    c.calc_dir = str(tmp_path)
    c.deck = str(deck)          # convenience for tests below
    return c


def _fn():
    return TOOLS["run_study"].fn


def test_registered_compute_and_long_running():
    t = TOOLS["run_study"]
    assert t.tier == "compute"
    assert "run_study" in LONG_RUNNING
    assert "resume" in t.description.lower() or \
        "RESUME" in t.description


def test_plan_previews_without_executing(ctx, tmp_path):
    r = _fn()(ctx, action="plan", lattice_path=ctx.deck, name="p",
              parameters=PARAMS)
    assert r["status"] == "ok"
    assert r["data"]["n_runs"] == 3
    assert r["data"]["first_runs"][0]["params"] == {"@3.gradient": 8.0}
    # nothing was created
    assert not (tmp_path / "studies").exists()


def test_run_resume_and_summarize(ctx, tmp_path):
    ticks = []
    r = _fn()(ctx, action="run", lattice_path=ctx.deck, name="s1",
              parameters=PARAMS,
              observables=[{"quantity": "sigma_x"}],
              progress_callback=lambda v, d, t: ticks.append(d))
    assert r["status"] == "ok"
    assert (r["data"]["n_runs"], r["data"]["completed"],
            r["data"]["failed"]) == (3, 3, 0)
    assert ticks and ticks[-1] == 3
    sd = Path(r["data"]["study_dir"])
    assert sd == tmp_path / "studies" / "s1"
    csv = (sd / "summary" / "summary.csv").read_text(encoding="utf-8")
    header, *rows = csv.splitlines()
    assert "@3.gradient" in header and "sigma_x" in header
    assert len(rows) == 3

    # resume skips everything (completed runs are never re-executed)
    r2 = _fn()(ctx, action="run", study_dir=str(sd))
    assert r2["status"] == "ok"
    assert r2["data"]["completed"] == 3

    r3 = _fn()(ctx, action="summarize", study_dir=str(sd))
    assert r3["status"] == "ok"
    assert Path(r3["data"]["summary_csv"]).is_file()


def test_refusals(ctx):
    # no parameters and no session lattice
    assert _fn()(ctx, action="run")["status"] == "refused"
    # URL is not a local path (SSRF guard)
    r = _fn()(ctx, action="run", lattice_path="https://evil/x.dat",
              parameters=PARAMS)
    assert r["status"] == "refused"
    # rangeless parameter
    r = _fn()(ctx, action="run", lattice_path=ctx.deck,
              parameters=[{"selector": "@3.gradient"}])
    assert r["status"] == "refused"
    assert "values" in r["data"]["message"]
    # summarize without a directory
    assert _fn()(ctx, action="summarize")["status"] == "refused"
    # missing deck
    r = _fn()(ctx, action="run", lattice_path="/nope/missing.dat",
              parameters=PARAMS)
    assert r["status"] == "error"


def test_session_lattice_requires_saved_file(ctx):
    """With no explicit lattice_path the tool gates on a SAVED session
    lattice — in-memory-only edits must refuse, not silently study
    different physics."""
    from linac_gen.core.config import BeamConfig
    from linac_gen.cli.common import load_lattice
    lat = load_lattice(ctx.deck)
    ctx.set_lattice(lat, "")            # loaded but no on-disk path
    ctx.set_beam_config(BeamConfig())
    r = _fn()(ctx, action="run", parameters=PARAMS)
    assert r["status"] == "refused"
    assert "saved" in r["data"]["message"].lower() or \
        "disk" in r["data"]["message"].lower()

    ctx.set_lattice(lat, ctx.deck)      # now with the real file
    r = _fn()(ctx, action="run", parameters=PARAMS, name="sess")
    assert r["status"] == "ok"
    assert r["data"]["completed"] == 3
