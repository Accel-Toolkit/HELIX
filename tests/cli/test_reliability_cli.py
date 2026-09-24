"""CLI: python -m linac_gen reliability plan|run|resume|summarize|export|import."""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from linac_gen.__main__ import main

REPO = Path(__file__).resolve().parents[2]
DEMO = REPO / "examples" / "reliability_demo"


@pytest.fixture()
def deck(tmp_path):
    for name in ("reliability_demo.dat", "reliability_demo.lgproj", "circuits.json"):
        shutil.copy2(DEMO / name, tmp_path / name)
    return tmp_path / "reliability_demo.lgproj"


def test_plan_prints_items_and_writes_spec(deck, tmp_path, capsys):
    spec_path = tmp_path / "spec.json"
    assert main(["reliability", "plan", str(deck), "--write-spec", str(spec_path)]) == 0
    out = capsys.readouterr().out
    assert "preset=quick" in out and "faults" in out and "availability" in out
    doc = json.loads(spec_path.read_text())
    assert doc["__kind__"] == "linac_gen_reliability" and doc["classes"] == ["S1", "S2", "S5", "S10"]
    assert doc["circuits"].endswith("circuits.json")
    assert main(["reliability", "plan", str(deck), "--preset", "full", "--legs", "faults"]) == 0
    out = capsys.readouterr().out
    assert "preset=full" in out and "imperfections" not in out


def test_run_resume_summarize_export_import(deck, tmp_path, capsys):
    spec = {"__kind__": "linac_gen_reliability", "__version__": 1, "name": "cli", "input": "x",
            "preset": "quick", "legs": {"faults": True, "availability": True, "imperfections": False, "foil": True},
            "classes": ["S1", "S5"], "compensation": {"top_n": 1}, "availability": {"n_trials": 10}}
    sp = tmp_path / "start.json"
    sp.write_text(json.dumps(spec))
    cdir = tmp_path / "camp"
    assert main(["reliability", "run", str(deck), "--spec", str(sp), "--dir", str(cdir), "--serial"]) == 0
    out = capsys.readouterr().out
    assert "complete" in out and (cdir / "report.html").exists() and (cdir / "summary.json").exists()
    assert main(["reliability", "resume", str(cdir), "--serial"]) == 0
    out = capsys.readouterr().out
    assert "0 to execute" in out
    assert main(["reliability", "summarize", str(cdir)]) == 0
    assert "summary written" in capsys.readouterr().out
    # export the foil leg of a second campaign that has not run it, run the job, import
    c2 = tmp_path / "camp2"
    assert main(["reliability", "run", str(deck), "--spec", str(sp), "--dir", str(c2), "--serial",
                 "--legs", "faults"]) == 0
    job = tmp_path / "job"
    assert main(["reliability", "export", str(c2), "--to", str(job), "--legs", "foil"]) == 0
    assert "job written" in capsys.readouterr().out and (job / "README.txt").exists()
    assert main(["reliability", "run", str(job), "--serial"]) == 0
    assert main(["reliability", "import", str(job), "--into", str(c2)]) == 0
    assert "imported 7 item(s)" in capsys.readouterr().out
    summ = json.loads((c2 / "summary.json").read_text())
    assert len(summ["foil"]) == 7 and summ["status"]["foil"]["done"] == 7


def test_bad_inputs_exit_2(tmp_path, capsys):
    assert main(["reliability", "run"]) == 2
    assert main(["reliability", "run", str(tmp_path / "nope.dat")]) == 2
    assert main(["reliability", "resume", str(tmp_path)]) == 2
    assert main(["reliability", "summarize", str(tmp_path)]) == 2
    assert main(["reliability", "export", str(tmp_path)]) == 2
    assert main(["reliability", "import", str(tmp_path)]) == 2
    (tmp_path / "d.dat").write_text("FREQ 162.5\nDRIFT 100 20\nEND\n")
    assert main(["reliability", "plan", str(tmp_path / "d.dat"), "--legs", "nope"]) == 2
    err = capsys.readouterr().err
    assert "unknown leg" in err
