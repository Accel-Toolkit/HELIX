"""CLI: python -m linac_gen study plan|run|resume|summarize."""
import json

import pytest

from linac_gen.__main__ import main

FODO = """DRIFT 100 20 0 0 0
QUAD 80 8.0 20 0 0 0 0 0 0
DRIFT 200 20 0 0 0
QUAD 80 -8.0 20 0 0 0 0 0 0
DRIFT 100 20 0 0 0
END
"""


@pytest.fixture()
def study_json(tmp_path):
    (tmp_path / "fodo.dat").write_text(FODO)
    doc = {
        "__kind__": "linac_gen_study", "__version__": 1,
        "name": "cli_smoke", "input": "fodo.dat",
        "mode": "envelope", "strategy": "grid",
        "parameters": [{"selector": "@2.gradient",
                        "start": 7.0, "stop": 9.0, "n": 2}],
        "beam": {"energy": 2.1, "current": 0.0, "n_particles": 300},
    }
    p = tmp_path / "study.json"
    p.write_text(json.dumps(doc))
    return p


def test_plan_lists_runs(study_json, capsys):
    assert main(["study", "plan", str(study_json)]) == 0
    out = capsys.readouterr().out
    assert "2 run(s)" in out and "@2.gradient=7" in out


def test_run_then_resume_noop(study_json, capsys):
    assert main(["study", "run", str(study_json), "-q"]) == 0
    sd = study_json.parent / "cli_smoke"
    assert (sd / "summary" / "summary.csv").exists()
    assert len(list((sd / "runs").iterdir())) == 2
    # resume on the directory: nothing to execute
    assert main(["study", "run", str(sd)]) == 0
    out = capsys.readouterr().out
    assert "0 to execute" in out


def test_summarize_rebuilds(study_json):
    assert main(["study", "run", str(study_json), "-q"]) == 0
    sd = study_json.parent / "cli_smoke"
    (sd / "summary" / "summary.csv").unlink()
    assert main(["study", "summarize", str(sd)]) == 0
    assert (sd / "summary" / "summary.csv").exists()


def test_bad_spec_exit_2(tmp_path):
    p = tmp_path / "study.json"
    p.write_text(json.dumps({"__kind__": "linac_gen_study",
                             "name": "x", "input": "missing.dat",
                             "parameters": []}))
    assert main(["study", "plan", str(p)]) == 2


def test_missing_target_exit_2(tmp_path):
    assert main(["study", "run", str(tmp_path / "nope.json")]) == 2
