"""The `python -m linac_gen batch` subcommand."""
import json
from pathlib import Path

from linac_gen.__main__ import main

_REPO = Path(__file__).resolve().parents[2]
_DAT = _REPO / "examples" / "csr_chicane.dat"


def test_batch_two_jobs(tmp_path):
    jobs = {"jobs": [
        {"name": "a", "input": str(_DAT), "mode": "envelope",
         "beam": {"energy": 800, "species": "H-"}},
        {"name": "b", "input": str(_DAT), "mode": "envelope",
         "beam": {"energy": 800, "species": "H-", "current": 2.0}},
    ]}
    jf = tmp_path / "jobs.json"
    jf.write_text(json.dumps(jobs))
    rc = main(["batch", str(jf), "--out", str(tmp_path), "-q"])
    assert rc == 0
    summary = tmp_path / "batch_summary.csv"
    assert summary.is_file()
    rows = summary.read_text().strip().splitlines()
    assert len(rows) == 3                       # header + 2 jobs
    assert rows[0].startswith("name,input,mode,")
    assert rows[1].startswith("a,")
    assert rows[2].startswith("b,")


def test_batch_bare_list(tmp_path):
    """A job file may be a bare list (no 'jobs' wrapper)."""
    jobs = [{"name": "solo", "input": str(_DAT), "mode": "envelope",
             "beam": {"energy": 800, "species": "H-"}}]
    jf = tmp_path / "jobs.json"
    jf.write_text(json.dumps(jobs))
    rc = main(["batch", str(jf), "--out", str(tmp_path), "-q"])
    assert rc == 0
    assert len((tmp_path / "batch_summary.csv").read_text()
               .strip().splitlines()) == 2


def test_batch_element_override(tmp_path):
    jobs = {"jobs": [
        {"name": "g4", "input": str(_DAT), "mode": "envelope",
         "beam": {"energy": 800, "species": "H-"},
         "set": {"@2.gradient": 4.0}},
    ]}
    jf = tmp_path / "jobs.json"
    jf.write_text(json.dumps(jobs))
    assert main(["batch", str(jf), "--out", str(tmp_path), "-q"]) == 0


def test_batch_missing_file(tmp_path):
    assert main(["batch", str(tmp_path / "nope.json")]) == 2


def test_batch_empty_jobs(tmp_path):
    jf = tmp_path / "empty.json"
    jf.write_text('{"jobs": []}')
    assert main(["batch", str(jf), "--out", str(tmp_path)]) == 2
