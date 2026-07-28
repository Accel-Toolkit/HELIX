"""The `python -m linac_gen scan` subcommand."""
from pathlib import Path

from linac_gen.__main__ import main

_REPO = Path(__file__).resolve().parents[2]
_DAT = _REPO / "examples" / "csr_chicane.dat"


def _rows(csv_path):
    return csv_path.read_text().strip().splitlines()


def test_scan_beam_param(tmp_path):
    out = tmp_path / "s.csv"
    rc = main(["scan", str(_DAT), "--vary", "current=0:4:2",
               "--mode", "envelope", "--energy", "800", "--species", "H-",
               "--out", str(out), "-q"])
    assert rc == 0
    rows = _rows(out)
    assert len(rows) == 4                 # header + 3 points
    assert rows[0].startswith("current,")


def test_scan_explicit_value_list(tmp_path):
    out = tmp_path / "s.csv"
    rc = main(["scan", str(_DAT), "--vary", "current=0,1,5,9",
               "--mode", "envelope", "--energy", "800", "--species", "H-",
               "--out", str(out), "-q"])
    assert rc == 0
    assert len(_rows(out)) == 5           # header + 4


def test_scan_element_param(tmp_path):
    out = tmp_path / "e.csv"
    rc = main(["scan", str(_DAT), "--vary", "@2.gradient=4:8:2",
               "--mode", "envelope", "--energy", "800", "--species", "H-",
               "--out", str(out), "-q"])
    assert rc == 0
    rows = _rows(out)
    assert len(rows) == 4
    assert rows[0].startswith("@2.gradient,")


def test_scan_two_vary_cartesian(tmp_path):
    out = tmp_path / "c.csv"
    rc = main(["scan", str(_DAT), "--vary", "current=0:2:2",
               "--vary", "@2.gradient=4:6:2",
               "--mode", "envelope", "--energy", "800", "--species", "H-",
               "--out", str(out), "-q"])
    assert rc == 0
    assert len(_rows(out)) == 5           # header + 2x2 grid


def test_scan_parallel(tmp_path):
    """The process-pool path produces the same row count as serial."""
    out = tmp_path / "p.csv"
    rc = main(["scan", str(_DAT), "--vary", "current=0:2:2",
               "--mode", "envelope", "--energy", "800", "--species", "H-",
               "--parallel", "2", "--out", str(out), "-q"])
    assert rc == 0
    assert len(_rows(out)) == 3


def test_scan_missing_input(tmp_path):
    assert main(["scan", str(tmp_path / "nope.dat"),
                 "--vary", "current=0:2:1"]) == 2
