"""``--tracewin-ini`` on the input-taking subcommands: a bare deck takes
its beam from the sibling (or named) TraceWin ``.ini``; a ``.lgproj``
refuses it; scalar overrides still win; scan/batch/failures share the
same resolution as ``run``."""
from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path

import h5py
import pytest

from linac_gen.__main__ import main
from linac_gen.cli import common
from linac_gen.core.config import BeamConfig
from linac_gen.io.project import write_project

REPO = Path(__file__).resolve().parents[2]
ADS = REPO / "tests" / "io" / "fixtures" / "tracewin_ini" / "ads.ini"
FODO = REPO / "tests" / "io" / "fixtures" / "simple_fodo.dat"


@pytest.fixture
def deck(tmp_path):
    shutil.copy(FODO, tmp_path / "deck.dat")
    shutil.copy(ADS, tmp_path / "deck.ini")
    return tmp_path / "deck.dat"


def _h5_beam(out_dir: Path) -> dict:
    files = list(out_dir.glob("*_results.h5"))
    assert len(files) == 1, files
    with h5py.File(files[0], "r") as f:
        return dict(f["beam_config"].attrs)


def test_run_takes_the_beam_from_the_sibling_ini(deck, tmp_path, capsys):
    out = tmp_path / "out"
    rc = main(["run", str(deck), "--tracewin-ini", "--mode", "envelope",
               "--out", str(out), "-q"])
    err = capsys.readouterr().err
    assert rc == 0
    assert "beam from deck.ini: proton, 20 MeV, 100 MHz, 5 mA, 50000 particles" in err
    assert "352.21" in err and "FREQ" in err          # deck FREQ ≠ beam frequency, warned
    b = _h5_beam(out)
    assert b["energy"] == 20.0 and b["species"] == "proton"
    assert abs(b["emit_nx"] - 0.2) < 1e-9 and abs(b["alpha_z"] + 0.17661) < 1e-9


def test_without_the_flag_nothing_reads_the_ini(deck, tmp_path, capsys):
    out = tmp_path / "out"
    assert main(["run", str(deck), "--mode", "envelope", "--out", str(out), "-q"]) == 0
    assert "deck.ini" not in capsys.readouterr().err
    assert _h5_beam(out)["energy"] == BeamConfig().energy


def test_command_line_overrides_beat_the_ini(deck, tmp_path):
    out = tmp_path / "out"
    rc = main(["run", str(deck), "--tracewin-ini", "--energy", "25",
               "--species", "H-", "--mode", "envelope", "--out", str(out), "-q"])
    assert rc == 0
    b = _h5_beam(out)
    assert b["energy"] == 25.0 and b["species"] == "H-"
    assert abs(b["emit_nx"] - 0.2) < 1e-9                 # the rest still from the .ini


def test_explicit_ini_path(deck, tmp_path):
    other = tmp_path / "elsewhere" / "beam.ini"
    other.parent.mkdir()
    shutil.move(str(tmp_path / "deck.ini"), other)
    out = tmp_path / "out"
    assert main(["run", str(deck), "--tracewin-ini", str(other), "--mode",
                 "envelope", "--out", str(out), "-q"]) == 0
    assert _h5_beam(out)["energy"] == 20.0


def test_missing_explicit_ini_is_a_clean_error(deck, tmp_path, capsys):
    rc = main(["run", str(deck), "--tracewin-ini", str(tmp_path / "missing.ini"),
               "--mode", "envelope", "--out", str(tmp_path), "-q"])
    assert rc == 2
    assert "missing.ini" in capsys.readouterr().err


def test_energy_override_on_top_of_the_ini_is_warned(deck, tmp_path, capsys):
    common._TW_INI_ANNOUNCED.discard("overrides")
    rc = main(["run", str(deck), "--tracewin-ini", "--energy", "30",
               "--mode", "envelope", "--out", str(tmp_path / "o"), "-q"])
    err = capsys.readouterr().err
    assert rc == 0
    assert "not re-derived" in err and "energy" in err
    b = _h5_beam(tmp_path / "o")
    assert b["energy"] == 30.0 and abs(b["beta_z"] - 37.11) < 0.01   # NOT re-derived, as warned


def test_missing_sibling_is_a_clean_error(tmp_path, capsys):
    shutil.copy(FODO, tmp_path / "lonely.dat")
    rc = main(["run", str(tmp_path / "lonely.dat"), "--tracewin-ini",
               "--mode", "envelope", "--out", str(tmp_path), "-q"])
    assert rc == 2
    assert "lonely.ini" in capsys.readouterr().err


def test_a_lattice_given_as_the_ini_is_refused(deck, tmp_path, capsys):
    rc = main(["run", str(deck), "--tracewin-ini", str(deck), "--mode",
               "envelope", "--out", str(tmp_path), "-q"])
    assert rc == 2
    assert "expects a TraceWin .ini" in capsys.readouterr().err


def test_project_input_refuses_the_flag(deck, tmp_path, capsys):
    proj = write_project(tmp_path / "deck.lgproj", lattice_path=deck,
                         beam=BeamConfig(energy=7.5))
    rc = main(["run", str(proj), "--tracewin-ini", "--mode", "envelope",
               "--out", str(tmp_path / "o"), "-q"])
    assert rc == 2
    assert "always wins" in capsys.readouterr().err
    # and without the flag the project's beam is used, untouched by the sibling .ini
    out = tmp_path / "o2"
    assert main(["run", str(proj), "--mode", "envelope", "--out", str(out), "-q"]) == 0
    assert _h5_beam(out)["energy"] == 7.5


def test_load_input_and_build_scan_point_agree(deck):
    _lat, cfg, _conv = common.load_input(str(deck), tracewin_ini="auto")
    point = common.build_scan_point(str(deck), tracewin_ini="auto")
    assert BeamConfig(**point.beam_config) == cfg
    assert cfg.energy == 20.0
    # overrides are applied after the .ini beam
    point2 = common.build_scan_point(str(deck), tracewin_ini=True,
                                     beam_overrides={"energy": "33"})
    assert point2.beam_config["energy"] == 33.0
    assert point2.beam_config["emit_ny"] == cfg.emit_ny
    proj = write_project(deck.with_suffix(".lgproj"), lattice_path=deck,
                         beam=BeamConfig(energy=7.5))
    with pytest.raises(ValueError, match="always wins"):
        common.build_scan_point(str(proj), tracewin_ini="auto")
    with pytest.raises(ValueError, match="always wins"):
        common.load_input(str(proj), tracewin_ini=True)
    assert common.build_scan_point(str(proj)).beam_config["energy"] == 7.5


def test_scan_uses_the_ini_for_every_point(deck, tmp_path, capsys):
    out = tmp_path / "scan.csv"
    rc = main(["scan", str(deck), "--tracewin-ini", "--vary", "current=0,1",
               "--mode", "envelope", "--out", str(out), "-q"])
    err = capsys.readouterr().err
    assert rc == 0, err
    assert err.count("beam from deck.ini") == 1           # announced once, not per point
    rows = list(csv.DictReader(out.open(encoding="utf-8")))
    assert len(rows) == 2
    assert all(abs(float(r["ref_w_kin"]) - 20.0) < 1e-6 for r in rows)


def test_batch_job_key(deck, tmp_path, capsys):
    jobs = [
        {"name": "auto", "input": "deck.dat", "mode": "envelope", "tracewin_ini": True},
        {"name": "rel", "input": "deck.dat", "mode": "envelope", "tracewin_ini": "deck.ini"},
        {"name": "plain", "input": "deck.dat", "mode": "envelope"},
    ]
    jf = tmp_path / "jobs.json"
    jf.write_text(json.dumps(jobs))
    assert main(["batch", str(jf), "--out", str(tmp_path), "-q"]) == 0
    rows = {r["name"]: r for r in csv.DictReader(
        (tmp_path / "batch_summary.csv").open(encoding="utf-8"))}
    assert abs(float(rows["auto"]["ref_w_kin"]) - 20.0) < 1e-6
    assert abs(float(rows["rel"]["ref_w_kin"]) - 20.0) < 1e-6
    assert abs(float(rows["plain"]["ref_w_kin"]) - BeamConfig().energy) < 1e-6


def test_failure_study_passes_it_through(deck):
    from linac_gen.failures.study import FailureStudy
    st = FailureStudy(str(deck), tracewin_ini="auto")
    assert st._point(()).beam_config["energy"] == 20.0
    assert FailureStudy(str(deck))._point(()).beam_config["energy"] == BeamConfig().energy


def test_twiss_and_export_accept_the_flag(deck, tmp_path, capsys):
    assert main(["twiss", str(deck), "--tracewin-ini", "--mode", "whole", "-q"]) == 0
    assert "beam from deck.ini" in capsys.readouterr().err
    out = tmp_path / "deck.madx"
    assert main(["export", str(deck), str(out), "--tracewin-ini"]) == 0
    assert out.is_file()
