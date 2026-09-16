"""``python -m linac_gen twini`` end to end on the LightWin ads.ini fixture."""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from linac_gen.__main__ import main
from linac_gen.io.project import load_project

REPO = Path(__file__).resolve().parents[2]
ADS = REPO / "tests" / "io" / "fixtures" / "tracewin_ini" / "ads.ini"
FODO = REPO / "tests" / "io" / "fixtures" / "simple_fodo.dat"


@pytest.fixture
def project(tmp_path):
    """A TraceWin-style project folder: deck.dat + deck.ini (ads beam)."""
    shutil.copy(ADS, tmp_path / "deck.ini")
    shutil.copy(FODO, tmp_path / "deck.dat")            # FREQ 352.21
    return tmp_path


def test_default_output_is_the_converted_beam(capsys):
    assert main(["twini", str(ADS)]) == 0
    out, err = capsys.readouterr()
    assert "ads.ini" in out and "beam 1: proton, 20 MeV, 100 MHz, 5 mA, 50000 particles" in out
    assert "alpha_z -0.17661" in out and "beta_z 37.1099 deg/MeV" in out
    assert "recorded, not applied: nbr_thread 8, PICNIC r/z 20x40, xy 9x9" in out
    assert "warning" not in err


def test_report_and_json(capsys):
    assert main(["twini", str(ADS), "--report"]) == 0
    out = capsys.readouterr().out
    assert "Applied to the HELIX beam" in out and "Not decoded" in out
    assert main(["twini", str(ADS), "--json"]) == 0
    data = json.loads(capsys.readouterr().out)
    assert data["beam"]["energy"] == 20.0 and data["beam"]["species"] == "proton"
    assert data["recorded"]["picnic_z_mesh"] == 40
    assert data["warnings"] == [] and data["beam_index"] == 1
    assert "i32@0x2ee0" in data["unknown_slots"]


def test_quiet_prints_only_warnings(capsys):
    assert main(["twini", str(ADS), "-q", "--species", "H-"]) == 0
    out, err = capsys.readouterr()
    assert out == ""
    assert "warning:" in err and "'Proton'" in err


def test_lgproj_next_to_the_deck(project, capsys):
    assert main(["twini", str(project / "deck.ini"), "--lgproj"]) == 0
    out, err = capsys.readouterr()
    assert "wrote" in out and "deck.lgproj" in out
    # the sibling deck has FREQ 352.21 while the beam is 100 MHz → warned
    assert "352.21" in err and "FREQ" in err
    proj = load_project(project / "deck.lgproj")
    assert proj.beam.energy == 20.0 and proj.beam.frequency == 100.0
    assert Path(proj.lattice_path).resolve() == (project / "deck.dat").resolve()
    assert json.loads((project / "deck.lgproj").read_text())["lattice_path"] == "deck.dat"
    assert proj.convergence["tracewin_ini"]["picnic_r_mesh"] == 20
    # second run refuses to overwrite; --force allows it
    assert main(["twini", str(project / "deck.ini"), "--lgproj"]) == 2
    assert "exists" in capsys.readouterr().err
    assert main(["twini", str(project / "deck.ini"), "--lgproj", "--force", "-q"]) == 0


def test_lgproj_explicit_paths(project, tmp_path, capsys):
    other = tmp_path / "elsewhere" / "line.dat"
    other.parent.mkdir()
    shutil.copy(FODO, other)
    out = tmp_path / "out" / "line.lgproj"
    assert main(["twini", str(project / "deck.ini"), "--lgproj", str(out),
                 "--lattice", str(other), "--species", "H-"]) == 0
    proj = load_project(out)
    assert proj.beam.species == "H-"
    assert Path(proj.lattice_path).resolve() == other.resolve()


def test_lgproj_without_a_deck_is_refused(tmp_path, capsys):
    ini = tmp_path / "lonely.ini"
    shutil.copy(ADS, ini)
    assert main(["twini", str(ini), "--lgproj"]) == 2
    assert "--lattice" in capsys.readouterr().err
    assert not (tmp_path / "lonely.lgproj").exists()
    assert main(["twini", str(ini), "--lattice", str(tmp_path / "nope.dat")]) == 2


def test_lgproj_pointing_at_a_directory_is_refused(project, tmp_path, capsys):
    d = tmp_path / "adir"
    d.mkdir()
    assert main(["twini", str(project / "deck.ini"), "--lgproj", str(d), "--force"]) == 2
    assert "is a directory" in capsys.readouterr().err


def test_unpopulated_beam_2_fails_cleanly(capsys):
    assert main(["twini", str(ADS), "--beam", "2"]) == 1
    assert "not populated" in capsys.readouterr().err


def test_bad_inputs(tmp_path, capsys):
    assert main(["twini", str(tmp_path / "missing.ini")]) == 2
    assert "not found" in capsys.readouterr().err
    txt = tmp_path / "text.ini"
    txt.write_text("[beam]\nenergy=1\n")
    assert main(["twini", str(txt)]) == 1
    assert "not a TraceWin options file" in capsys.readouterr().err


def test_unparseable_sibling_deck_is_a_note_not_a_failure(tmp_path, capsys):
    shutil.copy(ADS, tmp_path / "deck.ini")
    (tmp_path / "deck.dat").write_bytes(b"\xff\xfe garbage \x00\x00")
    rc = main(["twini", str(tmp_path / "deck.ini")])
    out, err = capsys.readouterr()
    assert rc == 0 and "beam 1: proton" in out
