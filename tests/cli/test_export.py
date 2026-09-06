"""The `python -m linac_gen export` subcommand (MAD-X output)."""
from pathlib import Path

import pytest

from linac_gen.__main__ import main
from linac_gen.elements.quadrupole import Quadrupole
from linac_gen.io.madx_parser import parse_madx
from linac_gen.io.tracewin_parser import parse_tracewin

_REPO = Path(__file__).resolve().parents[2]
_DAT = _REPO / "examples" / "fodo_cell.dat"
_LGPROJ = _REPO / "examples" / "csr_chicane.lgproj"


def test_export_writes_madx_that_reimports(tmp_path, capsys):
    out = tmp_path / "fodo.madx"
    rc = main(["export", str(_DAT), str(out), "--energy", "3",
               "--species", "H-", "--freq", "352.21"])
    assert rc == 0
    printed = capsys.readouterr().out
    assert "wrote" in printed and "H- 3 MeV" in printed
    back, meta = parse_madx(str(out))
    assert meta["warnings"] == []
    orig = parse_tracewin(str(_DAT))[0]
    q0 = [e.gradient for e in orig.elements if isinstance(e, Quadrupole)]
    q1 = [e.gradient for e in back.elements if isinstance(e, Quadrupole)]
    assert q1 == q0                      # exact inverse (repr formatting)
    # The BEAM line carried H- as an ion with the right charge.
    assert meta["reference"].species.charge == -1
    assert meta["reference"].w_kin == pytest.approx(3.0, rel=1e-9)


def test_export_from_project_uses_its_beam(tmp_path, monkeypatch):
    monkeypatch.chdir(_REPO)          # project lattice paths are repo-relative
    out = tmp_path / "chicane.seq"
    rc = main(["export", str(_LGPROJ), str(out), "-q"])
    assert rc == 0
    txt = out.read_text()
    assert "BEAM, particle=ion" in txt and "charge=-1" in txt  # H- project
    assert "SBEND" in txt


def test_bare_lattice_without_energy_is_refused(tmp_path, capsys):
    rc = main(["export", str(_DAT), str(tmp_path / "x.madx")])
    assert rc == 2
    assert "--energy" in capsys.readouterr().err


def test_unknown_output_format_is_refused(tmp_path, capsys):
    rc = main(["export", str(_DAT), str(tmp_path / "x.txt"), "--energy", "3"])
    assert rc == 2
    assert "--format" in capsys.readouterr().err


def test_missing_input(tmp_path):
    assert main(["export", str(tmp_path / "nope.dat"),
                 str(tmp_path / "x.madx"), "--energy", "3"]) == 2


def test_strict_refuses_unrepresentable_elements(tmp_path, capsys):
    deck = tmp_path / "elec.dat"
    deck.write_text("DRIFT 100 30 0\nTHIN_STEERING 0.001 0.0 30 1\n"
                    "DRIFT 100 30 0\nEND\n")
    out = tmp_path / "elec.madx"
    rc = main(["export", str(deck), str(out), "--energy", "2.1",
               "--strict", "-q"])
    assert rc == 1
    assert "refused" in capsys.readouterr().err
    assert not out.exists()
    # Default mode: written, with the warning on stderr and the geometry kept.
    rc = main(["export", str(deck), str(out), "--energy", "2.1", "-q"])
    assert rc == 0
    err = capsys.readouterr().err
    assert "export warning" in err and "MARKER" in err
    back, _ = parse_madx(str(out))
    assert sum(e.length for e in back.elements) == pytest.approx(200.0)


def test_output_equal_to_input_is_refused(tmp_path, capsys):
    src = tmp_path / "ring.madx"
    src.write_text((_REPO / "examples" / "madx" / "fodo.madx").read_text())
    before = src.read_text()
    rc = main(["export", str(src), str(src), "--energy", "3"])
    assert rc == 2
    assert "never overwritten" in capsys.readouterr().err
    assert src.read_text() == before


def test_linearize_flag_exports_explicit_matrix_as_madx_matrix(tmp_path, capsys):
    """An Elegant EMATRIX (→ MatrixElement) is a MARKER + body DRIFT by
    default and a MAD-X MATRIX under --linearize."""
    from linac_gen.elements.matrix_element import MatrixElement
    lte = tmp_path / "line.lte"
    lte.write_text("d: drift, l=0.1\n"
                   "m2: ematrix, l=0.1, r11=1, r12=0.5, r22=1, r33=1, r44=1, "
                   "r55=1, r66=1\n"
                   "l: line=(d, m2, d)\n")
    out = tmp_path / "line.madx"
    rc = main(["export", str(lte), str(out), "--energy", "5", "-q"])
    assert rc == 0
    err = capsys.readouterr().err
    assert "MARKER + body DRIFT" in err
    assert "MATRIX" not in out.read_text()
    rc = main(["export", str(lte), str(out), "--energy", "5", "--linearize", "-q"])
    assert rc == 0
    err = capsys.readouterr().err
    assert "linearised as MAD-X MATRIX" in err
    back, _ = parse_madx(str(out))
    (me,) = [e for e in back.elements if isinstance(e, MatrixElement)]
    assert me.matrix[0, 1] == pytest.approx(0.5, rel=1e-12)   # mm/mrad == m/rad
    assert me.length == pytest.approx(100.0, rel=1e-12)
