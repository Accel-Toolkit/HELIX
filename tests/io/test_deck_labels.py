"""Deck labels survive parsing and writing.

``Q01: QUAD …`` / ``D01T: THIN_STEERING …`` / ``D01BPM: DIAG_POSITION …``: the
parser stamps the RAW label on every element the line adds (``label``), the
generated ``name`` (QUAD_001, STEER_001, de-duplicated BPM names) is unchanged,
and the writer emits the label again so a saved deck keeps its device names.
Unlabeled decks must round-trip byte-identically (no cosmetic labels).
"""
from __future__ import annotations
import contextlib, io, os
from pathlib import Path

import pytest

from linac_gen.elements.marker import Marker
from linac_gen.elements.quadrupole import Quadrupole
from linac_gen.elements.steerer import Steerer
from linac_gen.io.tracewin_parser import parse_tracewin
from linac_gen.io.tracewin_writer import write_tracewin
from tests.dataguard import needs  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
DECK = (
    "FREQ 352.2\n"
    "Q01: QUAD 85.3 -8.09 20 0 0 0 0 0\n"          # glued colon
    "D01T : THIN_STEERING 4.56e-5 0 20 0\n"       # spaced colon
    "D01T: THIN_STEERING 0 4.67e-5 20 0\n"        # duplicate label (fnalscl style)
    "D01BPM:DIAG_POSITION 12 0.1 -0.2\n"          # colon glued on both sides
    "DRIFT 100 20 0\n"
    "QUAD 85.3 8.1 20\n"                          # unlabeled
    "D02BPM: DIAG_POSITION 11 0.3 0.4\n"
    "END\n"
)


def _parse(path):
    with contextlib.redirect_stdout(io.StringIO()):
        return parse_tracewin(str(path))[0]


def _write(lat, path):
    write_tracewin(lat, str(path))
    return Path(path).read_text(encoding="latin-1")


def test_labels_captured_names_unchanged(tmp_path):
    p = tmp_path / "d.dat"; p.write_text(DECK, encoding="latin-1")
    lat = _parse(p)
    quads = [e for e in lat.elements if isinstance(e, Quadrupole)]
    steer = [e for e in lat.elements if isinstance(e, Steerer)]
    bpms = [e for e in lat.elements if isinstance(e, Marker) and e.is_bpm]
    assert [q.label for q in quads] == ["Q01", None]
    assert [q.name for q in quads] == ["QUAD_001", "QUAD_002"]
    assert [s.label for s in steer] == ["D01T", "D01T"]           # raw, not de-duplicated
    assert [s.name for s in steer] == ["STEER_001", "STEER_002"]
    assert [b.label for b in bpms] == ["D01BPM", "D02BPM"]
    assert [b.name for b in bpms] == ["D01BPM", "D02BPM"]
    drift = [e for e in lat.elements if type(e).__name__ == "Drift"][0]
    assert drift.label is None


def test_writer_emits_labels_and_round_trips(tmp_path):
    p = tmp_path / "d.dat"; p.write_text(DECK, encoding="latin-1")
    lat = _parse(p)
    txt = _write(lat, tmp_path / "w.dat")
    lines = txt.splitlines()
    assert any(l.startswith("Q01: QUAD ") for l in lines)
    assert sum(l.startswith("D01T: THIN_STEERING ") for l in lines) == 2
    assert any(l.startswith("D01BPM: DIAG_POSITION 12 ") for l in lines)
    assert any(l.startswith("QUAD ") for l in lines)               # the unlabeled quad stays bare
    lat2 = _parse(tmp_path / "w.dat")
    assert [e.name for e in lat2.elements] == [e.name for e in lat.elements]
    assert [e.label for e in lat2.elements] == [e.label for e in lat.elements]
    # write → parse → write is idempotent
    assert _write(lat2, tmp_path / "w2.dat") == txt


def test_label_grammar_guard(tmp_path):
    """A label the parser could not re-read is not emitted."""
    p = tmp_path / "d.dat"; p.write_text("QUAD 85.3 -8.09 20\nEND\n", encoding="latin-1")
    lat = _parse(p)
    lat.elements[0].label = "bad label"                             # space → would corrupt the card
    txt = _write(lat, tmp_path / "w.dat")
    assert txt.splitlines()[0].startswith("QUAD ")
    lat.elements[0].label = "Q01"
    assert _write(lat, tmp_path / "w3.dat").splitlines()[0].startswith("Q01: QUAD ")


def test_latin1_byte_on_labelled_line(tmp_path):
    p = tmp_path / "d.dat"
    p.write_bytes(b"Q01: QUAD 85.3 -8.09 20 ; d\xe9viation\nEND\n")
    lat = _parse(p)
    assert lat.elements[0].label == "Q01" and lat.elements[0].name == "QUAD_001"
    assert _write(lat, tmp_path / "w.dat").splitlines()[0].startswith("Q01: QUAD ")


@pytest.mark.parametrize("deck", ["examples/fodo_cell.dat", "examples/correction_demo/correction_demo.dat"])
def test_unlabeled_decks_write_without_labels(deck):
    lat = _parse(REPO / deck)
    assert all(e.label is None for e in lat.elements)
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        txt = _write(lat, Path(td) / "w.dat")
    assert not any(":" in l.split(" ")[0] for l in txt.splitlines() if l and not l.startswith(";"))


@needs("examples/piplattice/fnalscl.dat")   # PIP-II deck: dev checkouts only
def test_fnalscl_labels():
    deck = REPO / "examples/piplattice/fnalscl.dat"
    cwd = os.getcwd(); os.chdir(deck.parent)
    try:
        lat = _parse(deck)
    finally:
        os.chdir(cwd)
    steer = [e for e in lat.elements if isinstance(e, Steerer)]
    assert steer[0].label == steer[1].label == "D01T" and steer[0].name != steer[1].name
    quads = [e for e in lat.elements if isinstance(e, Quadrupole)]
    assert [q.label for q in quads[:5]] == ["Q01", "Q02", "Q03", "Q04", "Q11"]
    bpm = next(e for e in lat.elements if isinstance(e, Marker) and getattr(e, "is_bpm", False))
    assert bpm.name == "D01BPM" and bpm.label == "D01BPM"
    assert sum(e.label is not None for e in lat.elements) >= 100


def _tiny_map(tmp_path):
    """A minimal 1-D field map (same recipe as tests/io/test_superpose_parsing.py)."""
    import numpy as np
    z = np.linspace(0, 300, 31); ez = np.zeros_like(z)
    with open(tmp_path / "sol.bsz", "w") as fh:
        fh.write(f"{len(z) - 1} {z[-1] / 1000.0}\n1.0\n")
        for v in ez:
            fh.write(f"{v}\n")


def test_label_not_stamped_on_flushed_cluster_or_restored_orphans(tmp_path):
    """A labelled line that also flushes a pending SUPERPOSE cluster or restores a
    SHIFT orphan must label only its own element (adversarial-review finding)."""
    _tiny_map(tmp_path)
    p = tmp_path / "d.dat"
    p.write_text("FREQ 352.2\nSUPERPOSE_MAP 0\nFIELD_MAP 10 300 0 16 2.0 0 0 0 sol\nQ01: QUAD 85.3 -8.09 20\nEND\n", encoding="latin-1")
    lat = _parse(p)
    labs = [(type(e).__name__, e.label) for e in lat.elements]
    assert ("Quadrupole", "Q01") in labs
    assert all(lab is None for t, lab in labs if t != "Quadrupole")
    p2 = tmp_path / "e.dat"
    p2.write_text("FREQ 352.2\nSHIFT_IN_FIELD_MAP 5\nD01: MARKER\nQ01: QUAD 85.3 -8.09 20\nEND\n", encoding="latin-1")
    lat = _parse(p2)
    labs = {type(e).__name__: e.label for e in lat.elements}
    assert labs["Quadrupole"] == "Q01" and labs["Marker"] == "D01"


def test_labels_on_commands_lattice_and_diag_phase_round_trip(tmp_path):
    p = tmp_path / "d.dat"
    p.write_text("FREQ 352.2\nLBL1: LATTICE 2 0\nSPH1: SET_SYNC_PHASE\nDP1: DIAG_PHASE 1\nQUAD 85.3 -8.09 20\nDRIFT 100 20 0\nLEND1: LATTICE_END\nEND\n", encoding="latin-1")
    lat = _parse(p)
    txt = _write(lat, tmp_path / "w.dat")
    for tok in ("LBL1: LATTICE ", "SPH1: SET_SYNC_PHASE", "DP1: DIAG_PHASE 1", "LEND1: LATTICE_END"):
        assert any(l.startswith(tok) for l in txt.splitlines()), tok
    lat2 = _parse(tmp_path / "w.dat")
    assert [e.label for e in lat2.elements] == [e.label for e in lat.elements]


def test_unicode_letter_label_round_trips(tmp_path):
    p = tmp_path / "d.dat"; p.write_bytes("Écart: QUAD 85.3 -8.09 20\nEND\n".encode("latin-1"))
    lat = _parse(p)
    assert lat.elements[0].label == "Écart"
    txt = _write(lat, tmp_path / "w.dat")
    assert txt.splitlines()[0].startswith("Écart: QUAD ")
    lat.elements[0].label = "Q\u03b1"                                 # not latin-1 encodable → dropped, not corrupted
    assert _write(lat, tmp_path / "w2.dat").splitlines()[0].startswith("QUAD ")


def test_interior_marker_label_survives_superpose_write(tmp_path):
    _tiny_map(tmp_path)
    p = tmp_path / "d.dat"
    p.write_text("FREQ 352.2\nSHIFT_IN_FIELD_MAP 5\nIM1: MARKER\nFIELD_MAP 10 300 0 16 2.0 0 0 0 sol\nDRIFT 10 20 0\nEND\n", encoding="latin-1")
    lat = _parse(p)
    txt = _write(lat, tmp_path / "w.dat")
    assert any(l.startswith("IM1: MARKER") for l in txt.splitlines())
