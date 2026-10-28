"""Ring detection on TraceWin decks, through the parser: a ``;@LG ring``
declaration, a closed ring of bends without one, and an open arc."""
from linac_gen.io.tracewin_parser import parse_tracewin
from linac_gen.ring.detect import detect_ring

DECK = """;@LG nx=96 ny=96 nz=96    # PIC grid
;@LG harmonic = 84
;@LG ring
DRIFT 100 20
END
"""


def _parse(tmp_path, text):
    p = tmp_path / "deck.dat"
    p.write_text(text)
    return parse_tracewin(str(p))


def test_ring_declaration_reaches_ring_detection_through_the_parser(tmp_path):
    lat, meta = _parse(tmp_path, DECK)
    v = detect_ring(lat, meta)
    assert v.declared and v.is_ring and v.harmonic == 84


def test_a_deck_without_directives_is_not_declared_a_ring(tmp_path):
    lat, meta = _parse(tmp_path, "DRIFT 100 20\nEND\n")
    assert meta["lg_options"] == {}
    v = detect_ring(lat, meta)
    assert not v.declared and not v.is_ring


def _arc_deck(n_bends: int) -> str:
    # n bends of 90° (rho 2 m) with 1 m drifts between them: four close a ring
    return "DRIFT 1000 30\nBEND 90 2000 0 30 0\n" * n_bends + "END\n"


def test_closed_tracewin_ring_is_reported_but_not_declared(tmp_path):
    lat, meta = _parse(tmp_path, _arc_deck(4))
    v = detect_ring(lat, meta)
    assert v.is_ring and not v.declared
    assert abs(v.total_angle_deg - 360.0) < 1e-9


def test_open_tracewin_arc_is_not_a_ring(tmp_path):
    lat, meta = _parse(tmp_path, _arc_deck(2))
    v = detect_ring(lat, meta)
    assert not v.is_ring and not v.declared
    assert abs(v.total_angle_deg - 180.0) < 1e-9


def test_harmonic_one_is_kept(tmp_path):
    # (h = 1 equals True in Python; a membership test once dropped it)
    lat, meta = _parse(tmp_path, ";@LG ring\n;@LG harmonic=1\nDRIFT 100 20\nEND\n")
    assert detect_ring(lat, meta).harmonic == 1
    lat, meta = _parse(tmp_path, ";@LG ring\n;@LG harmonic\nDRIFT 100 20\nEND\n")
    assert detect_ring(lat, meta).harmonic is None          # a bare flag is no number
