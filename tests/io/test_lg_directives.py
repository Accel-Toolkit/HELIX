"""``;@LG`` deck directives are parsed as the manual describes them: several
``key=value`` pairs on a line, a trailing ``#`` or ``;`` comment, a bare key as
a flag, a value running to the next ``key=`` (spaces included).  The parser
used to keep everything after the first ``=`` as one string (``nx`` became
``"96 ny=96 nz=96  # PIC grid"``); nothing read the options then; ring
detection does (tests/ring/test_detect_tracewin.py).  The deck writer writes
them back."""
import warnings

import pytest

from linac_gen.io.tracewin_parser import parse_tracewin
from linac_gen.io.tracewin_writer import write_tracewin

DECK = """;@LG nx=96 ny=96 nz=96    # PIC grid
;@LG kernel=tsc ; TraceWin-style comment
;@LG harmonic = 84
;@LG ring
;@LG use_gpu=auto
DRIFT 100 20
END
"""


def _parse(tmp_path, text=DECK):
    p = tmp_path / "lg.dat"
    p.write_text(text)
    return parse_tracewin(str(p))


def test_directives_follow_the_documented_syntax(tmp_path):
    lat, meta = _parse(tmp_path)
    assert meta["lg_options"] == {"nx": 96, "ny": 96, "nz": 96, "kernel": "tsc",
                                  "harmonic": 84, "ring": True, "use_gpu": "auto"}
    assert lat.lg_options == meta["lg_options"]
    assert len(lat.elements) == 1                  # directives make no elements


def test_a_value_with_spaces_is_one_value(tmp_path):
    """``note=booster ring study`` used to give note='booster' and two flags,
    ``ring`` and ``study`` — and ring detection then called the linac deck a
    ring.  A value runs to the next ``key=``, as the parser of 1.20 kept it."""
    from linac_gen.ring.detect import detect_ring
    lat, meta = _parse(tmp_path, ";@LG note=booster ring study   # what it is\n"
                                 ";@LG title=two  spaces kept kernel=tsc\nDRIFT 100 20\nEND\n")
    assert meta["lg_options"] == {"note": "booster ring study", "title": "two  spaces kept",
                                  "kernel": "tsc"}
    verdict = detect_ring(lat, meta)
    assert not verdict.declared and not verdict.is_ring


def test_flags_come_before_the_pairs_of_their_line(tmp_path):
    _lat, meta = _parse(tmp_path, ";@LG ring harmonic=84\n;@LG =5 lone\nDRIFT 100 20\nEND\n")
    assert meta["lg_options"] == {"ring": True, "harmonic": 84, "lone": True}


_ROUND_TRIP = """;@LG ring
;@LG harmonic=84
;@LG note=booster ring study
;@LG nx=96 ny=96 nz=96    # PIC grid
;@LG grid_extent=5.0 gpu_particles=false
DRIFT 100 20
END
"""


def test_the_writer_keeps_the_directives(tmp_path):
    """The manual says the directives are kept for round-tripping; the deck
    writer (GUI Save Lattice) used to drop them — a saved ring deck was no
    longer declared a ring."""
    from linac_gen.ring.detect import detect_ring
    lat, meta = _parse(tmp_path, _ROUND_TRIP)
    out = tmp_path / "saved.dat"
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        write_tracewin(lat, out)
    text = out.read_text()
    assert text.startswith(";@LG ring\n;@LG harmonic=84\n;@LG note=booster ring study\n")
    back, meta2 = parse_tracewin(str(out))
    assert meta2["lg_options"] == meta["lg_options"]
    assert list(meta2["lg_options"]) == list(meta["lg_options"])
    assert {k: type(v) for k, v in meta2["lg_options"].items()} == \
        {k: type(v) for k, v in meta["lg_options"].items()}
    verdict = detect_ring(back, meta2)
    assert verdict.declared and verdict.harmonic == 84
    assert [type(e).__name__ for e in back.elements] == ["Drift"]


def test_an_option_that_cannot_read_back_is_left_out_with_a_warning(tmp_path):
    lat, _meta = _parse(tmp_path, ";@LG ring\nDRIFT 100 20\nEND\n")
    lat.lg_options.update({"note": "a # b", "two words": 1, "ok": "fine"})
    out = tmp_path / "saved.dat"
    with pytest.warns(UserWarning, match="would not read back the same") as rec:
        write_tracewin(lat, out)
    assert len(rec) == 2
    assert parse_tracewin(str(out))[1]["lg_options"] == {"ring": True, "ok": "fine"}


def test_a_deck_without_directives_writes_none(tmp_path):
    lat, _meta = _parse(tmp_path, "DRIFT 100 20\nEND\n")
    out = tmp_path / "saved.dat"
    write_tracewin(lat, out)
    assert out.read_text() == "DRIFT 100 20\nEND\n"
