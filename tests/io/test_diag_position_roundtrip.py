# tests/io/test_diag_position_roundtrip.py
"""DIAG_POSITION operand parsing + write round-trip (diagnostic matching).

TraceWin: ``DIAG_POSITION N X Y [dm]`` — N = diagnostic (family) number
linking to ``ADJUST N v`` cards; X/Y = wanted beam centroid (mm) with
``|value| >= 1e50`` leaving that plane unconstrained; dm = diagnostic
accuracy (mm, default 1).  Native ``BPM :`` cards are target-less BPMs
and must round-trip as ``BPM`` (not be rewritten to DIAG_POSITION).
"""
import os
import tempfile

import pytest

from linac_gen.elements.marker import Marker
from linac_gen.io.tracewin_parser import parse_tracewin
from linac_gen.io.tracewin_writer import write_tracewin


def _parse(tmp_path, body):
    dat = tmp_path / "in.dat"
    dat.write_text("FREQ 352.21\n" + body + "END\n")
    return parse_tracewin(str(dat))


def _bpms(lat):
    return [e for e in lat.elements if getattr(e, "is_bpm", False)]


def _roundtrip(lat, tmp_path):
    out = str(tmp_path / "rt.dat")
    write_tracewin(lat, out)
    lat2, _ = parse_tracewin(out)
    return lat2, open(out).read()


def test_parse_operands_label_family_targets_dm(tmp_path):
    lat, meta = _parse(tmp_path,
                       "D07BPM: DIAG_POSITION 11 0.0579 -0.0603 0.25\n")
    assert not meta["warnings"]
    (m,) = _bpms(lat)
    assert m.name == "D07BPM"
    assert m.diag_family == 11
    assert m.x_target_mm == pytest.approx(0.0579)
    assert m.y_target_mm == pytest.approx(-0.0603)
    assert m.accuracy_mm == pytest.approx(0.25)


def test_sentinel_1e50_disables_plane(tmp_path):
    lat, _ = _parse(tmp_path, "DIAG_POSITION 3 1e50 -0.2\n"
                              "DIAG_POSITION 3 0.1 -1e51\n")
    a, b = _bpms(lat)
    assert a.x_target_mm is None and a.y_target_mm == pytest.approx(-0.2)
    assert b.x_target_mm == pytest.approx(0.1) and b.y_target_mm is None


def test_dm_defaults_to_1mm_and_family_only_card(tmp_path):
    lat, _ = _parse(tmp_path, "DIAG_POSITION 1\n")
    (m,) = _bpms(lat)
    assert m.diag_family == 1
    assert m.x_target_mm is None and m.y_target_mm is None
    assert m.accuracy_mm == 1.0


def test_bare_card_all_none(tmp_path):
    # correction_demo.dat idiom — plain BPM behavior must be unchanged.
    lat, _ = _parse(tmp_path, "DIAG_POSITION\n")
    (m,) = _bpms(lat)
    assert m.diag_family is None
    assert m.x_target_mm is None and m.y_target_mm is None
    lat2, txt = _roundtrip(lat, tmp_path)
    assert "DIAG_POSITION\n" in txt          # stays bare — no operands
    (m2,) = _bpms(lat2)
    assert m2.diag_family is None


def test_write_reparse_equality_with_label(tmp_path):
    lat, _ = _parse(tmp_path,
                    "D01BPM: DIAG_POSITION 12 0.0035567627 -7.20253e-05\n"
                    "DIAG_POSITION 11 1e50 0.5 0.1\n")
    lat2, txt = _roundtrip(lat, tmp_path)
    for a, b in zip(_bpms(lat), _bpms(lat2)):
        assert a.name == b.name
        assert a.diag_family == b.diag_family
        assert a.x_target_mm == b.x_target_mm
        assert a.y_target_mm == b.y_target_mm
        assert a.accuracy_mm == b.accuracy_mm
    assert "D01BPM: DIAG_POSITION 12" in txt
    # sentinel-disabled plane re-emits the sentinel, dm survives
    assert "DIAG_POSITION 11 1e50 0.5 0.1" in txt


def test_bpm_card_round_trips_as_bpm(tmp_path):
    lat, _ = _parse(tmp_path, "BPM :\nDRIFT 100 20 0\n")
    (m,) = _bpms(lat)
    assert m.origin_keyword == "BPM" and m.diag_family is None
    lat2, txt = _roundtrip(lat, tmp_path)
    assert "BPM\n" in txt and "DIAG_POSITION" not in txt
    (m2,) = _bpms(lat2)
    assert m2.origin_keyword == "BPM" and m2.is_bpm


def test_duplicate_labels_deduplicated(tmp_path):
    lat, _ = _parse(tmp_path, "DUP: DIAG_POSITION 1 0.1 0.2\n"
                              "DUP: DIAG_POSITION 1 0.3 0.4\n")
    names = [m.name for m in _bpms(lat)]
    assert names == ["DUP", "DUP_2"]


def test_target_overrides_never_serialize(tmp_path):
    lat, _ = _parse(tmp_path, "D1: DIAG_POSITION 2 0.1 0.2\n")
    (m,) = _bpms(lat)
    m.diag_target_override = (9.9, -9.9, None)   # runtime override
    _, txt = _roundtrip(lat, tmp_path)
    assert "9.9" not in txt              # deck keeps only card fields
    assert "0.1 0.2" in txt


def test_malformed_operands_keep_the_marker(tmp_path):
    """Bad operands must degrade to a bare BPM, not delete the element
    (dropping it would shift every legacy index-ADJUST downstream)."""
    lat, meta = _parse(tmp_path, "DIAG_POSITION 11 0,5 0.3\n")
    assert len(meta["warnings"]) == 1
    (m,) = _bpms(lat)
    assert m.diag_family is None and m.x_target_mm is None


def test_label_colliding_with_auto_name_deduped(tmp_path):
    lat, _ = _parse(tmp_path, "DIAG_POSITION 1 0.1 0.1\n"
                              "BPM_001: DIAG_POSITION 1 9.9 9.9\n")
    names = [m.name for m in _bpms(lat)]
    assert len(set(names)) == 2          # no aliasing in name-keyed maps


def test_unwritable_api_label_suppressed_not_corrupting(tmp_path):
    lat = _parse(tmp_path, "DRIFT 100 20 0\n")[0]
    lat.add(Marker("1BPM", is_bpm=True, diag_family=2, x_target_mm=1.0))
    lat2, txt = _roundtrip(lat, tmp_path)
    assert "1BPM:" not in txt            # unparseable label suppressed
    (m2,) = _bpms(lat2)
    assert m2.diag_family == 2 and m2.x_target_mm == pytest.approx(1.0)


def test_fnalscl_diag_position_operands_roundtrip():
    deck = os.path.join(os.path.dirname(__file__), "..", "..",
                        "examples", "piplattice", "fnalscl.dat")
    if not os.path.exists(deck):
        pytest.skip("fnalscl.dat not present")
    lat, meta = parse_tracewin(deck)
    assert not meta["warnings"]
    bpms = _bpms(lat)
    assert len(bpms) == 29
    fams = {}
    for b in bpms:
        fams[b.diag_family] = fams.get(b.diag_family, 0) + 1
    assert fams == {11: 25, 12: 4}
    d01 = next(b for b in bpms if b.name == "D01BPM")
    assert d01.diag_family == 12
    assert d01.x_target_mm == pytest.approx(0.0035567627)
    assert d01.y_target_mm == pytest.approx(-7.20253e-05)
    assert all(b.accuracy_mm == 1.0 for b in bpms)   # deck omits dm
    with tempfile.TemporaryDirectory() as d:
        out = os.path.join(d, "rt.dat")
        write_tracewin(lat, out)
        lat2, _ = parse_tracewin(out)
    bpms2 = _bpms(lat2)
    assert [(b.name, b.diag_family, b.x_target_mm, b.y_target_mm)
            for b in bpms] == \
           [(b.name, b.diag_family, b.x_target_mm, b.y_target_mm)
            for b in bpms2]
