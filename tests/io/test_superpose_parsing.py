# tests/io/test_superpose_parsing.py
"""SUPERPOSE_MAP cluster parsing + writer round-trip (phase B).

Synthetic canonical 1-D field files (Nz Zmax / Norm / Nz+1 values) are
written per-test; decks resolve them via base_dir.  geom 10 = 1-D
static magnetic (.bsz), geom 100 = 1-D RF electric (.edz).
"""
import glob

import numpy as np
import pytest

from linac_gen.elements.field_map import FieldMap
from linac_gen.elements.superposed_field_map import SuperposedFieldMap
from linac_gen.io.tracewin_parser import parse_tracewin
from linac_gen.io.tracewin_writer import write_tracewin


def _write_1d(path, vals, zmax_m):
    lines = [f"{len(vals) - 1} {zmax_m:.6f}", "1.0"]
    lines += [f"{v:.9g}" for v in vals]
    path.write_text("\n".join(lines) + "\n")


@pytest.fixture()
def maps(tmp_path):
    z = np.linspace(0.0, 0.3, 31)
    _write_1d(tmp_path / "sol.bsz",
              0.4 * (0.5 - 0.5 * np.cos(2 * np.pi * z / 0.3)), 0.3)
    _write_1d(tmp_path / "cav.edz",
              2.0 * np.sin(np.pi * z / 0.3), 0.3)
    return tmp_path


def _parse(tmp_path, text, **kw):
    p = tmp_path / "deck.dat"
    p.write_text(text)
    return parse_tracewin(str(p), **kw)


# ── cluster assembly ─────────────────────────────────────────────────────
def test_two_child_cluster(maps):
    lat, meta = _parse(maps,
        "DRIFT 50 16 0\n"
        "SUPERPOSE_MAP 0\n"
        "FIELD_MAP 10 300 0 16 2.0 0 0 0 sol\n"
        "SUPERPOSE_MAP 100\n"
        "FIELD_MAP 10 300 0 16 1.0 0 0 0 sol\n"
        "DRIFT 100 16 0\nEND\n")
    assert meta["warnings"] == []
    types = [type(e).__name__ for e in lat.elements]
    assert types == ["Drift", "SuperposedFieldMap", "Drift"]
    sup = lat.elements[1]
    assert sup.length == pytest.approx(400.0)     # max(0+300, 100+300)
    assert [z0 for z0, _c in sup.children] == [0.0, 100.0]
    assert sup.aperture == pytest.approx(16.0)    # z0==0 carrier
    # A well-formed cluster is fully supported — strict must pass too.
    _parse(maps,
        "SUPERPOSE_MAP 0\nFIELD_MAP 10 300 0 16 2.0 0 0 0 sol\n"
        "SUPERPOSE_MAP 100\nFIELD_MAP 10 300 0 16 1.0 0 0 0 sol\n"
        "DRIFT 100 16 0\nEND\n", strict=True)


def test_single_pair_z0_zero_emits_plain_fieldmap(maps):
    lat, meta = _parse(maps,
        "SUPERPOSE_MAP 0\n"
        "FIELD_MAP 10 300 0 16 2.0 0 0 0 sol\n"
        "DRIFT 100 16 0\nEND\n")
    assert meta["warnings"] == []
    assert isinstance(lat.elements[0], FieldMap)       # NOT a container
    assert lat.elements[0].superpose_z0 == 0.0         # writer provenance


def test_back_to_back_clusters(maps):
    lat, meta = _parse(maps,
        "SUPERPOSE_MAP 0\nFIELD_MAP 10 300 0 16 2.0 0 0 0 sol\n"
        "SUPERPOSE_MAP 50\nFIELD_MAP 10 300 0 16 1.0 0 0 0 sol\n"
        "DRIFT 10 16 0\n"
        "SUPERPOSE_MAP 0\nFIELD_MAP 10 300 0 16 2.0 0 0 0 sol\n"
        "SUPERPOSE_MAP 80\nFIELD_MAP 10 300 0 16 1.0 0 0 0 sol\n"
        "END\nEND\n")
    kinds = [type(e).__name__ for e in lat.elements]
    assert kinds == ["SuperposedFieldMap", "Drift", "SuperposedFieldMap"]
    assert meta["warnings"] == []


def test_freq_closes_cluster_with_warning(maps):
    deck = ("SUPERPOSE_MAP 0\nFIELD_MAP 10 300 0 16 2.0 0 0 0 sol\n"
            "FREQ 162.5\n"
            "FIELD_MAP 100 300 -30 16 0 1.0 0 0 cav\n"
            "DRIFT 10 16 0\nEND\n")
    lat, meta = _parse(maps, deck)
    assert any("FREQ closes the open SUPERPOSE cluster" in w
               and "SEQUENTIALLY" in w
               for w in meta["warnings"]), meta["warnings"]
    kinds = [type(e).__name__ for e in lat.elements]
    # single-pair z0=0 flushed to plain; FREQ command; plain RF map
    assert kinds == ["FieldMap", "Freq", "FieldMap", "Drift"]
    # Physics downgrade (overlap → sequential, lattice lengthened):
    # strict must REFUSE — it used to return the mislengthened lattice
    # with permissive/strict behaving identically.
    with pytest.raises(ValueError, match="FREQ closes"):
        _parse(maps, deck, strict=True)


def test_dangling_superpose_warns_and_strict_raises(maps):
    _, meta = _parse(maps, "SUPERPOSE_MAP 0\nDRIFT 100 16 0\nEND\n")
    assert any("not followed by a FIELD_MAP" in w
               for w in meta["warnings"]), meta["warnings"]
    with pytest.raises(ValueError, match="not followed by a FIELD_MAP"):
        _parse(maps, "SUPERPOSE_MAP 0\nDRIFT 100 16 0\nEND\n",
               strict=True)


def test_consecutive_superpose_cards_warn(maps):
    lat, meta = _parse(maps,
        "SUPERPOSE_MAP 40\n"
        "SUPERPOSE_MAP 0\n"
        "FIELD_MAP 10 300 0 16 2.0 0 0 0 sol\n"
        "DRIFT 10 16 0\nEND\n")
    assert any("consecutive" in w for w in meta["warnings"]), \
        meta["warnings"]
    assert isinstance(lat.elements[0], FieldMap)   # surviving pair, z0=0


def test_nonzero_extras_warn_but_strict_passes(maps):
    """TraceWin itself ignores X/Y/θ without SUPERPOSE_MAP_OUT —
    ignoring is FAITHFUL, so this is a courtesy warning, not a
    downgrade."""
    _, meta = _parse(maps,
        "SUPERPOSE_MAP 0 5.0 0 0 0 0\n"
        "FIELD_MAP 10 300 0 16 2.0 0 0 0 sol\n"
        "SUPERPOSE_MAP 100\nFIELD_MAP 10 300 0 16 1.0 0 0 0 sol\n"
        "DRIFT 10 16 0\nEND\n")
    assert any("transverse/rotation operands ignored" in w
               for w in meta["warnings"])
    lat2, _ = _parse(maps,
        "SUPERPOSE_MAP 0 5.0 0 0 0 0\n"
        "FIELD_MAP 10 300 0 16 2.0 0 0 0 sol\n"
        "SUPERPOSE_MAP 100\nFIELD_MAP 10 300 0 16 1.0 0 0 0 sol\n"
        "DRIFT 10 16 0\nEND\n", strict=True)
    assert isinstance(lat2.elements[0], SuperposedFieldMap)


def test_unbuildable_cluster_falls_back_to_sequential(maps):
    """Non-positive span (every map ends before the entrance) cannot be
    built — permissive falls back to today's end-to-end physics with a
    loud warning; strict refuses."""
    deck = ("SUPERPOSE_MAP -400\n"
            "FIELD_MAP 10 300 0 16 2.0 0 0 0 sol\n"
            "SUPERPOSE_MAP -350\n"
            "FIELD_MAP 10 300 0 16 1.0 0 0 0 sol\n"
            "DRIFT 10 16 0\nEND\n")
    lat, meta = _parse(maps, deck)
    assert any("falling back to END-TO-END" in w
               for w in meta["warnings"]), meta["warnings"]
    kinds = [type(e).__name__ for e in lat.elements]
    assert kinds == ["FieldMap", "FieldMap", "Drift"]
    with pytest.raises(ValueError, match="cannot be built"):
        _parse(maps, deck, strict=True)


def test_missing_child_file_drops_child(maps):
    deck = ("SUPERPOSE_MAP 0\n"
            "FIELD_MAP 10 300 0 16 2.0 0 0 0 sol\n"
            "SUPERPOSE_MAP 100\n"
            "FIELD_MAP 10 300 0 16 1.0 0 0 0 nofile\n"
            "DRIFT 10 16 0\nEND\n")
    lat, meta = _parse(maps, deck)
    assert any("file missing" in w for w in meta["warnings"])
    assert any("cluster child dropped" in w for w in meta["warnings"])
    # Remaining single z0=0 child flushes as a plain map.
    assert isinstance(lat.elements[0], FieldMap)
    with pytest.raises(ValueError, match="file missing"):
        _parse(maps, deck, strict=True)


def test_no_position_zero_child_warns(maps):
    _, meta = _parse(maps,
        "SUPERPOSE_MAP 50\nFIELD_MAP 10 300 0 16 2.0 0 0 0 sol\n"
        "SUPERPOSE_MAP 100\nFIELD_MAP 10 300 0 16 1.0 0 0 0 sol\n"
        "DRIFT 10 16 0\nEND\n")
    assert any("no map at position 0" in w for w in meta["warnings"]), \
        meta["warnings"]


def test_sync_phase_binds_first_electric_child(maps):
    lat, meta = _parse(maps,
        "FREQ 162.5\n"
        "SET_SYNC_PHASE\n"
        "SUPERPOSE_MAP 0\n"
        "FIELD_MAP 10 300 0 16 2.0 0 0 0 sol\n"       # static — skipped
        "SUPERPOSE_MAP 30\n"
        "FIELD_MAP 100 240 -30 16 0 1.0 0 0 cav\n"    # RF — binds
        "DRIFT 10 16 0\nEND\n")
    sup = next(e for e in lat.elements
               if isinstance(e, SuperposedFieldMap))
    kids = [c for _z0, c in sup.children]
    static = [c for c in kids
              if not any(ch.is_electric for ch in c.field_data.channels)]
    rf = [c for c in kids
          if any(ch.is_electric for ch in c.field_data.channels)]
    assert len(static) == 1 and len(rf) == 1
    assert static[0].p_flag == 0            # sync flag skipped the solenoid
    assert rf[0].p_flag == 1                # ...and bound to the cavity


# ── writer round-trip ────────────────────────────────────────────────────
def test_round_trip_byte_idempotent(maps):
    lat, _ = _parse(maps,
        "DRIFT 50 16 0\n"
        "SUPERPOSE_MAP 0\n"
        "FIELD_MAP 10 300 0 16 2.0 0 0 0 sol\n"
        "SUPERPOSE_MAP 100\n"
        "FIELD_MAP 10 300 0 16 1.0 0 0 0 sol\n"
        "DRIFT 100 16 0\nEND\n")
    out1 = maps / "rt1.dat"
    write_tracewin(lat, str(out1))
    lat2, meta2 = parse_tracewin(str(out1))
    assert meta2["warnings"] == []
    sup = lat2.elements[1]
    assert isinstance(sup, SuperposedFieldMap)
    assert [z0 for z0, _c in sup.children] == [0.0, 100.0]
    out2 = maps / "rt2.dat"
    write_tracewin(lat2, str(out2))
    assert out1.read_text() == out2.read_text()


def test_round_trip_single_pair_provenance(maps):
    # Explicit FREQ so the writer's auto-FREQ (from the stamped element
    # frequency) round-trips symmetrically, as in every real deck.
    lat, _ = _parse(maps,
        "FREQ 352.21\n"
        "SUPERPOSE_MAP 0\n"
        "FIELD_MAP 10 300 0 16 2.0 0 0 0 sol\n"
        "DRIFT 100 16 0\nEND\n")
    out = maps / "rt.dat"
    write_tracewin(lat, str(out))
    text = out.read_text()
    assert "SUPERPOSE_MAP 0\nFIELD_MAP 10" in text
    lat2, _ = parse_tracewin(str(out))
    fm = next(e for e in lat2.elements if isinstance(e, FieldMap))
    assert fm.superpose_z0 == 0.0


# ── shipped-deck bit-compat ──────────────────────────────────────────────
def test_shipped_decks_unaffected():
    """No shipped deck uses SUPERPOSE_MAP: parsing every example must
    produce ZERO containers and ZERO superpose-related warnings —
    the mechanical form of the bit-compat claim."""
    import warnings as _w
    decks = sorted(glob.glob("examples/**/*.dat", recursive=True))
    # floor low enough for the PUBLIC checkout (PIP-II decks are not
    # distributed) while still proving a real sweep happened
    assert len(decks) > 15
    for deck in decks:
        with _w.catch_warnings():
            _w.simplefilter("ignore")
            lat, meta = parse_tracewin(deck)
        assert not any(isinstance(e, SuperposedFieldMap)
                       for e in lat.elements), deck
        assert not any("SUPERPOSE" in w for w in meta["warnings"]), \
            (deck, meta["warnings"])


# ── SHIFT_IN_FIELD_MAP (phase D) ─────────────────────────────────────────
_SHIFT_DECK = (
    "DRIFT 100 16 0\n"
    "SHIFT_IN_FIELD_MAP 200\n"
    "DIA1: DIAG_SIZE 2\n"
    "SHIFT_IN_FIELD_MAP 360\n"
    "DIA2: DIAG_SIZE 5\n"
    "SUPERPOSE_MAP 0\n"
    "FIELD_MAP 10 300 0 16 2.0 0 0 0 sol\n"
    "SUPERPOSE_MAP 100\n"
    "FIELD_MAP 10 300 0 16 1.0 0 0 0 sol\n"
    "DRIFT 100 16 0\nEND\n")


def test_shift_diagnostics_consumed_into_cluster(maps):
    lat, meta = _parse(maps, _SHIFT_DECK)
    assert meta["warnings"] == []
    kinds = [type(e).__name__ for e in lat.elements]
    assert kinds == ["Drift", "SuperposedFieldMap", "Drift"]
    sup = lat.elements[1]
    assert [dz for dz, _m in sup.interior_markers] == [200.0, 360.0]
    assert all(m.origin_keyword == "DIAG_SIZE"
               for _dz, m in sup.interior_markers)
    # Fully supported: strict must pass.
    _parse(maps, _SHIFT_DECK, strict=True)


def test_shift_round_trip_byte_idempotent(maps):
    lat, _ = _parse(maps, _SHIFT_DECK)
    out1 = maps / "shift_rt1.dat"
    write_tracewin(lat, str(out1))
    text = out1.read_text()
    assert "SHIFT_IN_FIELD_MAP 200\nDIAG_SIZE 2" in text
    lat2, meta2 = parse_tracewin(str(out1))
    assert meta2["warnings"] == []
    sup2 = next(e for e in lat2.elements
                if isinstance(e, SuperposedFieldMap))
    assert [dz for dz, _m in sup2.interior_markers] == [200.0, 360.0]
    out2 = maps / "shift_rt2.dat"
    write_tracewin(lat2, str(out2))
    assert out1.read_text() == out2.read_text()


def test_shift_before_plain_map_wraps_container(maps):
    lat, meta = _parse(maps,
        "SHIFT_IN_FIELD_MAP 150\n"
        "DIAG_SIZE 2\n"
        "FIELD_MAP 10 300 0 16 2.0 0 0 0 sol\n"
        "DRIFT 100 16 0\nEND\n")
    assert meta["warnings"] == []
    sup = lat.elements[0]
    assert isinstance(sup, SuperposedFieldMap)
    assert len(sup.children) == 1
    assert [dz for dz, _m in sup.interior_markers] == [150.0]


def test_shift_orphans_warn(maps):
    # SHIFT not followed by a diagnostic.
    _, meta = _parse(maps,
        "SHIFT_IN_FIELD_MAP 100\nDRIFT 50 16 0\nEND\n")
    assert any("no following FIELD_MAP" in w or
               "not followed by a diagnostic" in w
               for w in meta["warnings"]), meta["warnings"]
    # SHIFT+diag but no following map: markers restored as ordinary.
    lat, meta = _parse(maps,
        "SHIFT_IN_FIELD_MAP 100\nDIAG_SIZE 2\nDRIFT 50 16 0\nEND\n")
    assert any("no following FIELD_MAP" in w for w in meta["warnings"])
    from linac_gen.elements.marker import Marker
    assert any(isinstance(e, Marker) for e in lat.elements)
    # dz beyond the map span: marker dropped, loudly.
    _, meta = _parse(maps,
        "SHIFT_IN_FIELD_MAP 900\nDIAG_SIZE 2\n"
        "FIELD_MAP 10 300 0 16 2.0 0 0 0 sol\nDRIFT 10 16 0\nEND\n")
    assert any("outside the following map span" in w
               for w in meta["warnings"]), meta["warnings"]
    # dz <= 0: card ignored.
    _, meta = _parse(maps,
        "SHIFT_IN_FIELD_MAP -5\nDIAG_SIZE 2\n"
        "FIELD_MAP 10 300 0 16 2.0 0 0 0 sol\nDRIFT 10 16 0\nEND\n")
    assert any("needs dz > 0" in w for w in meta["warnings"])


def test_shift_bpm_stays_lattice_element(maps):
    # BPMs are resolved ORDINALLY by orbit correction / diagnostic
    # matching — a SHIFT must never consume one (with or without
    # targets); the BPM keeps its lattice position, the shift is
    # dropped loudly, and strict refuses.
    from linac_gen.elements.marker import Marker
    deck = ("SHIFT_IN_FIELD_MAP 100\n"
            "DIAG_POSITION 3 0.5 -0.2\n"
            "FIELD_MAP 10 300 0 16 2.0 0 0 0 sol\n"
            "DRIFT 10 16 0\nEND\n")
    lat, meta = _parse(maps, deck)
    assert any("BPM" in w and "shift is ignored" in w
               for w in meta["warnings"]), meta["warnings"]
    bpms = [e for e in lat.elements
            if isinstance(e, Marker) and getattr(e, "is_bpm", False)]
    assert len(bpms) == 1                      # ordinal table intact
    assert type(lat.elements[1]).__name__ == "FieldMap"  # no container
    with pytest.raises(ValueError, match="BPM"):
        _parse(maps, deck, strict=True)
    # A bare BPM card (no targets) is refused the same way.
    deck2 = ("SHIFT_IN_FIELD_MAP 100\nBPM\n"
             "FIELD_MAP 10 300 0 16 2.0 0 0 0 sol\n"
             "DRIFT 10 16 0\nEND\n")
    lat2, meta2 = _parse(maps, deck2)
    assert any("BPM" in w for w in meta2["warnings"])
    assert sum(1 for e in lat2.elements
               if getattr(e, "is_bpm", False)) == 1


def test_shift_pending_at_eof_restored(maps):
    # Deck WITHOUT END: captured diagnostics must not vanish (count
    # conservation), and strict must refuse rather than return a
    # silently-mutilated lattice.
    from linac_gen.elements.marker import Marker
    deck = "DRIFT 100 16 0\nSHIFT_IN_FIELD_MAP 100\nDIAG_SIZE 2\n"
    lat, meta = _parse(maps, deck)
    assert any(isinstance(e, Marker) for e in lat.elements)
    assert any("no following FIELD_MAP" in w for w in meta["warnings"])
    with pytest.raises(ValueError):
        _parse(maps, deck, strict=True)
    # A dangling SHIFT alone at EOF warns too.
    _, meta2 = _parse(maps, "DRIFT 100 16 0\nSHIFT_IN_FIELD_MAP 100\n")
    assert any("not followed by a diagnostic" in w
               for w in meta2["warnings"]), meta2["warnings"]


def test_shift_marker_keyword_captured(maps):
    # A plain MARKER is a legitimate SHIFT target (named interior row).
    lat, meta = _parse(maps,
        "SHIFT_IN_FIELD_MAP 100\nMARKER\n"
        "FIELD_MAP 10 300 0 16 2.0 0 0 0 sol\nDRIFT 10 16 0\nEND\n")
    assert meta["warnings"] == []
    sup = next(e for e in lat.elements
               if isinstance(e, SuperposedFieldMap))
    assert [dz for dz, _m in sup.interior_markers] == [100.0]


def test_shift_structural_marker_not_captured(maps):
    # LATTICE furniture is NOT a diagnostic: it must never be captured
    # as the shift target (the shift is restored loudly instead).
    lat, meta = _parse(maps,
        "SHIFT_IN_FIELD_MAP 100\nLATTICE 4 0\nDIAG_SIZE 2\n"
        "FIELD_MAP 10 300 0 16 2.0 0 0 0 sol\nDRIFT 10 16 0\nEND\n")
    assert any("not followed by a diagnostic" in w
               for w in meta["warnings"]), meta["warnings"]
    assert not any(isinstance(e, SuperposedFieldMap)
                   for e in lat.elements)


def test_shift_missing_map_file_restores_markers(maps):
    # The map the diagnostics were bound to is dropped (missing field
    # file, permissive): the diagnostics must be restored, not left to
    # bind to a LATER map or vanish at EOF.
    from linac_gen.elements.marker import Marker
    lat, meta = _parse(maps,
        "SHIFT_IN_FIELD_MAP 100\nDIAG_SIZE 2\n"
        "FIELD_MAP 10 300 0 16 2.0 0 0 0 nosuchmap\n"
        "DRIFT 10 16 0\nEND\n")
    assert any(isinstance(e, Marker) and e.length == 0.0
               for e in lat.elements)
    assert any("restored as ordinary markers" in w
               for w in meta["warnings"]), meta["warnings"]


def test_shift_wrapped_plain_map_writes_plain_card(maps):
    # A container created ONLY to host SHIFT diagnostics around a plain
    # FIELD_MAP must not invent a SUPERPOSE_MAP line on write, and the
    # round-trip stays byte-idempotent.
    deck = ("SHIFT_IN_FIELD_MAP 150\nDIAG_SIZE 2\n"
            "FIELD_MAP 10 300 0 16 2.0 0 0 0 sol\n"
            "DRIFT 10 16 0\nEND\n")
    lat, meta = _parse(maps, deck)
    assert meta["warnings"] == []
    out1 = maps / "wrap_rt1.dat"
    write_tracewin(lat, str(out1))
    text = out1.read_text()
    assert "SUPERPOSE_MAP" not in text
    assert "SHIFT_IN_FIELD_MAP 150\nDIAG_SIZE 2" in text
    lat2, meta2 = parse_tracewin(str(out1))
    assert meta2["warnings"] == []
    out2 = maps / "wrap_rt2.dat"
    write_tracewin(lat2, str(out2))
    assert out1.read_text() == out2.read_text()
