# tests/io/test_parser_downgrades.py
"""Physics-downgrade honesty: every card whose meaning HELIX cannot
honour must WARN in permissive mode (metadata["warnings"], the channel
the GUI surfaces and the CLI echoes) and RAISE in strict mode.

Both regimes are exercised for every card class (house rule 4) — the
silent variants of these behaviors are exactly what the PRAB review
flagged as publication blockers.
"""
import pytest

from linac_gen.io.tracewin_parser import parse_tracewin


def _parse(tmp_path, text, **kw):
    p = tmp_path / "deck.dat"
    p.write_text(text)
    return parse_tracewin(str(p), **kw)


# ── superposition / beam-mutating cards ──────────────────────────────────
@pytest.mark.parametrize("card,frag", [
    # SUPERPOSE_MAP and SHIFT_IN_FIELD_MAP left the downgrade class in
    # the superposition round (tests/io/test_superpose_parsing.py); a
    # DANGLING card still warns via the cluster/shift machinery.
    ("SUPERPOSE_MAP_OUT 100 0 0 0 0 0", "not modelled"),
    ("SHIFT_IN_FIELD_MAP 5", "not followed by a diagnostic"),
    ("READ_DST beam.dst", "ignored"),
    ("BEAM_ROT 0 0 30", "ignored"),
])
def test_downgrade_cards_warn_permissive_raise_strict(tmp_path, card, frag):
    deck = f"DRIFT 100 15 0\n{card}\nDRIFT 100 15 0\nEND\n"
    lat, meta = _parse(tmp_path, deck)
    assert any(frag in w for w in meta["warnings"]), meta["warnings"]
    with pytest.raises(ValueError, match=card.split()[0]):
        _parse(tmp_path, deck, strict=True)


# ── dynamic / coupled error groups ───────────────────────────────────────
# Layout: N r dx dy φx φy φz dG dG3 dG4 dG5 dG6 — r=2 (gaussian) is the
# fully supported draw mode; φx/φy stay 0 (pitch/yaw are inert).
_QUAD_ERR_ARGS = "1 2 0.1 0.1 0 0 0 0.001 0 0 0 0"


def test_dynamic_error_card_declared_static(tmp_path):
    deck = (f"DRIFT 100 15 0\nERROR_QUAD_NCPL_DYN {_QUAD_ERR_ARGS}\n"
            "QUAD 100 5 15 0 0 0 0 0 0\nEND\n")
    lat, meta = _parse(tmp_path, deck)
    assert any("treated as STATIC" in w for w in meta["warnings"]), \
        meta["warnings"]
    # The card still takes effect (static downgrade, not a drop):
    assert getattr(lat, "errors", []), "error def was lost, not downgraded"
    with pytest.raises(ValueError, match="STATIC"):
        _parse(tmp_path, deck, strict=True)


def test_coupled_error_card_declared_independent(tmp_path):
    deck = (f"DRIFT 100 15 0\nERROR_QUAD_CPL_STAT {_QUAD_ERR_ARGS}\n"
            "QUAD 100 5 15 0 0 0 0 0 0\nEND\n")
    lat, meta = _parse(tmp_path, deck)
    assert any("INDEPENDENTLY" in w for w in meta["warnings"]), \
        meta["warnings"]
    assert getattr(lat, "errors", []), "error def was lost, not downgraded"
    with pytest.raises(ValueError, match="coupled"):
        _parse(tmp_path, deck, strict=True)


def test_ncpl_stat_card_stays_clean(tmp_path):
    """The fully-supported card class must NOT gain warnings."""
    deck = (f"DRIFT 100 15 0\nERROR_QUAD_NCPL_STAT {_QUAD_ERR_ARGS}\n"
            "QUAD 100 5 15 0 0 0 0 0 0\nEND\n")
    lat, meta = _parse(tmp_path, deck)
    assert meta["warnings"] == [], meta["warnings"]
    _parse(tmp_path, deck, strict=True)            # and strict still parses


def test_unimplemented_error_card_warns(tmp_path):
    deck = "DRIFT 100 15 0\nERROR_STAT_FILE errors.txt\nEND\n"
    lat, meta = _parse(tmp_path, deck)
    assert any("IGNORED (no errors" in w for w in meta["warnings"]), \
        meta["warnings"]
    with pytest.raises(ValueError, match="ERROR_STAT_FILE"):
        _parse(tmp_path, deck, strict=True)


def test_error_directive_caveats_reach_metadata(tmp_path):
    """tracewin_error_parsing's warnings.warn caveats (historically
    stderr-only, invisible to the GUI) must also land in metadata."""
    deck = ("DRIFT 100 15 0\n"
            "ERROR_QUAD_NCPL_STAT 1 0.1 0.1 0 0 0 0 0.001 0 0 0\n"  # 11 args
            "QUAD 100 5 15 0 0 0 0 0 0\nEND\n")
    _, meta = _parse(tmp_path, deck)
    assert any("malformed ERROR_QUAD" in w for w in meta["warnings"]), \
        meta["warnings"]


# ── missing / rejected field-map files ───────────────────────────────────
_MISSING_FM_DECK = ("FIELD_MAP_PATH .\nDRIFT 100 15 0\n"
                    "FIELD_MAP 100 200 0 15 0 1.0 0 0 nofile\n"
                    "DRIFT 100 15 0\nEND\n")


def test_missing_field_file_permissive_drops_with_loud_warning(tmp_path):
    lat, meta = _parse(tmp_path, _MISSING_FM_DECK)
    assert len(lat.elements) == 2                       # cavity dropped
    assert any("ELEMENT DROPPED" in w for w in meta["warnings"]), \
        meta["warnings"]


def test_missing_field_file_strict_raises(tmp_path):
    """A strict parse must never return a lattice missing a cavity —
    empirically, it used to (the element vanished with a metadata note)."""
    with pytest.raises(ValueError, match="FIELD_MAP file missing"):
        _parse(tmp_path, _MISSING_FM_DECK, strict=True)


def test_negative_geom_flag_declared(tmp_path):
    deck = ("FIELD_MAP_PATH .\n"
            "FIELD_MAP -100 200 0 15 0 1.0 0 0 nofile\nEND\n")
    _, meta = _parse(tmp_path, deck)
    assert any("second-order" in w for w in meta["warnings"]), \
        meta["warnings"]
    with pytest.raises(ValueError, match="second-order"):
        _parse(tmp_path, deck, strict=True)


def test_strict_negative_geom_message_not_double_wrapped(tmp_path):
    """Adversarial-review finding: the geom refusal used to be raised
    inside the FIELD_MAP read-try, whose except re-wrapped it as
    'Line N: FIELD_MAP rejected: Line N: …' — a doubled prefix and a
    read-failure label on a policy refusal."""
    deck = ("FIELD_MAP_PATH .\n"
            "FIELD_MAP -100 200 0 15 0 1.0 0 0 nofile\nEND\n")
    with pytest.raises(ValueError) as ei:
        _parse(tmp_path, deck, strict=True)
    msg = str(ei.value)
    assert "rejected" not in msg, msg
    assert msg.count("Line 2:") == 1, msg


def test_writer_warns_when_error_study_would_be_lost(tmp_path):
    """The writer serialises no ERROR_* cards — saving a deck that
    carries an error study silently deleted it (and the round-tripped
    deck no longer warned, because the cards were gone).  Now loud."""
    deck = (f"DRIFT 100 15 0\nERROR_QUAD_NCPL_STAT {_QUAD_ERR_ARGS}\n"
            "QUAD 100 5 15 0 0 0 0 0 0\nEND\n")
    lat, _ = _parse(tmp_path, deck)
    from linac_gen.io.tracewin_writer import write_tracewin
    with pytest.warns(UserWarning, match="does NOT serialise"):
        write_tracewin(lat, str(tmp_path / "out.dat"))


# ── grouped low-stakes hints ─────────────────────────────────────────────
def test_fit_hint_cards_grouped_single_line_not_strict_fatal(tmp_path):
    deck = ("RFQ_GEOM 1 2 3\nRFQ_GAP_RMS_FFS 0.5\nMATCH_FAM_GRAD 1\n"
            "DRIFT 100 15 0\nEND\n")
    _, meta = _parse(tmp_path, deck)
    hints = [w for w in meta["warnings"] if "no-op cards" in w]
    assert len(hints) == 1, meta["warnings"]
    assert "RFQ_GEOM" in hints[0] and "MATCH_FAM_GRAD" in hints[0]
    # hints are informative, NOT physics downgrades — strict still parses:
    _parse(tmp_path, deck, strict=True)


# ── CLI loader visibility ────────────────────────────────────────────────
def test_cli_load_lattice_echoes_warnings_to_stderr(tmp_path, capsys):
    """Every CLI path used to DISCARD parse metadata outright."""
    p = tmp_path / "deck.dat"
    p.write_text("DRIFT 100 15 0\nSUPERPOSE_MAP 0\nEND\n")
    from linac_gen.cli.common import load_lattice
    load_lattice(str(p))
    err = capsys.readouterr().err
    assert "parse warning" in err and "SUPERPOSE_MAP" in err
