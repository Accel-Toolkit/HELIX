"""The assistant's physicist-level lattice knowledge (2026-07-29):
describe_lattice, semantic list filters, the read_file window, and the
prompt hardware rollup — born from a live failure: 'how many cavities?'
was unanswerable and the assistant claimed it had no file access."""
from __future__ import annotations

import os

import pytest

from linac_gen.assist.prompts import build_system_prompt
from linac_gen.assist.tools import TOOLS, WorkContext

_DECK = os.path.join(os.path.dirname(__file__), "..", "..",
                     "examples", "MEBT_To_Foil", "mebt_to_foil.dat")


@pytest.fixture(scope="module")
def deck_ctx():
    if not os.path.isfile(_DECK):
        pytest.skip("MEBT_To_Foil example deck not present")
    from linac_gen.io.tracewin_parser import parse_tracewin
    ctx = WorkContext(calc_dir="runs")
    ctx.lattice, _ = parse_tracewin(_DECK)
    ctx.lattice_path = os.path.abspath(_DECK)
    # The deck's cavities are FieldMaps whose field FILES live in the
    # machine-local Fields/ directory (ANL/CEA data, never committed).
    # A checkout without them parses fine but silently DROPS every
    # cavity — these knowledge tests would then assert against a
    # gutted lattice (CI failed exactly this way).  Skip like every
    # other local-data-dependent test.
    if not any(type(e).__name__ in ("FieldMap", "FieldMap3D",
                                    "SuperposedFieldMap")
               for e in ctx.lattice.elements):
        pytest.skip("field-map data (Fields/) not present on this "
                    "machine — cavity content unavailable")
    return ctx


def test_describe_lattice_answers_how_many_cavities(deck_ctx):
    r = TOOLS["describe_lattice"].fn(deck_ctx)
    assert r["status"] == "ok"
    d = r["data"]
    assert d["cavities"] == 135
    assert d["cavities_powered"] == 123
    assert d["cavities_parked_zero_amplitude"] == 12
    assert d["solenoids"] == 37
    assert d["rf_sections_mhz"] == [162.5, 325.0, 650.0]


def test_describe_lattice_needs_a_lattice():
    ctx = WorkContext(calc_dir="runs")
    assert TOOLS["describe_lattice"].fn(ctx)["status"] == "error"


def test_list_elements_semantic_filter_and_total(deck_ctx):
    r = TOOLS["list_lattice_elements"].fn(deck_ctx, filter="cavity",
                                          limit=5)
    d = r["data"]
    assert d["total_matched"] == 135             # counted past the limit
    assert len(d["elements"]) == 5
    assert d["truncated"] is True
    row = d["elements"][0]
    assert row["kind"] == "cavity"
    assert "RF" in row["field_type"]
    assert row["geom"] == 7700
    assert "ke" in row["params"]
    # plural / synonym forms
    for syn in ("cavities", "rf"):
        assert TOOLS["list_lattice_elements"].fn(
            deck_ctx, filter=syn, limit=1)["data"]["total_matched"] == 135
    # solenoids via kind
    assert TOOLS["list_lattice_elements"].fn(
        deck_ctx, filter="solenoid", limit=1)["data"]["total_matched"] == 37


def test_list_elements_matches_field_file_label(deck_ctx):
    """Deck labels survive only in the field-map FILE name — the filter
    must find them (element names are synthesized FMAP_001…)."""
    r = TOOLS["list_lattice_elements"].fn(deck_ctx, filter="QWR",
                                          limit=3)
    assert r["data"]["total_matched"] > 0
    assert "QWR" in r["data"]["elements"][0]["field_file"]


def test_list_elements_exact_limit_not_marked_truncated(deck_ctx):
    """Old bug: a result of exactly `limit` rows was always flagged
    truncated even when nothing was cut."""
    n_foil = TOOLS["list_lattice_elements"].fn(
        deck_ctx, filter="foil", limit=1)["data"]
    assert n_foil["total_matched"] == 1
    assert n_foil["truncated"] is False


def test_read_file_defaults_to_loaded_deck(deck_ctx):
    r = TOOLS["read_file"].fn(deck_ctx, start_line=1, max_lines=5)
    d = r["data"]
    assert d["path"].endswith("mebt_to_foil.dat")
    assert len(d["lines"]) == 5
    assert d["total_lines"] > 2800
    assert d["truncated"] is True


def test_read_file_grep_finds_raw_cards(deck_ctx):
    r = TOOLS["read_file"].fn(deck_ctx, pattern=r"FIELD_MAP\s+7700")
    d = r["data"]
    assert d["matches_total"] >= 135             # the raw cavity cards
    assert "FIELD_MAP 7700" in d["lines"][0]


def test_read_file_arbitrary_local_file(tmp_path):
    ctx = WorkContext(calc_dir="runs")
    p = tmp_path / "notes.txt"
    p.write_text("alpha\nbeta\ngamma\n")
    r = TOOLS["read_file"].fn(ctx, path=str(p))
    assert [l.split("| ", 1)[1] for l in r["data"]["lines"]] == [
        "alpha", "beta", "gamma"]


def test_read_file_refuses_urls_and_summarizes_binary(tmp_path):
    ctx = WorkContext(calc_dir="runs")
    assert TOOLS["read_file"].fn(
        ctx, path="https://x/y.dat")["status"] == "refused"
    b = tmp_path / "blob.bin"
    b.write_bytes(b"\x00\x01\x02helix")
    r = TOOLS["read_file"].fn(ctx, path=str(b))
    assert r["status"] == "ok"
    assert r["data"]["binary"] is True
    assert "lines" not in r["data"]


def test_read_file_errors_without_path_or_lattice():
    ctx = WorkContext(calc_dir="runs")
    assert TOOLS["read_file"].fn(ctx)["status"] == "error"


def test_prompt_carries_hardware_rollup(deck_ctx):
    sp = build_system_prompt(deck_ctx)
    assert "135 RF cavities (12 parked at zero amplitude)" in sp
    assert "37 solenoids" in sp
    assert "162.5/325/650 MHz" in sp
    # the security line stays honest about the new scope
    assert "NO file access" not in sp
    assert "read-only file access" in sp or "read_file" in sp
