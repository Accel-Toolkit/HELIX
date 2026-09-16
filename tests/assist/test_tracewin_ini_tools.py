"""Assistant tools for the TraceWin ``.ini`` options file: the read-tier
``inspect_tracewin_ini`` (nothing touches the session) and the
``tracewin_ini`` argument of ``load_lattice`` (the converted beam becomes
the session beam; a project refuses it)."""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from linac_gen.assist.tools import TOOLS, WorkContext
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


def _ctx(tmp_path):
    return WorkContext(calc_dir=str(tmp_path))


def test_tools_are_registered_with_the_right_tiers():
    assert TOOLS["inspect_tracewin_ini"].tier == "read"
    assert TOOLS["load_lattice"].tier == "mutate"
    assert "tracewin_ini" in TOOLS["load_lattice"].schema["properties"]
    assert TOOLS["inspect_tracewin_ini"].schema["required"] == ["path"]


def test_inspect_decodes_without_touching_the_session(tmp_path):
    ctx = _ctx(tmp_path)
    res = TOOLS["inspect_tracewin_ini"].fn(ctx, path=str(ADS))
    assert res["status"] == "ok", res
    d = res["data"]
    assert d["layout"].startswith("TraceWin 2019") and d["project_name"] == "ads"
    assert d["particle"]["name"] == "Proton" and d["particle"]["helix_species"] == "proton"
    assert d["beam"]["energy"] == 20.0 and d["beam"]["species"] == "proton"
    assert abs(d["beam"]["alpha_z"] + 0.17661) < 1e-12
    assert d["raw_fields"]["alpz"] == 0.17661 and d["raw_fields"]["energy"] == 2e7
    assert d["identified_not_applied"]["picnic_r_mesh"] == 20
    assert "input_dist_type" in d["not_decoded"]
    assert "Applied to the HELIX beam" in d["report"]
    assert d["not_convertible"] is None and res["warnings"] == []
    assert ctx.lattice is None and ctx.beam_config in (None, BeamConfig())
    json.dumps(res, allow_nan=False)


def test_inspect_beam_2_reports_not_convertible(tmp_path):
    res = TOOLS["inspect_tracewin_ini"].fn(_ctx(tmp_path), path=str(ADS), beam=2)
    assert res["status"] == "ok"
    assert res["data"]["beam"] is None and "not populated" in res["data"]["not_convertible"]
    assert TOOLS["inspect_tracewin_ini"].fn(_ctx(tmp_path), path=str(ADS), beam=3)["status"] == "refused"


def test_inspect_refusals_and_errors(tmp_path):
    ctx = _ctx(tmp_path)
    assert TOOLS["inspect_tracewin_ini"].fn(ctx, path="https://x/ads.ini")["status"] == "refused"
    assert TOOLS["inspect_tracewin_ini"].fn(ctx, path=str(tmp_path / "nope.ini"))["status"] == "error"
    txt = tmp_path / "text.ini"
    txt.write_text("[beam]\n")
    res = TOOLS["inspect_tracewin_ini"].fn(ctx, path=str(txt))
    assert res["status"] == "refused" and "not a TraceWin options file" in res["data"]["message"]


def test_inspect_species_override_warns(tmp_path):
    res = TOOLS["inspect_tracewin_ini"].fn(_ctx(tmp_path), path=str(ADS), species="H-")
    assert res["status"] == "ok" and res["data"]["beam"]["species"] == "H-"
    assert any("'Proton'" in w for w in res["warnings"])


def test_load_lattice_with_the_sibling_ini(deck, tmp_path):
    ctx = _ctx(tmp_path)
    res = TOOLS["load_lattice"].fn(ctx, path=str(deck), tracewin_ini="auto")
    assert res["status"] == "ok", res
    d = res["data"]
    assert d["beam_loaded"] and d["beam_source"] == "tracewin_ini"
    assert d["tracewin_ini"] == "deck.ini" and d["beam"]["energy"] == 20.0
    assert res["provenance"]["tracewin_ini"].endswith("deck.ini")
    assert ctx.lattice is not None and ctx.beam_config.energy == 20.0
    assert ctx.beam_config.species == "proton" and abs(ctx.beam_config.alpha_z + 0.17661) < 1e-12
    assert any("FREQ" in w and "352.21" in w for w in res["warnings"])   # deck FREQ ≠ 100 MHz


def test_load_lattice_with_an_explicit_ini_path(deck, tmp_path):
    other = tmp_path / "other.ini"
    shutil.move(str(tmp_path / "deck.ini"), other)
    ctx = _ctx(tmp_path)
    res = TOOLS["load_lattice"].fn(ctx, path=str(deck), tracewin_ini=str(other))
    assert res["status"] == "ok" and res["data"]["tracewin_ini"] == "other.ini"
    assert ctx.beam_config.energy == 20.0


def test_load_lattice_without_the_flag_leaves_the_beam(deck, tmp_path):
    ctx = _ctx(tmp_path)
    ctx.set_beam_config(BeamConfig(energy=9.0))
    res = TOOLS["load_lattice"].fn(ctx, path=str(deck))
    assert res["status"] == "ok" and res["data"]["beam_source"] is None
    assert ctx.beam_config.energy == 9.0


def test_load_lattice_missing_sibling_is_refused(tmp_path):
    shutil.copy(FODO, tmp_path / "lonely.dat")
    ctx = _ctx(tmp_path)
    res = TOOLS["load_lattice"].fn(ctx, path=str(tmp_path / "lonely.dat"), tracewin_ini="auto")
    assert res["status"] == "refused" and "lonely.ini" in res["data"]["message"]
    assert ctx.lattice is None                       # nothing half-loaded


def test_load_lattice_project_refuses_the_flag(deck, tmp_path):
    proj = write_project(tmp_path / "deck.lgproj", lattice_path=deck, beam=BeamConfig(energy=7.5))
    ctx = _ctx(tmp_path)
    res = TOOLS["load_lattice"].fn(ctx, path=str(proj), tracewin_ini="auto")
    assert res["status"] == "refused" and "always wins" in res["data"]["message"]
    res2 = TOOLS["load_lattice"].fn(ctx, path=str(proj))
    assert res2["status"] == "ok" and res2["data"]["beam_source"] == "project"
    assert ctx.beam_config.energy == 7.5


def test_load_lattice_url_ini_is_refused(deck, tmp_path):
    res = TOOLS["load_lattice"].fn(_ctx(tmp_path), path=str(deck), tracewin_ini="https://x/deck.ini")
    assert res["status"] == "refused"


def test_inspect_is_strict_json_even_with_non_finite_slots(tmp_path):
    import struct
    from linac_gen.io.tracewin_ini import FIELDS
    off = {f.name: f.offset for f in FIELDS}
    buf = bytearray(ADS.read_bytes())
    buf[off["energy1"]:off["energy1"] + 8] = struct.pack("<d", float("nan"))
    buf[off["betz1"]:off["betz1"] + 8] = struct.pack("<d", float("inf"))
    p = tmp_path / "nan.ini"
    p.write_bytes(bytes(buf))
    res = TOOLS["inspect_tracewin_ini"].fn(_ctx(tmp_path), path=str(p))
    assert res["status"] == "ok" and res["data"]["beam"] is None
    assert "not finite" in res["data"]["not_convertible"]
    assert res["data"]["raw_fields"]["energy"] is None
    json.dumps(res, allow_nan=False)


def test_inspect_parameter_hygiene(tmp_path):
    ctx = _ctx(tmp_path)
    for bad in ("Proton", "h-", ""):
        r = TOOLS["inspect_tracewin_ini"].fn(ctx, path=str(ADS), species=bad)
        assert r["status"] == "refused" and "species" in r["data"]["message"]
    for bad in ("x", 0, 3, True):
        r = TOOLS["inspect_tracewin_ini"].fn(ctx, path=str(ADS), beam=bad)
        assert r["status"] == "refused" and "beam" in r["data"]["message"]
    assert TOOLS["inspect_tracewin_ini"].fn(ctx, path=str(ADS), beam=None)["status"] == "ok"


def test_load_lattice_argument_hygiene(deck, tmp_path):
    ctx = _ctx(tmp_path)
    r = TOOLS["load_lattice"].fn(ctx, path=str(deck), tracewin_ini=1)
    assert r["status"] == "refused" and "tracewin_ini" in r["data"]["message"]
    proj = write_project(tmp_path / "deck.lgproj.json", lattice_path=deck, beam=BeamConfig(energy=7.5))
    r = TOOLS["load_lattice"].fn(ctx, path=str(proj), tracewin_ini="auto")
    assert r["status"] == "refused" and "always wins" in r["data"]["message"]
    r2 = TOOLS["load_lattice"].fn(ctx, path=str(proj))
    assert r2["status"] == "ok" and ctx.beam_config.energy == 7.5
