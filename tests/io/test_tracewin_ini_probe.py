"""``scripts/tracewin_ini_probe.py`` — the read-only mapper for the
TraceWin ``.ini`` layout: ``find`` locates a typed value, ``diff``
isolates the slot a single TraceWin edit changed, ``dump`` prints the
reader's view."""
from __future__ import annotations

import importlib.util
import struct
from pathlib import Path

import pytest

from linac_gen.io.tracewin_ini import FIELDS

REPO = Path(__file__).resolve().parents[2]
ADS = REPO / "tests" / "io" / "fixtures" / "tracewin_ini" / "ads.ini"
SCRIPT = REPO / "scripts" / "tracewin_ini_probe.py"
OFF = {f.name: f.offset for f in FIELDS}


@pytest.fixture(scope="module")
def probe():
    spec = importlib.util.spec_from_file_location("tracewin_ini_probe", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _variant(tmp_path, name, patches):
    buf = bytearray(ADS.read_bytes())
    for off, b in patches.items():
        buf[off:off + len(b)] = b
    p = tmp_path / name
    p.write_bytes(bytes(buf))
    return p


def test_find_locates_the_energy_slot_and_names_it(probe, capsys):
    assert probe.main(["find", "20e6", str(ADS), "--types", "f64"]) == 0
    out = capsys.readouterr().out
    assert "0x2f44 f64" in out and "known: energy1 (verified" in out


def test_find_scales_and_ints(probe, capsys):
    # 20 MeV typed in MeV → the eV slot is found through the 1e6 scale
    assert probe.main(["find", "20", str(ADS), "--types", "f64"]) == 0
    assert "energy1" in capsys.readouterr().out
    assert probe.main(["find", "50000", str(ADS), "--types", "i32"]) == 0
    out = capsys.readouterr().out
    assert "0x2ed0 i32" in out and "nbr_part1" in out


def test_find_requires_the_value_in_every_file(probe, tmp_path, capsys):
    other = _variant(tmp_path, "e40.ini", {OFF["energy1"]: struct.pack("<d", 40e6)})
    assert probe.main(["find", "20e6", str(ADS), str(other), "--types", "f64"]) == 1
    assert "no f64 slot" in capsys.readouterr().out
    assert probe.main(["find", "100", str(ADS), str(other), "--types", "f64"]) == 0
    assert "freq1" in capsys.readouterr().out


def test_find_rejects_bad_arguments(probe, tmp_path, capsys):
    assert probe.main(["find", "abc", str(ADS)]) == 2
    assert probe.main(["find", "1", str(ADS), "--types", "f16"]) == 2
    assert probe.main(["find", "1", str(tmp_path / "nope.ini")]) == 2


def test_diff_isolates_one_changed_slot(probe, tmp_path, capsys):
    var = _variant(tmp_path, "e40.ini", {OFF["energy1"]: struct.pack("<d", 40e6)})
    assert probe.main(["diff", str(ADS), f"{var}:energy"]) == 0
    out = capsys.readouterr().out
    assert "1 differing run(s)" in out
    assert "0x2f44 f64 20000000.0 -> 40000000.0" in out and "known: energy1" in out


def test_diff_suggests_a_fieldspec_for_an_unknown_slot(probe, tmp_path, capsys):
    var = _variant(tmp_path, "dw.ini", {0x2FE4: struct.pack("<d", 2.5e-3)})
    assert probe.main(["diff", str(ADS), f"{var}:spreadw1"]) == 0
    out = capsys.readouterr().out
    assert 'FieldSpec("spreadw1", 0x2FE4, "<d"' in out


def test_diff_flags_more_than_one_change(probe, tmp_path, capsys):
    var = _variant(tmp_path, "two.ini", {OFF["energy1"]: struct.pack("<d", 40e6),
                                         OFF["current1"]: struct.pack("<d", 0.01)})
    assert probe.main(["diff", str(ADS), str(var), "--max-runs", "1"]) == 3
    out = capsys.readouterr().out
    assert "2 differing run(s)" in out and "more than one setting changed" in out
    assert "energy1" in out and "current1" in out


def test_dump_prints_report_table_and_raw(probe, capsys):
    assert probe.main(["dump", str(ADS), "--raw"]) == 0
    out = capsys.readouterr().out
    assert "Applied to the HELIX beam" in out
    assert "3: 939294308.1  -1  'H-'" in out
    assert "project_name" in out and "'ads'" in out
    assert "i32@0x2ed0 = 50000" in out and "nbr_part1 (verified)" in out
    assert "i32@0x2ee0 = -100" in out and "unknown" in out


def test_dump_refuses_a_foreign_file(probe, tmp_path, capsys):
    bad = tmp_path / "x.ini"
    bad.write_text("nope")
    assert probe.main(["dump", str(bad)]) == 2
