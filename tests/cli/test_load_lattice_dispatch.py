"""``load_lattice`` (the CLI loader) dispatches on the file suffix through
``linac_gen.io.formats`` — hermetic: the lattix branch is stubbed, so the
test runs without lattix installed."""
from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from linac_gen.cli.common import load_lattice
from linac_gen.core.lattice import Lattice
from linac_gen.elements.drift import Drift

EXAMPLES = Path(__file__).resolve().parents[2] / "examples"


@pytest.mark.parametrize("suffix", [".bmad", ".jl", ".scibmad", ".pals.yaml", ".pals.json",
                                    ".lattix.json"])
def test_lattix_suffixes_go_to_the_bridge(tmp_path, monkeypatch, capsys, suffix):
    import linac_gen.io.lattix_bridge as bridge
    calls = []

    def fake(path, **kw):
        calls.append(str(path))
        lat = Lattice()
        lat.add(Drift(name="d", length=123.0))
        return lat, {"warnings": ["LOSSY:SOMETHING x: approximated"], "format": "stub"}

    monkeypatch.setattr(bridge, "parse_with_lattix", fake)
    deck = tmp_path / ("cell" + suffix)
    deck.write_text("not parsed by the stub\n")
    lat = load_lattice(str(deck))
    assert calls == [str(deck)]
    assert len(lat.elements) == 1 and lat.elements[0].length == 123.0
    assert lat.parse_warnings == ["LOSSY:SOMETHING x: approximated"]
    assert "parse warning: LOSSY:SOMETHING x: approximated" in capsys.readouterr().err


def test_native_suffixes_do_not_touch_the_bridge(tmp_path, monkeypatch):
    import linac_gen.io.lattix_bridge as bridge
    monkeypatch.setattr(bridge, "parse_with_lattix",
                        lambda *a, **k: (_ for _ in ()).throw(AssertionError("bridge called")))
    lat = load_lattice(str(EXAMPLES / "fodo_cell.dat"))
    assert len(lat.elements) > 0
    lat2 = load_lattice(str(EXAMPLES / "madx" / "fodo.madx"))
    assert len(lat2.elements) == 13


def test_unknown_suffix_warns_on_stderr(tmp_path, capsys):
    odd = tmp_path / "cell.txt"
    shutil.copy(EXAMPLES / "fodo_cell.dat", odd)
    lat = load_lattice(str(odd))
    assert len(lat.elements) > 0
    err = capsys.readouterr().err
    assert "parse warning: unknown lattice suffix '.txt'" in err
    assert any("unknown lattice suffix" in w for w in lat.parse_warnings)
    # a .dat never carries that warning
    lat = load_lattice(str(EXAMPLES / "fodo_cell.dat"))
    assert not any("unknown lattice suffix" in w for w in lat.parse_warnings)


def test_scan_workers_use_the_shared_dispatcher(tmp_path, monkeypatch):
    """The parallel scan pool re-parses the lattice in each worker.  It used to
    carry a private copy of the suffix dispatch that sent .jl/.bmad decks to the
    TraceWin parser (a scan on examples/scibmad/fodo.jl then failed with
    "no element named 'qf'"); it must go through linac_gen.io.formats."""
    import linac_gen.io.lattix_bridge as bridge
    from linac_gen.parallel.scan_pool import _parse_lattice_for_scan
    seen = []

    def fake(path, **kw):
        seen.append(Path(path).name)
        lat = Lattice()
        lat.add(Drift(name="d", length=5.0))
        return lat, {"warnings": ["would be printed by the parent"]}

    monkeypatch.setattr(bridge, "parse_with_lattix", fake)
    deck = tmp_path / "cell.jl"
    deck.write_text("stub\n")
    lat = _parse_lattice_for_scan(str(deck))
    assert seen == ["cell.jl"] and [e.name for e in lat.elements] == ["d"]
    # native decks still parse natively in the worker, and an unknown suffix
    # is parsed as TraceWin without a second warning (the parent already warned)
    lat_dat = _parse_lattice_for_scan(str(EXAMPLES / "fodo_cell.dat"))
    assert len(lat_dat.elements) > 0 and seen == ["cell.jl"]
    lat_madx = _parse_lattice_for_scan(str(EXAMPLES / "madx" / "fodo.madx"))
    assert len(lat_madx.elements) == 13
