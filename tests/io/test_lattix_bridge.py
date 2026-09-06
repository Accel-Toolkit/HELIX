"""Bmad / SciBmad / PALS import through the optional ``lattix`` translator.

Hermetic tier (no lattix): suffix table, unknown-suffix warning, the
error message when lattix is absent.  External tier (skips without a
lattix that knows the SciBmad format — ``HELIX_LATTIX_ROOT`` or the
sibling ``Translator`` checkout): the three ``examples/*/fodo.*`` decks
must give the SAME HELIX lattice as the MAD-X importer gives for
``examples/madx/fodo.madx`` — the deck they were all written from.
"""
from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

import pytest

from linac_gen.io.formats import (
    IMPORT_SUFFIXES, LATTIX_SUFFIXES, import_format, is_foreign_source, parse_lattice_file,
)

REPO = Path(__file__).resolve().parents[2]
EXAMPLES = REPO / "examples"
FODO = {
    "bmad": EXAMPLES / "bmad" / "fodo.bmad",
    "scibmad": EXAMPLES / "scibmad" / "fodo.jl",
    "pals": EXAMPLES / "pals" / "fodo.pals.yaml",
}


def _lattix_or_skip():
    from linac_gen.io.lattix_bridge import _import_lattix
    try:
        lattix = _import_lattix()
    except ImportError as exc:
        pytest.skip(f"lattix not available: {exc}")
    from lattix.formats.base import FORMATS
    if "scibmad" not in FORMATS:
        pytest.skip("this lattix checkout predates the SciBmad format")
    return lattix


# ---------------------------------------------------------------- hermetic
@pytest.mark.parametrize("name, fmt", [
    ("x.dat", "tracewin"), ("X.DAT", "tracewin"), ("x.madx", "madx"), ("x.seq", "madx"),
    ("x.lat", "mad8"), ("x.flat", "mad8"), ("x.lte", "elegant"),
    ("x.bmad", "lattix"), ("x.jl", "lattix"), ("x.scibmad", "lattix"),
    ("x.pals.yaml", "lattix"), ("x.pals.yml", "lattix"), ("x.pals.json", "lattix"),
    ("x.lattix.json", "lattix"),
    ("x.yaml", None), ("x.json", None), ("x.txt", None), ("x", None),
])
def test_import_format_table(name, fmt):
    assert import_format(name) == fmt
    assert import_format(Path("/some/dir") / name) == fmt
    assert is_foreign_source(name) == (fmt not in (None, "tracewin"))


def test_every_lattix_suffix_is_an_import_suffix():
    assert set(LATTIX_SUFFIXES) <= set(IMPORT_SUFFIXES)
    assert ".dat" in IMPORT_SUFFIXES and ".madx" in IMPORT_SUFFIXES


def test_unknown_suffix_is_parsed_as_tracewin_with_a_warning(tmp_path):
    src = EXAMPLES / "fodo_cell.dat"
    odd = tmp_path / "cell.txt"
    shutil.copy(src, odd)
    ref, _ = parse_lattice_file(str(src))
    lat, meta = parse_lattice_file(str(odd))
    assert len(lat.elements) == len(ref.elements)
    assert any("unknown lattice suffix '.txt'" in w and "parsed as TraceWin" in w
               for w in meta["warnings"]), meta["warnings"]
    _, quiet = parse_lattice_file(str(odd), warn_unknown=False)
    assert not any("unknown lattice suffix" in w for w in quiet["warnings"])
    _, dat_meta = parse_lattice_file(str(src))
    assert not any("unknown lattice suffix" in w for w in dat_meta["warnings"])


def test_missing_lattix_error_names_both_lookups(tmp_path, monkeypatch):
    import linac_gen.io.lattix_bridge as bridge
    monkeypatch.setitem(sys.modules, "lattix", None)          # import lattix -> ImportError
    monkeypatch.setenv("HELIX_LATTIX_ROOT", str(tmp_path / "nowhere"))
    monkeypatch.setattr(bridge, "_SIBLINGS", (tmp_path / "no_sibling",))
    with pytest.raises(ImportError) as info:
        bridge.parse_with_lattix(str(tmp_path / "x.bmad"))
    msg = str(info.value)
    assert "install lattix into this environment" in msg and "HELIX_LATTIX_ROOT" in msg and "no_sibling" in msg


def test_sibling_lookup_covers_the_workspace_layout():
    """HELIX lives in <workspace>/HELIX_unzipped/HELIX_v3 and the translator
    in <workspace>/Translator: the search must look above the parent."""
    import linac_gen.io.lattix_bridge as bridge
    root = bridge._HELIX_ROOT
    assert root.parent / "Translator" in bridge._SIBLINGS
    assert root.parents[1] / "Translator" in bridge._SIBLINGS


def test_missing_lattix_is_also_a_value_error_for_the_cli(tmp_path, monkeypatch):
    """cli/run.py (and twiss/backtrack/export) catch ValueError around load_input:
    the bridge's error must be one, so the CLI prints `error: ...`, not a traceback."""
    import linac_gen.io.lattix_bridge as bridge
    monkeypatch.setitem(sys.modules, "lattix", None)
    monkeypatch.setenv("HELIX_LATTIX_ROOT", str(tmp_path / "nowhere"))
    monkeypatch.setattr(bridge, "_SIBLINGS", (tmp_path / "no_sibling",))
    with pytest.raises(ValueError) as info:
        bridge.parse_with_lattix(str(tmp_path / "x.bmad"))
    assert isinstance(info.value, ImportError) and isinstance(info.value, bridge.LattixUnavailable)


def test_old_lattix_without_the_format_is_named(tmp_path, monkeypatch):
    """A lattix checkout that predates a format must say so (root + format), not
    fall into lattix's own 'cannot guess the lattice format; pass fmt='."""
    import types
    import linac_gen.io.lattix_bridge as bridge
    fake = types.ModuleType("lattix")
    fake.__file__ = str(tmp_path / "old" / "lattix" / "__init__.py")
    fake.__version__ = "0.1.0"
    formats = types.ModuleType("lattix.formats")
    base = types.ModuleType("lattix.formats.base")
    base.FORMATS = {"bmad": object()}
    base.guess_format = lambda p: (_ for _ in ()).throw(AssertionError("guess_format must not run"))
    helix = types.ModuleType("lattix.formats.helix")
    helix.to_helix = None
    ir = types.ModuleType("lattix.ir")
    ref = types.ModuleType("lattix.ir.reference")
    for name, mod in (("lattix", fake), ("lattix.formats", formats), ("lattix.formats.base", base),
                      ("lattix.formats.helix", helix), ("lattix.ir", ir), ("lattix.ir.reference", ref)):
        monkeypatch.setitem(sys.modules, name, mod)
    with pytest.raises(bridge.LattixUnavailable) as info:
        bridge.parse_with_lattix(str(tmp_path / "cell.jl"))
    msg = str(info.value)
    assert "'scibmad'" in msg and str(tmp_path / "old") in msg and "0.1.0" in msg


def test_species_normalisation():
    from linac_gen.io.lattix_bridge import _norm_species
    assert _norm_species("Proton") == "proton" and _norm_species("#1H-") == "h-"
    assert _norm_species("H-") == "h-" and _norm_species("D") == "deuteron"


# ---------------------------------------------------------------- with lattix
def _signature(lat):
    """(type, length_mm, strength) per element, leading reference markers dropped."""
    out = []
    for e in lat.elements:
        t = type(e).__name__
        if t == "Marker" and not out and e.length == 0:
            continue                              # SciBmad's lat_begin reference marker
        strength = (getattr(e, "gradient", None) if t == "Quadrupole"
                    else getattr(e, "angle", None) if t == "Dipole"
                    else getattr(e, "pole_rotation", None) if t == "Edge" else None)
        out.append((t, float(e.length), None if strength is None else float(strength)))
    return out


@pytest.mark.parametrize("fmt", sorted(FODO))
def test_examples_match_the_madx_import(fmt):
    """fodo.{bmad,jl,pals.yaml} were written from fodo.madx: element for
    element the same HELIX lattice (types, lengths, gradients, angles)."""
    _lattix_or_skip()
    from linac_gen.io.madx_parser import parse_madx
    ref, _ = parse_madx(str(EXAMPLES / "madx" / "fodo.madx"))
    lat, meta = parse_lattice_file(str(FODO[fmt]))
    assert meta["format"] == fmt
    a, b = _signature(ref), _signature(lat)
    assert [x[0] for x in a] == [x[0] for x in b]
    for (ta, la, sa), (tb, lb, sb) in zip(a, b):
        assert la == pytest.approx(lb, rel=1e-9, abs=1e-9), ta
        if sa is not None:
            assert sa == pytest.approx(sb, rel=1e-9), ta
    assert sum(x[1] for x in b) == pytest.approx(6600.0, rel=1e-12)
    ref_particle = meta["reference"]
    assert ref_particle.species.name == "proton"
    assert ref_particle.w_kin == pytest.approx(800.0, rel=1e-6)
    # every quadrupole gradient is the MAD-X one to the last digit
    gq = [e.gradient for e in lat.elements if type(e).__name__ == "Quadrupole"]
    assert gq == pytest.approx([2.928617935551388, -2.928617935551388], rel=1e-12)


def test_meta_contract_and_parse_warnings(capsys):
    _lattix_or_skip()
    lat, meta = parse_lattice_file(str(FODO["scibmad"]))
    assert set(meta) >= {"title", "warnings", "reference", "fidelity", "format"}
    assert isinstance(meta["warnings"], list) and all(isinstance(w, str) for w in meta["warnings"])
    assert any("set the Beam tab to match" in w for w in meta["warnings"])
    assert set(meta["fidelity"]) == {"read", "to_helix"}
    assert "entries" in meta["fidelity"]["read"]
    # the CLI loader carries the ledger on the lattice, as for every other format
    from linac_gen.cli.common import load_lattice
    lat2 = load_lattice(str(FODO["scibmad"]))
    assert list(lat2.parse_warnings) == meta["warnings"]
    err = capsys.readouterr().err
    assert "parse warning:" in err


def test_positron_deck_needs_an_explicit_species(tmp_path):
    """Bmad's default particle is the positron, which HELIX cannot track:
    the error must say what the deck's reference is and how to override."""
    _lattix_or_skip()
    from linac_gen.io.lattix_bridge import parse_with_lattix
    deck = tmp_path / "ring.bmad"
    deck.write_text("parameter[p0c] = 1e9\nq1: quadrupole, l = 0.2, k1 = 1.5\n"
                    "d1: drift, l = 0.3\nring: line = (q1, d1)\nuse, ring\n")
    with pytest.raises(ValueError) as info:
        parse_with_lattix(str(deck))
    msg = str(info.value)
    assert "positron" in msg and "species=" in msg
    lat, meta = parse_with_lattix(str(deck), species="proton")
    assert [type(e).__name__ for e in lat.elements] == ["Quadrupole", "Drift"]
    assert any("deck reference is positron" in w for w in meta["warnings"])
    # k1 = 1.5 at p0c = 1 GeV: G = k1 * Brho with Brho = p/(e c)
    brho = 1e9 / 299792458.0
    assert lat.elements[0].gradient == pytest.approx(1.5 * brho, rel=1e-9)
    assert meta["reference"].species.name == "proton"


def test_hminus_flips_the_sign_of_every_normalised_strength(tmp_path):
    _lattix_or_skip()
    from linac_gen.io.lattix_bridge import parse_with_lattix
    deck = tmp_path / "q.jl"
    deck.write_text('using Beamlines\n@elements begin\n'
                    '  b0 = Marker(species_ref = Species("#1H-"), E_ref = 941391806.25)\n'
                    '  q1 = Quadrupole(L = 0.3, Kn1 = -2.0)\nend\nlattice = Beamline([b0, q1])\n')
    lat, meta = parse_with_lattix(str(deck))
    assert meta["reference"].species.name == "H-"
    q = [e for e in lat.elements if type(e).__name__ == "Quadrupole"][0]
    # Kn1 = Bn1 / (p/q) with q < 0: a negative Kn1 is a positive field gradient
    assert q.gradient > 0
    assert q.gradient == pytest.approx(2.0 * abs(meta["reference"].brho), rel=1e-6)


def test_importing_lattix_does_not_reload_linac_gen():
    """lattix reaches HELIX through its own adapter: inside HELIX that must be
    the already-imported package, never a second copy from a path insert."""
    _lattix_or_skip()
    import linac_gen
    before = sys.modules["linac_gen"]
    path_before = list(sys.path)
    parse_lattice_file(str(FODO["bmad"]))
    assert sys.modules["linac_gen"] is before is linac_gen
    added = [p for p in sys.path if p not in path_before]
    assert all("Translator" in p or os.environ.get("HELIX_LATTIX_ROOT", "\0") in p for p in added), added
