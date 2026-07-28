"""Portable persisted paths (2026-07-11).

Covers the shared helper (``linac_gen.io.portable_paths``), the TraceWin
writer's field-map relativization, and the CLI project loader's
project-dir-first resolution.  The GUI save/load side is covered in
``tests/gui/test_project_portable_paths.py``.
"""
from __future__ import annotations

import os as _os_guard

import pytest as _pytest_guard

if not _os_guard.path.isdir("Fields"):
    _pytest_guard.skip("Fields/ field-map data is not distributed with "
                       "the repository (third-party ANL/CEA data) — see "
                       "examples/FIELD_MAPS.md", allow_module_level=True)

import os
from pathlib import Path

import numpy as np
import pytest

from linac_gen.io.portable_paths import best_relpath, resolve_candidates

REPO = Path(__file__).resolve().parents[2]


# ---------------------------------------------------------------------------
# best_relpath
# ---------------------------------------------------------------------------
def test_same_dir(tmp_path):
    rel, ok = best_relpath(tmp_path / "a.dat", tmp_path)
    assert ok and rel == "a.dat"


def test_subdir_posix_separators(tmp_path):
    rel, ok = best_relpath(tmp_path / "maps" / "hwr" / "f.edz", tmp_path)
    assert ok and rel == "maps/hwr/f.edz"          # POSIX even on Windows


def test_escaping_parent_allowed(tmp_path):
    anchor = tmp_path / "examples" / "pipii"
    target = tmp_path / "Fields" / "HWRDonut"
    rel, ok = best_relpath(target, anchor)
    assert ok and rel == "../../Fields/HWRDonut"


def test_root_only_common_ancestor_stays_absolute():
    # /var/... vs /Users/... (or any two top-level trees): only common
    # path is the root → not meaningfully relocatable.
    t = os.path.join(os.sep, "some_root_tree_a", "f.edz")
    a = os.path.join(os.sep, "some_root_tree_b")
    rel, ok = best_relpath(t, a)
    assert not ok
    assert rel == os.path.abspath(t)


def test_cross_drive_stays_absolute(monkeypatch, tmp_path):
    import linac_gen.io.portable_paths as pp

    def raise_valueerror(*a, **kw):
        raise ValueError("path is on mount 'C:', start on mount 'D:'")

    monkeypatch.setattr(pp.os.path, "relpath", raise_valueerror)
    rel, ok = pp.best_relpath(tmp_path / "x", tmp_path)
    assert not ok


def test_resolve_candidates(tmp_path):
    (tmp_path / "sub").mkdir()
    f = tmp_path / "sub" / "lat.dat"
    f.write_text("END\n")
    assert resolve_candidates("lat.dat", [tmp_path / "sub"]) == str(f)
    assert resolve_candidates("lat.dat", [tmp_path]) is None
    assert resolve_candidates(str(f), []) == str(f)         # absolute hit
    assert resolve_candidates(str(f) + ".nope", []) is None


# ---------------------------------------------------------------------------
# writer relativization (real Fields-referencing deck)
# ---------------------------------------------------------------------------
MEBT_HWR = REPO / "examples" / "mebt_plus_hwr.dat"


@pytest.mark.skipif(not MEBT_HWR.exists(), reason="example deck missing")
def test_writer_relocatable_within_tree(tmp_path, monkeypatch):
    """Exporting inside the repo tree yields ONE FIELD_MAP_PATH card,
    bare filenames, no absolute tokens — and re-parses to the identical
    resolved field files, from an unrelated cwd."""
    from linac_gen.io.tracewin_parser import parse_tracewin
    from linac_gen.io.tracewin_writer import write_tracewin

    lat, _ = parse_tracewin(str(MEBT_HWR))
    orig = {e.name: os.path.realpath(e.field_file)
            for e in lat.elements
            if getattr(e, "field_file", None)}
    assert orig, "fixture deck should reference field maps"

    out_dir = REPO / "runs" / "_test_portable_writer"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "exported.dat"
    try:
        monkeypatch.chdir(tmp_path)               # unrelated cwd
        write_tracewin(lat, out)
        lines = out.read_text().splitlines()
        fmp = [l for l in lines if l.startswith("FIELD_MAP_PATH")]
        fm = [l for l in lines if l.startswith("FIELD_MAP ")]
        assert len(fmp) == 1                       # one shared directory
        assert not any(os.sep + "Users" in l or "/mnt/" in l for l in fm)
        lat2, _ = parse_tracewin(str(out))
        back = {e.name: os.path.realpath(e.field_file)
                for e in lat2.elements if getattr(e, "field_file", None)}
        assert back == orig
        # write→parse→write is byte-idempotent
        out2 = out_dir / "exported2.dat"
        write_tracewin(lat2, out2)
        assert out.read_text() == out2.read_text()
    finally:
        import shutil
        shutil.rmtree(out_dir, ignore_errors=True)


@pytest.mark.skipif(not MEBT_HWR.exists(), reason="example deck missing")
def test_writer_unrelated_tree_absolute_with_warning(tmp_path):
    """No usable common ancestor (system temp vs repo): absolute paths
    are kept and ONE warning per map directory is emitted."""
    from linac_gen.io.tracewin_parser import parse_tracewin
    from linac_gen.io.tracewin_writer import write_tracewin

    lat, _ = parse_tracewin(str(MEBT_HWR))
    out = tmp_path / "exported.dat"
    with pytest.warns(UserWarning, match="not be relocatable") as rec:
        write_tracewin(lat, out)
    reloca = [w for w in rec if "not be relocatable" in str(w.message)]
    assert len(reloca) == 1                        # deduped per directory
    lat2, _ = parse_tracewin(str(out))
    assert all(os.path.isabs(e.field_file) for e in lat2.elements
               if getattr(e, "field_file", None))


def test_writer_multiple_map_dirs(tmp_path):
    """Maps in different directories re-emit the (stateful) directive on
    each change and still resolve correctly."""
    from linac_gen.core.lattice import Lattice
    from linac_gen.elements.field_map import FieldMap
    from linac_gen.io.tracewin_parser import parse_tracewin
    from linac_gen.io.tracewin_writer import write_tracewin

    # Two synthetic 1-D map prefixes in different subdirs (reuse the
    # synthetic fixture content shipped for parser tests).
    fixture = REPO / "tests" / "io" / "fixtures"
    src = next(fixture.glob("*.edz"), None)
    if src is None:
        pytest.skip("no synthetic .edz fixture available")
    for sub in ("maps_a", "maps_b"):
        (tmp_path / sub).mkdir()
        (tmp_path / sub / "m.edz").write_bytes(src.read_bytes())

    lat = Lattice()
    for i, sub in enumerate(("maps_a", "maps_b", "maps_a")):
        fmap = FieldMap.from_file(
            f"FM{i}", str(tmp_path / sub / "m.edz"),
            phase=0.0, aperture=20.0, frequency=162.5,
        )
        # Writer provenance: geom code + extension-less prefix (what the
        # parser stores at load time).  ka=0: the synthetic fixture has
        # no .ouv pipe-radius file.
        fmap.geom = 100
        fmap.field_file = str(tmp_path / sub / "m")
        fmap.ka = 0
        lat.add(fmap)

    out = tmp_path / "out" / "multi.dat"
    out.parent.mkdir()
    write_tracewin(lat, out)
    lines = out.read_text().splitlines()
    fmp = [l for l in lines if l.startswith("FIELD_MAP_PATH")]
    assert len(fmp) == 3                           # a → b → a re-emissions
    lat2, _ = parse_tracewin(str(out))
    names = [os.path.realpath(e.field_file) for e in lat2.elements
             if getattr(e, "field_file", None)]
    assert names == [os.path.realpath(str(tmp_path / s / "m"))
                     for s in ("maps_a", "maps_b", "maps_a")]


# ---------------------------------------------------------------------------
# CLI project loader — project-dir-first
# ---------------------------------------------------------------------------
def _write_lgproj(dirpath, lattice_rel, beam_extra=None):
    import json
    beam = {"species": "proton", "energy": 3.0, "frequency": 162.5,
            "current": 0.0, "n_particles": 100}
    beam.update(beam_extra or {})
    (dirpath / "p.lgproj").write_text(json.dumps({
        "__kind__": "linac_gen_project",
        "__version__": 1,
        "lattice_path": lattice_rel,
        "beam": beam,
    }, indent=2))
    return dirpath / "p.lgproj"


def test_project_loader_project_dir_first(tmp_path, monkeypatch):
    """A same-named decoy in cwd must LOSE to the file next to the
    .lgproj."""
    from linac_gen.io.project import load_project
    proj_dir = tmp_path / "proj"
    cwd = tmp_path / "cwd"
    proj_dir.mkdir(), cwd.mkdir()
    real = proj_dir / "lat.dat"
    real.write_text("DRIFT 100 15 0\nEND\n")
    decoy = cwd / "lat.dat"
    decoy.write_text("QUAD 50 5 15\nEND\n")
    monkeypatch.chdir(cwd)
    pc = load_project(_write_lgproj(proj_dir, "lat.dat"))
    assert os.path.realpath(pc.lattice_path) == os.path.realpath(str(real))


def test_project_loader_resolves_relative_dst(tmp_path):
    from linac_gen.io.project import load_project
    proj_dir = tmp_path / "proj"
    proj_dir.mkdir()
    (proj_dir / "lat.dat").write_text("DRIFT 100 15 0\nEND\n")
    (proj_dir / "beam.dst").write_bytes(b"\x00" * 16)
    pc = load_project(_write_lgproj(
        proj_dir, "lat.dat",
        {"source": "file", "distribution_file": "beam.dst"}))
    assert os.path.isabs(pc.beam.distribution_file)
    assert os.path.realpath(pc.beam.distribution_file) == \
        os.path.realpath(str(proj_dir / "beam.dst"))


def test_repaired_examples_all_load():
    """Every tracked example .lgproj resolves to an existing lattice —
    the 12 dead /mnt/c/... projects are repaired and must stay repaired."""
    import subprocess
    from linac_gen.io.project import load_project
    files = subprocess.run(
        ["git", "ls-files", "examples/*.lgproj", "examples/**/*.lgproj"],
        capture_output=True, text=True, cwd=REPO,
    ).stdout.split()
    if not files:
        pytest.skip("not a git checkout")
    for f in files:
        pc = load_project(REPO / f)
        assert os.path.isfile(pc.lattice_path), (f, pc.lattice_path)
