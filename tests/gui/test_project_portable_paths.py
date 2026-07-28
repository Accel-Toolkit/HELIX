"""GUI project files are relocatable (2026-07-11).

Save writes ``lattice_path`` / ``beam.distribution_file`` RELATIVE to the
.lgproj; load resolves relative entries against the project directory
first.  Legacy absolute-path projects keep loading through the existing
fallbacks.
"""
from __future__ import annotations

import json
import os
import shutil

import pytest

pytest.importorskip("PyQt6")


@pytest.fixture()
def win(qapp):
    from linac_gen_gui.interphase.app import InterphaseWindow
    w = InterphaseWindow()
    yield w
    w.close()
    w.deleteLater()


def _make_project_dir(tmp_path, name="a"):
    d = tmp_path / name
    d.mkdir()
    (d / "lat.dat").write_text("FREQ 162.5\nDRIFT 100 15 0\nEND\n")
    return d


def test_collect_writes_relative_paths(win, tmp_path):
    d = _make_project_dir(tmp_path)
    from linac_gen.io.tracewin_parser import parse_tracewin
    lat, _ = parse_tracewin(str(d / "lat.dat"))
    win.state.set_lattice(lat, str(d / "lat.dat"))

    data = win._collect_project_dict(project_path=str(d / "p.lgproj"))
    assert data["lattice_path"] == "lat.dat"       # relative, POSIX

    # Without a project_path (dirty-check snapshots) paths stay as-is.
    data2 = win._collect_project_dict()
    assert os.path.isabs(data2["lattice_path"])


def test_save_move_load_round_trip(win, tmp_path):
    """Save under a/, copy the whole directory to b/, load from b/ —
    the lattice must resolve inside b/ (the whole point)."""
    a = _make_project_dir(tmp_path, "a")
    from linac_gen.io.tracewin_parser import parse_tracewin
    lat, _ = parse_tracewin(str(a / "lat.dat"))
    win.state.set_lattice(lat, str(a / "lat.dat"))

    data = win._collect_project_dict(project_path=str(a / "p.lgproj"))
    (a / "p.lgproj").write_text(json.dumps(data, indent=2))

    b = tmp_path / "b"
    shutil.copytree(a, b)
    shutil.rmtree(a)                                # original gone

    win._apply_project_dict(json.loads((b / "p.lgproj").read_text()),
                            project_path=str(b / "p.lgproj"), silent=True)
    assert win.state.lattice_path is not None
    assert os.path.realpath(win.state.lattice_path) == \
        os.path.realpath(str(b / "lat.dat"))


def test_legacy_absolute_project_still_loads(win, tmp_path):
    """Absolute lattice_path (legacy projects): loads as-is when it
    exists, and via the basename-next-to-project fallback when it
    doesn't (the WSL→macOS case)."""
    d = _make_project_dir(tmp_path, "legacy")
    # 1. absolute + existing
    win._apply_project_dict(
        {"__kind__": "linac_gen_project",
         "lattice_path": str(d / "lat.dat")},
        project_path=str(d / "p.lgproj"), silent=True)
    assert os.path.realpath(win.state.lattice_path) == \
        os.path.realpath(str(d / "lat.dat"))
    # 2. absolute + dead (foreign machine) → basename fallback
    win._apply_project_dict(
        {"__kind__": "linac_gen_project",
         "lattice_path": "/mnt/c/Project/Linac_Gen/examples/lat.dat"},
        project_path=str(d / "p.lgproj"), silent=True)
    assert os.path.realpath(win.state.lattice_path) == \
        os.path.realpath(str(d / "lat.dat"))
