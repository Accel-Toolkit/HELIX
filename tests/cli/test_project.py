"""The standalone .lgproj project loader (linac_gen/io/project.py)."""
import json
from pathlib import Path

import pytest

from linac_gen.io.project import load_project

_REPO = Path(__file__).resolve().parents[2]


def test_load_bundled_project():
    proj = load_project(_REPO / "examples" / "csr_chicane.lgproj")
    assert proj.beam.species == "H-"
    assert proj.beam.energy == 800.0
    assert proj.beam.current == 5.0
    assert proj.lattice_path.endswith("csr_chicane.dat")
    assert proj.convergence  # the convergence sub-dict is populated


def test_relative_lattice_path_resolved(tmp_path):
    """A relative lattice_path resolves to an absolute, existing file."""
    (tmp_path / "mylat.dat").write_text("DRIFT 100 20\nEND\n")
    proj_file = tmp_path / "p.lgproj"
    proj_file.write_text(json.dumps({
        "__kind__": "linac_gen_project",
        "lattice_path": "mylat.dat",
        "beam": {"species": "proton", "energy": 5.0},
    }))
    proj = load_project(proj_file)
    assert Path(proj.lattice_path).is_absolute()
    assert Path(proj.lattice_path).is_file()
    assert proj.beam.energy == 5.0


def test_unknown_beam_keys_ignored(tmp_path):
    """A project written by a newer GUI (extra beam keys) still loads."""
    proj_file = tmp_path / "p.lgproj"
    proj_file.write_text(json.dumps({
        "__kind__": "linac_gen_project",
        "lattice_path": "x.dat",
        "beam": {"species": "proton", "energy": 7.0, "future_key": 123},
    }))
    proj = load_project(proj_file)
    assert proj.beam.energy == 7.0


def test_rejects_non_project(tmp_path):
    bad = tmp_path / "bad.lgproj"
    bad.write_text('{"something": "else"}')
    with pytest.raises(ValueError, match="not a linac_gen project"):
        load_project(bad)
