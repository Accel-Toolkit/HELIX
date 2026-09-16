"""``write_project`` — the GUI-free ``.lgproj`` writer round-trips through
``load_project`` and stores relocatable paths."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from linac_gen.core.config import BeamConfig
from linac_gen.io import project as project_mod
from linac_gen.io.project import load_project, write_project

FIXTURES = Path(__file__).parent / "fixtures"


def _deck(tmp_path: Path, name="deck.dat") -> Path:
    p = tmp_path / name
    p.write_text((FIXTURES / "simple_fodo.dat").read_text(encoding="utf-8"),
                 encoding="utf-8")
    return p


def test_round_trip_beam_and_relative_lattice(tmp_path):
    deck = _deck(tmp_path)
    beam = BeamConfig(species="H-", energy=116.1, frequency=804.96, current=23.7,
                      n_particles=12345, emit_nx=0.918685, alpha_x=-0.2758,
                      beta_x=6.46037, emit_z=0.9079, alpha_z=0.5, beta_z=61.22)
    out = write_project(tmp_path / "sub" / "deck.lgproj", lattice_path=deck,
                        beam=beam, convergence={"grid_nx": 64, "tracewin_ini": {"file": "x.ini"}},
                        extra={"calc_dir": "runs"})
    assert out.is_file()
    raw = json.loads(out.read_text(encoding="utf-8"))
    assert raw["__kind__"] == "linac_gen_project" and raw["__version__"] == 1
    assert raw["lattice_path"] == "../deck.dat"          # relative, POSIX
    assert raw["calc_dir"] == "runs"
    assert raw["convergence"]["tracewin_ini"] == {"file": "x.ini"}
    proj = load_project(out)
    assert proj.beam == beam
    assert Path(proj.lattice_path).resolve() == deck.resolve()
    assert proj.convergence["grid_nx"] == 64


def test_refuses_to_overwrite_unless_asked(tmp_path):
    deck = _deck(tmp_path)
    out = tmp_path / "p.lgproj"
    write_project(out, lattice_path=deck, beam=BeamConfig())
    with pytest.raises(FileExistsError):
        write_project(out, lattice_path=deck, beam=BeamConfig(energy=9.0))
    assert load_project(out).beam.energy == BeamConfig().energy
    write_project(out, lattice_path=deck, beam=BeamConfig(energy=9.0), overwrite=True)
    assert load_project(out).beam.energy == 9.0


def test_distribution_file_is_relativised(tmp_path):
    deck = _deck(tmp_path)
    dst = tmp_path / "beam.dst"
    dst.write_bytes(b"\0")
    beam = BeamConfig(source="file", distribution_file=str(dst))
    out = write_project(tmp_path / "p.lgproj", lattice_path=deck, beam=beam)
    raw = json.loads(out.read_text(encoding="utf-8"))
    assert raw["beam"]["distribution_file"] == "beam.dst"
    assert Path(load_project(out).beam.distribution_file).resolve() == dst.resolve()


def test_symlinked_project_directory_still_relativises(tmp_path):
    real = tmp_path / "real"
    real.mkdir()
    link = tmp_path / "link"
    link.symlink_to(real, target_is_directory=True)
    deck = _deck(link)                                # written through the link
    out = write_project(link / "deck.lgproj", lattice_path=link / "deck.dat",
                        beam=BeamConfig())
    raw = json.loads(out.read_text(encoding="utf-8"))
    assert raw["lattice_path"] == "deck.dat"
    assert Path(load_project(out).lattice_path).resolve() == deck.resolve()


def test_non_portable_path_stays_absolute(tmp_path, monkeypatch):
    deck = _deck(tmp_path)
    import linac_gen.io.portable_paths as pp
    monkeypatch.setattr(pp, "best_relpath",
                        lambda target, anchor: (str(Path(target).resolve()), False))
    out = write_project(tmp_path / "p.lgproj", lattice_path=deck, beam=BeamConfig())
    raw = json.loads(out.read_text(encoding="utf-8"))
    assert Path(raw["lattice_path"]).is_absolute()
    assert Path(load_project(out).lattice_path) == deck.resolve()


def test_writer_is_gui_free():
    import inspect
    src = inspect.getsource(project_mod)
    assert "PyQt6" not in src and "linac_gen_gui" not in src
    assert "write_project" in project_mod.__all__
