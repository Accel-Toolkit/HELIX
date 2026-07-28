"""``; HELIX_SC_GRID`` — mid-lattice PIC grid-extent directive.

Motivating case (2026-07-25): the combined MEBT->Foil deck needs the
linac at the default grid extent but the BTL at 20 sigma; a single
per-run ``grid_extent`` forced one compromise on both sections.
"""
from __future__ import annotations

import os
import tempfile

import numpy as np
import pytest

from linac_gen.core.config import BeamConfig, SpaceChargeConfig
from linac_gen.core.lattice import Lattice
from linac_gen.elements.drift import Drift
from linac_gen.elements.sc_grid import ScGridDirective
from linac_gen.io.tracewin_parser import parse_tracewin
from linac_gen.io.tracewin_writer import write_tracewin
from linac_gen.pic.pic_solver import PicSolver


def _write_dat(text: str) -> str:
    fd, path = tempfile.mkstemp(suffix=".dat")
    with os.fdopen(fd, "w") as fh:
        fh.write(text)
    return path


# ---------------------------------------------------------------------------
# parse / write round-trip
# ---------------------------------------------------------------------------
def test_card_parses_to_directive():
    p = _write_dat("DRIFT 100 20 0\n"
                   "; HELIX_SC_GRID 20   ; widen for the BTL\n"
                   "DRIFT 100 20 0\nEND\n")
    lat, meta = parse_tracewin(p)
    kinds = [type(e).__name__ for e in lat.elements]
    assert kinds == ["Drift", "ScGridDirective", "Drift"]
    assert lat.elements[1].extent_sigma == 20.0
    assert meta["warnings"] == []
    os.unlink(p)


def test_card_round_trips_through_writer():
    p = _write_dat("DRIFT 100 20 0\n; HELIX_SC_GRID 12.5\nDRIFT 100 20 0\nEND\n")
    lat, _ = parse_tracewin(p)
    out = p + ".rt.dat"
    write_tracewin(lat, out)
    text = open(out).read()
    assert "; HELIX_SC_GRID 12.5" in text
    lat2, meta2 = parse_tracewin(out)
    assert [type(e).__name__ for e in lat2.elements] == \
           [type(e).__name__ for e in lat.elements]
    assert lat2.elements[1].extent_sigma == 12.5
    os.unlink(p); os.unlink(out)


def test_nonpositive_extent_downgrades_to_marker():
    p = _write_dat("DRIFT 100 20 0\n; HELIX_SC_GRID 0\nEND\n")
    lat, meta = parse_tracewin(p)
    assert any("HELIX_SC_GRID" in w for w in meta["warnings"])
    assert [type(e).__name__ for e in lat.elements] == ["Drift", "Marker"]
    with pytest.raises(ValueError):
        parse_tracewin(p, strict=True)
    os.unlink(p)


def test_directive_refuses_bad_extent_programmatically():
    with pytest.raises(ValueError):
        ScGridDirective("SCGRID_001", extent_sigma=-1.0)


# ---------------------------------------------------------------------------
# PicSolver: the override retargets the grid without touching the config
# ---------------------------------------------------------------------------
def _spans(solver):
    return np.asarray(solver._grid_max) - np.asarray(solver._grid_min)


def test_fixed_mode_rederives_grid_once_on_override():
    cfg = SpaceChargeConfig(nx=8, ny=8, nz=8, grid_extent=5.0,
                            grid_mode="fixed", use_gpu="cpu")
    solver = PicSolver(cfg)
    rng = np.random.default_rng(1)
    coords = rng.normal(scale=1e-3, size=(500, 3))
    solver._setup_grid(coords)
    span5 = _spans(solver).copy()
    # Frozen: a second call must not change the grid.
    solver._setup_grid(coords * 3.0)
    np.testing.assert_array_equal(_spans(solver), span5)
    # Override: exactly one re-derivation at the new extent...
    solver.set_grid_extent(20.0)
    solver._setup_grid(coords)
    np.testing.assert_allclose(_spans(solver), span5 * 4.0, rtol=1e-12)
    # ...then frozen again.
    span20 = _spans(solver).copy()
    solver._setup_grid(coords * 3.0)
    np.testing.assert_array_equal(_spans(solver), span20)
    # The caller-owned config is NEVER mutated (no cross-run leak).
    assert cfg.grid_extent == 5.0


def test_adaptive_mode_honours_override_every_kick():
    cfg = SpaceChargeConfig(nx=8, ny=8, nz=8, grid_extent=5.0,
                            grid_mode="adaptive", use_gpu="cpu")
    solver = PicSolver(cfg)
    rng = np.random.default_rng(2)
    coords = rng.normal(scale=1e-3, size=(500, 3))
    solver._setup_grid(coords)
    span5 = _spans(solver).copy()
    solver.set_grid_extent(10.0)
    solver._setup_grid(coords)
    np.testing.assert_allclose(_spans(solver), span5 * 2.0, rtol=1e-12)
    assert cfg.grid_extent == 5.0


# ---------------------------------------------------------------------------
# tracker end-to-end: the card reaches the live solver
# ---------------------------------------------------------------------------
def test_mp_run_applies_card_to_live_solver():
    from linac_gen.core.simulation import Simulation
    from linac_gen.distributions.factory import create_beam

    lat = Lattice()
    lat.add(Drift(name="D1", length=100.0, aperture=50.0))
    lat.add(ScGridDirective("SCGRID_001", extent_sigma=20.0))
    lat.add(Drift(name="D2", length=100.0, aperture=50.0))

    cfg = BeamConfig(species="H-", energy=2.0, frequency=162.5,
                     current=5.0, n_particles=300,
                     emit_nx=0.2, alpha_x=0.0, beta_x=0.5,
                     emit_ny=0.2, alpha_y=0.0, beta_y=0.5,
                     emit_z=0.06, alpha_z=0.0, beta_z=500.0)
    beam = create_beam(cfg, seed=7)
    sim = Simulation(lat, beam,
                     space_charge=SpaceChargeConfig(
                         nx=8, ny=8, nz=8, grid_extent=5.0,
                         grid_mode="fixed", use_gpu="cpu"))
    sim.run()
    solver = sim._pic_solver
    assert solver is not None
    assert solver._extent_override == 20.0
    # The frozen grid was re-derived at 20 sigma after the card.
    spans = _spans(solver)
    assert np.all(spans > 0)


def test_card_warns_once_when_no_pic_solver(recwarn):
    """SC off -> the card is inert and must SAY so (fail loud)."""
    from linac_gen.core.simulation import Simulation
    from linac_gen.distributions.factory import create_beam

    lat = Lattice()
    lat.add(Drift(name="D1", length=50.0, aperture=50.0))
    lat.add(ScGridDirective("SCGRID_001", extent_sigma=20.0))
    lat.add(ScGridDirective("SCGRID_002", extent_sigma=8.0))
    lat.add(Drift(name="D2", length=50.0, aperture=50.0))

    cfg = BeamConfig(species="H-", energy=2.0, frequency=162.5,
                     current=0.0, n_particles=100,
                     emit_nx=0.2, alpha_x=0.0, beta_x=0.5,
                     emit_ny=0.2, alpha_y=0.0, beta_y=0.5,
                     emit_z=0.06, alpha_z=0.0, beta_z=500.0)
    beam = create_beam(cfg, seed=7)
    sim = Simulation(lat, beam, space_charge="off")
    sim.run()
    inert = [w for w in recwarn.list
             if "HELIX_SC_GRID" in str(w.message)]
    assert len(inert) == 1          # warn once, not per card
