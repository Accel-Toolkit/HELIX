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


def test_nonfinite_extent_downgrades_to_marker():
    # 1e999 overflows to inf when parsed; it used to pass the "> 0" check.
    p = _write_dat("DRIFT 100 20 0\n; HELIX_SC_GRID 1e999\nEND\n")
    lat, meta = parse_tracewin(p)
    assert any("finite" in w for w in meta["warnings"])
    assert [type(e).__name__ for e in lat.elements] == ["Drift", "Marker"]
    with pytest.raises(ValueError):
        parse_tracewin(p, strict=True)
    os.unlink(p)


@pytest.mark.parametrize("extent", [-1.0, 0.0, float("inf"), float("nan"), 1.0e7])
def test_directive_refuses_bad_extent_programmatically(extent):
    with pytest.raises(ValueError):
        ScGridDirective("SCGRID_001", extent_sigma=extent)


@pytest.mark.parametrize("extent", [-1.0, 0.0, float("inf"), float("nan"), 1.0e150])
def test_solver_refuses_bad_extent_at_run_time(extent):
    """A card's extent can be changed after parsing (--set, the assistant);
    the solver checks it too instead of building a NaN Green's function."""
    solver = PicSolver(SpaceChargeConfig(nx=8, ny=8, nz=8, use_gpu="cpu"))
    with pytest.raises(ValueError):
        solver.set_grid_extent(extent)
    assert solver._extent_override is None


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


# ---------------------------------------------------------------------------
# grid_card: the named adaptive methods ignore the card; everything else
# applies it, as before.  Every card reached is recorded.
# ---------------------------------------------------------------------------
def _card_run(sc):
    from linac_gen.core.simulation import Simulation
    from linac_gen.distributions.factory import create_beam

    lat = Lattice()
    lat.add(Drift(name="D1", length=100.0, aperture=50.0))
    lat.add(ScGridDirective("SCGRID_001", extent_sigma=20.0))
    lat.add(Drift(name="D2", length=100.0, aperture=50.0))
    lat.add(ScGridDirective("SCGRID_002", extent_sigma=8.0))
    lat.add(Drift(name="D3", length=50.0, aperture=50.0))
    cfg = BeamConfig(species="H-", energy=2.0, frequency=162.5,
                     current=5.0, n_particles=300,
                     emit_nx=0.2, alpha_x=0.0, beta_x=0.5,
                     emit_ny=0.2, alpha_y=0.0, beta_y=0.5,
                     emit_z=0.06, alpha_z=0.0, beta_z=500.0)
    sim = Simulation(lat, create_beam(cfg, seed=7), space_charge=sc)
    return sim, sim.run()


def _card_warnings(recwarn):
    return [str(w.message) for w in recwarn.list
            if "HELIX_SC_GRID" in str(w.message)]


def test_named_adaptive_method_ignores_cards_and_records_them(recwarn):
    sc = SpaceChargeConfig(method="tsc_adaptive", nx=8, ny=8, nz=8,
                           use_gpu="cpu")
    assert sc.grid_card == "auto"
    sim, rec = _card_run(sc)
    assert sim._pic_solver._extent_override is None
    msgs = _card_warnings(recwarn)
    assert len(msgs) == 1
    assert "ignored" in msgs[0] and "tsc_adaptive" in msgs[0]
    assert [c["action"] for c in rec.sc_grid_cards] == ["ignored", "ignored"]
    assert [c["extent_sigma"] for c in rec.sc_grid_cards] == [20.0, 8.0]
    assert rec.sc_grid_cards[0]["s_mm"] == pytest.approx(100.0)
    assert [c["element"] for c in rec.sc_grid_cards] == ["SCGRID_001",
                                                         "SCGRID_002"]


def test_adaptive_config_without_a_method_name_applies_cards(recwarn):
    """Explicit fields, no method: the cards apply exactly as before."""
    sc = SpaceChargeConfig(nx=8, ny=8, nz=8, grid_mode="adaptive",
                           kernel="tsc", use_gpu="cpu")
    assert sc.method == "tsc_adaptive" and sc.grid_card == "apply"
    sim, rec = _card_run(sc)
    assert sim._pic_solver._extent_override == 8.0
    assert [c["action"] for c in rec.sc_grid_cards] == ["applied", "applied"]
    assert not _card_warnings(recwarn)


def test_named_adaptive_method_can_be_told_to_apply_cards():
    sc = SpaceChargeConfig(method="cic_adaptive", grid_card="apply",
                           nx=8, ny=8, nz=8, use_gpu="cpu")
    assert sc.method == "cic_adaptive"
    sim, _ = _card_run(sc)
    assert sim._pic_solver._extent_override == 8.0


def test_fixed_methods_apply_cards():
    for name in ("cic_fixed", "tsc_fixed"):
        assert SpaceChargeConfig(method=name).grid_card == "apply"


def test_card_under_the_torch_solver_names_it(recwarn):
    pytest.importorskip("torch")
    sc = SpaceChargeConfig(nx=8, ny=8, nz=8, grid_mode="adaptive",
                           sc_backend="torch", use_gpu="cpu")
    _, rec = _card_run(sc)
    msgs = _card_warnings(recwarn)
    assert len(msgs) == 1
    assert "TorchPicSolver" in msgs[0] and "no 3-D bunched" not in msgs[0]
    assert all(c["action"].startswith("inert") for c in rec.sc_grid_cards)


def test_grid_card_value_is_checked():
    with pytest.raises(ValueError, match="grid_card"):
        SpaceChargeConfig(grid_card="sometimes")


def test_card_record_reaches_the_hdf5_provenance(tmp_path):
    import json

    import h5py

    from linac_gen.io.hdf5_output import save_results_hdf5
    sc = SpaceChargeConfig(method="tsc_adaptive", nx=8, ny=8, nz=8,
                           use_gpu="cpu")
    _, rec = _card_run(sc)
    out = tmp_path / "r.h5"
    save_results_hdf5(rec, str(out))
    with h5py.File(out, "r") as f:
        cards = json.loads(f["provenance"].attrs["sc_grid_cards"])
        assert f["provenance"].attrs["sc_grid_card"] == "auto"
    assert [c["action"] for c in cards] == ["ignored", "ignored"]


def test_auto_applies_cards_on_a_fixed_grid():
    """The default follows the grid: a named adaptive method asked for a
    fixed grid is the fixed method, which applies the cards."""
    sc = SpaceChargeConfig(method="tsc_adaptive", grid_mode="fixed",
                           nx=8, ny=8, nz=8, use_gpu="cpu")
    assert sc.method == "tsc_fixed"
    sim, rec = _card_run(sc)
    assert sim._pic_solver._extent_override == 8.0
    assert [c["action"] for c in rec.sc_grid_cards] == ["applied", "applied"]


def test_explicit_ignore_on_a_fixed_grid_says_so(recwarn):
    sc = SpaceChargeConfig(nx=8, ny=8, nz=8, grid_card="ignore",
                           use_gpu="cpu")
    sim, rec = _card_run(sc)
    assert sim._pic_solver._extent_override is None
    msgs = _card_warnings(recwarn)
    assert len(msgs) == 1 and "grid_card='ignore'" in msgs[0]
    assert "resizes its grid" not in msgs[0]


def test_differentiable_pic_card_message_gives_no_advice_that_cannot_work(
        recwarn):
    pytest.importorskip("torch")
    sc = SpaceChargeConfig(method="differentiable_pic", nx=8, ny=8, nz=8,
                           use_gpu="cpu")
    _, rec = _card_run(sc)
    msgs = _card_warnings(recwarn)
    assert len(msgs) == 1 and "TorchPicSolver" in msgs[0]
    assert "Set grid_card" not in msgs[0]
