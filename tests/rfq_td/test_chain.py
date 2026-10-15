"""The time-domain RFQ through the real run path (Simulation -> Tracker),
on a small synthetic RFQ: a generated vane file and deck."""
from __future__ import annotations

import dataclasses
import warnings

import numpy as np
import pytest

pytest.importorskip("pyamg")

from linac_gen.core.config import BeamConfig
from linac_gen.core.simulation import Simulation
from linac_gen.distributions.factory import create_beam
from linac_gen.io.tracewin_parser import parse_tracewin
from linac_gen.rfq_td import build
from linac_gen.rfq_td.chain import RfqTdError

N_CELLS, L_CELL, GAP_IN, GAP_OUT = 8, 9.5, 5.0, 3.0
V, R0, M = 50_000.0, 3.0, 1.1


def _write_rfq(tmp_path):
    """Vane file (pole 1 = +y, 2 = +x, 3 = -y, 4 = -x) + deck."""
    z_end = N_CELLS * L_CELL - GAP_OUT
    z = np.linspace(GAP_IN, z_end, 400)
    amp = (M - 1) / (M + 1)
    c = np.cos(np.pi * (z - GAP_IN) / L_CELL)
    rx = R0 * (1 + amp * c) * 1e-3
    ry = R0 * (1 - amp * c) * 1e-3
    tc = np.full_like(z, 0.75 * R0 * 1e-3)
    zero = np.zeros_like(z)
    cols = [z * 1e-3,
            ry, tc, np.full_like(z, +V / 2), zero,      # pole 1, +y
            rx, tc, np.full_like(z, -V / 2), zero,      # pole 2, +x
            ry, tc, np.full_like(z, +V / 2), zero,      # pole 3, -y
            rx, tc, np.full_like(z, -V / 2), zero]      # pole 4, -x
    vane = tmp_path / "syn.vane"
    np.savetxt(vane, np.column_stack(cols), fmt="%.10f")
    lines = [f"RFQ_GEOM 1 {vane.name}",
             f"RFQ_GAP_RMS_FFS {GAP_IN} {GAP_OUT}", "DRIFT 0.001 20"]
    for k in range(N_CELLS):
        t = 2 if k % 2 else -2
        lines.append(f"RFQ_CELL {V:.0f} {R0} 0.05 {M} {L_CELL} -90 {t} "
                     f"{0.75 * R0} 0 0 0 0 0 0")
    lines.append("MARKER mk")
    lines.append("DRIFT 5 20")
    lines.append("END")
    deck = tmp_path / "syn.dat"
    deck.write_text("\n".join(lines) + "\n")
    return deck


def _beam(n=600, continuous=True, current=0.0):
    cfg = BeamConfig(species="proton", energy=0.050, frequency=162.5,
                     current=current, n_particles=n, distribution="gaussian",
                     cutoff=3.0, emit_nx=0.02, alpha_x=0.0, beta_x=0.05,
                     emit_ny=0.02, alpha_y=0.0, beta_y=0.05,
                     emit_z=0.0 if continuous else 0.3, alpha_z=0.0,
                     beta_z=1.0,
                     continuous=continuous, dc_energy_spread_keV=0.0)
    return create_beam(cfg, seed=7)


def _lattice(tmp_path, solver="time_domain"):
    deck = _write_rfq(tmp_path)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        lat, _ = parse_tracewin(str(deck))
    lat.step_config = dataclasses.replace(lat.step_config, rfq_solver=solver,
                                          rfq_td_steps_per_period=24)
    return lat


@pytest.fixture(autouse=True)
def _coarse_map(monkeypatch):
    # a coarse field map keeps the test fast; physics is checked elsewhere
    monkeypatch.setitem(build.DEFAULT_SETTINGS, "h", 0.2)
    monkeypatch.setitem(build.DEFAULT_SETTINGS, "x_max", 10.0)
    monkeypatch.setitem(build.DEFAULT_SETTINGS, "pad_mm", 10.0)
    monkeypatch.setitem(build.DEFAULT_SETTINGS, "upstream_mm", 12.0)
    monkeypatch.setitem(build.DEFAULT_SETTINGS, "downstream_mm", 6.0)


@pytest.mark.slow
def test_time_domain_chain_through_simulation(tmp_path):
    lat = _lattice(tmp_path)
    assert lat.rfq_gaps_mm == (GAP_IN, GAP_OUT)
    b = _beam()
    res = Simulation(lat, b, space_charge="off", rfq_geometry="off").run()
    # one row per element plus INPUT, rows named after the elements
    assert len(res.s) == 1 + len(lat.elements)
    assert list(res.element_names[1:]) == [e.name for e in lat.elements]
    info = res.rfq_td
    assert info["n_tracked"] == 600 and info["steps"] > 0
    assert 0 <= info["n_lost"] < 600
    # losses carry RFQ cell names and positions inside the chain
    lt = res.loss_table
    cells = {e.name for e in lat.elements if type(e).__name__ == "RfqCell"}
    assert set(lt["element_name"]) <= cells
    # the reference is exactly the per-cell model's
    lat_c = _lattice(tmp_path, solver="cells")
    b2 = _beam()
    res_c = Simulation(lat_c, b2, space_charge="off", rfq_geometry="off").run()
    np.testing.assert_array_equal(np.asarray(res.ref_w_kin),
                                  np.asarray(res_c.ref_w_kin))
    np.testing.assert_array_equal(np.asarray(res.ref_phi_s),
                                  np.asarray(res_c.ref_phi_s))
    np.testing.assert_array_equal(np.asarray(res.s), np.asarray(res_c.s))
    # the second run of the same RFQ reuses the cached map (no rebuild)
    b3 = _beam()
    res3 = Simulation(_lattice(tmp_path), b3, space_charge="off",
                      rfq_geometry="off").run()
    assert res3.rfq_td["map_key"] == info["map_key"]
    np.testing.assert_array_equal(b3.particles, b.particles)


@pytest.mark.slow
def test_time_domain_with_space_charge_runs(tmp_path):
    from linac_gen.core.config import SpaceChargeConfig
    lat = _lattice(tmp_path)
    b = _beam(n=800, current=5.0)
    res = Simulation(lat, b, space_charge=SpaceChargeConfig(nx=32, ny=32,
                                                            nz=32),
                     rfq_geometry="off").run()
    assert res.rfq_td["sc"] and res.rfq_td["sc_solves"] > 0


def test_refuses_bunched_input(tmp_path):
    lat = _lattice(tmp_path)
    b = _beam(continuous=False)
    with pytest.raises(RfqTdError, match="DC"):
        Simulation(lat, b, space_charge="off", rfq_geometry="off").run()


def test_refuses_element_between_cells(tmp_path):
    from linac_gen.elements.thin_lens import ThinLens
    lat = _lattice(tmp_path)
    cells = [i for i, e in enumerate(lat.elements)
             if type(e).__name__ == "RfqCell"]
    lat.elements.insert(cells[3], ThinLens("tl", fx=1e4, fy=1e4))
    with pytest.raises(RfqTdError, match="between RFQ"):
        Simulation(lat, _beam(), space_charge="off",
                   rfq_geometry="off").run()


def test_refuses_unequal_misalignment(tmp_path):
    lat = _lattice(tmp_path)
    cells = [e for e in lat.elements if type(e).__name__ == "RfqCell"]
    cells[2].dx = 0.1
    with pytest.raises(RfqTdError, match="misalignments"):
        Simulation(lat, _beam(), space_charge="off",
                   rfq_geometry="off").run()


def test_refuses_missing_vane_file(tmp_path):
    lat = _lattice(tmp_path)
    lat.rfq_geom_file = str(tmp_path / "nope.vane")
    with pytest.raises(RfqTdError, match="vane geometry"):
        Simulation(lat, _beam(), space_charge="off",
                   rfq_geometry="off").run()


def test_cells_mode_makes_no_driver(tmp_path):
    from linac_gen.tracking.tracker import Tracker
    lat = _lattice(tmp_path, solver="cells")
    t = Tracker(lat, _beam(), pic_solver="off")
    t.run()
    assert t._rfq_td is None


def test_backtracking_keeps_the_per_cell_model_and_says_so(tmp_path):
    from linac_gen.tracking.backtrack import BacktrackWarning
    lat = _lattice(tmp_path)
    b = _beam(n=50, continuous=False)
    cells = [i for i, e in enumerate(lat.elements)
             if type(e).__name__ == "RfqCell"]
    with pytest.warns(BacktrackWarning, match="forward only"):
        Simulation(lat, b, space_charge="off").run_backtrack(
            entrance_ref=b.ref.copy(), start=cells[0], end=cells[-1],
            approximate_backtracking=True)


def test_a_daemonic_worker_builds_the_map_in_process(tmp_path, monkeypatch):
    """A multiprocessing.Pool worker (parallel CMA-ES matching) may not
    start processes: the map is built serially there."""
    import multiprocessing

    class _Daemon:
        daemon = True
        name = "PoolWorker-1"

    def _no_pool(*a, **k):
        raise AssertionError("daemonic processes are not allowed to have "
                             "children")

    monkeypatch.delenv("HELIX_RFQTD_WORKERS", raising=False)
    monkeypatch.setattr(multiprocessing, "current_process", _Daemon)
    monkeypatch.setattr(build, "ProcessPoolExecutor", _no_pool)
    res = Simulation(_lattice(tmp_path), _beam(n=50), space_charge="off",
                     rfq_geometry="off").run()
    assert res.rfq_td["n_tracked"] == 50


def test_stop_during_the_field_map_build_is_a_clean_abort(tmp_path,
                                                          monkeypatch):
    """Stop pressed while the map is being built: the run ends as aborted
    (not failed) and no partial map is cached."""
    from linac_gen.tracking.tracker import Tracker
    flag = {"stop": False}
    real = build._solve_task

    def solve(task):
        flag["stop"] = True            # Stop pressed during the first window
        return real(task)

    monkeypatch.setattr(build, "_solve_task", solve)
    t = Tracker(_lattice(tmp_path), _beam(n=50), pic_solver="off",
                should_abort=lambda: flag["stop"])
    t.run()
    assert t.aborted
    assert not list((tmp_path / "cache").rglob("*.npz"))
    assert not list((tmp_path / "cache").rglob("*.lock"))


def test_record_substeps_rows_sit_on_the_per_cell_rows(tmp_path):
    """With record_substeps each cell gets SUBPLANES_PER_CELL rows, the
    last at its exit; every row sits at a position (and reference energy
    and phase) where the per-cell model records one too."""
    from linac_gen.rfq_td.chain import SUBPLANES_PER_CELL
    kw = dict(space_charge="off", rfq_geometry="off", record_substeps=True)
    lat = _lattice(tmp_path)
    res = Simulation(lat, _beam(n=200), **kw).run()
    res_c = Simulation(_lattice(tmp_path, solver="cells"), _beam(n=200),
                       **kw).run()
    cells = [e.name for e in lat.elements if type(e).__name__ == "RfqCell"]
    names = list(res.element_names)
    assert all(names.count(c) == SUBPLANES_PER_CELL for c in cells)
    rows = [i for i, nm in enumerate(names) if nm in set(cells)]
    s_td = np.asarray(res.s)[rows]
    s_c = np.asarray(res_c.s)
    w_c = np.asarray(res_c.ref_w_kin)
    pos = np.searchsorted(s_c, s_td)
    np.testing.assert_array_equal(s_c[pos], s_td)
    np.testing.assert_array_equal(w_c[pos], np.asarray(res.ref_w_kin)[rows])
    # the exit rows are those of a run without sub-steps
    res_x = Simulation(_lattice(tmp_path), _beam(n=200), space_charge="off",
                       rfq_geometry="off").run()
    last = [max(i for i, nm in enumerate(names) if nm == c) for c in cells]
    rows_x = [i for i, nm in enumerate(res_x.element_names) if nm in cells]
    for key in ("s", "sigma_x", "sigma_phi", "emit_x"):
        np.testing.assert_array_equal(np.asarray(getattr(res, key))[last],
                                      np.asarray(getattr(res_x, key))[rows_x])


@pytest.mark.parametrize("cores, expected", [
    ([90, 60, 40, 30], True),          # engaged from the DC end, held
    ([90, 60, 20, 30], False),         # released at 20, 30 < 35
    ([90, 20, 40], True),              # re-engaged at 40
    ([30, 30, 30], False),             # never reached 35
    ([10, 90, 50, 26], True),          # the first cell is not consulted
])
def test_train_gate_hysteresis(cores, expected):
    from linac_gen.rfq_td.chain import RfqTdDriver
    rng = np.random.default_rng(3)
    n = 4000
    base = rng.normal(0.0, 1.0, n)
    states = np.zeros((len(cores), n, 6), np.float32)
    for k, c in enumerate(cores):
        states[k, :, 4] = base * c      # core (half 16-84 span) ~ c
    crossed = np.ones((len(cores), n), bool)
    assert RfqTdDriver._train_gate(states, crossed,
                                   np.arange(len(cores))) is expected


def test_without_a_vane_file_the_cards_give_the_electrodes(tmp_path):
    lat = _lattice(tmp_path)
    lat.rfq_geom_file = None
    res = Simulation(lat, _beam(n=300), space_charge="off",
                     rfq_geometry="off").run()
    assert res.rfq_td["geometry"] == "RFQ_CELL cards"
    assert res.rfq_td["n_tracked"] == 300
    assert res.rfq_td["n_lost"] < 300


def test_the_cells_vane_profile_is_not_armed_in_time_domain(tmp_path,
                                                            caplog):
    """rfq_geometry="auto" arms the per-cell vane-geometry profile for
    multiparticle runs; with the time-domain solver the cells are not
    tracked, so it is neither armed nor announced."""
    import logging
    lat = _lattice(tmp_path)
    seen = []
    cells = [e for e in lat.elements if type(e).__name__ == "RfqCell"]
    real = type(cells[0]).track_rk4

    def spy(self, beam, ds):
        seen.append(self._geom_z is not None)
        return real(self, beam, ds)

    with caplog.at_level(logging.INFO, logger="linac_gen"):
        type(cells[0]).track_rk4 = spy
        try:
            Simulation(lat, _beam(n=50), space_charge="off").run()
        finally:
            type(cells[0]).track_rk4 = real
    assert seen and not any(seen)          # the ghost reference: no profile
    assert not any("vane-geometry profile" in r.getMessage()
                   for r in caplog.records)


def test_the_time_step_converges_at_second_order(tmp_path):
    """Each particle's exit energy at 48 and 96 steps per period against a
    192-step run: the error falls about fourfold per halving (second-order
    leapfrog; at least threefold here), and at 48 steps it is small
    against the beam's energy spread."""
    W, alive = {}, {}
    for steps in (48, 96, 192):
        lat = _lattice(tmp_path)
        lat.step_config = dataclasses.replace(lat.step_config,
                                              rfq_td_steps_per_period=steps)
        b = _beam(n=600)
        Simulation(lat, b, space_charge="off", rfq_geometry="off").run()
        W[steps] = b.ref.w_kin + b.particles[:, 5]
        alive[steps] = b.alive_mask.copy()
    both = alive[48] & alive[96] & alive[192]
    assert both.sum() >= 20
    err = {s: float(np.sqrt(np.mean((W[s][both] - W[192][both]) ** 2)))
           for s in (48, 96)}
    assert err[96] < err[48] / 3.0, err
    assert err[48] < 0.03 * float(np.std(W[192][both])), err
    assert abs(int(alive[48].sum()) - int(alive[192].sum())) <= 6


# ---- rfq_solver = "auto", the default ------------------------------------

def test_auto_is_the_default(tmp_path):
    from linac_gen.core.step_config import StepConfig
    assert StepConfig().rfq_solver == "auto"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        lat, _ = parse_tracewin(str(_write_rfq(tmp_path)))
    assert lat.step_config.rfq_solver == "auto"


@pytest.mark.slow
def test_auto_runs_the_time_domain_when_it_can(tmp_path, caplog):
    """Bit for bit the explicit time-domain run (the cells' vane profile
    is not armed), without a notice; the record says what tracked the RFQ."""
    import logging
    out = {}
    for solver in ("time_domain", "auto"):
        b = _beam()
        with caplog.at_level(logging.INFO, logger="linac_gen"):
            res = Simulation(_lattice(tmp_path, solver=solver), b,
                             space_charge="off").run()
        out[solver] = (res, b)
    (rt, bt), (ra, ba) = out["time_domain"], out["auto"]
    np.testing.assert_array_equal(ba.particles, bt.particles)
    np.testing.assert_array_equal(ba.lost, bt.lost)
    assert ra.rfq_td["map_key"] == rt.rfq_td["map_key"]
    assert (ra.rfq_solver_used, ra.rfq_solver_note) == ("time_domain", None)
    assert (rt.rfq_solver_used, rt.rfq_solver_note) == ("time_domain", None)
    msgs = [r.getMessage() for r in caplog.records]
    assert not any("RFQ solver auto" in m or "vane-geometry profile" in m
                   for m in msgs)


def _lattice_from_cards(tmp_path, solver):
    """The synthetic RFQ without its RFQ_GEOM line: no vane file, so the
    per-cell model runs without a vane-geometry profile."""
    deck = _write_rfq(tmp_path)
    cards = tmp_path / "cards.dat"
    cards.write_text("\n".join(line for line in deck.read_text().splitlines()
                               if not line.startswith("RFQ_GEOM")) + "\n")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        lat, _ = parse_tracewin(str(cards))
    assert lat.rfq_geom_file is None
    lat.step_config = dataclasses.replace(lat.step_config, rfq_solver=solver)
    return lat


def test_auto_keeps_the_cells_when_the_time_domain_cannot_run(tmp_path,
                                                             caplog):
    """A bunched input beam: bit for bit the explicit per-cell run, with
    one notice per lattice and reason.  (Compared without a vane file:
    per-cell runs with the vane-geometry profile differ from run to run
    at the 1e-7 level whichever solver setting chose them.)"""
    import logging
    bc = _beam(n=200, continuous=False)
    rc = Simulation(_lattice_from_cards(tmp_path, "cells"), bc,
                    space_charge="off").run()
    lat = _lattice_from_cards(tmp_path, "auto")
    ba = _beam(n=200, continuous=False)
    with caplog.at_level(logging.WARNING, logger="linac_gen"):
        ra = Simulation(lat, ba, space_charge="off").run()
        again = Simulation(lat, _beam(n=200, continuous=False),
                           space_charge="off").run()
    np.testing.assert_array_equal(ba.particles, bc.particles)
    np.testing.assert_array_equal(ba.lost, bc.lost)
    assert ra.rfq_solver_used == "cells" and "DC" in ra.rfq_solver_note
    assert again.rfq_solver_used == "cells"
    assert (rc.rfq_solver_used, rc.rfq_solver_note) == ("cells", None)
    assert getattr(ra, "rfq_td", None) is None
    notes = [r.getMessage() for r in caplog.records
             if "RFQ solver auto" in r.getMessage()]
    assert len(notes) == 1 and "DC" in notes[0]


def test_auto_falling_back_arms_the_vane_profile_like_the_cells(tmp_path,
                                                               caplog):
    """With a vane file the per-cell fallback gets the vane-geometry
    profile, as an explicit per-cell run does."""
    import logging
    with caplog.at_level(logging.INFO, logger="linac_gen"):
        rec = Simulation(_lattice(tmp_path, solver="auto"),
                         _beam(n=50, continuous=False),
                         space_charge="off").run()
    assert rec.rfq_solver_used == "cells"
    assert any("vane-geometry profile" in r.getMessage()
               for r in caplog.records)


def test_auto_keeps_the_cells_for_element_hooks(tmp_path):
    """Bunch-train runs (element hooks): auto tracks the RFQ per cell, the
    explicit time-domain choice refuses."""
    def hook(*_a, **_k):
        return None
    ra = Simulation(_lattice(tmp_path, solver="auto"), _beam(n=50),
                    space_charge="off", element_entry_hook=hook).run()
    assert ra.rfq_solver_used == "cells" and "hooks" in ra.rfq_solver_note
    with pytest.raises(RfqTdError, match="hooks"):
        Simulation(_lattice(tmp_path), _beam(n=50), space_charge="off",
                   element_entry_hook=hook).run()


def test_a_tracker_on_its_own_resolves_auto(tmp_path):
    from linac_gen.tracking.tracker import Tracker
    t = Tracker(_lattice(tmp_path, solver="auto"),
                _beam(n=50, continuous=False), pic_solver="off")
    rec = t.run()
    assert t._rfq_td is None
    assert rec.rfq_solver_used == "cells" and "DC" in rec.rfq_solver_note


def test_a_lattice_without_an_rfq_is_untouched():
    from linac_gen.core.lattice import Lattice
    from linac_gen.elements.drift import Drift
    from linac_gen.rfq_td.chain import resolve_rfq_solver
    from linac_gen.tracking.tracker import Tracker
    lat = Lattice()
    lat.add(Drift("D1", length=50.0))
    b = _beam(n=20)
    assert resolve_rfq_solver(lat, b) == (None, None)
    rec = Tracker(lat, b, pic_solver="off").run()
    assert not hasattr(rec, "rfq_solver_used")


def test_the_results_file_records_what_tracked_the_rfq(tmp_path):
    import h5py
    from linac_gen.io.hdf5_output import save_results_hdf5
    lat = _lattice(tmp_path, solver="auto")
    rec = Simulation(lat, _beam(n=50, continuous=False),
                     space_charge="off").run()
    path = tmp_path / "r.h5"
    save_results_hdf5(rec, str(path), lattice=lat)
    with h5py.File(path, "r") as f:
        prov = f["provenance"].attrs
        assert prov["rfq_solver"] == "auto"
        assert prov["rfq_solver_used"] == "cells"
        assert "DC" in prov["rfq_solver_note"]


@pytest.mark.parametrize("solver, says", [("auto", True), ("cells", False)])
def test_backtracking_says_it_undoes_the_rfq_per_cell(tmp_path, solver, says):
    from linac_gen.tracking.backtrack import BacktrackWarning
    lat = _lattice(tmp_path, solver=solver)
    b = _beam(n=50, continuous=False)
    cells = [i for i, e in enumerate(lat.elements)
             if type(e).__name__ == "RfqCell"]
    with warnings.catch_warnings(record=True) as got:
        warnings.simplefilter("always")
        Simulation(lat, b, space_charge="off").run_backtrack(
            entrance_ref=b.ref.copy(), start=cells[0], end=cells[-1],
            approximate_backtracking=True)
    hit = [w for w in got if issubclass(w.category, BacktrackWarning)
           and "forward only" in str(w.message)]
    assert bool(hit) is says


def test_a_fresh_field_map_is_the_same_every_time(tmp_path, monkeypatch):
    """Two builds of one field map (two empty caches) agree to the last
    bit: pyamg's set-up runs under linac_gen.core.amg.seeded."""
    maps = []
    for tag in ("a", "b"):
        monkeypatch.setenv("HELIX_CACHE_DIR", str(tmp_path / f"cache_{tag}"))
        build._MEMORY.clear()
        Simulation(_lattice(tmp_path), _beam(n=20), space_charge="off",
                   rfq_geometry="off").run()
        f, = (tmp_path / f"cache_{tag}" / "rfq_td").glob("map-*.npz")
        with np.load(f) as d:
            maps.append({k: d[k] for k in d.files})
    assert maps[0].keys() == maps[1].keys()
    for k in maps[0]:
        np.testing.assert_array_equal(maps[0][k], maps[1][k], err_msg=k)
