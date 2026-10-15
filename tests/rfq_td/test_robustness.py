"""Time-domain RFQ: phases, edge cases, the field-map cache and lock,
validation at run start, and the run paths that must keep the choice."""
from __future__ import annotations

import dataclasses
import logging
import math
import subprocess
import sys

import numpy as np
import pytest

pytest.importorskip("pyamg")

from linac_gen.core.simulation import Simulation
from linac_gen.rfq_td import build
from linac_gen.rfq_td.chain import RfqTdError
from tests.rfq_td.test_chain import _beam, _lattice, _write_rfq


@pytest.fixture(autouse=True)
def _coarse_map(monkeypatch):
    monkeypatch.setitem(build.DEFAULT_SETTINGS, "h", 0.2)
    monkeypatch.setitem(build.DEFAULT_SETTINGS, "x_max", 10.0)
    monkeypatch.setitem(build.DEFAULT_SETTINGS, "pad_mm", 10.0)
    monkeypatch.setitem(build.DEFAULT_SETTINGS, "upstream_mm", 12.0)
    monkeypatch.setitem(build.DEFAULT_SETTINGS, "downstream_mm", 6.0)


def _cell_rows(res, lat):
    names = list(res.element_names)
    return [names.index(e.name) for e in lat.elements
            if type(e).__name__ == "RfqCell"]


# ── phases ────────────────────────────────────────────────────────────────
@pytest.mark.parametrize("periodic", [False, True])
def test_one_phase_shift_for_every_particle_and_plane(tmp_path, monkeypatch,
                                                      periodic):
    """A particle that leaves the RFQ late keeps its lag at the exit
    (periodic_phase off: unfolded, as in a per-cell run; on: folded by the
    tracker) and is NOT shifted by whole periods at the cells before --
    one bucket index per particle, taken at the exit, once put particles
    tens of thousands of degrees off at every earlier cell."""
    from linac_gen.rfq_td import tracker as tdt
    real = tdt.TimeDomainTracker.run
    late = {}

    def run(self, *a, **k):
        res = real(self, *a, **k)
        out = np.flatnonzero(np.isfinite(res.rec[-1, :, 4]))
        i = int(out[0])
        late["i"] = i
        res.rec[-1, i, 4] += np.float32(5.0 * self.T)    # 5 periods late
        return res

    def go(patch):
        lat = _lattice(tmp_path)
        b = _beam(n=200)
        b.periodic_phase = periodic
        if patch:
            monkeypatch.setattr(tdt.TimeDomainTracker, "run", run)
        try:
            res = Simulation(lat, b, space_charge="off",
                             rfq_geometry="off").run()
        finally:
            monkeypatch.setattr(tdt.TimeDomainTracker, "run", real)
        return lat, b, res

    lat, b0, r0 = go(False)
    lat, b1, r1 = go(True)
    rows = _cell_rows(r1, lat)
    # every cell before the last: the same rows (to the float32 rounding of
    # the modified time, which moves the circular-mean origin by ~1e-6 deg)
    for key in ("sigma_phi", "sigma_x", "emit_z"):
        np.testing.assert_allclose(np.asarray(getattr(r1, key))[rows[:-1]],
                                   np.asarray(getattr(r0, key))[rows[:-1]],
                                   rtol=1e-6)
    i = late["i"]
    lag = b1.particles[i, 4] - b0.particles[i, 4]
    if periodic:
        assert abs(lag) < 0.01
    else:
        assert lag == pytest.approx(5 * 360.0, abs=0.01)


# ── edge cases ────────────────────────────────────────────────────────────
def test_no_particle_alive_at_the_rfq(tmp_path):
    lat = _lattice(tmp_path)
    b = _beam(n=50)
    b.lost[:] = True
    res = Simulation(lat, b, space_charge="off", rfq_geometry="off").run()
    assert res.rfq_td["n_tracked"] == 0
    lat_c = _lattice(tmp_path, solver="cells")
    b2 = _beam(n=50)
    b2.lost[:] = True
    res_c = Simulation(lat_c, b2, space_charge="off", rfq_geometry="off").run()
    np.testing.assert_array_equal(np.asarray(res.ref_w_kin),
                                  np.asarray(res_c.ref_w_kin))


def test_exit_energy_offset_is_recorded(tmp_path):
    res = Simulation(_lattice(tmp_path), _beam(n=200), space_charge="off",
                     rfq_geometry="off").run()
    assert isinstance(res.rfq_td["exit_w_offset_kev"], float)


# ── validation at run start ───────────────────────────────────────────────
def _run_start_error(lat, beam=None):
    """The error raised by Tracker.run before any element is tracked."""
    from linac_gen.tracking.tracker import Tracker
    t = Tracker(lat, beam or _beam(n=20), pic_solver="off")
    with pytest.raises(RfqTdError) as ei:
        t.run()
    assert len(t.recorder.s) == 0           # nothing recorded, not even INPUT
    return str(ei.value)


def test_asymmetric_vanes_are_refused_at_start(tmp_path):
    lat = _lattice(tmp_path)
    vane = np.loadtxt(lat.rfq_geom_file)
    vane[:, 9] *= 1.01                      # pole 3 (-y) 1 % wider
    np.savetxt(lat.rfq_geom_file, vane, fmt="%.10f")
    assert "poles differ" in _run_start_error(lat)


def test_tip_radii_and_voltages_of_opposite_poles_are_checked(tmp_path):
    lat = _lattice(tmp_path)
    vane = np.loadtxt(lat.rfq_geom_file)
    vane[:, 14] *= 1.2                      # pole 4 (-x) tip radius
    np.savetxt(lat.rfq_geom_file, vane, fmt="%.10f")
    assert "tip radii" in _run_start_error(lat)
    lat = _lattice(tmp_path)
    vane = np.loadtxt(lat.rfq_geom_file)
    vane[:, 11] *= 0.9                      # pole 3 (-y) voltage
    np.savetxt(lat.rfq_geom_file, vane, fmt="%.10f")
    assert "voltages" in _run_start_error(lat)


def test_rounding_level_pole_differences_are_accepted(tmp_path):
    lat = _lattice(tmp_path)
    vane = np.loadtxt(lat.rfq_geom_file)
    vane[:, 9] += 2e-7                      # 0.2 um on pole 3 (file in m)
    np.savetxt(lat.rfq_geom_file, vane, fmt="%.10f")
    res = Simulation(lat, _beam(n=50), space_charge="off",
                     rfq_geometry="off").run()
    assert res.rfq_td["n_tracked"] == 50


def test_too_few_cells_and_upstream_buncher_are_refused_at_start(tmp_path):
    from linac_gen.elements.rf_gap import RFGap
    lat = _lattice(tmp_path)
    cells = [i for i, e in enumerate(lat.elements)
             if type(e).__name__ == "RfqCell"]
    short = dataclasses.replace(lat.step_config)
    lat2 = _lattice(tmp_path)
    del lat2.elements[cells[2]:cells[-1] + 1]
    lat2.step_config = short
    assert "at least 3" in _run_start_error(lat2)
    lat.elements.insert(cells[0] - 1, RFGap("BUNCHER", voltage=0.01,
                                            phase=-90.0, frequency=162.5))
    assert "BUNCHER" in _run_start_error(lat)


def test_a_bunched_beam_is_refused_at_start(tmp_path):
    assert "starts bunched" in _run_start_error(_lattice(tmp_path),
                                                _beam(continuous=False))


# ── field-map cache and lock ──────────────────────────────────────────────
def _electrodes_and_edges(tmp_path):
    from linac_gen.rfq_td.chain import RfqTdDriver
    from linac_gen.tracking.tracker import Tracker
    lat = _lattice(tmp_path)
    drv = RfqTdDriver(Tracker(lat, _beam(n=10), pic_solver="off"))
    el, _ = drv._geom[0]
    cells = [e for e in lat.elements if type(e).__name__ == "RfqCell"]
    return el, np.concatenate([[0.0], np.cumsum([c.length for c in cells])])


def _dead_pid():
    p = subprocess.Popen([sys.executable, "-c", "pass"])
    p.wait()
    return p.pid


def test_a_stale_lock_of_a_dead_process_is_taken_over(tmp_path, caplog):
    import socket
    el, edges = _electrodes_and_edges(tmp_path)
    key = build.map_key(el, edges, dict(build.DEFAULT_SETTINGS))
    path = build._map_path(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    lock = path.with_suffix(path.suffix + ".lock")
    lock.write_text(f"{_dead_pid()}@{socket.gethostname()}")
    with caplog.at_level(logging.WARNING, logger="linac_gen.rfq_td"):
        fmap = build.get_field_map(el, edges)
    assert fmap.key == key and path.exists() and not lock.exists()
    assert any("stale field-map lock" in r.getMessage()
               for r in caplog.records)


def test_waiting_on_a_live_lock_honours_stop(tmp_path):
    import os
    import socket

    from linac_gen.rfq_td.tracker import Aborted
    el, edges = _electrodes_and_edges(tmp_path)
    key = build.map_key(el, edges, dict(build.DEFAULT_SETTINGS))
    path = build._map_path(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    lock = path.with_suffix(path.suffix + ".lock")
    lock.write_text(f"{os.getpid()}@{socket.gethostname()}")   # alive
    with pytest.raises(Aborted):
        build.get_field_map(el, edges, should_abort=lambda: True)
    assert lock.exists()                    # someone else's: left alone


def test_a_corrupt_cached_map_is_rebuilt(tmp_path, monkeypatch):
    el, edges = _electrodes_and_edges(tmp_path)
    key = build.map_key(el, edges, dict(build.DEFAULT_SETTINGS))
    path = build._map_path(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"not a map")
    calls = []
    real = build._build
    monkeypatch.setattr(build, "_build",
                        lambda *a, **k: calls.append(1) or real(*a, **k))
    fmap = build.get_field_map(el, edges)
    assert calls == [1] and fmap.key == key
    build._MEMORY.clear()
    assert build.RfqFieldMap.load(path).key == key     # replaced on disk


def test_a_cached_map_is_loaded_from_disk_without_a_build(tmp_path,
                                                          monkeypatch):
    el, edges = _electrodes_and_edges(tmp_path)
    first = build.get_field_map(el, edges)
    build._MEMORY.clear()

    def no_build(*a, **k):
        raise AssertionError("rebuilt a cached map")

    monkeypatch.setattr(build, "_build", no_build)
    again = build.get_field_map(el, edges)
    assert again is not first and again.key == first.key
    np.testing.assert_array_equal(again.phi, first.phi)
    assert build.map_cached(el, edges)


def test_an_unconverged_window_is_not_cached(tmp_path, monkeypatch):
    el, edges = _electrodes_and_edges(tmp_path)
    real = build._solve_task

    def bad(task):
        z_first, hz, phi, info = real(task)
        info["rel_residual"] = 1e-3
        return z_first, hz, phi, info

    monkeypatch.setattr(build, "_solve_task", bad)
    with pytest.raises(RuntimeError, match="did not converge"):
        build.get_field_map(el, edges)
    key = build.map_key(el, edges, dict(build.DEFAULT_SETTINGS))
    assert not build._map_path(key).exists()


@pytest.mark.slow
def test_stop_during_a_parallel_build(tmp_path, monkeypatch):
    """The process-pool build ends on Stop too (and caches nothing)."""
    from linac_gen.rfq_td.tracker import Aborted
    monkeypatch.setenv("HELIX_RFQTD_WORKERS", "2")
    el, edges = _electrodes_and_edges(tmp_path)
    seen = []

    def stop():
        seen.append(1)
        return len(seen) > 2

    with pytest.raises(Aborted):
        build.get_field_map(el, edges, should_abort=stop)
    key = build.map_key(el, edges, dict(build.DEFAULT_SETTINGS))
    assert not build._map_path(key).exists()
    assert not build._map_path(key).with_suffix(".npz.lock").exists()


def test_the_cache_is_pruned_to_its_budget(tmp_path, monkeypatch):
    import os
    import time
    d = build.cache_dir()
    d.mkdir(parents=True, exist_ok=True)
    old = []
    for i in range(3):
        f = d / f"map-old{i}.npz"
        f.write_bytes(b"x" * 1000)
        os.utime(f, (time.time() - 1000 + i, time.time() - 1000 + i))
        old.append(f)
    keep = d / "map-new.npz"
    keep.write_bytes(b"x" * 1000)
    monkeypatch.setenv("HELIX_RFQTD_CACHE_GB", str(2500 / 1024 ** 3))
    build._prune(keep)
    assert keep.exists() and old[2].exists()
    assert not old[0].exists() and not old[1].exists()


# ── space-charge Green's-function cache ───────────────────────────────────
def test_greens_cache_stays_within_its_byte_budget():
    from linac_gen.rfq_td.space_charge import PeriodicTrainSpaceCharge
    lam = 299.792458 / 0.1625
    rng = np.random.default_rng(8)
    n = 5000
    x, y, z = (rng.normal(0, 1, n), rng.normal(0, 1, n),
               rng.normal(0, 2.0, n))
    out = {}
    for budget in (40 * 1024 ** 2, 4 * 1024 ** 3):
        sc = PeriodicTrainSpaceCharge(1e-15, lam, nx=32, cache_bytes=budget)
        fields = []
        for beta in np.linspace(0.01, 0.03, 12):
            fields.append(sc.field(x, y, z, np.full(n, beta)))
        assert sc._cache_nbytes <= budget
        assert sc._cache_nbytes == sum(g.nbytes for g, _ in
                                       sc._cache.values())
        out[budget] = fields
    for a, b in zip(*out.values()):
        for u, v in zip(a, b):
            np.testing.assert_array_equal(u, v)


# ── run paths that must keep the solver choice ────────────────────────────
def _td_lattice():
    from linac_gen.core.lattice import Lattice
    from linac_gen.elements.drift import Drift
    lat = Lattice()
    lat.add(Drift("D", length=100.0))
    lat.step_config = dataclasses.replace(lat.step_config,
                                          rfq_solver="time_domain",
                                          rfq_td_steps_per_period=60)
    return lat


def _capture_run_mp_sim(monkeypatch, *modules):
    seen = []

    def fake(lattice, cfg, sc, step, **kw):
        seen.append(step)
        raise StopIteration

    for m in modules:
        monkeypatch.setattr(m, "run_mp_sim", fake)
    return seen


def test_failure_study_and_compensation_keep_the_choice(monkeypatch):
    from linac_gen.core.config import BeamConfig
    from linac_gen.failures import compensation, study
    seen = _capture_run_mp_sim(monkeypatch, study, compensation)
    cfg = BeamConfig(species="proton", energy=3.0, frequency=352.21,
                     current=0.0, n_particles=10)
    fs = study.FailureStudy(lattice=_td_lattice(), beam_config=cfg,
                            mode="mp")
    with pytest.raises(StopIteration):
        fs._inmem_metrics([])
    with pytest.raises(StopIteration):
        compensation._forward_metrics(_td_lattice(), cfg, "mp")
    assert [(s.rfq_solver, s.rfq_td_steps_per_period) for s in seen] \
        == [("time_domain", 60)] * 2
    assert all(s.integration_steps_per_metre == 100.0 for s in seen)


def test_the_matchers_default_step_config_keeps_the_choice(monkeypatch):
    from linac_gen.cli import common
    from linac_gen.core.config import BeamConfig
    from linac_gen.matching import engine
    seen = _capture_run_mp_sim(monkeypatch, common)
    cfg = BeamConfig(species="proton", energy=3.0, frequency=352.21,
                     current=0.0, n_particles=10)
    with pytest.raises(StopIteration):
        engine._run_mp(_td_lattice(), cfg, space_charge=False,
                       n_particles=10, seed=1)
    assert seen[0].rfq_solver == "time_domain"


# ── provenance ────────────────────────────────────────────────────────────
def test_provenance_records_the_settings_the_run_used(tmp_path):
    """A change of the live lattice's StepConfig after the run (the GUI
    combo mid-run) does not rewrite what the results say."""
    import h5py

    from linac_gen.core.config import BeamConfig
    from linac_gen.distributions.factory import create_beam
    from linac_gen.io.hdf5_output import save_results_hdf5
    lat = _td_lattice()
    cfg = BeamConfig(species="proton", energy=3.0, frequency=352.21,
                     current=0.0, n_particles=20)
    res = Simulation(lat, create_beam(cfg, seed=1), space_charge="off").run()
    lat.step_config = dataclasses.replace(lat.step_config, rfq_solver="cells")
    out = tmp_path / "r.h5"
    save_results_hdf5(res, str(out), beam_config=cfg, lattice=lat)
    with h5py.File(out, "r") as f:
        assert f["provenance"].attrs["rfq_solver"] == "time_domain"


# ── buckets at the exit ───────────────────────────────────────────────────
@pytest.mark.parametrize("side", ["up", "down"])
def test_bucket_periods_follow_the_injection_phase(side):
    """The time-domain bucket need not line up with the DC injection
    window: captured particles land in the main bucket or a neighbour
    according to their injection phase (one boundary).  Each particle gets
    the period its injection phase implies -- captured, straggling or lost
    -- so the captured leave as one bunch and the others keep their lag."""
    from linac_gen.rfq_td.chain import RfqTdDriver
    rng = np.random.default_rng(11)
    n = 4000
    phi = rng.uniform(-180.0, 180.0, n)
    b = 80.0 if side == "up" else -120.0
    k = np.where(phi >= b, 1, 0) if side == "up" else np.where(phi < b, -1, 0)
    r_main = -1                            # the bunch slips one period
    d = 360.0 * (r_main + k) + rng.normal(0.0, 8.0, n)
    # a few boundary mix-ups, stragglers and lost particles
    near = np.flatnonzero(np.abs(phi - b) < 2.0)[:5]
    d[near] += 360.0 * np.sign(b - phi[near])    # into the other bucket
    strag = rng.choice(n, 30, replace=False)
    d[strag] = 221 * 360.0 + rng.normal(0.0, 50.0, 30)
    lost = rng.choice(np.setdiff1d(np.arange(n), strag), 200, replace=False)
    d[lost] = np.nan
    per = RfqTdDriver._bucket_periods(phi, d)
    want = r_main + k
    others = np.setdiff1d(np.arange(n), near)
    # stragglers and lost particles: the injection-phase rule
    np.testing.assert_array_equal(per[strag], want[strag])
    np.testing.assert_array_equal(per[lost], want[lost])
    np.testing.assert_array_equal(per[others], want[others])
    cap = np.isfinite(d) & ~np.isin(np.arange(n), strag)
    exit_phase = d - 360.0 * per
    # every captured particle, the boundary mix-ups too, in one bunch
    assert np.abs(exit_phase[cap]).max() < 60.0
    assert np.all(exit_phase[strag] > 200 * 360.0)         # keep their lag


def test_first_cell_rows_are_folded_in_periodic_runs(tmp_path):
    """With periodic_phase the first cell's rows are folded too (the
    tracker folds a bunch train from the next element on)."""
    lat = _lattice(tmp_path)
    b = _beam(n=300)
    b.periodic_phase = True
    snap = {}
    from linac_gen.rfq_td import chain as ch
    real = ch.RfqTdDriver._load_plane

    def spy(self, run, p):
        real(self, run, p)
        if p == int(run.exit_idx[0]):
            al = self.tr.beam.alive_mask
            snap["phi"] = self.tr.beam.particles[al, 4].copy()

    ch.RfqTdDriver._load_plane = spy
    try:
        Simulation(lat, b, space_charge="off", rfq_geometry="off").run()
    finally:
        ch.RfqTdDriver._load_plane = real
    assert np.abs(snap["phi"]).max() <= 180.0


def test_an_rfq_upstream_is_refused_at_start(tmp_path):
    from types import SimpleNamespace

    from linac_gen.elements.vane_rfq import VaneRFQ
    lat = _lattice(tmp_path)
    v = object.__new__(VaneRFQ)
    v.name = "VRFQ"
    v.length = 10.0
    v.cells = [SimpleNamespace(voltage_V=6e4)]
    cells = [i for i, e in enumerate(lat.elements)
             if type(e).__name__ == "RfqCell"]
    lat.elements.insert(cells[0] - 1, v)
    assert "VRFQ" in _run_start_error(lat)


# ── cache and lock, continued ─────────────────────────────────────────────
def test_only_the_owner_removes_a_lock(tmp_path):
    import socket
    lock = tmp_path / "m.npz.lock"
    lock.write_text(f"{_dead_pid()}@{socket.gethostname()}")
    build._unlock(lock)                       # someone else's: kept
    assert lock.exists()
    import os
    lock.write_text(f"{os.getpid()}@{socket.gethostname()}")
    build._unlock(lock)
    assert not lock.exists()


def test_a_waiter_loads_the_map_once_it_is_complete(tmp_path):
    """Waiting for a lock whose owner cannot remove it (Windows: file in
    use) ends as soon as the finished map loads."""
    import os
    import socket
    el, edges = _electrodes_and_edges(tmp_path)
    fmap = build.get_field_map(el, edges)           # builds and caches
    build._MEMORY.clear()
    key = fmap.key
    path = build._map_path(key)
    lock = path.with_suffix(path.suffix + ".lock")
    lock.write_text(f"{os.getpid()}@{socket.gethostname()}")   # live owner
    # a fresh map file appears while the lock stays
    got = build._lock(path, timeout_s=30.0,
                      ready=lambda: build._try_load(path, key) is not None)
    assert got is None
    lock.unlink()


def test_an_unwritable_cache_is_refused_at_start(tmp_path, monkeypatch):
    import os
    import stat
    ro = tmp_path / "ro"
    ro.mkdir()
    monkeypatch.setenv("HELIX_CACHE_DIR", str(ro))
    os.chmod(ro, stat.S_IRUSR | stat.S_IXUSR)
    try:
        if os.access(ro, os.W_OK):
            pytest.skip("cannot make a read-only directory here (root?)")
        msg = _run_start_error(_lattice(tmp_path))
    finally:
        os.chmod(ro, stat.S_IRWXU)
    assert "cannot be cached" in msg
