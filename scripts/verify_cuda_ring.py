#!/usr/bin/env python3
"""Ring mode on a CUDA GPU: the validation of a CUDA session (run it on the
machine with the GPU and paste the output back).

    python scripts/verify_cuda_ring.py --quick      (about 5 minutes)
    python scripts/verify_cuda_ring.py --full

On the simple Fermilab Booster (tests/ring/fixtures), CPU against GPU.
The GPU compiler fuses multiply-adds and the GPU's sine, cosine, ... are not
the CPU's, so the two agree to rounding, not bit for bit; the thresholds
below are set from the CPU with fused multiply-adds (numba's ``contract``),
which differs from the plain CPU by 1e-15 in one element, 7e-13 after one
turn of the Booster and 5e-10 after 1000 turns:

* G1 — every node of the ring on its own, then one whole turn, for N
  particles: the largest difference per coordinate over that coordinate's
  largest value (1e-12 per node, 1e-10 per turn); with an aperture, the same
  particles lost at the same node and turn;
* G2 — 200 (``--full``: 1000) turns of small-amplitude particles (with the
  RF bucket): CPU and GPU within 1e-7 of each coordinate's largest value;
* G3 — on the GPU, the same run twice and a run restarted from a checkpoint
  give the same bits (the particles at the last turn, the statistics); the
  one-turn Jacobian (central differences) is symplectic to the
  finite-difference floor on both devices;
* ramp — a bunch through a generic ramp (a 15 Hz cycle from 800 MeV, a
  voltage table) on each device: the reference's record identical bit for
  bit (it is computed on the host), the particles within 1e-7 of each
  coordinate's largest value, the same losses; a restarted ramp on the GPU
  bit for bit;
* space charge — the 2.5-D model (particle-in-cell, 96 kicks per turn) on
  each device from the same matched beam: the particles within 1e-6 of each
  coordinate's largest value after the turns (the field solve is on the host
  for both; the deposit's integers are the same for the same particles, which
  is checked on its own, and so are the line density's bins chosen from
  them), the same run twice and a restarted run on the GPU bit for bit, and
  the time per turn of 5 000 particles on each device;
* injection — an H⁻ beam injected through a foil (stripped, scattered and
  slowed down on the host: the particles inside the foil found on the
  device, those alone and the injected slots moved to the host and back)
  for three quarters of the turns, the sextupoles following a waveform (the
  node table written on the device each turn): the particles within 1e-7 of
  each coordinate's largest value, the injection's record the same, the same
  run twice and a run restarted in the middle of the injection on the GPU
  bit for bit;
* timing — turns of 58 000 particles on the GPU and on every CPU core.

Every launch on the GPU is kept short (linac_gen.ring.kernels.cuda.Launcher):
under Windows a GPU that also drives a display stops a kernel that runs
longer than its timeout (2 s by default, TdrDelay); the report has the
device's timeout setting, its driver model and the longest launch.

The JSON report goes to ``--out`` (default: benchmarks/ring/dev/cuda_sessions/
in a development tree, else ./cuda_sessions/).  Run it from the repository
root with HELIX on the path (PYTHONPATH=.).  ``--tiny`` is the script's own
check with a handful of particles; with NUMBA_ENABLE_CUDASIM=1 it runs in
numba's CUDA simulator, on a machine without a GPU.
"""
from __future__ import annotations

import argparse
import copy
import datetime as _dt
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
DECK = ROOT / "tests" / "ring" / "fixtures" / "simple_booster.madx"
DEV_OUT = ROOT / "benchmarks" / "ring" / "dev"

G1_NODE, G1_TURN, G2_TOL, SYMPLECTIC_TOL = 1e-12, 1e-10, 1e-7, 1e-6
#: space charge: CPU and GPU after the turns, of each coordinate's largest value
SC_TOL = 1e-6
SC = {"intensity": 5e10, "kicks_per_turn": 96, "grid": [64, 64]}
#: injection: CPU and GPU after the turns, of each coordinate's largest value
INJ_TOL = 1e-7
#: a generic resonant cycle and voltage table (as tests/ring/test_ramp.py)
RAMP = {"energy": {"kind": "sine", "start_kinetic_mev": 800.0, "end_kinetic_mev": 6000.0,
                   "frequency_hz": 15.0, "t_min_s": 0.0005, "t_start_s": 0.0008},
        "voltage": {"kind": "table", "times_s": [0.0, 0.003, 0.03], "values_mv": [0.4, 1.2, 1.2]},
        "t_start_s": 0.004, "gamma_t": 5.449167323}


def _ring():
    from linac_gen.cli.common import load_lattice
    from linac_gen.ring.compile import compile_from_config
    from linac_gen.ring.config import RingConfig
    lat = load_lattice(str(DECK))
    return lat, compile_from_config(copy.deepcopy(lat), RingConfig())


def _beam():
    from linac_gen.core.config import BeamConfig
    return BeamConfig(species="proton", energy=800.0, frequency=44.7044096, current=0.0,
                      emit_nx=1.5, emit_ny=1.5, emit_z=5.0, beta_z=100.0, cutoff=4.0)


def _particles(n, seed=3, scale=1.0):
    rng = np.random.default_rng(seed)
    X = rng.normal(0.0, 1.0, (6, n)) * scale * np.array(
        [[2e-3], [6e-5], [1e-3], [2e-4], [0.2], [3e-4]])
    return np.ascontiguousarray(X)


def _arrays(rl, X):
    from linac_gen.ring.optics import _energy
    n = X.shape[1]
    pt, rp, rv = (np.ascontiguousarray(a) for a in _energy(X[5], rl.reference.beta0))
    return [np.array(X, copy=True), pt, rp, rv, np.ones(n, np.int8), np.full(n, -1, np.int32),
            np.full(n, -1, np.int32), np.zeros(n, np.int8)]


def _track(fn, rl, X, turns, wrap):
    a = _arrays(rl, X)
    fn(*a, *rl.node_table(), 0, rl.n_nodes, turns, 0, wrap)
    return a


def _scaled(a, b):
    """Largest |a - b| per coordinate over that coordinate's largest |a|."""
    scale = np.maximum(np.abs(a).max(axis=1), 1e-300)
    return np.abs(a - b).max(axis=1) / scale


def g1(rl, n):
    import dataclasses

    from linac_gen.ring.compile import blocks as B
    from linac_gen.ring.engine import wrap_length
    from linac_gen.ring.kernels import cuda as kc
    from linac_gen.ring.kernels import nodes as N
    from linac_gen.ring.kernels.cpu import backend
    be = backend(allow_python=False)
    L = wrap_length(rl) or 0.0
    X = _particles(n)
    # every node on its own, with the particles as they arrive there (CPU)
    worst, worst_node = 0.0, None
    A = _arrays(rl, X)
    for j in range(rl.n_nodes):
        cpu = [v.copy() for v in A]
        be.track_turns(*cpu, *rl.node_table(), j, j + 1, 1, 0, 0.0)
        gpu = [v.copy() for v in A]
        kc.track_turns(*gpu, *rl.node_table(), j, j + 1, 1, 0, 0.0)
        alive = cpu[4] == 1
        if alive.any():
            d = float(_scaled(cpu[0][:, alive], gpu[0][:, alive]).max())
            if d > worst:
                worst, worst_node = d, j
        A = cpu
    out = {"n": n, "per_node_max": worst,
           "per_node_worst": None if worst_node is None else {
               "node": worst_node, "kind": int(rl.kinds[worst_node])}}
    # one whole turn
    a = _track(be.track_turns, rl, X, 1, L)
    b = _track(kc.track_turns, rl, X, 1, L)
    per = _scaled(a[0], b[0])
    out["one_turn_max_per_coordinate"] = [float(v) for v in per]
    out["energy_helpers_max_rel"] = float(max(np.abs(a[k] / b[k] - 1).max() for k in (1, 2, 3)))
    # an aperture of 2 mm after node 10: the same particles lost, same node and turn
    j = 10
    kinds = np.insert(rl.kinds, j + 1, N.N_APERTURE)
    params = np.concatenate([rl.params, B.aperture("circle", 2e-3)])
    offsets = np.insert(rl.offsets, j + 1, rl.params.size)
    rla = dataclasses.replace(rl, kinds=kinds, offsets=offsets, params=params,
                              ref_beta0=np.insert(rl.ref_beta0, j + 1, rl.ref_beta0[0]),
                              ref_p0c=np.insert(rl.ref_p0c, j + 1, rl.ref_p0c[0]),
                              ref_bg2=np.insert(rl.ref_bg2, j + 1, rl.ref_bg2[0]))
    a = _track(be.track_turns, rla, X, 3, L)
    b = _track(kc.track_turns, rla, X, 3, L)
    same = (np.array_equal(a[4], b[4]) and np.array_equal(a[5], b[5])
            and np.array_equal(a[6], b[6]))
    out.update({"aperture_lost": int((a[4] == 0).sum()), "same_losses": bool(same)})
    out["pass"] = bool(worst <= G1_NODE and max(per) <= G1_TURN and same)
    return out


def g2(rl, n, turns):
    from linac_gen.ring.engine import wrap_length
    from linac_gen.ring.kernels import cuda as kc
    from linac_gen.ring.kernels.cpu import PARALLEL_LOCK, backend
    be = backend(allow_python=False)
    L = wrap_length(rl) or 0.0
    X = _particles(n, seed=5, scale=0.05)
    with PARALLEL_LOCK:
        a = _track(be.track_turns_parallel, rl, X, turns, L)
    b = _track(kc.track_turns, rl, X, turns, L)
    worst = float(_scaled(a[0], b[0]).max())
    same_lost = bool(np.array_equal(a[4], b[4]))
    return {"n": n, "turns": turns, "max_rel_to_coordinate_scale": worst,
            "same_lost": same_lost, "pass": bool(worst <= G2_TOL and same_lost)}


def g3(lat, n, turns, tmp):
    from linac_gen.ring import RingConfig, RingSimulation
    cfg = RingConfig(n_turns=turns, n_particles=n, source="matched", tune_probes=1,
                     device="cuda", checkpoint_every=max(1, turns // 2), seed=7,
                     snapshot_turns=["last"])
    a = RingSimulation(copy.deepcopy(lat), _beam(), cfg).run(checkpoint_path=tmp / "ck.npz")
    b = RingSimulation(copy.deepcopy(lat), _beam(), cfg).run()
    c = RingSimulation(copy.deepcopy(lat), _beam(), cfg).run(restart_from=tmp / "ck.npz")

    def same(r, s):
        return {"particles": bool(np.array_equal(r.snapshots[turns]["coords"],
                                                 s.snapshots[turns]["coords"])),
                "statistics": bool(np.array_equal(r.cov, s.cov)
                                   and np.array_equal(r.mean, s.mean)),
                "losses": bool(np.array_equal(r.losses["pid"], s.losses["pid"]))}
    run_to_run, restart = same(a, b), same(a, c)
    return {"n": n, "turns": turns, "run_to_run_bitwise": run_to_run,
            "restart_bitwise": restart,
            "pass": all(run_to_run.values()) and all(restart.values())}


def ramp(lat, n, turns, tmp):
    """A bunch through the generic ramp on the CPU and on the GPU; a restarted
    ramp on the GPU."""
    from linac_gen.ring import RingConfig, RingSimulation
    base = dict(n_turns=turns, n_particles=n, source="matched", tune_probes=0, seed=11,
                snapshot_turns=["last"], ramp=RAMP, stats_every=max(1, turns // 10))
    gpu_cfg = RingConfig(device="cuda", checkpoint_every=max(1, turns // 2), **base)
    cpu = RingSimulation(copy.deepcopy(lat), _beam(), RingConfig(device="cpu", **base)).run()
    gpu = RingSimulation(copy.deepcopy(lat), _beam(), gpu_cfg).run(
        checkpoint_path=tmp / "ramp_ck.npz")
    again = RingSimulation(copy.deepcopy(lat), _beam(), gpu_cfg).run(
        restart_from=tmp / "ramp_ck.npz")
    same_ref = all(np.array_equal(cpu.ramp[k], gpu.ramp[k])
                   for k in ("t_s", "kinetic_mev", "phi_deg", "voltage_mv", "frequency_hz"))
    same_lost = bool(np.array_equal(cpu.losses["pid"], gpu.losses["pid"]))
    a, b = cpu.snapshots[turns]["coords"], gpu.snapshots[turns]["coords"]
    worst = float(_scaled(a, b).max()) if a.shape == b.shape else float("inf")
    restart = bool(np.array_equal(gpu.snapshots[turns]["coords"],
                                  again.snapshots[turns]["coords"])
                   and np.array_equal(gpu.cov, again.cov))
    return {"n": n, "turns": turns, "kinetic_mev": [float(cpu.ramp["kinetic_mev"][0]),
                                                    float(cpu.ramp["kinetic_end_mev"])],
            "reference_bitwise": same_ref, "max_rel_to_coordinate_scale": worst,
            "same_lost": same_lost, "restart_bitwise": restart,
            "pass": bool(same_ref and same_lost and restart and worst <= G2_TOL)}


def space_charge(lat, n, turns, tmp):
    """The 2.5-D model on the CPU and on the GPU; run to run and a restart on
    the GPU; the deposit on the same particles on both."""
    from linac_gen.ring import RingConfig, RingSimulation
    base = dict(n_turns=turns, n_particles=n, source="matched", tune_probes=1, seed=13,
                snapshot_turns=["last"], space_charge=SC, stats_every=max(1, turns // 10))
    gpu_cfg = RingConfig(device="cuda", checkpoint_every=max(1, turns // 2), **base)
    cpu = RingSimulation(copy.deepcopy(lat), _beam(), RingConfig(device="cpu", **base)).run()
    gpu = RingSimulation(copy.deepcopy(lat), _beam(), gpu_cfg).run(
        checkpoint_path=tmp / "sc_ck.npz")
    again = RingSimulation(copy.deepcopy(lat), _beam(), gpu_cfg).run(
        restart_from=tmp / "sc_ck.npz")
    twice = RingSimulation(copy.deepcopy(lat), _beam(), RingConfig(device="cuda", **base)).run()
    a, b = cpu.snapshots[turns]["coords"], gpu.snapshots[turns]["coords"]
    worst = float(_scaled(a, b).max()) if a.shape == b.shape else float("inf")
    restart = bool(np.array_equal(gpu.snapshots[turns]["coords"],
                                  again.snapshots[turns]["coords"])
                   and np.array_equal(gpu.cov, again.cov))
    repeat = bool(np.array_equal(gpu.snapshots[turns]["coords"],
                                 twice.snapshots[turns]["coords"]))
    deposit = _deposit_same(lat, min(n, 4096))
    # (the line density's bins, "auto": chosen from the deposit's integers —
    # the same on both)
    bins = (cpu.meta["space_charge"]["z_bins_used"] == gpu.meta["space_charge"]["z_bins_used"]
            == again.meta["space_charge"]["z_bins_used"])
    return {"n": n, "turns": turns, "kicks_per_turn": gpu.meta["space_charge"]["kicks"],
            "max_rel_to_coordinate_scale": worst, "restart_bitwise": restart,
            "run_to_run_bitwise": repeat, "deposit_bitwise": deposit,
            "z_bins_used": gpu.meta["space_charge"]["z_bins_used"], "z_bins_same": bool(bins),
            "cpu_s_per_turn": cpu.wall_s / turns, "gpu_s_per_turn": gpu.wall_s / turns,
            "pass": bool(worst <= SC_TOL and restart and repeat and deposit and bins)}


def _deposit_same(lat, n):
    """The deposit of the same particles on the CPU and on the GPU: the same
    integers (grid, line density, particles off the grid)."""
    from numba import cuda

    from linac_gen.ring.compile import compile_from_config
    from linac_gen.ring.config import RingConfig
    from linac_gen.ring.sc.beams import CpuBeam, CudaBeam
    from linac_gen.ring.sc.solver import (
        SpaceCharge,
        _power_of_two_scale,
        combine_moments,
        half_widths,
        make_box,
    )
    rl = compile_from_config(copy.deepcopy(lat), RingConfig())
    sc = SpaceCharge(SC, rl, zwrap=rl.length / 84, rf_on=True)
    X = _particles(n, seed=5)
    X[4] = np.clip(X[4], -0.49 * rl.length / 84, 0.49 * rl.length / 84)
    w = np.full(n, 1e7)
    st = np.ones(n, np.int8)
    cb = CpuBeam(X.copy(), np.zeros(n), np.ones(n), np.ones(n), st.copy(), w)
    m = combine_moments(cb.moments(0.0, 0.0), 0.0, 0.0)
    box = make_box(64, 64, m["x"], m["y"], *half_widths(m, 4.0))
    kp, ki = sc._params(0, box, (m["x"], m["y"]), m["w"])
    s = _power_of_two_scale(m["w"])
    g1_, l1, o1 = cb.deposit(kp, ki, s, True)
    dev = [cuda.to_device(np.ascontiguousarray(v)) for v in
           (X.copy(), np.zeros(n), np.ones(n), np.ones(n), st.copy())]
    g2_, l2, o2 = CudaBeam(dev, w).deposit(kp, ki, s, True)
    return bool(np.array_equal(g1_, g2_) and np.array_equal(l1, l2) and o1 == o2)


def _injection(turns, per_turn):
    t_end = turns * 474.202752 / (0.841810706 * 299792458.0)
    return {"foil": {"at": "foil", "thickness_ug_cm2": 600.0, "x_min_m": -0.01,
                     "x_max_m": 0.01, "y_min_m": -0.01, "y_max_m": 0.01},
            "source": {"kind": "gaussian", "species": "H-", "kinetic_mev": 800.0,
                       "frequency_mhz": 162.5, "emit_nx_um": 0.3, "emit_ny_um": 0.3,
                       "beta_x_m": 30.0, "beta_y_m": 5.0, "sigma_dp": 1e-4,
                       "sigma_z_m": 0.002},
            # (three quarters of the turns: the checkpoint at half of them is in the
            # middle of the injection, the ring partly filled)
            "turns": max(2, (3 * turns) // 4), "particles_per_turn": per_turn, "current_ma": 2.0,
            "magnets": [{"name": "sextupoles", "elements": ["sxsxx"], "model": "scale",
                         "waveform": {"kind": "linear", "t_start_s": 0.0, "t_end_s": t_end,
                                      "start": 1.0, "end": 0.5}}]}


def injection(lat, per_turn, turns, tmp):
    """Injection through a foil on the CPU and on the GPU; run to run and a
    restart on the GPU from the middle of the injection."""
    from linac_gen.elements.marker import Marker
    from linac_gen.ring import RingConfig, RingSimulation
    ring = copy.deepcopy(lat)
    ring.elements.insert(1, Marker(name="foil"))
    base = dict(n_turns=turns, source="empty", injection=_injection(turns, per_turn),
                tune_probes=1, seed=17, snapshot_turns=["last"])
    gpu_cfg = RingConfig(device="cuda", checkpoint_every=max(1, turns // 2), **base)
    cpu = RingSimulation(copy.deepcopy(ring), _beam(), RingConfig(device="cpu", **base)).run()
    gpu = RingSimulation(copy.deepcopy(ring), _beam(), gpu_cfg).run(
        checkpoint_path=tmp / "inj_ck.npz")
    again = RingSimulation(copy.deepcopy(ring), _beam(), gpu_cfg).run(
        restart_from=tmp / "inj_ck.npz")
    twice = RingSimulation(copy.deepcopy(ring), _beam(),
                           RingConfig(device="cuda", **base)).run()
    a, b = cpu.snapshots[turns]["coords"], gpu.snapshots[turns]["coords"]
    worst = float(_scaled(a, b).max()) if a.shape == b.shape and a.size else float("inf")
    record = all(np.array_equal(cpu.injection[k], gpu.injection[k])
                 for k in ("injected", "stripped", "h0", "hminus", "missed", "foil_hits"))
    restart = bool(np.array_equal(gpu.snapshots[turns]["coords"],
                                  again.snapshots[turns]["coords"])
                   and np.array_equal(gpu.injection["foil_hits"], again.injection["foil_hits"]))
    repeat = bool(np.array_equal(gpu.snapshots[turns]["coords"],
                                 twice.snapshots[turns]["coords"]))
    return {"per_turn": per_turn, "turns": turns, "injected": int(gpu.ledger["injected"]),
            "circulating": int(gpu.ledger["circulating"]),
            "max_rel_to_coordinate_scale": worst, "record_same": bool(record),
            "restart_bitwise": restart, "run_to_run_bitwise": repeat,
            "cpu_s_per_turn": cpu.wall_s / turns, "gpu_s_per_turn": gpu.wall_s / turns,
            "pass": bool(worst <= INJ_TOL and record and restart and repeat)}


def symplecticity(rl):
    """|J^T S J - S| of the one-turn map, J by central differences, tracked on
    each device (the finite-difference floor is about 1e-8)."""
    from linac_gen.ring.engine import wrap_length
    from linac_gen.ring.kernels import cuda as kc
    from linac_gen.ring.kernels.cpu import backend
    from linac_gen.ring.optics import R_STEPS
    from linac_gen.ring.optics.normal_form import symplectic_form
    be = backend(allow_python=False)
    L = wrap_length(rl) or 0.0
    S = symplectic_form(6)
    out = {}
    for name, fn in (("cpu", be.track_turns), ("cuda", kc.track_turns)):
        X = np.zeros((6, 13))
        for k in range(6):
            X[k, 1 + 2 * k] += 10 * R_STEPS[k]
            X[k, 2 + 2 * k] -= 10 * R_STEPS[k]
        a = _track(fn, rl, X, 1, L)
        J = np.empty((6, 6))
        for k in range(6):
            J[:, k] = (a[0][:, 1 + 2 * k] - a[0][:, 2 + 2 * k]) / (20 * R_STEPS[k])
        out[name] = float(np.abs(J.T @ S @ J - S).max())
    out["pass"] = bool(max(out["cpu"], out["cuda"]) < SYMPLECTIC_TOL)
    return out


def timing(rl, n, turns):
    from linac_gen.ring.engine import wrap_length
    from linac_gen.ring.kernels import cuda as kc
    from linac_gen.ring.kernels.cpu import PARALLEL_LOCK, backend
    be = backend(allow_python=False)
    L = wrap_length(rl) or 0.0
    X = _particles(n)
    out = {"n": n, "turns": turns}
    with PARALLEL_LOCK:
        _track(be.track_turns_parallel, rl, X[:, :64], 1, L)
        t = time.perf_counter()
        _track(be.track_turns_parallel, rl, X, turns, L)
        out["cpu_s_per_turn"] = (time.perf_counter() - t) / turns
    # (the particles on the device; one turn first, which also sizes the launches)
    kc.kernel()
    dev = [kc.cuda.to_device(np.ascontiguousarray(v)) for v in _arrays(rl, X)]
    table = [kc.cuda.to_device(np.ascontiguousarray(v)) for v in rl.node_table()]
    ln = kc.Launcher(dev, table, n, 0, rl.n_nodes, L)
    ln.track(0, 1)
    t = time.perf_counter()
    ln.track(1, turns)
    out["cuda_s_per_turn"] = (time.perf_counter() - t) / turns
    out["speedup"] = out["cpu_s_per_turn"] / out["cuda_s_per_turn"]
    out["cuda_launches"] = ln.launches
    out["cuda_longest_launch_s"] = round(ln.longest_s, 4)
    return out


def _device_report(cuda) -> dict:
    d = cuda.get_current_device()
    name = d.name.decode() if isinstance(d.name, bytes) else str(d.name)
    rep = {"name": name, "compute_capability": list(d.compute_capability)}
    for key, attr in (("multiprocessors", "MULTIPROCESSOR_COUNT"),
                      ("clock_khz", "CLOCK_RATE"),
                      ("kernel_exec_timeout", "KERNEL_EXEC_TIMEOUT"),
                      ("tcc_driver", "TCC_DRIVER")):
        try:
            rep[key] = int(getattr(d, attr))
        except Exception:                                           # noqa: BLE001
            rep[key] = None
    return rep


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--quick", action="store_true")
    g.add_argument("--full", action="store_true")
    g.add_argument("--tiny", action="store_true",
                   help="a handful of particles and turns: the script's own check (in numba's "
                        "CUDA simulator when NUMBA_ENABLE_CUDASIM=1 is set)")
    ap.add_argument("--out", default=str(DEV_OUT / "cuda_sessions" if DEV_OUT.is_dir()
                                         else Path.cwd() / "cuda_sessions"))
    a = ap.parse_args(argv)
    full = bool(a.full)
    tiny = bool(a.tiny)
    from linac_gen.ring.kernels import cuda as kc
    report = {"date": _dt.datetime.now().isoformat(timespec="seconds"),
              "mode": "full" if full else ("tiny" if tiny else "quick"),
              "python": platform.python_version(),
              "machine": platform.platform(),
              "thresholds": {"g1_per_node": G1_NODE, "g1_one_turn": G1_TURN, "g2": G2_TOL,
                             "symplecticity": SYMPLECTIC_TOL, "space_charge": SC_TOL,
                             "injection": INJ_TOL},
              "max_launch_s": kc.MAX_LAUNCH_S}
    try:
        import numba
        from numba import cuda
        report["numba"] = numba.__version__
        report["cuda_available"] = bool(cuda.is_available())
        if report["cuda_available"]:
            if getattr(numba.config, "ENABLE_CUDASIM", 0):
                report["device"] = {"name": "numba CUDA simulator"}
            else:
                report["device"] = _device_report(cuda)
    except Exception as exc:                                        # noqa: BLE001
        report["cuda_available"] = False
        report["error"] = f"{type(exc).__name__}: {exc}"
    if not report.get("cuda_available") or not kc.available():
        print(json.dumps(report, indent=1))
        print("no CUDA device: nothing to verify (install numba with CUDA support and the "
              "NVIDIA driver)")
        return 2
    lat, rl = _ring()
    import tempfile
    tmp = Path(tempfile.mkdtemp(prefix="helix_cuda_"))
    size = {"full": (2000, 2048, 1000, 20000, 50, 58000, 5, 20000, 300, 5000, 100, 100, 100),
            "quick": (512, 2048, 200, 2000, 10, 58000, 2, 2000, 50, 5000, 20, 50, 20),
            "tiny": (3, 2, 2, 3, 2, 3, 1, 3, 2, 40, 2, 3, 2)}[report["mode"]]
    n1, n2, t2, n3, t3, n4, t4, n5, t5, n6, t6, n7, t7 = size
    steps = [("G1 every node and one turn, CPU vs GPU", lambda: g1(rl, n1)),
             ("G2 small-amplitude particles", lambda: g2(rl, n2, t2)),
             ("G3 run to run and restart on the GPU", lambda: g3(lat, n3, t3, tmp)),
             ("symplecticity of the one-turn map", lambda: symplecticity(rl)),
             ("ramp, CPU vs GPU and restart", lambda: ramp(lat, n5, t5, tmp)),
             ("space charge, CPU vs GPU and restart", lambda: space_charge(lat, n6, t6, tmp)),
             ("injection, CPU vs GPU and restart", lambda: injection(lat, n7, t7, tmp)),
             ("timing", lambda: timing(rl, n4, t4))]
    ok = True
    for name, fn in steps:
        t = time.perf_counter()
        try:
            r = fn()
        except Exception as exc:                                    # noqa: BLE001
            r = {"error": f"{type(exc).__name__}: {exc}", "pass": False}
        r["wall_s"] = round(time.perf_counter() - t, 2)
        report[name] = r
        ok &= bool(r.get("pass", True))
        print(f"{name}: {json.dumps(r)}", flush=True)
    report["all_pass"] = ok
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{_dt.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}_{report['mode']}.json"
    path.write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(f"{'ALL PASS' if ok else 'SOMETHING FAILED'} — report: {path}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
