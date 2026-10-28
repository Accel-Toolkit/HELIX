"""The ramp in the turn loop (milestone M3, L2): the beam follows the
reference's program (the same arithmetic as the reference alone), the
normalized emittance is conserved and the beam shrinks as (beta gamma)^-1/2,
the synchrotron tune is the accelerating bucket's, transition is crossed with
its phase jump, a restarted run and any number of threads give the same bits,
a flat ramp is the fixed-energy ring, and what the RF cannot give is capped
or refused."""
from __future__ import annotations

import copy
import json
import math
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from linac_gen.core.config import BeamConfig
from linac_gen.ring import RingConfig, RingSimulation
from linac_gen.ring.programs import Ramp, RampConfig

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "simple_booster.madx"
ROOT = Path(__file__).resolve().parents[2]
GAMMA_T = 5.449167323
#: a generic resonant cycle and voltage table (not any machine's)
SINE = {"kind": "sine", "start_kinetic_mev": 800.0, "end_kinetic_mev": 6000.0,
        "frequency_hz": 15.0, "t_min_s": 0.0005, "t_start_s": 0.0008}
VTABLE = {"kind": "table", "times_s": [0.0, 0.003, 0.03], "values_mv": [0.4, 1.2, 1.2]}


def _beam(**kw):
    d = dict(species="proton", energy=800.0, frequency=44.7044096, current=0.0, emit_nx=1.5,
             emit_ny=1.5, emit_z=3.0, beta_z=100.0, n_particles=300, cutoff=3.0)
    d.update(kw)
    return BeamConfig(**d)


def _ramp(**kw):
    d = {"energy": SINE, "voltage": VTABLE, "t_start_s": 0.004, "gamma_t": GAMMA_T}
    d.update(kw)
    return d


@pytest.fixture(scope="module")
def booster():
    from linac_gen.cli.common import load_lattice
    return load_lattice(str(FIXTURE))


def _run(lat, cfg, **kw):
    run_kw = {k: kw.pop(k) for k in list(kw) if k in ("checkpoint_path", "restart_from")}
    return RingSimulation(copy.deepcopy(lat), kw.pop("beam", None) or _beam(), cfg,
                          **kw).run(**run_kw)


@pytest.fixture(scope="module")
def ramp_run(booster):
    cfg = RingConfig(n_turns=300, n_particles=300, source="matched", tune_probes=1, seed=5,
                     stats_every=10, snapshot_turns=[0, 150, "last"], ramp=_ramp())
    return _run(booster, cfg)


def test_the_beam_follows_the_reference_program(ramp_run, booster):
    res = ramp_run
    r = res.ramp
    assert res.turns_done == 300 and res.survival[-1] == 1.0
    # the reference alone, with the same arithmetic: the same bits
    from linac_gen.ring.compile import compile_from_config
    cfg = RingConfig(ramp=_ramp(), design_energy_mev=float(r["kinetic_mev"][0]))
    rl = compile_from_config(copy.deepcopy(booster), cfg)
    o = Ramp(RampConfig.from_dict(cfg.ramp), rl).orbit(300)
    assert np.array_equal(r["t_s"], o["t_s"]) and np.array_equal(r["phi_deg"], o["phi_deg"])
    assert r["kinetic_end_mev"] == o["energy_end_mev"] - rl.reference.mass
    assert r["t_end_s"] == o["t_end_s"]
    # the bunch stays on the reference: its mean momentum deviation stays small
    # (measured 6e-6)
    assert np.abs(res.mean[:, 5]).max() < 1e-4
    # each statistics row carries its own reference
    g_rows = np.sqrt(1.0 + res.row_beta0gamma0 ** 2)
    assert g_rows[0] < g_rows[-1]
    assert res.meta["final_kinetic_mev"] == pytest.approx(r["kinetic_end_mev"], rel=1e-15)
    # the snapshots carry the reference and the RF of their turn
    ref150 = res.snapshots[150]["reference"]
    assert ref150["kinetic_mev"] > r["kinetic_mev"][0] and ref150["voltage_mv"] > 0.0


def test_adiabatic_damping(ramp_run):
    # no dispersion vertically: the projected emittance is the betatron one;
    # normalized, it is conserved — geometric, it shrinks as 1/(beta gamma)
    res = ramp_run
    ey = res.emit[:, 1]
    bg = res.row_beta0gamma0
    assert (ey[-1] * bg[-1]) / (ey[0] * bg[0]) == pytest.approx(1.0, abs=2e-3)   # (3.3e-4)
    assert ey[-1] / ey[0] == pytest.approx(bg[0] / bg[-1], rel=2e-3)
    assert bg[-1] / bg[0] > 1.02                     # (a ramp that matters: +3 %)
    # the longitudinal emittance in eV s, the adiabatic invariant of the
    # synchrotron motion (measured 3.7e-4)
    el = res.emit_long_evs
    assert el[-1] / el[0] == pytest.approx(1.0, abs=2e-3)


def test_the_synchrotron_tune_is_the_accelerating_buckets(ramp_run, booster):
    from linac_gen.ring.compile import compile_from_config
    from linac_gen.ring.optics import accelerating_bucket, twiss
    res = ramp_run
    cfg = RingConfig(ramp=_ramp(), design_energy_mev=float(res.ramp["kinetic_mev"][0]))
    rl = compile_from_config(copy.deepcopy(booster), cfg)
    ramp = Ramp(RampConfig.from_dict(cfg.ramp), rl)
    pl = ramp.plan(0, *ramp.start())
    o6 = twiss(ramp.stationary_equivalent(pl), six_d=True)
    g = pl.energy_mev / rl.reference.mass
    b = accelerating_bucket(pl.voltage_mv, pl.phi_deg, 84, o6.alpha_c - 1 / g ** 2,
                            math.sqrt(1 - 1 / g ** 2), pl.energy_mev, rl.length)
    # the stationary equivalent's one-turn map (22 discrete cavities, with
    # the dispersion at them) against the smooth formula of the accelerating
    # bucket: 1.2e-3 apart at this voltage (2.5e-4 at the 0.2 MV of the
    # fixed-energy check)
    assert o6.qs == pytest.approx(b["qs"], rel=2e-3)
    assert res.optics["bucket"]["qs"] == pytest.approx(b["qs"], rel=1e-12)
    # tracked: the probe's synchrotron tune over the run, between the
    # bucket's at its start and at its end
    qs_end = accelerating_bucket(res.ramp["voltage_mv"][-1], res.ramp["phi_deg"][-1], 84,
                                 o6.alpha_c - 1 / (1 + res.ramp["kinetic_mev"][-1]
                                                   / rl.reference.mass) ** 2,
                                 math.sqrt(1 - 1 / (1 + res.ramp["kinetic_mev"][-1]
                                                    / rl.reference.mass) ** 2),
                                 res.ramp["kinetic_mev"][-1] + rl.reference.mass,
                                 rl.length)["qs"]
    lo, hi = sorted((b["qs"], qs_end))
    assert lo * 0.98 < res.tunes["qs"] < hi * 1.02


def test_transition_is_crossed_with_the_phase_jump(booster):
    # start a little below transition: the rule's phase jumps to 180 - phi_s
    # on the first turn above gamma_t; the bunch is shortest there
    rc = RampConfig.from_dict(_ramp(t_start_s=0.0))
    from linac_gen.ring.compile import compile_from_config
    rl = compile_from_config(copy.deepcopy(booster), RingConfig(ramp=rc.to_dict()))
    o = Ramp(rc, rl).orbit(Ramp(rc, rl).turns_to_end())
    k_tr = o["transition_turn"]
    t_start = float(o["t_s"][k_tr - 150])
    cfg = RingConfig(n_turns=300, n_particles=300, source="matched", tune_probes=0, seed=2,
                     stats_every=5, ramp=_ramp(t_start_s=t_start))
    res = _run(booster, cfg)
    r = res.ramp
    assert r["transition_turn"] == 150
    assert r["phi_deg"][149] < 90.0 < r["phi_deg"][150]
    assert r["phi_deg"][150] == pytest.approx(180.0 - math.degrees(
        math.asin(r["gain_mev"][150] / r["voltage_mv"][150])), abs=1e-9)
    assert res.survival[-1] == 1.0
    sz = res.sigma[:, 4]
    k_min = int(res.turns[int(np.argmin(sz))])
    assert 100 <= k_min <= 200
    assert sz.min() < 0.97 * min(sz[0], sz[-1])       # (measured: 4.5 % shorter)


def test_restart_continues_a_ramp_bit_for_bit(booster, tmp_path):
    cfg = RingConfig(n_turns=40, n_particles=200, source="matched", tune_probes=1, seed=4,
                     stats_every=5, checkpoint_every=20, ramp=_ramp())
    ck = tmp_path / "ramp.ck.npz"
    full = _run(booster, cfg, checkpoint_path=ck)
    cont = _run(booster, cfg, restart_from=ck)
    assert np.array_equal(full.mean, cont.mean) and np.array_equal(full.cov, cont.cov)
    for k in full.ramp:
        a, b = full.ramp[k], cont.ramp[k]
        if isinstance(a, np.ndarray):
            assert np.array_equal(a, b, equal_nan=True), k
        else:
            assert a == b, k
    assert np.array_equal(full.row_beta0gamma0, cont.row_beta0gamma0)
    # a checkpoint of a ramp is not continued without it, nor the reverse
    with pytest.raises(ValueError, match="ramp"):
        _run(booster, RingConfig(n_turns=40, n_particles=200, source="matched", tune_probes=1,
                                 seed=4, stats_every=5), restart_from=ck)


def test_threads_do_not_change_a_bit(booster):
    pytest.importorskip("numba")
    base = dict(n_turns=12, n_particles=200, source="matched", tune_probes=0, seed=6,
                stats_every=3, ramp=_ramp())
    one = _run(booster, RingConfig(threads=1, **base))
    many = _run(booster, RingConfig(threads=4, **base))
    assert np.array_equal(one.mean, many.mean) and np.array_equal(one.cov, many.cov)


def test_a_flat_ramp_is_the_fixed_energy_ring(booster):
    # no gain, the cavities at their own voltage and lag 0: the ramp machinery
    # (reference changes doing nothing, the tables written every turn) tracks
    # as the fixed-energy ring, to rounding
    lat = copy.deepcopy(booster)
    base = dict(n_turns=30, n_particles=200, source="matched", tune_probes=0, seed=7,
                stats_every=30)
    fixed = _run(lat, RingConfig(**base))
    flat = _run(lat, RingConfig(ramp={"energy": {"kind": "flat"}, "gamma_t": GAMMA_T}, **base))
    assert flat.ramp["gain_mev"].max() == 0.0 and flat.ramp["phi_deg"].max() == 0.0
    np.testing.assert_allclose(flat.mean, fixed.mean, rtol=0, atol=1e-12)
    np.testing.assert_allclose(flat.cov, fixed.cov, rtol=1e-9, atol=1e-18)


@pytest.mark.parametrize("gamma0, phi_s, qs, z_ufp, z_other", [
    (3.0, 30.0, 0.0032729, -8.33333, 4.7699056),         # below transition
    (10.0, 150.0, 0.00176525, 8.33333, -4.76990),        # above: 180 - 30 deg
])
def test_the_accelerating_bucket_against_reference_code_a(gamma0, phi_s, qs, z_ufp, z_other):
    # reference code A's test of a bucket whose reference gains energy: a ring
    # of 1000 m, gamma_t 4, h 40, 100 kV, 50 keV a turn (phi_s 30 deg); its
    # numbers pinned to 1e-7 (Qs) and 1e-4 m (the bucket's ends)
    from linac_gen.ring.optics import accelerating_bucket
    m = 938.27208816                                    # (its proton mass, CODATA 2018)
    eta = 1.0 / 16.0 - 1.0 / gamma0 ** 2
    b = accelerating_bucket(0.1, phi_s, 40, eta, math.sqrt(1.0 - 1.0 / gamma0 ** 2), gamma0 * m,
                            1000.0)
    assert b["stable"] and b["qs"] == pytest.approx(qs, abs=1e-7)
    ends = sorted((z_ufp, z_other))
    assert b["zeta_min"] == pytest.approx(ends[0], abs=1e-4)
    assert b["zeta_max"] == pytest.approx(ends[1], abs=1e-4)
    # the unstable fixed point is the end at pi - phi_s
    assert (b["zeta_min"] if z_ufp < 0 else b["zeta_max"]) == pytest.approx(
        math.radians(phi_s - (180.0 - phi_s)) / (2 * math.pi * 40 / 1000.0), abs=1e-12)


def test_the_accelerating_separatrix_separates(booster):
    # a program gaining 0.1 MeV a turn at 0.2 MV (phi_s 30 deg): particles
    # started at 0.97 of the separatrix of the first turn's bucket stay inside
    # the bucket of every turn for 2.5 synchrotron periods; at 1.03 every one
    # leaves it (on the off-momentum closed orbit transversely)
    from linac_gen.ring.optics import accelerating_bucket, twiss
    ramp = {"energy": {"kind": "table", "times_s": [0.0, 0.001], "values": [800.0, 853.22]},
            "voltage": {"kind": "table", "times_s": [0.0, 1.0], "values_mv": [0.2, 0.2]},
            "gamma_t": GAMMA_T}
    sim = RingSimulation(copy.deepcopy(booster), _beam(), RingConfig(ramp=ramp))
    r = sim.ramp()
    pl = r.plan(0, *r.start())
    assert pl.phi_deg == pytest.approx(30.0, abs=1e-3)
    o4 = twiss(r.rl.with_rf(False), six_d=False, chromaticity=False)
    g = pl.energy_mev / r.mass
    b = accelerating_bucket(pl.voltage_mv, pl.phi_deg, r.h, o4.alpha_c - 1 / g ** 2,
                            math.sqrt(1 - 1 / g ** 2), pl.energy_mev, r.length)
    zz = np.linspace(b["zeta_min"], b["zeta_max"], 22)[1:-1]
    up = b["separatrix"](zz)
    Z, D = np.concatenate([zz, zz]), np.concatenate([up, -up])
    n = int(2.5 / b["qs"])
    inside = {}
    for scale in (0.97, 1.03):
        X = np.zeros((6, Z.size))
        X[4], X[5] = scale * Z, scale * D
        X[0], X[1] = o4.start["dx"] * X[5], o4.start["dpx"] * X[5]
        res = RingSimulation(copy.deepcopy(booster), _beam(), RingConfig(
            n_turns=n, tune_probes=0, stats_every=n, snapshot_turns=list(range(n + 1)),
            ramp=ramp), coords=X).run()
        ok = np.ones(Z.size, dtype=bool)
        for t in range(n + 1):
            sr = res.snapshots[t]["reference"]
            bt = accelerating_bucket(sr["voltage_eff_mv"], sr["phi_eff_deg"], sr["harmonic"],
                                     sr["eta"], sr["beta0"], sr["energy_mev"], sr["length_m"])
            z = res.snapshots[t]["coords"][4]
            ok &= (z >= bt["zeta_min"]) & (z <= bt["zeta_max"])
        inside[scale] = ok
    assert inside[0.97].all() and not inside[1.03].any()


def test_what_the_rf_cannot_give_is_capped_or_refused(booster):
    low = {"kind": "table", "times_s": [0.0], "values_mv": [0.05]}
    cfg = RingConfig(n_turns=5, n_particles=50, source="matched", tune_probes=0, seed=1,
                     ramp=_ramp(voltage=low))
    res = _run(booster, cfg)
    assert res.ramp["n_capped"] == 5 and res.ramp["capped"].all()
    assert np.allclose(res.ramp["phi_deg"], 90.0)
    # refused from the first turn: no run at all
    with pytest.raises(ValueError, match="more than the RF gives"):
        _run(booster, RingConfig(n_turns=5, n_particles=50, source="matched", tune_probes=0,
                                 ramp=_ramp(voltage=low, phase={"on_excess": "refuse"})))
    # the voltage falling to nothing after five turns (each 1.825 us from 4 ms):
    # capped from there, or the run stops there with the turns done
    drop = {"kind": "table", "times_s": [0.0, 0.0040085, 0.0040086, 1.0],
            "values_mv": [1.2, 1.2, 0.0, 0.0]}
    res = _run(booster, RingConfig(n_turns=8, n_particles=50, source="matched", tune_probes=0,
                                   snapshot_turns=[5], ramp=_ramp(voltage=drop)))
    assert list(np.flatnonzero(res.ramp["capped"])) == [5, 6, 7] and res.ramp["n_capped"] == 3
    assert np.array_equal(res.ramp["kinetic_mev"][6:], res.ramp["kinetic_mev"][5:7])  # (no gain)
    # (the snapshot before the turn without voltage: no bucket)
    assert res.snapshots[5]["reference"]["voltage_mv"] == 0.0
    res = _run(booster, RingConfig(n_turns=8, n_particles=50, source="matched", tune_probes=0,
                                   snapshot_turns=[5, "last"], ramp=_ramp(
                                       voltage=drop, phase={"on_excess": "refuse"})))
    assert res.truncated and res.turns_done == 5 and len(res.ramp["t_s"]) == 5
    assert "turn 5" in res.meta["ramp_refused"] and "more than the RF gives" in \
        res.meta["ramp_refused"]
    assert any(n.startswith("stopped: turn 5") for n in res.meta["notes"])
    # (the snapshot after five turns looks ahead to the refused turn: its RF is
    # not a number)
    assert math.isnan(res.snapshots[5]["reference"]["voltage_mv"])
    assert sorted(res.snapshots) == [5]


def test_a_blank_start_is_the_rings_energy_whatever_the_clock(booster):
    # the resonant cycle's start left blank: the ring's reference energy
    # (800 MeV), not the energy the ring is compiled at for a later clock
    given = _ramp(t_start_s=0.004)
    blank = _ramp(t_start_s=0.004, energy=dict(SINE, start_kinetic_mev=None))
    a, b = (_run(booster, RingConfig(n_turns=3, n_particles=20, source="matched",
                                     tune_probes=0, seed=1, ramp=r)) for r in (given, blank))
    assert a.ramp["kinetic_mev"][0] == b.ramp["kinetic_mev"][0] == pytest.approx(923.5528388,
                                                                                 rel=1e-9)
    assert np.array_equal(a.ramp["t_s"], b.ramp["t_s"]) and np.array_equal(a.cov, b.cov)
    assert b.ramp["program"]["energy"]["start_kinetic_mev"] == pytest.approx(800.0, rel=1e-15)


def test_a_long_program_does_not_hold_up_the_results(booster):
    # an energy table ending at 100 s (about 53 million turns of the program's
    # clock): the run's results say None for the turns to the end
    long = {"kind": "table", "times_s": [0.0, 100.0], "values": [800.0, 801.0]}
    res = _run(booster, RingConfig(n_turns=2, n_particles=20, source="matched", tune_probes=0,
                                   ramp=_ramp(energy=long, t_start_s=0.0)))
    assert res.turns_done == 2 and res.ramp["turns_to_end"] is None


def test_ramp_results_round_trip_through_hdf5(ramp_run, tmp_path):
    from linac_gen.ring.results import load_ring_hdf5, save_ring_hdf5
    p = save_ring_hdf5(ramp_run, tmp_path / "ramp.h5")
    back = load_ring_hdf5(p)
    assert np.array_equal(back.row_beta0gamma0, ramp_run.row_beta0gamma0)
    for k in ("t_s", "kinetic_mev", "phi_deg", "voltage_mv", "frequency_hz", "capped"):
        assert np.array_equal(back.ramp[k], ramp_run.ramp[k]), k
    assert back.ramp["transition_turn"] == ramp_run.ramp["transition_turn"]
    assert back.ramp["program"] == ramp_run.ramp["program"]
    assert back.snapshots[150]["reference"]["phi_deg"] == \
        ramp_run.snapshots[150]["reference"]["phi_deg"]
    np.testing.assert_array_equal(back.emit_normalized, ramp_run.emit_normalized)
    assert back.ramp["above_transition_at_start"] is False
    # (what is None is not written: read back as None)
    nogt = copy.copy(ramp_run)
    nogt.ramp = dict(ramp_run.ramp, gamma_t=None)
    assert load_ring_hdf5(save_ring_hdf5(nogt, tmp_path / "nogt.h5")).ramp["gamma_t"] is None


def test_the_cuda_kernel_tracks_a_ramp_as_the_cpu(tmp_path):
    pytest.importorskip("numba")
    code = r"""
import copy, json, sys, numpy as np
sys.path.insert(0, sys.argv[1])
from linac_gen.cli.common import load_lattice
from linac_gen.core.config import BeamConfig
from linac_gen.ring import RingConfig, RingSimulation
lat = load_lattice(sys.argv[2])
beam = BeamConfig(species="proton", energy=800.0, frequency=44.7044096, current=0.0,
                  emit_nx=1.5, emit_ny=1.5, emit_z=3.0, beta_z=100.0, n_particles=16)
ramp = json.loads(sys.argv[3])
out = {}
for dev in ("cpu", "cuda"):
    cfg = RingConfig(n_turns=3, n_particles=16, source="matched", tune_probes=0, seed=3,
                     device=dev, ramp=ramp, snapshot_turns=["last"])
    res = RingSimulation(copy.deepcopy(lat), beam, cfg).run()
    out[dev] = res.snapshots[3]["coords"]
d = np.abs(out["cpu"] - out["cuda"]).max(axis=1) / np.abs(out["cpu"]).max(axis=1)
print("CUDA_RAMP_OK", float(d.max()))
"""
    env = dict(os.environ, NUMBA_ENABLE_CUDASIM="1", OMP_NUM_THREADS="1")
    r = subprocess.run([sys.executable, "-c", code, str(ROOT), str(FIXTURE), json.dumps(_ramp())],
                       cwd=ROOT, env=env, capture_output=True, text=True, timeout=1500)
    assert "CUDA_RAMP_OK" in r.stdout, r.stderr[-3000:]
    assert float(r.stdout.split("CUDA_RAMP_OK")[1]) < 1e-14


def test_a_decelerating_ramp_reports_its_downward_crossing(booster):
    """(transition is the first turn on the other side, up or down: from 6 GeV
    down to 4 GeV the reference crosses it downwards)"""
    from linac_gen.ring.programs import transition_crossing
    assert transition_crossing([True, True, False, False]) == 2
    assert transition_crossing([False, True]) == 1 and transition_crossing([True, True]) is None
    down = {"energy": {"kind": "table", "times_s": [0.0, 1e-3], "values": [6000.0, 4000.0]},
            "voltage": {"kind": "table", "times_s": [0.0, 1.0], "values_mv": [10.0, 10.0]},
            "gamma_t": GAMMA_T}
    ramp = RingSimulation(copy.deepcopy(booster), _beam(energy=6000.0),
                          RingConfig(ramp=down)).ramp()
    n = ramp.turns_to_end()
    o = ramp.orbit(n)
    above = np.asarray(o["above_transition"], dtype=bool)
    assert above[0] and not above[-1] and o["above_transition_at_start"]
    k = o["transition_turn"]
    assert k is not None and above[k - 1] and not above[k]
    assert o["energy_end_mev"] < o["energy_mev"][0]


def test_a_checkpoint_is_not_continued_with_another_ramp(booster, tmp_path):
    cfg = RingConfig(n_turns=4, n_particles=100, source="matched", tune_probes=0,
                     checkpoint_every=2, ramp=_ramp())
    ck = tmp_path / "ck.npz"
    _run(booster, cfg, checkpoint_path=ck)
    other = RingConfig(n_turns=4, n_particles=100, source="matched", tune_probes=0,
                       checkpoint_every=2, ramp=_ramp(gamma_t=GAMMA_T + 0.1))
    with pytest.raises(ValueError, match="followed another ramp"):
        _run(booster, other, restart_from=ck)
