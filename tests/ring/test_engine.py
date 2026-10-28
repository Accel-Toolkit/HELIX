"""The ring turn loop (milestone M2, fixed energy): statistics, sources,
stop, probes and their NAFF tunes, the RF bucket, losses, zeta wrapping,
checkpoint/restart bit for bit, and what is refused."""
from __future__ import annotations

import copy
import dataclasses
from pathlib import Path

import numpy as np
import pytest

from linac_gen.core.config import BeamConfig
from linac_gen.elements.aperture import Aperture
from linac_gen.elements.rf_gap import RFGap
from linac_gen.io.madx_parser import parse_madx
from linac_gen.ring import RingConfig, RingSimulation
from linac_gen.ring.engine import ENGINE, load_checkpoint, wrap_length

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "simple_booster.madx"


def _beam(**kw):
    d = dict(species="proton", energy=800.0, frequency=44.7044096, current=0.0, emit_nx=1.5,
             emit_ny=1.5, emit_z=5.0, beta_z=100.0, n_particles=500, cutoff=4.0)
    d.update(kw)
    return BeamConfig(**d)


@pytest.fixture(scope="module")
def booster():
    # (through HELIX's loader, which keeps the deck's BEAM on the lattice —
    # the beam its magnets were set for — as the CLI and the GUI do)
    from linac_gen.cli.common import load_lattice
    return load_lattice(str(FIXTURE))


@pytest.fixture(scope="module")
def booster_no_rf(booster):
    lat = copy.deepcopy(booster)
    for e in lat.elements:
        if isinstance(e, RFGap):
            e.voltage = 0.0
    return lat


_RUN_KW = ("checkpoint_path", "restart_from", "progress_callback", "should_abort",
           "live_callback")


def _run(lat, cfg, beam=None, **kw):
    run_kw = {k: kw.pop(k) for k in list(kw) if k in _RUN_KW}
    return RingSimulation(copy.deepcopy(lat), beam or _beam(), cfg, **kw).run(**run_kw)


def test_turns_statistics_snapshots_and_ledger(booster):
    cfg = RingConfig(n_turns=12, n_particles=300, source="matched", snapshot_turns=[0, 5, 12],
                     tune_probes=0, seed=3)
    res = _run(booster, cfg)
    assert res.engine == ENGINE and not res.truncated and res.turns_done == 12
    assert list(res.turns) == list(range(13))
    assert list(res.n_alive) == [300] * 13
    assert res.mean.shape == (13, 6) and res.cov.shape == (13, 6, 6)
    assert sorted(res.snapshots) == [0, 5, 12]
    assert res.snapshots[5]["coords"].shape == (6, 300)
    assert sorted(res.snapshots[12]["pid"].tolist()) == list(range(300))
    assert res.ledger["injected"] == 300 and res.ledger["circulating"] == 300
    assert res.meta["device"] == "cpu" and res.meta["rf_on"] is True
    assert res.meta["zeta_wrap_m"] == pytest.approx(res.meta["length_m"] / 84, rel=1e-15)
    # the statistics are those of the snapshot taken at the same turn
    X = res.snapshots[12]["coords"]
    assert np.array_equal(res.mean[-1], X.mean(axis=1))


def test_the_run_stops_between_turns(booster):
    seen = []
    res = RingSimulation(copy.deepcopy(booster), _beam(),
                         RingConfig(n_turns=1000, n_particles=50, source="matched",
                                    tune_probes=0)).run(
        progress_callback=lambda t, n: seen.append((t, n)), should_abort=lambda: len(seen) >= 7)
    assert res.truncated and res.turns_done == 7
    assert seen == [(t, 1000) for t in range(1, 8)]


def test_the_lattice_given_is_the_only_one_touched(booster):
    before = [(type(e).__name__, e.name, e.length) for e in booster.elements]
    _run(booster, RingConfig(n_turns=2, n_particles=20, source="matched", tune_probes=0))
    assert [(type(e).__name__, e.name, e.length) for e in booster.elements] == before


def _periodic_beam(lat, **kw):
    """The Beam tab's beam with the ring's periodic Twiss and dispersion, as
    the Ring tab's 'Use the matched beam in the Beam tab' sets them (HELIX
    units)."""
    from linac_gen.ring.compile import compile_from_config
    from linac_gen.ring.optics import twiss
    rl = compile_from_config(copy.deepcopy(lat), RingConfig())
    o = twiss(rl, chromaticity=False)
    per_mev = 1e3 / (o.beta0 * rl.reference.p0c)
    st = o.start
    return _beam(alpha_x=st["alfx"], beta_x=st["betx"], alpha_y=st["alfy"], beta_y=st["bety"],
                 disp_x=st["dx"] * per_mev, disp_xp=st["dpx"] * per_mev, **kw)


@pytest.mark.parametrize("beta_x_scale, bmag", [(1.0, 1.0), (1.4, 1.0571)])
def test_the_beam_tab_beam_in_ring_coordinates(booster_no_rf, beta_x_scale, bmag):
    # the Beam tab's distribution, converted to ring coordinates: with the
    # ring's periodic Twiss and dispersion (as "Use the matched beam" sets them)
    # its betatron moments match the ring (BMAG 1 to the sampling noise);
    # with beta_x 40 % off, BMAG = (1.4 + 1/1.4)/2
    from linac_gen.ring.compile import compile_from_config
    from linac_gen.ring.optics import twiss
    b = _periodic_beam(booster_no_rf, n_particles=20000, emit_z=0.5, beta_z=2000.0)
    b = dataclasses.replace(b, beta_x=b.beta_x * beta_x_scale)
    res = _run(booster_no_rf, RingConfig(n_turns=3, source="beam", tune_probes=0, seed=7), b)
    assert list(res.n_alive) == [20000] * 4
    st = twiss(compile_from_config(copy.deepcopy(booster_no_rf), RingConfig()),
               chromaticity=False).start
    T = np.eye(6)
    T[0, 5], T[1, 5] = -st["dx"], -st["dpx"]
    S = T @ res.cov[0] @ T.T                     # betatron moments (dispersion removed)
    e = np.sqrt(S[0, 0] * S[1, 1] - S[0, 1] ** 2)
    beta, alpha = S[0, 0] / e, -S[0, 1] / e
    g1, g2 = (1 + alpha ** 2) / beta, (1 + st["alfx"] ** 2) / st["betx"]
    bm = 0.5 * (beta * g2 + st["betx"] * g1 - 2 * alpha * st["alfx"])
    assert bm == pytest.approx(bmag, abs=0.005)


def test_a_matched_beam_holds_its_emittances(booster):
    # 6-D matched Gaussian (the RF bucket of the deck): rms emittances flat to
    # the sampling noise over 200 turns
    res = _run(booster, RingConfig(n_turns=200, n_particles=4000, source="matched",
                                   tune_probes=0, stats_every=10, seed=11))
    e = res.emit
    assert np.abs(e / e[0] - 1.0).max() < 0.03


def test_probe_tunes_are_the_one_turn_maps(booster, booster_no_rf):
    # one small-amplitude probe, 1000 turns, NAFF: with the RF on, the 6-D
    # eigentunes (the transverse ones differ from the 4-D tunes by the
    # synchro-betatron coupling of dispersion at the cavities) and Qs; with
    # it off, the 4-D tunes — far inside the plan's 1e-4
    res = _run(booster, RingConfig(n_turns=1000, n_particles=4, source="matched",
                                   stats_every=100))
    q6 = res.optics["eigen_tunes_6d"]
    assert res.tunes["qx"] == pytest.approx(q6[0], abs=1e-8)
    assert res.tunes["qy"] == pytest.approx(q6[1], abs=1e-8)
    assert res.tunes["qs"] == pytest.approx(q6[2], abs=1e-7)
    assert res.tunes["qs"] == pytest.approx(res.optics["qs"], abs=1e-7)
    assert abs(q6[0] - res.optics["qx"] % 1.0) > 1e-6          # (the coupling is real)
    res4 = _run(booster_no_rf, RingConfig(n_turns=1000, n_particles=4, source="matched",
                                          stats_every=100))
    assert res4.tunes["qx"] == pytest.approx(res4.optics["qx"] % 1.0, abs=1e-8)
    assert res4.tunes["qy"] == pytest.approx(res4.optics["qy"] % 1.0, abs=1e-8)
    assert "qs" not in res4.tunes


def test_a_footprint_of_probes(booster_no_rf):
    # a 3 x 3 grid of probe amplitudes: tunes and the tune diffusion of each;
    # the sextupoles move the tunes with amplitude
    res = _run(booster_no_rf, RingConfig(n_turns=256, n_particles=4, source="matched",
                                         tune_probes=3, probe_max_sigma=4.0, stats_every=64))
    q = res.probes["tunes"]
    assert q.shape == (9, 3) and np.all(np.isfinite(q[:, :2]))
    assert np.all(np.isfinite(res.probes["diffusion"]))
    assert np.ptp(q[:, 0]) > 1e-6
    assert res.probes["amplitudes_sigma"].shape == (9, 2)


def test_zeta_is_wrapped_into_the_rf_period(booster):
    # particles outside the bucket slip through it turn after turn: wrapped,
    # zeta stays in [-L/2, L/2) while unwrapped it runs over several periods;
    # the motion is the same (zeta to a whole number of periods L)
    from linac_gen.ring.compile import compile_from_config
    from linac_gen.ring.optics import rf_bucket, twiss
    rl = compile_from_config(copy.deepcopy(booster), RingConfig())
    b = rf_bucket(rl, twiss(rl, chromaticity=False))
    X = np.zeros((6, 4))
    X[5] = np.array([-3.0, -2.0, 2.0, 3.0]) * b["delta_max"]
    n = 40
    runs = {}
    for wrap in ("auto", "off"):
        cfg = RingConfig(n_turns=n, tune_probes=0, zeta_wrap=wrap,
                         snapshot_turns=list(range(n + 1)))
        runs[wrap] = RingSimulation(copy.deepcopy(booster), _beam(), cfg, coords=X).run()
    period = runs["auto"].meta["zeta_wrap_m"]
    assert period == pytest.approx(runs["auto"].meta["length_m"] / 84, rel=1e-15)
    assert runs["off"].meta["zeta_wrap_m"] is None

    def coord(wrap, i):
        return np.array([runs[wrap].snapshots[t]["coords"][i] for t in range(n + 1)])
    za, zo = coord("auto", 4), coord("off", 4)
    assert np.all(za >= -period / 2) and np.all(za < period / 2)
    assert np.abs(zo).max() > 2 * period             # (several periods: really wrapped)
    k = (zo - za) / period
    assert np.abs(k - np.round(k)).max() < 1e-9
    for i in (0, 1, 2, 3, 5):
        assert np.allclose(coord("auto", i), coord("off", i), rtol=1e-8, atol=1e-15), i


def test_wrap_length_follows_the_harmonics(booster, booster_no_rf):
    from linac_gen.ring.compile import compile_from_config
    rl = compile_from_config(copy.deepcopy(booster), RingConfig())
    assert wrap_length(rl) == pytest.approx(rl.length / 84, rel=1e-15)
    rl0 = compile_from_config(copy.deepcopy(booster_no_rf), RingConfig())
    assert wrap_length(rl0) == pytest.approx(rl0.length, rel=1e-15)


def test_the_separatrix_separates(booster):
    # stationary bucket: started at zeta = 0 just inside the separatrix the
    # particle oscillates (delta changes sign), just outside it does not
    from linac_gen.ring.compile import compile_from_config
    from linac_gen.ring.optics import rf_bucket, twiss
    lat = copy.deepcopy(booster)
    rl = compile_from_config(lat, RingConfig())
    o = twiss(rl, chromaticity=False)
    b = rf_bucket(rl, o)
    co = np.zeros(6)
    X = np.zeros((6, 2))
    X[5] = [0.97 * b["delta_max"], 1.03 * b["delta_max"]]
    X += co[:, None]
    n = int(3 / b["qs"])
    res = RingSimulation(lat, _beam(), RingConfig(n_turns=n, tune_probes=0, stats_every=1,
                                                  snapshot_turns=list(range(n + 1))),
                         coords=X).run()
    d = np.array([res.snapshots[t]["coords"][5] for t in range(n + 1)])
    assert d[:, 0].min() < 0.0 < d[:, 0].max()               # inside: rotates
    assert np.all(d[:, 1] > 0.0)                               # outside: never turns back


def test_losses_at_an_aperture(booster_no_rf):
    lat = copy.deepcopy(booster_no_rf)
    lat.elements.insert(20, Aperture(name="tight", dx=3.0, dy=3.0, aperture_type=Aperture.CIRCULAR))
    res = _run(lat, RingConfig(n_turns=5, n_particles=2000, source="matched", tune_probes=0,
                               seed=5))
    L = res.losses
    n_lost = L["pid"].size
    assert 0 < n_lost < 2000
    assert res.ledger["lost"] == n_lost and res.ledger["circulating"] == 2000 - n_lost
    assert res.n_alive[-1] == 2000 - n_lost and np.all(np.diff(res.n_alive) <= 0)
    names = res.meta["element_names"]
    assert {names[i] for i in L["element"]} == {"tight"}
    assert np.all(L["cause"] == 1) and np.all((L["turn"] >= 0) & (L["turn"] < 5))
    r = np.hypot(L["coords"][0], L["coords"][2])
    assert np.all(r > 3e-3)


def test_checkpoint_restart_is_bitwise(booster, tmp_path):
    cfg = RingConfig(n_turns=12, n_particles=200, source="matched", tune_probes=1,
                     snapshot_turns=[0, 4, 11, 12], checkpoint_every=5, seed=9)
    ck = tmp_path / "ring_checkpoint.npz"
    full = _run(booster, cfg, checkpoint_path=ck)
    assert load_checkpoint(ck)["turn"] == 10
    again = _run(booster, cfg, restart_from=ck)
    for name in ("turns", "n_alive", "mean", "cov"):
        assert np.array_equal(getattr(full, name), getattr(again, name), equal_nan=True), name
    assert sorted(full.snapshots) == sorted(again.snapshots)
    for t in full.snapshots:
        assert np.array_equal(full.snapshots[t]["coords"], again.snapshots[t]["coords"])
    assert np.array_equal(full.probes["tbt"], again.probes["tbt"], equal_nan=True)
    assert full.tunes == again.tunes
    # a checkpoint of another ring is refused
    other = copy.deepcopy(booster)
    for e in other.elements:
        if type(e).__name__ == "Quadrupole":
            e.gradient *= 1.001
            break
    with pytest.raises(ValueError, match="another ring"):
        _run(other, cfg, restart_from=ck)


def test_one_thread_and_all_give_the_same_run(booster):
    a = _run(booster, RingConfig(n_turns=4, n_particles=300, source="matched", threads=1,
                                 tune_probes=0))
    b = _run(booster, RingConfig(n_turns=4, n_particles=300, source="matched", tune_probes=0))
    assert np.array_equal(a.cov, b.cov) and np.array_equal(a.mean, b.mean)


def test_an_unstable_bucket_is_refused(tmp_path):
    # the deck at 6 GeV keeps the 800 MeV RF phase, now on the wrong side of
    # transition: refused with the reason, not tracked
    deck = tmp_path / "booster_6gev.madx"
    deck.write_text(FIXTURE.read_text().replace("energy=0.8+pmass", "energy=6.0+pmass"))
    lat, meta = parse_madx(str(deck))
    with pytest.raises(ValueError, match="above it at 180 deg"):
        RingSimulation(lat, _beam(energy=meta["reference"].w_kin),
                       RingConfig(n_turns=10, source="matched")).run()


def test_an_accelerating_phase_is_refused(booster):
    lat = copy.deepcopy(booster)
    for e in lat.elements:
        if isinstance(e, RFGap):
            e.phase = -60.0
    with pytest.raises(ValueError, match="fixed energy"):
        _run(lat, RingConfig(n_turns=3, source="matched"))


def test_a_beam_at_another_energy_is_refused(booster):
    # a beam read from a file keeps its own reference (a linac distribution
    # at 2.1 MeV, say): it must be the ring's
    from linac_gen.distributions.factory import create_beam
    other = create_beam(BeamConfig(species="H-", energy=2.1226695, frequency=162.5,
                                   n_particles=100), seed=1)
    with pytest.raises(ValueError, match="is not the ring's"):
        RingSimulation(copy.deepcopy(booster), _beam(), RingConfig(n_turns=3),
                       beam=other).run()


def test_an_empty_lattice_is_refused():
    from linac_gen.core.lattice import Lattice
    with pytest.raises(ValueError, match="no elements"):
        RingSimulation(Lattice(), _beam(), RingConfig(n_turns=3)).run()


def test_the_loaders_keep_the_beam_its_magnets_were_scaled_with():
    # (the CLI and the assistant load through cli.common.load_lattice; the GUI
    # through its own wrapper, tested in tests/gui/test_ring_tab.py; the core
    # importers return the lattice exactly as before)
    from linac_gen.cli.common import load_lattice
    from linac_gen.io.formats import parse_lattice_file
    assert not hasattr(parse_lattice_file(str(FIXTURE))[0], "ring_reference_beam")
    lat = load_lattice(str(FIXTURE))
    rb = lat.ring_reference_beam
    assert rb["species"] == "proton" and rb["energy"] == pytest.approx(800.0, rel=1e-12)
    assert rb["declared"] is True             # (its BEAM gives the energy)
    assert rb["frequency"] == pytest.approx(44.7044096, rel=1e-8)


def test_g0_torch_default_dtype_changes_nothing(booster):
    # (G0) the ring engine works in explicit float64: torch's default dtype
    # set to float32 changes no bit of a run
    torch = pytest.importorskip("torch")
    cfg = RingConfig(n_turns=3, n_particles=100, source="matched", tune_probes=0, seed=2)
    a = _run(booster, cfg)
    old = torch.get_default_dtype()
    torch.set_default_dtype(torch.float32)
    try:
        b = _run(booster, cfg)
    finally:
        torch.set_default_dtype(old)
    assert np.array_equal(a.cov, b.cov) and np.array_equal(a.mean, b.mean)


def test_a_cuda_run_in_the_simulator_matches_the_cpu():
    # (G1 on the Mac) the device-resident CUDA path, run by numba's CUDA
    # simulator in a fresh process: the same particles, turns and losses as
    # the CPU to rounding
    import os
    import subprocess
    import sys
    root = Path(__file__).resolve().parents[2]
    code = f"""
import copy, numpy as np
from linac_gen.core.config import BeamConfig
from linac_gen.io.madx_parser import parse_madx
from linac_gen.ring import RingConfig, RingSimulation
lat, _ = parse_madx({str(FIXTURE)!r})
b = BeamConfig(species="proton", energy=800.0, frequency=44.7044096, current=0.0, emit_nx=1.5,
               emit_ny=1.5, emit_z=5.0, beta_z=100.0, cutoff=4.0)
out = {{}}
for dev in ("cpu", "cuda"):
    cfg = RingConfig(n_turns=2, n_particles=6, source="matched", tune_probes=1, device=dev,
                     snapshot_turns=[2], seed=4)
    r = RingSimulation(copy.deepcopy(lat), b, cfg).run()
    out[dev] = (r.snapshots[2]["coords"], r.probes["tbt"], r.meta["device"])
a, b2 = out["cpu"], out["cuda"]
assert b2[2] == "cuda", b2[2]
d1 = np.abs(a[0] - b2[0]).max() / np.abs(a[0]).max()
d2 = np.nanmax(np.abs(a[1] - b2[1])) / np.nanmax(np.abs(a[1]))
print("CUDASIM_RUN", d1, d2, flush=True)
"""
    env = dict(os.environ)
    env.update({"NUMBA_ENABLE_CUDASIM": "1", "PYTHONPATH": str(root), "OMP_NUM_THREADS": "1"})
    r = subprocess.run([sys.executable, "-c", code], cwd=root, env=env, capture_output=True,
                       text=True, timeout=1800)
    line = [ln for ln in r.stdout.splitlines() if ln.startswith("CUDASIM_RUN")]
    assert line, (r.returncode, r.stderr[-3000:])
    d1, d2 = (float(v) for v in line[-1].split()[1:])
    assert d1 <= 1e-15 and d2 <= 1e-15


def test_a_matched_beam_has_the_emittances_asked_for(booster, booster_no_rf):
    # cut at 3 sigma (rms 0.949 of the uncut), then set: the normal-mode
    # emittances are the Beam tab's exactly, with and without RF; without RF
    # delta has the Beam tab's rms energy spread sqrt(emit_z (1 + alpha_z^2)/beta_z)
    import math

    from linac_gen.ring.compile import compile_from_config
    b = _beam(cutoff=3.0, alpha_z=2.0)
    on = _run(booster, RingConfig(n_turns=1, n_particles=3000, tune_probes=0,
                                  snapshot_turns=[0]), b)
    off = _run(booster_no_rf, RingConfig(n_turns=1, n_particles=3000, tune_probes=0), b)
    bg = on.meta["beta0gamma0"]
    ref = compile_from_config(copy.deepcopy(booster), RingConfig()).reference
    ez = 5.0 * ref.beta0 * 299792458.0 / (360.0 * 44.7044096e6) / (ref.beta0 ** 2 * ref.energy0)
    assert np.allclose(np.array(on.meta["eps0_normal_modes"]) / [1.5e-6 / bg, 1.5e-6 / bg, ez],
                       1.0, rtol=1e-12, atol=0)
    assert np.allclose(np.array(off.meta["eps0_normal_modes"][:2]) * bg, 1.5e-6, rtol=1e-12)
    sd = math.sqrt(5.0 * (1 + 2.0 ** 2) / 100.0) / (ref.beta0 ** 2 * ref.energy0)
    assert math.sqrt(off.cov[0][5, 5]) == pytest.approx(sd, rel=1e-12)
    assert abs(off.mean[0][5]) < 1e-12 * sd
    # the cut stays a cut: in each normal mode every particle within about
    # 3/sqrt(0.949) = 3.08 rms (the setting mixes the modes by ~1/sqrt(n));
    # uncut, 3000 particles would reach 4.3
    from linac_gen.ring.optics import twiss
    o = twiss(compile_from_config(copy.deepcopy(booster), RingConfig()), six_d=True,
              chromaticity=False)
    Z = np.linalg.inv(np.asarray(o.W)) @ (on.snapshots[0]["coords"]
                                          - np.asarray(o.closed_orbit)[:, None])
    eps = on.meta["eps0_normal_modes"]
    r = np.array([np.hypot(Z[2 * k], Z[2 * k + 1]) / math.sqrt(eps[k]) for k in range(3)])
    assert 2.9 < r.max() < 3.3


def test_restart_after_every_particle_is_lost_is_bitwise(booster_no_rf, tmp_path):
    # the beam dies in the first turn and the checkpoint is written at that
    # turn: continued, the run ends where the uninterrupted one ends
    lat = copy.deepcopy(booster_no_rf)
    lat.elements.insert(20, Aperture(name="shut", dx=1e-6, dy=1e-6,
                                     aperture_type=Aperture.CIRCULAR))
    cfg = RingConfig(n_turns=5, n_particles=50, tune_probes=0, checkpoint_every=1, seed=2)
    ck = tmp_path / "ck.npz"
    full = _run(lat, cfg, checkpoint_path=ck)
    assert list(full.turns) == [0, 1] and full.meta["all_lost_at_turn"] == 1
    assert not full.truncated and load_checkpoint(ck)["turn"] == 1
    again = _run(lat, cfg, restart_from=ck)
    assert list(again.turns) == [0, 1] and again.meta["all_lost_at_turn"] == 1
    assert again.meta["restart_turn"] == 1 and not again.truncated


def test_a_restart_with_other_probes_is_refused(booster, tmp_path):
    cfg = RingConfig(n_turns=6, n_particles=50, tune_probes=1, checkpoint_every=3, seed=2)
    ck = tmp_path / "ck.npz"
    _run(booster, cfg, checkpoint_path=ck)
    for change in ({"tune_probes": 0}, {"stats_every": 2}, {"probe_max_sigma": 2.0},
                   {"seed": 3}):
        other = dataclasses.replace(cfg, **change)
        with pytest.raises(ValueError, match="continue it with the same settings"):
            _run(booster, other, restart_from=ck)
    # (the snapshots and the number of turns may change)
    more = dataclasses.replace(cfg, n_turns=9, snapshot_turns=["last"])
    res = _run(booster, more, restart_from=ck)
    assert res.turns_done == 9 and sorted(res.snapshots) == [9]


def test_the_probe_record_is_capped(booster, monkeypatch):
    # the probes' coordinates are kept for the first PROBE_RECORD_TURNS turns
    # (their tunes from those); tracked on to the end, unrecorded, bit for bit
    from linac_gen.ring import engine as E
    full = _run(booster, RingConfig(n_turns=60, n_particles=20, tune_probes=1, seed=1))
    monkeypatch.setattr(E, "PROBE_RECORD_TURNS", 40)
    cut = _run(booster, RingConfig(n_turns=60, n_particles=20, tune_probes=1, seed=1))
    assert cut.probes["tbt"].shape == (1, 41, 6) and full.probes["tbt"].shape == (1, 61, 6)
    assert np.array_equal(cut.probes["tbt"], full.probes["tbt"][:, :41])
    assert cut.meta["probe_record_turns"] == 40
    assert np.array_equal(cut.probes["state"], full.probes["state"])


def test_probes_advanced_unrecorded_are_the_recorded_ones(booster):
    from linac_gen.ring.compile import compile_from_config
    from linac_gen.ring.engine import _ProbeTracker
    from linac_gen.ring.optics import _energy
    rl = compile_from_config(copy.deepcopy(booster), RingConfig())
    X = np.zeros((6, 3))
    X[0] = [1e-3, 2e-3, -1e-3]
    X[5] = [1e-4, 0.0, -2e-4]
    pt, rp, rv = _energy(X[5], rl.reference.beta0)
    probe = {"X": X, "ptau": pt, "rpp": rp, "rvv": rv, "state": np.ones(3, np.int8)}
    a = _ProbeTracker(rl, probe, 5.6)
    b = _ProbeTracker(rl, probe, 5.6)
    a.track(0, 7)
    b.advance(0, 7)
    for k in ("X", "ptau", "rpp", "rvv", "state"):
        assert np.array_equal(a.arrays()[k], b.arrays()[k]), k


def test_the_live_callback_gets_the_reference(booster):
    got = []

    def live(stats, n_turns, ref):
        got.append((len(stats["turns"]), n_turns, dict(ref)))
    res = _run(booster, RingConfig(n_turns=3, n_particles=40, tune_probes=0),
               live_callback=live)
    assert [g[0] for g in got] == [2, 3, 4] and {g[1] for g in got} == {3}
    assert got[-1][2] == {"beta0gamma0": res.meta["beta0gamma0"], "p0c_mev": res.meta["p0c_mev"]}


def test_the_beam_tab_beam_is_made_at_the_rings_reference(booster_no_rf):
    # a Beam tab set for a linac (H-, 2.1 MeV, 162.5 MHz): its distribution is
    # generated at the ring's species and energy, said in the notes
    b = _beam(species="H-", energy=2.1226695, frequency=162.5, n_particles=200)
    res = _run(booster_no_rf, RingConfig(n_turns=2, source="beam", tune_probes=0), b)
    assert res.n_alive[0] == 200 and res.meta["kinetic_mev"] == pytest.approx(800.0)
    assert any("ring's reference, proton 800 MeV" in n for n in res.meta["notes"])


def test_snapshots_at_the_last_turn(booster):
    res = _run(booster, RingConfig(n_turns=5, n_particles=30, tune_probes=0,
                                   snapshot_turns=[0, "last"]))
    assert sorted(res.snapshots) == [0, 5]


def test_moments_and_products_without_blas():
    from linac_gen.ring.engine import _apply, _moments
    rng = np.random.default_rng(4)
    X = np.ascontiguousarray(rng.normal(size=(6, 5000)) * [[1e-3], [1e-5], [2e-3], [3e-5],
                                                            [0.3], [1e-4]])
    mu, C = _moments(X)
    assert np.allclose(mu, X.mean(axis=1), rtol=0, atol=0)
    assert np.allclose(C, np.cov(X, bias=True), rtol=1e-12, atol=0)
    M = rng.normal(size=(6, 6))
    assert np.allclose(_apply(M, X), M @ X, rtol=1e-13, atol=1e-18)


def test_invariants_do_not_drift():
    # (L2) a small-amplitude particle in a thin-lens FODO ring, 10^6 turns:
    # its Courant-Snyder invariants after the last turn are those of the start
    # to 1e-9 (a symplectic map: no secular drift, rounding only)
    from linac_gen.core.lattice import Lattice
    from linac_gen.elements.drift import Drift
    from linac_gen.elements.multipole import Multipole
    from linac_gen.ring.compile import compile_ring
    from linac_gen.ring.optics import twiss
    from linac_gen.ring.reference import RingReference
    lat = Lattice()
    for i in range(4):
        lat.add(Multipole(name=f"qf{i}", knl=[0.0, 0.25]))
        lat.add(Drift(name=f"d1_{i}", length=3e3))
        lat.add(Multipole(name=f"qd{i}", knl=[0.0, -0.25]))
        lat.add(Drift(name=f"d2_{i}", length=3e3))
    rl = compile_ring(lat, RingReference.from_kinetic("proton", 800.0))
    Winv = np.linalg.inv(np.asarray(twiss(rl, chromaticity=False).W))
    n = 1_000_000
    X = np.zeros((6, 1))
    X[0], X[2] = 1e-4, 5e-5
    res = RingSimulation(lat, _beam(), RingConfig(n_turns=n, tune_probes=0, stats_every=n),
                         coords=X, ring_lattice=rl).run()
    assert res.turns_done == n and res.n_alive[-1] == 1

    def J(v):
        z = Winv @ v[:4]
        return np.array([(z[0] ** 2 + z[1] ** 2) / 2, (z[2] ** 2 + z[3] ** 2) / 2])

    assert np.all(np.abs(J(res.mean[-1]) / J(X[:, 0]) - 1.0) < 1e-9)


def test_a_tune_step_across_an_integer_is_the_short_way():
    from linac_gen.ring.engine import _tune_step
    assert _tune_step(0.999, 0.001) == pytest.approx(-0.002, abs=1e-15)
    assert _tune_step(0.001, 0.999) == pytest.approx(0.002, abs=1e-15)
    # (under half a turn: the plain difference, bit for bit)
    for a, b in ((0.3213, 0.3211), (0.71, 0.28), (0.5, 0.0)):
        assert _tune_step(a, b) == a - b
