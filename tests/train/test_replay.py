"""M6 hybrid two-pass replay anchors.

1. LOSSLESS CONSTRUCTION (load-bearing): a replay fed a TRACKED train
   run's own recorded per-bunch state (applied voltage_rel/phase_offset,
   psi pins, HOM kicks — via overrides_for_tracked_slot) reproduces the
   tracked bunch BIT-IDENTICALLY on a fresh lattice.
2. Override round-trip: every exported attribute (voltage_rel,
   phase_offset, sync_phase_pin, hom_kick_x/y) survives selector
   application on a RE-PARSED deck, float-exact, by @index and by NAME.
3. Parallel scan_pool replay == in-process replay (2 workers).
4. Lattice fingerprint mismatch (deck != live lattice) refused loudly.
5. Zero-coupling: hybrid with the coupling physics off is REFUSED at
   config construction (documented choice — beam_loading is the
   minimum; a physics-off "hybrid" is a mislabelled single-bunch run),
   plus the sibling refusals (distinct neighbours, bad selections,
   direct TrainRunner construction, parallel without lattice_path).
6. select_bunches="auto" hits every pattern edge (first/last bunch of
   each filled run), stays within the cap, picks filled slots only, and
   log-spaces the interior.
7. Teardown: an in-process hybrid run leaves the live lattice exactly
   as it entered (priors restored, no pins, no hom_kick_* instance
   attributes) — the zero-coupling contract for the new attributes.
"""
from __future__ import annotations

import json

import numpy as np
import pytest

from linac_gen.cli.common import apply_element_override
from linac_gen.core.config import BeamConfig, SpaceChargeConfig
from linac_gen.core.lattice import Lattice
from linac_gen.core.simulation import Simulation
from linac_gen.distributions.factory import create_beam
from linac_gen.elements.drift import Drift
from linac_gen.elements.ncells import NCells
from linac_gen.elements.rf_gap import RFGap
from linac_gen.train import (HybridReplayRunner, PulsePattern, TrainConfig,
                             TrainPhysics, TrainRunner, auto_select_bunches,
                             run_train)
from linac_gen.train.replay import (build_overrides, lattice_fingerprint,
                                    overrides_for_tracked_slot)

F = 162.5
ROQ = 200.0
QL = 5.0e6
I_MA = 5.0

_CMP_KEYS = ("s", "sigma_x", "sigma_y", "sigma_phi", "sigma_w",
             "emit_x", "emit_y", "emit_z", "transmission",
             "ref_w_kin", "ref_phi_s")


def _assert_identical(res_a, res_b, keys=_CMP_KEYS):
    for k in keys:
        a, b = getattr(res_a, k, None), getattr(res_b, k, None)
        if a is None and b is None:
            continue
        np.testing.assert_array_equal(np.asarray(a, float),
                                      np.asarray(b, float), err_msg=k)


def _cfg(**kw):
    base = dict(species="proton", energy=3.0, frequency=F,
                current=I_MA, n_particles=200, distribution="waterbag",
                emit_nx=0.25, alpha_x=0.0, beta_x=0.4,
                emit_ny=0.25, alpha_y=0.0, beta_y=0.4,
                emit_z=0.15, alpha_z=0.0, beta_z=1.2,
                centroid_x=1.0, bunch_frequency_MHz=F)
    base.update(kw)
    return BeamConfig(**base)


def _mode(f_MHz=900.0, roqt=500.0, ql=1.0e5, pol=0.0):
    return dict(f_MHz=f_MHz, r_over_q_t=roqt, q_loaded=ql,
                polarization_deg=pol)


def _sidecar(tmp_path, patterns=("CAV*", "NC*"), name="cav.json"):
    entry = {"r_over_q": ROQ, "q_loaded": QL, "hom_modes": [_mode()]}
    p = tmp_path / name
    p.write_text(json.dumps({pat: entry for pat in patterns}))
    return str(p)


def _lattice():
    """Two thin gaps + one SET_SYNC_PHASE NCells (exercises the pin
    transport) with drifts."""
    lat = Lattice()
    lat.add(Drift("D0", 40.0, aperture=30.0))
    lat.add(RFGap(name="CAV1", voltage=0.8, phase=-20.0, frequency=F))
    lat.add(Drift("D1", 40.0, aperture=30.0))
    lat.add(RFGap(name="CAV2", voltage=0.6, phase=-25.0, frequency=F))
    lat.add(Drift("D2", 40.0, aperture=30.0))
    lat.add(NCells("NC", mode=1, n_cells=4, beta_g=0.081, eot_v_per_m=8e5,
                   theta_s_deg=-40.0, aperture_mm=20.0, sync_phase=True,
                   frequency_mhz=F))
    lat.add(Drift("D3", 40.0, aperture=30.0))
    return lat


def _tc(mode="hybrid", loading=True, hom=True, rle="1*3 0*5 1*1",
        side=None, **kw):
    physics = kw.pop("physics",
                     TrainPhysics(beam_loading=loading, hom=hom))
    return TrainConfig(bunch_frequency_MHz=F,
                       pattern=PulsePattern.from_rle(rle), mode=mode,
                       physics=physics, cavity_params=side, **kw)


# ------------------------------------------------------------------
# 1. LOSSLESS CONSTRUCTION
# ------------------------------------------------------------------
def test_lossless_construction_tracked_state(tmp_path):
    """Replaying bunch k of a tracked (loading+hom) train from its OWN
    recorded applied values on a FRESH lattice is bit-identical to the
    tracked bunch k — mid-train and post-chopped-gap bunches, pin
    included."""
    side = _sidecar(tmp_path)
    cfg = _cfg()
    tc = _tc(mode="mp", side=side)
    runner = TrainRunner(_lattice(), cfg, tc)
    res_t = runner.run()
    assert res_t.slots == [0, 1, 2, 8]
    assert "NC" in res_t.pins and res_t.pin_by_index      # pin recorded
    assert not res_t.pins_unindexed
    assert res_t.applied_loading and res_t.applied_hom    # ledgers filled

    for pos, slot in [(2, 2), (3, 8)]:
        ov = overrides_for_tracked_slot(res_t, slot)
        # loading values on all 3 cavities + the NC pin + the (x-pol)
        # HOM kicks are transported
        attrs = {sel.rsplit(".", 1)[1] for sel, _ in ov}
        assert {"voltage_rel", "phase_offset", "sync_phase_pin",
                "hom_kick_x"} <= attrs
        lat2 = _lattice()                                 # virgin lattice
        for sel, val in ov:
            apply_element_override(lat2, sel, val)
        single = Simulation(
            lat2, create_beam(runner.beam_config, seed=tc.seed)).run()
        _assert_identical(res_t.bunch_results[pos], single)

    # teeth: WITHOUT the overrides the loaded bunch is NOT reproduced
    bare = Simulation(
        _lattice(), create_beam(runner.beam_config, seed=tc.seed)).run()
    assert not np.array_equal(
        np.asarray(res_t.bunch_results[3].ref_w_kin, float),
        np.asarray(bare.ref_w_kin, float))


# ------------------------------------------------------------------
# 2. override round-trip on a re-parsed deck
# ------------------------------------------------------------------
_DECK = """FREQ 162.5
DRIFT 50 30
GAP 800000 -20 30 0
DRIFT 50 30
GAP 600000 -25 30 0
DRIFT 50 30
SET_SYNC_PHASE
NCELLS 1 4 0.081 800000 -40 20
DRIFT 50 30
END
"""


def _write_deck(tmp_path, text=_DECK, name="replay_deck.dat"):
    p = tmp_path / name
    p.write_text(text)
    return str(p)


def test_override_roundtrip_on_reparsed_lattice(tmp_path):
    """Every transported attribute survives apply_element_override on a
    RE-PARSED lattice float-exactly, via @index selectors (what the
    replay emits) and via NAME selectors; sync_phase_pin (a None slot on
    a virgin element) becomes a float, never a string."""
    from linac_gen.io.tracewin_parser import parse_tracewin
    path = _write_deck(tmp_path)
    lat = parse_tracewin(path)[0]
    gaps = [(i, el) for i, el in enumerate(lat.elements)
            if isinstance(el, RFGap)]
    ncs = [(i, el) for i, el in enumerate(lat.elements)
           if isinstance(el, NCells)]
    assert len(gaps) == 2 and len(ncs) == 1
    (i1, _g1), (i2, g2) = gaps
    (i3, nc) = ncs[0]
    assert nc.sync_phase                                  # SET_SYNC_PHASE
    vals = {"voltage_rel": -3.9481726352845637e-9,
            "phase_offset": 1.2345678913579246e-7,
            "hom_kick_x": 1.1723094817320472e-5,
            "hom_kick_y": -8.413011012345679e-6}
    pin = -32.71828182845905
    ov = [(f"@{i1 + 1}.{a}", v) for a, v in vals.items()]
    ov += [(f"{g2.name}.{a}", 2.0 * v) for a, v in vals.items()]
    ov += [(f"@{i3 + 1}.sync_phase_pin", pin)]

    lat2 = parse_tracewin(path)[0]                        # re-parse
    for sel, val in ov:
        apply_element_override(lat2, sel, val)
    e1, e2, e3 = (lat2.elements[i] for i in (i1, i2, i3))
    for a, v in vals.items():
        got1, got2 = getattr(e1, a), getattr(e2, a)
        assert isinstance(got1, float) and isinstance(got2, float), a
        assert got1 == v, a                               # bit-exact
        assert got2 == 2.0 * v, a
    assert isinstance(e3.sync_phase_pin, float)           # not a string
    assert e3.sync_phase_pin == pin
    # NAME selector reaches the pin slot too
    apply_element_override(lat2, f"{nc.name}.sync_phase_pin", 2.0 * pin)
    assert lat2.elements[i3].sync_phase_pin == 2.0 * pin
    # virgin elements untouched by the transport stay at inert defaults
    fresh = parse_tracewin(path)[0]
    assert fresh.elements[i1].hom_kick_x == 0.0
    assert fresh.elements[i1].hom_kick_y == 0.0
    assert fresh.elements[i3].sync_phase_pin is None
    # and build_overrides skips zero HOM kicks entirely
    out = build_overrides({(4, "g"): (0.1, 0.2)}, {(4, "g"): (0.0, 0.0)},
                          {})
    assert [s for s, _ in out] == ["@5.voltage_rel", "@5.phase_offset"]


# ------------------------------------------------------------------
# 3. parallel == in-process   (+ HDF5 extension smoke)
# ------------------------------------------------------------------
def test_parallel_replay_matches_in_process(tmp_path, monkeypatch):
    """Two-worker scan_pool replay == in-process replay, array-exact
    (same deck, same overrides, same SC).  FFT workers pinned to 1 so
    the parent matches the pool workers' threading."""
    monkeypatch.setenv("LINAC_GEN_FFT_WORKERS", "1")
    from linac_gen.io.tracewin_parser import parse_tracewin
    path = _write_deck(tmp_path)
    side = _sidecar(tmp_path, patterns=("GAP_*", "NC*"))
    cfg = _cfg(n_particles=120)
    sc = SpaceChargeConfig(nx=8, ny=8, nz=8, grid_extent=4.0,
                           use_gpu="cpu")

    def _run(parallel):
        lat = parse_tracewin(path)[0]
        tc = _tc(side=side, rle="1*2 0*4 1*2", select_bunches=[0, 7])
        if parallel:
            return HybridReplayRunner(
                lat, cfg, tc, sc_config=sc, lattice_path=path,
                replay_parallel=True, max_workers=2).run()
        # in-process branch through the run_train dispatch seam
        return run_train(lat, cfg, tc, sc_config=sc, lattice_path=path)

    res_ip = _run(parallel=False)
    res_par = _run(parallel=True)
    assert res_ip.mode == res_par.mode == "hybrid"
    assert sorted(res_ip.replay_bunches) == [0, 7]
    assert sorted(res_par.replay_bunches) == [0, 7]
    # the parallel results really came through the worker/HDF5 path
    from types import SimpleNamespace
    assert isinstance(res_par.replay_bunches[0], SimpleNamespace)
    assert not isinstance(res_ip.replay_bunches[0], SimpleNamespace)
    assert res_ip.fast is not None and res_par.fast is not None
    np.testing.assert_array_equal(res_ip.fast.w_exit_MeV,
                                  res_par.fast.w_exit_MeV)
    assert res_ip.replay_overrides == res_par.replay_overrides
    for slot in (0, 7):
        _assert_identical(res_ip.replay_bunches[slot],
                          res_par.replay_bunches[slot])

    # save_hdf5 minimal extension: replay bunches + slot index present
    out = tmp_path / "hybrid.h5"
    res_ip.save_hdf5(str(out))
    import h5py
    with h5py.File(out) as f:
        tr = f["train"]
        assert tr.attrs["mode"] == "hybrid"
        assert tr.attrs["n_bunches_tracked"] == 0
        assert tr.attrs["n_bunches_replayed"] == 2
        assert tr["replay_slots"][:].tolist() == [0, 7]
        assert "train/fast" in f
        for slot in (0, 7):
            g = f[f"bunches/b_{slot:04d}"]
            np.testing.assert_array_equal(
                g["ref_w_kin"][:],
                np.asarray(res_ip.replay_bunches[slot].ref_w_kin, float))


# ------------------------------------------------------------------
# 4. fingerprint mismatch refused
# ------------------------------------------------------------------
def test_fingerprint_mismatch_refused(tmp_path):
    from linac_gen.io.tracewin_parser import parse_tracewin
    path_a = _write_deck(tmp_path, name="deck_a.dat")
    path_b = _write_deck(tmp_path, _DECK.replace(
        "DRIFT 50 30\nEND", "DRIFT 50 30\nDRIFT 20 30\nEND"),
        name="deck_b.dat")
    lat = parse_tracewin(path_a)[0]
    side = _sidecar(tmp_path, patterns=("GAP_*",))
    tc = _tc(side=side, rle="1*2", select_bunches=[0, 1])
    fp_a, fp_b = (lattice_fingerprint(parse_tracewin(p)[0])
                  for p in (path_a, path_b))
    assert fp_a != fp_b
    assert fp_a == lattice_fingerprint(lat)               # stable
    with pytest.raises(ValueError, match="fingerprint mismatch"):
        HybridReplayRunner(lat, _cfg(), tc,
                           sc_config=SpaceChargeConfig(
                               nx=8, ny=8, nz=8, use_gpu="cpu"),
                           lattice_path=path_b, replay_parallel=True)


# ------------------------------------------------------------------
# 5. refusals (documented choices)
# ------------------------------------------------------------------
def test_hybrid_with_physics_off_refused():
    with pytest.raises(ValueError, match="beam_loading"):
        _tc(loading=False, hom=False, side=None)


def test_hybrid_hom_only_refused(tmp_path):
    """beam_loading is the hybrid minimum — hom alone is refused too."""
    with pytest.raises(ValueError, match="beam_loading"):
        _tc(loading=False, hom=True, side=_sidecar(tmp_path))


def test_hybrid_distinct_neighbors_refused(tmp_path):
    with pytest.raises(ValueError, match="images"):
        _tc(side=_sidecar(tmp_path),
            physics=TrainPhysics(beam_loading=True, direct_sc=True),
            direct_sc_neighbors="distinct")


def test_select_bunches_validation(tmp_path):
    side = _sidecar(tmp_path)
    with pytest.raises(ValueError, match="empty"):
        _tc(side=side, select_bunches=[])
    with pytest.raises(ValueError, match="select_bunches"):
        _tc(side=side, rle="1*2 0*2", select_bunches=[0, 3])  # chopped slot
    with pytest.raises(ValueError, match="select_bunches"):
        _tc(side=side, select_bunches="some")
    tc = _tc(side=side, rle="1*3", select_bunches=(2, 0, 2))
    assert tc.select_bunches == [0, 2]                    # normalised


def test_direct_hybrid_construction_and_parallel_prereqs(tmp_path):
    side = _sidecar(tmp_path)
    tc = _tc(side=side)
    with pytest.raises(ValueError, match="HybridReplayRunner"):
        TrainRunner(_lattice(), _cfg(), tc)
    with pytest.raises(ValueError, match="lattice_path"):
        HybridReplayRunner(_lattice(), _cfg(), tc, replay_parallel=True)
    # current > 0 without a reproducible SC config: refused
    with pytest.raises(ValueError, match="SpaceChargeConfig"):
        HybridReplayRunner(_lattice(), _cfg(), tc,
                           lattice_path=_write_deck(tmp_path),
                           replay_parallel=True)


# ------------------------------------------------------------------
# 5b. direct_sc composes into the replays (and: hom-off regime)
# ------------------------------------------------------------------
def test_hybrid_direct_sc_composes_in_replays(tmp_path):
    """physics.direct_sc + hybrid: every replay's freshly built PIC is
    armed with ITS slot's pattern image factors (spied at the
    pic_setup seam; the factor->field physics itself is M5-anchored),
    the factors are recorded in results.direct_sc, and the forced
    images actually engage.  Also covers the hom-OFF hybrid regime
    (dual-regime rule).  Parallel + direct_sc refused."""
    import warnings as _warnings

    side = _sidecar(tmp_path)
    cfg = _cfg(n_particles=100)
    sc = SpaceChargeConfig(nx=8, ny=8, nz=8, grid_extent=4.0,
                           use_gpu="cpu")

    def _tc_dsc():
        return _tc(side=side, rle="1*2 0*1 1*1",
                   physics=TrainPhysics(beam_loading=True, hom=False,
                                        direct_sc=True),
                   direct_sc_force_engage=True, select_bunches=[1, 3])

    runner = HybridReplayRunner(_lattice(), cfg, _tc_dsc(), sc_config=sc)
    armed = []
    orig = runner._pic_setup

    def spy(pic):
        orig(pic)
        armed.append((pic.train_image_factors, pic.train_force_engage))

    runner._pic_setup = spy
    with _warnings.catch_warnings(record=True) as caught:
        _warnings.simplefilter("always")
        res = runner.run()
    assert not any("never engaged" in str(w.message) for w in caught)
    assert res.direct_sc == {1: (1.0, 0.0), 3: (0.0, 0.0)}
    assert sorted(res.replay_bunches) == [1, 3]
    assert armed == [((1.0, 0.0), True), ((0.0, 0.0), True)]
    assert runner._dsc_engaged_any
    assert all("hom_kick" not in sel
               for ov in res.replay_overrides.values() for sel, _ in ov)
    with pytest.raises(ValueError, match="in-process only"):
        HybridReplayRunner(_lattice(), cfg, _tc_dsc(), sc_config=sc,
                           lattice_path=_write_deck(tmp_path),
                           replay_parallel=True)


# ------------------------------------------------------------------
# 6. auto selection
# ------------------------------------------------------------------
def test_auto_select_hits_every_pattern_edge():
    pat = PulsePattern.from_rle("1*10 0*54 1*26 0*9 1*1")   # 100 slots
    sel = auto_select_bunches(pat)
    assert sel == sorted(set(sel))
    assert {0, 9, 64, 89, 99} <= set(sel)                 # every edge
    assert all(pat.filled[s] for s in sel)                # filled only
    assert len(sel) <= 24

    # log-spaced interior on a long uniform pulse: denser early
    pat2 = PulsePattern.uniform(5000)
    sel2 = auto_select_bunches(pat2)
    assert {0, 4999} <= set(sel2) and len(sel2) <= 24
    gaps = np.diff(sel2)
    assert gaps[0] < gaps[-1]

    # cap bounds only the interior fill — edges are never dropped
    pat3 = PulsePattern.from_duty(40, 1, 2)               # 20 one-bunch runs
    sel3 = auto_select_bunches(pat3, cap=5)
    assert sel3 == [int(s) for s in pat3.filled_slots]

    # config keeps "auto" as-is for the runner to resolve
    assert _tc(side="cav.json").select_bunches == "auto"


# ------------------------------------------------------------------
# 7. teardown — the live lattice leaves the run as it entered
# ------------------------------------------------------------------
def test_in_process_hybrid_leaves_lattice_clean(tmp_path):
    side = _sidecar(tmp_path)
    lat = _lattice()
    cav = lat.elements[1]
    cav.voltage_rel, cav.phase_offset = 0.03, 1.5         # user prior
    cfg = _cfg(n_particles=100)
    tc = _tc(side=side, rle="1*2 0*3 1*1", select_bunches=[1, 5])
    res = run_train(lat, cfg, tc)
    assert sorted(res.replay_bunches) == [1, 5]
    # priors restored exactly; pins cleared; hom_kick_* left inert with
    # no instance attributes behind (class-default state)
    assert cav.voltage_rel == 0.03 and cav.phase_offset == 1.5
    for el in lat.elements:
        assert getattr(el, "sync_phase_pin", None) is None
        assert getattr(el, "hom_kick_x", 0.0) == 0.0
        assert getattr(el, "hom_kick_y", 0.0) == 0.0
        assert "hom_kick_x" not in el.__dict__
        assert "hom_kick_y" not in el.__dict__
    # bit-identity probe: the used lattice tracks like a fresh one
    cav.voltage_rel, cav.phase_offset = 0.0, 0.0
    after = Simulation(lat, create_beam(cfg, seed=7)).run()
    fresh = Simulation(_lattice(), create_beam(cfg, seed=7)).run()
    _assert_identical(after, fresh)
