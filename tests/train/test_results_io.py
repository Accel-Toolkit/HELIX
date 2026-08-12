"""M7 results/HDF5 anchors.

1. Full-schema round-trip for a tracked mp train with loading + HOM on:
   every array np.array_equal through the file, strings equal, ledgers
   and pins dict-exact, pattern <-> RLE round-trip.
2. Fast-mode round-trip on the full PIP-II 89,375-slot pattern with a
   history stride, stride respected through the file.
3. Hybrid round-trip including the replay/ override group.
4. THE INDEPENDENT ANCHOR: the LOADED per-arrival beam phasor of a
   chopped fast train equals the exponential wake sum / geometric-series
   steady-state-and-droop closed forms computed HERE from (R/Q, Q_L,
   detuning, q, T_slot, pattern) alone — through the file, never against
   the in-memory objects (round-trips cancel symmetric errors).
5. Single-bunch compatibility: the single-bunch writer/loader are
   untouched — no train/ group, no run_type, loader refusal both ways.
6. Summary-only (keep_full_results=False) round-trip with a REAL summary
   table; abort-truncation flag; pre-M7 file refusal.

Fixture lattices carry a FREQ card and a SET_SYNC_PHASE NCells cavity
(sync_phase=True) — cardless test lattices masked an M1 crash on every
real deck.
"""
from __future__ import annotations

import cmath
import json
import math

import h5py
import numpy as np
import pytest

from linac_gen.core.config import BeamConfig
from linac_gen.core.lattice import Lattice
from linac_gen.core.simulation import Simulation
from linac_gen.distributions.factory import create_beam
from linac_gen.elements.drift import Drift
from linac_gen.elements.lattice_commands import Freq
from linac_gen.elements.ncells import NCells
from linac_gen.elements.rf_gap import RFGap
from linac_gen.train import (FastPulseRunner, PulsePattern, TrainConfig,
                             TrainPhysics, TrainRunner, load_train_results,
                             run_train)
from linac_gen.train.results import _BUNCH_FLOAT_KEYS

F = 162.5           # MHz
ROQ = 200.0         # Ohm
QL = 5.0e6
I_MA = 5.0


def _lattice():
    """FREQ card + two thin gaps + one SET_SYNC_PHASE NCells + drifts."""
    lat = Lattice()
    lat.add(Freq("FREQ", F))
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


def _cfg(**kw):
    base = dict(species="proton", energy=3.0, frequency=F,
                current=I_MA, n_particles=200, distribution="waterbag",
                emit_nx=0.25, alpha_x=0.0, beta_x=0.4,
                emit_ny=0.25, alpha_y=0.0, beta_y=0.4,
                emit_z=0.15, alpha_z=0.0, beta_z=1.2,
                centroid_x=1.0, bunch_frequency_MHz=F)
    base.update(kw)
    return BeamConfig(**base)


def _sidecar(tmp_path, hom=True, name="cav.json", **extra):
    entry = {"r_over_q": ROQ, "q_loaded": QL, **extra}
    if hom:
        entry["hom_modes"] = [dict(f_MHz=900.0, r_over_q_t=500.0,
                                   q_loaded=1.0e5, polarization_deg=0.0)]
    p = tmp_path / name
    p.write_text(json.dumps({pat: entry for pat in ("CAV*", "NC*")}))
    return str(p)


def _assert_bunch_equal(loaded_ns, r):
    """Every persisted per-bunch series equals the in-memory result."""
    n_checked = 0
    for k in _BUNCH_FLOAT_KEYS:
        arr = getattr(r, k, None)
        if arr is None or not len(arr):
            assert getattr(loaded_ns, k, None) is None
            continue
        assert np.array_equal(getattr(loaded_ns, k),
                              np.asarray(arr, np.float64)), k
        n_checked += 1
    assert n_checked >= 10
    cent = getattr(r, "centroid", None)
    if cent is not None and len(cent):
        assert np.array_equal(loaded_ns.centroid,
                              np.asarray(cent, np.float64))
    idx = getattr(r, "element_exit_idx", None)
    if idx is not None and len(idx):
        assert loaded_ns.element_exit_idx.dtype == np.int64
        assert np.array_equal(loaded_ns.element_exit_idx,
                              np.asarray(idx, np.int64))


# ------------------------------------------------------------------
# 1. tracked mp round-trip (loading + HOM)
# ------------------------------------------------------------------
def test_roundtrip_tracked_mp_loading_hom(tmp_path):
    side = _sidecar(tmp_path)
    tc = TrainConfig(bunch_frequency_MHz=F,
                     pattern=PulsePattern.from_rle("1*3 0*4 1*2"),
                     mode="mp",
                     physics=TrainPhysics(beam_loading=True, hom=True),
                     cavity_params=side)
    res = TrainRunner(_lattice(), _cfg(), tc).run()
    out = tmp_path / "train_mp.h5"
    res.save_hdf5(str(out))
    ld = load_train_results(str(out))

    # identity / config scalars
    assert ld.schema_version == 1
    assert ld.run_type == "train"
    assert ld.mode == "mp"
    assert ld.bunch_frequency_MHz == F
    assert ld.seed == tc.seed
    assert ld.keep_full_results is True
    assert ld.physics == TrainPhysics(beam_loading=True, hom=True)
    assert ld.cavity_params == side
    assert ld.charge_scale == "none"
    assert ld.jitter is None
    # pattern <-> RLE round-trip through the file
    assert ld.pattern_rle == "1*3 0*4 1*2"
    assert np.array_equal(ld.pattern.filled, tc.pattern.filled)
    assert np.array_equal(
        PulsePattern.from_rle(ld.pattern_rle).filled, tc.pattern.filled)
    assert ld.n_slots == 9 and ld.n_bunches_pattern == 5
    assert ld.n_bunches_tracked == 5

    # summary table exact (incl. the new ordinal + centroid columns)
    s0 = res.summary()
    assert set(ld.summary) == set(s0)
    for k, v in s0.items():
        assert np.array_equal(ld.summary[k], v), k
    assert ld.summary["slot"].dtype == np.int64
    assert ld.summary["ordinal"].tolist() == [0, 1, 2, 3, 4]
    assert np.isfinite(ld.summary["mean_x"]).all()

    # per-bunch groups mirror the in-memory recorders exactly
    assert ld.slots == res.slots == [0, 1, 2, 7, 8]
    for r, lr in zip(res.bunch_results, ld.bunch_results):
        _assert_bunch_equal(lr, r)

    # pins (the SET_SYNC_PHASE NCells was calibrated + pinned)
    assert res.pins and ld.pins == res.pins
    assert ld.pin_by_index == res.pin_by_index
    assert ld.pins_unindexed == []

    # tracked ledgers dict-exact through the file
    assert res.applied_loading and ld.applied_loading == res.applied_loading
    assert res.v_beam_loading and ld.v_beam_loading == res.v_beam_loading
    assert res.applied_hom and ld.applied_hom == res.applied_hom
    assert res.hom_w and ld.hom_w == res.hom_w

    # cavity_state / hom static attrs vs the runner's registry table
    assert len(res.cavity_table) == 3
    for rec in res.cavity_table:
        key = (rec["element_index"], rec["name"])
        ns = ld.cavity_state[key]
        for a in ("frequency_MHz", "v_design_MV", "phi_s_deg",
                  "dw_design_MeV", "r_over_q", "q_loaded", "detuning_Hz"):
            assert getattr(ns, a) == rec[a], a
        assert ns.name == rec["name"]
        hom_ns = ld.hom[key]
        assert hom_ns.modes == rec["hom_modes"]
        assert hom_ns.slot.tolist() == [0, 1, 2, 7, 8]

    # design-pass exit energy stamped
    assert ld.w_design_exit_MeV == float(res.design_result.ref_w_kin[-1])
    # tracked run: no replay/fast content
    assert ld.fast is None and ld.replay_slots == []
    assert ld.replay_overrides == {} and ld.replay_bunches == {}
    # provenance follows the single-bunch convention
    assert ld.provenance["run_type"] == "train"
    assert "linac_gen_version" in ld.provenance
    assert ld.provenance["beam_seed"] == tc.seed
    assert ld.beam_config["n_particles"] == 200


# ------------------------------------------------------------------
# 1b. envelope-mode round-trip (dual-regime rule: loader covers every mode)
# ------------------------------------------------------------------
def test_envelope_train_roundtrip(tmp_path):
    tc = TrainConfig(bunch_frequency_MHz=F, mode="envelope",
                     pattern=PulsePattern.from_rle("1*2 0*2 1*1"))
    res = TrainRunner(_lattice(), _cfg(), tc).run()
    out = tmp_path / "train_env.h5"
    res.save_hdf5(str(out))
    ld = load_train_results(str(out))
    assert ld.mode == "envelope"
    assert ld.slots == [0, 1, 4]
    for r, lr in zip(res.bunch_results, ld.bunch_results):
        _assert_bunch_equal(lr, r)
    s0 = res.summary()
    assert set(ld.summary) == set(s0)
    for k, v in s0.items():
        if v.dtype.kind == "f":
            # envelope results carry no transmission -> NaN column
            assert np.array_equal(ld.summary[k], v, equal_nan=True), k
        else:
            assert np.array_equal(ld.summary[k], v), k


# ------------------------------------------------------------------
# 2. fast round-trip: full PIP-II pattern, stride respected
# ------------------------------------------------------------------
def test_fast_pip2_pattern_stride_roundtrip(tmp_path):
    side = _sidecar(tmp_path, hom=False)
    pattern = PulsePattern.from_duty(89375, 2, 6)     # the 0.55 ms pulse
    tc = TrainConfig(bunch_frequency_MHz=F, pattern=pattern, mode="fast",
                     physics=TrainPhysics(beam_loading=True),
                     cavity_params=side)
    stride = 512
    res = FastPulseRunner(_lattice(), _cfg(), tc,
                          history_stride=stride).run()
    out = tmp_path / "train_fast.h5"
    res.save_hdf5(str(out))
    ld = load_train_results(str(out))

    fs, lf = res.fast, ld.fast
    assert ld.mode == "fast"
    assert lf.history_stride == stride
    assert lf.truncated is False
    assert lf.w_design_exit_MeV == fs.w_design_exit_MeV
    assert lf.charge_per_bunch_C == fs.charge_per_bunch_C
    assert np.array_equal(lf.slot, fs.slot)
    assert np.array_equal(lf.w_exit_MeV, fs.w_exit_MeV)
    assert lf.slot.size == pattern.n_bunches            # every bunch
    # stride respected: histories are the ::stride decimation
    assert np.array_equal(lf.history_slot, fs.history_slot)
    assert np.array_equal(lf.history_slot, fs.slot[::stride])
    assert lf.history_slot.size == \
        (pattern.n_bunches + stride - 1) // stride
    assert len(lf.cavities) == len(fs.cavities) == 3
    for a, b in zip(lf.cavities, fs.cavities):
        assert (a.element_index, a.name) == (b.element_index, b.name)
        assert a.voltage_rel.size == lf.history_slot.size
        assert np.array_equal(a.voltage_rel, b.voltage_rel)
        assert np.array_equal(a.phase_offset_deg, b.phase_offset_deg)
        assert np.array_equal(a.v_beam_MV, b.v_beam_MV)
        for attr in ("frequency_MHz", "v_design_MV", "phi_s_deg",
                     "dw_design_MeV", "r_over_q", "q_loaded",
                     "detuning_Hz"):
            assert getattr(a, attr) == getattr(b, attr), attr
        assert a.centroid is None and a.hom_w is None
        assert a.hom_modes == ()
    # fast mode tracks no bunches
    assert ld.slots == [] and ld.bunch_results == []
    with h5py.File(out) as f:
        assert "bunches" not in f
        assert f["train/fast"].attrs["truncated"] == np.False_


# ------------------------------------------------------------------
# 3. hybrid round-trip including the replay group
# ------------------------------------------------------------------
def test_hybrid_roundtrip_with_replay_group(tmp_path):
    side = _sidecar(tmp_path)
    tc = TrainConfig(bunch_frequency_MHz=F,
                     pattern=PulsePattern.from_rle("1*3 0*5 1*1"),
                     mode="hybrid",
                     physics=TrainPhysics(beam_loading=True, hom=True),
                     cavity_params=side, select_bunches=[0, 8])
    res = run_train(_lattice(), _cfg(), tc)
    out = tmp_path / "train_hybrid.h5"
    res.save_hdf5(str(out))
    ld = load_train_results(str(out))

    assert ld.mode == "hybrid"
    assert ld.select_bunches == [0, 8]
    assert ld.n_bunches_replayed == 2
    assert ld.replay_slots == [0, 8]
    assert sorted(ld.replay_bunches) == [0, 8]
    # override ledgers exact (selector strings + float values)
    assert ld.replay_overrides == res.replay_overrides
    for ov in ld.replay_overrides.values():
        assert ov and all(isinstance(s, str) and s.startswith("@")
                          for s, _v in ov)
    # replayed full results mirror the in-memory recorders
    for slot in (0, 8):
        _assert_bunch_equal(ld.replay_bunches[slot],
                            res.replay_bunches[slot])
    # pass-1 fast product round-trips too
    assert ld.fast is not None
    assert np.array_equal(ld.fast.w_exit_MeV, res.fast.w_exit_MeV)
    assert ld.pins == res.pins and ld.pin_by_index == res.pin_by_index
    # hybrid: no tracked bunches, no tracked ledgers
    assert ld.slots == [] and ld.applied_loading == {}
    with h5py.File(out) as f:
        assert sorted(f["train/replay"]) == ["b_0000", "b_0008"]


# ------------------------------------------------------------------
# 4. THE INDEPENDENT ANCHOR: loaded phasor history vs analytic wake sum
# ------------------------------------------------------------------
def test_loaded_phasor_history_matches_analytic(tmp_path):
    """Chopped fast train -> HDF5 -> reload; the LOADED per-arrival beam
    phasor equals the wake sum computed here from first principles:

        dv  = -(omega (R/Q) q / 2) 1e-6 exp(-i phi_s)      [MV]
        rho = exp(-T_slot/tau + i 2 pi df T_slot),  tau = 2 Q_L/omega
        v_beam[k] = sum_{j <= k} dv rho^(s_k - s_j)

    plus the geometric-series steady-state/droop closed forms — all
    through the file, never against the in-memory objects."""
    DET = 300.0                 # Hz detuning
    PH = -20.0                  # deg synchronous phase (RFGap setting)
    lat = Lattice()
    lat.add(Freq("FREQ", F))
    lat.add(Drift("D0", 40.0, aperture=30.0))
    lat.add(RFGap(name="CAV1", voltage=0.8, phase=PH, frequency=F))
    lat.add(Drift("D1", 40.0, aperture=30.0))
    side = _sidecar(tmp_path, hom=False, detuning_Hz=DET)
    filled = np.array([True] * 10 + [False] * 40 + [True] * 6)
    tc = TrainConfig(bunch_frequency_MHz=F,
                     pattern=PulsePattern.from_array(filled), mode="fast",
                     physics=TrainPhysics(beam_loading=True),
                     cavity_params=side)
    out = tmp_path / "chopped_fast.h5"
    FastPulseRunner(lat, _cfg(), tc).run().save_hdf5(str(out))

    ld = load_train_results(str(out))               # through the file
    assert np.array_equal(ld.pattern.filled, filled)
    rec = ld.fast.cavities[0]
    assert rec.name == "CAV1"
    v_loaded = rec.v_beam_MV

    # ---- analytic, from (R/Q, Q_L, df, q, T_slot, pattern) alone ----
    omega = 2.0 * math.pi * F * 1e6
    tau = 2.0 * QL / omega
    T = 1.0 / (F * 1e6)
    q = I_MA * 1e-3 / (F * 1e6)                     # loss-free I/f_bunch
    dv = -(omega * ROQ * q / 2.0) * 1e-6 * cmath.exp(
        -1j * math.radians(PH))
    rho = cmath.exp(-T / tau + 1j * 2.0 * math.pi * DET * T)
    slots = np.flatnonzero(filled)
    expect = np.array([sum(dv * rho ** int(sk - sj)
                           for sj in slots[:k + 1])
                       for k, sk in enumerate(slots)])
    np.testing.assert_allclose(v_loaded, expect, rtol=1e-9, atol=0)

    # geometric-series build-up along the leading uniform run
    for k in (0, 4, 9):
        geo = dv * (1 - rho ** (k + 1)) / (1 - rho)
        assert v_loaded[k] == pytest.approx(geo, rel=1e-9)
    # droop across the chopped gap: pure decay over 41 slots + one kick
    assert v_loaded[10] == pytest.approx(v_loaded[9] * rho ** 41 + dv,
                                         rel=1e-9)
    # steady-state magnitude bound: |v| approaches |dv|/(1 - |rho|)
    v_inf = abs(dv) / (1.0 - abs(rho))
    assert abs(v_loaded[9]) < v_inf
    # loaded exit energies droop monotonically along the uniform run
    assert np.all(np.diff(ld.fast.w_exit_MeV[:10]) < 0)


# ------------------------------------------------------------------
# 5. single-bunch files unchanged + loader refusals
# ------------------------------------------------------------------
def test_single_bunch_file_unchanged_and_refused(tmp_path):
    from linac_gen.io.hdf5_output import (load_results_hdf5,
                                          save_results_hdf5)
    lat = _lattice()
    cfg = _cfg()
    res = Simulation(lat, create_beam(cfg, seed=42)).run()
    p = tmp_path / "single.h5"
    save_results_hdf5(res, str(p), beam_config=cfg, lattice=lat, seed=42)
    with h5py.File(p) as f:
        assert "train" not in f
        assert "run_type" not in f["provenance"].attrs
        assert "envelope" in f and "reference" in f
    d = load_results_hdf5(str(p))
    assert np.array_equal(d["sigma_x"], np.asarray(res.sigma_x, float))
    assert np.array_equal(d["ref_w_kin"],
                          np.asarray(res.ref_w_kin, float))
    with pytest.raises(ValueError, match="not a multibunch train"):
        load_train_results(str(p))


def test_pre_m7_train_file_refused(tmp_path):
    p = tmp_path / "old_train.h5"
    with h5py.File(p, "w") as f:
        f.create_group("train")
    with pytest.raises(ValueError, match="schema_version"):
        load_train_results(str(p))


def test_unicode_cavity_name_roundtrip(tmp_path):
    """utf-8-safe strings through cavity_state attrs, pins and ledgers
    (Windows-round house rule: never assume ASCII element names)."""
    import dataclasses

    from linac_gen.train.results import TrainResults
    tc = TrainConfig(bunch_frequency_MHz=F,
                     pattern=PulsePattern.uniform(2),
                     physics=TrainPhysics(beam_loading=True),
                     cavity_params="sidecar.json")
    res = TrainResults(tc, "mp")
    name = "CAVITÉ_β1"                    # CAVITÉ_β1
    res.cavity_table = [dict(element_index=3, name=name,
                             frequency_MHz=F, v_design_MV=0.8,
                             phi_s_deg=-20.0, dw_design_MeV=0.5,
                             r_over_q=ROQ, q_loaded=QL, detuning_Hz=0.0,
                             hom_modes=())]
    res.applied_loading = {(0, 3, name): (0.01, -0.2),
                           (1, 3, name): (0.02, -0.3)}
    res.v_beam_loading = {(0, 3, name): complex(-1e-4, 2e-5),
                          (1, 3, name): complex(-2e-4, 4e-5)}
    res.pins = {name: -12.5}
    res.pin_by_index = {3: -12.5}
    out = tmp_path / "unicode.h5"
    res.save_hdf5(str(out))
    ld = load_train_results(str(out))
    assert ld.pins == res.pins
    assert ld.pin_by_index == res.pin_by_index
    assert ld.applied_loading == res.applied_loading
    assert ld.v_beam_loading == res.v_beam_loading
    assert ld.cavity_state[(3, name)].name == name
    assert dataclasses.asdict(ld.physics) == dataclasses.asdict(tc.physics)


# ------------------------------------------------------------------
# 6. summary-only round-trip + abort truncation
# ------------------------------------------------------------------
def test_summary_only_roundtrip(tmp_path):
    tc = TrainConfig(bunch_frequency_MHz=F,
                     pattern=PulsePattern.from_rle("1*2 0*3 1*1"),
                     keep_full_results=False)
    res = TrainRunner(_lattice(), _cfg(), tc).run()
    assert res.bunch_results == [None, None, None]
    s0 = res.summary()
    # the M7 fix: rows captured before the full result is dropped
    assert np.isfinite(s0["sigma_x"]).all()
    assert np.isfinite(s0["ref_w_kin"]).all()
    assert s0["ordinal"].tolist() == [0, 1, 2]
    out = tmp_path / "summary_only.h5"
    res.save_hdf5(str(out))
    with h5py.File(out) as f:
        assert "bunches" not in f
    ld = load_train_results(str(out))
    assert ld.keep_full_results is False
    assert ld.slots == [0, 1, 5]
    assert ld.bunch_results == [None, None, None]
    for k, v in s0.items():
        assert np.array_equal(ld.summary[k], v), k


def test_fast_abort_truncation_flag_roundtrip(tmp_path):
    side = _sidecar(tmp_path, hom=False)
    tc = TrainConfig(bunch_frequency_MHz=F,
                     pattern=PulsePattern.uniform(12), mode="fast",
                     physics=TrainPhysics(beam_loading=True),
                     cavity_params=side)
    calls = {"n": 0}

    def abort():
        calls["n"] += 1
        return calls["n"] > 5

    with pytest.warns(UserWarning, match="aborted"):
        res = FastPulseRunner(_lattice(), _cfg(), tc,
                              should_abort=abort).run()
    assert res.fast.truncated is True
    assert res.fast.slot.size == 5
    out = tmp_path / "aborted_fast.h5"
    res.save_hdf5(str(out))
    ld = load_train_results(str(out))
    assert ld.fast.truncated is True
    assert ld.fast.slot.size == 5 < ld.n_bunches_pattern
    assert np.array_equal(ld.fast.w_exit_MeV, res.fast.w_exit_MeV)
