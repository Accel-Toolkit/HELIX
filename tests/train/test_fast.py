"""M3 fast full-pulse anchors: fast-vs-tracked equivalence (phasor
histories, applied slot values, per-bunch exit energy — one physics
implementation, two drivers), PIP-II-scale pattern in seconds with the
droop/recovery structure, chopped-gap decay through the fast runner,
the trivial no-loading path, prior composition/restoration, and the
adversarial FREQ-jump / pinned-psi equivalences."""
from __future__ import annotations

import json
import math
import time

import numpy as np
import pytest

from linac_gen.core.config import BeamConfig
from linac_gen.core.lattice import Lattice
from linac_gen.elements.drift import Drift
from linac_gen.elements.rf_gap import RFGap
from linac_gen.pic.macrocharge import macro_charge_coulombs
from linac_gen.train import (FastPulseRunner, PulsePattern, TrainConfig,
                             TrainPhysics, TrainRunner, run_train)
from linac_gen.train.cavity_state import (CavityMode, CavityStateRegistry,
                                          _CavityState)

F = 162.5           # MHz
ROQ = 200.0         # Ohm
QL = 5.0e6
V_MV = 0.8
I_MA = 5.0


def _lattice(phase=0.0):
    lat = Lattice()
    lat.add(Drift("D0", 50.0, aperture=30.0))
    lat.add(RFGap(name="CAV1", voltage=V_MV, phase=phase, frequency=F))
    lat.add(Drift("D1", 50.0, aperture=30.0))
    return lat


def _cfg(n=400):
    return BeamConfig(species="proton", energy=3.0, frequency=F,
                      current=I_MA, n_particles=n, distribution="waterbag",
                      emit_nx=0.25, alpha_x=0.0, beta_x=0.4,
                      emit_ny=0.25, alpha_y=0.0, beta_y=0.4,
                      emit_z=0.15, alpha_z=0.0, beta_z=1.2)


def _sidecar(tmp_path, **extra):
    d = {"CAV*": {**dict(r_over_q=ROQ, q_loaded=QL), **extra}}
    p = tmp_path / "cav.json"
    p.write_text(json.dumps(d))
    return str(p)


def _state(detuning=0.0):
    return _CavityState(mode=CavityMode(r_over_q=ROQ, q_loaded=QL,
                                        detuning_Hz=detuning),
                        frequency_MHz=F, v_design_MV=V_MV, phi_s_deg=0.0)


def _instrumented_tracked_run(lat, cfg, tc):
    """Tracked TrainRunner run snapshotting, after every bunch, each
    bound cavity's APPLIED slot values (still on the element until the
    teardown restore) and its post-exit-hook phasor."""
    runner = TrainRunner(lat, cfg, tc)
    items = list(runner._loading.reg.items())
    cavs = [lat.elements[idx] for (idx, _name), _st in items]
    snaps = {"vr": [], "po": [], "vb": []}

    def snap(_i, _n):
        snaps["vr"].append([c.voltage_rel for c in cavs])
        snaps["po"].append([c.phase_offset for c in cavs])
        snaps["vb"].append([st.v_beam for _k, st in items])

    runner.progress_callback = snap
    res = runner.run()
    vr = np.array(snaps["vr"]).T        # (n_cav, n_bunches)
    po = np.array(snaps["po"]).T
    vb = np.array(snaps["vb"]).T
    w = np.array([r.ref_w_kin[-1] for r in res.bunch_results])
    return res, runner, vr, po, vb, w


# ------------------------------------------------- anchor 1: equivalence
@pytest.mark.parametrize("phase,detuning", [(0.0, 0.0), (-20.0, 300.0)])
def test_fast_matches_tracked(tmp_path, phase, detuning):
    """20-bunch lossless uniform train: per-cavity phasor histories,
    applied (voltage_rel, phase_offset) sequences and per-bunch exit
    energies from the fast recursion equal the tracked TrainRunner run
    to float tolerance (same bunch_passage physics, same charge)."""
    n = 20
    side = _sidecar(tmp_path, detuning_Hz=detuning)

    def _tc(mode):
        return TrainConfig(bunch_frequency_MHz=F,
                           pattern=PulsePattern.uniform(n), mode=mode,
                           physics=TrainPhysics(beam_loading=True),
                           cavity_params=side)

    res_mp, _run_mp, vr_t, po_t, vb_t, w_t = _instrumented_tracked_run(
        _lattice(phase), _cfg(), _tc("mp"))
    res_f = FastPulseRunner(_lattice(phase), _cfg(), _tc("fast")).run()
    fs = res_f.fast
    assert fs.slot.tolist() == list(range(n))
    rec = fs.cavities[0]
    np.testing.assert_allclose(rec.voltage_rel, vr_t[0], rtol=1e-9, atol=0)
    np.testing.assert_allclose(rec.phase_offset_deg, po_t[0], rtol=1e-9,
                               atol=0)
    np.testing.assert_allclose(rec.v_beam_MV, vb_t[0], rtol=1e-9, atol=0)
    np.testing.assert_allclose(fs.w_exit_MeV, w_t, rtol=1e-9, atol=0)
    assert fs.w_design_exit_MeV == pytest.approx(
        res_mp.design_result.ref_w_kin[-1], rel=1e-12)


# --------------------------------------- anchor 2: full PIP-II-scale run
def test_full_pip2_pattern_droop_recovery(tmp_path):
    """PulsePattern.from_duty(89375, 2, 6) — the full 0.55 ms pulse —
    runs in seconds and shows the droop/recovery structure (sign and
    monotonicity only; Q_L chosen so tau ~ 98 us << pulse, i.e. the
    steady chopped sawtooth is reached inside the pulse)."""
    pattern = PulsePattern.from_duty(89375, 2, 6)
    side = _sidecar(tmp_path, q_loaded=5.0e4)
    tc = TrainConfig(bunch_frequency_MHz=F, pattern=pattern, mode="fast",
                     physics=TrainPhysics(beam_loading=True),
                     cavity_params=side)
    t0 = time.perf_counter()
    res = FastPulseRunner(_lattice(), _cfg(), tc).run()
    wall = time.perf_counter() - t0
    assert wall < 30.0, f"fast full-pulse run took {wall:.1f} s"
    fs = res.fast
    assert fs.slot.size == pattern.n_bunches == 29792
    w = fs.w_exit_MeV
    w1, w2 = w[0::2], w[1::2]           # first / second bunch of each pair
    assert w1.size == w2.size == 14896
    # droop WITHIN every filled pair (second bunch sees one more kick)
    assert np.all(w2 < w1)
    # pulse head: monotone droop while v_beam builds up
    assert np.all(np.diff(w[:200]) < 0)
    # steady sawtooth (last 20 %): RECOVERY across every 4-slot gap
    i0 = int(0.8 * w1.size)
    assert np.all(w1[i0 + 1:] > w2[i0:-1])
    # net droop from pulse head to steady state, all below design
    assert w2[-1] < w[0] < fs.w_design_exit_MeV


# ------------------------------------------- anchor 3: chopped-gap decay
def test_chopped_gap_pure_decay_fast(tmp_path):
    """M2's analytic chopped-gap expectation through the fast runner: a
    gap of G slots decays v_beam by exactly exp(-G T/tau)."""
    side = _sidecar(tmp_path)

    def _run_fast(rle):
        tc = TrainConfig(bunch_frequency_MHz=F,
                         pattern=PulsePattern.from_rle(rle), mode="fast",
                         physics=TrainPhysics(beam_loading=True),
                         cavity_params=side)
        runner = FastPulseRunner(_lattice(), _cfg(), tc)
        runner.run()
        return next(iter(runner._loading.reg.items()))[1]

    v3 = _run_fast("1*3").v_beam
    st2 = _run_fast("1*3 0*40 1*1")
    tau = 2 * QL / (2 * math.pi * F * 1e6)
    r_gap = math.exp(-(41.0 / (F * 1e6)) / tau)
    q_b = macro_charge_coulombs(I_MA, F, 1)   # fast charge: I/f, loss-free
    dv = CavityStateRegistry.induced_dv_MV(_state(), q_b)
    assert st2.v_beam == pytest.approx(v3 * r_gap + dv, rel=1e-9)


# ------------------------------------------------ anchor 4: zero-coupling
def test_fast_without_loading_is_flat_design():
    """DOCUMENTED CHOICE (fast.py docstring): beam_loading=False with
    mode='fast' runs TRIVIALLY — no cavities are bound, the recursion is
    empty, and every bunch exits at the design energy.  Also covers the
    run_train mode dispatch."""
    tc = TrainConfig(bunch_frequency_MHz=F,
                     pattern=PulsePattern.from_rle("1*5 0*3 1*2"),
                     mode="fast")
    lat = _lattice()
    res = run_train(lat, _cfg(), tc)
    fs = res.fast
    assert res.mode == "fast" and fs is not None
    assert fs.cavities == []
    w_design = res.design_result.ref_w_kin[-1]
    assert np.all(fs.w_exit_MeV == w_design)
    assert fs.slot.tolist() == [0, 1, 2, 3, 4, 8, 9]
    assert res.slots == [] and res.bunch_results == []   # nothing tracked
    cav = lat.elements[1]
    assert cav.voltage_rel == 0.0 and cav.phase_offset == 0.0


def test_tracked_runner_refuses_fast_mode():
    tc = TrainConfig(bunch_frequency_MHz=F, pattern=PulsePattern.uniform(2),
                     mode="fast")
    with pytest.raises(ValueError, match="FastPulseRunner"):
        TrainRunner(_lattice(), _cfg(), tc)


def test_fast_runner_refuses_tracked_mode():
    tc = TrainConfig(bunch_frequency_MHz=F, pattern=PulsePattern.uniform(2))
    with pytest.raises(ValueError, match="mode='fast'"):
        FastPulseRunner(_lattice(), _cfg(), tc)


# ------------------------------------------------------- anchor 5: priors
def test_priors_composed_and_restored_fast(tmp_path):
    """M2's prior contract through the fast runner: a pre-existing
    voltage_rel/phase_offset (error study, manual setting) is COMPOSED
    into the recorded applied values and RESTORED after the run."""
    lat = _lattice()
    cav = lat.elements[1]
    cav.voltage_rel, cav.phase_offset = 0.03, 1.5
    tc = TrainConfig(bunch_frequency_MHz=F,
                     pattern=PulsePattern.from_rle("1*2"), mode="fast",
                     physics=TrainPhysics(beam_loading=True),
                     cavity_params=_sidecar(tmp_path))
    runner = FastPulseRunner(lat, _cfg(), tc)
    res = runner.run()
    assert cav.voltage_rel == pytest.approx(0.03)
    assert cav.phase_offset == pytest.approx(1.5)
    st = next(iter(runner._loading.reg.items()))[1]
    # design derived WITH the prior active (the erred cavity), as in M2
    assert st.v_design_MV == pytest.approx(V_MV * 1.03, rel=1e-9)
    assert st.phi_s_deg == pytest.approx(1.5)
    # applied slot values COMPOSE on the prior — first bunch by hand
    rec = res.fast.cavities[0]
    q_b = macro_charge_coulombs(I_MA, F, 1)
    st0 = _CavityState(mode=CavityMode(r_over_q=ROQ, q_loaded=QL),
                       frequency_MHz=F, v_design_MV=st.v_design_MV,
                       phi_s_deg=1.5)
    dv = CavityStateRegistry.induced_dv_MV(st0, q_b)
    v_tot = st.v_design_MV + 0.5 * dv
    vr_raw = abs(v_tot) / st.v_design_MV - 1.0
    po_raw = math.degrees(math.atan2(v_tot.imag, v_tot.real))
    assert rec.voltage_rel[0] == pytest.approx(
        (1.0 + vr_raw) * 1.03 - 1.0, rel=1e-9)
    assert rec.phase_offset_deg[0] == pytest.approx(1.5 + po_raw, rel=1e-9)
    # and the first-bunch energy deficit still obeys the theorem
    omega = 2 * math.pi * F * 1e6
    deficit = res.fast.w_design_exit_MeV - res.fast.w_exit_MeV[0]
    assert deficit == pytest.approx((q_b * omega * ROQ / 4) * 1e-6, rel=1e-6)


# --------------------------------------- adversarial: FREQ-jump lattice
def test_fast_matches_tracked_across_freq_jump(tmp_path):
    """Slot clock is 1/f_bunch while each cavity decays/kicks at its OWN
    omega: a 162.5 -> 325 MHz two-cavity lattice must stay fast-vs-
    tracked equivalent (for p_flag=0 RFGaps the prescribed-phase energy
    model is exact even downstream of the jump)."""
    def _lat2():
        lat = Lattice()
        lat.add(Drift("D0", 50.0, aperture=30.0))
        lat.add(RFGap(name="CAV1", voltage=V_MV, phase=-20.0, frequency=F))
        lat.add(Drift("D1", 50.0, aperture=30.0))
        lat.add(RFGap(name="CAV2", voltage=0.6, phase=-30.0,
                      frequency=2 * F))
        lat.add(Drift("D2", 50.0, aperture=30.0))
        return lat

    side = _sidecar(tmp_path, detuning_Hz=200.0)

    def _tc(mode):
        return TrainConfig(bunch_frequency_MHz=F,
                           pattern=PulsePattern.from_rle("1*4 0*9 1*4"),
                           mode=mode,
                           physics=TrainPhysics(beam_loading=True),
                           cavity_params=side)

    with pytest.warns(UserWarning, match="FREQ card"):
        res_mp, _r, vr_t, po_t, vb_t, w_t = _instrumented_tracked_run(
            _lat2(), _cfg(), _tc("mp"))
        runner_f = FastPulseRunner(_lat2(), _cfg(), _tc("fast"))
    res_f = runner_f.run()
    fs = res_f.fast
    assert len(fs.cavities) == 2
    assert fs.cavities[0].frequency_MHz == F
    assert fs.cavities[1].frequency_MHz == 2 * F
    for j in range(2):
        rec = fs.cavities[j]
        np.testing.assert_allclose(rec.voltage_rel, vr_t[j], rtol=1e-9)
        np.testing.assert_allclose(rec.phase_offset_deg, po_t[j], rtol=1e-9)
        np.testing.assert_allclose(rec.v_beam_MV, vb_t[j], rtol=1e-9)
    np.testing.assert_allclose(fs.w_exit_MeV, w_t, rtol=1e-9)


# ------------------------------------- adversarial: pinned-psi (NCells)
def test_fast_matches_tracked_ncells_pinned_psi(tmp_path):
    """A sync_phase multi-gap cavity — psi pinned by the design pass —
    must give an identical phasor recursion in fast and tracked mode;
    the energy ledger is first-order there (multi-gap RK4 response), so
    the deviation from design is compared loosely, not at float
    tolerance."""
    from linac_gen.elements.ncells import NCells

    def _latn():
        lat = Lattice()
        lat.add(Drift("D0", 30.0, aperture=30.0))
        lat.add(NCells("NC", mode=1, n_cells=4, beta_g=0.081,
                       eot_v_per_m=8e5, theta_s_deg=-40.0,
                       aperture_mm=20.0, sync_phase=True,
                       frequency_mhz=F))
        lat.add(Drift("D1", 30.0, aperture=30.0))
        return lat

    p = tmp_path / "nc.json"
    p.write_text(json.dumps({"NC*": {"r_over_q": ROQ, "q_loaded": QL}}))

    def _tc(mode):
        return TrainConfig(bunch_frequency_MHz=F,
                           pattern=PulsePattern.uniform(6), mode=mode,
                           physics=TrainPhysics(beam_loading=True),
                           cavity_params=str(p))

    cfg = _cfg(n=64)
    res_mp, _r, vr_t, po_t, vb_t, w_t = _instrumented_tracked_run(
        _latn(), cfg, _tc("mp"))
    assert "NC" in res_mp.pins
    res_f = FastPulseRunner(_latn(), cfg, _tc("fast")).run()
    assert res_f.pins["NC"] == pytest.approx(res_mp.pins["NC"])
    rec = res_f.fast.cavities[0]
    np.testing.assert_allclose(rec.voltage_rel, vr_t[0], rtol=1e-9)
    np.testing.assert_allclose(rec.phase_offset_deg, po_t[0], rtol=1e-9)
    np.testing.assert_allclose(rec.v_beam_MV, vb_t[0], rtol=1e-9)
    # centroid ledger: same sign, same magnitude scale as tracked
    dev_f = res_f.fast.w_exit_MeV - res_f.fast.w_design_exit_MeV
    dev_t = w_t - res_mp.design_result.ref_w_kin[-1]
    assert np.all(dev_t < 0) and np.all(dev_f < 0)
    np.testing.assert_allclose(dev_f, dev_t, rtol=0.3)


# ------------------------------------------- storage: stride + HDF5
def test_history_stride_and_hdf5(tmp_path):
    side = _sidecar(tmp_path)

    def _run(stride):
        tc = TrainConfig(bunch_frequency_MHz=F,
                         pattern=PulsePattern.uniform(20), mode="fast",
                         physics=TrainPhysics(beam_loading=True),
                         cavity_params=side)
        return FastPulseRunner(_lattice(), _cfg(), tc,
                               history_stride=stride).run()

    full = _run(1).fast
    dec = _run(7).fast
    # decimation touches STORAGE only — physics identical
    np.testing.assert_array_equal(dec.w_exit_MeV, full.w_exit_MeV)
    assert dec.history_slot.tolist() == [0, 7, 14]
    np.testing.assert_array_equal(dec.cavities[0].voltage_rel,
                                  full.cavities[0].voltage_rel[::7])
    np.testing.assert_array_equal(dec.cavities[0].v_beam_MV,
                                  full.cavities[0].v_beam_MV[::7])

    res = _run(1)
    out = tmp_path / "fast.h5"
    res.save_hdf5(str(out))
    import h5py
    with h5py.File(out) as f:
        g = f["train/fast"]
        assert g.attrs["n_bunches"] == 20
        assert g.attrs["history_stride"] == 1
        assert g.attrs["w_design_exit_MeV"] == res.fast.w_design_exit_MeV
        np.testing.assert_array_equal(g["w_exit_MeV"][:],
                                      res.fast.w_exit_MeV)
        c = g["cavities/c_0001"]
        assert c.attrs["name"] == "CAV1"
        np.testing.assert_array_equal(c["voltage_rel"][:],
                                      res.fast.cavities[0].voltage_rel)
        np.testing.assert_array_equal(
            c["v_beam_re_MV"][:] + 1j * c["v_beam_im_MV"][:],
            res.fast.cavities[0].v_beam_MV)
        assert f["train"].attrs["n_bunches_tracked"] == 0
        assert "bunches" not in f


# ------------------------------- fallback ledger branch: pure buncher
def test_fast_matches_tracked_buncher_fallback(tmp_path):
    """|cos(phi_s)| < 1e-6 (phi_s = -90 buncher): V_design must come from
    the sidecar and the ledger falls back to A = q * v_design_MV.  The
    fundamental theorem still fixes the first-bunch deficit (it is
    phi_s-independent), and tracked-vs-fast stays exact for the thin
    gap."""
    side = _sidecar(tmp_path, v_design_MV=V_MV)

    def _tc(mode):
        return TrainConfig(bunch_frequency_MHz=F,
                           pattern=PulsePattern.uniform(5), mode=mode,
                           physics=TrainPhysics(beam_loading=True),
                           cavity_params=side)

    res_mp, _r, vr_t, po_t, vb_t, w_t = _instrumented_tracked_run(
        _lattice(phase=-90.0), _cfg(), _tc("mp"))
    res_f = FastPulseRunner(_lattice(phase=-90.0), _cfg(), _tc("fast")).run()
    fs = res_f.fast
    rec = fs.cavities[0]
    assert rec.v_design_MV == V_MV                 # sidecar, not derived
    np.testing.assert_allclose(rec.voltage_rel, vr_t[0], rtol=1e-9)
    np.testing.assert_allclose(rec.phase_offset_deg, po_t[0], rtol=1e-9)
    np.testing.assert_allclose(rec.v_beam_MV, vb_t[0], rtol=1e-9)
    np.testing.assert_allclose(fs.w_exit_MeV, w_t, rtol=1e-9)
    q_b = macro_charge_coulombs(I_MA, F, 1)
    omega = 2 * math.pi * F * 1e6
    deficit = fs.w_design_exit_MeV - fs.w_exit_MeV[0]
    assert deficit == pytest.approx((q_b * omega * ROQ / 4) * 1e-6, rel=1e-6)


# ----------------------------------------------- abort truncates cleanly
def test_fast_abort_returns_partial(tmp_path):
    side = _sidecar(tmp_path)
    tc = TrainConfig(bunch_frequency_MHz=F,
                     pattern=PulsePattern.uniform(10), mode="fast",
                     physics=TrainPhysics(beam_loading=True),
                     cavity_params=side)
    calls = {"n": 0}

    def abort():
        calls["n"] += 1
        return calls["n"] > 3          # allow bunches 0..2, stop at i=3

    runner = FastPulseRunner(_lattice(), _cfg(), tc, should_abort=abort,
                             history_stride=2)
    with pytest.warns(UserWarning, match="aborted"):
        res = runner.run()
    fs = res.fast
    assert fs.slot.tolist() == [0, 1, 2]
    assert fs.w_exit_MeV.shape == (3,)
    assert fs.history_slot.tolist() == [0, 2]      # stride-2 records
    assert fs.cavities[0].voltage_rel.shape == (2,)
    # teardown still ran: lattice slots restored
    cav = runner.lattice.elements[1]
    assert cav.voltage_rel == 0.0 and cav.phase_offset == 0.0
    with pytest.raises(ValueError, match="history_stride"):
        FastPulseRunner(_lattice(), _cfg(), tc, history_stride=0)


# -------------------------------------- adversarial: species sign at -90
def _cfg_hminus(n=64):
    cfg = _cfg(n)
    import dataclasses
    return dataclasses.replace(cfg, species="H-")


def test_pure_buncher_negative_species_prescribed(tmp_path):
    """H- through a PRESCRIBED-phase RFGap at phi_s=-90 (pure-buncher
    amp branch, |cos| < 1e-6): the raw law dW = q V T cos(phi) keeps the
    species sign, and the fast ledger must reproduce the tracked train
    exactly (thin gap).  Guards the q_sign path of _ledger_amplitudes."""
    side = _sidecar(tmp_path, v_design_MV=V_MV)

    def _tc(mode):
        return TrainConfig(bunch_frequency_MHz=F,
                           pattern=PulsePattern.uniform(6), mode=mode,
                           physics=TrainPhysics(beam_loading=True),
                           cavity_params=side)

    cfg = _cfg_hminus()
    _res, _r, vr_t, po_t, vb_t, w_t = _instrumented_tracked_run(
        _lattice(phase=-90.0), cfg, _tc("mp"))
    res_f = FastPulseRunner(_lattice(phase=-90.0), cfg, _tc("fast")).run()
    fs = res_f.fast
    np.testing.assert_allclose(fs.cavities[0].phase_offset_deg, po_t[0],
                               rtol=1e-9, atol=0)
    np.testing.assert_allclose(fs.w_exit_MeV, w_t, rtol=1e-9, atol=0)


def test_pure_buncher_negative_species_calibrated(tmp_path):
    """H- through a psi-CALIBRATED (sync_phase pin) multi-gap cavity at
    theta_s=-90: the calibration absorbs the species charge, so the
    tracked dW(po) response is +A sin(po) — a q*V ledger amplitude flips
    its sign (the bug caught on the PIP-II MEBT demo).  Fast per-bunch
    energy deviations must carry the TRACKED sign, with first-order
    magnitude (multi-gap RK4 response, no intra-cavity feedback)."""
    from linac_gen.elements.ncells import NCells

    def _latn():
        lat = Lattice()
        lat.add(Drift("D0", 30.0, aperture=30.0))
        lat.add(NCells("NC", mode=1, n_cells=4, beta_g=0.081,
                       eot_v_per_m=8e5, theta_s_deg=-90.0,
                       aperture_mm=20.0, sync_phase=True,
                       frequency_mhz=F))
        lat.add(Drift("D1", 30.0, aperture=30.0))
        return lat

    p = tmp_path / "nc.json"
    p.write_text(json.dumps(
        {"NC*": {"r_over_q": ROQ, "q_loaded": QL, "v_design_MV": V_MV}}))

    def _tc(mode):
        return TrainConfig(bunch_frequency_MHz=F,
                           pattern=PulsePattern.uniform(6), mode=mode,
                           physics=TrainPhysics(beam_loading=True),
                           cavity_params=str(p))

    cfg = _cfg_hminus()
    res_mp, _r, vr_t, po_t, vb_t, w_t = _instrumented_tracked_run(
        _latn(), cfg, _tc("mp"))
    assert "NC" in res_mp.pins
    res_f = FastPulseRunner(_latn(), cfg, _tc("fast")).run()
    np.testing.assert_allclose(res_f.fast.cavities[0].phase_offset_deg,
                               po_t[0], rtol=1e-9, atol=0)
    dev_t = w_t - res_mp.design_result.ref_w_kin[-1]
    dev_f = (np.asarray(res_f.fast.w_exit_MeV)
             - res_f.fast.w_design_exit_MeV)
    assert np.abs(dev_t[1:]).min() > 0
    assert np.array_equal(np.sign(dev_f[1:]), np.sign(dev_t[1:]))
    # Order-of-magnitude envelope only: at a zero crossing the ledger is
    # per-cavity first-order with no intra-cavity feedback and measures
    # ~4x the tracked multi-gap response here (~1.9x on the 4-cavity
    # MEBT FieldMap chain).  The strict assertion above is the SIGN —
    # that is what the q*V bug flipped; this envelope catches gross
    # unit/convention regressions without certifying first-order
    # magnitude at pure bunchers.
    ratio = dev_f[1:] / dev_t[1:]
    assert np.all(ratio > 0.2) and np.all(ratio < 8.0), ratio
