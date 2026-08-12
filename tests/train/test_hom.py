"""M4 dipole-HOM / cumulative-BBU anchors.

1. Zero-HOM identity: hom on with missing/empty mode tables refused
   loudly; hom OFF with hom tables merely PRESENT in the sidecar is
   bit-identical to M3 (parsing alone perturbs nothing).
2. Single-cavity single-mode: phasor excitation magnitude, decay/rotation
   and kick pinned by an independent hand computation (plus the
   destabilizing kick SIGN and the Z^2 species invariance at unit level).
3. THE Delayen anchor: on the reduced BBU lattice (identical thin
   cavities, pure drifts, no focusing, no acceleration) the fast-mode
   recursion must match the module's independent explicit-wake-sum
   reference (`delayen_bbu_reference`) to ~1e-9 for (a) a steady
   periodic train, (b) a transient truncated-train profile and (c) every
   point of an on/off-resonance sweep — which must also show the
   exquisite on-resonance structure (exact train-harmonic null vs
   near-resonance peak orders of magnitude apart).
4. Tracked-vs-fast consistency (one implementation, two drivers):
   lossless short train — exact at 1e-9 on the drift lattice, documented
   bracket for the accelerating lattice where the fast perturbation
   transport is drift-only.
5. Kick sign produces GROWTH: the first kicked bunch is deflected in the
   direction of the source offset (deflection amplifies — daisy-chain
   cumulative BBU) and the deviation grows along the train and along the
   cavity chain.
6. Polarization decoupling: orthogonal modes leave x and y evolutions
   independent.
"""
from __future__ import annotations

import json
import math

import numpy as np
import pytest

from linac_gen.core.config import BeamConfig
from linac_gen.core.constants import C_LIGHT
from linac_gen.core.lattice import Lattice
from linac_gen.elements.drift import Drift
from linac_gen.elements.rf_gap import RFGap
from linac_gen.pic.macrocharge import macro_charge_coulombs
from linac_gen.train import (FastPulseRunner, PulsePattern, TrainConfig,
                             TrainPhysics, TrainRunner)
from linac_gen.train.cavity_state import CavityStateRegistry, _CavityState
from linac_gen.train.cavity_state import CavityMode
from linac_gen.train.hom import (HomManager, HOMMode, _HomState,
                                 delayen_bbu_reference)

F = 162.5           # MHz bunch frequency
I_MA = 5.0
V_MV = 0.8
MASS_P = 938.27208816


# --------------------------------------------------------------- helpers
def _lattice(n_cav=6, drift_mm=250.0, voltage=0.0, phase=0.0,
             first_drift_mm=300.0):
    """The reduced BBU lattice: identical thin RF gaps separated by pure
    drifts.  voltage=0 makes every gap a transverse no-op, so the only
    transverse physics is drifts + HOM kicks — the Delayen model
    contract."""
    lat = Lattice()
    lat.add(Drift("D0", first_drift_mm))
    for k in range(n_cav):
        lat.add(RFGap(name=f"CAV{k + 1}", voltage=voltage, phase=phase,
                      frequency=F))
        lat.add(Drift(f"D{k + 1}", drift_mm))
    return lat


def _cfg(n=300, x0=1.0, y0=0.0):
    return BeamConfig(species="proton", energy=3.0, frequency=F,
                      current=I_MA, n_particles=n, distribution="waterbag",
                      emit_nx=0.25, alpha_x=0.0, beta_x=0.4,
                      emit_ny=0.25, alpha_y=0.0, beta_y=0.4,
                      emit_z=0.15, alpha_z=0.0, beta_z=1.2,
                      centroid_x=x0, centroid_y=y0)


def _hom_sidecar(tmp_path, modes, fundamental=False, name="hom.json"):
    entry = {"hom_modes": modes}
    if fundamental:
        entry.update(r_over_q=200.0, q_loaded=5.0e6)
    p = tmp_path / name
    p.write_text(json.dumps({"CAV*": entry}))
    return str(p)


def _mode(f_MHz, roqt=500.0, ql=1.0e5, pol=0.0):
    return dict(f_MHz=f_MHz, r_over_q_t=roqt, q_loaded=ql,
                polarization_deg=pol)


def _tc(pattern, side, mode="fast", loading=False, hom=True):
    return TrainConfig(bunch_frequency_MHz=F, pattern=pattern, mode=mode,
                       physics=TrainPhysics(beam_loading=loading, hom=hom),
                       cavity_params=side)


def _hom_recs(res):
    return [c for c in res.fast.cavities if c.centroid is not None]


def _q_signed(z=1.0):
    return macro_charge_coulombs(I_MA, F, 1) * z


# ------------------------------------------------- 1. zero-HOM identity
def test_hom_on_without_mode_table_refused(tmp_path):
    """physics.hom demands an explicit per-cavity dipole-mode table —
    a missing OR empty hom_modes list is refused loudly (plan 3b: physics
    inputs are never silently defaulted)."""
    p_missing = tmp_path / "nohom.json"
    p_missing.write_text(json.dumps({"CAV*": {"r_over_q": 200.0,
                                              "q_loaded": 5e6}}))
    tc = _tc(PulsePattern.uniform(2), str(p_missing), mode="mp")
    with pytest.raises(ValueError, match="hom_modes.*missing"):
        TrainRunner(_lattice(), _cfg(), tc)
    p_empty = tmp_path / "emptyhom.json"
    p_empty.write_text(json.dumps({"CAV*": {"hom_modes": []}}))
    tc = _tc(PulsePattern.uniform(2), str(p_empty), mode="mp")
    with pytest.raises(ValueError, match="hom_modes.*empty"):
        TrainRunner(_lattice(), _cfg(), tc)
    # no cavity_params at all is caught at TrainConfig construction
    with pytest.raises(ValueError, match="cavity_params"):
        TrainConfig(bunch_frequency_MHz=F, pattern=PulsePattern.uniform(2),
                    physics=TrainPhysics(hom=True))
    # and hom + envelope is an explicit non-feature
    side = _hom_sidecar(tmp_path, [_mode(650.0)])
    with pytest.raises(NotImplementedError, match="envelope"):
        TrainRunner(_lattice(), _cfg(),
                    _tc(PulsePattern.uniform(2), side, mode="envelope"))
    # unmatched sidecar names refuse loudly and name the hom channel
    p_bad = tmp_path / "bad.json"
    p_bad.write_text(json.dumps({"NOPE*": {"hom_modes": [_mode(650.0)]}}))
    with pytest.raises(ValueError, match="hom.*matched no"):
        TrainRunner(_lattice(), _cfg(),
                    _tc(PulsePattern.uniform(2), str(p_bad), mode="mp"))


def test_hom_off_is_m3_bit_identical(tmp_path):
    """hom OFF must be bit-identical to M3 even when the sidecar CARRIES
    hom tables: parsing them binds inert state and nothing else.  Run
    the same beam-loading train with and without hom_modes present and
    require bitwise-equal loading histories and energies (fast) and
    bitwise-equal per-bunch exit energies (tracked)."""
    plain = tmp_path / "plain.json"
    plain.write_text(json.dumps({"CAV*": {"r_over_q": 200.0,
                                          "q_loaded": 5e6}}))
    withhom = tmp_path / "withhom.json"
    withhom.write_text(json.dumps(
        {"CAV*": {"r_over_q": 200.0, "q_loaded": 5e6,
                  "hom_modes": [_mode(650.7)]}}))
    pat = PulsePattern.from_rle("1*6 0*3 1*4")

    def _fast(side):
        tc = _tc(pat, side, mode="fast", loading=True, hom=False)
        return FastPulseRunner(_lattice(n_cav=2, voltage=V_MV), _cfg(),
                               tc, sc_config="off").run()

    a, b = _fast(str(plain)), _fast(str(withhom))
    np.testing.assert_array_equal(a.fast.w_exit_MeV, b.fast.w_exit_MeV)
    for ra, rb in zip(a.fast.cavities, b.fast.cavities):
        np.testing.assert_array_equal(ra.voltage_rel, rb.voltage_rel)
        np.testing.assert_array_equal(ra.v_beam_MV, rb.v_beam_MV)
        # hom tables were parsed but stayed inert — no records
        assert ra.centroid is None and rb.centroid is None
        assert rb.hom_modes == ()

    def _tracked(side):
        tc = _tc(pat, side, mode="mp", loading=True, hom=False)
        res = TrainRunner(_lattice(n_cav=2, voltage=V_MV), _cfg(), tc,
                          sc_config="off").run()
        return np.array([r.ref_w_kin[-1] for r in res.bunch_results])

    np.testing.assert_array_equal(_tracked(str(plain)),
                                  _tracked(str(withhom)))


# ------------------------------- 2. single-cavity single-mode analytics
def test_single_mode_excitation_decay_kick_analytic():
    """Unit anchor a la M2 test 1, with every expectation computed BY
    HAND in the test (no module helpers): excitation magnitude
    q*u*omega^2*(R/Q)_t/(2c), phasor decay exp(-dt/tau) and rotation
    exp(i*omega*dt) at the MODE's own frequency, kick
    1e3*Z*Im[w]/(bg*m), polarization projection, destabilizing sign, and
    Z^2 species invariance."""
    f_h, roqt, ql, pol = 657.13, 50.0, 2.0e4, 30.0
    mode = HOMMode(f_MHz=f_h, r_over_q_t=roqt, q_loaded=ql,
                   polarization_deg=pol)
    st = _CavityState(mode=CavityMode(r_over_q=0.0, q_loaded=0.0),
                      frequency_MHz=F, hom=[_HomState(mode=mode)])
    mgr = HomManager(CavityStateRegistry(), F)
    q, x, y, bg, z = 2.0e-10, 1.5, -0.4, 0.08, 1.0

    # --- passage 1: no stored wake -> zero kick; excitation by hand
    dxp, dyp = mgr.hom_passage(st, 0, q, x, y, bg, MASS_P, z)
    assert dxp == 0.0 and dyp == 0.0
    omega = 2.0 * math.pi * f_h * 1e6
    kappa = omega ** 2 * roqt / (2.0 * C_LIGHT)          # V/(C m)
    a = math.radians(pol)
    u_m = (x * math.cos(a) + y * math.sin(a)) * 1e-3
    w1_expect = q * u_m * kappa * 1e-6                   # MV, purely real
    assert st.hom[0].w == pytest.approx(w1_expect, rel=1e-12)
    assert st.hom[0].w.imag == 0.0                       # no self-kick

    # --- decay/rotation over a 7-slot gap, then the kick on passage 2
    dt = 7.0 / (F * 1e6)
    tau = 2.0 * ql / omega
    w2_expect = w1_expect * math.exp(-dt / tau) * complex(
        math.cos(omega * dt), math.sin(omega * dt))
    dxp, dyp = mgr.hom_passage(st, 7, q, 0.0, 0.0, bg, MASS_P, z)
    kick = 1e3 * z * w2_expect.imag / (bg * MASS_P)
    assert dxp == pytest.approx(kick * math.cos(a), rel=1e-12)
    assert dyp == pytest.approx(kick * math.sin(a), rel=1e-12)
    # passage at zero offset excites nothing further
    assert st.hom[0].w == pytest.approx(w2_expect, rel=1e-12)

    # --- destabilizing SIGN at unit level: source at +x, sin(omega dt)>0
    # -> trailing kick along +x (deflection toward the source's side)
    st2 = _CavityState(mode=CavityMode(r_over_q=0.0, q_loaded=0.0),
                       frequency_MHz=F,
                       hom=[_HomState(mode=HOMMode(f_MHz=f_h,
                                                   r_over_q_t=roqt,
                                                   q_loaded=ql))])
    mgr.hom_passage(st2, 0, q, +2.0, 0.0, bg, MASS_P, z)
    assert math.sin(omega * 1.0 / (F * 1e6)) > 0.0       # chosen lobe
    dxp2, _ = mgr.hom_passage(st2, 1, q, 0.0, 0.0, bg, MASS_P, z)
    assert dxp2 > 0.0

    # --- Z^2 invariance: an H-minus-like species (z=-1, signed charge
    # flipped) must be deflected the SAME way (wake ~ q_source*q_test)
    st3 = _CavityState(mode=CavityMode(r_over_q=0.0, q_loaded=0.0),
                       frequency_MHz=F,
                       hom=[_HomState(mode=HOMMode(f_MHz=f_h,
                                                   r_over_q_t=roqt,
                                                   q_loaded=ql))])
    mgr.hom_passage(st3, 0, -q, +2.0, 0.0, bg, MASS_P, -1.0)
    dxp3, _ = mgr.hom_passage(st3, 1, -q, 0.0, 0.0, bg, MASS_P, -1.0)
    assert dxp3 == pytest.approx(dxp2, rel=1e-12)


# --------------------------------------------------- 3. Delayen anchor
def _run_fast_bbu(tmp_path, pattern, f_h, n_cav=6, roqt=500.0, ql=1.0e5,
                  n_particles=200, name="bbu.json"):
    side = _hom_sidecar(tmp_path, [_mode(f_h, roqt=roqt, ql=ql)], name=name)
    runner = FastPulseRunner(_lattice(n_cav=n_cav), _cfg(n=n_particles),
                             _tc(pattern, side), sc_config="off")
    res = runner.run()
    recs = _hom_recs(res)
    assert len(recs) == n_cav
    fast_cent = np.stack([c.centroid for c in recs])     # (ncav, nb, 4)
    states = [st for _k, st in runner._registry.items()]
    s_mm = np.array([st.s_design_mm for st in states])
    bgs = np.array([st.bg_design for st in states])
    # no acceleration on the reduced lattice: bg identical everywhere
    assert np.ptp(bgs) == 0.0
    ref = delayen_bbu_reference(
        pattern.filled_slots, 1.0 / (F * 1e6), _q_signed(), 1.0,
        list(recs[0].hom_modes), s_mm, states[0].centroid_design,
        float(bgs[0]), MASS_P)
    return fast_cent, ref, states


def test_delayen_anchor_steady_periodic(tmp_path):
    """(a) steady periodic train: the fast recursion equals the
    independent explicit-wake-sum reference at every cavity, every
    bunch, all four centroid components."""
    fast_cent, ref, _ = _run_fast_bbu(
        tmp_path, PulsePattern.uniform(40), f_h=4.005 * F)
    np.testing.assert_allclose(fast_cent, ref, rtol=1e-9, atol=1e-12)
    # and the wake actually did something (guard against a null anchor)
    assert np.max(np.abs(fast_cent[-1, :, 1] - ref[-1, 0, 1])) > 1e-7


def test_delayen_anchor_transient_profile(tmp_path):
    """(b) transient truncated-train profile (Delayen's arbitrary
    current profile): gaps decay the wake, refills re-excite it — the
    explicit sums see exactly the same arrival times."""
    pat = PulsePattern.from_rle("1*6 0*5 1*12 0*3 1*10")
    fast_cent, ref, _ = _run_fast_bbu(tmp_path, pat, f_h=4.005 * F)
    np.testing.assert_allclose(fast_cent, ref, rtol=1e-9, atol=1e-12)


def test_delayen_anchor_resonance_sweep(tmp_path):
    """(c) ON vs OFF resonance — the adversarial checkpoint.  Sweep
    f_HOM around the 4th train harmonic: growth must show the exquisite
    on-resonance structure (the EXACT harmonic is a point-bunch null:
    every arrival samples sin(2*pi*h*k) = 0, so a rigid train couples
    to nothing; detuning by ~1/(n_train*T) turns the wake fully on;
    far detuning partially cancels), and fast-vs-reference equality
    must hold at EVERY swept point (atol floors the null where both
    paths are pure float noise)."""
    deltas = [0.0, 0.002, 0.005, 0.01, 0.05, 0.25]
    growth = {}
    for i, d in enumerate(deltas):
        pat = PulsePattern.uniform(40)
        fast_cent, ref, states = _run_fast_bbu(
            tmp_path, pat, f_h=(4.0 + d) * F, name=f"sweep{i}.json")
        np.testing.assert_allclose(fast_cent, ref, rtol=1e-9, atol=1e-12)
        x_des = states[-1].centroid_design[0]
        growth[d] = float(np.max(np.abs(fast_cent[-1, :, 0] - x_des)))
    g_peak = max(growth.values())
    # the peak is a real signal, far above float noise (measured 5.6e-4)
    assert g_peak > 1e-6
    # EXACT harmonic: point-bunch null — every kick is Im[real phasor]
    # ~ 1e-15-scale, invisible against the mm-scale design centroid
    # (measured exactly 0.0 here; allow generous float headroom)
    assert growth[0.0] < 1e-8 * g_peak
    # rising flank toward the peak at delta ~ 0.4/n_train
    assert growth[0.002] < growth[0.005] < growth[0.01]
    assert growth[0.01] == pytest.approx(g_peak)
    # fall-off outside the train-length bandwidth ~ 1/(n T) ...
    assert growth[0.05] < 0.3 * g_peak
    # ... and far off resonance the alternating phases nearly cancel
    assert growth[0.25] < 0.06 * g_peak


# ------------------------------------- 4. tracked-vs-fast consistency
def _instrumented_tracked_hom(lat, cfg, tc):
    """Tracked TrainRunner run snapshotting each HOM cavity's phasor
    list after every bunch (the registry is live until teardown)."""
    runner = TrainRunner(lat, cfg, tc, sc_config="off")
    states = [st for _k, st in runner._registry.items() if st.hom]
    snaps = []

    def snap(_i, _n):
        snaps.append([[h.w for h in st.hom] for st in states])

    runner.progress_callback = snap
    res = runner.run()
    # (ncav, nmodes, nbunches)
    w = np.array(snaps, complex).transpose(1, 2, 0)
    return res, runner, w


def test_tracked_vs_fast_drift_lattice_exact(tmp_path):
    """10 bunches, 3 cavities, V=0 gaps (pure drift transport, lossless):
    the tracked driver's per-bunch mean-of-alive centroid model and the
    fast per-bunch centroid recursion are the SAME model here, so the
    per-cavity phasor histories must agree to float tolerance — the
    one-implementation-two-drivers invariant for the hom channel."""
    side = _hom_sidecar(tmp_path, [_mode(4.005 * F)])
    pat = PulsePattern.uniform(10)
    lat = _lattice(n_cav=3)
    res_t, run_t, w_t = _instrumented_tracked_hom(
        lat, _cfg(), _tc(pat, side, mode="mp"))
    assert all(len(r.ref_w_kin) for r in res_t.bunch_results)
    res_f = FastPulseRunner(_lattice(n_cav=3), _cfg(),
                            _tc(pat, side, mode="fast"),
                            sc_config="off").run()
    recs = _hom_recs(res_f)
    w_f = np.stack([r.hom_w for r in recs])              # (ncav, nm, nb)
    np.testing.assert_allclose(w_f, w_t, rtol=1e-9, atol=0)
    # lossless bracket contract of the comparison
    for r in res_t.bunch_results:
        assert r.transmission[-1] == pytest.approx(100.0)


def test_tracked_vs_fast_accelerating_bracket(tmp_path):
    """Accelerating lattice (V=0.8 MV, phi=-20): the fast perturbation
    transport is drift-only while the tracked centroid feels adiabatic
    damping and RF defocusing inside the gaps — the DOCUMENTED
    centroid-model tolerance.  The per-bunch excitation offsets u_n
    (recovered from the phasor increments) must agree to the bracket
    below; the design (bunch-1) offsets are exact by construction.

    Bracket: |du_fast - du_tracked| <= 25% of max|du| + 1e-9 mm floor,
    where du_n = u_n - u_1 is the BBU-driven part of the offset (the
    perturbation the drift-only transport approximates).  Measured on
    this lattice: 9.5% — the bracket carries ~2.6x headroom."""
    f_h = 4.005 * F
    side = _hom_sidecar(tmp_path, [_mode(f_h, roqt=2000.0)])
    pat = PulsePattern.uniform(10)
    cfg = _cfg()
    res_t, run_t, w_t = _instrumented_tracked_hom(
        cfg=cfg, lat=_lattice(n_cav=3, voltage=V_MV, phase=-20.0),
        tc=_tc(pat, side, mode="mp"))
    res_f = FastPulseRunner(_lattice(n_cav=3, voltage=V_MV, phase=-20.0),
                            cfg, _tc(pat, side, mode="fast"),
                            sc_config="off").run()
    recs = _hom_recs(res_f)
    w_f = np.stack([r.hom_w for r in recs])
    mode = recs[0].hom_modes[0]
    om, tau = mode.omega, mode.tau_s
    T = 1.0 / (F * 1e6)
    dec = np.exp((1j * om - 1.0 / tau) * T)
    kq = _q_signed() * mode.kappa_V_per_C_m * 1e-6 * 1e-3   # per mm

    def offsets(w):
        # u_n [mm] = (w_n - w_{n-1} * decay) / (q * kappa)
        prev = np.concatenate([np.zeros_like(w[:, :, :1]),
                               w[:, :, :-1]], axis=2)
        return ((w - prev * dec) / kq).real
    u_t, u_f = offsets(w_t), offsets(w_f)
    # bunch 1 sees the pure design centroid in both drivers: exact
    np.testing.assert_allclose(u_f[:, :, 0], u_t[:, :, 0],
                               rtol=1e-9, atol=1e-12)
    du_t = u_t - u_t[:, :, :1]
    du_f = u_f - u_f[:, :, :1]
    scale = np.max(np.abs(du_t))
    assert scale > 1e-6                    # the BBU signal is real
    assert np.max(np.abs(du_f - du_t)) <= 0.25 * scale + 1e-9


# --------------------------------------------- 5. kick sign -> growth
def test_kick_sign_growth(tmp_path):
    """The sign convention must AMPLIFY: (i) the first kicked bunch is
    deflected toward the source's offset side (+x here — with no
    focusing a flipped sign would displace bunches in -x instead of
    +x); (ii) deviations grow along the train (resonant buildup);
    (iii) deviations grow along the cavity chain (cumulative BBU)."""
    pat = PulsePattern.uniform(80)
    fast_cent, ref, states = _run_fast_bbu(
        tmp_path, pat, f_h=4.004 * F, n_cav=8, roqt=2.0e4)
    x_des = np.array([st.centroid_design[0] for st in states])
    xp_des = np.array([st.centroid_design[1] for st in states])
    dxp = fast_cent[:, :, 1] - xp_des[:, None]
    dx = fast_cent[:, :, 0] - x_des[:, None]
    # (i) first kicked bunch, first cavity: deflection along +x
    assert math.sin(2 * math.pi * 4.004) > 0        # chosen wake lobe
    assert dxp[0, 1] > 0.0
    # (ii) growth along the train: envelope of |dx| at the last cavity
    g = np.abs(dx[-1])
    assert g[20] < g[40] < g[60] < g[79]
    assert g[79] > 30.0 * max(g[10], 1e-15)
    # (iii) growth along the cavity chain for the late train
    tail = np.abs(dx[:, -1])
    assert tail[-1] > 10.0 * tail[1]
    assert np.all(np.diff(tail[1:]) > 0.0)


# ------------------------------------------ 6. polarization decoupling
def test_polarization_decoupling(tmp_path):
    """Two orthogonal dipole modes (0 and 90 deg): the x evolution with
    both modes present equals the x-mode-only run, and likewise for y —
    the planes evolve independently (cross-leakage only at the 1e-16
    trig-of-90-degrees floor)."""
    f_h = 4.005 * F
    pat = PulsePattern.uniform(30)

    def _run(modes, name):
        side = _hom_sidecar(tmp_path, modes, name=name)
        res = FastPulseRunner(_lattice(n_cav=4), _cfg(x0=1.2, y0=0.5),
                              _tc(pat, side), sc_config="off").run()
        return np.stack([c.centroid for c in _hom_recs(res)])

    both = _run([_mode(f_h, pol=0.0), _mode(f_h, pol=90.0)], "both.json")
    only_x = _run([_mode(f_h, pol=0.0)], "onlyx.json")
    only_y = _run([_mode(f_h, pol=90.0)], "onlyy.json")
    # x plane: identical with and without the orthogonal y mode
    np.testing.assert_allclose(both[:, :, :2], only_x[:, :, :2],
                               rtol=1e-9, atol=1e-12)
    # y plane: identical with and without the orthogonal x mode
    np.testing.assert_allclose(both[:, :, 2:], only_y[:, :, 2:],
                               rtol=1e-9, atol=1e-12)
    # and both planes really moved (nonzero seeds in both)
    assert np.max(np.abs(both[-1, :, 1] - both[-1, 0, 1])) > 1e-9
    assert np.max(np.abs(both[-1, :, 3] - both[-1, 0, 3])) > 1e-9
    # x-only run leaves y perturbation at absolute float floor
    dyp = only_x[:, :, 3] - only_x[:, 0:1, 3]
    assert np.max(np.abs(dyp)) < 1e-12


# ----------------------------------------- combined channels + storage
def test_combined_loading_and_hom_tracked(tmp_path):
    """Both channels through the SAME chained entry hook: loading still
    obeys the M2 first-bunch theorem while hom kicks the centroid; the
    lattice slots are restored at teardown (hom itself never touches
    element attributes)."""
    side = _hom_sidecar(tmp_path, [_mode(4.005 * F, roqt=2000.0)],
                        fundamental=True)
    pat = PulsePattern.uniform(4)
    lat = _lattice(n_cav=3, voltage=V_MV, phase=0.0)
    tc = _tc(pat, side, mode="mp", loading=True, hom=True)
    runner = TrainRunner(lat, _cfg(), tc, sc_config="off")
    res = runner.run()
    # loading: first-bunch energy deficit = fundamental-theorem self-loss
    q_b = macro_charge_coulombs(I_MA, F, 300) * 300
    omega = 2 * math.pi * F * 1e6
    w_design = res.design_result.ref_w_kin[-1]
    w1 = res.bunch_results[0].ref_w_kin[-1]
    n_cav = 3
    assert w_design - w1 == pytest.approx(
        n_cav * (q_b * omega * 200.0 / 4) * 1e-6, rel=1e-5)
    # hom: the registry accumulated a wake at every cavity
    for _k, st in runner._registry.items():
        assert st.hom and abs(st.hom[0].w) > 0.0
    # teardown restored the loading slots
    for el in lat.elements:
        if isinstance(el, RFGap):
            assert el.voltage_rel == 0.0 and el.phase_offset == 0.0


def test_hom_hdf5_roundtrip(tmp_path):
    """Fast-mode HOM records reach the HDF5 sidecar schema: centroid,
    phasor history and the mode table."""
    side = _hom_sidecar(tmp_path, [_mode(4.005 * F)])
    res = FastPulseRunner(_lattice(n_cav=2), _cfg(),
                          _tc(PulsePattern.uniform(6), side),
                          sc_config="off").run()
    out = tmp_path / "hom.h5"
    res.save_hdf5(str(out))
    import h5py
    with h5py.File(out) as f:
        g = f["train/fast/cavities"]
        keys = sorted(g.keys())
        assert len(keys) == 2
        c = g[keys[0]]
        rec = _hom_recs(res)[0]
        np.testing.assert_array_equal(c["centroid"][:], rec.centroid)
        np.testing.assert_array_equal(
            c["hom_w_re_MV"][:] + 1j * c["hom_w_im_MV"][:], rec.hom_w)
        assert c["hom_modes/f_MHz"][:] == pytest.approx([4.005 * F])
        assert c["hom_modes/polarization_deg"][:] == pytest.approx([0.0])


def test_design_beam_death_before_hom_cavity_refused(tmp_path):
    """A design beam that dies upstream of a HOM-bound cavity leaves the
    excitation offset undefined — loud refusal, not a silent skip."""
    side = _hom_sidecar(tmp_path, [_mode(650.0)])
    lat = Lattice()
    lat.add(Drift("D0", 300.0))
    lat.add(Drift("KILL", 50.0, aperture=1e-6))          # scrapes all
    lat.add(RFGap(name="CAV1", voltage=0.0, phase=0.0, frequency=F))
    lat.add(Drift("D1", 250.0))
    tc = _tc(PulsePattern.uniform(2), side, mode="fast")
    with pytest.raises(ValueError, match="never reached"):
        FastPulseRunner(lat, _cfg(), tc, sc_config="off").run()
