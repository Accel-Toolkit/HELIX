"""M2 beam-loading anchors: analytic phasor decay, the fundamental
theorem of beam loading (half-self-kick), steady-state geometric series,
chopped-gap droop, no-match refusal, and design-voltage derivation."""
from __future__ import annotations

import json
import math

import numpy as np
import pytest

from linac_gen.core.config import BeamConfig
from linac_gen.core.lattice import Lattice
from linac_gen.elements.drift import Drift
from linac_gen.elements.rf_gap import RFGap
from linac_gen.pic.macrocharge import macro_charge_coulombs
from linac_gen.train import PulsePattern, TrainConfig, TrainPhysics, TrainRunner
from linac_gen.train.cavity_state import CavityMode, CavityStateRegistry, _CavityState

F = 162.5           # MHz
ROQ = 200.0         # Ohm
QL = 5.0e6
V_MV = 0.8
I_MA = 5.0


# ---------------------------------------------------------------- units
def _state(detuning=0.0):
    return _CavityState(mode=CavityMode(r_over_q=ROQ, q_loaded=QL,
                                        detuning_Hz=detuning),
                        frequency_MHz=F, v_design_MV=V_MV, phi_s_deg=0.0)


def test_phasor_decay_analytic():
    st = _state(detuning=300.0)
    st.v_beam = 0.01 + 0j
    dt = 2.5e-6
    CavityStateRegistry.decay(st, dt)
    tau = 2 * QL / (2 * math.pi * F * 1e6)
    expect = 0.01 * math.exp(-dt / tau) * np.exp(1j * 2 * math.pi * 300.0 * dt)
    assert st.v_beam == pytest.approx(expect, rel=1e-12)


def test_induced_phasor_magnitude_and_selfloss():
    st = _state()
    q = 1e-9
    dv = CavityStateRegistry.induced_dv_MV(st, q)
    omega = 2 * math.pi * F * 1e6
    assert abs(dv) * 1e6 == pytest.approx(omega * ROQ * q / 2, rel=1e-12)
    # fundamental theorem: self-loss q^2 omega RoQ / 4, independent of phi_s
    for phi in (0.0, -30.0, -90.0):
        st.phi_s_deg = phi
        dvp = CavityStateRegistry.induced_dv_MV(st, q)
        w_self_J = -q * 0.5 * (dvp.real * math.cos(math.radians(phi))
                               - dvp.imag * math.sin(math.radians(phi))) * 1e6
        assert w_self_J == pytest.approx(q * q * omega * ROQ / 4, rel=1e-12)


# ------------------------------------------------------------ integration
def _lattice():
    lat = Lattice()
    lat.add(Drift("D0", 50.0, aperture=30.0))
    lat.add(RFGap(name="CAV1", voltage=V_MV, phase=0.0, frequency=F))
    lat.add(Drift("D1", 50.0, aperture=30.0))
    return lat


def _cfg(n=400):
    return BeamConfig(species="proton", energy=3.0, frequency=F,
                      current=I_MA, n_particles=n, distribution="waterbag",
                      emit_nx=0.25, alpha_x=0.0, beta_x=0.4,
                      emit_ny=0.25, alpha_y=0.0, beta_y=0.4,
                      emit_z=0.15, alpha_z=0.0, beta_z=1.2)


def _sidecar(tmp_path, **extra):
    d = {"CAV*": dict(r_over_q=ROQ, q_loaded=QL, **extra)}
    p = tmp_path / "cav.json"
    p.write_text(json.dumps(d))
    return str(p)


def _run(tmp_path, rle, **extra):
    tc = TrainConfig(bunch_frequency_MHz=F,
                     pattern=PulsePattern.from_rle(rle),
                     physics=TrainPhysics(beam_loading=True),
                     cavity_params=_sidecar(tmp_path, **extra))
    lat = _lattice()
    runner = TrainRunner(lat, _cfg(), tc)
    res = runner.run()
    return res, lat, runner


def test_first_bunch_selfkick_energy(tmp_path):
    """First bunch: energy gain reduced vs unloaded by exactly the
    half-self-kick (theorem), at crest phi_s = 0."""
    res, lat, runner = _run(tmp_path, "1*1")
    w_loaded = res.bunch_results[0].ref_w_kin[-1]
    w_design = res.design_result.ref_w_kin[-1]
    q_b = macro_charge_coulombs(I_MA, F, 400) * 400
    omega = 2 * math.pi * F * 1e6
    dw_expected_MeV = (q_b * omega * ROQ / 4) * 1e-6
    assert w_design - w_loaded == pytest.approx(dw_expected_MeV, rel=1e-6)


def test_steady_state_geometric_series(tmp_path):
    """Uniform train: accumulated V_beam follows the geometric series."""
    n = 12
    res, lat, runner = _run(tmp_path, f"1*{n}")
    st = next(iter(runner._loading.reg.items()))[1]
    q_b = macro_charge_coulombs(I_MA, F, 400) * 400
    dv = CavityStateRegistry.induced_dv_MV(
        _state(), q_b)
    tau = 2 * QL / (2 * math.pi * F * 1e6)
    r = math.exp(-(1.0 / (F * 1e6)) / tau)
    expect = dv * (1 - r ** n) / (1 - r)
    # v_beam holds n full kicks, decayed; compare geometric sum
    assert st.v_beam == pytest.approx(expect, rel=1e-9)


def test_chopped_gap_pure_decay(tmp_path):
    """A gap of G slots decays V_beam by exactly exp(-G T/tau)."""
    res1, _, run1 = _run(tmp_path, "1*3")
    v3 = next(iter(run1._loading.reg.items()))[1].v_beam
    res2, _, run2 = _run(tmp_path, "1*3 0*40 1*1")
    st2 = next(iter(run2._loading.reg.items()))[1]
    tau = 2 * QL / (2 * math.pi * F * 1e6)
    r_gap = math.exp(-(41.0 / (F * 1e6)) / tau)
    q_b = macro_charge_coulombs(I_MA, F, 400) * 400
    dv = CavityStateRegistry.induced_dv_MV(_state(), q_b)
    expect = v3 * r_gap + dv
    assert st2.v_beam == pytest.approx(expect, rel=1e-9)


def test_bunch_by_bunch_droop_monotone(tmp_path):
    """Along a uniform train the per-bunch exit energy droops
    monotonically toward steady state (the sawtooth DC level)."""
    res, _, _ = _run(tmp_path, "1*10")
    w = np.array([r.ref_w_kin[-1] for r in res.bunch_results])
    assert np.all(np.diff(w) < 0)
    assert np.diff(w)[0] < 0 and abs(np.diff(w)[-1]) < abs(np.diff(w)[0])


def test_lattice_restored_after_train(tmp_path):
    res, lat, _ = _run(tmp_path, "1*4")
    cav = [e for e in lat.elements if e.name == "CAV1"][0]
    assert cav.voltage_rel == 0.0 and cav.phase_offset == 0.0


def test_no_match_refused(tmp_path):
    p = tmp_path / "bad.json"
    p.write_text(json.dumps({"NOPE*": {"r_over_q": 1.0, "q_loaded": 1e6}}))
    tc = TrainConfig(bunch_frequency_MHz=F,
                     pattern=PulsePattern.uniform(2),
                     physics=TrainPhysics(beam_loading=True),
                     cavity_params=str(p))
    with pytest.raises(ValueError, match="matched no"):
        TrainRunner(_lattice(), _cfg(), tc)


def test_design_voltage_derived(tmp_path):
    """V_design derived from the design pass equals the RFGap voltage
    (phi_s = 0, TTF-free thin gap)."""
    res, lat, runner = _run(tmp_path, "1*1")
    st = next(iter(runner._loading.reg.items()))[1]
    assert st.v_design_MV == pytest.approx(V_MV, rel=1e-9)
    assert st.phi_s_deg == pytest.approx(0.0)


def test_priors_composed_and_restored(tmp_path):
    """Adversarial check 2: a cavity carrying a pre-existing
    voltage_rel/phase_offset (error study, manual setting) must get the
    loading COMPOSED on top during the train and the prior RESTORED
    after — never clobbered to zero."""
    lat = _lattice()
    cav = [e for e in lat.elements if e.name == "CAV1"][0]
    cav.voltage_rel, cav.phase_offset = 0.03, 1.5
    tc = TrainConfig(bunch_frequency_MHz=F,
                     pattern=PulsePattern.from_rle("1*2"),
                     physics=TrainPhysics(beam_loading=True),
                     cavity_params=_sidecar(tmp_path))
    runner = TrainRunner(lat, _cfg(), tc)
    res = runner.run()
    assert cav.voltage_rel == pytest.approx(0.03)
    assert cav.phase_offset == pytest.approx(1.5)
    # design voltage was derived WITH the prior active (erred cavity)
    st = next(iter(runner._loading.reg.items()))[1]
    # dW = V*1.03*cos(phi_s + 1.5deg); the frame phase IS the effective
    # phase (1.5deg), so the derivation returns exactly V*1.03.
    assert st.v_design_MV == pytest.approx(V_MV * 1.03, rel=1e-9)
    # and the first bunch's deficit still obeys the theorem
    w_loaded = res.bunch_results[0].ref_w_kin[-1]
    w_design = res.design_result.ref_w_kin[-1]
    q_b = macro_charge_coulombs(I_MA, F, 400) * 400
    omega = 2 * math.pi * F * 1e6
    assert w_design - w_loaded == pytest.approx(
        (q_b * omega * ROQ / 4) * 1e-6, rel=1e-6)
