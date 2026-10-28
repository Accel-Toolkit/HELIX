"""Ring-mode programs (milestone M3, L0): the momentum, voltage and phase
programs, the phase rule with its transition mirror, the cap and the refusal,
paraphasing (both branches and the refusal), what a turn writes into the node
table, the reference-change node's exact map, and the running totals over a
ramp — the turn times add up to the clock, the gains to the energy, the
transverse determinants to p_start / p_end."""
from __future__ import annotations

import copy
import math
from pathlib import Path

import numpy as np
import pytest

from linac_gen.cli.common import load_lattice
from linac_gen.ring.compile import compile_from_config
from linac_gen.ring.config import RingConfig
from linac_gen.ring.kernels import nodes as N
from linac_gen.ring.kernels.cartesian import energy_from_delta
from linac_gen.ring.programs import (
    C_LIGHT,
    EnergyProgram,
    PhaseProgram,
    Ramp,
    RampConfig,
    RampError,
    VoltageProgram,
)

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "simple_booster.madx"
MASS = 938.27208816

#: a generic resonant cycle and voltage table (not any machine's)
SINE = {"kind": "sine", "start_kinetic_mev": 800.0, "end_kinetic_mev": 6000.0,
        "frequency_hz": 15.0, "t_min_s": 0.0005, "t_start_s": 0.0008}
VTABLE = {"kind": "table", "times_s": [0.0, 0.003, 0.03], "values_mv": [0.4, 1.2, 1.2]}


def _p(w):
    return math.sqrt(w * w + 2.0 * w * MASS)


@pytest.fixture(scope="module")
def booster():
    return load_lattice(str(FIXTURE))


def _ramp_ring(booster, ramp, **kw):
    cfg = RingConfig(ramp=ramp, **kw)
    rl = compile_from_config(copy.deepcopy(booster), cfg)
    return rl, Ramp(RampConfig.from_dict(cfg.ramp), rl,
                    corrector_mode=kw.get("corrector_mode", "follow"))


# ---------------------------------------------------------------------------
# programs
# ---------------------------------------------------------------------------
def test_sine_program_meets_its_ends():
    ep = EnergyProgram(**SINE)
    p0 = _p(800.0)
    assert ep.momentum(0.0, MASS, p0) == p0                       # held before t_start
    assert ep.momentum(0.0008, MASS, p0) == pytest.approx(p0, rel=1e-13)
    t_end = ep.t_end()
    assert t_end == pytest.approx(0.0005 + 1.0 / 30.0, rel=1e-15)
    assert ep.momentum(t_end, MASS, p0) == pytest.approx(_p(6000.0), rel=1e-13)
    # the lowest field at t_min, below the start momentum; in between, the
    # fraction of the way from the lowest to the top is (1 - cos)/2
    _start, centre, amplitude = ep._cycle(MASS, p0)
    p_low, p_top = centre - amplitude, centre + amplitude
    assert p_low < p0 < p_top and p_top == pytest.approx(_p(6000.0), rel=1e-15)
    w = 2 * math.pi * 15.0
    for t in (0.0008, 0.004, 0.02):
        frac = (ep.momentum(t, MASS, p0) - p_low) / (p_top - p_low)
        assert frac == pytest.approx(0.5 * (1 - math.cos(w * (t - 0.0005))), rel=1e-12)
    # (start energy left out: the ring's reference)
    ep2 = EnergyProgram(**{**SINE, "start_kinetic_mev": None})
    assert ep2.momentum(0.0008, MASS, 1234.5) == pytest.approx(1234.5, rel=1e-13)


@pytest.mark.parametrize("quantity,values", [("kinetic_mev", [800.0, 900.0]),
                                             ("momentum_mevc", [1463.296, 1600.0]),
                                             ("total_mev", [1738.27, 1838.27])])
def test_table_program_is_linear_and_holds_its_ends(quantity, values):
    ep = EnergyProgram(kind="table", times_s=[0.001, 0.002], values=values, quantity=quantity)
    p = [ep.momentum(t, MASS, 1.0) for t in (0.0, 0.001, 0.0015, 0.002, 0.01)]
    assert p[0] == p[1] and p[3] == p[4]                          # ends held
    if quantity == "momentum_mevc":
        assert p[2] == pytest.approx(0.5 * (values[0] + values[1]), rel=1e-15)
    if quantity == "kinetic_mev":
        assert p[2] == pytest.approx(_p(850.0), rel=1e-15)
    if quantity == "total_mev":
        e = 0.5 * (values[0] + values[1])
        assert p[2] == pytest.approx(math.sqrt(e * e - MASS * MASS), rel=1e-14)


@pytest.mark.parametrize("bad,match", [
    ({"kind": "sinus"}, "kind"),
    ({"kind": "sine", "end_kinetic_mev": 6000.0}, "frequency"),
    ({"kind": "sine", "frequency_hz": 15.0}, "top kinetic"),
    ({"kind": "sine", "frequency_hz": 15.0, "end_kinetic_mev": 6e3, "t_min_s": 0.01,
      "t_start_s": 0.0}, "t_start"),
    ({"kind": "sine", "frequency_hz": 15.0, "end_kinetic_mev": 6e3, "start_kinetic_mev": 0.0},
     "start kinetic energy must be above 0"),
    ({"kind": "sine", "frequency_hz": 15.0, "end_kinetic_mev": float("nan")}, "finite"),
    ({"kind": "sine", "frequency_hz": True, "end_kinetic_mev": 6e3}, "not a number"),
    ({"kind": "sine", "frequency_hz": 15.0, "end_kinetic_mev": 6e3, "times_s": [0.0, 1.0],
      "values": [1.0]}, "as many"),
    ({"kind": "table", "times_s": [0.0, 0.0], "values": [1.0, 2.0]}, "increase"),
    ({"kind": "table", "times_s": [0.0], "values": [1.0, 2.0]}, "as many"),
    ({"kind": "table", "times_s": [0.0, 1.0], "values": [1.0, -2.0]}, "positive"),
    ({"kind": "table", "times_s": [0.0], "values": [1.0], "quantity": "watts"}, "quantity"),
])
def test_energy_programs_are_validated(bad, match):
    with pytest.raises(RampError, match=match):
        EnergyProgram(**bad)


def test_numbers_written_as_text_are_numbers():
    # (YAML reads 6e3 as the text '6e3')
    e = EnergyProgram(kind="sine", start_kinetic_mev="800", end_kinetic_mev="6e3",
                      frequency_hz="15", t_min_s="5e-4", t_start_s="8e-4")
    assert (e.start_kinetic_mev, e.end_kinetic_mev, e.t_min_s) == (800.0, 6000.0, 0.0005)
    assert RampConfig(gamma_t="5.4", t_start_s="4e-3").gamma_t == 5.4


def test_other_programs_and_the_config_are_validated():
    with pytest.raises(RampError, match="negative"):
        VoltageProgram(kind="table", times_s=[0.0], values_mv=[-1.0])
    # (a table kept with another kind must still load back)
    with pytest.raises(RampError, match="as many"):
        VoltageProgram(kind="constant", times_s=[0.0, 1.0], values_mv=[1.0])
    with pytest.raises(RampError, match="as many"):
        PhaseProgram(kind="auto", times_s=[0.0], values_deg=[1.0, 2.0])
    with pytest.raises(RampError, match="on_excess"):
        PhaseProgram(on_excess="ignore")
    with pytest.raises(RampError, match="unknown key"):
        RampConfig.from_dict({"energy": {"kind": "flat"}, "radial_loop": True})
    with pytest.raises(RampError, match="gamma_t"):
        RampConfig(gamma_t=0.5)
    with pytest.raises(ValueError, match="ramp"):
        RingConfig(ramp={"energy": {"kind": "sine"}})
    rc = RampConfig.from_dict({"energy": SINE, "voltage": VTABLE, "gamma_t": 5.4,
                               "paraphasing": [{"cavities": ["rf0*"], "angle_deg": 10.0}]})
    d = rc.to_dict()
    assert RampConfig.from_dict(d).to_dict() == d
    assert RingConfig(ramp=d).ramp == d


# ---------------------------------------------------------------------------
# a turn
# ---------------------------------------------------------------------------
def test_the_phase_rule_gives_the_gain_the_turn_needs(booster):
    rl, ramp = _ramp_ring(booster, {"energy": SINE, "voltage": VTABLE, "t_start_s": 0.004,
                                    "gamma_t": 5.449167323})
    t, e = ramp.start()
    pl = ramp.plan(0, t, e)
    beta = math.sqrt(1 - (MASS / e) ** 2)
    dt = rl.length / (beta * C_LIGHT)
    want = EnergyProgram(**SINE).total_energy(t + dt, MASS, rl.reference.p0c) - e
    assert pl.gain_mev == pytest.approx(want, rel=1e-14)
    assert pl.voltage_mv == pytest.approx(np.interp(0.004, VTABLE["times_s"],
                                                    VTABLE["values_mv"]), rel=1e-15)
    assert math.sin(math.radians(pl.phi_deg)) * pl.voltage_mv == pytest.approx(pl.gain_mev,
                                                                                rel=1e-14)
    assert pl.dE_mev.sum() == pytest.approx(pl.gain_mev, rel=1e-13)
    assert np.allclose(pl.lags_deg, pl.phi_deg) and not pl.above_transition and not pl.capped
    assert pl.frequency_hz == pytest.approx(84 * beta * C_LIGHT / rl.length, rel=1e-15)
    assert pl.energy_end_mev == pytest.approx(e + pl.gain_mev, rel=1e-15)


def test_above_transition_the_phase_is_mirrored(booster):
    rl, ramp = _ramp_ring(booster, {"energy": SINE, "voltage": VTABLE, "t_start_s": 0.004,
                                    "gamma_t": 1.5})              # (already above)
    t, e = ramp.start()
    pl = ramp.plan(0, t, e)
    assert pl.above_transition
    assert 90.0 < pl.phi_deg < 180.0
    assert math.sin(math.radians(pl.phi_deg)) * pl.voltage_mv == pytest.approx(pl.gain_mev,
                                                                                rel=1e-13)


def test_a_gain_beyond_the_voltage_is_capped_or_refused(booster):
    low = {"kind": "table", "times_s": [0.0], "values_mv": [0.01]}
    rl, ramp = _ramp_ring(booster, {"energy": SINE, "voltage": low, "t_start_s": 0.004})
    t, e = ramp.start()
    pl = ramp.plan(0, t, e)
    assert pl.capped and pl.phi_deg == pytest.approx(90.0) and ramp.n_capped == 1
    assert pl.dE_mev.sum() == pytest.approx(0.01, rel=1e-12)       # all the RF can give
    rl, ramp = _ramp_ring(booster, {"energy": SINE, "voltage": low, "t_start_s": 0.004,
                                    "phase": {"on_excess": "refuse"}})
    with pytest.raises(RampError, match="more than the RF gives"):
        ramp.plan(0, *ramp.start())


@pytest.mark.parametrize("above", [False, True])
def test_paraphasing_groups(booster, above):
    # half the cavities at +theta, half at -theta: R = V cos(theta), the rule's
    # phase asin(gain/R) (mirrored above transition), every cavity's lag
    # phi +- theta, the reference's gain the turn's
    # (the deck's cavities share one name: chosen by their number)
    n_cav = len(compile_from_config(copy.deepcopy(booster), RingConfig()).cavities)
    half = list(range(n_cav // 2))
    groups = [{"cavities": half, "angle_deg": 25.0},
              {"cavities": list(range(n_cav // 2, n_cav)), "angle_deg": -25.0}]
    rl, ramp = _ramp_ring(booster, {"energy": SINE, "voltage": VTABLE, "t_start_s": 0.004,
                                    "paraphasing": groups, "gamma_t": 1.5 if above else 50.0})
    pl = ramp.plan(0, *ramp.start())
    R = pl.voltage_mv * math.cos(math.radians(25.0))
    x = math.degrees(math.asin(pl.gain_mev / R))
    assert pl.phi_deg == pytest.approx(180.0 - x if above else x, abs=1e-12)
    k = len(half)
    assert np.allclose(pl.lags_deg[:k], pl.phi_deg + 25.0) and np.allclose(pl.lags_deg[k:],
                                                                         pl.phi_deg - 25.0)
    assert pl.dE_mev.sum() == pytest.approx(pl.gain_mev, rel=1e-12)
    # unequal angles: the rule subtracts the sum's argument
    groups2 = [{"cavities": half, "angle_deg": 40.0}]
    rl, ramp = _ramp_ring(booster, {"energy": SINE, "voltage": VTABLE, "t_start_s": 0.004,
                                    "paraphasing": groups2, "gamma_t": 1.5 if above else 50.0})
    pl = ramp.plan(0, *ramp.start())
    assert pl.dE_mev.sum() == pytest.approx(pl.gain_mev, rel=1e-12)
    # by name pattern (every cavity is "rfc"): one offset for all, the same gain
    rl, ramp = _ramp_ring(booster, {"energy": SINE, "voltage": VTABLE, "t_start_s": 0.004,
                                    "paraphasing": [{"cavities": ["rf*"], "angle_deg": 20.0}],
                                    "gamma_t": 1.5 if above else 50.0})
    pl = ramp.plan(0, *ramp.start())
    assert np.allclose(pl.lags_deg, pl.phi_deg + 20.0)
    assert pl.dE_mev.sum() == pytest.approx(pl.gain_mev, rel=1e-12)
    # a phase table gives the phase of the net voltage
    rl, ramp = _ramp_ring(booster, {"energy": SINE, "voltage": VTABLE, "t_start_s": 0.004,
                                    "paraphasing": groups2,
                                    "phase": {"kind": "table", "times_s": [0.0],
                                              "values_deg": [30.0]}})
    pl = ramp.plan(0, *ramp.start())
    assert pl.phi_deg + math.degrees(ramp.arg) == pytest.approx(30.0, abs=1e-12)


def test_paraphasing_refusals(booster):
    n_cav = len(compile_from_config(copy.deepcopy(booster), RingConfig()).cavities)
    half = n_cav // 2
    names = list(range(n_cav))
    with pytest.raises(RampError, match="cancel"):
        _ramp_ring(booster, {"energy": SINE, "voltage": VTABLE, "paraphasing": [
            {"cavities": names[:half], "angle_deg": 90.0},
            {"cavities": names[half:], "angle_deg": -90.0}]})
    with pytest.raises(RampError, match="beyond"):
        _ramp_ring(booster, {"energy": SINE, "paraphasing": [{"cavities": [n_cav]}]})
    with pytest.raises(RampError, match="no cavity matches"):
        _ramp_ring(booster, {"energy": SINE, "paraphasing": [{"cavities": ["nothing*"]}]})
    with pytest.raises(RampError, match="in two groups"):
        _ramp_ring(booster, {"energy": SINE, "paraphasing": [
            {"cavities": [0, 1], "angle_deg": 10.0}, {"cavities": [1, 2], "angle_deg": -10.0}]})
    # a gain beyond R (here R << V) is refused when asked
    rl, ramp = _ramp_ring(booster, {"energy": SINE, "voltage": VTABLE, "t_start_s": 0.004,
                                    "phase": {"on_excess": "refuse"}, "paraphasing": [
                                        {"cavities": names[:half], "angle_deg": 89.9},
                                        {"cavities": names[half:], "angle_deg": -89.9}]})
    with pytest.raises(RampError, match="more than the RF gives"):
        ramp.plan(0, *ramp.start())


@pytest.mark.parametrize("above", [False, True])
def test_negative_voltages_still_give_the_gain(booster, above):
    # an H- deck: its charge signs every cavity's voltage — the rule counts
    # each with an offset of pi, so the net voltage gives the gain and the RF
    # focuses as on the proton deck
    ramp_d = {"energy": SINE, "voltage": VTABLE, "t_start_s": 0.004,
              "gamma_t": 1.5 if above else 50.0}
    _, rp = _ramp_ring(booster, ramp_d)
    hm = copy.deepcopy(booster)
    hm.ring_reference_beam = dict(hm.ring_reference_beam, species="H-")
    rlh, rh = _ramp_ring(hm, ramp_d)
    assert all(c["voltage_mv"] < 0.0 for c in rlh.cavities)
    pp, ph = rp.plan(0, *rp.start()), rh.plan(0, *rh.start())
    assert ph.dE_mev.sum() == pytest.approx(ph.gain_mev, rel=1e-12) and ph.gain_mev > 0.0
    focus_p = float(np.sum(pp.volts_mv * np.cos(np.radians(pp.lags_deg))))
    focus_h = float(np.sum(ph.volts_mv * np.cos(np.radians(ph.lags_deg))))
    assert np.sign(focus_h) == np.sign(focus_p) == (-1.0 if above else 1.0)
    assert ph.phi_deg + math.degrees(rh.arg) == pytest.approx(pp.phi_deg, abs=1e-6)
    o = rh.orbit(50)
    assert o["energy_end_mev"] > o["energy_mev"][0]
    # one cavity's voltage reversed on the proton deck: the rest make up for it
    rl1 = compile_from_config(copy.deepcopy(booster), RingConfig(ramp=ramp_d))
    rl1.cavities[3]["voltage_mv"] = -rl1.cavities[3]["voltage_mv"]
    r1 = Ramp(RampConfig.from_dict(RingConfig(ramp=ramp_d).ramp), rl1)
    p1 = r1.plan(0, *r1.start())
    assert p1.dE_mev.sum() == pytest.approx(p1.gain_mev, rel=1e-12)
    assert r1.r_unit == pytest.approx(20.0 / 22.0, rel=1e-14)


def test_a_ramp_needs_one_harmonic(booster):
    ramp_d = RingConfig(ramp={"energy": SINE, "voltage": VTABLE, "t_start_s": 0.004}).ramp
    rl = compile_from_config(copy.deepcopy(booster), RingConfig(ramp=ramp_d))
    rl.cavities[2]["harmonic"] = 168
    with pytest.raises(RampError, match="harmonics \\[168\\]"):
        Ramp(RampConfig.from_dict(ramp_d), rl)
    rl.cavities[2]["harmonic"] = None
    with pytest.raises(RampError, match="fixed frequency"):
        Ramp(RampConfig.from_dict(ramp_d), rl)


def test_no_turns_to_run_and_the_transition_crossing(booster):
    from linac_gen.ring.programs import transition_crossing
    past = {"energy": {"kind": "table", "times_s": [0.0, 0.0005], "values": [800.0, 830.0]},
            "voltage": VTABLE, "t_start_s": 0.001, "gamma_t": 5.4}
    rl, ramp = _ramp_ring(booster, past)
    assert ramp.turns_to_end() == 0
    with pytest.raises(RampError, match="no turns to run: the clock at turn 0"):
        ramp.orbit(0)
    assert transition_crossing([False, False, True, True]) == 2
    assert transition_crossing([True, True, False]) == 2              # (downwards)
    assert transition_crossing([True, True]) is None
    rl, ramp = _ramp_ring(booster, {"energy": SINE, "voltage": VTABLE, "t_start_s": 0.004,
                                    "gamma_t": 1.5})
    o = ramp.orbit(5)
    assert o["transition_turn"] is None and o["above_transition_at_start"]


def test_a_turn_written_into_the_node_table(booster):
    rl, ramp = _ramp_ring(booster, {"energy": SINE, "voltage": VTABLE, "t_start_s": 0.004})
    pl = ramp.plan(0, *ramp.start())
    params, rb, rp, rg = rl.params.copy(), rl.ref_beta0.copy(), rl.ref_p0c.copy(), rl.ref_bg2.copy()
    ramp.apply(pl, params, rb, rp, rg)
    for k, c in enumerate(rl.cavities):
        o = rl.offsets[c["node"]]
        assert (params[o], params[o + 1], params[o + 2]) == (pl.volts_mv[k], pl.frequency_hz,
                                                             pl.lags_deg[k])
        j = c["refshift_node"]
        assert rl.kinds[j] == N.N_REFSHIFT and j == c["node"] + 1
        o = rl.offsets[j]
        e_after = pl.energy_after_mev[k]
        assert params[o] == pl.dE_mev[k]
        assert params[o + 1] == pytest.approx(math.sqrt(e_after ** 2 - MASS ** 2), rel=1e-14)
        # every node after it carries the new reference, every node before the old
        assert rp[j + 1] == params[o + 1] and rb[j + 1] == params[o + 2]
        assert rp[j] == (rp[0] if k == 0 else params[rl.offsets[rl.cavities[k - 1]["refshift_node"]] + 1])
    assert np.allclose(rg, (rp / MASS) ** 2, rtol=1e-14)


def test_correctors_that_keep_their_field_kick_less_as_the_momentum_grows(booster):
    from linac_gen.elements.steerer import Steerer
    lat = copy.deepcopy(booster)
    lat.elements.insert(5, Steerer("hk", by_l=0.002, bx_l=-0.001))
    for mode in ("follow", "fixed_field"):
        cfg = RingConfig(ramp={"energy": SINE, "voltage": VTABLE, "t_start_s": 0.004},
                         corrector_mode=mode)
        rl = compile_from_config(lat, cfg)
        ramp = Ramp(RampConfig.from_dict(cfg.ramp), rl, corrector_mode=mode)
        c = rl.correctors[0]
        k0 = rl.params[rl.offsets[c["node"]]]
        pl = ramp.plan(0, *ramp.start())
        params = rl.params.copy()
        ramp.apply(pl, params, rl.ref_beta0.copy(), rl.ref_p0c.copy(), rl.ref_bg2.copy())
        k1 = params[rl.offsets[c["node"]]]
        if mode == "follow":
            assert k1 == k0 == c["kick"][0]
        else:
            # field / rigidity at the turn's reference, which is above the lattice's
            p_lat = abs(rl.brho_norm) * C_LIGHT * 1e-6
            assert k1 == pytest.approx(c["kick"][0] * p_lat / math.sqrt(pl.energy_mev ** 2
                                                                        - MASS ** 2), rel=1e-13)
            assert abs(k1) < abs(c["kick"][0])


# ---------------------------------------------------------------------------
# the reference-change node
# ---------------------------------------------------------------------------
def _refshift(p0c, beta0, dE, X):
    """One particle through a reference change of gain dE at (p0c, beta0)."""
    e0 = p0c / beta0
    e1 = e0 + dE
    p1 = math.sqrt(e1 * e1 - MASS * MASS)
    b1 = p1 / e1
    x, px, y, py, zeta, delta = X
    ptau, rpp, rvv = energy_from_delta(delta, beta0)
    out = N.track_node(N.N_REFSHIFT, np.array([dE, p1, b1]), 0, beta0, p0c, (p0c / MASS) ** 2,
                       x, px, y, py, zeta, delta, ptau, rpp, rvv)
    return out, (p1, b1)


def test_reference_change_keeps_energy_momentum_and_time():
    p0c = _p(800.0)
    beta0 = p0c / math.hypot(p0c, MASS)
    X = (1e-3, 2e-4, -3e-4, 1e-4, 0.25, 1.5e-3)
    out, (p1, b1) = _refshift(p0c, beta0, 0.5, X)
    x, px, y, py, zeta, delta, ptau, rpp, rvv, lost = out
    assert lost == N.LOSS_NONE and (x, y) == (X[0], X[2])
    # the total energy and transverse momentum of the particle are unchanged
    e_old = p0c / beta0 + p0c * energy_from_delta(X[5], beta0)[0]
    assert p1 / b1 + p1 * ptau == pytest.approx(e_old, rel=1e-15)
    assert px * p1 == pytest.approx(X[1] * p0c, rel=1e-15)
    assert py * p1 == pytest.approx(X[3] * p0c, rel=1e-15)
    # its time lag t = -zeta/(beta0 c) too, and delta is that of the new reference
    assert zeta / b1 == pytest.approx(X[4] / beta0, rel=1e-15)
    assert (1 + delta) * p1 == pytest.approx((1 + X[5]) * p0c, rel=1e-15)
    # the transverse determinant: p0c / p0c'
    assert (px / X[1]) == pytest.approx(p0c / p1, rel=1e-15)


def test_no_gain_is_no_change_at_all():
    p0c = _p(800.0)
    beta0 = p0c / math.hypot(p0c, MASS)
    X = (1e-3, 2e-4, -3e-4, 1e-4, 0.25, 1.5e-3)
    ptau, rpp, rvv = energy_from_delta(X[5], beta0)
    out = N.track_node(N.N_REFSHIFT, np.array([0.0, p0c, beta0]), 0, beta0, p0c,
                       (p0c / MASS) ** 2, *X, ptau, rpp, rvv)
    assert out[:9] == (*X, ptau, rpp, rvv)                        # bit for bit


# ---------------------------------------------------------------------------
# running totals over a ramp
# ---------------------------------------------------------------------------
def test_running_totals_over_a_ramp(booster):
    rl, ramp = _ramp_ring(booster, {"energy": SINE, "voltage": VTABLE, "t_start_s": 0.003,
                                    "gamma_t": 5.449167323})
    n = 2000
    o = ramp.orbit(n)
    t0, e0 = ramp.start()
    # the turn times add up to the clock, the gains to the energy
    t, e = t0, e0
    for k in range(n):
        pl = ramp.plan(k, o["t_s"][k], o["energy_mev"][k])
        t += pl.turn_time_s
        e += pl.dE_mev.sum()
    assert t == pytest.approx(o["t_end_s"], rel=1e-13)
    assert e == pytest.approx(o["energy_end_mev"], rel=1e-13)
    # and the reference follows the program (catch-up every turn)
    ep = RampConfig.from_dict(rl and ramp.cfg.to_dict()).energy
    e_prog = ep.total_energy(o["t_end_s"], MASS, rl.reference.p0c)
    assert o["energy_end_mev"] == pytest.approx(e_prog, rel=1e-6)
    # the program's clock: turns to the end
    n_end = ramp.turns_to_end()
    t_end = ep.t_end()
    tt = t0
    for _ in range(n_end):
        tt += rl.length / (math.sqrt(1 - (MASS / ep.total_energy(tt, MASS, rl.reference.p0c))
                                     ** 2) * C_LIGHT)
    assert tt >= t_end and tt - rl.length / C_LIGHT / 0.5 < t_end


def test_determinant_over_turns_is_the_momentum_ratio(booster):
    # the linear map of whole turns of a ramp: every plane shrinks by
    # p_start / p_end (adiabatic damping) — the 6x6 determinant
    # (p_start / p_end)^3, the vertical block (no dispersion there) p_start / p_end
    from linac_gen.ring.kernels.cpu import backend
    from linac_gen.ring.optics import _energy
    rl, ramp = _ramp_ring(booster, {"energy": SINE, "voltage": VTABLE, "t_start_s": 0.004})
    t, e = ramp.start()
    be = backend(allow_python=True)
    h = 1e-7
    X0 = np.zeros((6, 12))                   # (central differences: +-h in each)
    for i in range(6):
        X0[i, 2 * i], X0[i, 2 * i + 1] = h, -h
    X = np.ascontiguousarray(X0)
    # (the particles' energy helpers at the reference of the first turn, as
    # the engine, which compiles the ring at the ramp's start, gives them)
    ramp.apply(ramp.plan(0, t, e), rl.params, rl.ref_beta0, rl.ref_p0c, rl.ref_bg2)
    ptau, rpp, rvv = (np.ascontiguousarray(a) for a in _energy(X[5], rl.ref_beta0[0]))
    st = np.ones(12, np.int8)
    lt, ln, lc = np.full(12, -1, np.int32), np.full(12, -1, np.int32), np.zeros(12, np.int8)
    p_start = math.sqrt(e * e - MASS * MASS)
    for k in range(20):
        pl = ramp.plan(k, t, e)
        ramp.apply(pl, rl.params, rl.ref_beta0, rl.ref_p0c, rl.ref_bg2)
        be.track_turns(X, ptau, rpp, rvv, st, lt, ln, lc, *rl.node_table(), 0, rl.n_nodes, 1, k,
                       0.0)
        t += pl.turn_time_s
        e = pl.energy_end_mev
    p_end = math.sqrt(e * e - MASS * MASS)
    J = (X[:, 0::2] - X[:, 1::2]) / (2 * h)
    # (the vertical block carries no dispersion: exact to the differences'
    # rounding, measured 1.6e-12; the 6x6 determinant mixes entries of very
    # different sizes — the slip, the RF focusing — and its finite-difference
    # floor is ~3e-7, as at fixed energy, where it must be exactly 1)
    assert np.linalg.det(J[2:4, 2:4]) == pytest.approx(p_start / p_end, rel=1e-10)
    assert np.linalg.det(J) == pytest.approx((p_start / p_end) ** 3, rel=2e-6)
    assert p_end > p_start


def test_a_file_that_is_not_a_csv_table_is_a_ramp_error(tmp_path):
    # (e.g. a binary or huge-line file picked by mistake: the CSV reader's own
    # error comes back as the program's)
    from linac_gen.ring.programs import RampError, read_table_csv
    f = tmp_path / "picked.csv"
    f.write_text("t_s,kinetic_mev\n" + "1" * 200_000 + ",800\n")
    with pytest.raises(RampError, match="not a CSV table"):
        read_table_csv(f, "energy")
