"""Ring optics where they must refuse, and the RF bucket on both sides of
transition."""
from __future__ import annotations

import math
from pathlib import Path

import pytest

from linac_gen.core.lattice import Lattice
from linac_gen.elements.drift import Drift
from linac_gen.elements.multipole import Multipole
from linac_gen.elements.rf_gap import RFGap
from linac_gen.io.madx_parser import parse_madx
from linac_gen.ring.compile import compile_ring
from linac_gen.ring.optics import rf_bucket, twiss
from linac_gen.ring.optics.normal_form import UnstableError
from linac_gen.ring.reference import RingReference

DECK = Path(__file__).resolve().parent / "fixtures" / "simple_booster.madx"


def _fodo(f, cavities=()):
    lat = Lattice()
    for i in range(12):
        lat.add(Multipole(name=f"qf{i}", knl=[0.0, 1.0 / f]))
        lat.add(Drift(name=f"d1_{i}", length=3e3))
        lat.add(Multipole(name=f"qd{i}", knl=[0.0, -1.0 / f]))
        lat.add(Drift(name=f"d2_{i}", length=3e3))
    for c in cavities:
        lat.add(c)
    return compile_ring(lat, RingReference.from_kinetic("proton", 800.0))


def _cavity(name, harmonic, phase=-90.0, voltage=0.1):
    # (HELIX's RF phase; the ring cavity's sine lag is the phase + 90 deg)
    c = RFGap(name=name, voltage=voltage, phase=phase, frequency=1.0)
    c.harmonic = harmonic
    return c


def test_six_d_optics_need_a_cavity():
    rl = _fodo(4.0)
    with pytest.raises(ValueError, match="need RF"):
        twiss(rl, six_d=True)
    with pytest.raises(ValueError, match="no cavity"):
        rf_bucket(rl, twiss(rl))


def test_a_cavity_at_zero_voltage_is_no_rf():
    rl = _fodo(4.0, [_cavity("rf", 12, voltage=0.0)])
    with pytest.raises(ValueError, match="need RF"):
        twiss(rl, six_d=True)


def test_an_accelerating_phase_has_no_six_d_closed_orbit():
    # the reference energy is fixed in this milestone: a cavity that gives
    # the reference particle energy has no periodic orbit
    rl = _fodo(4.0, [_cavity("rf", 12, phase=-60.0)])
    with pytest.raises(ValueError, match="stationary bucket"):
        twiss(rl, six_d=True)
    with pytest.raises(ValueError, match="stationary bucket"):
        rf_bucket(rl, twiss(rl))


def test_cavities_at_two_harmonics_have_no_single_bucket():
    rl = _fodo(4.0, [_cavity("rf1", 12), _cavity("rf2", 24)])
    o = twiss(rl)
    with pytest.raises(ValueError, match="one harmonic"):
        rf_bucket(rl, o)


def test_an_unstable_ring_is_refused_with_its_energy():
    # thin FODO with 3 m drifts is unstable when f < 1.5 m
    with pytest.raises(UnstableError, match="unstable at 800 MeV"):
        twiss(_fodo(1.4))


@pytest.fixture(scope="module")
def booster_above_transition():
    # the Booster's magnets ramped to 6 GeV (gamma 7.4 > gamma_t 5.45)
    lat, meta = parse_madx(str(DECK))
    return lat, meta["reference"].w_kin, RingReference.from_kinetic("proton", 6000.0)


@pytest.mark.parametrize("phase, stable", [(-90.0, False), (90.0, True)])
def test_the_bucket_above_transition(booster_above_transition, phase, stable):
    # above transition the stable phase is the other zero crossing (sine lag
    # 180 deg): lag 0 is unstable, lag 180 is the bucket, and the 6-D
    # synchrotron tune agrees with the small-amplitude formula there
    lat, w, ref = booster_above_transition
    for e in lat.elements:
        if isinstance(e, RFGap):
            e.phase = phase
    rl = compile_ring(lat, ref, lattice_energy=w)
    o = twiss(rl, chromaticity=False)
    assert o.eta > 0.0 and o.gamma0 > o.gamma_t
    b = rf_bucket(rl, o)
    assert b["stable"] is stable
    assert b["lag_deg"] == pytest.approx(phase + 90.0, abs=1e-9)
    if stable:
        o6 = twiss(rl, six_d=True, chromaticity=False)
        assert o6.qs == pytest.approx(b["qs"], rel=2e-3)
        assert b["delta_max"] == pytest.approx(math.sqrt(
            2 * b["voltage_mv"] / (math.pi * 84 * o.eta * o.beta0 ** 2 * ref.energy0)), rel=1e-12)
    else:
        assert math.isnan(b["qs"]) and math.isnan(b["delta_max"])
        with pytest.raises(UnstableError):
            twiss(rl, six_d=True, chromaticity=False)


def test_a_stop_request_ends_the_optics_between_passes():
    from linac_gen.core.cancelled import OperationCancelled
    rl = _fodo(4.0)
    calls = []

    def stop():
        calls.append(1)
        return len(calls) > 3

    with pytest.raises(OperationCancelled):
        twiss(rl, should_stop=stop)
    assert len(calls) == 4
    # (the hook belongs to that computation only)
    assert twiss(rl).qx == pytest.approx(twiss(rl, should_stop=lambda: False).qx, abs=0)
