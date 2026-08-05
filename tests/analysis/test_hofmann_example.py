"""Regression smoke of the 46-period Hofmann example deck.

Pins the example's headline outcomes at the 5 mA working point: the
higher-order l=3-odd flag in section A (in-domain), the l=2 flag on the
R=1 band in section B, section C quiet, and the coupled refusal of the
solenoid section D.  Growth expectations are loose (abs 5e-3) — the
bitwise contract lives in test_hofmann_dispersion.py; this test guards
the deck + parser + probe + adapter chain end-to-end.
"""
from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pytest

from linac_gen.analysis.hofmann_stability import hofmann_stability
from linac_gen.analysis.period_detect import detect_periods
from linac_gen.core.particle import PROTON
from linac_gen.core.reference import ReferenceParticle
from linac_gen.io.tracewin_parser import parse_tracewin
from linac_gen.tracking.envelope import EnvelopeSolver
from linac_gen.tracking.matrix_tracking import (
    compute_transfer_matrix,
    compute_twiss,
)

DECK = (Path(__file__).resolve().parents[2] / "examples"
        / "hofmann_stability" / "hofmann_demo_linac.dat")


@pytest.mark.physics
def test_demo_deck_scenario_outcomes():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        lat, meta = parse_tracewin(str(DECK))
    assert not meta["warnings"], meta["warnings"]
    periods = [p for p in detect_periods(lat) if p.source != "fallback"]
    assert [p.n_repeats for p in periods] == [16, 12, 12, 6]

    ref = ReferenceParticle(species=PROTON, w_kin=2.5, frequency=162.5)
    a, b = periods[0].spans()[0]
    M = compute_transfer_matrix(lat, ref, start=a, end=b - 1)
    twx, twy, twz = (compute_twiss(M, pl) for pl in ("x", "y", "z"))
    init = dict(alpha_x=twx["alpha"], beta_x=twx["beta"], emit_x=2.0,
                alpha_y=twy["alpha"], beta_y=twy["beta"], emit_y=2.0,
                alpha_z=twz["alpha"], beta_z=twz["beta"], emit_z=0.035)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = EnvelopeSolver(lat, ref.copy(), init, current=5.0,
                             phase_probe=True).run()
        tabs = [hofmann_stability(res, p) for p in periods]
    sec_a, sec_b, sec_c, sec_d = tabs

    # Section A: one in-domain higher-order (l=3 odd) flag.
    assert sec_a["reason"] is None
    assert sec_a["n_flagged"] >= 1
    k = int(np.nanargmax(np.where(sec_a["flagged"],
                                  sec_a["g_combined"], np.nan)))
    assert sec_a["g_l3_odd"][k] == pytest.approx(0.0331, abs=5e-3)
    assert sec_a["g_l3_odd"][k] == sec_a["g_combined"][k]
    assert sec_a["g_l2"][k] < sec_a["g_l3_odd"][k]     # l=2 alone misses it
    assert sec_a["S2"][k] <= 10.0

    # Section B: a weak l=2 flag on the R = 1 band.
    assert sec_b["reason"] is None
    flag_b = sec_b["flagged"]
    assert flag_b.any()
    j = int(np.argmax(flag_b))
    assert sec_b["R"][j] == pytest.approx(1.0, abs=0.05)
    assert sec_b["g_l2"][j] == sec_b["g_combined"][j]
    # Hot cells outside the gate stay extrapolations, never valid flags.
    assert not (sec_b["flagged"] & ~sec_b["valid"]).any()

    # Section C: quiet at this working point.
    assert sec_c["reason"] is None
    assert sec_c["n_flagged"] == 0

    # Section D: coupled refusal.
    assert sec_d["reason"] is not None and "coupled" in sec_d["reason"]
