"""A ramp of the simple Booster against reference code B (milestone M3, L2).

``tests/ring/fixtures/refs/simple_booster_ramp_ref.npz`` holds the reference
code's run of a generic program (a 15 Hz resonant cycle from 800 MeV to
6 GeV, a voltage table; the ramp rule written again for it in a dev-only
generator that never imports HELIX), in two segments — from the start of the
ramp, 600 turns, and across transition, 400 turns with the phase jump at turn
200: the reference particle's time and gamma every turn, and a bunch of 200
protons at the recorded turns.  HELIX runs the same program with the
curvilinear model at the reference's settings (sixth order, six steps in
every magnet; its constants) from the same particles."""
from __future__ import annotations

import copy
import json
import math
from pathlib import Path

import numpy as np
import pytest

from linac_gen.core.config import BeamConfig
from linac_gen.ring import RingConfig, RingSimulation
from linac_gen.ring import coords as C

FIX = Path(__file__).resolve().parent / "fixtures"
REF = FIX / "refs" / "simple_booster_ramp_ref.npz"
DATA = np.load(REF)
PROGRAM = json.loads(str(DATA["program_json"]))
MASS = float(DATA["mass_mev"])
SEGMENTS = {"start": (600, (1, 300, 600)), "transition": (400, (1, 200, 400))}
SETTINGS = dict(model="curvilinear", integrator="symp6", bend_steps=6, magnet_integrator="symp6",
                magnet_steps=6, constants="codata2022", harmonic=84, zeta_wrap="off")


def _beta(gamma):
    return math.sqrt(1.0 - 1.0 / (gamma * gamma))


@pytest.fixture(scope="module")
def booster():
    from linac_gen.cli.common import load_lattice
    return load_lattice(str(FIX / "simple_booster.madx"))


@pytest.fixture(scope="module", params=list(SEGMENTS))
def segment(request, booster):
    name = request.param
    n, recorded = SEGMENTS[name]
    t0 = float(DATA[f"{name}_t0_s"])
    g0 = float(DATA[f"{name}_gamma"][0])
    X0 = C.from_time_lag(*DATA[f"{name}_in"], beta0=_beta(g0))
    beam = BeamConfig(species="proton", energy=800.0, frequency=44.7, current=0.0)
    cfg = RingConfig(n_turns=n, n_particles=X0.shape[1], tune_probes=0, stats_every=n,
                     snapshot_turns=list(recorded), ramp=dict(PROGRAM, t_start_s=t0), **SETTINGS)
    res = RingSimulation(copy.deepcopy(booster), beam, cfg, coords=X0).run()
    return name, res


def test_the_reference_particle_turn_by_turn(segment):
    name, res = segment
    g_ref = DATA[f"{name}_gamma"]
    t_ref = DATA[f"{name}_t_s"]
    g = 1.0 + res.ramp["kinetic_mev"] / MASS
    np.testing.assert_allclose(g, g_ref, rtol=1e-12, atol=0)
    np.testing.assert_allclose(res.ramp["t_s"], t_ref, rtol=1e-12, atol=0)
    assert res.ramp["gamma_end"] == pytest.approx(float(DATA[f"{name}_gamma_end"]), rel=1e-12)
    assert res.ramp["t_end_s"] == pytest.approx(float(DATA[f"{name}_t_end_s"]), rel=1e-12)
    if name == "transition":
        assert res.ramp["transition_turn"] == 200 == int(np.argmax(g_ref > 5.449167323))


def test_the_bunch_at_the_recorded_turns(segment):
    name, res = segment
    n, recorded = SEGMENTS[name]
    g_ref = DATA[f"{name}_gamma"]
    out = {}
    for k in recorded:
        # (after turn k the coordinates are relative to the reference of the
        # start of turn k + 1)
        g_after = float(DATA[f"{name}_gamma_end"]) if k == n else float(g_ref[k])
        X = res.snapshots[k]["coords"]
        got = np.vstack(C.to_time_lag(X, beta0=_beta(g_after)))
        want = DATA[f"{name}_turn{k}"]
        out[k] = np.abs(got - want).max(axis=1)
    for k, d in out.items():
        # x, y, t [m]; px, py, pt.  Measured (macOS): 2.5e-12 m in x after a
        # turn — the reference's own rounding, as at fixed energy — growing to
        # 5.9e-11 m after 600 turns and, across transition, 1.0e-10 m in x and
        # 6.3e-10 m in t after 400; the bounds are ten times that, growing with
        # the turns, for other machines' rounding
        tol = np.array([3e-11, 1e-11, 1e-10, 3e-12, 1e-12, 2e-12]) * max(1.0, k / 10.0)
        assert np.all(d <= tol), (name, k, d)
