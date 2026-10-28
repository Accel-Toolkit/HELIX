"""Losses on a ramp of the simple Booster against reference code B (milestone
M6, L5).

``tests/ring/fixtures/refs/simple_booster_aperture_ref.npz`` holds the
reference code's run of the generic program of the ramp check (a 15 Hz
resonant cycle from 800 MeV to 6 GeV, a voltage table; the ramp rule written
again for it in a dev-only generator that never imports HELIX) with a
rectangular aperture (19 mm by 4 mm half widths) at the start of every cell:
1000 protons from 300 turns before transition for 600 turns, the ring
tracked cell by cell so that each loss has its cell.  The bunch is not
matched to the bucket: its momentum spread swings to 2.7e-3 within 100
turns, and the dispersion at the cells' starts (3.19 m) takes particles
into the apertures, before and after transition.  HELIX runs the same
program and apertures with the curvilinear model at the reference's
settings from the same particles: the same particles are lost, on the same
turn, at the same cell, and the survivors end where the reference's do."""
from __future__ import annotations

import copy
import json
import math
from pathlib import Path

import numpy as np
import pytest

from linac_gen.core.config import BeamConfig
from linac_gen.elements.aperture import Aperture
from linac_gen.ring import RingConfig, RingSimulation
from linac_gen.ring import coords as C

FIX = Path(__file__).resolve().parent / "fixtures"
DATA = np.load(FIX / "refs" / "simple_booster_aperture_ref.npz")
PROGRAM = json.loads(str(DATA["program_json"]))
SETTINGS = dict(model="curvilinear", integrator="symp6", bend_steps=6, magnet_integrator="symp6",
                magnet_steps=6, constants="codata2022", harmonic=84, zeta_wrap="off")


def _beta(gamma):
    return math.sqrt(1.0 - 1.0 / (gamma * gamma))


@pytest.fixture(scope="module")
def run():
    from linac_gen.cli.common import load_lattice
    lat = load_lattice(str(FIX / "simple_booster.madx"))
    ax, ay = (float(v) * 1e3 for v in DATA["aperture_m"])
    els = []
    for e in lat.elements:
        if str(e.name).lower() == "sa":
            els.append(Aperture(name="cellap", dx=ax, dy=ay, aperture_type=Aperture.RECTANGULAR))
        els.append(e)
    lat.elements[:] = els
    X0 = C.from_time_lag(*DATA["bunch_in"], beta0=_beta(float(DATA["gamma"][0])))
    n_turns = len(DATA["alive"]) - 1
    beam = BeamConfig(species="proton", energy=800.0, frequency=44.7, current=0.0)
    cfg = RingConfig(n_turns=n_turns, n_particles=X0.shape[1], tune_probes=0, stats_every=1,
                     snapshot_turns=[n_turns],
                     ramp=dict(PROGRAM, t_start_s=float(DATA["t0_s"])), **SETTINGS)
    return RingSimulation(copy.deepcopy(lat), beam, cfg, coords=X0).run(), lat


def test_the_reference_particle_is_the_references(run):
    res, _ = run
    g = np.sqrt(1.0 + (np.asarray(res.row_p0c_mev) / float(DATA["mass_mev"])) ** 2)
    assert np.max(np.abs(g / DATA["gamma"] - 1.0)) < 1e-13
    assert res.ramp["transition_turn"] == 300


def test_the_same_particles_are_lost_on_the_same_turn_at_the_same_cell(run):
    res, lat = run
    L = float(DATA["length_m"])
    lo = res.losses
    order = np.argsort(np.asarray(lo["pid"]), kind="stable")
    pid = np.asarray(lo["pid"])[order]
    turn = np.asarray(lo["turn"])[order]
    cell = np.floor(np.asarray(lo["s"])[order] / (L / 24) + 1e-9).astype(int)
    ref = np.argsort(DATA["lost_index"], kind="stable")
    assert np.array_equal(pid, DATA["lost_index"][ref])            # 584 particles
    assert np.array_equal(turn, DATA["lost_turn"][ref])
    assert np.array_equal(cell, DATA["lost_cell"][ref])
    assert np.array_equal(res.n_alive, DATA["alive"])
    # (losses before transition and after it — 483 and 101; in HELIX's record
    # 567 beyond the horizontal half width, from turn 10 on, and 17 beyond the
    # vertical one, all in the first turn — 7 of them outside it from the start)
    assert np.any(DATA["lost_turn"] < 300) and np.any(DATA["lost_turn"] >= 300)
    X = np.asarray(lo["coords"])
    assert np.any(np.abs(X[0]) > 0.019) and np.any(np.abs(X[2]) > 0.004)


def test_the_survivors_end_where_the_references_do(run):
    res, _ = run
    snap = res.snapshots[len(DATA["alive"]) - 1]
    order = np.argsort(snap["pid"])
    assert np.array_equal(snap["pid"][order], DATA["end_index"])
    R = C.from_time_lag(*DATA["end_bunch"], beta0=_beta(float(DATA["gamma"][-1])))
    d = np.abs(snap["coords"][:, order] - R)
    # (rounding grown over 600 turns of a nonlinear map, as without apertures:
    # measured 1.6e-10 m in x, 3.7e-11 m in y, 1.2e-10 m in zeta)
    assert d[0].max() < 2e-9 and d[2].max() < 2e-9 and d[4].max() < 2e-9
