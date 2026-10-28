"""Ring mode, milestone M4: HELIX's space charge against the reference codes' stored
runs on the simple Booster (marker ``bench``: run with HELIX_RUN_BENCH=1).

The bunch matched inside the 200 kV bucket (1.5e10, 0.01 eV s: its
longitudinal space charge, at the bunch's centre, about as strong as the RF's
focusing — the bunch length swings from 0.30 m to 0.41 m and back), the same
50 000 initial particles as each reference run (drawn again here with the
fixture's recipe, seed and mass), 20 turns:

* against reference code A (its averaged 2.5-D model, the longitudinal kick
  on), HELIX's Cartesian model at its settings: the bunch length, the energy
  spread, the longitudinal and the betatron emittances (measured: within
  2.4e-3, 1.9e-3, 2.2e-3 and 9e-4 over the 20 turns; 4.2e-3, 2.9e-3, 2.2e-3
  and 9e-4 over 150);
* against reference code B (its 2.5-D model), HELIX's curvilinear model at
  its settings: the betatron emittances (within 7e-4 over 100 turns) and the
  bunch length at turn 5 — after that reference code B's unfiltered line
  density heats the bunch longitudinally (its longitudinal emittance +9 % by
  turn 20, +5 % over 300 turns with 32 bins instead of 128; see the
  validation page).
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
FIXTURE = ROOT / "tests" / "ring" / "fixtures" / "refs" / "simple_booster_sc_ref.npz"
BOOSTER = ROOT / "tests" / "ring" / "fixtures" / "simple_booster.madx"
C_LIGHT = 299792458.0
TURNS = 20


def _bunch(s: dict, n: int, mass: float):
    """(x, px, y, py, t, pt) of the fixture's bunch: Gaussian in normalized
    coordinates with the start's periodic Twiss, the longitudinal emittance
    converted as the reference code's example converts it (6 pi times the rms),
    dispersion added through each particle's delta — the recipe and the draw
    order of the reference runs' generator."""
    g0 = 1.0 + s["kinetic_mev"] / mass
    b0 = math.sqrt(1.0 - 1.0 / (g0 * g0))
    bg = b0 * g0
    ex = ey = s["emit_n_rms"] / bg
    et = s["emit_ev_s"] * 1e-6 * C_LIGHT / (mass * bg * 6.0 * math.pi)
    u = np.random.default_rng(s["seed"]).standard_normal((6, n))
    t = math.sqrt(et * s["beta_t"]) * u[4]
    pt = math.sqrt(et / s["beta_t"]) * u[5]
    ptau = -pt
    delta = np.sqrt(1.0 + 2.0 * ptau / b0 + ptau * ptau) - 1.0
    w = s["twiss"]
    x = math.sqrt(ex * w["beta_x"]) * u[0] + w["disp_x"] * delta
    px = math.sqrt(ex / w["beta_x"]) * (u[1] - w["alpha_x"] * u[0]) + w["disp_px"] * delta
    y = math.sqrt(ey * w["beta_y"]) * u[2]
    py = math.sqrt(ey / w["beta_y"]) * (u[3] - w["alpha_y"] * u[2])
    return x, px, y, py, t, pt


def _emit(c, i, j=None):
    """The emittance of plane (i, i+1) (without its correlation with row j)."""
    if j is None:
        return math.sqrt(max(c[i, i] * c[i + 1, i + 1] - c[i, i + 1] ** 2, 0.0))
    idx = [i, i + 1, j]
    c3 = c[np.ix_(idx, idx)]
    return math.sqrt(max(np.linalg.det(c3) / c3[2, 2], 0.0))


def _run(d, key, settings, sc):
    """HELIX from the reference run ``key``'s particles, TURNS turns."""
    from linac_gen.cli.common import load_lattice
    from linac_gen.core.config import BeamConfig
    from linac_gen.ring import RingConfig, RingSimulation
    from linac_gen.ring import coords as C
    s = json.loads(str(d["settings_json"]))["B_seed1"]       # (the recipe's numbers)
    s = dict(s, seed=json.loads(str(d["settings_json"]))[key]["seed"])
    mass = float(d[f"{key}_mass_mev"])
    n = int(s["particles"])
    x, px, y, py, t, pt = _bunch(s, n, mass)
    b0 = math.sqrt(1.0 - 1.0 / (1.0 + s["kinetic_mev"] / mass) ** 2)
    X0 = C.from_time_lag(x, y, t, px, py, pt, beta0=b0)
    beam = BeamConfig(species="proton", energy=s["kinetic_mev"], frequency=44.7, current=0.0)
    cfg = RingConfig(n_turns=TURNS, n_particles=n, tune_probes=0, stats_every=s["every"],
                     harmonic=s["harmonic"], space_charge=dict(sc, intensity=s["intensity"]),
                     **settings)
    return RingSimulation(load_lattice(str(BOOSTER)), beam, cfg, coords=X0).run(), b0


def _rows(res, d, key):
    """(turn, HELIX's covariance, the reference's) at the reference's rows."""
    turns = np.asarray(d[f"{key}_turn"])
    for k, turn in enumerate(np.asarray(res.turns)):
        j = int(np.searchsorted(turns, turn))
        assert j < len(turns) and turns[j] == turn
        yield int(turn), np.asarray(res.cov[k]), np.asarray(d[f"{key}_cov"])[j]


@pytest.fixture(scope="module")
def fixture_data():
    if not FIXTURE.is_file():
        pytest.skip("the reference runs' fixture is not there")
    return np.load(FIXTURE)


@pytest.mark.bench
def test_the_bunch_in_the_bucket_against_reference_code_a(fixture_data):
    d = fixture_data
    res, _b0 = _run(d, "A_seed1", dict(model="cartesian", edge_model="full", constants="helix"),
                    {"kicks_per_turn": 96, "grid": [64, 64], "deposit": "tsc", "box_sigma": 4.0,
                     "z_bins": 128, "z_smoothing_bins": 1.0})
    for turn, ch, cr in _rows(res, d, "A_seed1"):
        # (both in zeta, delta)
        for got, want, tol, what in (
                (math.sqrt(ch[4, 4]), math.sqrt(cr[4, 4]), 5e-3, "sigma_z"),
                (math.sqrt(ch[5, 5]), math.sqrt(cr[5, 5]), 5e-3, "sigma_delta"),
                (_emit(ch, 4), _emit(cr, 4), 5e-3, "longitudinal emittance"),
                (_emit(ch, 0, 5), _emit(cr, 0, 5), 1.5e-3, "betatron emittance x"),
                (_emit(ch, 2, 5), _emit(cr, 2, 5), 1.5e-3, "betatron emittance y")):
            assert got == pytest.approx(want, rel=tol), (turn, what)


@pytest.mark.bench
def test_the_bunch_in_the_bucket_against_reference_code_b(fixture_data):
    d = fixture_data
    res, b0 = _run(d, "B_seed1", dict(model="curvilinear", integrator="symp6", bend_steps=6,
                                       magnet_integrator="symp6", magnet_steps=6,
                                       constants="codata2022"),
                   {"kicks_per_turn": 192, "grid": [64, 64], "deposit": "tsc",
                    "box_sigma": 100.0, "z_bins": 256, "z_smoothing_bins": 1.0})
    for turn, ch, cr in _rows(res, d, "B_seed1"):
        # (the reference's t and pt: delta's correlations are pt's; sigma_z = beta0 sigma_t)
        assert _emit(ch, 0, 5) == pytest.approx(_emit(cr, 0, 5), rel=1e-3), turn
        assert _emit(ch, 2, 5) == pytest.approx(_emit(cr, 2, 5), rel=1e-3), turn
        if turn == 5:
            assert math.sqrt(ch[4, 4]) == pytest.approx(b0 * math.sqrt(cr[4, 4]), rel=5e-3)
