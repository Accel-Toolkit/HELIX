"""Ring mode, milestone M6 (L5): space charge through transition on a ramp of
the simple Booster against reference code B's stored run (marker ``bench``:
run with HELIX_RUN_BENCH=1).

``tests/ring/fixtures/refs/simple_booster_sc_ramp_ref.npz``: the generic
program of the ramp check (a 15 Hz resonant cycle from 800 MeV to 6 GeV;
the ramp rule written again for it in a dev-only generator that never
imports HELIX), 5000 protons standing for 1.5e10, from 300 turns before
transition for 600 turns, with the reference code's 2.5-D space charge —
its transverse kick alone: its longitudinal kick, from an unfiltered line
density, heats a bunch by itself (the M4 comparison, validation page).  The
bunch, 0.1 m long (rms), starts with sigma_delta = sigma_z / 154 m, 2.9
times the bucket's small-amplitude ratio there (53 m): it rotates, to
3.4 cm at turn 146 while its momentum spread swells, and its horizontal
betatron emittance grows by more than 40 % before transition.  HELIX runs the same
program from the same particles with the curvilinear model at the
reference's settings and its 2.5-D model, the longitudinal kick off: the
betatron emittances, the bunch length, the energy spread and the
longitudinal emittance turn by turn (the reference's p_t taken to delta to
first order: up to 5e-4 in the energy spread at the end, the bunch
2.4e-3 off momentum by then; without space charge the codes' particles
end within 1.3e-5 m of each other and these quantities within 2.4e-6).

The line density must resolve that bunch: 4096 bins over the 5.6 m RF
period (1.4 mm).  With 256 bins (2.2 cm: the bins and the filter widen
what they see) its peak is 20 % low at the shortest bunch and the growth to
transition +13 % instead of the reference's +47 %; with 2048 bins +38 %.  At
5000 particles the growth also carries the line density's noise: at 20 000
(dev runs, benchmarks/ring/dev/README.md) it is +42 % in the reference and
+41 % in HELIX (+38 % with 2048 bins), the codes then within 1 % turn by
turn.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
FIXTURE = ROOT / "tests" / "ring" / "fixtures" / "refs" / "simple_booster_sc_ramp_ref.npz"
BOOSTER = ROOT / "tests" / "ring" / "fixtures" / "simple_booster.madx"
SETTINGS = dict(model="curvilinear", integrator="symp6", bend_steps=6, magnet_integrator="symp6",
                magnet_steps=6, constants="codata2022", harmonic=84)


def _emit(c, i, j=None):
    """The emittance of plane (i, i+1) (without its correlation with row j)."""
    if j is None:
        return math.sqrt(max(c[i, i] * c[i + 1, i + 1] - c[i, i + 1] ** 2, 0.0))
    idx = [i, i + 1, j]
    c3 = c[np.ix_(idx, idx)]
    return math.sqrt(max(np.linalg.det(c3) / c3[2, 2], 0.0))


def quantities(cov_helix, cov_ref, beta0):
    """Per row: (betatron x, betatron y, sigma_z, sigma_delta, longitudinal
    emittance) of HELIX's covariance (x, px, y, py, zeta, delta) and of the
    reference's (x, px, y, py, t, pt): sigma_z = beta0 sigma_t, sigma_delta =
    sigma_pt / beta0 (to first order), the longitudinal area the same."""
    h, r = [], []
    for ch, cr, b0 in zip(cov_helix, cov_ref, beta0, strict=True):
        h.append((_emit(ch, 0, 5), _emit(ch, 2, 5), math.sqrt(ch[4, 4]), math.sqrt(ch[5, 5]),
                  _emit(ch, 4)))
        r.append((_emit(cr, 0, 5), _emit(cr, 2, 5), b0 * math.sqrt(cr[4, 4]),
                  math.sqrt(cr[5, 5]) / b0, _emit(cr, 4)))
    return np.array(h), np.array(r)


def run_helix(d):
    from linac_gen.cli.common import load_lattice
    from linac_gen.core.config import BeamConfig
    from linac_gen.ring import RingConfig, RingSimulation
    from linac_gen.ring import coords as C
    s = json.loads(str(d["settings_json"]))
    g0 = float(d["gamma"][0])
    X0 = C.from_time_lag(*d["bunch_in"], beta0=math.sqrt(1.0 - 1.0 / g0 ** 2))
    n_turns = len(d["gamma"]) - 1
    sc = {"intensity": s["intensity"], "kicks_per_turn": 192, "grid": [64, 64],
          "deposit": "tsc", "box_sigma": 100.0, "z_bins": 4096, "z_smoothing_bins": 1.0,
          "longitudinal": "off"}
    beam = BeamConfig(species="proton", energy=800.0, frequency=44.7, current=0.0)
    cfg = RingConfig(n_turns=n_turns, n_particles=X0.shape[1], tune_probes=0, stats_every=1,
                     space_charge=sc,
                     ramp=dict(json.loads(str(d["program_json"])), t_start_s=float(d["t0_s"])),
                     **SETTINGS)
    return RingSimulation(load_lattice(str(BOOSTER)), beam, cfg, coords=X0).run()


@pytest.fixture(scope="module")
def fixture_data():
    if not FIXTURE.is_file():
        pytest.skip("the reference run's fixture is not there")
    return np.load(FIXTURE)


@pytest.mark.bench
def test_space_charge_through_transition_against_reference_code_b(fixture_data):
    d = fixture_data
    res = run_helix(d)
    g = np.asarray(d["gamma"])
    b0 = np.sqrt(1.0 - 1.0 / g ** 2)
    gh = np.sqrt(1.0 + (np.asarray(res.row_p0c_mev) / float(d["mass_mev"])) ** 2)
    assert np.max(np.abs(gh / g - 1.0)) < 1e-13
    assert res.ramp["transition_turn"] == 300
    h, r = quantities(res.cov, d["cov"], b0)
    rel = np.abs(h / r - 1.0)
    for k, what in enumerate(("betatron x", "betatron y", "sigma_z", "sigma_delta",
                              "longitudinal emittance")):
        assert rel[:, k].max() < TOLERANCES[what], (what, float(rel[:, k].max()),
                                                    int(np.argmax(rel[:, k])))


#: the largest relative difference over the 600 turns allowed (measured: 2.7 %
#: at turn 227, 0.64 %, 0.15 %, 0.30 %, 1.8 %)
TOLERANCES = {"betatron x": 0.04, "betatron y": 0.01, "sigma_z": 0.003, "sigma_delta": 0.005,
              "longitudinal emittance": 0.03}
