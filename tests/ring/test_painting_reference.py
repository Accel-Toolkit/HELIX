"""Painting a beam into a ring through a foil against reference code C
(milestone M6, L5).

``tests/ring/fixtures/refs/painting_ref.npz`` holds the reference code's run
(a dev-only generator that never imports HELIX) of a generic FODO ring — 12
cells QF D B D QD D B D, quads 0.4 m at k1 = +-0.35 /m^2, 201.6 m, 800 MeV,
no RF: a coasting beam; its quadrupoles without the reference code's
hard-edge fringe (HELIX's have none), HELIX's drifts expanded as the
reference's.  The first drift holds a vertical bump of four thin kickers
(kicks t, -t, -t, t, t = 7.5 mrad: 6 mm at the foil) whose strength falls
as 1 - sqrt(t/T) over 300 injection turns, held through each turn at its
value at the turn's start (as HELIX applies a program); 20 protons a turn
injected at y = 6 mm — the same 6000 in both codes, in the same order, at
the same energies (HELIX reads them from a DST file with ``draw:
in_order``; the reference's canonical momenta written as slopes); the
foil, 2 to 20 mm in y, a carbon one of 1e-6 ug/cm^2 (the case
``painting``: no scattering, the foil counts its hits) or of 600 ug/cm^2
(``scattering``); 400 turns.  The tunes agree to 1e-10.

Painting: the same particles circulate; the foil is hit as often, turn by
turn (604 089 hits in all against 604 109) and per particle (100.75 on
average); the emittances agree turn by turn (3e-6 in y, 1.7e-5 in x) and
each particle's vertical action at the end (6e-7 at the median); the
particles end 5 um apart at the median in y (2 um in x).

Scattering: each circulating particle crosses the foil about 100 times.
HELIX gives a particle's k-th crossing the growth of Highland's theta0^2
from k - 1 to k thicknesses, so that its crossings add up to Highland at
their total thickness (theta0^2 is not additive over layers: kicks of the
one-crossing width would be 1.30 times too narrow in angle for 99
crossings).  The reference code's single scatters add up as the physics
does: the painted vertical action agrees to 1.4 % at the median over five
of its runs (its foil's random numbers are not seeded), and its foil widens
the horizontal action (the plane the painting leaves alone) 0.84-0.90
times as much as HELIX's — a single-scatter estimate gives 0.88: Highland
at the total thickness (Lynch & Dahl's form; x/X0 = 1.4e-3, inside the
range it was fitted over, to about 11 %) is 6 % wider than the core of 100
crossings' single scatters, and at different betatron phases that core
grows 5 % less than in one plane; its tails, which HELIX does not model,
much more.  The actions are betatron actions, the dispersion taken out:
HELIX's foil also slows the particles (a mean delta of -8.6e-5 after 100
crossings), the reference's scattering-only foil does not.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

FIX = Path(__file__).resolve().parent / "fixtures" / "refs" / "painting_ref.npz"
DATA = np.load(FIX)
S = json.loads(str(DATA["settings_json"]))
TW_START = json.loads(str(DATA["twiss_start_json"]))
#: the reference code's drifts are expanded (paraxial)
MODEL = {"drift_model": "expanded"}


def _emit(c, i):
    c = np.asarray(c)
    return np.sqrt(np.maximum(c[:, i, i] * c[:, i + 1, i + 1] - c[:, i, i + 1] ** 2, 0.0))


def run_helix(tmp: Path, case: str):
    from linac_gen.cli.common import load_lattice
    from linac_gen.io.tracewin_dst import write_dst
    from linac_gen.ring import RingConfig, RingSimulation
    deck = tmp / "painting_ring.madx"
    deck.write_text(str(DATA["deck_madx"]))
    q = DATA["injected"]                                   # x, px, y, py [m, 1], dE [GeV]
    n = q.shape[1]
    # (the reference's momenta are canonical, px/p0: the file takes the
    # slopes x' = px/pz, which HELIX reads back to px exactly)
    pz = np.sqrt((1.0 + _delta(q[4])) ** 2 - q[1] ** 2 - q[3] ** 2)
    P = np.column_stack([q[0] * 1e3, q[1] / pz * 1e3, q[2] * 1e3, q[3] / pz * 1e3,
                         np.zeros(n), q[4] * 1e3])
    dst = tmp / "injected.dst"
    write_dst(str(dst), P, current_mA=2.0, frequency_MHz=162.5,
              mass_MeV=float(DATA["mass_gev"]) * 1e3, w_kin_ref=float(DATA["kinetic_gev"]) * 1e3)
    box = S["foil_box"]
    th = S["theta"]
    inj = {"foil": {"at": "foil", "material": "C",
                    "thickness_ug_cm2": S["thickness_ug_cm2"][case], "x_min_m": box[0],
                    "x_max_m": box[1], "y_min_m": box[2], "y_max_m": box[3]},
           "source": {"kind": "file", "path": str(dst), "species": "proton",
                      "frequency_mhz": 162.5, "draw": "in_order", "y_m": S["y_inj"]},
           "turns": S["turns_inj"], "particles_per_turn": S["per_turn"], "current_ma": 2.0,
           "magnets": [{"name": "paint", "elements": ["kb1", "kb2", "kb3", "kb4"],
                        "model": "kick", "plane": "y",
                        "kick_rad": [th * k[2] for k in S["kickers"]],
                        "waveform": {"kind": "sqrt", "t_start_s": 0.0,
                                     "t_end_s": S["turns_inj"] * S["turn_time_s"],
                                     "start": 1.0, "end": 0.0}}]}
    cfg = RingConfig(n_turns=S["turns"], source="empty", injection=inj, tune_probes=0,
                     stats_every=1, snapshot_turns=[S["turns"]], seed=1, **MODEL)
    return RingSimulation(load_lattice(str(deck)), _beam(), cfg).run()


def _beam():
    from linac_gen.core.config import BeamConfig
    return BeamConfig(species="proton", energy=800.0, frequency=162.5, current=0.0)


def _delta(dE_gev):
    """delta of the reference code's energy offsets [GeV] (its p0 at 800 MeV)."""
    m, k = float(DATA["mass_gev"]), float(DATA["kinetic_gev"])
    p0 = np.sqrt((m + k) ** 2 - m * m)
    return np.sqrt((m + k + np.asarray(dE_gev)) ** 2 - m * m) / p0 - 1.0


@pytest.fixture(scope="module")
def runs(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("painting")
    return {case: run_helix(tmp, case) for case in ("painting", "scattering")}


def _hits_turn(res):
    r = res.injection
    return np.asarray(r["foil_hits"]) + np.asarray(r["foil_hits_injected"])


def test_the_same_particles_circulate_and_hit_the_foil_as_often(runs):
    res = runs["painting"]
    assert np.array_equal(res.n_alive[1:], DATA["painting_n"][1:])         # 5996 at the end
    assert res.ledger["injected"] == S["turns_inj"] * S["per_turn"]
    h, r = _hits_turn(res), DATA["painting_hits_turn"]
    # (a particle at the foil's edge counted on one turn or the next: measured
    # at most 7 hits apart in a turn, 20 in 604 109)
    assert np.all(np.abs(h - r) <= 0.03 * r + 5), int(np.argmax(np.abs(h - r) - 0.03 * r))
    assert abs(int(h.sum()) / int(r.sum()) - 1.0) < 1e-4
    mean_hits = res.injection["mean_hits_circulating"]
    assert mean_hits == pytest.approx(DATA["painting_hits_particle"].mean(), rel=1e-4)


def test_the_painted_emittances_turn_by_turn(runs):
    res = runs["painting"]
    for i, tol in ((2, 2e-5), (0, 1e-4)):               # (measured 3.1e-6 in y, 1.7e-5 in x)
        eh = _emit(res.cov, i)[1:]
        er = _emit(DATA["painting_cov"], i)[1:]
        rel = np.abs(eh / er - 1.0)
        assert rel.max() < tol, (i, float(rel.max()), int(np.argmax(rel)))


def _end(res):
    """HELIX's particles at the end in the order injected (x, px, y, py, zeta, delta)."""
    snap = res.snapshots[S["turns"]]
    return snap["coords"][:, np.argsort(snap["pid"])]


def _reference_end(case):
    """The reference's particles at the end (x, px, y, py, z, delta)."""
    R = np.array(DATA[f"{case}_end"], dtype=float)         # ..., dE [GeV]
    R[5] = _delta(R[5])
    return R


def _actions(X, plane, optics):
    """Each particle's betatron action in a plane at the ring's start: the
    reference's Twiss there, the momentum's share taken out with the
    dispersion there (HELIX's optics of the same ring) — HELIX's foil slows
    the particles a little, the reference's scattering-only foil does not;
    the momenta of both codes canonical."""
    i = 2 if plane == "y" else 0
    a, b = TW_START[f"alpha_{plane}"], TW_START[f"beta_{plane}"]
    d, dp = optics[f"d{plane}_start"], optics[f"dp{plane}_start"]
    y, yp = X[i] - d * X[5], X[i + 1] - dp * X[5]
    return 0.5 * ((1.0 + a * a) / b * y * y + 2.0 * a * y * yp + b * yp * yp)


def test_each_particle_ends_with_the_references_action(runs):
    res = runs["painting"]
    X, R = _end(res), _reference_end("painting")
    jh, jr = _actions(X, "y", res.optics), _actions(R, "y", res.optics)
    rel = np.abs(jh - jr) / jr
    # (measured 5.6e-7 at the median, 6.4e-6 at the 99th percentile)
    assert np.median(rel) < 5e-6 and np.quantile(rel, 0.99) < 5e-5
    for plane, tol in (("y", 2e-5), ("x", 5e-5)):       # (measured 2.0e-6, 4.1e-6)
        q = (0.5, 0.9, 0.99)
        got = (np.quantile(_actions(X, plane, res.optics), q)
               / np.quantile(_actions(R, plane, res.optics), q))
        assert np.all(np.abs(got - 1.0) < tol), (plane, got)


def test_repeated_crossings_scatter_as_highland_at_the_total_thickness(runs):
    """The foil's share of the median horizontal action: HELIX's crossings,
    each the growth of Highland's theta0^2 from one thickness to the next,
    against the reference's single scatters (measured 0.88 times as much in
    the reference, 0.84-0.90 over five of its runs; kicks of the one-crossing
    width gave 1.47-1.59).  The median, not the rms: the reference's single
    scatters have the Rutherford tail, and its rms moves from run to run with
    a few particles."""
    opt = runs["painting"].optics
    ends = {case: (_end(runs[case]), _reference_end(case)) for case in ("painting", "scattering")}
    med = {c: [float(np.median(_actions(X, "x", opt))) for X in ends[c]] for c in ends}
    grow_h = med["scattering"][0] - med["painting"][0]
    grow_r = med["scattering"][1] - med["painting"][1]
    assert grow_h > 0.0 and 0.75 < grow_r / grow_h < 1.10, (grow_h, grow_r)
    # (the painted vertical action, the painting's mostly: 1.2 % apart at the
    # median, -1.4 to +0.8 % over five runs)
    jy = [float(np.median(_actions(X, "y", opt))) for X in ends["scattering"]]
    assert jy[0] == pytest.approx(jy[1], rel=0.05)
