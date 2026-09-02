"""Dead-beam-aware orbit correction (2026-09 defect fix).

The defect: ``DiagnosticRecorder.record`` appends a ``zeros(6)`` centroid
placeholder when no particle is alive, so a corrector kick that KILLS the
beam made every downstream BPM read exactly 0.0 — the corrector declared
"converged" (rms ~1e-12) on a 0 %-transmission beam.

Contract pinned here (house rule: both regimes):

* Regime A — beam alive at every used BPM: kicks and legacy history
  values bit-compatible with the pre-fix tree (exact equality proven
  locally by the bitproof capture; rtol 1e-12 here for cross-platform
  CI), plus the new additive keys.
* Regime B — >= 1 used BPM dead (MP backend): NaN rms, no convergence,
  ``status == "beam_lost"``, first dead element named, early stop.

Identity claims for regime B are restricted to ``bpm_noise == 0`` and
dead-at-baseline readings: with noise a dead BPM used to contribute
``R = -noise/delta`` garbage, and a beam killed by the response probe
itself used to yield ``R = -r0/delta`` — both now yield NO kick (the
NEW, pinned behaviour).
"""
from __future__ import annotations

import logging
import math

import numpy as np
import pytest

from linac_gen.core.beam import Beam
from linac_gen.core.config import BeamConfig
from linac_gen.core.lattice import Lattice
from linac_gen.core.particle import PROTON
from linac_gen.core.reference import ReferenceParticle
from linac_gen.distributions.factory import create_beam
from linac_gen.elements.drift import Drift
from linac_gen.elements.lattice_commands import AdjustSteerer
from linac_gen.elements.marker import Marker
from linac_gen.elements.quadrupole import Quadrupole
from linac_gen.elements.steerer import Steerer
from linac_gen.errors.correction import (_beam_lost_at, _live_bpm_reading,
                                         apply_correction, correction_status,
                                         run_correction_from_lattice)
from linac_gen.tracking.tracker import Tracker
from tests.errors.test_correction_from_lattice import (_factory,
                                                       _make_pair_lattice)
from tests.errors.test_correction_iter import _fodo_with_pair
from tests.helpers import stub_results


# ---------------------------------------------------------------------
# The ORIGINAL correction_demo geometry (BPM 50 mm downstream of its
# steerer — the ill-conditioned lever arm that produced the -35 mrad
# beam-killing kick).  Kept here in memory so the regression survives
# the demo deck's re-conditioning.
# ---------------------------------------------------------------------
def _original_demo_lattice() -> Lattice:
    lat = Lattice()
    qn = dn = pn = 0

    def quad(g):
        nonlocal qn
        qn += 1
        return Quadrupole(f"QUAD_{qn:03d}", length=100.0, gradient=g,
                          aperture=20.0)

    def drift(length):
        nonlocal dn
        dn += 1
        return Drift(f"DR_{dn:03d}", length, aperture=30.0)

    lat.add(drift(100.0))
    for cell in range(1, 7):
        if cell in (1, 3, 5, 6):        # steerer/BPM cells, as in the deck
            pn += 1
            lat.add(quad(8.0))
            lat.add(drift(50.0))
            lat.add(AdjustSteerer(f"ADJ_{pn}", diag_n=pn, vmax=0.02,
                                  first_step=1e-4))
            lat.add(Steerer(f"STEER_{pn:03d}", bx_l=0.0, by_l=0.0))
            lat.add(drift(50.0))
            lat.add(Marker(f"BPM_{pn:03d}", is_bpm=True))   # 50 mm after STEER
            lat.add(drift(100.0))
            lat.add(quad(-8.0))
            lat.add(drift(100.0))
        else:                           # plain FODO cells 2 and 4
            lat.add(quad(8.0))
            lat.add(drift(250.0))
            lat.add(quad(-8.0))
            lat.add(drift(100.0))
    return lat


def _plant_demo_misalignments(lat, sigma_mm=0.2, seed=2026) -> None:
    rng = np.random.default_rng(seed)
    for e in lat.elements:
        if isinstance(e, Quadrupole):
            e.dx = float(rng.normal(0.0, sigma_mm))
            e.dy = float(rng.normal(0.0, sigma_mm))


def _demo_cfg(n_particles=500) -> BeamConfig:
    return BeamConfig(
        species="proton", energy=5.0, frequency=352.21,
        current=0.0, n_particles=n_particles,
        distribution="gaussian", cutoff=4.0,
        emit_nx=0.05, emit_ny=0.05, emit_z=0.10,
        alpha_x=0.0, beta_x=2.0,
        alpha_y=0.0, beta_y=2.0,
        alpha_z=0.0, beta_z=1.0,
    )


def _demo_factory(cfg=None, seed=1):
    cfg = cfg or _demo_cfg()
    return lambda: create_beam(cfg, seed=seed)


def _first_dead_element(lat, factory) -> str | None:
    rec = Tracker(lat, factory()).run()
    for i, t in enumerate(rec.transmission):
        if not (float(t) > 0.0):
            return rec.element_names[i]
    return None


def _pre_dead_lattice() -> Lattice:
    """Beam 100 % dead BEFORE any corrector pass (razor aperture)."""
    lat = Lattice()
    lat.add(Drift("D_kill", 50.0, aperture=1e-3))
    lat.add(AdjustSteerer("ADJ_1", diag_n=1, vmax=0.0, first_step=1e-4))
    lat.add(Steerer("STEER_1", bx_l=0.0, by_l=0.0))
    lat.add(Drift("D_post", 100.0))
    lat.add(Marker("BPM_1", is_bpm=True))
    lat.add(Drift("D_after", 100.0))
    return lat


# =====================================================================
# Regime B — corrector-induced loss (the shipped defect)
# =====================================================================
def test_original_demo_geometry_one_to_one_reports_beam_lost():
    lat = _original_demo_lattice()
    _plant_demo_misalignments(lat)
    factory = _demo_factory()
    res = run_correction_from_lattice(lat, factory, n_iter=5, tol_mm=0.01,
                                      history=True)
    assert res["method"] == "one_to_one"
    assert res["status"] == "beam_lost"
    assert res["converged"] is False
    h = res["history"][-1]
    assert math.isnan(h["rms_orbit_mm"])       # never < tol_mm
    assert h["n_dead_bpms"] == 2               # BPM_003 and BPM_004
    assert h["transmission_pct"] == 0.0
    assert h["stop_reason"] == "beam_lost"
    assert len(res["history"]) == 1            # the dead pass is the last
    # N-independent cross-check: the reported element IS the first row
    # with zero transmission on an independent post-correction run.
    assert res["beam_lost_at"] == _first_dead_element(lat, factory)
    # Pairs downstream of the loss applied no kick (identical to the old
    # outcome, where R == 0 silently skipped them).
    assert res["kicks"]["STEER_003"] == {"bx_l": 0.0, "by_l": 0.0}
    assert res["kicks"]["STEER_004"] == {"bx_l": 0.0, "by_l": 0.0}


def test_original_demo_geometry_svd_masks_dead_rows_and_reports(caplog):
    # one_to_one run for the kick cross-check.
    lat1 = _original_demo_lattice()
    _plant_demo_misalignments(lat1)
    res1 = run_correction_from_lattice(lat1, _demo_factory(), n_iter=5,
                                       tol_mm=0.01, history=True)

    lat2 = _original_demo_lattice()
    _plant_demo_misalignments(lat2)
    with caplog.at_level(logging.WARNING, logger="linac_gen.errors.correction"):
        res2 = run_correction_from_lattice(lat2, _demo_factory(),
                                           override_method="svd",
                                           n_iter=5, tol_mm=0.01,
                                           history=True)
    assert res2["status"] == "beam_lost"
    assert res2["converged"] is False
    h = res2["history"][-1]
    assert math.isnan(h["rms_orbit_mm"])
    assert h["n_dead_bpms"] == 2
    # The beam is ALIVE during the SVD solve passes (it dies only after
    # the solved kicks land), so the solve itself is untouched: the two
    # upstream steerers get the same kicks as one_to_one (rtol 1e-9 —
    # the two solvers agree to ~8e-12 relative, measured).
    for name in ("STEER_001", "STEER_002"):
        for knob in ("bx_l", "by_l"):
            assert res2["kicks"][name][knob] == pytest.approx(
                res1["kicks"][name][knob], rel=1e-9)
    assert "beam LOST" in caplog.text
    assert "NOT converged" in caplog.text


@pytest.mark.parametrize("method", [None, "svd"])
def test_pre_dead_beam_both_methods_no_kicks_and_mask(method, caplog):
    """Beam dead BEFORE correction: the dead-row mask (SVD) and the
    skip-response-passes path (one_to_one) execute; zero kicks; the
    'already lost' warning fires."""
    lat = _pre_dead_lattice()
    with caplog.at_level(logging.WARNING, logger="linac_gen.errors.correction"):
        res = run_correction_from_lattice(lat, _factory(),
                                          override_method=method,
                                          n_iter=3, tol_mm=0.01,
                                          history=True)
    assert res["status"] == "beam_lost"
    assert res["converged"] is False
    assert res["beam_lost_at"] == "D_kill"
    assert res["kicks"]["STEER_1"] == {"bx_l": 0.0, "by_l": 0.0}
    assert len(res["history"]) == 1
    h = res["history"][0]
    assert h["n_dead_bpms"] == 1
    assert h["transmission_pct"] == 0.0
    assert math.isnan(h["rms_orbit_mm"])
    assert "already lost" in caplog.text
    if method == "svd":
        assert "excluded from the solve" in caplog.text
    else:
        assert "left unchanged" in caplog.text


# =====================================================================
# Regime A — alive beams stay bit-compatible (baseline-comparison test)
# =====================================================================
def test_alive_regime_pinned_baseline_one_to_one_svd_and_noise():
    # -- one_to_one, 3 pairs (values captured on the pre-fix tree;
    #    exact equality proven locally by the bitproof diff, rtol 1e-12
    #    here for cross-platform CI).
    lat = _make_pair_lattice(3)
    res = run_correction_from_lattice(lat, _factory(), n_iter=3,
                                      tol_mm=1e-6, history=True)
    expect = {
        "STEER_1": (-0.002409793932236104, -0.004956947122189652),
        "STEER_2": (0.007222976992183668, 0.01487253235606924),
        "STEER_3": (-0.014445953984358755, -0.02974506471167223),
    }
    for name, (bx, by) in expect.items():
        assert res["kicks"][name]["bx_l"] == pytest.approx(bx, rel=1e-12)
        assert res["kicks"][name]["by_l"] == pytest.approx(by, rel=1e-12)
    assert len(res["history"]) == 1
    h = res["history"][0]
    assert h["rms_orbit_mm"] == pytest.approx(7.521197586654734e-11, abs=1e-9)
    assert res["status"] == "converged"
    assert res["converged"] is True
    assert res["beam_lost_at"] is None
    assert h["n_dead_bpms"] == 0
    assert h["transmission_pct"] == 100.0
    assert h["beam_lost_at"] is None
    assert h["stop_reason"] == "converged"

    # -- svd
    lat = _make_pair_lattice(3)
    res = run_correction_from_lattice(lat, _factory(),
                                      override_method="svd",
                                      n_iter=2, history=True)
    expect = {
        "STEER_1": (-0.002409793932236099, -0.0049569471221896505),
        "STEER_2": (0.0072229769921697655, 0.014872532356047229),
        "STEER_3": (-0.01444595398435343, -0.029745064712050434),
    }
    for name, (bx, by) in expect.items():
        assert res["kicks"][name]["bx_l"] == pytest.approx(bx, rel=1e-12)
        assert res["kicks"][name]["by_l"] == pytest.approx(by, rel=1e-12)
    assert res["history"][0]["rms_orbit_mm"] == pytest.approx(
        4.7353209987049904e-12, abs=1e-9)
    assert res["status"] == "converged"

    # -- bpm_noise: pins the rng noise-draw ORDER (dead rows must still
    #    draw so alive streams stay identical).
    lat = _fodo_with_pair()
    kicks, hist = apply_correction(lat, _factory(), bpm_noise=0.05,
                                   noise_seed=7, n_iter=3, tol_mm=1e-9,
                                   history=True)
    assert kicks["STEER_1"]["bx_l"] == pytest.approx(
        -0.0011671109091573692, rel=1e-12)
    assert kicks["STEER_1"]["by_l"] == pytest.approx(
        -0.0024473702442634826, rel=1e-12)
    expect_rms = [0.043161773231963824, 0.08442906275923838,
                  0.0489854020414229]
    assert [h["rms_orbit_mm"] for h in hist] == pytest.approx(
        expect_rms, rel=1e-12)
    assert all(h["n_dead_bpms"] == 0 for h in hist)
    assert hist[-1]["stop_reason"] == "max_iter"


def test_envelope_backend_has_no_dead_state():
    """EnvelopeResults carries no ``transmission`` (no loss model) —
    readings are never flagged dead; transmission_pct is None."""
    cfg = BeamConfig(
        species="proton", energy=3.0, frequency=352.21,
        current=0.0, n_particles=100,
        distribution="gaussian", cutoff=4.0,
        emit_nx=0.05, emit_ny=0.05, emit_z=0.10,
        alpha_x=0.0, beta_x=2.0, alpha_y=0.0, beta_y=2.0,
        alpha_z=0.0, beta_z=1.0)
    lat = _make_pair_lattice(2)
    res = run_correction_from_lattice(lat, None, reading_backend="envelope",
                                      beam_config=cfg, n_iter=2,
                                      history=True)
    for h in res["history"]:
        assert h["n_dead_bpms"] == 0
        assert h["transmission_pct"] is None
        assert h["beam_lost_at"] is None
    assert res["status"] in ("converged", "max_iter")
    assert res["converged"] == (res["history"][-1]["rms_orbit_mm"] < 0.05)
    assert res["beam_lost_at"] is None


# =====================================================================
# Unit level — both regimes of the helpers
# =====================================================================
def test_live_bpm_reading_alive_rows_bit_identical_and_dead_rows_none():
    lat = _make_pair_lattice(3)
    rec = Tracker(lat, _factory()()).run()
    for i, e in enumerate(lat.elements):
        if getattr(e, "is_bpm", False):
            row = rec.element_exit_idx[i]
            c = np.array(rec.centroid[row])
            # EXACT equality with what every caller computed before.
            assert _live_bpm_reading(lat, rec, e) == (float(c[0]),
                                                      float(c[2]))

    lat1 = Lattice()
    bpm = Marker("BPM_1", is_bpm=True)
    lat1.add(bpm)
    dead = stub_results(transmission=[100.0, 0.0],
                        centroid=[[1, 2, 3, 4, 5, 6], [0.0] * 6],
                        element_names=["INPUT", "D1"],
                        element_exit_idx=[1])
    assert _live_bpm_reading(lat1, dead, bpm) is None
    assert _beam_lost_at(dead) == "D1"

    no_loss_model = stub_results(centroid=[[1, 2, 3, 4, 5, 6],
                                           [0.5, 0.0, -0.25, 0.0, 0.0, 0.0]],
                                 element_exit_idx=[1])
    assert _live_bpm_reading(lat1, no_loss_model, bpm) == (0.5, -0.25)
    assert _beam_lost_at(no_loss_model) is None


def test_correction_status_vocabulary():
    assert correction_status([]) == {"status": "none", "converged": False,
                                     "beam_lost_at": None,
                                     "rms_final_mm": None}
    st = correction_status([{"rms_orbit_mm": 0.01, "stop_reason": "converged",
                             "beam_lost_at": None}])
    assert st["status"] == "converged" and st["converged"] is True
    st = correction_status([{"rms_orbit_mm": float("nan"),
                             "stop_reason": "beam_lost",
                             "beam_lost_at": "QUAD_007"}])
    assert st["status"] == "beam_lost"
    assert st["converged"] is False
    assert st["beam_lost_at"] == "QUAD_007"
