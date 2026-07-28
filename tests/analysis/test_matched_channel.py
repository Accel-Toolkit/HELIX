"""Matched-channel state (full-Σ fixed point) + beam-integral guards.

``find_matched_period_sigma`` iterates Σ → R(M_dep·Σ·M_depᵀ) through
real envelope passes with eigenemittance renormalization (normalized-
coordinate matching — raw covariance iteration collapses under
acceleration since det(M₄) < 1).  ``channel_phase_advance_matched``
runs the phase probe from that state: the primary σ_model, independent
of how well the user's tracked beam was matched.

Also pinned: the new ``beam_phase_advance`` validity machinery —
covariance (BMAG) mismatch, resolution flag, and the coupled-interior
(projected-β) guard.
"""
from __future__ import annotations

import warnings

import numpy as np
import pytest

from linac_gen.analysis.period_detect import PeriodicStructure, detect_periods
from linac_gen.analysis.phase_advance import (
    beam_phase_advance, channel_phase_advance,
    channel_phase_advance_matched, mismatch_factor,
)
from linac_gen.core.lattice import Lattice
from linac_gen.core.particle import PROTON
from linac_gen.core.reference import ReferenceParticle
from linac_gen.elements.drift import Drift
from linac_gen.elements.quadrupole import Quadrupole
from linac_gen.elements.solenoid import Solenoid
from linac_gen.matching.periodic import (
    _renormalize_eigenemittances, _sigma_modes, find_matched_period_sigma,
    find_periodic_twiss,
)
from linac_gen.tracking.envelope import EnvelopeSolver


def _fodo(n_cells: int = 3) -> Lattice:
    lat = Lattice()
    for _ in range(n_cells):
        lat.add(Drift(name="D", length=100.0, aperture=10.0))
        lat.add(Quadrupole(name="QF", length=50.0, gradient=+10.0,
                           aperture=10.0))
        lat.add(Drift(name="D", length=100.0, aperture=10.0))
        lat.add(Quadrupole(name="QD", length=50.0, gradient=-10.0,
                           aperture=10.0))
    return lat


def _ref() -> ReferenceParticle:
    return ReferenceParticle(species=PROTON, w_kin=2.5, frequency=162.5)


def _dc_initial(tw) -> dict:
    return dict(alpha_x=tw["alpha_x"], beta_x=tw["beta_x"], emit_x=1.0,
                alpha_y=tw["alpha_y"], beta_y=tw["beta_y"], emit_y=1.0,
                alpha_z=0.0, beta_z=1.0, emit_z=0.0, continuous=True)


def test_sigma_mode_roundtrip():
    """Decompose → rebuild must reproduce Σ; renormalizing to its own
    emittances is the identity."""
    rng = np.random.default_rng(7)
    A = rng.normal(size=(6, 6))
    Sigma = A @ A.T + 6.0 * np.eye(6)
    modes = _sigma_modes(Sigma)
    assert len(modes) == 3
    rebuilt = _renormalize_eigenemittances(Sigma, modes)
    np.testing.assert_allclose(rebuilt, Sigma, rtol=1e-9, atol=1e-9)


def test_matched_period_sigma_fodo_converges_and_is_periodic():
    lat = _fodo()
    ref = _ref()
    tw = find_periodic_twiss(lat, ref)
    period = next(p for p in detect_periods(lat) if p.n_repeats >= 3)
    ms = find_matched_period_sigma(lat, ref, period, current=20.0,
                                   base_initial=_dc_initial(tw))
    assert ms["converged"], ms
    # The matched Σ must be (numerically) periodic over the cell:
    # one more pass returns it unchanged.
    a0, b0 = period.spans()[0]
    cell = Lattice()
    for e in lat.elements[a0:b0]:
        cell.add(e)
    env = EnvelopeSolver(cell, ms["ref_entry"].copy(), _dc_initial(tw),
                         current=20.0, initial_sigma=ms["sigma_entry"],
                         bunch_frequency=ms["bunch_frequency"],
                         sc_factor=ms["sc_factor"]).run()
    S_out = np.asarray(env.sigma_matrix[-1])
    # Transverse block periodic to fixed-point precision; the (near-)
    # zero-emittance z sector of a DC beam is preserved rather than
    # solved and only needs to stay near zero.
    scale = np.abs(ms["sigma_entry"][:4, :4]).max()
    d4 = np.abs(S_out[:4, :4] - ms["sigma_entry"][:4, :4]).max() / scale
    assert d4 < 1e-5, d4
    assert np.abs(S_out[4:, :]).max() < 1e-3 * scale


def test_channel_matched_agrees_with_tracked_matched_run():
    """When the user's beam IS matched, the matched-channel tunes equal
    the plain channel tunes of the tracked run."""
    from scipy.optimize import root
    lat = _fodo()
    ref = _ref()
    period = next(p for p in detect_periods(lat) if p.n_repeats >= 3)
    tw = find_periodic_twiss(lat, ref)

    out_m = channel_phase_advance_matched(lat, ref, period, 20.0,
                                          _dc_initial(tw))
    assert out_m["matched_state"]["converged"]

    # Independent tracked-matched run (root on the cell periodicity).
    cell = Lattice()
    for e in lat.elements[:4]:
        cell.add(e)

    def _out(state):
        init = dict(alpha_x=state[0], beta_x=state[1], emit_x=1.0,
                    alpha_y=state[2], beta_y=state[3], emit_y=1.0,
                    alpha_z=0.0, beta_z=1.0, emit_z=0.0, continuous=True)
        r = EnvelopeSolver(cell, ref.copy(), init, current=20.0).run()
        S = np.asarray(r.sigma_matrix[-1])
        o = []
        for i, j in ((0, 1), (2, 3)):
            eps = np.sqrt(max(S[i, i] * S[j, j] - S[i, j] ** 2, 1e-30))
            o += [-S[i, j] / eps, S[i, i] / eps]
        return np.asarray(o)

    seed = np.asarray([tw["alpha_x"], tw["beta_x"],
                       tw["alpha_y"], tw["beta_y"]])
    sol = root(lambda v: _out(v) - v, seed, method="hybr", tol=1e-12)
    init = dict(alpha_x=sol.x[0], beta_x=sol.x[1], emit_x=1.0,
                alpha_y=sol.x[2], beta_y=sol.x[3], emit_y=1.0,
                alpha_z=0.0, beta_z=1.0, emit_z=0.0, continuous=True)
    res = EnvelopeSolver(lat, ref.copy(), init, current=20.0,
                         phase_probe=True).run()
    out_t = channel_phase_advance(res, period)

    assert out_m["mu_x_dep_deg"][0] == pytest.approx(
        out_t["mu_x_dep_deg"][0], rel=1e-4)
    assert out_m["mu_y_dep_deg"][0] == pytest.approx(
        out_t["mu_y_dep_deg"][0], rel=1e-4)


def test_frozen_longitudinal_is_labelled():
    lat = _fodo()
    ref = _ref()
    tw = find_periodic_twiss(lat, ref)
    period = next(p for p in detect_periods(lat) if p.n_repeats >= 3)
    ms = find_matched_period_sigma(lat, ref, period, current=5.0,
                                   base_initial=_dc_initial(tw),
                                   longitudinal="frozen")
    assert ms["sigma_model_approx"] == "frozen-longitudinal"


def test_bmag_mismatch_flags_alpha_mismatch():
    """β-endpoints can agree while α is mismatched — BMAG must catch it."""
    assert mismatch_factor(0.0, 2.0, 0.0, 2.0) == pytest.approx(1.0)
    assert mismatch_factor(1.0, 2.0, -1.0, 2.0) > 1.5

    # An envelope launched with flipped α returns to the same β at the
    # cell end (time-reversal symmetry of the mismatch oscillation) but
    # is NOT matched; the covariance check must veto it.
    lat = _fodo()
    ref = _ref()
    tw = find_periodic_twiss(lat, ref)
    period = next(p for p in detect_periods(lat) if p.n_repeats >= 3)
    good = dict(alpha_x=tw["alpha_x"], beta_x=tw["beta_x"], emit_x=1.0,
                alpha_y=tw["alpha_y"], beta_y=tw["beta_y"], emit_y=1.0,
                alpha_z=0.0, beta_z=1.0, emit_z=0.0, continuous=True)
    res_good = EnvelopeSolver(lat, ref.copy(), good, current=0.0).run()
    out_good = beam_phase_advance(res_good, period)
    assert out_good["matched"]
    assert out_good["mismatch_bmag_x"] == pytest.approx(1.0, abs=1e-6)

    bad = dict(good, alpha_x=-tw["alpha_x"], alpha_y=-tw["alpha_y"])
    res_bad = EnvelopeSolver(lat, ref.copy(), bad, current=0.0).run()
    out_bad = beam_phase_advance(res_bad, period)
    assert not out_bad["matched"]
    x_ok = out_bad["mismatch_bmag_x"] is not None and \
        out_bad["mismatch_bmag_x"] > 1.05
    y_ok = out_bad["mismatch_bmag_y"] is not None and \
        out_bad["mismatch_bmag_y"] > 1.05
    assert x_ok or y_ok


def test_resolution_keys():
    lat = _fodo()
    ref = _ref()
    tw = find_periodic_twiss(lat, ref)
    period = next(p for p in detect_periods(lat) if p.n_repeats >= 3)
    init = _dc_initial(tw)
    coarse = EnvelopeSolver(lat, ref.copy(), init, current=0.0).run()
    fine = EnvelopeSolver(lat, ref.copy(), init, current=0.0,
                          record_substeps=True).run()
    out_c = beam_phase_advance(coarse, period)
    out_f = beam_phase_advance(fine, period)
    assert out_f["n_samples"] > out_c["n_samples"]
    assert out_f["samples_per_period"] > out_c["samples_per_period"]
    assert out_f["resolution_ok"]


def test_projected_beta_guard_fires_inside_solenoid():
    """Substep records inside a solenoid are x–y coupled: the guard
    must warn and set projected_only."""
    lat = Lattice()
    for _ in range(2):
        lat.add(Drift(name="D", length=100.0, aperture=20.0))
        lat.add(Solenoid(name="SOL", length=200.0, field=1.2, aperture=20.0))
        lat.add(Drift(name="D", length=100.0, aperture=20.0))
    ref = _ref()
    period = PeriodicStructure(start=0, end=6, inner_period_length=3,
                               inner_slice_end=3, n_repeats=2,
                               label="sol", source="manual")
    # An ASYMMETRIC beam: a round beam is rotation-invariant, so a
    # solenoid never develops x-y cross terms for it and the guard
    # (correctly) stays silent.
    init = dict(alpha_x=0.0, beta_x=1.0, emit_x=0.5,
                alpha_y=0.0, beta_y=3.0, emit_y=0.2,
                alpha_z=0.0, beta_z=1.0, emit_z=0.0, continuous=True)
    res = EnvelopeSolver(lat, ref.copy(), init, current=0.0,
                         record_substeps=True).run()
    with pytest.warns(UserWarning, match="projected"):
        out = beam_phase_advance(res, period)
    assert out["projected_only"]

    # An uncoupled lattice (quads only) must stay quiet.
    fodo = _fodo()
    ref2 = _ref()
    tw = find_periodic_twiss(fodo, ref2)
    period_f = next(p for p in detect_periods(fodo) if p.n_repeats >= 3)
    res_f = EnvelopeSolver(fodo, ref2.copy(), _dc_initial(tw),
                           current=0.0, record_substeps=True).run()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        out_f = beam_phase_advance(res_f, period_f)
    assert not out_f["projected_only"]
