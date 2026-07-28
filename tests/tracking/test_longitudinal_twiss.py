# tests/tracking/test_longitudinal_twiss.py
"""Longitudinal Twiss recording (2026-07) — makes SET_TWISS kaz/kbz real.

Conventions pinned here (per the migration appendix):
* alpha_z is HELIX-internal: alpha_z = -<dphi.dW>/eps_z = -Sigma45/eps_z
  (TW's alpha_z is the NEGATIVE of this).
* beta_z = Sigma44/eps_z in deg/MeV at the local machine clock —
  scales f_new/f_old across a FREQ card; alpha_z is invariant there.
"""
import warnings

import numpy as np
import pytest

from linac_gen.core.beam import Beam
from linac_gen.core.config import BeamConfig
from linac_gen.core.lattice import Lattice
from linac_gen.core.particle import PROTON, H_MINUS
from linac_gen.core.reference import ReferenceParticle
from linac_gen.elements.drift import Drift
from linac_gen.elements.lattice_commands import (
    AdjustBeamTwiss, Freq, SetTwiss,
)
from linac_gen.elements.quadrupole import Quadrupole
from linac_gen.tracking.envelope import EnvelopeSolver

INITIAL = dict(beta_x=2.0, alpha_x=0.0, emit_x=0.25,
               beta_y=2.0, alpha_y=0.0, emit_y=0.25,
               beta_z=5.0, alpha_z=0.7, emit_z=0.30)


def _lattice():
    lat = Lattice()
    lat.add(Quadrupole("QF", 50.0, gradient=5.0, n_steps=5))
    lat.add(Drift("D1", 300.0))
    lat.add(Quadrupole("QD", 50.0, gradient=-5.0, n_steps=5))
    lat.add(Drift("D2", 300.0))
    return lat


def _run_envelope(lat, w_kin=3.0, freq=352.21, initial=None):
    ref = ReferenceParticle(species=PROTON, w_kin=w_kin, frequency=freq)
    return EnvelopeSolver(lat, ref, dict(initial or INITIAL),
                         current=0.0).run()


def test_envelope_rows_match_sigma_block():
    """Every recorded (alpha_z, beta_z) row equals an independent
    recomputation from the stored 6x6 sigma matrix z-block."""
    res = _run_envelope(_lattice())
    assert len(res.alpha_z) == len(res.s) == len(res.beta_z)
    for i, sig in enumerate(res.sigma_matrix):
        s44, s45, s55 = sig[4, 4], sig[4, 5], sig[5, 5]
        eps = np.sqrt(max(s44 * s55 - s45 * s45, 0.0))
        assert res.alpha_z[i] == pytest.approx(-s45 / eps, rel=1e-12)
        assert res.beta_z[i] == pytest.approx(s44 / eps, rel=1e-12)


def test_row0_matches_seed_and_drift_evolves_alpha():
    """Row 0 reproduces the seeded z-Twiss exactly; a drift changes it
    (proving the constraint residual has a usable gradient)."""
    res = _run_envelope(_lattice())
    assert res.alpha_z[0] == pytest.approx(INITIAL["alpha_z"], rel=1e-9)
    assert res.beta_z[0] == pytest.approx(INITIAL["beta_z"], rel=1e-9)
    assert abs(res.alpha_z[-1] - res.alpha_z[0]) > 1e-6


def test_sign_convention_pinned_against_raw_moments():
    """External-truth pin (not a round-trip): alpha_z = +0.5 must mean
    a NEGATIVE raw <dphi.dW> correlation on generated particles, and
    the MP recorder must report back +0.5 at s=0."""
    from linac_gen.distributions.factory import create_beam
    cfg = BeamConfig(species="proton", energy=3.0, frequency=352.21,
                     current=0.0, duty_cycle=100.0, n_particles=20000,
                     distribution="gaussian", cutoff=4.0,
                     emit_nx=0.25, alpha_x=0.0, beta_x=2.0,
                     emit_ny=0.25, alpha_y=0.0, beta_y=2.0,
                     emit_z=0.30, alpha_z=+0.5, beta_z=5.0)
    beam = create_beam(cfg, seed=7)
    p = beam.particles
    dphi = p[:, 4] - p[:, 4].mean()
    dw = p[:, 5] - p[:, 5].mean()
    assert float(np.mean(dphi * dw)) < 0.0        # alpha>0 <=> corr<0

    from linac_gen.diagnostics.moments import compute_twiss_from_particles
    tw = compute_twiss_from_particles(p, "z")
    assert tw["alpha"] == pytest.approx(0.5, abs=0.03)
    assert tw["beta"] == pytest.approx(5.0, rel=0.05)


def test_mp_recorder_matches_envelope_at_zero_current():
    """MP recorder z-Twiss agrees with the envelope at I=0 (parity)."""
    from linac_gen.distributions.factory import create_beam
    from linac_gen.tracking.tracker import Tracker

    lat = _lattice()
    env = _run_envelope(lat)

    cfg = BeamConfig(species="proton", energy=3.0, frequency=352.21,
                     current=0.0, duty_cycle=100.0, n_particles=50000,
                     distribution="gaussian", cutoff=6.0,
                     emit_nx=0.25 * 0.0731 * 1.0, alpha_x=0.0, beta_x=2.0,
                     emit_ny=0.25 * 0.0731 * 1.0, alpha_y=0.0, beta_y=2.0,
                     emit_z=0.30, alpha_z=0.7, beta_z=5.0)
    beam = create_beam(cfg, seed=11)
    lat2 = _lattice()
    rec = Tracker(lat2, beam).run()
    assert rec.alpha_z, "MP recorder must carry alpha_z"
    # entrance parity (exact seed) and exit parity (transport)
    assert rec.alpha_z[0] == pytest.approx(0.7, abs=0.03)
    assert rec.beta_z[0] == pytest.approx(5.0, rel=0.05)
    assert rec.alpha_z[-1] == pytest.approx(env.alpha_z[-1], abs=0.05)
    assert rec.beta_z[-1] == pytest.approx(env.beta_z[-1], rel=0.05)


def test_freq_jump_alpha_invariant_beta_scales():
    """Across a FREQ card f->r*f: alpha_z continuous, beta_z x r."""
    lat = Lattice()
    lat.add(Drift("D1", 100.0))
    lat.add(Freq("FREQ", frequency_mhz=325.0))
    lat.add(Drift("D2", 100.0))
    res = _run_envelope(lat, w_kin=2.1, freq=162.5)
    fr = np.asarray(res.ref_frequency)
    j = int(np.where(np.diff(fr) != 0)[0][0])
    r = fr[j + 1] / fr[j]
    assert r == pytest.approx(2.0)
    assert res.alpha_z[j + 1] == pytest.approx(res.alpha_z[j], rel=1e-9)
    assert res.beta_z[j + 1] == pytest.approx(res.beta_z[j] * r, rel=1e-9)


# ---------------------------------------------------------------------------
# SET_TWISS kaz/kbz — evaluator + end-to-end matcher
# ---------------------------------------------------------------------------
def test_set_twiss_z_axes_evaluated_no_warning():
    """kaz/kbz now produce genuine residual entries with no z-skip
    warning, equal to (recorded - target)."""
    from linac_gen.matching.constraints import collect_constraints

    lat = _lattice()
    res = _run_envelope(lat)
    lat.add(SetTwiss("ST", "", 0, 0, 0, 0, 0.2, 6.0,
                     0, 0, 0, 0, 1, 1))
    cs = [c for c in collect_constraints(lat)
          if c.label.startswith("SET_TWISS")]
    assert len(cs) == 1
    with warnings.catch_warnings():
        warnings.simplefilter("error")            # any warning -> fail
        r = cs[0].evaluate(res, lat)
    assert r.shape == (2,)
    assert r[0] == pytest.approx(res.alpha_z[-1] - 0.2, rel=1e-9)
    assert r[1] == pytest.approx(res.beta_z[-1] - 6.0, rel=1e-9)


def test_set_twiss_z_legacy_results_warn_and_fill_zero():
    """Legacy results (no z record): warn once, and the residual keeps
    its FIXED length with the z slots filled 0 (never resized)."""
    from types import SimpleNamespace

    from linac_gen.matching.constraints import _make_set_twiss_evaluator

    ev = _make_set_twiss_evaluator(
        SetTwiss("ST", "", 0, 0, 0, 0, 0.2, 6.0, 0, 0, 0, 0, 1, 1))
    from linac_gen.matching.constraints import _Z_DEGENERATE_PENALTY
    legacy = SimpleNamespace(s=[0.0], alpha_x=[0.0], beta_x=[1.0],
                             alpha_y=[0.0], beta_y=[1.0])
    with pytest.warns(UserWarning, match="unusable"):
        r = ev(legacy, None)
    # fixed length, stateless penalty fill (worker-safe)
    np.testing.assert_allclose(
        r, [_Z_DEGENERATE_PENALTY, _Z_DEGENERATE_PENALTY])
    with warnings.catch_warnings():
        warnings.simplefilter("error")            # warn-once only
        r2 = ev(legacy, None)
    np.testing.assert_allclose(r2, r)


def test_set_twiss_z_degenerate_dc_beam_warns_and_fills_zero():
    from types import SimpleNamespace

    from linac_gen.matching.constraints import _make_set_twiss_evaluator

    ev = _make_set_twiss_evaluator(
        SetTwiss("ST", "", 0, 0, 0, 0, 0.2, 6.0, 0, 0, 0, 0, 1, 1))
    from linac_gen.matching.constraints import _Z_DEGENERATE_PENALTY
    dc = SimpleNamespace(s=[0.0], alpha_x=[0.0], beta_x=[1.0],
                         alpha_y=[0.0], beta_y=[1.0],
                         alpha_z=[0.1], beta_z=[2.0], emit_z=[45.0],
                         continuous_at=[True])
    with pytest.warns(UserWarning, match="unusable"):
        r = ev(dc, None)
    np.testing.assert_allclose(
        r, [_Z_DEGENERATE_PENALTY, _Z_DEGENERATE_PENALTY])


def test_set_twiss_z_residual_length_constant_mid_run():
    """Reviewer follow-up: the residual vector must NEVER change length
    under scipy least_squares.  Healthy baseline -> real values; a
    trial point that goes degenerate -> same length, both z slots =
    the fixed penalty; recovery -> real values again."""
    from types import SimpleNamespace

    from linac_gen.matching.constraints import (
        _Z_DEGENERATE_PENALTY, _make_set_twiss_evaluator,
    )

    ev = _make_set_twiss_evaluator(
        SetTwiss("ST", "", 0, 0, 0, 0, 0.2, 6.0, 0, 0, 0, 0, 1, 1))
    healthy = SimpleNamespace(s=[0.0], alpha_x=[0.0], beta_x=[1.0],
                              alpha_y=[0.0], beta_y=[1.0],
                              alpha_z=[0.5], beta_z=[7.0],
                              emit_z=[0.3])
    degenerate = SimpleNamespace(s=[0.0], alpha_x=[0.0], beta_x=[1.0],
                                 alpha_y=[0.0], beta_y=[1.0],
                                 alpha_z=[0.0], beta_z=[0.0],
                                 emit_z=[0.0])
    r0 = ev(healthy, None)                        # baseline: healthy
    np.testing.assert_allclose(r0, [0.5 - 0.2, 7.0 - 6.0])
    with pytest.warns(UserWarning, match="penalty"):
        r1 = ev(degenerate, None)                 # trial went bad
    assert r1.shape == r0.shape
    np.testing.assert_allclose(
        r1, [_Z_DEGENERATE_PENALTY, _Z_DEGENERATE_PENALTY])
    r2 = ev(healthy, None)                        # recovery
    np.testing.assert_allclose(r2, r0)
    # STATELESS (worker-safety): a fresh closure whose FIRST call is
    # the degenerate trial must score it identically — never 0.0
    # (cmaes_parallel workers rebuild closures and never see x0).
    ev2 = _make_set_twiss_evaluator(
        SetTwiss("ST", "", 0, 0, 0, 0, 0.2, 6.0, 0, 0, 0, 0, 1, 1))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r_worker = ev2(degenerate, None)
    np.testing.assert_allclose(r_worker, r1)


def test_z_penalty_infeasible_is_weight_aware():
    """per_constraint_residuals stores POST-weight values; the flag
    must detect weight*penalty, not only the bare constant, so a
    future weighted SET_TWISS cannot silently break it."""
    from types import SimpleNamespace

    from linac_gen.matching.constraints import _Z_DEGENERATE_PENALTY
    from linac_gen.matching.engine import MatchResult

    def result(weight, values):
        c = SimpleNamespace(label="SET_TWISS:END", weight=weight)
        return MatchResult(
            success=True, message="", n_iter=1, elapsed_s=0.0,
            x0=np.zeros(1), x_final=np.zeros(1),
            residuals=np.asarray(values), cost=0.0,
            variables=[], constraints=[c],
            per_constraint_residuals={"SET_TWISS:END":
                                      np.asarray(values)})

    p = _Z_DEGENERATE_PENALTY
    assert result(1.0, [p, p]).z_penalty_infeasible is True
    assert result(2.5, [2.5 * p, 2.5 * p]).z_penalty_infeasible is True
    assert result(2.5, [p, p]).z_penalty_infeasible is False   # not w*p
    assert result(1.0, [0.3, 1.0]).z_penalty_infeasible is False


def test_match_baseline_raises_on_degenerate_z_flags():
    """End-to-end: a DC (continuous) beam through a transverse line
    with SET_TWISS kaz=kbz=1 must RAISE at the baseline evaluation
    (naming the card) unless allow_inert_constraints=True, which
    warns and completes."""
    from linac_gen.elements.lattice_commands import Adjust
    from linac_gen.matching import match

    def build():
        lat = Lattice()
        lat.add(Drift("D1", length=100.0))
        lat.add(Adjust("A1", target="QF", param_idx=2, link_group=1,
                       vmin=0.0, vmax=15.0, start_step=0.5))
        lat.add(Quadrupole("QF", length=80.0, gradient=5.0,
                           aperture=20.0))
        lat.add(Drift("D2", length=200.0))
        lat.add(SetTwiss("ST", "", 0, 0, 0, 0, 0.2, 6.0,
                         0, 0, 0, 0, 1, 1))
        return lat

    cfg = BeamConfig(species="proton", energy=3.0, frequency=352.21,
                     current=0.0, duty_cycle=100.0, n_particles=10,
                     distribution="waterbag", cutoff=3.0,
                     emit_nx=0.25, alpha_x=0.0, beta_x=1.0,
                     emit_ny=0.25, alpha_y=0.0, beta_y=1.0,
                     emit_z=0.30, alpha_z=0.0, beta_z=1.0,
                     continuous=True)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with pytest.raises(ValueError, match="SET_TWISS"):
            match(build(), cfg, algorithm="least_squares", max_iter=3,
                  cost_solver="envelope")
        res = match(build(), cfg, algorithm="least_squares", max_iter=3,
                    cost_solver="envelope", allow_inert_constraints=True)
    assert res is not None
    # machine-readable infeasibility: the override solution's z slots
    # carry the penalty, and the report says so next to the status
    assert res.z_penalty_infeasible is True
    assert "INFEASIBLE" in res.report()


def test_set_twiss_z_dc_to_bunched_exit_is_evaluated():
    """Adversarial-review fix: the envelope's scalar `continuous` flag
    is the run-START state and goes stale across a DC->bunched
    transition.  A results object with continuous=True but a genuinely
    bunched exit (emit_z end > 0) must be EVALUATED, matching the MP
    cost solver's per-row behaviour — not falsely skipped."""
    from types import SimpleNamespace

    from linac_gen.matching.constraints import (
        _make_set_twiss_evaluator, _z_twiss_degenerate,
    )

    stale = SimpleNamespace(s=[0.0, 1.0], alpha_x=[0.0, 0.0],
                            beta_x=[1.0, 1.0], alpha_y=[0.0, 0.0],
                            beta_y=[1.0, 1.0],
                            alpha_z=[0.0, -1.2], beta_z=[1.0, 8.5],
                            emit_z=[0.0, 0.104],       # bunched exit
                            continuous=True)           # stale seed flag
    assert not _z_twiss_degenerate(stale)
    ev = _make_set_twiss_evaluator(
        SetTwiss("ST", "", 0, 0, 0, 0, -1.0, 8.0, 0, 0, 0, 0, 1, 1))
    with warnings.catch_warnings():
        warnings.simplefilter("error")                 # no skip warning
        r = ev(stale, None)
    np.testing.assert_allclose(r, [-1.2 - (-1.0), 8.5 - 8.0])

    # MP-style per-row flag still authoritative: DC AT the exit skips
    # even when the sampled emit_z is numerically large (uniform phase).
    mp_dc = SimpleNamespace(s=[0.0], alpha_x=[0.0], beta_x=[1.0],
                            alpha_y=[0.0], beta_y=[1.0],
                            alpha_z=[0.1], beta_z=[2.0],
                            emit_z=[45.0], continuous_at=[True])
    assert _z_twiss_degenerate(mp_dc)


def test_set_twiss_z_end_to_end_matcher():
    """least_squares with ADJUST_BEAM_TWISS (z axes) + SET_TWISS
    kaz=kbz=1 genuinely reduces the z-Twiss residual — the flags are
    matchable knobs now, not warn-and-skip."""
    from linac_gen.matching import match

    def build():
        lat = Lattice()
        lat.add(AdjustBeamTwiss("ABT", 0, 0, 0, 0, 0, 1, 1))
        lat.add(Drift("D1", 400.0))
        lat.add(SetTwiss("ST", "", 0, 0, 0, 0, -0.3, 4.0,
                         0, 0, 0, 0, 1, 1))
        return lat

    # 80 MeV keeps the longitudinal drift term modest (~3 deg/MeV over
    # 400 mm) so the (-0.3, 4.0) target is reachable from in-bounds
    # input Twiss; at 3 MeV the beta^3 drift factor makes it not so.
    cfg = BeamConfig(species="proton", energy=80.0, frequency=352.21,
                     current=0.0, duty_cycle=100.0, n_particles=10,
                     distribution="waterbag", cutoff=3.0,
                     emit_nx=0.25, alpha_x=0.0, beta_x=1.0,
                     emit_ny=0.25, alpha_y=0.0, beta_y=1.0,
                     emit_z=0.30, alpha_z=0.9, beta_z=9.0)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = match(build(), cfg, algorithm="least_squares",
                    max_iter=80, cost_solver="envelope")
    assert res is not None
    # 2 knobs (input alpha_z/beta_z) vs 2 targets through a drift —
    # an exactly-solvable problem: the matcher must essentially zero it.
    assert res.cost < 1e-6, res.cost
    assert not np.allclose(res.x_final, res.x0)   # knobs actually moved
    # a genuinely-met z target is NOT flagged infeasible
    assert res.z_penalty_infeasible is False
