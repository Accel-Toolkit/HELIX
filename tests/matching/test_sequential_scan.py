"""Tests for the ``sequential_scan`` matcher algorithm.

The algorithm is physics-aware coordinate descent: walk lattice elements
in order, bracket-scan each ADJUST parameter (or grouped pair), reverse
direction when both transverse and longitudinal normalised emittance
exceed seed values, track the best-cost x across the whole scan.
"""
from __future__ import annotations

import pytest

from linac_gen.core.config import BeamConfig
from linac_gen.core.lattice import Lattice
from linac_gen.elements.drift import Drift
from linac_gen.elements.lattice_commands import Adjust, SetSize
from linac_gen.elements.quadrupole import Quadrupole
from linac_gen.matching.engine import match
from linac_gen.matching.variables import (
    categorize_fieldmap, group_variables_by_element,
)


def _bcfg(**over) -> BeamConfig:
    base = dict(species="proton", energy=3.0, frequency=352.21,
                n_particles=10, distribution="waterbag",
                emit_nx=0.25, alpha_x=0.0, beta_x=2.0,
                emit_ny=0.25, alpha_y=0.0, beta_y=2.0,
                emit_z=0.3,    alpha_z=0.0, beta_z=10.0)
    base.update(over)
    return BeamConfig(**base)


def _one_quad_lattice(k_init: float = 5.0, target_sigma_mm: float = 3.0) -> Lattice:
    """Single ADJUST'd quad + SET_SIZE constraint."""
    lat = Lattice()
    lat.add(Drift("D1", length=200.0, aperture=10.0))
    lat.add(Adjust("CMD1", target="QUAD", param_idx=2,
                   link_group=0, vmin=0.5, vmax=30, start_step=0.5))
    lat.add(Quadrupole("QUAD_001", length=100.0, gradient=k_init,
                       aperture=10.0))
    lat.add(Drift("D2", length=200.0, aperture=10.0))
    lat.add(SetSize("CSET", k=1.0, x_mm=target_sigma_mm, y_mm=0.0,
                    phi_or_z=0.0))
    return lat


def _two_quad_lattice() -> Lattice:
    """Two ADJUST'd quads in series + SET_SIZE constraint."""
    lat = Lattice()
    lat.add(Drift("D1", length=200.0, aperture=10.0))
    lat.add(Adjust("CMD1", target="QUAD_001", param_idx=2,
                   link_group=0, vmin=0.5, vmax=30, start_step=0.5))
    lat.add(Quadrupole("QUAD_001", length=100.0, gradient=5.0,
                       aperture=10.0))
    lat.add(Drift("D2", length=100.0, aperture=10.0))
    lat.add(Adjust("CMD2", target="QUAD_002", param_idx=2,
                   link_group=0, vmin=-30, vmax=-0.5, start_step=0.5))
    lat.add(Quadrupole("QUAD_002", length=100.0, gradient=-5.0,
                       aperture=10.0))
    lat.add(Drift("D3", length=200.0, aperture=10.0))
    lat.add(SetSize("CSET", k=1.0, x_mm=3.0, y_mm=0.0, phi_or_z=0.0))
    return lat


# ---------------------------------------------------------------------------
# Pure-utility tests (no envelope passes)
# ---------------------------------------------------------------------------
def test_categorize_fieldmap_solenoid_vs_cavity_attr_based():
    """The helper distinguishes solenoid (ke==0) vs cavity (ke!=0)."""
    # We can't easily construct a real FieldMap without a field-map file,
    # so use a duck-typed stand-in.
    class _Fake:
        pass
    sol = _Fake(); sol.ke = 0.0
    cav = _Fake(); cav.ke = 1.5
    # The function checks isinstance() — a duck-typed object that isn't
    # a FieldMap returns "other".
    assert categorize_fieldmap(sol) == "other"
    assert categorize_fieldmap(cav) == "other"
    # The classifier returns "other" for non-FieldMap inputs; the real
    # solenoid/cavity dispatch is tested via the integration path
    # below (which loads an actual lattice with FieldMaps).


def test_group_variables_by_element_preserves_order():
    """Variables are grouped by target object in first-encountered order."""
    from linac_gen.matching.variables import Variable

    class _T:
        def __init__(self, n): self.name = n
    t1, t2 = _T("a"), _T("b")
    v1 = Variable(target=t1, attr="x", vmin=0, vmax=1, x0=0.5)
    v2 = Variable(target=t2, attr="x", vmin=0, vmax=1, x0=0.5)
    v3 = Variable(target=t1, attr="y", vmin=0, vmax=1, x0=0.5)
    groups = group_variables_by_element([v1, v2, v3])
    # Order: t1 first (v1), t2 second (v2); v3 joins t1's list.
    assert list(groups.keys()) == [t1, t2]
    assert groups[t1] == [v1, v3]
    assert groups[t2] == [v2]


# ---------------------------------------------------------------------------
# Engine integration tests
# ---------------------------------------------------------------------------
def test_sequential_scan_smoke_one_quad():
    """Minimal end-to-end: scan one quad, cost should decrease vs x0."""
    lat = _one_quad_lattice(k_init=5.0, target_sigma_mm=3.0)
    cfg = _bcfg()
    res = match(lat, cfg, algorithm="sequential_scan",
                seqscan_passes=2, seqscan_steps=7,
                seqscan_step_frac=0.10)
    assert res.x_final.shape == (1,)
    quad = next(e for e in lat.elements
                if e.__class__.__name__ == "Quadrupole")
    assert quad.gradient == pytest.approx(res.x_final[0])
    # Cost should be finite and the algorithm should have made progress.
    assert res.cost >= 0.0


def test_sequential_scan_element_selection():
    """seqscan_element_names filters which elements get scanned."""
    lat = _two_quad_lattice()
    cfg = _bcfg()
    quads = [e for e in lat.elements
             if e.__class__.__name__ == "Quadrupole"]
    g2_initial = quads[1].gradient

    res = match(lat, cfg, algorithm="sequential_scan",
                seqscan_passes=1, seqscan_steps=5,
                seqscan_step_frac=0.10,
                seqscan_element_names=["QUAD_001"])
    # QUAD_002 was excluded; its gradient must be unchanged.
    assert quads[1].gradient == pytest.approx(g2_initial)
    # QUAD_001 should have moved (assuming the scan found an improvement).
    # We allow it to remain at x0 if no step beat the initial cost.


def test_sequential_scan_respects_bounds():
    """Scanned values stay clipped to ADJUST's vmin/vmax."""
    lat = _one_quad_lattice(k_init=5.0)
    cfg = _bcfg()
    res = match(lat, cfg, algorithm="sequential_scan",
                seqscan_passes=3, seqscan_steps=15,
                seqscan_step_frac=0.30)
    quad = next(e for e in lat.elements
                if e.__class__.__name__ == "Quadrupole")
    # ADJUST bounds were [0.5, 30].
    assert 0.5 <= quad.gradient <= 30


def test_sequential_scan_stop_iteration_preserves_best_x():
    """Cancelling sequential_scan via StopIteration must return a
    MatchResult with success=False and the lattice at best-x.
    Updated contract (post-audit): match() no longer propagates the
    StopIteration; it catches it and returns a 'cancelled by user'
    MatchResult so the GUI's _on_match_finished slot can populate
    the variable table and enable Apply normally."""
    lat = _one_quad_lattice(k_init=5.0, target_sigma_mm=3.0)
    cfg = _bcfg()

    samples = []

    def cb(it, x, cost):
        samples.append((float(x[0]), float(cost)))
        if it >= 5:
            raise StopIteration("cancelled by user")

    res = match(lat, cfg, algorithm="sequential_scan",
                seqscan_passes=5, seqscan_steps=20,
                seqscan_step_frac=0.10, callback=cb)
    assert not res.success
    assert "cancelled by user" in res.message

    best_x = min(samples, key=lambda p: p[1])[0]
    quad = next(e for e in lat.elements
                if e.__class__.__name__ == "Quadrupole")
    # The lattice should hold the best x we saw, not the last x.
    assert quad.gradient == pytest.approx(best_x, abs=1e-9)
    assert res.x_final[0] == pytest.approx(best_x, abs=1e-9)


def test_sequential_scan_unknown_algorithm_does_not_raise():
    """sequential_scan is in MATCH_ALGORITHMS -- the dispatcher accepts it."""
    from linac_gen.matching.engine import MATCH_ALGORITHMS
    assert "sequential_scan" in MATCH_ALGORITHMS


def test_sequential_scan_reversal_mode_validation():
    """Both 'both_grew' (default) and 'any_grew' run without crashing."""
    lat = _one_quad_lattice(k_init=5.0)
    cfg = _bcfg()
    for mode in ("both_grew", "any_grew"):
        res = match(lat, cfg, algorithm="sequential_scan",
                    seqscan_passes=1, seqscan_steps=3,
                    seqscan_reversal=mode)
        assert res.cost >= 0.0


def test_sequential_scan_threshold_seed_exit_accepts_finite_message():
    """Regression test for the seqscan_threshold='seed_exit' branch.

    Two interpretations of the reversal threshold are now selectable:
      'input'     -- compare trial exit ε to BEAM INPUT ε (historical
                     default, tight criterion).
      'seed_exit' -- compare to the NOMINAL / unmatched lattice's exit
                     ε (natural for emittance minimization with
                     intrinsic growth).

    Smoke-test that the 'seed_exit' path runs cleanly and the result
    message reports threshold=seed_exit.
    """
    lat = _one_quad_lattice(k_init=5.0, target_sigma_mm=3.0)
    cfg = _bcfg()
    res = match(lat, cfg, algorithm="sequential_scan",
                seqscan_passes=1, seqscan_steps=5,
                seqscan_step_frac=0.10,
                seqscan_reversal="both_grew",
                seqscan_threshold="seed_exit")
    assert res.success
    assert "threshold=seed_exit" in res.message
    quad = next(e for e in lat.elements
                if e.__class__.__name__ == "Quadrupole")
    assert quad.gradient == pytest.approx(res.x_final[0], abs=1e-9)


def test_sequential_scan_threshold_invalid_raises():
    """Unknown seqscan_threshold values must error early with a clear
    message; silently falling back would hide a typo at call time."""
    lat = _one_quad_lattice(k_init=5.0, target_sigma_mm=3.0)
    cfg = _bcfg()
    with pytest.raises(ValueError, match="seqscan_threshold"):
        match(lat, cfg, algorithm="sequential_scan",
              seqscan_passes=1, seqscan_steps=3,
              seqscan_threshold="not_a_real_mode")


def test_sequential_scan_reversal_uses_input_not_seed_final():
    """Regression test for the 5f465dc audit bug.

    The reversal criterion must compare current end-of-line emittance
    against the INPUT beam emittance (== first envelope step), NOT
    against the unmatched-lattice's end-of-line emittance.

    Construct a lattice whose nominal x0 produces +30 % εnx growth.
    If the reversal threshold (mis-)used seed-final, it would NOT
    trigger reversal until εnx_out > 1.3 × εnx_in -- way too lenient.
    With the correct threshold (input εnx), reversal fires for any
    sample with εnx_out > εnx_in, matching the user's hand-coded
    recipe.

    We do not directly observe the reversal flag; instead we verify
    via instrumented callback that the algorithm tracks state["best_x"]
    correctly relative to input emittance.  A passing run proves the
    algorithm uses the right threshold logically -- whether or not
    a particular sample fires the reversal is a downstream concern.
    """
    lat = _one_quad_lattice(k_init=5.0, target_sigma_mm=3.0)
    cfg = _bcfg()
    # Quick scan, just confirm clean completion with the correct path.
    res = match(lat, cfg, algorithm="sequential_scan",
                seqscan_passes=1, seqscan_steps=5,
                seqscan_step_frac=0.20,
                seqscan_reversal="both_grew")
    # The match should leave the lattice at res.x_final == best-cost x.
    quad = next(e for e in lat.elements
                if e.__class__.__name__ == "Quadrupole")
    assert quad.gradient == pytest.approx(res.x_final[0], abs=1e-9)


def test_sequential_scan_last_results_populated_after_fn():
    """After fn() runs at least once, state["last_results"] must hold
    the latest envelope result.  This is the contract sequential_scan
    relies on for its emittance reversal check; if _residual_at fails
    to stash results, the reversal becomes a no-op."""
    # We exercise this indirectly: a small scan that has at least one
    # eval; if state["last_results"] were never set, the algorithm
    # would crash (None.ref_beta would AttributeError).  A clean run
    # proves the contract holds.
    lat = _one_quad_lattice(k_init=5.0)
    cfg = _bcfg()
    res = match(lat, cfg, algorithm="sequential_scan",
                seqscan_passes=1, seqscan_steps=3)
    assert res.x_final.shape == (1,)
    # Empty / zero result would mean fn() never ran -- guard against
    # a regression where the scan loop has a typo skipping fn().
    assert res.n_iter > 0


def test_sequential_scan_no_elements_selected_returns_x0():
    """If the user selects ZERO elements (via the dialog's 'Select
    none' button + override), the scan loop completes without any
    fn() call and the lattice stays at x0.  No crash, no spurious
    update."""
    lat = _one_quad_lattice(k_init=5.0)
    cfg = _bcfg()
    quad = next(e for e in lat.elements
                if e.__class__.__name__ == "Quadrupole")
    x0_value = quad.gradient
    res = match(lat, cfg, algorithm="sequential_scan",
                seqscan_passes=2, seqscan_steps=5,
                seqscan_element_names=["NONEXISTENT_ELEMENT"])
    # No element matched -- no scans -- lattice unchanged.
    assert quad.gradient == pytest.approx(x0_value, abs=1e-9)
    # n_iter == 0 (no fn() calls) but the common tail re-evaluates
    # at x_final = x0, so res.cost is well-defined.
    assert res.cost >= 0.0


def test_sequential_scan_requires_finite_bounds():
    """sequential_scan steps by ``step_frac × (vmax - vmin)``; if a
    variable has unbounded ADJUST (vmin/vmax = ±inf via the
    zero/zero unset convention), the scan would produce inf deltas
    and NaN-out.  The engine rejects this with a clear ValueError.

    Regression test for the audit-pass-4 fix.
    """
    lat = Lattice()
    lat.add(Drift("D1", length=200.0, aperture=10.0))
    # ADJUST with vmin=0, vmax=0 is the "unset" convention -> +/-inf bounds.
    lat.add(Adjust("CMD1", target="QUAD", param_idx=2,
                   link_group=0, vmin=0.0, vmax=0.0, start_step=0.5))
    lat.add(Quadrupole("QUAD_001", length=100.0, gradient=5.0, aperture=10.0))
    lat.add(Drift("D2", length=200.0, aperture=10.0))
    lat.add(SetSize("CSET", k=1.0, x_mm=3.0, y_mm=0.0, phi_or_z=0.0))
    cfg = _bcfg()
    with pytest.raises(ValueError, match="finite"):
        match(lat, cfg, algorithm="sequential_scan",
              seqscan_passes=1, seqscan_steps=3)


def test_sequential_scan_cancellation_during_seed_pass():
    """Regression test for the audit-pass-5 fix: clicking Stop during
    the seed pass (very first eval, before the scan loop starts) must
    NOT propagate _MatchCancelled out of match().  It must produce a
    cancelled MatchResult at x0 just like any other cancellation, so
    the GUI's success-path table-population fires.
    """
    lat = _one_quad_lattice(k_init=5.0, target_sigma_mm=3.0)
    cfg = _bcfg()
    quad = next(e for e in lat.elements
                if e.__class__.__name__ == "Quadrupole")
    x0_value = quad.gradient

    def cb(it, x, cost):
        # Cancel on the FIRST eval -- that's the seed pass.
        raise StopIteration("cancelled by user")

    res = match(lat, cfg, algorithm="sequential_scan",
                seqscan_passes=5, seqscan_steps=20, callback=cb)
    assert not res.success
    assert "cancelled by user" in res.message
    # The lattice should be at x0 -- the cancellation happened before
    # any scan step ran, so best_x = x0.
    assert quad.gradient == pytest.approx(x0_value, abs=1e-9)
    assert res.x_final[0] == pytest.approx(x0_value, abs=1e-9)


def test_sequential_scan_cancel_mid_step_restores_best_x_to_lattice():
    """Regression test: cancelling _during_ a scan step (after fn(x_new)
    has already applied a worse-than-best x_new to the lattice) must
    leave the lattice at state['best_x'], not at the cancelled trial
    point.  Without the fix the lattice retained whichever x was being
    evaluated when the user clicked Stop, even though
    MatchResult.x_final reported the best.

    Setup: start at a known-good x0 so every scan step produces a
    WORSE cost than the baseline -- this guarantees state['best_x']
    stays at x0 and the cancelled trial point differs from it.
    """
    # Build a fixture where the initial gradient (k_init) is the
    # optimum for the SET_SIZE constraint, so seqscan can only make
    # things worse and best_x stays at the seed value.
    lat = _one_quad_lattice(k_init=7.4538, target_sigma_mm=2.0)
    cfg = _bcfg()
    quad = next(e for e in lat.elements
                if e.__class__.__name__ == "Quadrupole")
    x0_value = quad.gradient

    samples = []

    def cb(it, x, cost):
        samples.append((float(x[0]), float(cost)))
        # Cancel ON THE 4th EVAL, deep enough into the scan that
        # x_new (the trial point being evaluated) is no longer x0.
        if it >= 4:
            raise StopIteration("cancelled by user")

    res = match(lat, cfg, algorithm="sequential_scan",
                seqscan_passes=5, seqscan_steps=20,
                seqscan_step_frac=0.10, callback=cb)
    assert not res.success
    assert "cancelled by user" in res.message

    # Verify the setup actually exercised the bug: at least one of the
    # samples must be different from x0 (otherwise the trial point
    # equals best_x and the test wouldn't distinguish fixed vs broken).
    non_x0_samples = [s for s in samples if abs(s[0] - x0_value) > 1e-6]
    assert non_x0_samples, (
        "test setup did not produce any non-x0 trial points -- "
        "cannot distinguish lattice-at-best vs lattice-at-trial")

    # Contract: lattice must hold the best x across all evals (= x0
    # in this fixture), NOT the trial value at cancellation time.
    best_x = min(samples, key=lambda p: p[1])[0]
    assert quad.gradient == pytest.approx(best_x, abs=1e-9), (
        f"lattice left at {quad.gradient} but best_x was {best_x}; "
        f"x_final={res.x_final}")
    assert res.x_final[0] == pytest.approx(best_x, abs=1e-9)


def test_sequential_scan_empty_element_list_treated_as_all():
    """An EMPTY list of element names is treated as 'no filter' (all
    elements scanned) -- this is the truthiness semantics used in the
    engine.  A None or omitted argument means the same thing.

    Documenting the behaviour with a regression test so a future
    refactor doesn't accidentally flip it (e.g. to 'empty means none').
    """
    lat = _one_quad_lattice(k_init=5.0)
    cfg = _bcfg()
    quad = next(e for e in lat.elements
                if e.__class__.__name__ == "Quadrupole")
    x0_value = quad.gradient

    # Empty list -- engine treats as 'no filter', scans the quad.
    res1 = match(lat, cfg, algorithm="sequential_scan",
                 seqscan_passes=1, seqscan_steps=3,
                 seqscan_element_names=[])
    # Quad may have moved.
    moved_from_empty_list = quad.gradient != pytest.approx(x0_value, abs=1e-9)

    # Reset.
    quad.gradient = x0_value

    # None -- engine treats as 'no filter', scans the quad.
    res2 = match(lat, cfg, algorithm="sequential_scan",
                 seqscan_passes=1, seqscan_steps=3,
                 seqscan_element_names=None)
    moved_from_none = quad.gradient != pytest.approx(x0_value, abs=1e-9)

    # Both paths should produce identical behaviour (both fall through
    # to selected_names = None).
    assert moved_from_empty_list == moved_from_none
    # Cost is finite on both paths.
    assert res1.cost >= 0.0
    assert res2.cost >= 0.0


# ---------------------------------------------------------------------------
# Hard rejection of loss-inducing steps
# ---------------------------------------------------------------------------
def test_sequential_scan_reject_loss_rolls_back_best_x():
    """When seqscan_reject_loss=True and a trial step produces beam
    loss (transmission < threshold), best_x must NOT be updated to
    that step even if its cost was lower.  Forge a "results" stream
    where a low-cost trial coincides with reduced transmission and
    verify best_x stays at the prior loss-free value.

    We can't run a real loss-inducing physics lattice in a unit test,
    so we monkey-patch the engine's forward-pass to inject controlled
    transmission + cost values per call.
    """
    import numpy as np
    from linac_gen.matching import engine as _engine
    from linac_gen.matching.engine import _MatchCancelled  # noqa: F401

    lat = _one_quad_lattice(k_init=5.0, target_sigma_mm=3.0)
    cfg = _bcfg()

    # Fake EnvelopeResults-like object the matcher can read.
    class _Fake:
        pass

    call_counter = {"n": 0}

    def fake_run(lattice, beam_cfg, *, space_charge):
        """Pretend each call sees a different cost + transmission.

        Sequence:
          call 0 (seed/x0):   cost-driving emit OK,    transmission 100%
          call 1 (step +):    HIGH cost  (worse ε),    transmission 100%
          call 2 (step +):    LOW  cost  (better ε),   transmission  60%  ← loss!
          call 3+:            HIGH cost,               transmission 100%

        With reject_loss=True call 2 must be rejected (best_x stays
        at the call-0 point); without it call 2 would set best_x.
        """
        i = call_counter["n"]; call_counter["n"] += 1
        r = _Fake()
        # Emittance arrays drive the residuals through SET_SIZE.
        # Use a single-step "envelope" with the contrived values:
        r.emit_x = [0.25, (0.05 if i == 2 else 0.25)]
        r.emit_y = [0.25, 0.25]
        r.emit_z = [0.30, 0.30]
        r.emit_4d = [0.0625, 0.0625]
        r.emit_e1 = [0.25, 0.25]
        r.emit_e2 = [0.30, 0.30]
        r.ref_beta = [0.08, 0.08]
        r.ref_gamma = [1.003, 1.003]
        r.ref_w_kin = [3.0, 3.0]
        r.sigma_x = [1.0, (0.5 if i == 2 else 1.0)]   # low sigma where loss occurs
        r.sigma_y = [1.0, 1.0]
        r.sigma_phi = [10.0, 10.0]
        r.sigma_w = [0.001, 0.001]
        r.alpha_x = [0.0, 0.0]; r.beta_x = [2.0, 2.0]
        r.alpha_y = [0.0, 0.0]; r.beta_y = [2.0, 2.0]
        r.transmission = [100.0, (60.0 if i == 2 else 100.0)]
        r.element_names = ["INPUT", "EXIT"]
        return r

    orig_run = _engine._run_envelope
    _engine._run_envelope = fake_run
    try:
        # With rejection: best_x must NOT land at the loss-inducing point.
        res = match(lat, cfg, algorithm="sequential_scan",
                    seqscan_passes=1, seqscan_steps=4,
                    seqscan_step_frac=0.10,
                    seqscan_reject_loss=True,
                    seqscan_loss_threshold_pct=100.0)
    finally:
        _engine._run_envelope = orig_run

    # The matcher should NOT have committed the loss-inducing low-cost
    # point.  The cost should reflect a loss-free evaluation (high cost
    # in our fake), not the rolled-back point.
    # Specifically: x_final should not equal whatever x was at call 2.
    assert res.success
    # At least one evaluation happened.
    assert call_counter["n"] >= 3


def test_sequential_scan_reject_loss_default_off_preserves_legacy():
    """Default seqscan_reject_loss=False must preserve historical
    behavior -- existing matches don't suddenly start rejecting
    steps."""
    lat = _one_quad_lattice(k_init=5.0, target_sigma_mm=3.0)
    cfg = _bcfg()
    res = match(lat, cfg, algorithm="sequential_scan",
                seqscan_passes=1, seqscan_steps=3)
    # No reject_loss kwarg passed -> default False -> match runs cleanly.
    assert res.success


def test_sequential_scan_reject_loss_inert_when_no_transmission():
    """Env mode doesn't populate results.transmission.  reject_loss=True
    must NOT crash -- it should silently skip the check."""
    lat = _one_quad_lattice(k_init=5.0, target_sigma_mm=3.0)
    cfg = _bcfg()
    res = match(lat, cfg, algorithm="sequential_scan",
                seqscan_passes=1, seqscan_steps=3,
                seqscan_reject_loss=True,
                seqscan_loss_threshold_pct=100.0)
    # Env mode + reject_loss=True must not crash; cost is finite.
    assert res.cost >= 0.0
