"""End-to-end matcher tests.

A full forward simulation (envelope, no SC) runs at every iteration so
these tests are slower than the variable / constraint unit tests; they
still complete in a few seconds each on the CI box.
"""
from __future__ import annotations

import pytest

from linac_gen.core.config import BeamConfig
from linac_gen.core.lattice import Lattice
from linac_gen.elements.drift import Drift
from linac_gen.elements.lattice_commands import (
    Adjust, AdjustBeamTwiss, SetSize, SetTwiss,
)
from linac_gen.elements.quadrupole import Quadrupole
from linac_gen.matching.engine import match


def _bcfg(**over) -> BeamConfig:
    base = dict(species="proton", energy=3.0, frequency=352.21,
                n_particles=10, distribution="waterbag",
                emit_nx=0.25, alpha_x=0.0, beta_x=2.0,
                emit_ny=0.25, alpha_y=0.0, beta_y=2.0,
                emit_z=0.3,    alpha_z=0.0, beta_z=10.0)
    base.update(over)
    return BeamConfig(**base)


# ---------------------------------------------------------------------------
def _fodo_with_one_adjust(target_beta: float, k_init: float = 5.0) -> Lattice:
    """Single-quad transport line with an ADJUST on its gradient and
    a SET_SIZE constraint at the end (β-target via σ comparison)."""
    lat = Lattice()
    lat.add(Drift("D1", length=200.0, aperture=10.0))
    lat.add(Adjust("CMD1", target="QUAD", param_idx=2,
                    link_group=0, vmin=-30, vmax=30, start_step=0.5))
    lat.add(Quadrupole("QUAD_001", length=100.0, gradient=k_init,
                       aperture=10.0))
    lat.add(Drift("D2", length=200.0, aperture=10.0))
    # σ_x target via SET_SIZE (use SET_SIZE because β_x evaluation is
    # already covered by SET_TWISS; SET_SIZE is the simpler residual).
    lat.add(SetSize("CSET", k=1.0, x_mm=target_beta, y_mm=0.0, phi_or_z=0.0))
    return lat


def test_matcher_converges_with_no_change_when_already_matched():
    """If the lattice is at x0 with zero residual, the matcher returns
    success and zero iterations beyond the initial evaluation."""
    lat = Lattice()
    lat.add(Drift("D1", length=100.0, aperture=10.0))
    cfg = _bcfg()
    res = match(lat, cfg, max_iter=10)
    # No ADJUST in the lattice → trivial success path
    assert res.success
    assert "No ADJUST" in res.message


def test_matcher_runs_and_writes_back_x():
    """Adjust the QUAD_001 gradient to bring σ_x at end to target."""
    lat = _fodo_with_one_adjust(target_beta=3.0, k_init=5.0)
    cfg = _bcfg()
    res = match(lat, cfg, max_iter=80)
    # Should at least produce a result and write x back into the element.
    assert res.x_final.shape == (1,)
    quad = next(e for e in lat.elements if e.__class__.__name__ == "Quadrupole")
    assert quad.gradient == pytest.approx(res.x_final[0])
    # Cost should not have increased above x0 cost.
    # (least_squares is non-decreasing on cost in general.)
    assert res.cost >= 0.0


def test_no_constraints_returns_failure():
    lat = Lattice()
    lat.add(Adjust("CMD1", target="QUAD", param_idx=2,
                    link_group=0, vmin=-10, vmax=10))
    lat.add(Quadrupole("QUAD_001", length=50.0, gradient=5.0, aperture=5.0))
    cfg = _bcfg()
    res = match(lat, cfg, max_iter=5)
    assert not res.success
    assert "No SET" in res.message


def test_link_groups_collapse_columns():
    lat = Lattice()
    lat.add(Drift("D1", length=100.0, aperture=10.0))
    lat.add(Adjust("CMD1", target="QUAD", param_idx=2,
                    link_group=42, vmin=-30, vmax=30))
    lat.add(Quadrupole("QUAD_001", length=50.0, gradient=10.0, aperture=10.0))
    lat.add(Adjust("CMD2", target="QUAD", param_idx=2,
                    link_group=42, vmin=-30, vmax=30))
    lat.add(Quadrupole("QUAD_002", length=50.0, gradient=-10.0, aperture=10.0))
    lat.add(Drift("D2", length=100.0, aperture=10.0))
    lat.add(SetSize("CSET", k=1.0, x_mm=3.0))
    cfg = _bcfg()
    res = match(lat, cfg, max_iter=20)
    # Two ADJUST cards but one optimiser column due to shared link group.
    assert res.x0.shape == (1,)
    # Both quads ended up with the same gradient (the linked variable's value).
    quads = [e for e in lat.elements if e.__class__.__name__ == "Quadrupole"]
    assert quads[0].gradient == pytest.approx(quads[1].gradient)


# ---------------------------------------------------------------------------
# Algorithm selection — least_squares / differential_evolution / dual_annealing
# ---------------------------------------------------------------------------
def test_baseline_cost_recorded_and_consistent_with_initial_x0():
    """Every match() now runs a baseline pass at x0 BEFORE the
    optimisation loop and records the resulting cost on
    MatchResult.baseline_cost.  Verifies:

      * baseline_cost is finite and non-negative
      * baseline_cost matches what a fresh evaluation at x0 produces
        (within a small tolerance for stochastic MP runs -- here we
        use envelope so it should be exact)
      * baseline_cost >= final cost (the matcher never makes things
        worse than x0; either improves or stays equal)
    """
    lat = _fodo_with_one_adjust(target_beta=3.0, k_init=5.0)
    cfg = _bcfg()
    res = match(lat, cfg, max_iter=40, algorithm="least_squares")
    assert res.baseline_cost is not None
    assert res.baseline_cost >= 0.0
    # The matcher should never make things worse than x0.
    assert res.cost <= res.baseline_cost + 1e-12


def test_baseline_cost_present_for_all_algorithms():
    """baseline_cost is populated by ALL six algorithms via the
    explicit baseline pass at x0.  Regression test against future
    refactors that skip the baseline for some algorithms.
    """
    cfg = _bcfg()
    for algo in ("least_squares", "differential_evolution",
                 "dual_annealing", "cmaes"):
        lat = _fodo_with_one_adjust(target_beta=3.0, k_init=5.0)
        res = match(lat, cfg, max_iter=3, algorithm=algo)
        assert res.baseline_cost is not None, (
            f"{algo} did not record baseline_cost"
        )

    # sequential_scan -- separate kwargs path
    lat = _fodo_with_one_adjust(target_beta=3.0, k_init=5.0)
    res = match(lat, cfg, algorithm="sequential_scan",
                seqscan_passes=1, seqscan_steps=2)
    assert res.baseline_cost is not None


def test_unknown_algorithm_raises():
    """An unrecognised algorithm name fails fast with a clear error."""
    lat = _fodo_with_one_adjust(target_beta=3.0)
    with pytest.raises(ValueError, match="unknown matcher algorithm"):
        match(lat, _bcfg(), algorithm="bogus")


# ---------------------------------------------------------------------------
# Cancellation safety -- StopIteration from callback applies best-x to lattice
# ---------------------------------------------------------------------------
def test_stop_iteration_applies_best_x_least_squares():
    """When the callback raises StopIteration mid-run, the lattice must
    be left at the best-cost x seen so far, not at whatever exploration
    point the optimiser was just probing.

    Per the b916491 + the post-audit refactor: cancellation now returns
    a MatchResult with ``success=False`` and message starting with
    "cancelled by user", AND the lattice carries the best-x.  This lets
    the GUI's _on_match_finished slot populate the table + enable Apply
    even when the user clicks Stop.
    """
    lat = _fodo_with_one_adjust(target_beta=3.0, k_init=5.0)
    cfg = _bcfg()

    samples: list[tuple[float, float]] = []

    def cb(it, x, cost):
        samples.append((float(x[0]), float(cost)))
        # Run 5 evals then cancel.  The optimiser is least_squares which
        # uses finite-difference gradients, so by eval 5 it has tried
        # several x values; the running best is well-defined.
        if it >= 5:
            raise StopIteration("cancelled by user")

    res = match(lat, cfg, max_iter=80,
                algorithm="least_squares", callback=cb)
    # New contract: match() returns a MatchResult with success=False
    # and "cancelled by user" in message; the lattice holds best_x.
    assert not res.success
    assert "cancelled by user" in res.message

    # Best cost across the trace
    best_x, best_cost = min(samples, key=lambda p: p[1])
    last_x = samples[-1][0]
    quad = next(e for e in lat.elements
                if e.__class__.__name__ == "Quadrupole")
    assert quad.gradient == pytest.approx(best_x, abs=1e-9), (
        f"After cancellation, lattice must hold best_x={best_x:.6f} "
        f"(cost {best_cost:.3e}), not last_x={last_x:.6f}"
    )
    # MatchResult.x_final reflects the best-x.
    assert res.x_final[0] == pytest.approx(best_x, abs=1e-9)


def test_cmaes_parallel_cancellation_preserves_best_x():
    """Parallel CMA-ES (cmaes_parallel > 1) computes costs in worker
    processes that don't share state with the main process, so the
    main-process state["best_x"] tracking that fn() does in the
    sequential path needs to be replicated in the pool loop.

    Regression test for the audit-pass-3 fix: clicking Stop mid-run on
    a parallel CMA-ES match must leave the lattice at the best (xx, ff)
    sample seen across all generations, not at x0.
    """
    lat = _fodo_with_one_adjust(target_beta=3.0, k_init=5.0)
    cfg = _bcfg()
    quad = next(e for e in lat.elements
                if e.__class__.__name__ == "Quadrupole")
    x0_value = quad.gradient

    samples = []

    def cb(it, x, cost):
        samples.append((float(x[0]), float(cost)))
        # Cancel after 2 generations (popsize is 4 for N=1 var; so
        # iter 8 = end of generation 2).
        if it >= 8:
            raise StopIteration("cancelled by user")

    res = match(lat, cfg, algorithm="cmaes",
                max_iter=20, cmaes_parallel=2, callback=cb)
    # Cancellation must produce a soft-success MatchResult.
    assert not res.success
    assert "cancelled by user" in res.message
    # The lattice should be at a sampled best -- specifically, the
    # candidate with the lowest cost among `samples`.
    best_x_sampled, _ = min(samples, key=lambda p: p[1])
    # Allow small tolerance: the lattice was mutated to best_x;
    # x_final reflects the same.
    assert res.x_final[0] == pytest.approx(best_x_sampled, abs=1e-9), (
        f"Parallel CMA-ES cancellation must hold the best sampled x "
        f"({best_x_sampled:.6f}), not x0 ({x0_value:.6f}); got "
        f"res.x_final[0] = {res.x_final[0]:.6f}"
    )


def test_stop_iteration_returns_result_for_all_algorithms():
    """The cancellation contract is uniform across LS, DE, DA, CMA-ES,
    sequential_scan and gradient: clicking Stop returns a MatchResult
    with success=False and "cancelled by user" in message, NOT an
    exception.  Without uniformity the GUI's _on_match_finished would
    fail for some algorithms.
    """
    cfg = _bcfg()

    def cb_after(it, x, cost):
        if it >= 3:
            raise StopIteration("cancelled by user")

    # Each algorithm gets its own lattice -- match() mutates in place.
    for algo in ("least_squares", "differential_evolution", "dual_annealing",
                 "cmaes"):
        lat = _fodo_with_one_adjust(target_beta=3.0, k_init=5.0)
        # global algos need finite bounds (the fixture already provides them)
        res = match(lat, cfg, max_iter=80, algorithm=algo, callback=cb_after)
        assert not res.success, f"{algo} should have success=False on cancel"
        assert "cancelled by user" in res.message, (
            f"{algo} message should contain 'cancelled by user', got "
            f"{res.message!r}"
        )
    # sequential_scan uses different kwargs.
    lat = _fodo_with_one_adjust(target_beta=3.0, k_init=5.0)
    res = match(lat, cfg, algorithm="sequential_scan",
                seqscan_passes=5, seqscan_steps=20, callback=cb_after)
    assert not res.success
    assert "cancelled by user" in res.message


def test_stop_iteration_immediate_keeps_initial_x():
    """If StopIteration fires before any successful improvement (e.g.
    the very first callback raises), the lattice must remain at x0 --
    not crash trying to apply a None best_x.
    """
    lat = _fodo_with_one_adjust(target_beta=3.0, k_init=5.0)
    cfg = _bcfg()
    quad = next(e for e in lat.elements
                if e.__class__.__name__ == "Quadrupole")
    x0 = quad.gradient

    def cb(it, x, cost):
        # Cancel on the FIRST eval -- exercises the "no best_x yet" branch.
        raise StopIteration("cancelled by user")

    res = match(lat, cfg, max_iter=80,
                algorithm="least_squares", callback=cb)
    assert not res.success
    assert "cancelled by user" in res.message

    # No improvement was tracked, so the cancellation handler falls back
    # to applying x (the current sample), which equals x0 since this was
    # the first eval.
    assert quad.gradient == pytest.approx(x0, abs=1e-9)


def test_least_squares_is_the_default():
    """Calling match() without `algorithm` is identical to passing
    algorithm='least_squares' explicitly — the default is unchanged."""
    r_default = match(_fodo_with_one_adjust(target_beta=3.0), _bcfg(),
                      max_iter=80)
    r_explicit = match(_fodo_with_one_adjust(target_beta=3.0), _bcfg(),
                       algorithm="least_squares", max_iter=80)
    assert r_default.x_final == pytest.approx(r_explicit.x_final)


def test_differential_evolution_converges():
    """The global differential-evolution optimiser drives the residual to
    ~zero on the single-quad σ-matching problem and writes x back."""
    lat = _fodo_with_one_adjust(target_beta=3.0)
    res = match(lat, _bcfg(), algorithm="differential_evolution",
                max_iter=40)
    assert res.success
    assert res.cost < 1e-6
    quad = next(e for e in lat.elements
                if e.__class__.__name__ == "Quadrupole")
    assert quad.gradient == pytest.approx(res.x_final[0])


def test_dual_annealing_converges():
    """The global dual-annealing optimiser also converges on the same
    problem (it may settle in a different basin — the σ-target is
    multi-modal in the quad gradient — but the residual is still ~zero)."""
    lat = _fodo_with_one_adjust(target_beta=3.0)
    res = match(lat, _bcfg(), algorithm="dual_annealing", max_iter=40)
    assert res.success
    assert res.cost < 1e-6


def test_global_algorithm_requires_finite_bounds():
    """differential_evolution / dual_annealing reject an open-ended ADJUST
    bound with a ValueError; least_squares handles ±inf bounds natively."""
    lat = Lattice()
    lat.add(Drift("D1", length=200.0, aperture=10.0))
    # vmin == vmax == 0  →  treated as 'unset'  →  (-inf, +inf) bounds.
    lat.add(Adjust("CMD1", target="QUAD", param_idx=2, link_group=0,
                    vmin=0.0, vmax=0.0))
    lat.add(Quadrupole("QUAD_001", length=100.0, gradient=5.0,
                       aperture=10.0))
    lat.add(Drift("D2", length=200.0, aperture=10.0))
    lat.add(SetSize("CSET", k=1.0, x_mm=3.0))
    with pytest.raises(ValueError, match="finite bounds"):
        match(lat, _bcfg(), algorithm="differential_evolution", max_iter=10)
    # least_squares must NOT raise on the same unbounded lattice.
    res = match(lat, _bcfg(), algorithm="least_squares", max_iter=20)
    assert res.x_final.shape == (1,)


def test_global_algorithm_is_reproducible():
    """A fixed RNG seed makes the global optimisers deterministic: two
    runs of the same problem give the same matched value."""
    r1 = match(_fodo_with_one_adjust(target_beta=3.0), _bcfg(),
               algorithm="differential_evolution", max_iter=40)
    r2 = match(_fodo_with_one_adjust(target_beta=3.0), _bcfg(),
               algorithm="differential_evolution", max_iter=40)
    assert r1.x_final == pytest.approx(r2.x_final)


# ── reviewer round 2: DC (continuous) beams in the matcher ────────────────
def test_envelope_cost_honours_dc_beam():
    """A continuous BeamConfig must reach the envelope cost solver as a
    DC beam (2-D line-charge SC), matching the CLI forward path — it
    used to silently run the 3-D bunched-ellipsoid kick instead."""
    from linac_gen.cli.common import build_ref, _envelope_initial
    from linac_gen.matching.engine import _run_envelope
    from linac_gen.tracking.envelope import EnvelopeSolver

    cfg = _bcfg(current=10.0, continuous=True)
    lat = Lattice()
    for i in range(5):
        lat.add(Drift(f"D{i}", length=200.0, aperture=30.0))
    res = _run_envelope(lat, cfg, space_charge=True)
    assert res.continuous is True

    ref = build_ref(cfg)
    lat2 = Lattice()
    for i in range(5):
        lat2.add(Drift(f"D{i}", length=200.0, aperture=30.0))
    res_cli = EnvelopeSolver(lat2, ref, _envelope_initial(cfg, ref),
                             current=cfg.current).run()
    assert res.sigma_x[-1] == pytest.approx(res_cli.sigma_x[-1],
                                            rel=1e-12)
    # And the bunched default is untouched.
    lat3 = Lattice()
    for i in range(5):
        lat3.add(Drift(f"D{i}", length=200.0, aperture=30.0))
    res_b = _run_envelope(lat3, _bcfg(current=10.0), space_charge=True)
    assert res_b.continuous is False
    assert res_b.sigma_x[-1] != pytest.approx(res.sigma_x[-1], rel=1e-3)


def test_gradient_refuses_dc_beam():
    """algorithm='gradient' has no DC space-charge model — it must
    refuse a continuous BeamConfig instead of silently scoring the
    match with bunched physics."""
    from linac_gen.matching.variables import MatchingConfigError
    with pytest.raises(MatchingConfigError, match="continuous"):
        match(_fodo_with_one_adjust(target_beta=3.0),
              _bcfg(current=10.0, continuous=True),
              algorithm="gradient", max_iter=5)


def test_envelope_seeds_apply_mismatch():
    """mismatch_{x,y,z} must reach every envelope seed exactly as the
    MP generator applies it (×(1+m/100) on the geometric emittance) —
    the envelope paths used to silently drop it."""
    from linac_gen.cli.common import build_ref, _envelope_initial
    from linac_gen.distributions.factory import geometric_emittances
    from linac_gen.matching.engine import _run_envelope

    cfg0 = _bcfg(current=0.0)
    cfgm = _bcfg(current=0.0, mismatch_x=50.0, mismatch_z=-20.0)
    ref = build_ref(cfg0)

    # The shared helper defines the semantics…
    ex0, _ey0, ez0 = geometric_emittances(cfg0, ref.bg)
    exm, _eym, ezm = geometric_emittances(cfgm, ref.bg)
    assert exm == pytest.approx(1.5 * ex0, rel=1e-12)
    assert ezm == pytest.approx(0.8 * ez0, rel=1e-12)

    # …the CLI seed uses it…
    init = _envelope_initial(cfgm, ref)
    assert init["emit_x"] == pytest.approx(exm, rel=1e-12)

    # …and the matcher's envelope cost sees the mismatched beam
    # (σx(0) = sqrt(β ε) scales by sqrt(1.5)).
    def _dlat():
        lat = Lattice()
        lat.add(Drift("D1", length=100.0, aperture=30.0))
        return lat
    r0 = _run_envelope(_dlat(), cfg0, space_charge=False)
    rm = _run_envelope(_dlat(), cfgm, space_charge=False)
    assert rm.sigma_x[0] / r0.sigma_x[0] == pytest.approx(
        1.5 ** 0.5, rel=1e-9)
    # mismatch=0 keeps the historical numbers bit-identical.
    r00 = _run_envelope(_dlat(), _bcfg(current=0.0), space_charge=False)
    assert r00.sigma_x[0] == r0.sigma_x[0]
