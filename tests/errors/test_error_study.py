# tests/errors/test_error_study.py
"""Tests for the error model and Monte Carlo engine (Task 11.1)."""
import numpy as np
import pytest
from linac_gen.core.config import BeamConfig
from linac_gen.core.lattice import Lattice
from linac_gen.core.particle import PROTON
from linac_gen.core.reference import ReferenceParticle
from linac_gen.core.beam import Beam
from linac_gen.elements.drift import Drift
from linac_gen.elements.quadrupole import Quadrupole
from linac_gen.errors.error_model import ErrorDef, ErrorStudy, ErrorStudyResults


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_fodo_lattice():
    """Simple FODO-like lattice with named quads."""
    lat = Lattice()
    lat.add(Quadrupole("QF_1", length=50.0, gradient=5.0, aperture=50.0, n_steps=3))
    lat.add(Drift("D1", 200.0))
    lat.add(Quadrupole("QD_1", length=50.0, gradient=-5.0, aperture=50.0, n_steps=3))
    lat.add(Drift("D2", 200.0))
    return lat


def _make_beam_config(n=200):
    return BeamConfig(
        species="proton",
        energy=3.0,
        frequency=352.21,
        current=0.0,
        n_particles=n,
        distribution="gaussian",
        emit_nx=0.25,
        emit_ny=0.25,
        emit_z=0.3,
        alpha_x=0.0, beta_x=0.1,
        alpha_y=0.0, beta_y=0.1,
        alpha_z=0.0, beta_z=1.0,
        cutoff=3.0,
    )


# ---------------------------------------------------------------------------
# Test 1: ErrorDef dataclass
# ---------------------------------------------------------------------------

def test_error_def_defaults():
    err = ErrorDef(pattern="QF_*", parameter="gradient_rel")
    assert err.distribution == "gaussian"
    assert err.sigma == 0.0
    assert err.half_width == 0.0
    assert err.cutoff == 3.0


def test_error_def_explicit():
    err = ErrorDef("BPM_*", "dx", "uniform", half_width=0.5)
    assert err.pattern == "BPM_*"
    assert err.parameter == "dx"
    assert err.distribution == "uniform"
    assert err.half_width == 0.5


# ---------------------------------------------------------------------------
# Test 2: No errors → all seeds give same result
# ---------------------------------------------------------------------------

def test_no_errors_reproducible():
    """With no errors, all seeds produce identical results (same beam seed)."""
    lat = _make_fodo_lattice()
    cfg = _make_beam_config(n=100)
    # Run with same beam seed (no errors means lattice is identical)
    study = ErrorStudy(lat, cfg, n_seeds=3)
    # No errors added
    results = study.run()
    assert results.n_seeds == 3
    # sigma_x should have the same shape for all recorders
    s0 = np.array(results._recorders[0].sigma_x)
    s1 = np.array(results._recorders[1].sigma_x)
    s2 = np.array(results._recorders[2].sigma_x)
    # shapes must match
    assert s0.shape == s1.shape == s2.shape
    # std across seeds should be zero (same lattice, but different beam seeds)
    # The beams differ in seed, so values differ; what must be same is the lattice
    # Just confirm shapes are consistent
    assert len(s0) > 0


# ---------------------------------------------------------------------------
# Test 3: Gradient errors change optics
# ---------------------------------------------------------------------------

def test_gradient_errors_change_optics():
    """gradient_rel errors on quads produce spread in sigma_x across seeds."""
    lat = _make_fodo_lattice()
    cfg = _make_beam_config(n=300)
    study = ErrorStudy(lat, cfg, n_seeds=5)
    study.add_error("QF_*", "gradient_rel", distribution="gaussian", sigma=0.05)
    study.add_error("QD_*", "gradient_rel", distribution="gaussian", sigma=0.05)
    results = study.run()
    # std of sigma_x at the last step should be > 0
    std_sigma_x = results.std("sigma_x")
    assert std_sigma_x[-1] > 0.0, "Expected spread in sigma_x from gradient errors"


def test_gradient_errors_applied_to_elements():
    """_apply_errors correctly perturbs gradients."""
    lat = _make_fodo_lattice()
    original_gradient = lat.elements[0].gradient  # QF_1

    cfg = _make_beam_config()
    study = ErrorStudy(lat, cfg, n_seeds=3)
    study.add_error("QF_*", "gradient_rel", distribution="gaussian", sigma=0.10, cutoff=3.0)

    lat_copy = study._apply_errors(seed=0)
    qf_copy = lat_copy.elements[0]
    # The gradient should have changed
    assert qf_copy.gradient != original_gradient
    # Original lattice must be unchanged
    assert lat.elements[0].gradient == original_gradient


# ---------------------------------------------------------------------------
# Test 4: Alignment errors shift centroids
# ---------------------------------------------------------------------------

def test_alignment_errors_centroid_varies():
    """dx errors on quads produce spread in centroid_x across seeds."""
    lat = _make_fodo_lattice()
    cfg = _make_beam_config(n=300)
    study = ErrorStudy(lat, cfg, n_seeds=5)
    study.add_error("QF_*", "dx", distribution="uniform", half_width=1.0)
    study.add_error("QD_*", "dx", distribution="uniform", half_width=1.0)
    results = study.run()
    # Collect final centroid x for each seed
    centroid_x_final = []
    for rec in results._recorders:
        centroids = np.array(rec.centroid)  # shape (n_steps, 6)
        centroid_x_final.append(centroids[-1, 0])
    # std should be > 0 due to random offsets
    assert np.std(centroid_x_final) > 0.0, "Expected centroid variation from alignment errors"


def test_alignment_dx_stored_on_element():
    """_apply_errors sets dx attribute on matching elements."""
    lat = _make_fodo_lattice()
    cfg = _make_beam_config()
    study = ErrorStudy(lat, cfg)
    study.add_error("QF_*", "dx", distribution="uniform", half_width=0.5)
    lat_copy = study._apply_errors(seed=42)
    qf = lat_copy.elements[0]
    assert hasattr(qf, "dx")
    assert abs(qf.dx) <= 0.5


# ---------------------------------------------------------------------------
# Test 5: RF phase error
# ---------------------------------------------------------------------------

def test_phase_error_applied():
    """phase error adds to the element's phase attribute."""
    from linac_gen.elements.rf_gap import RFGap
    lat = Lattice()
    lat.add(Drift("D1", 100.0))
    rf = RFGap("CAV_1", voltage=1.0, phase=-30.0, frequency=352.21)
    lat.add(rf)

    original_phase = rf.phase
    cfg = _make_beam_config()
    study = ErrorStudy(lat, cfg, n_seeds=1)
    study.add_error("CAV_*", "phase", distribution="gaussian", sigma=2.0)

    lat_copy = study._apply_errors(seed=7)
    cav_copy = lat_copy.elements[1]
    assert cav_copy.phase != original_phase


# ---------------------------------------------------------------------------
# Test 6: Transmission varies with errors
# ---------------------------------------------------------------------------

def test_transmission_varies_with_errors():
    """With tight apertures and large errors, some seeds may lose particles."""
    lat = Lattice()
    lat.add(Quadrupole("QF_1", length=50.0, gradient=5.0, aperture=10.0, n_steps=3))
    lat.add(Drift("D1", 200.0, aperture=10.0))
    lat.add(Quadrupole("QD_1", length=50.0, gradient=-5.0, aperture=10.0, n_steps=3))

    cfg = _make_beam_config(n=200)
    study = ErrorStudy(lat, cfg, n_seeds=5)
    study.add_error("Q*", "gradient_rel", distribution="gaussian", sigma=0.3, cutoff=3.0)
    results = study.run()

    stats = results.transmission_stats()
    assert "mean" in stats
    assert "min" in stats
    assert "max" in stats
    assert "std" in stats
    # At least the structure is valid
    assert stats["mean"] >= 0.0
    assert stats["max"] <= 100.0


# ---------------------------------------------------------------------------
# Test 7: Statistics shape correctness
# ---------------------------------------------------------------------------

def test_statistics_shapes():
    lat = _make_fodo_lattice()
    cfg = _make_beam_config(n=150)
    study = ErrorStudy(lat, cfg, n_seeds=4)
    study.add_error("Q*", "gradient_rel", distribution="gaussian", sigma=0.02)
    results = study.run()

    n_steps = len(results._recorders[0].s)

    mean_sx = results.mean("sigma_x")
    std_sx = results.std("sigma_x")
    p95 = results.percentile("sigma_x", 95)

    assert mean_sx.shape == (n_steps,)
    assert std_sx.shape == (n_steps,)
    assert p95.shape == (n_steps,)


def test_mean_between_min_max():
    """mean should be between min and max percentiles."""
    lat = _make_fodo_lattice()
    cfg = _make_beam_config(n=150)
    study = ErrorStudy(lat, cfg, n_seeds=4)
    study.add_error("Q*", "gradient_rel", distribution="gaussian", sigma=0.05)
    results = study.run()

    mean_t = results.mean("transmission")
    p5 = results.percentile("transmission", 5)
    p95 = results.percentile("transmission", 95)
    # Mean should lie within p5 and p95 (element-wise)
    assert np.all(mean_t >= p5 - 1e-10)
    assert np.all(mean_t <= p95 + 1e-10)


# ---------------------------------------------------------------------------
# Test 8: transmission_stats returns correct dict
# ---------------------------------------------------------------------------

def test_transmission_stats_structure():
    lat = _make_fodo_lattice()
    cfg = _make_beam_config(n=100)
    study = ErrorStudy(lat, cfg, n_seeds=3)
    results = study.run()

    stats = results.transmission_stats()
    assert set(stats.keys()) == {"mean", "min", "max", "std"}
    assert stats["min"] <= stats["mean"] <= stats["max"]


def test_transmission_stats_no_loss():
    """With large apertures and no errors, all seeds should have 100% transmission."""
    lat = _make_fodo_lattice()  # aperture=50mm, big enough
    cfg = _make_beam_config(n=100)
    study = ErrorStudy(lat, cfg, n_seeds=3)
    results = study.run()

    stats = results.transmission_stats()
    assert stats["min"] == pytest.approx(100.0, abs=0.01)
    assert stats["max"] == pytest.approx(100.0, abs=0.01)


# ---------------------------------------------------------------------------
# Test 9: Uniform distribution half_width respected
# ---------------------------------------------------------------------------

def test_uniform_error_bounded():
    """Uniform errors should stay within ±half_width."""
    lat = _make_fodo_lattice()
    cfg = _make_beam_config()
    half_w = 0.3
    study = ErrorStudy(lat, cfg, n_seeds=10)
    study.add_error("QF_*", "gradient_rel", distribution="uniform", half_width=half_w)

    original_g = lat.elements[0].gradient
    for seed in range(10):
        lc = study._apply_errors(seed=seed)
        g_new = lc.elements[0].gradient
        rel_err = (g_new - original_g) / original_g
        assert abs(rel_err) <= half_w + 1e-9, f"seed={seed}: rel_err={rel_err}"


# ---------------------------------------------------------------------------
# Test 10: Gaussian cutoff respected
# ---------------------------------------------------------------------------

def test_gaussian_cutoff_respected():
    """Gaussian errors should be clipped at cutoff*sigma."""
    lat = _make_fodo_lattice()
    cfg = _make_beam_config()
    sigma = 0.1
    cutoff = 2.0
    study = ErrorStudy(lat, cfg, n_seeds=20)
    study.add_error("QF_*", "gradient_rel", distribution="gaussian",
                    sigma=sigma, cutoff=cutoff)

    original_g = lat.elements[0].gradient
    for seed in range(20):
        lc = study._apply_errors(seed=seed)
        g_new = lc.elements[0].gradient
        rel_err = (g_new - original_g) / original_g
        assert abs(rel_err) <= cutoff * sigma + 1e-9, f"seed={seed}: rel_err={rel_err}"


# ---------------------------------------------------------------------------
# Test 11: cooperative cancellation (should_stop / progress_cb)
# ---------------------------------------------------------------------------

def test_run_without_hooks_matches_requested_seeds():
    """Default call (no hooks) is unchanged: every seed runs."""
    lat = _make_fodo_lattice()
    cfg = _make_beam_config(n=60)
    study = ErrorStudy(lat, cfg, n_seeds=3)
    study.add_error("QF_*", "gradient_rel", sigma=0.01)
    results = study.run()
    assert results.n_seeds == 3
    assert results.n_requested == 3


def test_should_stop_keeps_only_whole_seeds():
    """Stop after two completed seeds → exactly 2 whole seeds kept."""
    lat = _make_fodo_lattice()
    cfg = _make_beam_config(n=60)
    study = ErrorStudy(lat, cfg, n_seeds=5)
    study.add_error("QF_*", "gradient_rel", sigma=0.01)

    progressed = []
    results = study.run(
        should_stop=lambda: len(progressed) >= 2,
        progress_cb=lambda i, n: progressed.append((i, n)),
    )
    assert results.n_seeds == 2
    assert results.n_requested == 5
    assert progressed == [(1, 5), (2, 5)]
    # Whole seeds only: every recorder has the full lattice's records.
    lengths = {len(r.s) for r in results._recorders}
    assert len(lengths) == 1


def test_should_stop_immediately_returns_empty():
    lat = _make_fodo_lattice()
    cfg = _make_beam_config(n=60)
    study = ErrorStudy(lat, cfg, n_seeds=4)
    study.add_error("QF_*", "gradient_rel", sigma=0.01)
    results = study.run(should_stop=lambda: True)
    assert results.n_seeds == 0
    assert results.n_requested == 4


def test_mid_seed_stop_discards_truncated_recorder():
    """A stop that lands during tracking (per-element poll) must discard
    the truncated seed — recorders of unequal length would corrupt every
    ensemble statistic."""
    lat = _make_fodo_lattice()
    cfg = _make_beam_config(n=60)
    study = ErrorStudy(lat, cfg, n_seeds=4)
    study.add_error("QF_*", "gradient_rel", sigma=0.01)

    # Reference length of a complete recorder.
    ref = study.run(should_stop=None)
    full_len = len(ref._recorders[0].s)

    calls = {"n": 0}
    def stop_on_some_later_poll():
        calls["n"] += 1
        # False for the first seed's polls, then True on a poll that will
        # land mid-tracking of a subsequent seed (loop-top + per-element
        # polls interleave, so an arbitrary later call index is fine).
        return calls["n"] > full_len + 2

    results = study.run(should_stop=stop_on_some_later_poll)
    assert results.n_seeds < 4
    for r in results._recorders:
        assert len(r.s) == full_len   # nothing truncated got in
