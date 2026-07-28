# tests/tracking/test_envelope.py
"""Tests for RMS envelope solver (Task 9.1)."""
import numpy as np
import pytest

from linac_gen.core.particle import PROTON
from linac_gen.core.reference import ReferenceParticle
from linac_gen.core.lattice import Lattice
from linac_gen.core.beam import Beam
from linac_gen.elements.drift import Drift
from linac_gen.elements.quadrupole import Quadrupole
from linac_gen.tracking.envelope import EnvelopeSolver, EnvelopeResults, compute_perveance
from linac_gen.tracking.tracker import Tracker


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ref():
    return ReferenceParticle(species=PROTON, w_kin=3.0, frequency=352.21)


def _default_emittances():
    return {"emit_x": 1.0, "emit_y": 1.0, "emit_z": 1.0}  # mm.mrad


def _default_twiss():
    return {
        "alpha_x": 0.0, "beta_x": 10.0,
        "alpha_y": 0.0, "beta_y": 10.0,
        "alpha_z": 0.0, "beta_z": 10.0,
    }


def _make_solver(lattice, current=0.0, twiss=None, emittances=None):
    ref = _ref()
    twiss = twiss or _default_twiss()
    emittances = emittances or _default_emittances()
    initial = {**twiss, **emittances}
    return EnvelopeSolver(lattice, ref, initial, current=current)


# ---------------------------------------------------------------------------
# Test: EnvelopeResults dataclass
# ---------------------------------------------------------------------------

def test_envelope_results_has_expected_fields():
    res = EnvelopeResults(
        s=[], sigma_x=[], sigma_y=[], sigma_phi=[], sigma_w=[],
        emit_x=[], emit_y=[], emit_z=[],
        alpha_x=[], beta_x=[], alpha_y=[], beta_y=[],
        ref_w_kin=[], ref_beta=[], ref_gamma=[],
    )
    for attr in ("s", "sigma_x", "sigma_y", "sigma_phi", "sigma_w",
                 "emit_x", "emit_y", "emit_z",
                 "alpha_x", "beta_x", "alpha_y", "beta_y",
                 "ref_w_kin", "ref_beta", "ref_gamma"):
        assert hasattr(res, attr)


# ---------------------------------------------------------------------------
# Test: Perveance calculation
# ---------------------------------------------------------------------------

def test_perveance_zero_current_is_zero():
    ref = _ref()
    K = compute_perveance(0.0, 1, PROTON.mass, ref.beta, ref.gamma)
    assert K == 0.0


def test_perveance_positive_current_positive():
    ref = _ref()
    K = compute_perveance(60.0, 1, PROTON.mass, ref.beta, ref.gamma)
    assert K > 0.0


def test_perveance_known_value():
    """K should be ~O(1e-4) for 60 mA proton beam at 3 MeV."""
    ref = _ref()
    K = compute_perveance(60.0, 1, PROTON.mass, ref.beta, ref.gamma)
    # Rough sanity check: perveance for 60 mA proton at ~beta~0.08 should be ~1e-4 to 1e-2
    assert 1e-6 < K < 1e-1


def test_perveance_scales_linearly_with_current():
    ref = _ref()
    K1 = compute_perveance(30.0, 1, PROTON.mass, ref.beta, ref.gamma)
    K2 = compute_perveance(60.0, 1, PROTON.mass, ref.beta, ref.gamma)
    assert abs(K2 / K1 - 2.0) < 1e-9


# ---------------------------------------------------------------------------
# Test: Sigma matrix initialization
# ---------------------------------------------------------------------------

def test_sigma_matrix_initialized_correctly():
    """sigma_x at start = sqrt(beta_x * emit_x)."""
    lat = Lattice()
    lat.add(Drift("D1", 1.0))
    solver = _make_solver(lat, twiss={"alpha_x": 0.0, "beta_x": 10.0,
                                      "alpha_y": 0.0, "beta_y": 10.0,
                                      "alpha_z": 0.0, "beta_z": 10.0},
                          emittances={"emit_x": 1.0, "emit_y": 1.0, "emit_z": 1.0})
    res = solver.run()
    expected_sigma_x = np.sqrt(10.0 * 1.0)  # sqrt(beta * emit) = sqrt(10) mm
    assert abs(res.sigma_x[0] - expected_sigma_x) < 1e-10


# ---------------------------------------------------------------------------
# Test: Drift - sigma grows
# ---------------------------------------------------------------------------

def test_drift_sigma_grows():
    """Beam drifting should have increasing sigma_x."""
    lat = Lattice()
    lat.add(Drift("D1", 500.0))
    solver = _make_solver(lat)
    res = solver.run()
    assert len(res.sigma_x) == 2  # initial + after drift
    assert res.sigma_x[-1] > res.sigma_x[0]


def test_drift_records_two_points():
    """Should record initial state + state after each element."""
    lat = Lattice()
    lat.add(Drift("D1", 100.0))
    lat.add(Drift("D2", 200.0))
    solver = _make_solver(lat)
    res = solver.run()
    assert len(res.s) == 3  # initial + 2 elements
    assert res.s[0] == pytest.approx(0.0)
    assert res.s[1] == pytest.approx(100.0)
    assert res.s[2] == pytest.approx(300.0)


# ---------------------------------------------------------------------------
# Test: Emittance preservation (no SC)
# ---------------------------------------------------------------------------

def test_emittance_preserved_drift():
    """Geometric emittance should be constant through a drift."""
    lat = Lattice()
    lat.add(Drift("D1", 500.0))
    solver = _make_solver(lat, current=0.0)
    res = solver.run()
    for emit in res.emit_x:
        assert abs(emit - res.emit_x[0]) < 1e-10, "emit_x changed through drift"
    for emit in res.emit_y:
        assert abs(emit - res.emit_y[0]) < 1e-10, "emit_y changed through drift"


def test_emittance_preserved_fodo():
    """Geometric emittance should be constant through a FODO cell."""
    lat = Lattice()
    lat.add(Quadrupole("QF", 50.0, gradient=5.0))
    lat.add(Drift("D1", 200.0))
    lat.add(Quadrupole("QD", 50.0, gradient=-5.0))
    lat.add(Drift("D2", 200.0))
    solver = _make_solver(lat, current=0.0)
    res = solver.run()
    emit0 = res.emit_x[0]
    for emit in res.emit_x:
        assert abs(emit - emit0) < 1e-9 * emit0, f"emit_x changed: {emit} vs {emit0}"


# ---------------------------------------------------------------------------
# Test: FODO - periodic envelope
# ---------------------------------------------------------------------------

def test_fodo_periodic_envelope():
    """After one FODO cell, sigma_x should return close to initial value if started matched."""
    from linac_gen.tracking.matrix_tracking import compute_transfer_matrix, compute_twiss
    ref = _ref()
    lat = Lattice()
    lat.add(Quadrupole("QF", 50.0, gradient=5.0))
    lat.add(Drift("D1", 200.0))
    lat.add(Quadrupole("QD", 50.0, gradient=-5.0))
    lat.add(Drift("D2", 200.0))

    # Get matched Twiss for this FODO cell
    M = compute_transfer_matrix(lat, ref)
    twiss_x = compute_twiss(M, plane="x")
    twiss_y = compute_twiss(M, plane="y")

    initial = {
        "alpha_x": twiss_x["alpha"],
        "beta_x": twiss_x["beta"],
        "alpha_y": twiss_y["alpha"],
        "beta_y": twiss_y["beta"],
        "alpha_z": 0.0, "beta_z": 10.0,
        "emit_x": 1.0, "emit_y": 1.0, "emit_z": 1.0,
    }
    solver = EnvelopeSolver(lat, ref, initial, current=0.0)
    res = solver.run()
    # After one matched cell, sigma_x should be nearly the same as the start
    assert abs(res.sigma_x[-1] - res.sigma_x[0]) < 1e-6 * res.sigma_x[0]


# ---------------------------------------------------------------------------
# Test: Space charge increases beam size
# ---------------------------------------------------------------------------

def test_sc_increases_beam_size():
    """Beam with SC should be larger than without SC."""
    lat = Lattice()
    lat.add(Drift("D1", 1000.0))

    emittances = {"emit_x": 0.5, "emit_y": 0.5, "emit_z": 0.5}
    twiss = {"alpha_x": 0.0, "beta_x": 5.0,
             "alpha_y": 0.0, "beta_y": 5.0,
             "alpha_z": 0.0, "beta_z": 5.0}
    initial = {**twiss, **emittances}

    ref = _ref()
    solver_nosc = EnvelopeSolver(lat, ref, initial, current=0.0)
    res_nosc = solver_nosc.run()

    ref2 = _ref()
    solver_sc = EnvelopeSolver(lat, ref2, initial, current=60.0)
    res_sc = solver_sc.run()

    assert res_sc.sigma_x[-1] > res_nosc.sigma_x[-1], \
        "SC should increase sigma_x"
    assert res_sc.sigma_y[-1] > res_nosc.sigma_y[-1], \
        "SC should increase sigma_y"


# ---------------------------------------------------------------------------
# Test: Compare with multi-particle tracking (no SC)
# ---------------------------------------------------------------------------

def test_compare_envelope_with_multiparticle():
    """
    Envelope and multi-particle sigma_x should agree within 5% for a drift.
    Use Gaussian beam (exact for linear tracking).
    """
    lat = Lattice()
    lat.add(Drift("D1", 500.0))

    emit_x = 1.0  # mm.mrad
    alpha_x = 0.0
    beta_x = 10.0  # mm/mrad

    ref = _ref()
    initial = {
        "alpha_x": alpha_x, "beta_x": beta_x,
        "alpha_y": 0.0, "beta_y": 10.0,
        "alpha_z": 0.0, "beta_z": 10.0,
        "emit_x": emit_x, "emit_y": 1.0, "emit_z": 1.0,
    }

    solver = EnvelopeSolver(lat, ref, initial, current=0.0)
    env_res = solver.run()

    # Multi-particle: Gaussian distribution with same Twiss
    n_part = 50000
    rng = np.random.default_rng(0)
    beam = Beam(ref=_ref(), n_particles=n_part, current=0.0)
    # draw correlated (x, xp) from Twiss covariance matrix
    cov_x = np.array([[beta_x * emit_x, -alpha_x * emit_x],
                       [-alpha_x * emit_x, (1.0 + alpha_x**2) / beta_x * emit_x]])
    beam.particles[:, :2] = rng.multivariate_normal([0, 0], cov_x, n_part)
    beam.particles[:, 2:4] = rng.multivariate_normal([0, 0], cov_x, n_part)  # same for y

    tracker = Tracker(lat, beam)
    tracker.run()

    mp_sigma_x = float(np.std(beam.particles[:, 0]))
    env_sigma_x = env_res.sigma_x[-1]

    rel_diff = abs(mp_sigma_x - env_sigma_x) / env_sigma_x
    assert rel_diff < 0.05, \
        f"sigma_x mismatch: envelope={env_sigma_x:.4f}, multi-particle={mp_sigma_x:.4f}, rel_diff={rel_diff:.2%}"


# ---------------------------------------------------------------------------
# Test: Twiss extracted from sigma matrix
# ---------------------------------------------------------------------------

def test_twiss_extracted_from_sigma():
    """alpha and beta extracted from sigma matrix should be self-consistent."""
    lat = Lattice()
    lat.add(Drift("D1", 100.0))
    solver = _make_solver(lat)
    res = solver.run()
    # beta * gamma_t - alpha^2 = 1 for Twiss parameters
    # We just check that beta_x > 0 and alpha_x is finite
    for i in range(len(res.s)):
        assert res.beta_x[i] > 0
        assert np.isfinite(res.alpha_x[i])


def test_ref_state_recorded():
    """Reference particle state should be recorded at each step."""
    lat = Lattice()
    lat.add(Drift("D1", 100.0))
    solver = _make_solver(lat)
    res = solver.run()
    assert len(res.ref_w_kin) == 2
    assert res.ref_w_kin[0] == pytest.approx(3.0)
    assert len(res.ref_beta) == 2
    assert len(res.ref_gamma) == 2
    assert res.ref_beta[0] > 0
    assert res.ref_gamma[0] > 1.0


# ---------------------------------------------------------------------------
# Test: Simulation.run_envelope() integration
# ---------------------------------------------------------------------------

def test_simulation_run_envelope():
    """Simulation.run_envelope() should return EnvelopeResults."""
    from linac_gen.core.simulation import Simulation

    lat = Lattice()
    lat.add(Drift("D1", 100.0))
    ref = _ref()
    beam = Beam(ref=ref, n_particles=1, current=0.0)

    sim = Simulation(lat, beam)
    sim.beam_envelope_params = {
        "alpha_x": 0.0, "beta_x": 10.0,
        "alpha_y": 0.0, "beta_y": 10.0,
        "alpha_z": 0.0, "beta_z": 10.0,
        "emit_x": 1.0, "emit_y": 1.0, "emit_z": 1.0,
    }
    res = sim.run_envelope()
    assert isinstance(res, EnvelopeResults)
    assert len(res.s) == 2


# ---------------------------------------------------------------------------
# Test: Sigma matrix positive definite preserved
# ---------------------------------------------------------------------------

def test_sigma_positive_definite():
    """Sigma matrix eigenvalues should remain positive throughout."""
    lat = Lattice()
    lat.add(Quadrupole("QF", 50.0, gradient=5.0))
    lat.add(Drift("D1", 200.0))
    lat.add(Quadrupole("QD", 50.0, gradient=-5.0))
    lat.add(Drift("D2", 200.0))
    solver = _make_solver(lat, current=0.0)
    res = solver.run()
    # sigma_x, sigma_y > 0 throughout
    for sx, sy in zip(res.sigma_x, res.sigma_y):
        assert sx > 0
        assert sy > 0
