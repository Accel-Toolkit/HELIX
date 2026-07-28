"""Standard physics benchmarks for the envelope tracker.

Each test pits LG against a closed-form analytic result.  These act as
regression gates: any code change that breaks one of these benchmarks
has broken physics, not just numerics.

Markers:
* ``@pytest.mark.physics`` — opt-in via ``pytest -m physics``.  Default
  test runs skip these, but the CI matrix should include them.
"""
import numpy as np
import pytest

from linac_gen.core.particle import H_MINUS
from linac_gen.core.reference import ReferenceParticle
from linac_gen.core.lattice import Lattice
from linac_gen.elements.drift import Drift
from linac_gen.elements.quadrupole import Quadrupole
from linac_gen.tracking.sacherer import SachererSolver

pytestmark = pytest.mark.physics


def _ref_dc():
    return ReferenceParticle(species=H_MINUS, w_kin=0.030, frequency=162.5)


def _make_lat(*elems):
    lat = Lattice()
    for e in elems:
        lat.add(e)
    return lat


# ---------------------------------------------------------------------------
def test_drift_expansion_no_sc_matches_analytic():
    """A 2 m drift with α=0, β=1 m, ε=1 µm·rad → analytic σ(s).

    Closed form: σ(s) = √(β·ε) · √(1 + (s/β)²)  for α₀ = 0.
    """
    init = dict(
        alpha_x=0.0, beta_x=1.0, emit_x=1.0,
        alpha_y=0.0, beta_y=1.0, emit_y=1.0,
    )
    lat = _make_lat(Drift(name="d", length=2000.0))
    sol = SachererSolver(lat, _ref_dc(), init, current=0.0).run()
    s_m = np.asarray(sol.s) * 1e-3
    sx = np.asarray(sol.sigma_x)
    sx_init = np.sqrt(1e-6) * 1e3
    expected = sx_init * np.sqrt(1.0 + s_m ** 2)
    np.testing.assert_allclose(sx, expected, rtol=1e-4, atol=1e-4)


# ---------------------------------------------------------------------------
def test_drift_expansion_with_sc_grows_more_than_no_sc():
    """SC defocusing must increase σ vs the no-SC baseline."""
    init = dict(
        alpha_x=-1.0, beta_x=1.0, emit_x=1.0,
        alpha_y=-1.0, beta_y=1.0, emit_y=1.0,
    )
    lat = _make_lat(Drift(name="d", length=2000.0))
    no_sc = SachererSolver(lat, _ref_dc(), init, current=0.0).run()
    with_sc = SachererSolver(lat, _ref_dc(), init, current=15.0).run()
    assert with_sc.sigma_x[-1] > no_sc.sigma_x[-1]
    # SC should produce a meaningful (>5 %) growth at this current.
    rel_growth = (with_sc.sigma_x[-1] / no_sc.sigma_x[-1] - 1.0) * 100
    assert rel_growth > 5.0, f"SC growth only {rel_growth:.2f} %"


# ---------------------------------------------------------------------------
def test_focus_quad_reduces_sigma_x():
    """A focusing quadrupole must reduce σ_x downstream relative to drift only."""
    init = dict(
        alpha_x=0.0, beta_x=1.0, emit_x=1.0,
        alpha_y=0.0, beta_y=1.0, emit_y=1.0,
    )
    drift = _make_lat(Drift(name="d1", length=200.0))
    fodo = _make_lat(
        Drift(name="d1", length=100.0),
        Quadrupole(name="qf", length=100.0, gradient=10.0, aperture=0.0),
        Drift(name="d2", length=0.001),
    )
    sd = SachererSolver(drift, _ref_dc(), init, current=0.0).run()
    sf = SachererSolver(fodo, _ref_dc(), init, current=0.0).run()
    # Focusing should pull σ_x in compared to drift expansion.
    assert sf.sigma_x[-1] < sd.sigma_x[-1]


# ---------------------------------------------------------------------------
def test_zero_emittance_matches_geometric_drift():
    """A pencil beam (ε → 0) drifts as a geometric ray; σ_x should
    track the input slope with no expansion from emittance."""
    init = dict(
        alpha_x=0.0, beta_x=1.0, emit_x=1e-6,
        alpha_y=0.0, beta_y=1.0, emit_y=1e-6,
    )
    lat = _make_lat(Drift(name="d", length=1000.0))
    sol = SachererSolver(lat, _ref_dc(), init, current=0.0).run()
    sx = np.asarray(sol.sigma_x)
    # σ should stay tiny across the drift (no significant growth).
    assert np.max(sx) < 1e-2  # mm
