"""Unit tests for the Sacherer / Kapchinsky-Vladimirsky ODE envelope solver."""
import numpy as np
import pytest

from linac_gen.core.particle import H_MINUS
from linac_gen.core.reference import ReferenceParticle
from linac_gen.elements.drift import Drift
from linac_gen.elements.quadrupole import Quadrupole
from linac_gen.core.lattice import Lattice
from linac_gen.tracking.sacherer import SachererSolver


@pytest.fixture
def ref_dc():
    return ReferenceParticle(species=H_MINUS, w_kin=0.030, frequency=162.5)


@pytest.fixture
def init_twiss():
    return dict(
        # ecosystem dialect: GEOMETRIC emittance in mm·mrad
        # (1.0 mm·mrad = 1 µm·rad — same beam as before the 2026-07-25
        # unit-contract fix, only the numeral changed)
        alpha_x=0.0, beta_x=1.0, emit_x=1.0,
        alpha_y=0.0, beta_y=1.0, emit_y=1.0,
    )


def _make_lat(*elems):
    lat = Lattice()
    for e in elems:
        lat.add(e)
    return lat


def test_drift_only_no_sc_matches_free_expansion(ref_dc, init_twiss):
    """A 1 m drift with α=0, β=1, ε=1 µm·rad and no SC → analytic σ(s)."""
    lat = _make_lat(Drift(name="drift1", length=1000.0))  # 1 m
    sol = SachererSolver(lat, ref_dc, init_twiss, current=0.0).run()
    sx = np.asarray(sol.sigma_x)
    s_mm = np.asarray(sol.s)
    # σ_x(s) = sqrt(β·ε) · sqrt(1 + (s/β)²)  for α₀=0
    sx_init_mm = np.sqrt(1.0 * 1e-6) * 1e3      # mm
    s_m = s_mm * 1e-3
    expected_mm = sx_init_mm * np.sqrt(1.0 + s_m ** 2)
    np.testing.assert_allclose(sx, expected_mm, rtol=1e-4, atol=1e-4)


def test_quadrupole_focusing_sign(ref_dc, init_twiss):
    """A focusing quad reduces σ_x; a defocusing quad grows it.

    Verifies κ_x sign convention is consistent with the rest of LG.
    """
    # Hard-edge quad with positive gradient: focuses x, defocuses y.
    lat_focus = _make_lat(
        Drift(name="d1", length=100.0),
        Quadrupole(name="qf", length=100.0, gradient=10.0, aperture=0.0),
        Drift(name="d2", length=100.0),
    )
    lat_defoc = _make_lat(
        Drift(name="d1", length=100.0),
        Quadrupole(name="qd", length=100.0, gradient=-10.0, aperture=0.0),
        Drift(name="d2", length=100.0),
    )
    f = SachererSolver(lat_focus, ref_dc, init_twiss, current=0.0).run()
    d = SachererSolver(lat_defoc, ref_dc, init_twiss, current=0.0).run()
    # Focusing quad should keep σ_x smaller than defocusing quad at exit.
    assert f.sigma_x[-1] < d.sigma_x[-1]
    # And the OPPOSITE for σ_y.
    assert f.sigma_y[-1] > d.sigma_y[-1]


def test_space_charge_grows_envelope(ref_dc, init_twiss):
    """Adding a finite current should grow σ for a drift envelope."""
    lat = _make_lat(Drift(name="d", length=2000.0))  # 2 m
    no_sc = SachererSolver(lat, ref_dc, init_twiss, current=0.0).run()
    with_sc = SachererSolver(lat, ref_dc, init_twiss, current=10.0).run()
    assert with_sc.sigma_x[-1] > no_sc.sigma_x[-1]
    assert with_sc.sigma_y[-1] > no_sc.sigma_y[-1]


def test_zero_current_matches_no_sc(ref_dc, init_twiss):
    """current=0 must equal the baseline drift expansion."""
    lat = _make_lat(Drift(name="d", length=500.0))
    a = SachererSolver(lat, ref_dc, init_twiss, current=0.0).run()
    b = SachererSolver(lat, ref_dc, init_twiss, current=0.0).run()
    np.testing.assert_array_equal(a.sigma_x, b.sigma_x)


def test_production_path_parity_with_matrix_solver():
    """Unit-contract guard (2026-07-25): the SAME BeamConfig through
    ``run_envelope_sim``'s two branches must describe the same beam.

    This is the test class that was missing: solver-only tests can
    never see a boundary-convention mismatch, and before the fix every
    production caller fed SachererSolver mm·mrad where it expected
    m·rad — σ came out exactly 1000× large on this very check."""
    from linac_gen.cli import common
    from linac_gen.core.config import BeamConfig

    cfg = BeamConfig(species="H-", energy=0.030, frequency=162.5,
                     current=0.0,
                     emit_nx=0.21, alpha_x=0.5, beta_x=0.8,
                     emit_ny=0.19, alpha_y=-0.3, beta_y=0.6,
                     emit_z=0.06, alpha_z=0.0, beta_z=800.0)
    lat = _make_lat(Drift(name="d", length=500.0, aperture=50.0))

    res_m = common.run_envelope_sim(lat, cfg)                     # matrix
    res_s = common.run_envelope_sim(lat, cfg, env_solver="sacherer")

    # Row 0 is pure unit bookkeeping — must agree tightly.
    assert res_s.sigma_x[0] == pytest.approx(res_m.sigma_x[0], rel=1e-9)
    assert res_s.sigma_y[0] == pytest.approx(res_m.sigma_y[0], rel=1e-9)
    # End of drift: exact linear transport vs the ODE integrator.
    assert res_s.sigma_x[-1] == pytest.approx(res_m.sigma_x[-1], rel=1e-3)
    assert res_s.sigma_y[-1] == pytest.approx(res_m.sigma_y[-1], rel=1e-3)


def test_hard_edge_solenoid_focuses_and_matches_matrix_solver():
    """Claim 1 (2026-07-25): the hard-edge SOLENOID card was silently a
    drift.  Now: Larmor focusing kappa=(B/2Brho)^2 in both planes,
    checked against the matrix envelope solver through the PRODUCTION
    dispatch on a round DC beam (rotation cancels for round beams)."""
    from linac_gen.cli import common
    from linac_gen.core.config import BeamConfig
    from linac_gen.elements.solenoid import Solenoid

    cfg = BeamConfig(species="H-", energy=0.030, frequency=162.5,
                     current=0.0,
                     emit_nx=0.2, alpha_x=0.0, beta_x=1.0,
                     emit_ny=0.2, alpha_y=0.0, beta_y=1.0,
                     emit_z=0.06, alpha_z=0.0, beta_z=800.0)
    sol_lat = _make_lat(Drift(name="d1", length=200.0),
                        Solenoid(name="S1", length=300.0, field=0.05,
                                 aperture=50.0),
                        Drift(name="d2", length=200.0))
    drift_lat = _make_lat(Drift(name="d", length=700.0))

    res_sac = common.run_envelope_sim(sol_lat, cfg, env_solver="sacherer")
    res_mat = common.run_envelope_sim(sol_lat, cfg)
    res_drift = common.run_envelope_sim(drift_lat, cfg,
                                        env_solver="sacherer")

    # 1. The solenoid FOCUSES relative to a pure drift (was identical).
    assert res_sac.sigma_x[-1] < res_drift.sigma_x[-1] * 0.98
    # 2. Cross-solver parity through the production path.
    assert res_sac.sigma_x[-1] == pytest.approx(res_mat.sigma_x[-1],
                                                rel=2e-3)
    assert res_sac.sigma_y[-1] == pytest.approx(res_mat.sigma_y[-1],
                                                rel=2e-3)
