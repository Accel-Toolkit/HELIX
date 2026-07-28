"""Regression tests for the 3D-ellipsoid envelope SC model.

Pins three claims:

1. The depolarization factors (M_x, M_y, M_z) satisfy M_x + M_y + M_z = 1
   and are 1/3 each for a sphere.
2. The envelope now includes a longitudinal SC kick (σ_φ grows with I>0).
3. On the FODO cell used for TraceWin validation, envelope-3D agrees
   with the TraceWin reference σ_x, σ_y, σ_φ to within 2 % at I = 5 mA.
"""
import math

import numpy as np
import pytest

from linac_gen.core.particle import H_MINUS
from linac_gen.core.reference import ReferenceParticle
from linac_gen.tracking.envelope import (
    EnvelopeSolver,
    _ellipsoid_depolarization_factors,
    _sc_kick_matrix_3d,
)


def test_depolarization_factors_sum_to_one_and_sphere_is_1_third():
    # Sphere
    fx, fy, fz = _ellipsoid_depolarization_factors(1.0, 1.0, 1.0)
    assert fx == pytest.approx(1.0 / 3.0, abs=1e-12)
    assert fy == pytest.approx(1.0 / 3.0, abs=1e-12)
    assert fz == pytest.approx(1.0 / 3.0, abs=1e-12)
    # Prolate (a = b < c)
    fx, fy, fz = _ellipsoid_depolarization_factors(1.0, 1.0, 5.0)
    assert fx + fy + fz == pytest.approx(1.0, rel=1e-10)
    assert fx == pytest.approx(fy, rel=1e-12)
    assert fz < 1.0 / 3.0   # long axis → smaller depolarization
    # Oblate (a = b > c)
    fx, fy, fz = _ellipsoid_depolarization_factors(5.0, 5.0, 1.0)
    assert fx + fy + fz == pytest.approx(1.0, rel=1e-10)
    assert fx == pytest.approx(fy, rel=1e-12)
    assert fz > 1.0 / 3.0   # short axis → larger depolarization
    # Fully triaxial
    fx, fy, fz = _ellipsoid_depolarization_factors(1.0, 2.0, 5.0)
    assert fx + fy + fz == pytest.approx(1.0, rel=1e-10)
    assert fz < fy < fx    # longest axis has smallest form factor


def test_sc_kick_matrix_is_identity_at_zero_current():
    M = _sc_kick_matrix_3d(
        current_mA=0.0, charge_state=1, mass_MeV=939.301,
        beta=0.067, gamma=1.0023, frequency_MHz=162.5,
        sigma_x_mm=1.0, sigma_y_mm=1.0, sigma_phi_deg=5.0,
        ds_mm=10.0,
    )
    np.testing.assert_allclose(M, np.eye(6), atol=1e-14)


def test_sc_kick_has_longitudinal_coupling():
    """With I > 0 the [5, 4] element is non-zero (3D model adds longitudinal
    defocusing that the old 2D Sacherer formula lacked)."""
    M = _sc_kick_matrix_3d(
        current_mA=5.0, charge_state=1, mass_MeV=939.301,
        beta=0.067, gamma=1.0023, frequency_MHz=162.5,
        sigma_x_mm=1.0, sigma_y_mm=0.6, sigma_phi_deg=7.0,
        ds_mm=10.0,
    )
    assert M[1, 0] > 0.0
    assert M[3, 2] > 0.0
    # Longitudinal: Δφ>0 (late particle) → ΔW<0, so M[5,4] is negative
    assert M[5, 4] < 0.0


def test_fodo_envelope_3d_matches_tracewin_at_5mA():
    """FODO cell extracted from examples/fodo_cell.dat: envelope 3D at
    I=5 mA should agree with TraceWin's multiparticle σ_x, σ_y, σ_φ
    end-of-lattice to within 2 %."""
    from linac_gen.io.tracewin_parser import parse_tracewin
    lattice, _ = parse_tracewin("examples/fodo_cell.dat")

    W0, FREQ = 2.1226695, 162.5
    # Initial Twiss extracted & validated in the PR description; matches TW
    # no-SC output to 0.01 % end-of-lattice in the pure linear regime.
    initial = dict(
        alpha_x=1.228, beta_x=0.316, emit_x=3.122,
        alpha_y=-0.095, beta_y=0.113, emit_y=3.122,
        alpha_z=0.0,
        beta_z=7.144 * 7.144 / (7.144 * 8.72e-3),
        emit_z=7.144 * 8.72e-3,
    )
    ref = ReferenceParticle(species=H_MINUS, w_kin=W0, frequency=FREQ)
    res = EnvelopeSolver(lattice, ref, initial, current=5.0).run()

    # TraceWin end-of-lattice values (from spacechargeenvelope.txt, I≈5 mA)
    tw_sx = 6.421   # mm
    tw_sy = 8.864   # mm
    # σ_z anchor converted to degrees of the MACHINE frequency: the deck's
    # ``FREQ 352.21`` card now governs the reporting clock from s=0 (TW
    # semantics), so the 15.703° @ 162.5 MHz value becomes ×(352.21/162.5).
    # Same physical bunch length — only the degree units changed.
    tw_sphi = 15.703 * 352.21 / 162.5  # deg @ 352.21 MHz

    assert abs(res.sigma_x[-1] - tw_sx) / tw_sx < 0.02
    assert abs(res.sigma_y[-1] - tw_sy) / tw_sy < 0.02
    assert abs(res.sigma_phi[-1] - tw_sphi) / tw_sphi < 0.02


def test_sc_off_gives_envelope_identical_to_tw_nosc_at_end():
    """At I=0, envelope must still match TraceWin no-SC to < 0.1 %."""
    from linac_gen.io.tracewin_parser import parse_tracewin
    lattice, _ = parse_tracewin("examples/fodo_cell.dat")

    W0, FREQ = 2.1226695, 162.5
    initial = dict(
        alpha_x=1.228, beta_x=0.316, emit_x=3.122,
        alpha_y=-0.095, beta_y=0.113, emit_y=3.122,
        alpha_z=0.0,
        beta_z=7.144 * 7.144 / (7.144 * 8.72e-3),
        emit_z=7.144 * 8.72e-3,
    )
    ref = ReferenceParticle(species=H_MINUS, w_kin=W0, frequency=FREQ)
    res = EnvelopeSolver(lattice, ref, initial, current=0.0).run()

    # TraceWin no-SC end-of-lattice
    assert abs(res.sigma_x[-1] - 5.864) / 5.864 < 0.001
    assert abs(res.sigma_y[-1] - 7.509) / 7.509 < 0.001
