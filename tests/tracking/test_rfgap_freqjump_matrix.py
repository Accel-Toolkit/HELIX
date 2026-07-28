"""RFGap.kick_matrix across a frequency jump: the map is K·D, not K.

Particle tracking (``apply_kick``) rescales the stored phase deviation
by r = f_new/f_old BEFORE the RF energy shear reads it, so the
linearized gap map across a jump is

    D = diag(…, r, 1)      (phase-coordinate unit change)
    K = shear (M[5,4] = k_φW) + transverse damping/defocus
    M = K·D  →  M[4,4] = r,  M[5,4] = k_φW · r

``kick_matrix`` used to carry neither factor, so
``compute_transfer_matrix`` disagreed with actual tracking across every
162.5 → 325 MHz boundary.  The envelope solver is unaffected: it
pre-rescales σ and updates ref.frequency before requesting the matrix,
so it sees r = 1 (exactly one rescale per path).
"""
from __future__ import annotations

import numpy as np
import pytest

from linac_gen.core.beam import Beam
from linac_gen.core.lattice import Lattice
from linac_gen.core.particle import PROTON
from linac_gen.core.reference import ReferenceParticle
from linac_gen.elements.rf_gap import RFGap
from linac_gen.tracking.envelope import EnvelopeSolver

F_OLD = 162.5
F_NEW = 325.0
R = F_NEW / F_OLD


def _gap(phase=-35.0, voltage=0.3) -> RFGap:
    return RFGap(name="G", voltage=voltage, phase=phase, frequency=F_NEW)


def _ref() -> ReferenceParticle:
    return ReferenceParticle(species=PROTON, w_kin=2.5, frequency=F_OLD)


def _fd_jacobian(gap: RFGap, ref: ReferenceParticle,
                 delta: float = 1e-6) -> np.ndarray:
    """Finite-difference Jacobian of apply_kick around the zero orbit.

    One beam carries all 12 probe particles (±δ per coordinate) plus an
    on-axis one; apply_kick is per-particle so a single call suffices.
    """
    beam = Beam(ref.copy(), n_particles=13, current=0.0)
    for j in range(6):
        beam.particles[2 * j, j] = +delta
        beam.particles[2 * j + 1, j] = -delta
    gap.apply_kick(beam)
    J = np.empty((6, 6))
    for j in range(6):
        J[:, j] = (beam.particles[2 * j] - beam.particles[2 * j + 1]) / (2 * delta)
    # On-axis particle must stay on axis (fixed point of the map).
    np.testing.assert_allclose(beam.particles[12], 0.0, atol=1e-12)
    return J


def test_kick_matrix_matches_tracking_jacobian_across_jump():
    gap = _gap()
    ref = _ref()
    M = gap.kick_matrix(ref)
    J = _fd_jacobian(gap, ref)
    np.testing.assert_allclose(M, J, rtol=1e-6, atol=1e-9)


def test_kick_matrix_kd_structure():
    gap = _gap()
    ref = _ref()
    M = gap.kick_matrix(ref)
    # D factor on the phase coordinate.
    assert M[4, 4] == pytest.approx(R)
    # Shear reads the POST-rescale phase: k_φW · r.
    charge = ref.species.charge
    k_pw = (-charge * gap.effective_voltage * gap.ttf
            * np.sin(np.deg2rad(gap.effective_phase)) * np.pi / 180.0)
    assert M[5, 4] == pytest.approx(k_pw * R, rel=1e-12)


def test_kick_matrix_unchanged_without_jump():
    gap = RFGap(name="G", voltage=0.3, phase=-35.0, frequency=F_OLD)
    ref = _ref()
    M = gap.kick_matrix(ref)
    assert M[4, 4] == 1.0
    J = _fd_jacobian(gap, ref)
    np.testing.assert_allclose(M, J, rtol=1e-6, atol=1e-9)


def test_envelope_applies_frequency_rescale_exactly_once():
    """σ_φ through a jump gap scales by r once (envelope pre-rescale),
    not r² (double count with the new matrix factor)."""
    lat = Lattice()
    # φ_s = −90°: pure buncher — no acceleration, no damping, so the
    # only effect on σ_φ is the frequency-unit change.
    lat.add(RFGap(name="G", voltage=0.1, phase=-90.0, frequency=F_NEW))
    ref = _ref()
    sigma_phi_in = 10.0
    initial = dict(alpha_x=0.0, beta_x=1.0, emit_x=1.0,
                   alpha_y=0.0, beta_y=1.0, emit_y=1.0,
                   alpha_z=0.0, beta_z=float(sigma_phi_in**2 / 0.05),
                   emit_z=0.05)
    res = EnvelopeSolver(lat, ref, initial, current=0.0).run()
    assert res.sigma_phi[0] == pytest.approx(sigma_phi_in, rel=1e-12)
    assert res.sigma_phi[-1] == pytest.approx(R * sigma_phi_in, rel=1e-9)
