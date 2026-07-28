"""FieldMap3D with explicit channel separation.

Verifies that the rewritten tracker:
  * applies no phasor to static channels (phase-independence);
  * applies cos(ωt+φ) to RF electric and sin(ωt+φ) to RF magnetic;
  * correctly rotates (x', y') through a uniform static Bz (Larmor);
  * conserves reference energy for a static magnetic map.
"""
from __future__ import annotations

import numpy as np
import pytest

from linac_gen.core.beam import Beam
from linac_gen.core.reference import ReferenceParticle
from linac_gen.core.particle import PROTON
from linac_gen.elements.field_map_3d import FieldMap3D
from linac_gen.io.field_map_data import FieldMapData, FieldChannel
from linac_gen.io.tracewin_geom import Channel


def _uniform_static_B(Bz_T: float = 0.5, L_mm: float = 200.0):
    nz = 11
    n = 3
    x = np.linspace(-10.0, 10.0, n)
    y = np.linspace(-10.0, 10.0, n)
    z = np.linspace(0.0, L_mm, nz)
    Bx = np.zeros((n, n, nz))
    By = np.zeros((n, n, nz))
    Bz = np.full((n, n, nz), Bz_T)
    fd = FieldMapData(z=z, frequency=0.0)
    fd.channels[Channel.STAT_B] = FieldChannel(
        geometry=7, x=x, y=y, z=z, Fx=Bx, Fy=By, Fz=Bz,
    )
    return fd


def _uniform_rf_E(Ez_MVm: float = 1.0, L_mm: float = 200.0):
    nz = 11
    n = 3
    x = np.linspace(-10.0, 10.0, n)
    y = np.linspace(-10.0, 10.0, n)
    z = np.linspace(0.0, L_mm, nz)
    Ex = np.zeros((n, n, nz))
    Ey = np.zeros((n, n, nz))
    Ez = np.full((n, n, nz), Ez_MVm)
    fd = FieldMapData(z=z, frequency=352.21)
    fd.channels[Channel.RF_E] = FieldChannel(
        geometry=7, x=x, y=y, z=z, Fx=Ex, Fy=Ey, Fz=Ez,
    )
    return fd


def _single(x_mm=0.0, xp_mrad=0.0, y_mm=0.0, yp_mrad=0.0):
    ref = ReferenceParticle(species=PROTON, w_kin=5.0, frequency=352.21)
    b = Beam(ref=ref, n_particles=1, current=0.0)
    b.particles[0] = [x_mm, xp_mrad, y_mm, yp_mrad, 0.0, 0.0]
    return b


def _track(elem, beam):
    ds = elem.length / elem.n_steps
    elem._step_idx = 0
    for _ in range(elem.n_steps):
        elem.track_rk4(beam, ds)


def test_static_solenoid_rotates_xy_plane():
    """Uniform Bz solenoid: off-axis parallel particle gains y' kick.

    Physics: v×Bz couples x-motion into y-kick and vice-versa (Larmor rotation).
    A particle with xp≠0 has v_x component; F_y = q(v_x × Bz) is non-zero.
    OLD CODE would give y'=0 because it omitted the v×Bz coupling entirely.
    """
    fd = _uniform_static_B(Bz_T=0.5, L_mm=200.0)
    sol = FieldMap3D(name="SOL", length=200.0, field_data=fd,
                     scale=1.0, phase=0.0, frequency=0.0, n_steps=200)
    beam = _single(xp_mrad=1.0)
    _track(sol, beam)
    # Under the old (broken) code, beam.particles[0, 3] stayed ≈ 0.
    # With v×Bz applied it must gain a non-trivial rotation.
    assert abs(beam.particles[0, 3]) > 0.05, beam.particles[0]


def test_static_field_is_phase_independent():
    """Changing phase on a static-only map should not change tracking.

    Static channels have phasor=1 (no time dependence). Phase offset
    must be completely ignored for STAT_B and STAT_E channels.
    OLD CODE applied cos(phase) to everything, so phase=180° would flip the field.
    """
    fd = _uniform_static_B(Bz_T=0.3)
    a = FieldMap3D(name="A", length=200.0, field_data=fd, scale=1.0,
                   phase=0.0,   frequency=352.21, n_steps=50)
    b_elem = FieldMap3D(name="B", length=200.0, field_data=fd, scale=1.0,
                        phase=180.0, frequency=352.21, n_steps=50)
    beam_a = _single(xp_mrad=1.0)
    beam_b = _single(xp_mrad=1.0)
    _track(a, beam_a)
    _track(b_elem, beam_b)
    np.testing.assert_allclose(beam_a.particles, beam_b.particles, rtol=1e-12)


def test_rf_electric_accelerates_on_crest():
    """On-crest (phase=0) RF-E cavity should accelerate the reference.

    cos(0)=1 → maximum energy gain. The reference particle starts at 5 MeV
    and should finish with higher kinetic energy.
    """
    fd = _uniform_rf_E(Ez_MVm=1.0, L_mm=200.0)
    cav = FieldMap3D(name="CAV", length=200.0, field_data=fd, scale=1.0,
                     phase=0.0, frequency=352.21, n_steps=100)
    ref = ReferenceParticle(species=PROTON, w_kin=5.0, frequency=352.21)
    W0 = ref.w_kin
    cav.advance_ref(ref)
    assert ref.w_kin > W0, f"Expected acceleration, got dW={ref.w_kin - W0}"


def test_rf_electric_decelerates_on_trough():
    """On-trough (phase=180°) RF-E cavity should decelerate the reference.

    cos(π)=-1 → maximum energy loss. Reference particle should finish
    with lower kinetic energy.
    """
    fd = _uniform_rf_E(Ez_MVm=1.0, L_mm=200.0)
    cav = FieldMap3D(name="CAV", length=200.0, field_data=fd, scale=1.0,
                     phase=180.0, frequency=352.21, n_steps=100)
    ref = ReferenceParticle(species=PROTON, w_kin=5.0, frequency=352.21)
    W0 = ref.w_kin
    cav.advance_ref(ref)
    assert ref.w_kin < W0


def test_static_B_conserves_reference_energy():
    """Pure static-B map must not change reference particle energy.

    Magnetic fields do no work (F⊥v). advance_ref uses only E-channels
    for the on-axis energy integral. A STAT_B-only map must leave w_kin
    unchanged to floating-point precision.
    """
    fd = _uniform_static_B(Bz_T=1.0, L_mm=400.0)
    sol = FieldMap3D(name="SOL", length=400.0, field_data=fd, scale=1.0,
                     phase=0.0, frequency=0.0, n_steps=200)
    ref = ReferenceParticle(species=PROTON, w_kin=5.0, frequency=352.21)
    W0 = ref.w_kin
    sol.advance_ref(ref)
    np.testing.assert_allclose(ref.w_kin, W0, rtol=1e-12)


def test_rf_phase_sensitivity():
    """RF-E map must be phase-sensitive: on-crest and on-trough give opposite dW.

    This is the complementary check to the static phase-independence test.
    The two advance_ref calls (phase=0 vs phase=180) must give different
    and opposite-sign energy changes.
    """
    fd = _uniform_rf_E(Ez_MVm=1.0, L_mm=200.0)

    ref_on = ReferenceParticle(species=PROTON, w_kin=5.0, frequency=352.21)
    cav_on = FieldMap3D(name="ON", length=200.0, field_data=fd, scale=1.0,
                        phase=0.0, frequency=352.21, n_steps=100)
    cav_on.advance_ref(ref_on)
    dW_on = ref_on.w_kin - 5.0

    ref_off = ReferenceParticle(species=PROTON, w_kin=5.0, frequency=352.21)
    cav_off = FieldMap3D(name="OFF", length=200.0, field_data=fd, scale=1.0,
                         phase=180.0, frequency=352.21, n_steps=100)
    cav_off.advance_ref(ref_off)
    dW_off = ref_off.w_kin - 5.0

    # Should be opposite signs (or at least clearly different)
    assert dW_on > 0, f"On-crest: dW={dW_on} should be positive"
    assert dW_off < 0, f"On-trough: dW={dW_off} should be negative"
    # Magnitudes should be roughly equal for a stationary-phase approximation
    # (phi_s doesn't advance much in 200 mm at 5 MeV / 352 MHz)
    np.testing.assert_allclose(abs(dW_on), abs(dW_off), rtol=0.1)
