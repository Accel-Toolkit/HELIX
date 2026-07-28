"""FieldMap (1-D / 2-D cyl) multi-channel tracking."""
from __future__ import annotations

import numpy as np
import pytest

from linac_gen.core.beam import Beam
from linac_gen.core.reference import ReferenceParticle
from linac_gen.core.particle import PROTON
from linac_gen.elements.field_map import FieldMap
from linac_gen.io.field_map_data import FieldMapData, FieldChannel
from linac_gen.io.tracewin_geom import Channel


def _rf_1d_cav(Ez_peak_MVm=1.0, L_mm=100.0):
    z = np.linspace(0, L_mm, 101)
    Ez = np.full_like(z, Ez_peak_MVm)  # uniform for clean energy-gain sum
    fd = FieldMapData(z=z, frequency=352.21)
    fd.channels[Channel.RF_E] = FieldChannel(geometry=1, z=z, Fz=Ez)
    return fd


def _static_1d_sol(Bz_T=0.5, L_mm=200.0):
    z = np.linspace(0, L_mm, 201)
    Bz = np.full_like(z, Bz_T)
    fd = FieldMapData(z=z, frequency=0.0)
    fd.channels[Channel.STAT_B] = FieldChannel(geometry=1, z=z, Fz=Bz)
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


def test_1d_rf_cavity_accelerates_on_crest():
    fd = _rf_1d_cav(1.0, 100.0)
    cav = FieldMap(name="C", length=100.0, field_data=fd, scale=1.0,
                   phase=0.0, frequency=352.21, n_steps=100)
    ref = ReferenceParticle(species=PROTON, w_kin=5.0, frequency=352.21)
    W0 = ref.w_kin
    cav.advance_ref(ref)
    assert ref.w_kin > W0


def test_1d_static_solenoid_rotates_offaxis_particle():
    """A 1-D static Bz solenoid: on-axis v×Bz couples x' to y'."""
    fd = _static_1d_sol(0.5, 200.0)
    sol = FieldMap(name="SOL", length=200.0, field_data=fd, scale=1.0,
                   phase=0.0, frequency=0.0, n_steps=200)
    beam = _single(xp_mrad=1.0)
    _track(sol, beam)
    assert abs(beam.particles[0, 3]) > 0.05, beam.particles[0]


def test_static_field_is_phase_independent():
    fd = _static_1d_sol(0.3, 200.0)
    a = FieldMap(name="A", length=200.0, field_data=fd, scale=1.0,
                 phase=0.0,   frequency=352.21, n_steps=50)
    b = FieldMap(name="B", length=200.0, field_data=fd, scale=1.0,
                 phase=180.0, frequency=352.21, n_steps=50)
    ba = _single(xp_mrad=1.0);  bb = _single(xp_mrad=1.0)
    _track(a, ba); _track(b, bb)
    np.testing.assert_allclose(ba.particles, bb.particles, rtol=1e-12)


def test_paraxial_Er_recovers_1d_offaxis_focus():
    """1-D RF cavity with cos(πz/L) axial profile: a slightly off-axis
    particle must feel a non-zero transverse kick (via -r/2·dEz/dz)."""
    z = np.linspace(0, 100, 101)
    Ez = np.cos(np.pi * z / 100) * 1.0  # MV/m
    fd = FieldMapData(z=z, frequency=352.21)
    fd.channels[Channel.RF_E] = FieldChannel(geometry=1, z=z, Fz=Ez)
    cav = FieldMap(name="CAV", length=100.0, field_data=fd, scale=1.0,
                   phase=0.0, frequency=352.21, n_steps=100)
    beam = _single(x_mm=2.0)
    _track(cav, beam)
    xp = beam.particles[0, 1]
    # Old code: xp was 1000× too small (bug #2), so this was ~1e-9 mrad.
    # New code: magnitude of paraxial kick for 1 MV/m, 2mm, 100mm cavity
    # is ~0.1 mrad order.
    assert abs(xp) > 1e-3, f"Expected paraxial kick ~0.1 mrad, got {xp}"


def test_static_B_conserves_reference_energy():
    """Pure static-B map must not change reference particle energy."""
    fd = _static_1d_sol(Bz_T=1.0, L_mm=400.0)
    sol = FieldMap(name="SOL", length=400.0, field_data=fd, scale=1.0,
                   phase=0.0, frequency=0.0, n_steps=200)
    ref = ReferenceParticle(species=PROTON, w_kin=5.0, frequency=352.21)
    W0 = ref.w_kin
    sol.advance_ref(ref)
    np.testing.assert_allclose(ref.w_kin, W0, rtol=1e-12)


def test_rf_electric_decelerates_on_trough():
    """On-trough (phase=180°) RF-E cavity should decelerate the reference."""
    fd = _rf_1d_cav(Ez_peak_MVm=1.0, L_mm=200.0)
    cav = FieldMap(name="CAV", length=200.0, field_data=fd, scale=1.0,
                   phase=180.0, frequency=352.21, n_steps=100)
    ref = ReferenceParticle(species=PROTON, w_kin=5.0, frequency=352.21)
    W0 = ref.w_kin
    cav.advance_ref(ref)
    assert ref.w_kin < W0


def test_geometry9_quad_gives_transverse_kick():
    """geometry=9 (quad gradient) must give equal and opposite xp/yp kicks.

    A normal quadrupole with G>0 focuses x (kicks inward for x>0) and
    defocuses y (kicks outward for y>0).  The Lorentz formula with
    Bx=G·y, By=G·x gives:
      dxp ~ -charge·ds·By / (γβ·mc²) = -charge·ds·G·x / (γβ·mc²) < 0 for x>0
      dyp ~ +charge·ds·Bx / (γβ·mc²) = +charge·ds·G·y / (γβ·mc²) > 0 for y>0
    """
    L_mm = 200.0
    z = np.linspace(0, L_mm, 201)
    G = np.full_like(z, 5.0)   # 5 T/m gradient
    fd = FieldMapData(z=z, frequency=0.0)
    fd.channels[Channel.STAT_B] = FieldChannel(geometry=9, z=z, Fz=G)

    quad = FieldMap(name="Q", length=L_mm, field_data=fd, scale=1.0,
                    phase=0.0, frequency=0.0, n_steps=200)
    # x=2 mm, y=0 → should feel -x kick from By=G·x
    beam_x = _single(x_mm=2.0)
    _track(quad, beam_x)
    # x=0, y=2 mm → should feel +y kick from Bx=G·y
    beam_y = _single(y_mm=2.0)
    quad2 = FieldMap(name="Q2", length=L_mm, field_data=fd, scale=1.0,
                     phase=0.0, frequency=0.0, n_steps=200)
    _track(quad2, beam_y)

    xp_kick = beam_x.particles[0, 1]
    yp_kick = beam_y.particles[0, 3]
    # Both kicks should be non-zero
    assert abs(xp_kick) > 1e-3, f"Expected quad x-kick, got xp={xp_kick}"
    assert abs(yp_kick) > 1e-3, f"Expected quad y-kick, got yp={yp_kick}"
    # For G>0 and q=+1: x-particle focuses (xp<0), y-particle defocuses (yp>0)
    assert xp_kick < 0, f"Quad should focus x, got xp={xp_kick}"
    assert yp_kick > 0, f"Quad should defocus y, got yp={yp_kick}"


def test_2d_cyl_rf_e_accelerates():
    """2-D cyl geometry=4 RF_E channel: reference should accelerate on crest."""
    L_mm = 100.0
    nz, nr = 51, 11
    z = np.linspace(0, L_mm, nz)
    r = np.linspace(0, 20.0, nr)
    Ez = np.ones((nz, nr)) * 1.0  # uniform 1 MV/m
    Fr = np.zeros((nz, nr))
    fd = FieldMapData(z=z, frequency=352.21)
    fd.channels[Channel.RF_E] = FieldChannel(
        geometry=4, z=z, r=r, Fz=Ez, Fr=Fr,
    )
    cav = FieldMap(name="CAV2D", length=L_mm, field_data=fd, scale=1.0,
                   phase=0.0, frequency=352.21, n_steps=50)
    ref = ReferenceParticle(species=PROTON, w_kin=5.0, frequency=352.21)
    W0 = ref.w_kin
    cav.advance_ref(ref)
    assert ref.w_kin > W0


def test_2d_cyl_offaxis_radial_kick():
    """2-D cyl geometry=4: off-axis particle gets transverse kick from Fr."""
    L_mm = 100.0
    nz, nr = 101, 21
    z = np.linspace(0, L_mm, nz)
    r = np.linspace(0, 20.0, nr)
    # paraxial Er = -(r/2) * dEz/dz for cos(pi*z/L) axial profile
    E0 = 1.0  # MV/m
    ez_axis = E0 * np.cos(np.pi * z / L_mm)
    dez_dz = np.gradient(ez_axis, z)
    Ez = np.outer(np.ones(nz), np.ones(nr)) * ez_axis[:, np.newaxis]
    Fr = np.zeros((nz, nr))
    for jr in range(nr):
        Fr[:, jr] = -0.5 * r[jr] * dez_dz
    fd = FieldMapData(z=z, frequency=352.21)
    fd.channels[Channel.RF_E] = FieldChannel(
        geometry=4, z=z, r=r, Fz=Ez, Fr=Fr,
    )
    cav = FieldMap(name="CAV2D", length=L_mm, field_data=fd, scale=1.0,
                   phase=0.0, frequency=352.21, n_steps=100)
    beam = _single(x_mm=2.0)
    _track(cav, beam)
    xp = beam.particles[0, 1]
    assert abs(xp) > 1e-3, f"Expected 2-D cyl transverse kick ~0.1 mrad, got {xp}"
