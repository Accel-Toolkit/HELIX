"""SET_SYNC_PHASE calibration across the +-180 deg wrap of psi.

The calibration solves psi = Psi(theta_s - psi) by under-relaxed iteration,
where Psi is the phase of the cavity's voltage integral.  psi and Psi are
angles.  Before 2026-10-17 the relaxation averaged them linearly, so when
psi sat near +-180 deg and the beta feedback put successive Psi on both
sides of the wrap, the iterate jumped far away, never converged, and the
cavity was left phased ~100 deg wrong without a warning (a half-wave
resonator entered at 3.69 MeV decelerated the beam).

Pinned on a synthetic two-gap map whose psi crosses the wrap between 3.70
and 3.74 MeV (protons, 162.5 MHz, 20 MV/m), for the 3-D and the 1-D
field-map elements (NCells and SUPERPOSE_MAP share the same helpers).
"""
from __future__ import annotations

import numpy as np
import pytest

from linac_gen.core.particle import PROTON
from linac_gen.core.reference import ReferenceParticle
from linac_gen.elements.base import angle_nearest, wrap_deg
from linac_gen.elements.field_map import FieldMap
from linac_gen.elements.field_map_3d import FieldMap3D
from linac_gen.io.field_map_data import FieldChannel, FieldMapData
from linac_gen.io.tracewin_geom import Channel

L_MM, F_MHZ, THETA_S = 250.0, 162.5, -30.0
W_ACROSS_WRAP = (3.70, 3.72, 3.74)


def _two_gap_ez(z, e0=20.0, gap=12.0, sep=90.0):
    zc = L_MM / 2
    return e0 * (np.exp(-0.5 * ((z - zc + sep / 2) / gap) ** 2)
                 - np.exp(-0.5 * ((z - zc - sep / 2) / gap) ** 2))


def _cavity_3d():
    x = y = np.linspace(-10.0, 10.0, 5)
    z = np.linspace(0.0, L_MM, 201)
    ez = np.broadcast_to(_two_gap_ez(z)[None, None, :], (5, 5, z.size)).copy()
    data = FieldMapData(x=x, y=y, z=z, Ez=ez, symmetry="3d")
    return FieldMap3D(name="G2", length=L_MM, field_data=data, phase=THETA_S,
                      frequency=F_MHZ, n_steps=200, p_flag=1)


def _cavity_1d():
    z = np.linspace(0.0, L_MM, 201)
    fd = FieldMapData(z=z, frequency=F_MHZ)
    fd.channels[Channel.RF_E] = FieldChannel(geometry=1, z=z, Fz=_two_gap_ez(z))
    return FieldMap(name="G1", length=L_MM, field_data=fd, phase=THETA_S,
                    frequency=F_MHZ, n_steps=200, p_flag=1)


def _ref(w):
    return ReferenceParticle(species=PROTON, w_kin=float(w), frequency=F_MHZ)


def _psi_of(cav, ref, phi_in_deg):
    re_v, im_v = cav._probe_voltage_integral(ref, phi_in_deg)
    return float(np.degrees(np.arctan2(im_v, re_v)))


def _pre_fix_iteration(cav, ref):
    """The calibration loop as it was before the fix (linear averaging)."""
    psi = 0.0
    for _ in range(30):
        new = _psi_of(cav, ref, THETA_S - psi)
        if abs(new - psi) < 0.005:
            return new
        psi = 0.3 * psi + 0.7 * new
    return psi


def test_angle_helpers():
    assert angle_nearest(179.0, -179.0) == -181.0
    assert angle_nearest(-179.0, 179.0) == 181.0
    assert angle_nearest(-170.0, 545.0) == 550.0
    for a, ref in ((10.0, 0.0), (123.456789, 100.0), (-179.9, -1.0)):
        assert angle_nearest(a, ref) == a          # untouched within 180 deg
    assert wrap_deg(181.0) == -179.0
    assert wrap_deg(-181.0) == 179.0
    assert wrap_deg(541.0) == -179.0
    for a in (180.0, -180.0, 37.25, -0.0):
        assert wrap_deg(a) == a                    # untouched inside [-180, 180]


@pytest.mark.parametrize("make", [_cavity_3d, _cavity_1d], ids=["3d", "1d"])
@pytest.mark.parametrize("w", W_ACROSS_WRAP)
def test_calibration_converges_across_the_wrap(make, w):
    cav, ref = make(), _ref(w)
    cav._calibrate_sync_phase(ref)
    psi = cav._sync_offset_deg
    assert -180.0 <= psi <= 180.0
    assert abs(abs(psi) - 180.0) < 5.0             # the case really sits at the wrap
    residual = wrap_deg(_psi_of(cav, ref, THETA_S - psi) - psi)
    assert abs(residual) < 0.005                   # psi = Psi(theta_s - psi)


@pytest.mark.parametrize("make", [_cavity_3d, _cavity_1d], ids=["3d", "1d"])
def test_the_cases_defeat_the_pre_fix_iteration(make):
    """Guard on the fixture itself: the old loop leaves a phase that
    violates the fixed point by more than a degree in at least one case."""
    bad = 0
    for w in W_ACROSS_WRAP:
        cav, ref = make(), _ref(w)
        psi = _pre_fix_iteration(cav, ref)
        if abs(wrap_deg(_psi_of(cav, ref, THETA_S - psi) - psi)) > 1.0:
            bad += 1
    assert bad >= 1


def test_energy_gain_is_continuous_through_the_wrap():
    """The reference's gain at the calibrated phase changes smoothly with the
    entry energy while psi passes through +-180 deg (the bug made it drop)."""
    ws = np.arange(3.60, 3.861, 0.02)
    gains = []
    for w in ws:
        cav, ref = _cavity_3d(), _ref(w)
        cav.advance_ref(ref)
        gains.append(ref.w_kin - w)
    gains = np.asarray(gains)
    assert np.all(gains > 0)
    assert np.max(np.abs(np.diff(gains))) < 0.02 * np.max(gains)
