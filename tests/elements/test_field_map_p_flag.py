"""p_flag=1 means θᵢ is absolute phase — independent of ref.phi_s.

Per TraceWin manual §18036-18042.
"""
import numpy as np
import pytest

from linac_gen.core.reference import ReferenceParticle
from linac_gen.core.particle import PROTON
from linac_gen.elements.field_map import FieldMap
from linac_gen.io.field_map_data import FieldMapData, FieldChannel
from linac_gen.io.tracewin_geom import Channel


def _rf_1d_cav(Ez_peak_MVm=1.0, L_mm=100.0):
    """Create a 1-D RF cavity with uniform field."""
    z = np.linspace(0, L_mm, 101)
    Ez = np.full_like(z, Ez_peak_MVm)
    fd = FieldMapData(z=z, frequency=352.21)
    fd.channels[Channel.RF_E] = FieldChannel(geometry=1, z=z, Fz=Ez)
    return fd


def _energy_gain(p_flag: int, phase_deg: float, phi_s_start: float) -> float:
    """Return energy gain (MeV) when tracking ref through cavity."""
    fd = _rf_1d_cav(1.0, 100.0)
    cav = FieldMap(name="C", length=100.0, field_data=fd,
                   scale=1.0, ke=1.0, kb=1.0,
                   phase=phase_deg, frequency=352.21,
                   n_steps=50, p_flag=p_flag)
    ref = ReferenceParticle(species=PROTON, w_kin=5.0, frequency=352.21)
    ref.phi_s = phi_s_start
    W0 = ref.w_kin
    cav.advance_ref(ref)
    return ref.w_kin - W0


def test_p_flag_relative_sees_ref_phi_s():
    """With p_flag=0, changing ref.phi_s changes the effective phase."""
    dW_a = _energy_gain(p_flag=0, phase_deg=0.0, phi_s_start=0.0)
    dW_b = _energy_gain(p_flag=0, phase_deg=0.0, phi_s_start=90.0)
    assert not np.isclose(dW_a, dW_b, rtol=1e-6), (
        f"relative phase should change result: dW_a={dW_a}, dW_b={dW_b}"
    )


def test_p_flag_absolute_ignores_ref_phi_s():
    """With p_flag=1, the phasor uses only self.phase; ref.phi_s is ignored."""
    dW_a = _energy_gain(p_flag=1, phase_deg=0.0, phi_s_start=0.0)
    dW_b = _energy_gain(p_flag=1, phase_deg=0.0, phi_s_start=90.0)
    np.testing.assert_allclose(dW_a, dW_b, rtol=1e-12)
