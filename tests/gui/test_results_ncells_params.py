"""NCELLS in the Results-tab lattice-parameter charts (V₀ / E_acc / φ_s).

Regression for "the RF voltage / synchronous-phase popups show 'no RF
elements' on an NCELLS deck": NCELLS now feeds all three charts.  The φ_s
chart's P=1 behavior is deliberate: raw θs is a meaningless RF-clock ramp
(62/242/422/… on fnalscl), so P=1 cavities plot the RUN-RESOLVED phase the
beam saw at gap 1 — and nothing before a run.
"""
from __future__ import annotations

import numpy as np
import pytest

from linac_gen_gui.interphase.tabs.results_tab import (
    _ncells_v0_MV, _ncells_phase_deg, _rf_phase_value,
)
from linac_gen.elements.ncells import NCells
from linac_gen.core.particle import H_MINUS
from linac_gen.core.reference import ReferenceParticle


def _mk(**kw):
    base = dict(mode=1, n_cells=4, beta_g=0.5, eot_v_per_m=5e6,
                theta_s_deg=0.0, aperture_mm=15, p_flag=0,
                frequency_mhz=650.0)
    base.update(kw)
    return NCells("NC", **base)


def test_v0_is_sum_of_gap_voltages():
    nc = _mk()
    lc = nc._interior_cell_length(0.5)               # mm
    expected = 4 * 5e6 * (lc * 1e-3) * 1e-6          # MV
    assert _ncells_v0_MV(nc) == pytest.approx(expected, rel=1e-12)
    # ERROR_CAV amplitude factor shows up (mirrors the errored-RFGap chart)
    nc.voltage_rel = 0.10
    assert _ncells_v0_MV(nc) == pytest.approx(1.10 * expected, rel=1e-12)


def test_v0_none_for_unresolved_betaG_zero_then_value_after_run():
    nc = _mk(beta_g=0.0, n_cells=8)
    assert _ncells_v0_MV(nc) is None                 # geometry not resolved yet
    ref = ReferenceParticle(species=H_MINUS, w_kin=150.0, frequency=650.0)
    nc.advance_ref(ref)
    v = _ncells_v0_MV(nc)
    assert v is not None and v > 0


def test_phase_relative_and_sync_show_theta_wrapped():
    assert _ncells_phase_deg(_mk(theta_s_deg=-30.0)) == pytest.approx(-30.0)
    assert _ncells_phase_deg(_mk(theta_s_deg=190.0)) == pytest.approx(-170.0)
    nc = _mk(theta_s_deg=-25.0, p_flag=1, sync_phase=True)
    assert _ncells_phase_deg(nc) == pytest.approx(-25.0)   # sync wins over P=1


def test_phase_p1_none_before_run_resolved_after():
    nc = _mk(p_flag=1, theta_s_deg=62.0)
    assert _ncells_phase_deg(nc) is None             # raw ramp never shown
    ref = ReferenceParticle(species=H_MINUS, w_kin=150.0, frequency=650.0)
    ref.phi_s = 0.0
    nc.reset_run_state()
    nc.advance_ref(ref)
    got = _ncells_phase_deg(nc)
    assert got is not None
    expected = ((nc._phi_s_at_gap1 - 62.0 + 180.0) % 360.0) - 180.0
    assert got == pytest.approx(expected, abs=1e-9)
    assert -180.0 <= got < 180.0


def test_phase_p1_crest_theta_resolves_to_zero():
    """If θs equals the gap-1 clock reading, the beam sees crest (0°)."""
    probe = _mk(p_flag=1, theta_s_deg=0.0)
    ref = ReferenceParticle(species=H_MINUS, w_kin=150.0, frequency=650.0)
    ref.phi_s = 0.0
    probe.reset_run_state()
    probe.advance_ref(ref)
    theta_crest = probe._phi_s_at_gap1 % 360.0
    nc = _mk(p_flag=1, theta_s_deg=theta_crest)
    ref2 = ReferenceParticle(species=H_MINUS, w_kin=150.0, frequency=650.0)
    ref2.phi_s = 0.0
    nc.reset_run_state()
    nc.advance_ref(ref2)
    assert _ncells_phase_deg(nc) == pytest.approx(0.0, abs=1e-9)


def test_rf_phase_dispatch_flat_phase_elements_unchanged():
    class _FakeGap:
        phase = -32.5
    assert _rf_phase_value(_FakeGap()) == pytest.approx(-32.5)
    assert _rf_phase_value(_mk(theta_s_deg=10.0)) == pytest.approx(10.0)
