"""Vane-field campaign (2026-07-30): the smooth TW calibration and the
surface/fit machinery that produced it.

Evidence chain pinned here:
  * an FD Laplace solve of the true constant-Tc electrode (prototype,
    validated to 0.1-0.7 % on the exact two-term surface control)
    predicts quad/accel corrections of the same SIGN and cell-trend as
    the ones TW's own matrices demand — the physical origin;
  * the constant-Tc model overshoots (Toutatis's reconstruction is not
    a pure constant-Tc arc), so magnitudes are calibrated to the 203
    ground-truth matrices — the fnalscl-T(β) mode-faithful precedent;
  * RAW per-cell fitted corrections make the BEAM worse (envelope y
    7.4 → 16 %): cell-to-cell jitter breaks AG coherence.  Smoothness
    is load-bearing; only the A10-parameterised smooth form ships.

Results with the calibration: envelope σ 1.1 % (x) / 1.7 % (y) vs the
vane-based TW export (its own-matrix floor is ~4 %), per-cell median
1.25 %, ramp exact, MP transmission/emittances unchanged (halo-driven).
"""
from __future__ import annotations

import numpy as np
import pytest

from tests.rfq import ref_loaders as rl


def test_tw_calibration_disable_flag():
    """The documented opt-out must really give the uncalibrated model."""
    import linac_gen.elements.rfq_coefficients as rc
    assert rc.tw_calibration(0.134) != (1.0, 1.0)
    rc.TW_CALIBRATION_ENABLED = False
    try:
        assert rc.tw_calibration(0.134) == (1.0, 1.0)
    finally:
        rc.TW_CALIBRATION_ENABLED = True
    assert rc.tw_calibration(0.134) != (1.0, 1.0)


def test_tw_calibration_shape():
    from linac_gen.elements.rfq_coefficients import tw_calibration
    # identity at the m→1 / A10→0 end and clamped beyond the last node
    assert tw_calibration(0.0) == (1.0, 1.0)
    q_end, a_end = tw_calibration(0.95)
    assert q_end == pytest.approx(1.0)
    assert a_end == pytest.approx(1.025)
    # the calibrated dip: quad minimum at A10 ≈ 0.134
    q_mid, a_mid = tw_calibration(0.134)
    assert q_mid == pytest.approx(0.990, abs=1e-9)
    assert a_mid == pytest.approx(0.984, abs=1e-9)
    # smoothness: monotone segments, no excursions beyond node values
    a10s = np.linspace(0, 0.9, 200)
    qs = np.array([tw_calibration(a)[0] for a in a10s])
    accs = np.array([tw_calibration(a)[1] for a in a10s])
    assert qs.min() >= 0.990 - 1e-12 and qs.max() <= 1.0 + 1e-12
    assert accs.min() >= 0.984 - 1e-12 and accs.max() <= 1.025 + 1e-12


def test_calibration_does_not_touch_longitudinal_channel(pxie_deck,
                                                         env_nosc):
    """The ramp must stay EXACT — the calibration scopes to the
    transverse quad/defocus only (K1/K2/E_z stay on the card A10)."""
    from linac_gen.core.particle import H_MINUS
    from linac_gen.core.reference import ReferenceParticle
    from linac_gen.elements.rfq_cell import RfqCell
    cells = [e for e in pxie_deck.elements if isinstance(e, RfqCell)]
    g0 = 1.0 + np.interp(1.9561, env_nosc["s_m"], env_nosc["gam1"])
    ref = ReferenceParticle(species=H_MINUS,
                            w_kin=(g0 - 1) * H_MINUS.mass,
                            frequency=162.5)
    saved = [c.field_model for c in cells]
    for c in cells:
        c.field_model = "tw2term"
    try:
        for c in cells:
            c.advance_ref(ref)
    finally:
        for c, fm in zip(cells, saved):
            c.field_model = fm
    assert ref.w_kin == pytest.approx(1.955717, abs=5e-6)


def test_vane_surface_sampler_geometry():
    from linac_gen.elements.rfq_vane_surface import (
        card_aperture_functions, sample_cell_surface)
    ax, ay = card_aperture_functions(5.576, 0.6, 36.47, -2)
    pts, v = sample_cell_surface(ax, ay, 36.47, 4.182,
                                 n_z=8, n_theta=9)
    assert pts.shape == (8 * 4 * 9, 3)
    # x-pair points carry +1, y-pair −1, in equal numbers
    assert (v > 0).sum() == (v < 0).sum()
    # the tip point (alpha = 0, middle of each fan) reproduces a_v(z)
    tips_x = pts[(v > 0)][4::9]          # middle of each 9-point fan
    for p in tips_x[:8]:
        assert abs(abs(p[0]) - ax(p[2])) < 1e-9
        assert abs(p[1]) < 1e-12
    # arc curvature: all fan points lie exactly Tc from the arc centre
    zi = pts[0, 2]
    fan = pts[:9]
    cx = ax(zi) + 4.182
    rr = np.sqrt((fan[:, 0] - cx) ** 2 + fan[:, 1] ** 2)
    np.testing.assert_allclose(rr, 4.182, rtol=1e-12)


def test_surface_fit_recovers_quad_reduction():
    """The 8-term surface fit on the true Tc arc must show the quad
    reduction (A(1,0) < 1) that motivated the calibration — and stay
    ≈ 1 for the near-unmodulated geometry only in the A10→0,
    Tc→matched limit (not tested here; FD control covers it)."""
    from linac_gen.elements.rfq_vane_surface import (
        card_aperture_functions, sample_cell_surface)
    from linac_gen.elements.vane_rfq_8term_full import fit_cell_multipoles
    ax, ay = card_aperture_functions(5.576, 0.6, 36.47, -2)
    pts, v = sample_cell_surface(ax, ay, 36.47, 4.182)
    cc = fit_cell_multipoles(pts, v, 0.0, 36.47, 5.576, 60000.0)
    assert 0.90 < cc.A[0] < 1.0          # quad reduced by the Tc arc
    assert cc.residual < 0.05
    assert np.isfinite(cc.cond_number)
