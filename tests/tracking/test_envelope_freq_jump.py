# tests/tracking/test_envelope_freq_jump.py
"""Envelope-mode RF frequency-jump Σ transform.

Locks the exact ``Σ → D Σ Dᵀ`` rescale (D = diag(1,1,1,1,ratio,1)) applied
when the beam crosses an RF frequency transition:

* for an UNCOUPLED beam it is bit-identical to rescaling the longitudinal
  2×2 block alone (so it cannot move any existing baseline / TraceWin
  validation — every PIP-II section is uncoupled at its 162.5→325→650
  jumps);
* for a transversely–longitudinally COUPLED beam it also rescales the
  phase↔transverse cross-moments, keeping the transform exact;
* projected σ / ε / Twiss and the SC kick (diagonal-only) are untouched
  either way.

Both the transform helper and its integration into ``EnvelopeSolver`` are
exercised, so a regression that dropped the cross-moment rescale (or the
helper call) fails here.
"""
import numpy as np
import pytest

from linac_gen.core.particle import PROTON
from linac_gen.core.reference import ReferenceParticle
from linac_gen.core.lattice import Lattice
from linac_gen.elements.drift import Drift
from linac_gen.elements.rf_gap import RFGap
from linac_gen.tracking.envelope import (
    EnvelopeSolver, _build_sigma_matrix, _rescale_sigma_for_freq_jump,
    _sigma_to_twiss,
)

# Non-power-of-2 ratios included on purpose: bit-identity for uncoupled
# beams must not rely on 162.5→325 happening to be exactly ×2.
RATIOS = [2.0, 325.0 / 162.5, 650.0 / 325.0, 1.3, 0.5]


def _uncoupled_sigma():
    return _build_sigma_matrix(0.5, 2.0, 0.25, -0.3, 1.5, 0.22,
                               0.1, 3.0, 0.40)


def _coupled_sigma(rho_x=0.5, rho_y=0.3):
    S = _uncoupled_sigma()
    S[4, 0] = S[0, 4] = rho_x * np.sqrt(S[0, 0] * S[4, 4])
    S[4, 2] = S[2, 4] = rho_y * np.sqrt(S[2, 2] * S[4, 4])
    return S


def _D(ratio):
    return np.diag([1.0, 1.0, 1.0, 1.0, ratio, 1.0])


# ---------------------------------------------------------------------------
# The transform helper
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("ratio", RATIOS)
def test_uncoupled_bit_identical_to_longitudinal_only(ratio):
    """Uncoupled beam: the full transform equals the old longitudinal-only
    rescale to the BIT — cannot perturb any existing result."""
    S = _uncoupled_sigma()
    longitudinal_only = S.copy()
    longitudinal_only[4, 4] *= ratio * ratio
    longitudinal_only[4, 5] *= ratio
    longitudinal_only[5, 4] *= ratio
    got = _rescale_sigma_for_freq_jump(S, ratio)
    assert np.array_equal(got, longitudinal_only)      # exact, not approx


@pytest.mark.parametrize("ratio", RATIOS)
def test_coupled_equals_full_DSD(ratio):
    """Coupled beam: the transform is exactly D Σ Dᵀ."""
    S = _coupled_sigma()
    expected = _D(ratio) @ S @ _D(ratio).T
    np.testing.assert_allclose(_rescale_sigma_for_freq_jump(S, ratio),
                               expected, rtol=0, atol=1e-15)


@pytest.mark.parametrize("ratio", RATIOS)
def test_projected_observables_untouched(ratio):
    """Transverse σ/ε and the diagonal the SC kick reads are unaffected by
    the cross-moment rescale (coupled vs uncoupled give identical
    projections); σ_φ scales by ratio as physics demands."""
    Sc, Su = _coupled_sigma(), _uncoupled_sigma()
    Rc = _rescale_sigma_for_freq_jump(Sc, ratio)
    Ru = _rescale_sigma_for_freq_jump(Su, ratio)
    # transverse block: coupling must not leak into projected x/y sizes
    for i in (0, 1, 2, 3):
        assert Rc[i, i] == pytest.approx(Ru[i, i], rel=1e-14)
    assert _sigma_to_twiss(Rc, "x")["emit"] == pytest.approx(
        _sigma_to_twiss(Su, "x")["emit"], rel=1e-14)
    # SC-critical σ_φ² scales by ratio² (the whole point of the rescale)
    assert Rc[4, 4] == pytest.approx((ratio ** 2) * Sc[4, 4], rel=1e-14)
    assert Ru[4, 4] == pytest.approx((ratio ** 2) * Su[4, 4], rel=1e-14)


def test_symmetry_preserved():
    S = _coupled_sigma()
    R = _rescale_sigma_for_freq_jump(S, 2.0)
    np.testing.assert_allclose(R, R.T, rtol=0, atol=1e-15)


def test_does_not_mutate_input():
    S = _coupled_sigma()
    S0 = S.copy()
    _rescale_sigma_for_freq_jump(S, 2.0)
    assert np.array_equal(S, S0)


# ---------------------------------------------------------------------------
# Integration: the solver applies the exact transform at a real jump
# ---------------------------------------------------------------------------

def _solver_with_gap(entrance_freq, gap_freq, current):
    """A Drift + RFGap lattice; the gap carries `gap_freq` so crossing it
    triggers the frequency-jump rescale when the ref is at entrance_freq."""
    lat = Lattice()
    lat.add(Drift("D0", 50.0))
    lat.add(RFGap("GAP", voltage=0.3, phase=-25.0, frequency=gap_freq,
                  ttf=0.85))
    ref = ReferenceParticle(species=PROTON, w_kin=3.0, frequency=entrance_freq)
    initial = _build_sigma_matrix(0.5, 2.0, 0.25, -0.3, 1.5, 0.22,
                                  0.1, 3.0, 0.40)
    twiss = dict(alpha_x=0.5, beta_x=2.0, emit_x=0.25,
                 alpha_y=-0.3, beta_y=1.5, emit_y=0.22,
                 alpha_z=0.1, beta_z=3.0, emit_z=0.40)
    return EnvelopeSolver(lat, ref, twiss, current=current), lat, initial


@pytest.mark.parametrize("current", [0.0, 15.0])
def test_solver_freq_jump_is_full_transform(current):
    """End-to-end: propagating a coupled Σ through the gap with the ref at
    the OLD frequency (rescale fires) must equal propagating D Σ Dᵀ with the
    ref already at the NEW frequency (rescale is a no-op, only optics act).
    This holds the solver's integration of the exact transform — with SC on
    too, since the SC kick reads the (identical) diagonal σ."""
    ratio = 2.0
    gap_freq, entrance_freq = 325.0, 325.0 / ratio
    Sc = _coupled_sigma()

    # code path: ref at old freq → the gap triggers the Σ rescale
    s_code, _, _ = _solver_with_gap(entrance_freq, gap_freq, current)
    gap = s_code.lattice.elements[1]
    assert s_code._ref.frequency == pytest.approx(entrance_freq)
    out_code = s_code._propagate_element(gap, Sc.copy())
    assert s_code._ref.frequency == pytest.approx(gap_freq)   # jump happened

    # reference path: ref already at new freq (no rescale), feed D Σ Dᵀ
    s_ref, _, _ = _solver_with_gap(gap_freq, gap_freq, current)
    gap_ref = s_ref.lattice.elements[1]
    pre = _D(ratio) @ Sc @ _D(ratio).T
    out_ref = s_ref._propagate_element(gap_ref, pre.copy())

    np.testing.assert_allclose(out_code, out_ref, rtol=1e-10, atol=1e-12)


def test_solver_uncoupled_matches_longitudinal_only_run():
    """A full uncoupled envelope run through a frequency jump is bit-for-bit
    reproducible by the longitudinal-only rescale — the guarantee that this
    change moves nothing for PIP-II-style (uncoupled) beams."""
    ratio = 2.0
    s, lat, _ = _solver_with_gap(325.0 / ratio, 325.0, current=10.0)
    gap = lat.elements[1]
    Su = _uncoupled_sigma()
    out_full = s._propagate_element(gap, Su.copy())

    # redo with a hand-rolled longitudinal-only rescale in place of the full
    s2, lat2, _ = _solver_with_gap(325.0 / ratio, 325.0, current=10.0)
    gap2 = lat2.elements[1]
    import linac_gen.tracking.envelope as E
    orig = E._rescale_sigma_for_freq_jump
    def longitudinal_only(sigma, r):
        s_ = sigma.copy()
        s_[4, 4] *= r * r; s_[4, 5] *= r; s_[5, 4] *= r
        return s_
    E._rescale_sigma_for_freq_jump = longitudinal_only
    try:
        out_long = s2._propagate_element(gap2, Su.copy())
    finally:
        E._rescale_sigma_for_freq_jump = orig
    assert np.array_equal(out_full, out_long)


# ---------------------------------------------------------------------------
# frequency_offset asymmetry warning (multibunch M3): the envelope field-map
# carry-over uses element.frequency while MP's advance_ref carries the
# EFFECTIVE frequency — warn the user on both envelope paths (SC on and the
# no-SC substep path — dual-regime rule).
# ---------------------------------------------------------------------------

def _rf_fieldmap3d(L_mm=100.0, freq=162.5):
    from linac_gen.elements.field_map_3d import FieldMap3D
    from linac_gen.io.field_map_data import FieldMapData, FieldChannel
    from linac_gen.io.tracewin_geom import Channel
    n, nz = 3, 11
    x = np.linspace(-20.0, 20.0, n)
    y = np.linspace(-20.0, 20.0, n)
    z = np.linspace(0.0, L_mm, nz)
    fd = FieldMapData(z=z, frequency=freq)
    fd.channels[Channel.RF_E] = FieldChannel(
        geometry=7, x=x, y=y, z=z,
        Fx=np.zeros((n, n, nz)), Fy=np.zeros((n, n, nz)),
        Fz=np.full((n, n, nz), 1.0e5))
    return FieldMap3D(name="RFM", length=L_mm, field_data=fd,
                      scale=1.0, n_steps=10)


def _rf_map_solver(fm, current, **kw):
    lat = Lattice()
    lat.add(fm)
    ref = ReferenceParticle(species=PROTON, w_kin=3.0, frequency=162.5)
    twiss = dict(alpha_x=0.0, beta_x=2.0, emit_x=0.25,
                 alpha_y=0.0, beta_y=2.0, emit_y=0.25,
                 alpha_z=0.0, beta_z=3.0, emit_z=0.30)
    return EnvelopeSolver(lat, ref, twiss, current=current, **kw)


@pytest.mark.parametrize("current,record_substeps",
                         [(5.0, False), (0.0, True)])
def test_envelope_warns_frequency_offset_ignored(current, record_substeps):
    fm = _rf_fieldmap3d()
    fm.frequency_offset = 1.0
    solver = _rf_map_solver(fm, current, record_substeps=record_substeps)
    with pytest.warns(UserWarning, match="frequency_offset"):
        solver.run()


def test_envelope_no_offset_no_frequency_warning():
    import warnings as _w
    solver = _rf_map_solver(_rf_fieldmap3d(), current=5.0)
    with _w.catch_warnings(record=True) as rec:
        _w.simplefilter("always")
        solver.run()
    assert not [r for r in rec if "frequency_offset" in str(r.message)]
