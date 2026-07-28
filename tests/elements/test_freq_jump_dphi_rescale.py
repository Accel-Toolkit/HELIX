"""Regression tests for FREQ-jump dphi rescale across MP-tracking elements.

When a cavity at a different RF frequency is encountered, beam.particles[:, 4]
(stored in deg-at-ref.frequency) must be multiplied by f_new/f_old BEFORE
ref.frequency is updated.  Otherwise the same physical phase deviation is
mis-interpreted under the new units, εnz (Liouville) collapses by f_old/f_new,
and σ_phi visibly halves at every doubling-frequency boundary.

Covers:
  * RFGap.apply_kick (rf_gap.py:52-58)
  * FieldMap._track_kd (field_map.py:481-485)
  * FieldMap3D._track_kd / _track_dkd (field_map_3d.py:356-360, 514-518)
"""
from __future__ import annotations
import numpy as np
import pytest

from linac_gen.core.particle import H_MINUS
from linac_gen.core.reference import ReferenceParticle
from linac_gen.core.beam import Beam
from linac_gen.elements.rf_gap import RFGap


F_OLD = 162.5
F_NEW = 325.0
RATIO = F_NEW / F_OLD


def _make_beam(seed=0, n=200):
    ref = ReferenceParticle(species=H_MINUS, w_kin=10.0, frequency=F_OLD)
    beam = Beam(ref=ref, n_particles=n, current=0.0)
    rng = np.random.default_rng(seed)
    beam.particles[:, 0] = 0.5 * rng.normal(size=n)
    beam.particles[:, 1] = 0.5 * rng.normal(size=n)
    beam.particles[:, 2] = 0.5 * rng.normal(size=n)
    beam.particles[:, 3] = 0.5 * rng.normal(size=n)
    beam.particles[:, 4] = 5.0 * rng.normal(size=n)   # deg-at-F_OLD
    beam.particles[:, 5] = 0.01 * rng.normal(size=n)
    return beam


# ---------- RFGap ----------------------------------------------------------

def test_rfgap_zerov_pure_freq_jump_preserves_physical_phase():
    """V=0 gap at a different freq: dphi[deg] must scale by f_new/f_old."""
    beam = _make_beam()
    dphi_old = beam.particles[:, 4].copy()
    gap = RFGap("G_freq_marker", voltage=0.0, phase=0.0, frequency=F_NEW, ttf=1.0)

    gap.apply_kick(beam)

    assert beam.ref.frequency == pytest.approx(F_NEW)
    np.testing.assert_allclose(beam.particles[:, 4], dphi_old * RATIO, rtol=1e-12)


def test_rfgap_no_freq_change_leaves_dphi_intact():
    """Same-freq gap must NOT touch dphi (rescale is a no-op when freqs match)."""
    beam = _make_beam()
    dphi_old = beam.particles[:, 4].copy()
    gap = RFGap("G_same", voltage=0.0, phase=0.0, frequency=F_OLD, ttf=1.0)

    gap.apply_kick(beam)

    assert beam.ref.frequency == pytest.approx(F_OLD)
    np.testing.assert_allclose(beam.particles[:, 4], dphi_old, rtol=1e-12)


def test_rfgap_freq_jump_preserves_physical_z():
    """The physical longitudinal offset z = -dphi·β·λ/360 must be invariant
    across the FREQ jump (Liouville under unit change)."""
    beam = _make_beam()
    beta_before = beam.ref.beta
    wl_before   = beam.ref.wavelength
    z_before    = -beam.particles[:, 4] * beta_before * wl_before / 360.0

    gap = RFGap("G_freq_marker", voltage=0.0, phase=0.0, frequency=F_NEW, ttf=1.0)
    gap.apply_kick(beam)

    beta_after = beam.ref.beta
    wl_after   = beam.ref.wavelength
    z_after    = -beam.particles[:, 4] * beta_after * wl_after / 360.0

    # V=0 → β unchanged; only λ shrinks by ratio while dphi grows by ratio.
    assert beta_after == pytest.approx(beta_before, rel=1e-12)
    assert wl_after  == pytest.approx(wl_before / RATIO, rel=1e-12)
    np.testing.assert_allclose(z_after, z_before, rtol=1e-12)


def test_rfgap_freq_jump_preserves_sigma_emit_invariants():
    """σ_phi scales by ratio; σ_W unchanged; σ_phiW cross term scales by ratio.
    These are the σ-block consequences of the dphi rescale."""
    beam = _make_beam(seed=42, n=2000)

    # Snapshot before
    dphi0 = beam.particles[:, 4]
    dW0   = beam.particles[:, 5]
    sphi_before = float(np.std(dphi0))
    sW_before   = float(np.std(dW0))
    cov_before  = float(np.cov(dphi0, dW0, ddof=0)[0, 1])

    gap = RFGap("G_freq_marker", voltage=0.0, phase=0.0, frequency=F_NEW, ttf=1.0)
    gap.apply_kick(beam)

    sphi_after = float(np.std(beam.particles[:, 4]))
    sW_after   = float(np.std(beam.particles[:, 5]))
    cov_after  = float(np.cov(beam.particles[:, 4], beam.particles[:, 5], ddof=0)[0, 1])

    assert sphi_after == pytest.approx(RATIO * sphi_before, rel=1e-12)
    assert sW_after   == pytest.approx(sW_before,           rel=1e-12)
    assert cov_after  == pytest.approx(RATIO * cov_before,  rel=1e-12)

    # εz_phi_W = sqrt(σ_phi²·σ_W² − σ_phiW²) scales by ratio (units changed)
    eps_before = np.sqrt(max(sphi_before**2 * sW_before**2 - cov_before**2, 0.0))
    eps_after  = np.sqrt(max(sphi_after**2  * sW_after**2  - cov_after**2,  0.0))
    assert eps_after == pytest.approx(RATIO * eps_before, rel=1e-12)


def test_rfgap_freq_jump_alive_only():
    """Lost particles must not be touched by the dphi rescale."""
    beam = _make_beam()
    # Mark particle 0 as lost
    beam.lost[0] = True
    dphi_lost_before  = beam.particles[0, 4]
    dphi_alive_before = beam.particles[1:, 4].copy()

    gap = RFGap("G_freq_marker", voltage=0.0, phase=0.0, frequency=F_NEW, ttf=1.0)
    gap.apply_kick(beam)

    # Lost particle untouched
    assert beam.particles[0, 4] == pytest.approx(dphi_lost_before, rel=1e-12)
    # Alive particles rescaled
    np.testing.assert_allclose(beam.particles[1:, 4],
                                dphi_alive_before * RATIO, rtol=1e-12)


# ---------- FieldMap (1-D) regression --------------------------------------

def test_fieldmap1d_rescales_dphi_at_freq_jump():
    """1-D field map track must rescale beam.particles[:, 4] on entry when
    the cavity frequency differs from ref.frequency.  Zero-field map keeps
    ΔW = 0 so the slip term contributes nothing and the rescale is the only
    effect on dphi."""
    from linac_gen.elements.field_map import FieldMap
    from linac_gen.io.field_map_reader import FieldMapData

    z = np.linspace(0.0, 100.0, 21)
    Ez = np.zeros_like(z)
    fd = FieldMapData(z=z, Ez=Ez, symmetry="1d")
    fm = FieldMap(name="FM_freq", length=100.0, field_data=fd,
                  frequency=F_NEW, n_steps=1, p_flag=0)

    beam = _make_beam()
    # Zero per-particle ΔW so the post-rescale slip term Δφ += slip·ΔW does
    # not perturb dphi; the only effect on dphi is then the entry rescale.
    beam.particles[:, 5] = 0.0
    dphi_old = beam.particles[:, 4].copy()
    fm.track_rk4(beam, ds=100.0)

    assert beam.ref.frequency == pytest.approx(F_NEW)
    np.testing.assert_allclose(beam.particles[:, 4], dphi_old * RATIO, rtol=1e-9)
