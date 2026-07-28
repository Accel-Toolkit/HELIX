"""Envelope-solver treatment of the stripper foil.

The foil's mean map is the identity, but its zero-mean stochastic kicks
diffuse the second moments: the envelope solver must apply

    Σ ← M Σ Mᵀ + D,   M = I,
    D = diag(0, θ₀², 0, θ₀², 0, σ_E²)     (mrad² / MeV² blocks)

with θ₀ the Highland rms plane angle and σ_E the Bohr straggling width,
plus the Bethe-Bloch mean loss on the reference energy.  Before the
2026-07 fix the envelope solver saw the foil as a pure identity (the
paper draft even claimed "the envelope trace is unaffected") — these
tests pin the corrected behaviour.
"""
import numpy as np
import pytest

from linac_gen.core.lattice import Lattice
from linac_gen.core.particle import H_MINUS
from linac_gen.core.reference import ReferenceParticle
from linac_gen.elements.drift import Drift
from linac_gen.elements.foil import Foil
from linac_gen.tracking.envelope import EnvelopeSolver

W_KIN = 800.0     # MeV — paper foil case
FREQ = 162.5      # MHz

INITIAL = {
    "alpha_x": 0.0, "beta_x": 5.0, "emit_x": 1.0,     # mm·mrad
    "alpha_y": 0.0, "beta_y": 3.0, "emit_y": 0.8,
    "alpha_z": 0.0, "beta_z": 10.0, "emit_z": 0.5,    # deg·MeV
}


def _run_envelope(current=0.0):
    lat = Lattice()
    lat.add(Drift(name="D1", length=100.0, aperture=20.0))
    lat.add(Foil(name="STRIP", material="C", thickness_ug_cm2=600.0))
    lat.add(Drift(name="D2", length=100.0, aperture=20.0))
    ref = ReferenceParticle(species=H_MINUS, w_kin=W_KIN, frequency=FREQ)
    solver = EnvelopeSolver(lat, ref, INITIAL, current=current)
    results = solver.run()
    i_foil = results.element_names.index("STRIP")
    return results, i_foil


def _expected_moments():
    """Model moments computed from the Foil element's own helpers."""
    foil = Foil(name="STRIP", material="C", thickness_ug_cm2=600.0)
    ref = ReferenceParticle(species=H_MINUS, w_kin=W_KIN, frequency=FREQ)
    theta_mrad = foil._highland_theta_rms(ref) * 1e3
    sigma_E = foil._energy_loss_sigma_MeV(ref)
    dE_mean = foil._mean_energy_loss_MeV(ref)
    return theta_mrad, sigma_E, dE_mean


def test_envelope_foil_adds_angular_diffusion():
    """Σ_x'x' and Σ_y'y' each grow by exactly θ₀² (mrad²) at the foil."""
    results, i = _run_envelope()
    theta_mrad, _, _ = _expected_moments()
    before = results.sigma_matrix[i - 1]
    after = results.sigma_matrix[i]
    assert after[1, 1] - before[1, 1] == pytest.approx(theta_mrad ** 2,
                                                       rel=1e-9)
    assert after[3, 3] - before[3, 3] == pytest.approx(theta_mrad ** 2,
                                                       rel=1e-9)


def test_envelope_foil_adds_energy_diffusion():
    """Σ_WW grows by exactly σ_E² (Bohr, MeV²) at the foil."""
    results, i = _run_envelope()
    _, sigma_E, _ = _expected_moments()
    before = results.sigma_matrix[i - 1]
    after = results.sigma_matrix[i]
    assert after[5, 5] - before[5, 5] == pytest.approx(sigma_E ** 2,
                                                       rel=1e-9)


def test_envelope_foil_leaves_other_moments_untouched():
    """Positions, phase, and all cross-moments pass through unchanged."""
    results, i = _run_envelope()
    before = results.sigma_matrix[i - 1].copy()
    after = results.sigma_matrix[i].copy()
    delta = after - before
    for j in (1, 3, 5):
        delta[j, j] = 0.0     # the three diffusion entries, tested above
    assert np.allclose(delta, 0.0, atol=1e-15)


def test_envelope_foil_drops_reference_energy_by_bethe_bloch_mean():
    """Reference energy drops by ⟨ΔE⟩ at the foil (and only there)."""
    results, i = _run_envelope()
    _, _, dE_mean = _expected_moments()
    assert results.ref_w_kin[i - 1] == pytest.approx(W_KIN, rel=1e-12)
    assert results.ref_w_kin[i] == pytest.approx(W_KIN - dE_mean, rel=1e-12)
    # No further energy change in the downstream drift
    assert results.ref_w_kin[i + 1] == pytest.approx(W_KIN - dE_mean,
                                                     rel=1e-12)


def test_envelope_foil_grows_transverse_emittance():
    """Diffusion is a real emittance source: ε_x, ε_y increase at the foil."""
    results, i = _run_envelope()
    assert results.emit_x[i] > results.emit_x[i - 1]
    assert results.emit_y[i] > results.emit_y[i - 1]


def test_envelope_foil_diffusion_identical_with_space_charge_on():
    """The foil branch bypasses SC sub-stepping (zero length): the Σ jump
    across the foil record is exactly D with the SC kick active too."""
    results, i = _run_envelope(current=5.0)
    theta_mrad, sigma_E, _ = _expected_moments()
    before = results.sigma_matrix[i - 1]
    after = results.sigma_matrix[i]
    assert after[1, 1] - before[1, 1] == pytest.approx(theta_mrad ** 2,
                                                       rel=1e-9)
    assert after[5, 5] - before[5, 5] == pytest.approx(sigma_E ** 2,
                                                       rel=1e-9)


def test_envelope_diffusion_matrix_shape_and_units():
    """Foil.envelope_diffusion: exactly three non-zero entries, all on the
    diagonal, with the documented values."""
    foil = Foil(name="STRIP", material="C", thickness_ug_cm2=600.0)
    ref = ReferenceParticle(species=H_MINUS, w_kin=W_KIN, frequency=FREQ)
    D, dE_mean = foil.envelope_diffusion(ref)
    assert D.shape == (6, 6)
    assert np.count_nonzero(D) == 3
    theta_mrad = foil._highland_theta_rms(ref) * 1e3
    assert D[1, 1] == pytest.approx(theta_mrad ** 2, rel=1e-12)
    assert D[3, 3] == pytest.approx(theta_mrad ** 2, rel=1e-12)
    assert D[5, 5] == pytest.approx(foil._energy_loss_sigma_MeV(ref) ** 2,
                                    rel=1e-12)
    assert dE_mean == pytest.approx(foil._mean_energy_loss_MeV(ref),
                                    rel=1e-12)


def test_zero_thickness_foil_is_identity_for_envelope():
    lat = Lattice()
    lat.add(Drift(name="D1", length=100.0, aperture=20.0))
    lat.add(Foil(name="STRIP", material="C", thickness_ug_cm2=0.0))
    ref = ReferenceParticle(species=H_MINUS, w_kin=W_KIN, frequency=FREQ)
    results = EnvelopeSolver(lat, ref, INITIAL, current=0.0).run()
    i = results.element_names.index("STRIP")
    assert np.allclose(results.sigma_matrix[i], results.sigma_matrix[i - 1])
    assert results.ref_w_kin[i] == pytest.approx(W_KIN, rel=1e-15)
