"""Longitudinal beam phase advance: effective β̃_z units.

The recorded σ-matrix longitudinal block is in HELIX's native
(Δφ [deg], ΔW [MeV]) pair, so σ_φφ/ε_φW is a deg/MeV beta.  Feeding it
to the transverse mrad→deg integrand made μ_z ~685× too small at PIP-II
injection (k_φ/k_W ≈ 685 at 2.12 MeV / 162.5 MHz).  The fix converts to
locally-normalized (z, z′) coordinates:

    β̃_z = γ² · (k_w / k_φ) · σ_φφ/ε_φW      [mm/mrad]
    k_φ = 360/(βλ) [deg/mm],  k_w = β²γm/1000 [MeV/mrad]

Pinned here: (a) the conversion factor itself on a hand-built results
object, and (b) physics consistency — on a matched zero-current buncher
channel the beam ∫ds/β̃_z must reproduce the transfer-matrix μ_z.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from linac_gen.analysis.period_detect import PeriodicStructure
from linac_gen.analysis.phase_advance import (
    beam_phase_advance, structure_phase_advance, _beta_z_eff_from_sigma,
)
from linac_gen.core.lattice import Lattice
from linac_gen.core.particle import PROTON
from linac_gen.core.reference import ReferenceParticle
from linac_gen.elements.drift import Drift
from linac_gen.elements.rf_gap import RFGap
from linac_gen.tracking.envelope import EnvelopeResults, EnvelopeSolver
from linac_gen.tracking.matrix_tracking import (
    compute_transfer_matrix, compute_twiss,
)


def test_beta_z_eff_conversion_factor():
    """β̃_z == γ²·(k_w/k_φ)·(σ_φφ/ε_φW) on a hand-built record."""
    n = 3
    beta, gamma, freq, mass = 0.0728, 1.00266, 162.5, PROTON.mass
    s44, s55 = 16.0, 1e-4          # σ_φ=4 deg, σ_W=0.01 MeV, uncorrelated
    sm = np.zeros((6, 6))
    sm[4, 4], sm[5, 5] = s44, s55
    results = EnvelopeResults(
        s=[0.0, 100.0, 200.0],
        beta_x=[1.0] * n, beta_y=[1.0] * n,
        alpha_x=[0.0] * n, alpha_y=[0.0] * n,
        sigma_matrix=[sm.copy() for _ in range(n)],
        element_names=["INPUT", "E0", "E1"],
        ref_w_kin=[(gamma - 1.0) * mass] * n,
        ref_beta=[beta] * n, ref_gamma=[gamma] * n,
        ref_frequency=[freq] * n,
        mass_mev=mass,
    )
    bz = _beta_z_eff_from_sigma(results)
    assert bz is not None and bz.size == n

    eps_phiw = math.sqrt(s44 * s55)
    beta_phiw = s44 / eps_phiw                     # deg/MeV
    wavelength = 299792.458 / freq                 # mm
    k_phi = 360.0 / (beta * wavelength)            # deg/mm
    k_w = beta * beta * gamma * mass * 1e-3        # MeV/mrad
    expected = gamma * gamma * (k_w / k_phi) * beta_phiw
    np.testing.assert_allclose(bz, expected, rtol=1e-12)
    # Order-of-magnitude guard: the native deg/MeV number is ~685× off
    # at these parameters — the conversion must not be a no-op.
    assert not np.isclose(expected, beta_phiw, rtol=0.5)


def _buncher_channel(n_cells: int = 2):
    """Drift–buncher–drift cells: longitudinally focusing, no
    acceleration (φ_s = −90°)."""
    lat = Lattice()
    for _ in range(n_cells):
        for _ in range(5):
            lat.add(Drift(name="D", length=20.0, aperture=20.0))
        lat.add(RFGap(name="G", voltage=0.05, phase=-90.0, frequency=162.5))
        for _ in range(5):
            lat.add(Drift(name="D", length=20.0, aperture=20.0))
    return lat


def test_beam_mu_z_matches_structure_mu_z_at_zero_current():
    lat = _buncher_channel(n_cells=2)
    ref = ReferenceParticle(species=PROTON, w_kin=2.5, frequency=162.5)
    n_cell_elems = len(lat.elements) // 2
    period = PeriodicStructure(
        start=0, end=len(lat.elements),
        inner_period_length=n_cell_elems, inner_slice_end=n_cell_elems,
        n_repeats=2, label="cell", source="manual",
    )

    sigma0 = structure_phase_advance(lat, ref, period)
    assert sigma0["mu_z_deg"] is not None and sigma0["mu_z_deg"] > 0

    # Matched longitudinal Twiss from the bare cell matrix.
    M = compute_transfer_matrix(lat, ref, start=0, end=n_cell_elems - 1)
    twz = compute_twiss(M, "z")
    initial = dict(alpha_x=0.0, beta_x=1.0, emit_x=1.0,
                   alpha_y=0.0, beta_y=1.0, emit_y=1.0,
                   alpha_z=twz["alpha"], beta_z=twz["beta"], emit_z=0.05)
    res = EnvelopeSolver(lat, ref.copy(), initial, current=0.0).run()

    out = beam_phase_advance(res, period)
    # The beam integral is the unwrapped magnitude; the structure value
    # is folded to [0, 180°] — they must agree on this sub-180° cell.
    assert out["mu_z_deg"] is not None
    assert out["mu_z_deg"] == pytest.approx(sigma0["mu_z_deg"], rel=1e-2)
    # Longitudinally matched (β_z closes over the cell).
    assert out["mismatch_z"] is not None and out["mismatch_z"] < 0.05
