"""Determinant-normalized Twiss extraction for accelerating cells.

Through an accelerating element the transverse (x, x′) block is
conformally symplectic: det M₂ = (βγ)_in/(βγ)_out, eigenvalues
λ = √det·e^{±iμ}.  ``compute_twiss`` used to run ``acos(tr/2)`` on the
raw block — biased for every accelerating cell (det ≈ 0.863 per PIP-II
HWR cavity) and mislabeling strongly-damped stable cells as unstable.
Same for the coupled 4×4 eigen extraction (det(M₄)^{1/4}) and the CS
propagation helpers (β/α scale by 1/det).

Magnetostatic maps (|det − 1| ≤ 1e-9) must take the historical code
path bit-for-bit.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from linac_gen.core.lattice import Lattice
from linac_gen.core.particle import PROTON
from linac_gen.core.reference import ReferenceParticle
from linac_gen.elements.drift import Drift
from linac_gen.elements.quadrupole import Quadrupole
from linac_gen.elements.rf_gap import RFGap
from linac_gen.elements.solenoid import Solenoid
from linac_gen.analysis.period_detect import PeriodicStructure
from linac_gen.analysis.phase_advance import (
    coupled_phase_advance, structure_phase_advance,
)
from linac_gen.tracking.matrix_tracking import (
    compute_transfer_matrix, compute_twiss, propagate_twiss,
)


def _rot_block(mu_rad: float, beta: float = 2.0, alpha: float = 0.3):
    """2×2 pseudo-rotation with Twiss (α, β): the standard matched-cell
    parameterization M = I·cosμ + J·sinμ."""
    c, s = math.cos(mu_rad), math.sin(mu_rad)
    return np.array([
        [c + alpha * s, beta * s],
        [-(1 + alpha ** 2) / beta * s, c - alpha * s],
    ])


def _embed(block, plane="x"):
    M = np.eye(6)
    i = {"x": 0, "y": 2, "z": 4}[plane]
    M[i:i + 2, i:i + 2] = block
    return M


def test_damped_rotation_recovers_mu_beta_alpha_and_damping():
    """g·R(μ): μ, β, α equal R's parameters exactly; damping == g."""
    mu, beta, alpha, g = 0.9, 2.0, 0.3, 0.97
    M = _embed(g * _rot_block(mu, beta, alpha))
    tw = compute_twiss(M, "x")
    assert tw["mu"] == pytest.approx(math.degrees(mu), abs=1e-12)
    assert tw["beta"] == pytest.approx(beta, rel=1e-12)
    assert tw["alpha"] == pytest.approx(alpha, abs=1e-12)
    assert tw["damping"] == pytest.approx(g, abs=1e-12)
    assert tw["mu_folded"] == pytest.approx(math.degrees(mu), abs=1e-12)


def test_symplectic_fast_path_bit_identical():
    """det == 1 must reproduce the historical formula bit-for-bit."""
    mu, beta, alpha = 1.1, 3.7, -0.8
    block = _rot_block(mu, beta, alpha)
    M = _embed(block)
    tw = compute_twiss(M, "x")
    # Inline copy of the pre-change extraction.
    m11, m12, m21, m22 = block[0, 0], block[0, 1], block[1, 0], block[1, 1]
    cos_mu = 0.5 * (m11 + m22)
    mu_ref = math.acos(cos_mu)
    sin_ref = math.sin(mu_ref)
    if m12 < 0:
        mu_ref = 2 * math.pi - mu_ref
        sin_ref = math.sin(mu_ref)
    assert tw["mu"] == math.degrees(mu_ref)              # bit-identical
    assert tw["beta"] == m12 / sin_ref                   # bit-identical
    assert tw["alpha"] == (m11 - m22) / (2.0 * sin_ref)  # bit-identical
    assert tw["damping"] == 1.0


def test_nonpositive_determinant_raises():
    M = _embed(np.array([[1.0, 0.5], [2.0, 1.0]]))  # det = 0
    with pytest.raises(ValueError, match="determinant"):
        compute_twiss(M, "x")


def test_folded_branch_key():
    """μ > 180° (m12 < 0 branch): mu_folded is the principal value."""
    mu = 2 * math.pi - 0.9          # oriented 308.4°
    M = _embed(_rot_block(mu, 2.0, 0.0))
    tw = compute_twiss(M, "x")
    assert tw["mu"] == pytest.approx(math.degrees(mu), abs=1e-9)
    assert tw["mu_folded"] == pytest.approx(math.degrees(0.9), abs=1e-9)


def _accelerating_cell() -> Lattice:
    # g=14 T/m, V=0.3 MV @ -30 deg: measured stable in both planes with
    # det = 0.9517 (genuinely accelerating).
    lat = Lattice()
    lat.add(Drift(name="D", length=100.0, aperture=10.0))
    lat.add(Quadrupole(name="QF", length=50.0, gradient=14.0, aperture=10.0))
    lat.add(Drift(name="D", length=100.0, aperture=10.0))
    lat.add(RFGap(name="G", voltage=0.3, phase=-30.0, frequency=162.5))
    lat.add(Drift(name="D", length=100.0, aperture=10.0))
    lat.add(Quadrupole(name="QD", length=50.0, gradient=-14.0, aperture=10.0))
    lat.add(Drift(name="D", length=100.0, aperture=10.0))
    return lat


def test_accelerating_fodo_mu_equals_eigenvalue_phase():
    """structure_phase_advance μ on an accelerating cell must equal the
    eigenvalue phase of the 2×2 block — the det-normalized definition."""
    lat = _accelerating_cell()
    ref = ReferenceParticle(species=PROTON, w_kin=2.5, frequency=162.5)
    M = compute_transfer_matrix(lat, ref.copy())
    blk = M[0:2, 0:2]
    det = np.linalg.det(blk)
    assert det < 1.0 - 1e-6          # genuinely accelerating
    lam = np.linalg.eigvals(blk / math.sqrt(det))
    mu_eig = math.degrees(abs(np.angle(lam[0])))

    period = PeriodicStructure(
        start=0, end=len(lat.elements),
        inner_period_length=len(lat.elements),
        inner_slice_end=len(lat.elements),
        n_repeats=1, label="cell", source="manual",
    )
    res = structure_phase_advance(lat, ref, period)
    assert res["mu_x_deg"] == pytest.approx(mu_eig, abs=1e-9)


def test_propagate_twiss_det_correction_matches_sigma_transport():
    """Twiss propagated through a damped block equals the Twiss of the
    transported Σ (β = σ11/ε with ε scaled by det)."""
    alpha0, beta0, eps = 0.4, 2.5, 1.7
    g = 0.93
    blk = g * _rot_block(0.7, 1.4, -0.2)
    T0 = eps * np.array([[beta0, -alpha0],
                         [-alpha0, (1 + alpha0 ** 2) / beta0]])
    T1 = blk @ T0 @ blk.T
    eps1 = math.sqrt(np.linalg.det(T1))
    beta_ref = T1[0, 0] / eps1
    alpha_ref = -T1[0, 1] / eps1
    a1, b1 = propagate_twiss(blk, alpha0, beta0)
    assert b1 == pytest.approx(beta_ref, rel=1e-12)
    assert a1 == pytest.approx(alpha_ref, rel=1e-12)


def test_propagate_twiss_symplectic_bit_identical():
    """det == 1 block: historical formula bit-for-bit."""
    blk = _rot_block(0.7, 1.4, -0.2)
    alpha0, beta0 = 0.4, 2.5
    c11, c12 = blk[0, 0], blk[0, 1]
    c21, c22 = blk[1, 0], blk[1, 1]
    g0 = (1.0 + alpha0 * alpha0) / beta0
    beta_ref = c11 * c11 * beta0 - 2.0 * c11 * c12 * alpha0 + c12 * c12 * g0
    alpha_ref = (-c11 * c21 * beta0 + (c11 * c22 + c12 * c21) * alpha0
                 - c12 * c22 * g0)
    a1, b1 = propagate_twiss(blk, alpha0, beta0)
    assert b1 == beta_ref
    assert a1 == alpha_ref


def test_coupled_damped_cell_is_stable_after_normalization():
    """Solenoid + accelerating gap cell: adiabatic damping must not be
    labeled 'unstable' (raw |λ| < 1 was failing the old check)."""
    lat = Lattice()
    lat.add(Drift(name="D", length=100.0, aperture=20.0))
    lat.add(Solenoid(name="S", length=200.0, field=1.5, aperture=20.0))
    lat.add(Drift(name="D", length=100.0, aperture=20.0))
    lat.add(RFGap(name="G", voltage=1.5, phase=-30.0, frequency=162.5))
    lat.add(Drift(name="D", length=100.0, aperture=20.0))
    ref = ReferenceParticle(species=PROTON, w_kin=2.5, frequency=162.5)
    period = PeriodicStructure(
        start=0, end=len(lat.elements),
        inner_period_length=len(lat.elements),
        inner_slice_end=len(lat.elements),
        n_repeats=1, label="cell", source="manual",
    )
    res = coupled_phase_advance(lat, ref, period)
    assert res["damping"] < 1.0 - 1e-6      # genuinely accelerating
    assert res["stable_I"] and res["stable_II"]
    assert 0.0 < res["mu_I_deg"] <= 180.0
    assert 0.0 < res["mu_II_deg"] <= 180.0


def test_torch_twin_parity_on_damped_matrix():
    torch = pytest.importorskip("torch")
    from linac_gen.tracking.torch_tracking import compute_twiss_torch
    mu, beta, alpha, g = 0.9, 2.0, 0.3, 0.95
    M = _embed(g * _rot_block(mu, beta, alpha))
    tw_np = compute_twiss(M, "x")
    tw_t = compute_twiss_torch(torch.as_tensor(M, dtype=torch.float64), "x")
    for key in ("alpha", "beta", "gamma_t", "mu", "mu_folded", "damping"):
        v = tw_t[key]
        v = float(v.detach()) if hasattr(v, "detach") else float(v)
        assert v == pytest.approx(tw_np[key], rel=1e-12, abs=1e-12), key
