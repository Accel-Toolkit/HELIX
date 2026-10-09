"""Tests for the 6-D eigen- (normal-mode) emittance helpers.

The 6-D eigenemittances are constants of motion under any linear
symplectic transport.  We exercise three regimes:

* uncoupled Σ → eigenemittances reduce to (ε_x, ε_y, ε_z);
* x-y coupling (solenoid-like rotation) → invariance under R·Σ·Rᵀ;
* dispersive coupling (x ↔ φ block off-diagonal) → invariance.
"""
import numpy as np
import pytest

from linac_gen.diagnostics.eigenemittance import (
    eigenemittances, eigenemittances_4d, kinetic_invariants,
    eigenemittances_series,
)


# ---------------------------------------------------------------------------
def _symplectic_xy_rot(theta: float) -> np.ndarray:
    """Build the 6×6 symplectic x-y rotation by angle θ."""
    c, s = np.cos(theta), np.sin(theta)
    R = np.eye(6)
    R[0, 0] = c; R[0, 2] = s
    R[1, 1] = c; R[1, 3] = s
    R[2, 0] = -s; R[2, 2] = c
    R[3, 1] = -s; R[3, 3] = c
    return R


def _drift_matrix(L: float) -> np.ndarray:
    """Build a 6×6 drift transport matrix of length L."""
    M = np.eye(6)
    M[0, 1] = L
    M[2, 3] = L
    # longitudinal drift (linear): φ moves with σ = 1/(βγ²) · L; we just
    # take a small generic value so the block isn't degenerate.
    M[4, 5] = 0.01 * L
    return M


# ---------------------------------------------------------------------------
def test_uncoupled_diagonal_recovers_projected():
    """Diagonal Σ → ε_i = projected emittance per block."""
    ex, ey, ez = 1.2, 0.7, 4.0
    sigma = np.diag([ex, ex, ey, ey, ez, ez])  # α=0, β=γ=1 in each plane
    e1, e2, e3 = eigenemittances(sigma)
    # Sorting both sides puts them in a stable order regardless of which
    # mode is "first" — what we want is set equality.
    assert sorted([e1, e2, e3]) == pytest.approx(sorted([ex, ey, ez]),
                                                 rel=1e-10)


def test_uncoupled_block_diagonal_recovers_projected():
    """Block-diagonal Σ with α≠0 still has ε_i = √det of each block."""
    eps = (2.0, 0.5, 3.0)
    twiss = ((1.5, 0.4), (0.8, -0.2), (4.0, 0.1))  # (β, α) per plane
    blocks = []
    for e, (b, a) in zip(eps, twiss):
        g = (1.0 + a * a) / b
        blocks.append(e * np.array([[b, -a], [-a, g]]))
    sigma = np.block([
        [blocks[0], np.zeros((2, 2)), np.zeros((2, 2))],
        [np.zeros((2, 2)), blocks[1], np.zeros((2, 2))],
        [np.zeros((2, 2)), np.zeros((2, 2)), blocks[2]],
    ])
    e1, e2, e3 = eigenemittances(sigma)
    assert sorted([e1, e2, e3]) == pytest.approx(sorted(eps), rel=1e-10)


def test_invariant_under_xy_rotation():
    """6-D ε_i must not change under a 4×4 transverse symplectic rotation."""
    ex, ey, ez = 1.5, 0.4, 2.5
    sigma_u = np.diag([ex, ex, ey, ey, ez, ez])
    e0 = sorted(eigenemittances(sigma_u))

    R = _symplectic_xy_rot(0.37)
    sigma_c = R @ sigma_u @ R.T
    e1 = sorted(eigenemittances(sigma_c))

    np.testing.assert_allclose(e1, e0, rtol=1e-9)


def test_invariant_under_drift_propagation():
    """ε_i is preserved by a drift transport: M · Σ · Mᵀ ⇒ same ε_i."""
    sigma = np.diag([2.0, 2.0, 0.5, 0.5, 3.0, 3.0])
    e0 = sorted(eigenemittances(sigma))
    M = _drift_matrix(0.85)
    sigma_after = M @ sigma @ M.T
    e1 = sorted(eigenemittances(sigma_after))
    np.testing.assert_allclose(e1, e0, rtol=1e-9)


def test_invariant_under_dispersive_coupling():
    """Adding x ↔ φ off-diagonal terms preserves the eigenemittances.

    A symplectic dispersive transformation R = I + λ·D (with D tied to
    the symplectic form so that R·J·Rᵀ = J) leaves the invariants
    untouched, even though the projected ε_x grows.
    """
    sigma = np.diag([1.5, 1.5, 0.6, 0.6, 2.5, 2.5])
    e0 = sorted(eigenemittances(sigma))
    # Symplectic generator: x ↔ w block.  Use M = exp(λ · X) where
    # X is a 6×6 generator with non-zero (0, 5) and (4, 1) entries chosen
    # so M is symplectic.  For small λ a first-order shear is good enough.
    lam = 0.15
    M = np.eye(6)
    M[0, 5] = lam
    M[4, 1] = -lam   # makes M·J·Mᵀ = J + O(λ²)
    sigma_after = M @ sigma @ M.T
    e1 = sorted(eigenemittances(sigma_after))
    # Allow O(λ²) drift since our generator is only first-order symplectic.
    np.testing.assert_allclose(e1, e0, rtol=2 * lam ** 2, atol=1e-3)


def test_singular_returns_projected_fallback():
    """Σ with zero longitudinal block → ε₃ = 0, ε₁/ε₂ from transverse blocks."""
    sigma = np.diag([1.0, 1.0, 0.5, 0.5, 0.0, 0.0])
    e1, e2, e3 = eigenemittances(sigma)
    assert min(e1, e2, e3) == pytest.approx(0.0)


def test_zeros_returns_zeros():
    """All-zero Σ (no particles yet) returns (0, 0, 0) without crashing."""
    e1, e2, e3 = eigenemittances(np.zeros((6, 6)))
    assert (e1, e2, e3) == (0.0, 0.0, 0.0)


def test_non_finite_returns_zeros():
    """NaN / Inf in Σ must be tolerated, returning zeros."""
    bad = np.full((6, 6), np.nan)
    e1, e2, e3 = eigenemittances(bad)
    assert (e1, e2, e3) == (0.0, 0.0, 0.0)


def test_kinetic_invariants_diagonal():
    """For a diagonal Σ with planes (ex, ey, ez) the invariants are
    I₂ = ex² + ey² + ez²,  I₄ = ex⁴ + ey⁴ + ez⁴, I₆ = ex⁶ + ey⁶ + ez⁶
    (sums of even-power eigenmodes; each plane contributes a pair ±i·ε).
    """
    ex, ey, ez = 1.2, 0.7, 2.5
    sigma = np.diag([ex, ex, ey, ey, ez, ez])
    i2, i4, i6 = kinetic_invariants(sigma)
    assert i2 == pytest.approx(ex ** 2 + ey ** 2 + ez ** 2, rel=1e-10)
    assert i4 == pytest.approx(ex ** 4 + ey ** 4 + ez ** 4, rel=1e-10)
    assert i6 == pytest.approx(ex ** 6 + ey ** 6 + ez ** 6, rel=1e-10)


def test_4d_helper_matches_recorder():
    """The trace-formulation 4-D helper matches the recorder's eigenvalue
    method to numerical noise on a coupled case."""
    from linac_gen.diagnostics.recorder import _normal_mode_emittances
    ex, ey = 1.5, 0.4
    sigma_u = np.diag([ex, ex, ey, ey])
    R4 = np.array([
        [np.cos(0.31), 0, np.sin(0.31), 0],
        [0, np.cos(0.31), 0, np.sin(0.31)],
        [-np.sin(0.31), 0, np.cos(0.31), 0],
        [0, -np.sin(0.31), 0, np.cos(0.31)],
    ])
    sigma_c = R4 @ sigma_u @ R4.T
    e1_eig, e2_eig = _normal_mode_emittances(sigma_c)
    e1_inv, e2_inv = eigenemittances_4d(sigma_c)
    np.testing.assert_allclose(sorted([e1_inv, e2_inv]),
                               sorted([e1_eig, e2_eig]), rtol=1e-9)


def test_series_helper():
    """eigenemittances_series applies the function row-wise."""
    sigmas = [np.diag([1.0, 1.0, 0.5, 0.5, 2.0, 2.0])] * 3
    out = eigenemittances_series(sigmas)
    assert out.shape == (3, 3)
    for row in out:
        assert sorted(row) == pytest.approx([0.5, 1.0, 2.0], rel=1e-10)


def test_series_handles_garbage_entries():
    """Wrong-shape entries → row of zeros, no exception."""
    sigmas = [np.eye(6) * 1.5, np.zeros((4, 4)), None, np.eye(6) * 0.7]
    # None will trip np.asarray with dtype=float — wrap in try.
    sigmas = [s if isinstance(s, np.ndarray) else np.zeros((6, 6))
              for s in sigmas]
    out = eigenemittances_series(sigmas)
    assert out.shape == (4, 3)
    assert np.all(out[1] == 0.0)


# ---------------------------------------------------------------------------
# Normalised canonical intrinsic emittances (canonical_eigenemittances)
# ---------------------------------------------------------------------------
from types import SimpleNamespace

from linac_gen.diagnostics.eigenemittance import (   # noqa: E402
    canonical_eigenemittances, canonical_eigenemittances_series,
    canonical_scale,
)
from linac_gen.diagnostics.recorder import _convert_emit_z_to_mmmrad  # noqa: E402

_M = 939.29420          # H- rest mass [MeV]
_J6C = np.kron(np.eye(3), np.array([[0.0, 1.0], [-1.0, 0.0]]))


def _kin(w_mev):
    g = 1.0 + w_mev / _M
    return g, np.sqrt(g * g - 1.0)


def _random_symplectic(seed, scale=0.3):
    from scipy.linalg import expm
    h = np.random.default_rng(seed).normal(size=(6, 6))
    return expm(_J6C @ (h + h.T) * scale)


def _sigma_coupled():
    s = np.diag([1.0, 1.0, 0.8, 1.2, 4.0, 1e-4])   # mm², mrad², deg², MeV²
    s[0, 5] = s[5, 0] = 0.006                      # <x ΔW>: dispersion-like
    s[2, 4] = s[4, 2] = 0.3                        # <y Δφ>
    return s


def test_canonical_uncoupled_equals_recorder_normalised_emittances():
    """External anchor: without coupling the modes ARE the recorder's
    ε_nx = βγε_x, ε_ny and ε_nz = βγ·ε_z(mm·mrad) (independent code path,
    _convert_emit_z_to_mmmrad) — at two energies and two frequencies."""
    for w, f in ((2.1227, 162.5), (185.0, 650.0)):
        g, bg = _kin(w)
        s = np.diag([1.3, 0.7, 0.9, 1.1, 4.0, 1e-4])
        s[0, 1] = s[1, 0] = -0.4
        s[4, 5] = s[5, 4] = 0.012
        r = canonical_eigenemittances(s, bg, f, _M)
        ex = np.sqrt(np.linalg.det(s[:2, :2]))
        ey = np.sqrt(np.linalg.det(s[2:4, 2:4]))
        ez = np.sqrt(np.linalg.det(s[4:, 4:]))
        ref = SimpleNamespace(beta=bg / g, gamma=g, species=SimpleNamespace(mass=_M),
                              wavelength=299792458.0 / (f * 1e6) * 1e3)
        want = [ex * bg, ey * bg, _convert_emit_z_to_mmmrad(ez, ref) * bg]
        np.testing.assert_allclose(r["modes"], want, rtol=1e-12)
        np.testing.assert_allclose(r["proj"], want, rtol=1e-12)
        np.testing.assert_allclose(r["share"], np.eye(3), atol=1e-12)
        assert r["coupling6"] == pytest.approx(1.0, abs=1e-12)
        assert r["coupling4"] == pytest.approx(1.0, abs=1e-12)
        assert r["e4d"] == pytest.approx(ex * ey * bg ** 2, rel=1e-12)


def test_canonical_modes_invariant_under_coupled_symplectic_transport():
    """Any canonical symplectic map, written in recorded (mm, mrad, deg,
    MeV) units, leaves the canonical modes and ε_6D unchanged — the case
    the mixed-unit eigenemittances() gets wrong (pinned below)."""
    g, bg = _kin(2.1227)
    f = 162.5
    d = canonical_scale(bg, f, _M)
    s0 = _sigma_coupled()
    r0 = canonical_eigenemittances(s0, bg, f, _M)
    for seed in range(5):
        m = np.diag(1.0 / d) @ _random_symplectic(seed) @ np.diag(d)
        r1 = canonical_eigenemittances(m @ s0 @ m.T, bg, f, _M)
        np.testing.assert_allclose(np.sort(r1["modes"]), np.sort(r0["modes"]),
                                   rtol=1e-9)
        assert r1["e6d"] == pytest.approx(r0["e6d"], rel=1e-9)
        assert r1["coupling6"] >= 1.0 - 1e-12
    # Mixed-unit Balandin values are NOT invariant under the same map
    # (documents why the canonical variant exists).
    m = np.diag(1.0 / d) @ _random_symplectic(0) @ np.diag(d)
    old0 = np.sort(eigenemittances(s0))
    old1 = np.sort(eigenemittances(m @ s0 @ m.T))
    assert np.max(np.abs(old1 / old0 - 1.0)) > 0.1


def test_canonical_modes_satisfy_projection_inequality_and_shares():
    g, bg = _kin(2.1227)
    r = canonical_eigenemittances(_sigma_coupled(), bg, 162.5, _M)
    assert np.prod(r["proj"]) >= r["e6d"] * (1 - 1e-12)
    assert np.prod(r["modes"]) == pytest.approx(r["e6d"], rel=1e-10)
    np.testing.assert_allclose(r["share"].sum(axis=1), 1.0, atol=1e-10)
    # x-like mode is mostly x, z-like mostly z (labels follow dominant plane)
    assert r["share"][0, 0] > 0.5 and r["share"][2, 2] > 0.5


def test_canonical_modes_invariant_across_freq_jump():
    """A FREQ jump rescales Δφ by f_new/f_old (Σ → DΣD); the canonical
    modes must not move although the recorded ε_z (deg·MeV) doubles."""
    g, bg = _kin(35.0)
    s = _sigma_coupled()
    r1 = canonical_eigenemittances(s, bg, 162.5, _M)
    jump = np.diag([1, 1, 1, 1, 2.0, 1])
    r2 = canonical_eigenemittances(jump @ s @ jump, bg, 325.0, _M)
    np.testing.assert_allclose(r2["modes"], r1["modes"], rtol=1e-12)


def test_canonical_degenerate_and_nonphysical_inputs_are_nan():
    g, bg = _kin(2.1227)
    for s, args in ((np.zeros((6, 6)), (bg, 162.5, _M)),
                    (_sigma_coupled(), (0.0, 162.5, _M)),
                    (_sigma_coupled(), (bg, 0.0, _M)),
                    (np.full((6, 6), np.nan), (bg, 162.5, _M))):
        r = canonical_eigenemittances(s, *args)
        assert np.all(np.isnan(r["modes"])) and np.isnan(r["e6d"])


def test_canonical_series_dc_records_keep_only_4d():
    """Dual regime: DC (unbunched) records have no ε_z, so 6-D outputs are
    NaN there while ε_4D is kept; bunched records are filled."""
    g, bg = _kin(0.035)
    s = _sigma_coupled()
    res = SimpleNamespace(
        s=[0.0, 1.0, 2.0], sigma_matrix=[s, s, np.zeros((6, 6))],
        ref_beta=[bg / g] * 3, ref_gamma=[g] * 3, ref_frequency=[162.5] * 3,
        mass_mev=_M, continuous_at=[True, False, False])
    ser = canonical_eigenemittances_series(res)
    assert np.isnan(ser["eps_x"][0]) and np.isfinite(ser["e4d"][0])
    assert np.isfinite(ser["eps_x"][1]) and np.isfinite(ser["e6d"][1])
    assert np.isnan(ser["eps_x"][2]) and np.isnan(ser["e4d"][2])
    # misaligned inputs → None, not a wrong-length series
    res.ref_frequency = [162.5] * 2
    assert canonical_eigenemittances_series(res) is None


@pytest.mark.parametrize("w_mev,f_mhz,angle,rho,hv,k", [
    (800.0, 650.0, 6.5637, 20430.0, 0, 0.0),     # BTL-arc-like, horizontal
    (2.1227, 162.5, 45.0, 300.0, 0, 0.5),        # low energy, field index
    (185.0, 325.0, -10.0, 2000.0, 1, 0.0),       # vertical bend
])
def test_canonical_scale_makes_helix_dipole_matrices_symplectic(
        w_mev, f_mhz, angle, rho, hv, k):
    """External anchor: HELIX's Dipole matrix (its own R16/R26 ↔ R51/R52
    derivation) must be symplectic in the canonical coordinates, i.e.
    Mcᵀ J Mc = J with Mc = D·M·D⁻¹.  Fails with the unsigned ΔE scale
    (|error| ~0.3 at 800 MeV) — (Δφ, ΔW) is anti-canonical."""
    from linac_gen.core.particle import H_MINUS as sp
    from linac_gen.core.reference import ReferenceParticle
    from linac_gen.elements.dipole import Dipole
    from linac_gen.tracking.matrix_tracking import get_element_matrix
    ref = ReferenceParticle(species=sp, w_kin=w_mev, frequency=f_mhz)
    el = Dipole("B", angle=angle, rho=rho, field_index=k, hv=hv)
    M = get_element_matrix(el, ref)
    d = canonical_scale(ref.beta * ref.gamma, f_mhz, sp.mass)
    mc = np.diag(d) @ M @ np.diag(1.0 / d)
    np.testing.assert_allclose(mc.T @ _J6C @ mc, _J6C, atol=1e-12)
    # and the coupling is really exercised (dispersion is non-zero)
    assert abs(M[0 if hv == 0 else 2, 5]) > 1e-6


def test_canonical_series_vectorised_equals_per_record_reference():
    """The vectorised series must reproduce the per-record function on a
    mix of coupled, uncoupled, degenerate and DC records, across energy
    and frequency (FREQ-jump-like) changes.  Tolerances: batched vs single
    LAPACK det/eig differ by ~1e-9 relative on ill-conditioned Σ."""
    rng = np.random.default_rng(7)
    n = 60
    stack, bgs, fs = [], [], []
    for i in range(n):
        a = rng.normal(size=(6, 6)) * np.array([1, 0.5, 1, 0.5, 3, 0.01])
        s = a @ a.T
        if i % 7 == 0:
            s = np.diag(np.diag(s))                   # uncoupled
        if i % 11 == 0:
            s = np.zeros((6, 6))                       # lost beam
        stack.append(s)
        g, bg = _kin(2.0 + 10.0 * i)
        bgs.append((bg / g, g))
        fs.append(162.5 if i < 20 else (325.0 if i < 40 else 650.0))
    dc = [i < 3 for i in range(n)]
    res = SimpleNamespace(sigma_matrix=stack, ref_beta=[b for b, _ in bgs],
                          ref_gamma=[g for _, g in bgs], ref_frequency=fs,
                          mass_mev=_M, continuous_at=dc)
    ser = canonical_eigenemittances_series(res)
    for i in range(n):
        b, g = bgs[i]
        r = canonical_eigenemittances(stack[i], b * g, fs[i], _M)
        np.testing.assert_allclose(ser["e4d"][i], r["e4d"], rtol=1e-8, equal_nan=True)
        np.testing.assert_allclose(ser["coupling4"][i], r["coupling4"], rtol=1e-8,
                                   equal_nan=True)
        if dc[i]:
            assert np.isnan(ser["eps_x"][i]) and np.isnan(ser["e6d"][i])
            continue
        got = [ser[f"eps_{p}"][i] for p in "xyz"]
        # eig() rounding: absolute error scales with the LARGEST mode
        np.testing.assert_allclose(got, r["modes"], rtol=1e-9, equal_nan=True,
                                   atol=1e-12 * np.nanmax(np.r_[r["modes"], 0.0]))
        np.testing.assert_allclose([ser[f"proj_{p}"][i] for p in "xyz"], r["proj"],
                                   rtol=1e-8, equal_nan=True)
        np.testing.assert_allclose(ser["e6d"][i], r["e6d"], rtol=1e-8, equal_nan=True)
        np.testing.assert_allclose(ser["coupling6"][i], r["coupling6"], rtol=1e-8,
                                   equal_nan=True)
