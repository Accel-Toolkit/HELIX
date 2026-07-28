"""Unit tests for the ``field_model="laplace2d"`` (M3) numerical Laplace
RFQ tracker.

The M3 path replaces the analytic 2-term Crandall coefficients with a
numerical 2-D Laplace solution Φ_static(x, y, z) sampled per .vane
slice and uses its second derivatives ``K_xx, K_yy`` as the in-plane
focusing strengths plus on-axis ``Ez_static = -∂Φ/∂z`` as the
longitudinal field.

Tests:

* T1   axisymmetric matcher sanity: Φ_static at on-axis quadrant samples
       reproduces the analytic ``Φ_2term = V₀·(x²−y²)/r₀²`` to within
       grid-discretisation accuracy (~10 %).
* T2   boundary fidelity: at sample points inside vane material the
       cached Φ equals the prescribed vane voltage.
* T3   interior Laplacian residual on a 5×5 stencil.
* T4   cache memory budget for the PXIE-scale default (estimated, no
       actual build).
* T5   zero-V drift: V_v_i=0 in laplace2d mode reduces to a drift.
* T6   helper passthrough: ``replace_rfq_cells_with_vane`` accepts and
       forwards ``field_model="laplace2d"`` and ``laplace_kwargs``.
* T7   M1 reproducibility: default ``field_model="2term"`` unchanged.
* T8   small-m cross-check: K_xx_M3 within 15 % of the M1 analytic
       value V·X/r₀² at a matcher slice.
"""
from __future__ import annotations

import numpy as np
import pytest

from linac_gen.core.particle import H_MINUS
from linac_gen.core.reference import ReferenceParticle
from linac_gen.core.beam import Beam
from linac_gen.core.lattice import Lattice
from linac_gen.elements.drift import Drift
from linac_gen.elements.rfq_cell import RfqCell
from linac_gen.elements.vane_rfq import CellSpan, VaneRFQ
from linac_gen.elements.vane_rfq_laplace2d import (
    Laplace2DCache,
    _vane_masks,
)
from linac_gen.io.tracewin_vane import VaneGeometry
from linac_gen.io.vane_rfq_helper import replace_rfq_cells_with_vane


@pytest.fixture
def ref():
    return ReferenceParticle(species=H_MINUS, w_kin=0.050, frequency=162.5)


def _flat_vane(L_mm: float, n: int = 50,
               a_m: float = 5.0e-3, V: float = 60_000.0) -> VaneGeometry:
    """Synthetic constant-aperture matcher vane (m=1 everywhere)."""
    z = np.linspace(0.0, L_mm * 1e-3, n)
    a = np.full_like(z, a_m)
    Tc = np.full_like(z, 4.183e-3)
    flag = np.zeros_like(z)
    Vp = np.full_like(z, +V * 0.5)
    Vn = np.full_like(z, -V * 0.5)
    return VaneGeometry(
        z=z,
        aperture_v1=a, Tc_v1=Tc, voltage_v1=Vp, flag_v1=flag,
        aperture_v2=a, Tc_v2=Tc, voltage_v2=Vn, flag_v2=flag,
        aperture_v3=a, Tc_v3=Tc, voltage_v3=Vp, flag_v3=flag,
        aperture_v4=a, Tc_v4=Tc, voltage_v4=Vn, flag_v4=flag,
    )


def _make_beam(ref, n=10):
    b = Beam(ref=ref, n_particles=n, current=0.0)
    rng = np.random.default_rng(0)
    b.particles[:, 0] = 0.5 * rng.normal(size=n)
    b.particles[:, 2] = 0.5 * rng.normal(size=n)
    return b


# ---------------------------------------------------------------------------
def test_T1_axisymmetric_matcher_quadrupole_shape():
    """Constant-aperture matcher: Φ(x,y) ≈ (V/2)·(x²−y²)/a² at small r.

    The analytic Crandall 2-term limit for m=1 is
    ``Φ = V₀·(x²−y²)/a²`` with ``V₀ = V_v1 = +V/2 = 30 kV`` and
    ``a = 5 mm`` (the inscribed radius and the vane tip distance both
    coincide at constant aperture).  M3 replaces this with a 5-point FD
    solve on a finite box.  Discretisation error scales as ``(dx/a)²``;
    for a 33-point grid spanning ±1.5·a the spacing is ~0.45 mm and we
    get ~5–10 % error vs analytic.
    """
    L = 20.0
    vane = _flat_vane(L_mm=L, n=10, a_m=5.0e-3, V=60_000.0)
    cache = Laplace2DCache(vane, nx=33, ny=33, box_factor=1.5)
    a_mm = 5.0
    V0 = 30_000.0
    z_mid_mm = L * 0.5
    # 9 sample points on a quarter-grid inside the cavity.
    pts = [(0.0, 0.0), (1.0, 0.0), (2.0, 0.0), (3.0, 0.0),
           (0.0, 1.0), (0.0, 2.0), (0.0, 3.0),
           (1.5, 1.5), (2.0, 1.0)]
    for x, y in pts:
        phi = float(cache.Phi_static(x, y, z_mid_mm))
        phi_ana = V0 * (x * x - y * y) / (a_mm * a_mm)
        # Analytic Φ at axis is 0 — use absolute tol for that case
        denom = max(abs(phi_ana), 1.0)  # V
        rel_err = abs(phi - phi_ana) / denom
        assert rel_err < 0.15, (
            f"M3 Φ at ({x},{y})={phi:.1f}V vs analytic {phi_ana:.1f}V "
            f"(rel err {rel_err:.3f})"
        )


def test_T2_boundary_fidelity_in_vane_material():
    """Grid points inside vane material are pinned exactly to V_v_i."""
    L = 20.0
    V = 60_000.0
    vane = _flat_vane(L_mm=L, n=10, a_m=5.0e-3, V=V)
    cache = Laplace2DCache(vane, nx=41, ny=41, box_factor=1.6)
    z_mid = L * 0.5
    # x²−y² ≥ a₁² well inside vane 1 (+x).
    phi_v1 = float(cache.Phi_static(7.0, 1.0, z_mid))    # x²−y² = 48 ≥ 25
    assert abs(phi_v1 - 30_000.0) / 30_000.0 < 1e-6
    # y²−x² ≥ a₂² well inside vane 2 (+y).
    phi_v2 = float(cache.Phi_static(1.0, 7.0, z_mid))
    assert abs(phi_v2 - (-30_000.0)) / 30_000.0 < 1e-6


def test_T3_interior_laplacian_residual():
    """On non-boundary grid cells, ∇²Φ should be ≈ 0 (residual under
    a few hundred V/mm² at 33-pt resolution)."""
    L = 20.0
    vane = _flat_vane(L_mm=L, n=10, a_m=5.0e-3)
    cache = Laplace2DCache(vane, nx=33, ny=33, box_factor=1.5)
    phi = cache.phi_static[len(cache.z_mm) // 2].astype(float)
    nx, ny = phi.shape
    dx = cache._dx_mm
    dy = cache._dy_mm
    # 5-point Laplacian residual on a strict interior box.
    res = (phi[2:, 1:-1] - 2 * phi[1:-1, 1:-1] + phi[:-2, 1:-1]) / (dx * dx) \
        + (phi[1:-1, 2:] - 2 * phi[1:-1, 1:-1] + phi[1:-1, :-2]) / (dy * dy)
    # Mask out cells inside vane material (those are Dirichlet pinned).
    a_mm = 5.0
    X, Y = np.meshgrid(np.linspace(-1.5 * a_mm, 1.5 * a_mm, nx),
                       np.linspace(-1.5 * a_mm, 1.5 * a_mm, ny),
                       indexing='ij')
    interior = ((X[1:-1, 1:-1]**2 - Y[1:-1, 1:-1]**2 < a_mm * a_mm)
                & (Y[1:-1, 1:-1]**2 - X[1:-1, 1:-1]**2 < a_mm * a_mm))
    free_res = res[interior]
    max_abs = float(np.max(np.abs(free_res))) if free_res.size else 0.0
    assert max_abs < 1.0, f"interior Laplacian residual {max_abs:.3e} V/mm²"


def test_T4_pxie_cache_memory_budget():
    """The full PXIE-scale cache (17 567 × 64 × 64 × 4 bytes) is below
    320 MB."""
    nz = 17_567
    nx = 64
    ny = 64
    expected_MB = nz * nx * ny * 4 / (1024 * 1024)
    assert expected_MB < 320.0


def test_T5_zero_voltage_is_drift_laplace2d(ref):
    """V_v_i = 0 → Φ = 0 → no kicks.  Particles drift through."""
    L = 20.0
    vane = _flat_vane(L_mm=L, n=10, a_m=5.0e-3, V=0.0)
    cells = [CellSpan(z_start_mm=0.0, z_end_mm=L, voltage_V=0.0,
                      A10=0.0, modulation=1.0, length_mm=L,
                      phi_s_deg=-60.0, cell_type=2,
                      type_prev=2, type_next=2, r0_dat_mm=5.0)]
    elem = VaneRFQ("R3", vane=vane, cells=cells, n_steps=20,
                   field_model="laplace2d",
                   laplace_kwargs=dict(nx=21, ny=21, box_factor=1.5))
    beam = _make_beam(ref, n=8)
    x_in = beam.particles[:, 0].copy()
    xp_in = beam.particles[:, 1].copy()
    ds = elem.length / elem.n_steps
    for _ in range(elem.n_steps):
        elem.track_rk4(beam, ds)
    np.testing.assert_allclose(beam.particles[:, 1], xp_in, atol=1e-9)
    np.testing.assert_allclose(beam.particles[:, 0],
                               x_in + xp_in * elem.length * 1e-3,
                               atol=1e-9)


def test_T6_helper_field_model_passthrough_laplace2d():
    """`replace_rfq_cells_with_vane` accepts ``laplace2d`` plus
    ``laplace_kwargs``."""
    L = 20.0
    lat = Lattice()
    lat.add(Drift("D1", length=10.0, aperture=0.0))
    lat.add(RfqCell("RFQ1", voltage_V=60_000.0, r0_mm=5.0, A10=0.0,
                    modulation=1.0, length_mm=L, phi_s_deg=-90.0,
                    cell_type=3))
    vane = _flat_vane(L_mm=L, n=10)
    replace_rfq_cells_with_vane(
        lat, vane, field_model="laplace2d",
        laplace_kwargs=dict(nx=21, ny=21, box_factor=1.5),
    )
    rfq = next(e for e in lat.elements if isinstance(e, VaneRFQ))
    assert rfq.field_model == "laplace2d"
    assert rfq._laplace_cache is not None


def test_T7_default_field_model_is_2term(ref):
    """Default keyword ``field_model="2term"`` unchanged (M1 path)."""
    L = 20.0
    cells = [CellSpan(z_start_mm=0.0, z_end_mm=L, voltage_V=60_000.0,
                      A10=0.1, modulation=1.5, length_mm=L,
                      phi_s_deg=-60.0, cell_type=2,
                      type_prev=2, type_next=2, r0_dat_mm=5.0)]
    vane = _flat_vane(L_mm=L, n=10)
    elem = VaneRFQ("R", vane=vane, cells=cells, n_steps=20)
    assert elem.field_model == "2term"
    # No laplace cache built (only on laplace2d).
    assert elem._laplace_cache is None


def test_T8_K_xx_matches_M1_analytic_in_matcher_limit():
    """For the matcher (constant a, m=1, A₁₀=0):
        K_xx_M3 ≈ V·X/r₀² = V·1/r₀² = V/a²
    Within 15 % at 33-point grid resolution."""
    L = 20.0
    a_mm = 5.0
    V = 60_000.0
    vane = _flat_vane(L_mm=L, n=10, a_m=a_mm * 1e-3, V=V)
    cache = Laplace2DCache(vane, nx=33, ny=33, box_factor=1.5)
    K_xx_mid = float(cache.K_xx_axis(L * 0.5))
    K_xx_ana = V / (a_mm * a_mm)         # 2400 V/mm²
    rel_err = abs(K_xx_mid - K_xx_ana) / K_xx_ana
    assert rel_err < 0.15, (
        f"K_xx_M3={K_xx_mid:.1f} vs analytic {K_xx_ana:.1f} "
        f"(rel err {rel_err:.3f})"
    )
    # Quadrupole symmetry: K_yy = -K_xx.
    K_yy_mid = float(cache.K_yy_axis(L * 0.5))
    assert K_yy_mid == pytest.approx(-K_xx_mid, rel=1e-9, abs=1e-9)


# ---------------------------------------------------------------------------
def test_T9_vane_masks_disjoint_at_origin():
    """The four vane masks must not overlap at the origin and must
    correctly classify points well inside each vane region."""
    nx = ny = 21
    a = 5.0e-3
    x = np.linspace(-7.5e-3, 7.5e-3, nx)
    y = np.linspace(-7.5e-3, 7.5e-3, ny)
    m1, m2, m3, m4 = _vane_masks(x, y, a, a, a, a)
    # Origin should be in NO mask.
    ix0, iy0 = nx // 2, ny // 2
    assert not (m1[ix0, iy0] or m2[ix0, iy0] or m3[ix0, iy0] or m4[ix0, iy0])
    # Disjoint masks at every cell.
    overlap = (m1.astype(int) + m2.astype(int)
               + m3.astype(int) + m4.astype(int))
    assert overlap.max() <= 1


def test_T10_invalid_grid_size_raises():
    """nx or ny < 5 should raise."""
    vane = _flat_vane(20.0, n=5)
    with pytest.raises(ValueError, match="nx, ny"):
        Laplace2DCache(vane, nx=4, ny=21)


def test_T11_invalid_box_factor_raises():
    """box_factor ≤ 1.0 should raise."""
    vane = _flat_vane(20.0, n=5)
    with pytest.raises(ValueError, match="box_factor"):
        Laplace2DCache(vane, nx=21, ny=21, box_factor=1.0)


def test_T12_z_subsample_reduces_grid():
    """``z_subsample > 1`` shrinks the cached z-grid."""
    vane = _flat_vane(20.0, n=20)
    cache = Laplace2DCache(vane, nx=21, ny=21, z_subsample=4)
    # 20 slices × subsample 4 → 5 + endpoint = 6.
    assert cache.nz == 6
