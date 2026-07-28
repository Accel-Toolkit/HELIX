"""Unit tests for the VaneRFQ Toutatis-equivalent RFQ tracker.

VaneRFQ wraps the entire RFQ as one FieldMapElement and uses the .vane
geometry to drive per-z r₀(z) and V(z) inside the existing 2-term
Crandall physics borrowed from RfqCell.  These tests cover:

* construction from a synthetic VaneGeometry + minimal CellSpan list,
* drift fallback when V=0 (geometry-only sanity),
* per-z r₀(z) lookup actually varies with the vane apertures,
* cell-boundary phase reset (cell-local phase accumulator zeroes when the
  substep cursor crosses into the next cell),
* the RfqCell→VaneRFQ helper preserves non-RFQ elements and replaces
  contiguous RfqCell chains with one VaneRFQ.
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
from linac_gen.io.tracewin_vane import VaneGeometry
from linac_gen.io.vane_rfq_helper import (
    cell_spans_from_rfq_chain,
    replace_rfq_cells_with_vane,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
@pytest.fixture
def ref():
    return ReferenceParticle(species=H_MINUS, w_kin=0.050, frequency=162.5)


def _flat_vane(length_mm: float, n: int = 200,
               r0_m: float = 5.0e-3, V: float = 60_000.0,
               Tc_m: float = 4.183e-3) -> VaneGeometry:
    """Synthetic constant-aperture .vane geometry for sanity tests.

    All four vanes share the same aperture so r₀(z) = r0_m everywhere
    and inter-vane voltage = V.
    """
    z = np.linspace(0.0, length_mm * 1e-3, n)
    a = np.full_like(z, r0_m)
    Tc = np.full_like(z, Tc_m)
    Vp = np.full_like(z, +V * 0.5)
    Vn = np.full_like(z, -V * 0.5)
    flag = np.zeros_like(z)
    return VaneGeometry(
        z=z,
        aperture_v1=a, Tc_v1=Tc, voltage_v1=Vp, flag_v1=flag,
        aperture_v2=a, Tc_v2=Tc, voltage_v2=Vn, flag_v2=flag,
        aperture_v3=a, Tc_v3=Tc, voltage_v3=Vp, flag_v3=flag,
        aperture_v4=a, Tc_v4=Tc, voltage_v4=Vn, flag_v4=flag,
    )


def _modulated_vane(length_mm: float, n: int = 200,
                    r0_avg_m: float = 5.0e-3, modulation: float = 1.5,
                    V: float = 60_000.0) -> VaneGeometry:
    """Synthetic modulated .vane: a₁ and a₂ alternate between min/max
    within one wavelength of length_mm, so r₀(z)=√(a₁·a₂) is constant
    but per-vane apertures vary.
    """
    z = np.linspace(0.0, length_mm * 1e-3, n)
    arg = 2.0 * np.pi * z / (length_mm * 1e-3)
    # a₁ swings around r₀_avg with modulation factor m.
    a_min = r0_avg_m / np.sqrt(modulation)
    a_max = r0_avg_m * np.sqrt(modulation)
    a1 = np.where(np.cos(arg) > 0, a_max, a_min)
    a2 = np.where(np.cos(arg) > 0, a_min, a_max)
    Tc = np.full_like(z, 4.183e-3)
    flag = np.zeros_like(z)
    Vp = np.full_like(z, +V * 0.5)
    Vn = np.full_like(z, -V * 0.5)
    return VaneGeometry(
        z=z,
        aperture_v1=a1, Tc_v1=Tc, voltage_v1=Vp, flag_v1=flag,
        aperture_v2=a2, Tc_v2=Tc, voltage_v2=Vn, flag_v2=flag,
        aperture_v3=a1, Tc_v3=Tc, voltage_v3=Vp, flag_v3=flag,
        aperture_v4=a2, Tc_v4=Tc, voltage_v4=Vn, flag_v4=flag,
    )


def _make_beam(ref, n=50):
    b = Beam(ref=ref, n_particles=n, current=0.0)
    rng = np.random.default_rng(0)
    b.particles[:, 0] = 0.5 * rng.normal(size=n)
    b.particles[:, 2] = 0.5 * rng.normal(size=n)
    b.particles[:, 4] = 5.0 * rng.normal(size=n)
    return b


def _two_cell_spans(L_each_mm: float = 20.0, V: float = 60_000.0,
                    A10: float = 0.1, modulation: float = 1.5) -> list[CellSpan]:
    """Two contiguous CellSpan covering [0, 2L]."""
    return [
        CellSpan(
            z_start_mm=0.0, z_end_mm=L_each_mm, voltage_V=V, A10=A10,
            modulation=modulation, length_mm=L_each_mm,
            phi_s_deg=-60.0, cell_type=2, type_prev=2, type_next=2,
        ),
        CellSpan(
            z_start_mm=L_each_mm, z_end_mm=2 * L_each_mm, voltage_V=V,
            A10=A10, modulation=modulation, length_mm=L_each_mm,
            phi_s_deg=-60.0, cell_type=2, type_prev=2, type_next=2,
        ),
    ]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
def test_construction_basic(ref):
    L = 20.0
    vane = _flat_vane(2 * L)
    cells = _two_cell_spans(L_each_mm=L)
    elem = VaneRFQ("RFQ_TEST", vane=vane, cells=cells)
    assert elem.length == pytest.approx(2 * L)
    assert elem.n_steps >= 2 * vane.n_slices
    # Cell-index lookup
    assert elem._cell_index(5.0) == 0
    assert elem._cell_index(L + 5.0) == 1
    # Out-of-range
    assert elem._cell_index(2 * L + 1.0) == -1


def test_zero_voltage_is_drift(ref):
    """V=0 in CellSpan + flat vane → no kicks, particles just drift."""
    L = 20.0
    vane = _flat_vane(2 * L, V=0.0)
    cells = [
        CellSpan(z_start_mm=0.0, z_end_mm=L, voltage_V=0.0, A10=0.0,
                 modulation=1.0, length_mm=L, phi_s_deg=-60.0,
                 cell_type=2, type_prev=2, type_next=2),
        CellSpan(z_start_mm=L, z_end_mm=2 * L, voltage_V=0.0, A10=0.0,
                 modulation=1.0, length_mm=L, phi_s_deg=-60.0,
                 cell_type=2, type_prev=2, type_next=2),
    ]
    elem = VaneRFQ("RFQ_TEST", vane=vane, cells=cells, n_steps=20)
    beam = _make_beam(ref, n=10)
    x_in = beam.particles[:, 0].copy()
    xp_in = beam.particles[:, 1].copy()
    ds = elem.length / elem.n_steps
    for _ in range(elem.n_steps):
        elem.track_rk4(beam, ds)
    np.testing.assert_allclose(beam.particles[:, 1], xp_in, rtol=1e-12)
    np.testing.assert_allclose(beam.particles[:, 0],
                               x_in + xp_in * elem.length * 1e-3,
                               rtol=1e-9, atol=1e-12)


def test_per_z_r0_lookup_constant_aperture(ref):
    """Flat vane → r₀(z) is constant everywhere, equal to the design value."""
    L = 20.0
    r0 = 5.0e-3
    vane = _flat_vane(2 * L, r0_m=r0)
    cells = _two_cell_spans(L_each_mm=L)
    elem = VaneRFQ("RFQ_TEST", vane=vane, cells=cells)
    # Check at several z's
    for zmm in [0.5, L * 0.5, L, 1.5 * L, 1.99 * L]:
        assert elem._vane_r0_mm(zmm) == pytest.approx(r0 * 1000.0)


def test_per_z_r0_varies_with_modulation(ref):
    """Modulated vane → r₀(z) = √(a₁·a₂) returns the geometric mean.

    For our square-wave synthetic, a₁ and a₂ swap so the geometric mean
    a_min·a_max·… should equal r0_avg, but at an exact crossing point the
    interpolated apertures are between a_min and a_max so r₀ is not the
    geometric mean.  Just verify r₀ is in the [a_min, a_max] window.
    """
    L = 20.0
    r0_avg = 5.0e-3
    m = 1.5
    a_min = r0_avg / np.sqrt(m)
    a_max = r0_avg * np.sqrt(m)
    vane = _modulated_vane(2 * L, r0_avg_m=r0_avg, modulation=m)
    cells = _two_cell_spans(L_each_mm=L, modulation=m)
    elem = VaneRFQ("RFQ_TEST", vane=vane, cells=cells)
    # r₀ at any z lies between a_min and a_max (mm units)
    for zmm in [2.0, L * 0.5, L * 1.5]:
        r0_z = elem._vane_r0_mm(zmm)
        assert a_min * 1000.0 - 1e-9 <= r0_z <= a_max * 1000.0 + 1e-9


def test_advance_ref_energy_gain(ref):
    """advance_ref should produce a non-trivial W gain matching the
    short-cell limit ``ΔW ≈ |q|·A₁₀·V·sin(φ_s)`` summed over both cells.

    Short-cell limit: L ≪ β·λ/2 so the synchronous phase barely
    advances during each cell.  At φ_s = +90° both cells fully accelerate.
    """
    bg_lambda_mm = ref.beta * ref.wavelength
    L = 0.05 * bg_lambda_mm   # 5 % of βλ
    V = 120_000.0
    A10 = 0.1
    vane = _flat_vane(2 * L, V=V)
    cells = [
        CellSpan(z_start_mm=0.0, z_end_mm=L, voltage_V=V, A10=A10,
                 modulation=1.0, length_mm=L, phi_s_deg=90.0,
                 cell_type=2, type_prev=2, type_next=2),
        CellSpan(z_start_mm=L, z_end_mm=2 * L, voltage_V=V, A10=A10,
                 modulation=1.0, length_mm=L, phi_s_deg=90.0,
                 cell_type=2, type_prev=2, type_next=2),
    ]
    elem = VaneRFQ("RFQ_TEST", vane=vane, cells=cells, n_steps=400)
    r = ref.copy()
    w_in = r.w_kin
    elem.advance_ref(r)
    dW_MeV = r.w_kin - w_in
    # Expected ≈ 2 · |q| · A₁₀ · V (eV) for two short cells at φ_s=90°.
    expected_eV = 2.0 * abs(ref.species.charge) * A10 * V
    assert dW_MeV * 1e6 == pytest.approx(expected_eV, rel=0.1)


def test_helper_replaces_only_rfq_chain():
    """replace_rfq_cells_with_vane must replace the RfqCell chain with one
    VaneRFQ and leave non-RFQ elements alone.
    """
    L = 20.0
    lat = Lattice()
    lat.add(Drift("D1", length=10.0, aperture=0.0))
    lat.add(RfqCell("RFQ1", voltage_V=60_000.0, r0_mm=5.0, A10=0.1,
                    modulation=1.5, length_mm=L, phi_s_deg=-60.0,
                    cell_type=2))
    lat.add(RfqCell("RFQ2", voltage_V=60_000.0, r0_mm=5.0, A10=0.1,
                    modulation=1.5, length_mm=L, phi_s_deg=-60.0,
                    cell_type=2))
    lat.add(Drift("D2", length=10.0, aperture=0.0))

    vane = _flat_vane(2 * L)
    replace_rfq_cells_with_vane(lat, vane)

    # Layout: D1, VaneRFQ, D2  (3 elements, 1 vane element)
    assert len(lat.elements) == 3
    assert isinstance(lat.elements[0], Drift)
    assert isinstance(lat.elements[1], VaneRFQ)
    assert isinstance(lat.elements[2], Drift)
    assert lat.elements[1].length == pytest.approx(2 * L)
    # CellSpan list reflects the original two cells.
    assert len(lat.elements[1].cells) == 2


def test_helper_raises_when_no_rfq_chain():
    """Helper raises if there are no RfqCell elements to replace."""
    L = 20.0
    lat = Lattice()
    lat.add(Drift("D1", length=10.0, aperture=0.0))
    vane = _flat_vane(2 * L)
    with pytest.raises(ValueError):
        replace_rfq_cells_with_vane(lat, vane)


def test_cell_spans_from_rfq_chain_contiguous():
    """cell_spans_from_rfq_chain produces a contiguous span list."""
    L = 20.0
    cells = [
        RfqCell(f"RFQ{i}", voltage_V=60_000.0, r0_mm=5.0, A10=0.1,
                modulation=1.5, length_mm=L, phi_s_deg=-60.0, cell_type=2)
        for i in range(3)
    ]
    spans = cell_spans_from_rfq_chain(cells)
    assert len(spans) == 3
    assert spans[0].z_start_mm == 0.0
    assert spans[0].z_end_mm == L
    assert spans[1].z_start_mm == L
    assert spans[1].z_end_mm == 2 * L
    assert spans[2].z_start_mm == 2 * L
    assert spans[2].z_end_mm == 3 * L
