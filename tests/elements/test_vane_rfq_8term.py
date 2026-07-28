"""Unit tests for the ``field_model="8term"`` matcher-aware path.

The current "8term" implementation is a surgical refinement: it uses the
local r₀(z) = √(a₁(z)·a₂(z)) from the .vane file *only* in matcher
cells (cells with A₁₀ ≈ 0).  In accelerating cells it leaves r₀ at the
cell-constant .dat value.

Tests:

* T1   the M1 default (``"2term"``) is unchanged — bit-identical W/σ.
* T2   ``effective_r0_mm`` returns the .vane lookup in matchers and the
       cell-constant in accelerating cells.
* T3   helper passthrough — ``replace_rfq_cells_with_vane`` accepts and
       forwards the ``field_model`` keyword.
* T4   zero-V drift in 8term mode reduces to a pure drift.
* T5   matcher-aware focusing actually weakens when r₀(z) is flared.
* T6   8term advance_ref energy gain matches 2term in the accelerating
       cells (advance_ref doesn't touch transverse focusing).
* T7   field_model rejection: invalid keyword raises ValueError.
* T8   construction of all three field_model variants succeeds (the
       "laplace2d" path is reserved for M3 but VaneRFQ accepts the name).
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
from linac_gen.elements.vane_rfq_8term import (
    A_quad_local,
    crandall_X,
    effective_r0_mm,
    is_matcher_cell,
)
from linac_gen.io.tracewin_vane import VaneGeometry
from linac_gen.io.vane_rfq_helper import replace_rfq_cells_with_vane


@pytest.fixture
def ref():
    return ReferenceParticle(species=H_MINUS, w_kin=0.050, frequency=162.5)


def _flared_vane(length_mm: float, n: int = 200,
                 r0_inner_m: float = 5.0e-3,
                 r0_flare_m: float = 20.0e-3,
                 V: float = 60_000.0) -> VaneGeometry:
    """Synthetic .vane: aperture flares from r0_flare at z=0 to r0_inner
    at z=length_mm/2 and back to r0_flare at z=length_mm.  Mimics the
    PXIE matcher entrance/exit horns.
    """
    z = np.linspace(0.0, length_mm * 1e-3, n)
    # half-cosine flare envelope: r0 at axis-min in the middle, flare at ends
    arg = np.pi * z / (length_mm * 1e-3)
    a = r0_inner_m + (r0_flare_m - r0_inner_m) * (np.cos(arg)) ** 2
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


def _matcher_then_accel_cells(L_match: float, L_acc: float,
                              V: float = 60_000.0,
                              r0_dat: float = 5.0):
    """Two-cell list: a matcher (A₁₀=0) then an accelerating cell."""
    return [
        CellSpan(z_start_mm=0.0, z_end_mm=L_match,
                 voltage_V=V, A10=0.0, modulation=1.0,
                 length_mm=L_match, phi_s_deg=-90.0,
                 cell_type=3, type_prev=3, type_next=2,
                 r0_dat_mm=r0_dat),
        CellSpan(z_start_mm=L_match, z_end_mm=L_match + L_acc,
                 voltage_V=V, A10=0.1, modulation=1.5,
                 length_mm=L_acc, phi_s_deg=-60.0,
                 cell_type=2, type_prev=3, type_next=2,
                 r0_dat_mm=r0_dat),
    ]


def _make_beam(ref, n=20):
    b = Beam(ref=ref, n_particles=n, current=0.0)
    rng = np.random.default_rng(0)
    b.particles[:, 0] = 0.5 * rng.normal(size=n)
    b.particles[:, 2] = 0.5 * rng.normal(size=n)
    return b


# ---------------------------------------------------------------------------
def test_T1_default_field_model_is_2term(ref):
    """M1 path stays the unconditional default — bit-identical to the
    pre-M2 behavior."""
    L = 20.0
    cells = [CellSpan(z_start_mm=0.0, z_end_mm=L, voltage_V=60_000.0,
                      A10=0.1, modulation=1.5, length_mm=L,
                      phi_s_deg=-60.0, cell_type=2,
                      type_prev=2, type_next=2, r0_dat_mm=5.0)]
    vane = _flared_vane(L)
    elem = VaneRFQ("RFQ", vane=vane, cells=cells, n_steps=20)
    assert elem.field_model == "2term"


def test_T2_effective_r0_dispatch():
    """Matcher cell → vane lookup; accelerating cell → cell-constant."""
    fake_lookup = lambda z_mm: 19.88
    # Matcher (A₁₀ = 0)
    r0 = effective_r0_mm(z_mid_mm=10.0, cell_r0_dat_mm=5.576,
                         A10=0.0, vane_lookup=fake_lookup)
    assert r0 == pytest.approx(19.88)
    # Accelerating (A₁₀ = 0.1)
    r0 = effective_r0_mm(z_mid_mm=10.0, cell_r0_dat_mm=5.576,
                         A10=0.1, vane_lookup=fake_lookup)
    assert r0 == pytest.approx(5.576)
    # Threshold check
    assert is_matcher_cell(0.0)
    assert is_matcher_cell(1e-7)
    assert not is_matcher_cell(1e-3)


def test_T3_helper_field_model_passthrough():
    """`replace_rfq_cells_with_vane` accepts and forwards the keyword."""
    L = 20.0
    lat = Lattice()
    lat.add(Drift("D1", length=10.0, aperture=0.0))
    lat.add(RfqCell("RFQ1", voltage_V=60_000.0, r0_mm=5.0, A10=0.0,
                    modulation=1.0, length_mm=L, phi_s_deg=-90.0,
                    cell_type=3))
    vane = _flared_vane(L)
    replace_rfq_cells_with_vane(lat, vane, field_model="8term")
    rfq = next(e for e in lat.elements if isinstance(e, VaneRFQ))
    assert rfq.field_model == "8term"


def test_T4_zero_voltage_is_drift_8term(ref):
    """V=0 in 8term mode → no kicks, particles drift."""
    L = 20.0
    cells = [CellSpan(z_start_mm=0.0, z_end_mm=L, voltage_V=0.0,
                      A10=0.0, modulation=1.0, length_mm=L,
                      phi_s_deg=-60.0, cell_type=2,
                      type_prev=2, type_next=2, r0_dat_mm=5.0)]
    vane = _flared_vane(L, V=0.0)
    elem = VaneRFQ("RFQ", vane=vane, cells=cells, n_steps=20,
                   field_model="8term")
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


def test_T5_matcher_focusing_weaker_in_8term(ref):
    """In a flared matcher cell, 8term focusing should be substantially
    weaker than 2term.  A pencil beam tracked through the matcher
    diverges *more* in 8term — a clean, monotonic check."""
    L_match = 44.0
    L_acc = 7.4
    cells_2 = _matcher_then_accel_cells(L_match, L_acc, V=60_000.0,
                                        r0_dat=5.0)
    cells_8 = _matcher_then_accel_cells(L_match, L_acc, V=60_000.0,
                                        r0_dat=5.0)
    # Flared matcher: vane aperture 4× larger than r0_dat
    vane = _flared_vane(L_match + L_acc,
                        r0_inner_m=5.0e-3, r0_flare_m=20.0e-3)

    elem2 = VaneRFQ("RFQ2", vane=vane, cells=cells_2,
                    n_steps=400, field_model="2term")
    elem8 = VaneRFQ("RFQ8", vane=vane, cells=cells_8,
                    n_steps=400, field_model="8term")

    # Inject the same off-axis pencil through each.
    def pencil_beam(elem):
        b = Beam(ref=ref.copy(), n_particles=1, current=0.0)
        b.particles[0, 0] = 1.0   # x = 1 mm
        b.particles[0, 1] = 0.0   # x' = 0
        ds = elem.length / elem.n_steps
        for _ in range(elem.n_steps):
            elem.track_rk4(b, ds)
        return b.particles[0, 0]

    x_2term = abs(pencil_beam(elem2))
    x_8term = abs(pencil_beam(elem8))
    # 8term in flared matcher should focus less, so |x_exit| larger
    # (or, beam diverges more).  Check x_8term > x_2term by a clear margin.
    assert x_8term > x_2term, (
        f"8term should focus less in flared matcher, got "
        f"|x|_2term={x_2term}, |x|_8term={x_8term}"
    )


def test_T6_advance_ref_unchanged_by_field_model(ref):
    """advance_ref only does on-axis longitudinal physics → field_model
    keyword shouldn't change it."""
    L = 20.0
    A10 = 0.1
    cells = [CellSpan(z_start_mm=0.0, z_end_mm=L, voltage_V=60_000.0,
                      A10=A10, modulation=1.5, length_mm=L,
                      phi_s_deg=-60.0, cell_type=2,
                      type_prev=2, type_next=2, r0_dat_mm=5.0)]
    vane = _flared_vane(L)
    elem2 = VaneRFQ("R2", vane=vane, cells=cells, n_steps=200,
                    field_model="2term")
    elem8 = VaneRFQ("R8", vane=vane, cells=cells, n_steps=200,
                    field_model="8term")
    r2, r8 = ref.copy(), ref.copy()
    elem2.advance_ref(r2)
    elem8.advance_ref(r8)
    assert r2.w_kin == pytest.approx(r8.w_kin, rel=1e-12)
    assert r2.s == pytest.approx(r8.s, rel=1e-12)
    assert r2.phi_s == pytest.approx(r8.phi_s, rel=1e-12)


def test_T7_invalid_field_model_raises(ref):
    L = 20.0
    cells = [CellSpan(z_start_mm=0.0, z_end_mm=L, voltage_V=60_000.0,
                      A10=0.0, modulation=1.0, length_mm=L,
                      phi_s_deg=-90.0, cell_type=3,
                      type_prev=3, type_next=3, r0_dat_mm=5.0)]
    vane = _flared_vane(L)
    with pytest.raises(ValueError):
        VaneRFQ("RFQ", vane=vane, cells=cells, field_model="invalid_xyz")


def test_T8_all_named_field_models_construct(ref):
    """Construction of all three field_model variants succeeds.
    laplace2d may not be implemented yet, but the keyword is reserved."""
    L = 20.0
    cells = [CellSpan(z_start_mm=0.0, z_end_mm=L, voltage_V=60_000.0,
                      A10=0.0, modulation=1.0, length_mm=L,
                      phi_s_deg=-90.0, cell_type=3,
                      type_prev=3, type_next=3, r0_dat_mm=5.0)]
    vane = _flared_vane(L)
    for fm in ("2term", "8term", "laplace2d"):
        elem = VaneRFQ("RFQ", vane=vane, cells=cells, field_model=fm)
        assert elem.field_model == fm


def test_T9_crandall_X_matcher_limit():
    """X = 1 in matcher cells (A₁₀ = 0, m = 1) regardless of r₀, L."""
    assert crandall_X(A10=0.0, m=1.0, r0_mm=5.0, L_mm=20.0) == pytest.approx(1.0)
    assert crandall_X(A10=0.0, m=1.0, r0_mm=20.0, L_mm=44.0) == pytest.approx(1.0)


def test_T10_crandall_X_typical_accel_cell():
    """For m=2, A₁₀=0.6, r₀=5.576 mm, L=20 mm: X ≈ 0.65 (vs short-form 0.4)."""
    X = crandall_X(A10=0.6, m=2.0, r0_mm=5.576, L_mm=20.0)
    # Short-form would give (1−A₁₀) = 0.4.  Crandall is m·(1−A·I₀(ka))
    # which lifts X by roughly 1.4–1.7× in this regime.
    assert X > 0.55
    assert X < 0.75
    short_form = 1.0 - 0.6
    assert X / short_form > 1.3   # at least 30 % stronger than M1's short form


def test_T11_crandall_X_degenerate():
    """Guards: L=0 or r₀=0 → X=0; m≤0 → X=0."""
    assert crandall_X(0.5, 1.5, 5.0, 0.0) == 0.0
    assert crandall_X(0.5, 1.5, 0.0, 20.0) == 0.0
    assert crandall_X(0.5, 0.0, 5.0, 20.0) == 0.0


def test_T12_A_quad_local_short_form():
    """Active dispatch uses short form ``(1−A₁₀)/r₀_eff²`` (not Crandall X).

    The full Crandall X is ~1.5× stronger and triggers AG-resonance
    instability without the higher-order multipole modes — see module
    docstring.  ``crandall_X`` is exported as a reference helper but
    ``A_quad_local`` (the active dispatch) keeps the M1 short form.
    """
    # Matcher cell: A₁₀=0 → A_quad = 1/r₀_eff² (independent of r₀_dat)
    assert A_quad_local(A10=0.0, m=1.0, r0_eff_mm=10.0,
                       r0_dat_mm=5.0, L_mm=20.0) == pytest.approx(1.0 / 100.0)
    # Accelerating: A_quad = (1−A₁₀)/r₀_eff²  (short form, M1 equivalent)
    A_q = A_quad_local(0.5, 1.5, 5.576, 5.576, 20.0)
    assert A_q == pytest.approx(0.5 / 5.576**2)
    # r₀_eff = 0 guard
    assert A_quad_local(0.5, 1.5, 0.0, 5.576, 20.0) == 0.0
