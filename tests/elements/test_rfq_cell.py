"""Unit tests for the TraceWin-compatible RFQ_CELL element.

Tests the substep integration against analytic predictions of the
on-axis longitudinal field (manual image 268) and the per-Type
transverse coefficients (images 281–290).  No TraceWin reference output
is available in the repo, so these tests cross-check the model against
its own published equations rather than against TraceWin numerics.
"""
import numpy as np
import pytest

from linac_gen.core.particle import H_MINUS
from linac_gen.core.reference import ReferenceParticle
from linac_gen.core.beam import Beam
from linac_gen.core.lattice import Lattice
from linac_gen.elements.rfq_cell import RfqCell


@pytest.fixture
def ref():
    # 50 keV H- at 162.5 MHz — typical post-source pre-RFQ energy
    return ReferenceParticle(species=H_MINUS, w_kin=0.050, frequency=162.5)


def _make_beam(ref, n=200):
    b = Beam(ref=ref, n_particles=n, current=0.0)
    rng = np.random.default_rng(0)
    b.particles[:, 0] = 0.5 * rng.normal(size=n)   # mm
    b.particles[:, 2] = 0.5 * rng.normal(size=n)
    b.particles[:, 4] = 5.0 * rng.normal(size=n)   # deg
    return b


# ---------------------------------------------------------------------------
def test_rfq_cell_constructs_and_default_A_quad(ref):
    cell = RfqCell(name="rfq1", voltage_V=120000.0, r0_mm=5.0,
                   A10=0.1, modulation=1.5, length_mm=20.0,
                   phi_s_deg=-60.0, cell_type=2)
    # Default A_quad = (1 - A10)/Ro²  in 1/mm²
    assert cell._A_quad == pytest.approx((1.0 - 0.1) / 25.0)
    assert cell.length == 20.0
    assert cell.cell_type == 2


def test_E_z_zero_at_endpoints(ref):
    """E_z(z, t) is sin(πz/L), so it must vanish at z=0 and z=L."""
    cell = RfqCell(name="rfq", voltage_V=120000.0, r0_mm=5.0, A10=0.1,
                   modulation=1.5, length_mm=20.0, phi_s_deg=-60.0,
                   cell_type=2)
    assert cell._Ez_onaxis(0.0, np.pi / 2) == 0.0
    assert abs(cell._Ez_onaxis(20.0, np.pi / 2)) < 1e-9


def test_E_z_peak_value_matches_formula(ref):
    """At z = L/2 and the peak phase, E_z = π·A₁₀·V / (2 L)."""
    L = 20.0
    V = 120000.0
    A10 = 0.1
    cell = RfqCell(name="rfq", voltage_V=V, r0_mm=5.0, A10=A10,
                   modulation=1.5, length_mm=L, phi_s_deg=0.0,
                   cell_type=2)
    expected_peak = np.pi * A10 * V / (2.0 * L)         # V/mm
    got = cell._Ez_onaxis(L / 2.0, np.pi / 2)           # phase at peak
    assert got == pytest.approx(expected_peak, rel=1e-12)


def test_synchronous_energy_gain_sign_and_short_cell_limit(ref):
    """For a cell much shorter than β·λ/2 the synchronous phase barely
    advances during the cell, so the integrated dW reduces to

        dW ≈ q · A₁₀ · V · sin(φ_s)

    (analytic ∫₀ᴸ sin(πz/L) dz = 2L/π cancels the π/2L prefactor).
    Verify (a) the sign flips with φ_s, (b) the magnitude in the short-
    cell limit matches q·A₁₀·V·sin(φ_s) within a few percent.

    Realistic RFQ cells satisfy L ≈ β·λ/2, in which case a transit-time
    factor of π/4 enters; we don't claim that as an exact match here.
    """
    V = 120000.0
    A10 = 0.1
    bg_lambda_mm = ref.beta * ref.wavelength
    L = 0.05 * bg_lambda_mm   # 5% of βλ — phase barely sweeps
    n_steps = 200

    # φ_s = 90° → maximum acceleration (sin = +1).
    cell_pos = RfqCell(name="rfq", voltage_V=V, r0_mm=5.0, A10=A10,
                       modulation=1.0, length_mm=L, phi_s_deg=90.0,
                       cell_type=2, n_steps=n_steps)
    beam = _make_beam(ref.copy(), n=10)
    w_in = beam.ref.w_kin
    ds = L / n_steps
    for _ in range(n_steps):
        cell_pos.track_rk4(beam, ds)
    dW_pos = beam.ref.w_kin - w_in

    # φ_s = -90° → maximum deceleration.
    cell_neg = RfqCell(name="rfq", voltage_V=V, r0_mm=5.0, A10=A10,
                       modulation=1.0, length_mm=L, phi_s_deg=-90.0,
                       cell_type=2, n_steps=n_steps)
    beam2 = _make_beam(ref.copy(), n=10)
    w_in2 = beam2.ref.w_kin
    for _ in range(n_steps):
        cell_neg.track_rk4(beam2, ds)
    dW_neg = beam2.ref.w_kin - w_in2

    # (a) Sign flips with φ_s.  Note the absolute |q| convention
    #     (manual image 265): for *both* H+ and H-, φ_s = +90° gives
    #     positive ΔW, φ_s = −90° gives negative ΔW.  No species-sign
    #     flip in the formula.
    assert dW_pos > 0
    assert dW_neg < 0
    # (b) Short-cell limit: |dW| ≈ A₁₀ · V (q magnitude is 1 here).
    #     Allow 10 % for the residual transit-time factor at L = 0.05·βλ.
    expected_eV = abs(ref.species.charge) * A10 * V
    assert abs(dW_pos) * 1e6 == pytest.approx(expected_eV, rel=0.1)
    assert abs(dW_neg) * 1e6 == pytest.approx(expected_eV, rel=0.1)


def test_zero_voltage_is_drift(ref):
    """V = 0 → no kicks anywhere, particles drift through unchanged."""
    L = 20.0
    cell = RfqCell(name="rfq", voltage_V=0.0, r0_mm=5.0, A10=0.0,
                   modulation=1.0, length_mm=L, phi_s_deg=-60.0,
                   cell_type=2, n_steps=10)
    beam = _make_beam(ref, n=50)
    x_in = beam.particles[:, 0].copy()
    xp_in = beam.particles[:, 1].copy()
    ds = L / 10
    for _ in range(10):
        cell.track_rk4(beam, ds)
    # Pure drift: x' unchanged, x advanced by xp·L
    np.testing.assert_allclose(beam.particles[:, 1], xp_in, rtol=1e-12)
    np.testing.assert_allclose(beam.particles[:, 0],
                               x_in + xp_in * L * 1e-3,
                               rtol=1e-9, atol=1e-12)


def test_per_type_C2_envelope_shapes(ref):
    """Verify the C₂(z) envelope shapes per the TraceWin manual:
       Type ±2 → cos(πz/L)            (manual img 282)
       Type ±3 → 0                    (manual img 285 — no C₂ term in front-end cells)
       Type +4 → ½(cos(πz/L) + 1)     (manual img 289)
       Type −4 → ½(1 − cos(πz/L))     (manual img 291)
    """
    L = 10.0
    z = np.linspace(0.0, L, 11)
    cases = [
        (2,  np.cos(np.pi * z / L)),
        (3,  np.zeros_like(z)),
        (-3, np.zeros_like(z)),
        (4,  0.5 * (np.cos(np.pi * z / L) + 1.0)),
        (-4, 0.5 * (1.0 - np.cos(np.pi * z / L))),
    ]
    for cell_type, expected in cases:
        cell = RfqCell(name=f"rfq{cell_type}", voltage_V=1.0, r0_mm=5.0,
                       A10=0.1, modulation=1.0, length_mm=L,
                       phi_s_deg=0.0, cell_type=cell_type)
        got = np.array([cell._type_coeffs(zi)[1] for zi in z])
        np.testing.assert_allclose(got, expected, atol=1e-12)


def test_S_sign_follows_neighbour_for_type3(ref):
    """For Type +3, S = −sign(type_next); flipping the next-cell type
    must flip S."""
    L = 10.0
    cell_a = RfqCell(name="rfq_a", voltage_V=1.0, r0_mm=5.0, A10=0.1,
                     modulation=1.0, length_mm=L, phi_s_deg=0.0,
                     cell_type=3, type_next=2)
    cell_b = RfqCell(name="rfq_b", voltage_V=1.0, r0_mm=5.0, A10=0.1,
                     modulation=1.0, length_mm=L, phi_s_deg=0.0,
                     cell_type=3, type_next=-2)
    _, _, S_a = cell_a._type_coeffs(L * 0.5)
    _, _, S_b = cell_b._type_coeffs(L * 0.5)
    assert S_a == -1
    assert S_b == +1


def test_parser_reads_rfq_cell_lines(tmp_path):
    """A .dat file with a chain of RFQ_CELL lines must produce a
    sequence of RfqCell elements with the correct type_prev / type_next
    back-links."""
    from linac_gen.io.tracewin_parser import parse_tracewin
    dat = tmp_path / "rfq_chain.dat"
    dat.write_text(
        "FREQ 162.5\n"
        "RFQ_CELL 120000 5.0  0.1   1.0  20.0  -60  3\n"
        "RFQ_CELL 120000 5.0  0.11  1.1  21.0  -60  4\n"
        "RFQ_CELL 120000 5.0  0.12  1.2  22.0  -60  2\n"
        "END\n"
    )
    lat, _ = parse_tracewin(str(dat))
    rfq_cells = [e for e in lat.elements if isinstance(e, RfqCell)]
    assert len(rfq_cells) == 3
    # Type chain: 3 → 4 → 2
    assert [c.cell_type for c in rfq_cells] == [3, 4, 2]
    # Back-links populated
    assert rfq_cells[0].type_next == 4
    assert rfq_cells[1].type_prev == 3
    assert rfq_cells[1].type_next == 2
    assert rfq_cells[2].type_prev == 4
