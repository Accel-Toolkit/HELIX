"""Tests for structure_phase_advance / beam_phase_advance."""
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
from linac_gen.matching.periodic import find_periodic_twiss
from linac_gen.analysis.period_detect import detect_periods, PeriodicStructure
from linac_gen.analysis.phase_advance import (
    structure_phase_advance, beam_phase_advance,
    structure_phase_advance_along_s, beam_phase_advance_along_s,
    coupled_phase_advance, coupled_phase_advance_per_cell,
    coupled_beam_phase_advance_per_cell, _normal_mode_beta_from_sigma,
)


def _fodo() -> Lattice:
    lat = Lattice()
    for _ in range(3):
        lat.add(Drift(name="D", length=100.0, aperture=10.0))
        lat.add(Quadrupole(name="QF", length=50.0, gradient=+10.0, aperture=10.0))
        lat.add(Drift(name="D", length=100.0, aperture=10.0))
        lat.add(Quadrupole(name="QD", length=50.0, gradient=-10.0, aperture=10.0))
    return lat


def _ref() -> ReferenceParticle:
    return ReferenceParticle(species=PROTON, w_kin=2.5, frequency=162.5)


def test_structure_phase_advance_matches_find_periodic_twiss():
    """Per-cell μ × n_repeats == whole-lattice μ for a non-accelerating FODO."""
    lat = _fodo()
    ref = _ref()
    periods = detect_periods(lat)
    cell = next(p for p in periods if p.source == "type_sequence")
    res = structure_phase_advance(lat, ref, cell)
    fpt = find_periodic_twiss(lat, ref)
    # 3 reps × cell μ should equal the whole-lattice μ to fp precision.
    assert res["stable_x"] and res["stable_y"]
    assert pytest.approx(res["mu_x_deg"] * cell.n_repeats, abs=1e-6) == fpt["mu_x"]
    assert pytest.approx(res["mu_y_deg"] * cell.n_repeats, abs=1e-6) == fpt["mu_y"]


def test_structure_phase_advance_no_energy_change_in_quad_only_section():
    """A FODO with no gaps has dW = 0."""
    lat = _fodo()
    ref = _ref()
    periods = detect_periods(lat)
    cell = next(p for p in periods if p.source == "type_sequence")
    res = structure_phase_advance(lat, ref, cell)
    assert res["dw"] == pytest.approx(0.0, abs=1e-9)
    assert res["w_in"] == pytest.approx(res["w_out"])


def test_structure_phase_advance_labels_accelerating_section():
    """A period containing an RFGap reports dW > 0 — but is still computed."""
    lat = Lattice()
    lat.add(Drift(name="D",  length=100.0, aperture=10.0))
    lat.add(Quadrupole(name="QF", length=50.0, gradient=10.0, aperture=10.0))
    lat.add(Drift(name="D",  length=100.0, aperture=10.0))
    # Voltage of the gap is in MV, phase in degrees, freq in MHz.
    lat.add(RFGap(name="G",  voltage=0.5, phase=-30.0, frequency=162.5))
    lat.add(Drift(name="D",  length=100.0, aperture=10.0))
    lat.add(Quadrupole(name="QD", length=50.0, gradient=-10.0, aperture=10.0))
    period = PeriodicStructure(
        start=0, end=len(lat.elements),
        inner_period_length=len(lat.elements),
        inner_slice_end=len(lat.elements),
        n_repeats=1, label="cell", source="manual",
    )
    res = structure_phase_advance(lat, _ref(), period)
    assert res["dw"] > 0.0
    # Even with energy gain, transverse μ is still defined at entry energy.
    # Stability is lattice-dependent — accept either outcome but the call
    # must not raise.
    assert res["w_in"] == pytest.approx(2.5)
    assert res["w_out"] > res["w_in"]


def test_beam_phase_advance_matched_returns_finite_mu():
    """Build a fake EnvelopeResults with a known constant β(s) and check
    the integrator gives μ = (Δs[mm] / β[mm/mrad]) · 180/π / 1000.

    s is in mm, β is in mm/mrad (equivalent to m/rad numerically), so
    ∫ds/β yields mrad, not rad — convert mrad→deg."""
    from linac_gen.tracking.envelope import EnvelopeResults
    n = 11
    s = np.linspace(0.0, 200.0, n).tolist()
    beta = 5.0
    results = EnvelopeResults(
        s=list(s),
        sigma_x=[1.0] * n, sigma_y=[1.0] * n,
        sigma_phi=[0.0] * n, sigma_w=[0.0] * n,
        emit_x=[1.0] * n, emit_y=[1.0] * n,
        emit_z=[0.0] * n, emit_z_mmmrad=[0.0] * n,
        alpha_x=[0.0] * n, beta_x=[beta] * n,
        alpha_y=[0.0] * n, beta_y=[beta] * n,
        ref_w_kin=[2.5] * n, ref_beta=[0.07] * n, ref_gamma=[1.0] * n,
        sigma_matrix=[],
        element_names=[f"E{i}" for i in range(n)],
    )
    period = PeriodicStructure(
        start=0, end=n, inner_period_length=n - 1,
        inner_slice_end=n, n_repeats=1, label="cell", source="manual",
    )
    out = beam_phase_advance(results, period)
    expected = (200.0 / beta) * (180.0 / math.pi) * 1e-3
    assert out["mu_x_deg"] == pytest.approx(expected, rel=1e-6)
    assert out["mu_y_deg"] == pytest.approx(expected, rel=1e-6)
    assert out["matched"] is True   # constant β


def test_beam_phase_advance_flags_mismatch():
    """β(end) ≠ β(start) ⇒ matched=False."""
    from linac_gen.tracking.envelope import EnvelopeResults
    n = 5
    results = EnvelopeResults(
        s=[0.0, 50.0, 100.0, 150.0, 200.0],
        sigma_x=[1.0] * n, sigma_y=[1.0] * n,
        sigma_phi=[0.0] * n, sigma_w=[0.0] * n,
        emit_x=[1.0] * n, emit_y=[1.0] * n,
        emit_z=[0.0] * n, emit_z_mmmrad=[0.0] * n,
        alpha_x=[0.0] * n, beta_x=[1.0, 1.5, 2.0, 2.5, 3.0],   # blowing up
        alpha_y=[0.0] * n, beta_y=[1.0, 1.0, 1.0, 1.0, 1.0],
        ref_w_kin=[2.5] * n, ref_beta=[0.07] * n, ref_gamma=[1.0] * n,
        sigma_matrix=[],
        element_names=[f"E{i}" for i in range(n)],
    )
    period = PeriodicStructure(
        start=0, end=n, inner_period_length=n - 1,
        inner_slice_end=n, n_repeats=1, label="cell", source="manual",
    )
    out = beam_phase_advance(results, period)
    assert out["matched"] is False
    assert out["mismatch_x"] > 0.5     # 1.0 → 3.0 = 200% change


def test_structure_phase_advance_along_s_accumulates_per_cell():
    """μ₀(s) at the end of n cells should equal n × per-cell μ₀."""
    lat = _fodo()
    ref = _ref()
    period = next(p for p in detect_periods(lat) if p.source == "type_sequence")
    sigma0 = structure_phase_advance(lat, ref, period)
    curves = structure_phase_advance_along_s(lat, ref, period)
    # End of cell k → element index period.start + k * inner_period_length.
    for k in range(1, period.n_repeats + 1):
        idx = period.start + k * period.inner_period_length
        assert curves["mu_x_deg"][idx] == pytest.approx(
            k * sigma0["mu_x_deg"], abs=1e-6
        )
        assert curves["mu_y_deg"][idx] == pytest.approx(
            k * sigma0["mu_y_deg"], abs=1e-6
        )


def test_beam_phase_advance_along_s_constant_beta():
    """Constant β over Δs of 200 mm → μ = (Δs/β) · 180/π."""
    from linac_gen.tracking.envelope import EnvelopeResults
    n = 11
    s = np.linspace(0.0, 200.0, n).tolist()
    beta = 4.0
    results = EnvelopeResults(
        s=s, sigma_x=[1.0] * n, sigma_y=[1.0] * n,
        sigma_phi=[0.0] * n, sigma_w=[0.0] * n,
        emit_x=[1.0] * n, emit_y=[1.0] * n,
        emit_z=[0.0] * n, emit_z_mmmrad=[0.0] * n,
        alpha_x=[0.0] * n, beta_x=[beta] * n,
        alpha_y=[0.0] * n, beta_y=[beta] * n,
        ref_w_kin=[2.5] * n, ref_beta=[0.07] * n, ref_gamma=[1.0] * n,
        sigma_matrix=[], element_names=[f"E{i}" for i in range(n)],
    )
    out = beam_phase_advance_along_s(results, start_index=0)
    expected_total = (200.0 / beta) * (180.0 / math.pi) * 1e-3
    assert out["mu_x_deg"][-1] == pytest.approx(expected_total, rel=1e-6)
    assert out["mu_y_deg"][0] == pytest.approx(0.0)


def test_beam_phase_advance_with_sigma0_yields_ratio():
    from linac_gen.tracking.envelope import EnvelopeResults
    n = 5
    results = EnvelopeResults(
        s=[0.0, 50.0, 100.0, 150.0, 200.0],
        sigma_x=[1.0] * n, sigma_y=[1.0] * n,
        sigma_phi=[0.0] * n, sigma_w=[0.0] * n,
        emit_x=[1.0] * n, emit_y=[1.0] * n,
        emit_z=[0.0] * n, emit_z_mmmrad=[0.0] * n,
        alpha_x=[0.0] * n, beta_x=[10.0] * n,    # β = 10 → big period, small μ
        alpha_y=[0.0] * n, beta_y=[10.0] * n,
        ref_w_kin=[2.5] * n, ref_beta=[0.07] * n, ref_gamma=[1.0] * n,
        sigma_matrix=[],
        element_names=[f"E{i}" for i in range(n)],
    )
    period = PeriodicStructure(
        start=0, end=n, inner_period_length=n - 1,
        inner_slice_end=n, n_repeats=1, label="cell", source="manual",
    )
    sigma0 = {"mu_x_deg": 100.0, "mu_y_deg": 100.0}
    out = beam_phase_advance(results, period, sigma0=sigma0)
    assert out["sigma_over_sigma0_x"] == pytest.approx(out["mu_x_deg"] / 100.0)
    assert out["sigma_over_sigma0_y"] == pytest.approx(out["mu_y_deg"] / 100.0)


def test_coupled_phase_advance_decoupled_lattice_matches_xy():
    """For a decoupled FODO (no solenoid), eigenmode μ_I, μ_II must
    coincide with the standard σ₀_x, σ₀_y from compute_twiss."""
    lat = _fodo()
    ref = _ref()
    period = next(p for p in detect_periods(lat) if p.source == "type_sequence")
    decoupled = structure_phase_advance(lat, ref, period)
    coupled = coupled_phase_advance(lat, ref, period)
    expected = sorted([decoupled["mu_x_deg"], decoupled["mu_y_deg"]])
    got = sorted([coupled["mu_I_deg"], coupled["mu_II_deg"]])
    assert got[0] == pytest.approx(expected[0], abs=1e-6)
    assert got[1] == pytest.approx(expected[1], abs=1e-6)
    assert coupled["stable_I"] and coupled["stable_II"]


def test_coupled_phase_advance_handles_solenoid():
    """A simple solenoid-bracketed cell xy-couples the lattice; the
    plain compute_twiss raises but coupled_phase_advance still returns
    finite μ_I, μ_II (the eigenvalues live on the unit circle)."""
    from linac_gen.elements.solenoid import Solenoid
    lat = Lattice()
    lat.add(Drift(name="D", length=50.0, aperture=10.0))
    lat.add(Solenoid(name="SOL", length=200.0, field=0.4, aperture=10.0))
    lat.add(Drift(name="D", length=50.0, aperture=10.0))
    ref = _ref()
    period = PeriodicStructure(
        start=0, end=len(lat.elements),
        inner_period_length=len(lat.elements),
        inner_slice_end=len(lat.elements),
        n_repeats=1, label="solenoid cell", source="manual",
    )
    out = coupled_phase_advance(lat, ref, period)
    assert math.isfinite(out["mu_I_deg"])
    assert math.isfinite(out["mu_II_deg"])
    assert 0.0 <= out["mu_I_deg"] <= 180.0
    assert 0.0 <= out["mu_II_deg"] <= 180.0


def test_coupled_phase_advance_per_cell_decoupled_constant():
    """For a periodic decoupled FODO, every cell gives the same μ_I,
    μ_II — and they match σ₀_x, σ₀_y from compute_twiss."""
    lat = _fodo()
    ref = _ref()
    period = next(p for p in detect_periods(lat) if p.source == "type_sequence")
    per_cell = coupled_phase_advance_per_cell(lat, ref, period)
    assert per_cell["cells"].size == period.n_repeats
    expected = sorted([structure_phase_advance(lat, ref, period)["mu_x_deg"],
                       structure_phase_advance(lat, ref, period)["mu_y_deg"]])
    for k in range(period.n_repeats):
        got = sorted([float(per_cell["mu_I_deg"][k]),
                      float(per_cell["mu_II_deg"][k])])
        assert got[0] == pytest.approx(expected[0], abs=1e-6)
        assert got[1] == pytest.approx(expected[1], abs=1e-6)


def test_normal_mode_beta_from_sigma_decoupled_limit():
    """For a block-diagonal σ (no xy coupling), the eigenmode β from
    σ-eigendecomposition reduces to the standard β_x, β_y."""
    bx, by = 4.0, 6.0      # mm/mrad
    ex, ey = 0.5, 0.7      # mm·mrad
    sigma4 = np.zeros((4, 4))
    sigma4[0, 0] = bx * ex
    sigma4[1, 1] = ex / bx              # γ·ε with α=0
    sigma4[2, 2] = by * ey
    sigma4[3, 3] = ey / by
    out = _normal_mode_beta_from_sigma(sigma4)
    assert out is not None
    # In the decoupled limit, mode I ≈ x, mode II ≈ y; emittances and
    # β functions match.  We accept either ordering of the modes.
    eps = sorted([out["eps_I"], out["eps_II"]])
    bxs = sorted([out["beta_I_x"], out["beta_II_x"]])
    bys = sorted([out["beta_I_y"], out["beta_II_y"]])
    assert eps[0] == pytest.approx(min(ex, ey), rel=1e-6)
    assert eps[1] == pytest.approx(max(ex, ey), rel=1e-6)
    # The largest x-projection should be the x-mode's β_x = bx.
    assert max(bxs) == pytest.approx(bx, rel=1e-6)
    assert max(bys) == pytest.approx(by, rel=1e-6)


def test_coupled_beam_phase_advance_per_cell_decoupled_matches_standard():
    """For a decoupled FODO with constant matched β recorded in
    sigma_matrix, coupled_beam_phase_advance_per_cell agrees with
    beam_phase_advance from β_x/β_y to within numerical noise."""
    from linac_gen.tracking.envelope import EnvelopeResults
    from linac_gen.analysis.period_detect import PeriodicStructure
    n = 21
    s = np.linspace(0, 1500, n).tolist()
    bx, by = 4.0, 6.0
    ex, ey = 0.5, 0.7
    sigmas = []
    for _ in range(n):
        sigma = np.zeros((6, 6))
        sigma[0, 0] = bx * ex; sigma[1, 1] = ex / bx
        sigma[2, 2] = by * ey; sigma[3, 3] = ey / by
        sigmas.append(sigma)
    results = EnvelopeResults(
        s=s, sigma_x=[1.0]*n, sigma_y=[1.0]*n, sigma_phi=[0.0]*n, sigma_w=[0.0]*n,
        emit_x=[ex]*n, emit_y=[ey]*n, emit_z=[0.0]*n, emit_z_mmmrad=[0.0]*n,
        alpha_x=[0.0]*n, beta_x=[bx]*n,
        alpha_y=[0.0]*n, beta_y=[by]*n,
        ref_w_kin=[2.5]*n, ref_beta=[0.07]*n, ref_gamma=[1.0]*n,
        sigma_matrix=sigmas, element_names=[f"E{i}" for i in range(n)],
    )
    period = PeriodicStructure(
        start=0, end=n, inner_period_length=5,
        # Per the PeriodicStructure contract, inner_slice_end is the index
        # immediately after the FIRST repeat (raw span 5 here — the cells
        # have no interleaved markers), NOT the end of the whole bracket.
        # The old value (21) only passed because the per-cell stride
        # wrongly ignored this field and used inner_period_length.
        inner_slice_end=5, n_repeats=4,
        label="cell", source="manual",
    )
    out = coupled_beam_phase_advance_per_cell(results, period)
    # Each cell spans Δs = (1500/20)*5 = 375 mm.  Expected per-cell
    # depressed mode tune ≈ 375/β · 180/π / 1000.  We don't know
    # beforehand which mode picks up x vs y, so check the *set* of mode
    # tunes against {μ_x, μ_y}.
    cell_dz = 1500.0 / 20 * 5
    expected_x = (cell_dz / bx) * (180.0 / np.pi) * 1e-3
    expected_y = (cell_dz / by) * (180.0 / np.pi) * 1e-3
    expected = sorted([expected_x, expected_y])
    for k in range(period.n_repeats):
        got = sorted([float(out["mu_I_deg"][k]), float(out["mu_II_deg"][k])])
        assert got[0] == pytest.approx(expected[0], rel=1e-3)
        assert got[1] == pytest.approx(expected[1], rel=1e-3)


def test_coupled_beam_phase_advance_via_M_zero_current():
    """At I=0 mA the depressed transfer matrix equals the bare matrix,
    so coupled_beam_phase_advance_per_cell_via_M must return σ_I, σ_II
    that exactly equal σ₀_I, σ₀_II from coupled_phase_advance_per_cell."""
    from linac_gen.elements.solenoid import Solenoid
    from linac_gen.tracking.envelope import EnvelopeSolver
    from linac_gen.analysis.phase_advance import (
        coupled_beam_phase_advance_per_cell_via_M,
    )
    lat = Lattice()
    for _ in range(2):
        lat.add(Drift(name="D", length=100.0, aperture=20.0))
        lat.add(Solenoid(name="SOL", length=100.0, field=0.3, aperture=20.0))
        lat.add(Drift(name="D", length=100.0, aperture=20.0))
    ref = _ref()
    period = PeriodicStructure(
        start=0, end=6, inner_period_length=3,
        inner_slice_end=6, n_repeats=2,
        label="sol cell", source="manual",
    )
    initial = dict(alpha_x=0.0, beta_x=1.0, emit_x=0.5,
                   alpha_y=0.0, beta_y=1.0, emit_y=0.5,
                   alpha_z=0.0, beta_z=1.0, emit_z=0.05)
    res = EnvelopeSolver(lat, ref, initial, current=0.0).run()
    cpc = coupled_phase_advance_per_cell(lat, ref, period)
    bpc = coupled_beam_phase_advance_per_cell_via_M(lat, ref, period, res)
    for k in range(period.n_repeats):
        # σ_I, σ_II at I=0 must equal σ₀_I, σ₀_II to numerical precision.
        struct = sorted([float(cpc["mu_I_deg"][k]), float(cpc["mu_II_deg"][k])])
        beam = sorted([float(bpc["mu_I_deg"][k]), float(bpc["mu_II_deg"][k])])
        assert beam[0] == pytest.approx(struct[0], abs=1e-6)
        assert beam[1] == pytest.approx(struct[1], abs=1e-6)


# ---------------------------------------------------------------------------
# Cooperative cancellation (should_stop) — raises, never partial dicts
# ---------------------------------------------------------------------------

def _cell(lat):
    periods = detect_periods(lat)
    return next(p for p in periods if p.source == "type_sequence")


def test_structure_phase_advance_cancel_raises():
    from linac_gen.core.cancelled import OperationCancelled
    lat = _fodo()
    with pytest.raises(OperationCancelled):
        structure_phase_advance(lat, _ref(), _cell(lat),
                                should_stop=lambda: True)


def test_along_s_cancel_raises_and_is_not_swallowed_as_missing_seed():
    """The seed call sits in a try/except Exception — a cancel must
    re-raise, not degrade into 'no seed' NaN curves."""
    from linac_gen.core.cancelled import OperationCancelled
    lat = _fodo()
    with pytest.raises(OperationCancelled):
        structure_phase_advance_along_s(lat, _ref(), _cell(lat),
                                        should_stop=lambda: True)


def test_along_s_cancel_mid_walk():
    from linac_gen.core.cancelled import OperationCancelled
    lat = _fodo()
    cell = _cell(lat)
    seed = structure_phase_advance(lat, _ref(), cell)   # no hook: fine
    polls = {"n": 0}

    def stop_after_two_elements():
        polls["n"] += 1
        return polls["n"] > 2

    with pytest.raises(OperationCancelled):
        structure_phase_advance_along_s(lat, _ref(), cell, seed=seed,
                                        should_stop=stop_after_two_elements)


def test_coupled_along_s_cancel_raises():
    from linac_gen.core.cancelled import OperationCancelled
    from linac_gen.analysis.phase_advance import coupled_phase_advance_along_s
    lat = _fodo()
    with pytest.raises(OperationCancelled):
        coupled_phase_advance_along_s(lat, _ref(), _cell(lat),
                                      should_stop=lambda: True)


def test_no_hook_unchanged():
    lat = _fodo()
    res = structure_phase_advance(lat, _ref(), _cell(lat), should_stop=None)
    assert res["stable_x"] and res["stable_y"]
