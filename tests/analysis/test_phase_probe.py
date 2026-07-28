"""Envelope phase probe + channel tunes (σ_model / σ₀_model).

The probe accumulates, slice by slice, every matrix the envelope solver
applies to Σ — the run's exact tangent map.  Channel tunes come from
det-normalized eigen extraction of the per-cell monodromies, with the
bare (SC→identity) maps accumulated in the SAME walk.

The key acceptance here is the matched-FODO case: the legacy via-M
approximation (whole-element bare map + entrance-σ SC kick) read ~9 %
low against the fine ∫ds/β on a converged matched channel; the probe
monodromy must agree to < 1 %.
"""
from __future__ import annotations

import numpy as np
import pytest

from linac_gen.analysis.period_detect import PeriodicStructure, detect_periods
from linac_gen.analysis.phase_advance import (
    beam_phase_advance, channel_phase_advance,
    coupled_beam_phase_advance_per_cell_via_M,
    coupled_phase_advance_per_cell, element_record_span,
    run_phase_probe, structure_phase_advance,
)
from linac_gen.core.lattice import Lattice
from linac_gen.core.particle import PROTON
from linac_gen.core.reference import ReferenceParticle
from linac_gen.elements.drift import Drift
from linac_gen.elements.quadrupole import Quadrupole
from linac_gen.elements.solenoid import Solenoid
from linac_gen.matching.periodic import find_periodic_twiss
from linac_gen.tracking.envelope import EnvelopeSolver


def _fodo(n_cells: int = 3) -> Lattice:
    lat = Lattice()
    for _ in range(n_cells):
        lat.add(Drift(name="D", length=100.0, aperture=10.0))
        lat.add(Quadrupole(name="QF", length=50.0, gradient=+10.0,
                           aperture=10.0))
        lat.add(Drift(name="D", length=100.0, aperture=10.0))
        lat.add(Quadrupole(name="QD", length=50.0, gradient=-10.0,
                           aperture=10.0))
    return lat


def _ref() -> ReferenceParticle:
    return ReferenceParticle(species=PROTON, w_kin=2.5, frequency=162.5)


def _bare_initial(tw) -> dict:
    return dict(alpha_x=tw["alpha_x"], beta_x=tw["beta_x"], emit_x=1.0,
                alpha_y=tw["alpha_y"], beta_y=tw["beta_y"], emit_y=1.0,
                alpha_z=0.0, beta_z=1.0, emit_z=0.05)


def test_probe_is_purely_additive():
    """σ evolution with the probe on is bit-identical to probe off."""
    lat = _fodo()
    ref = _ref()
    tw = find_periodic_twiss(lat, ref)
    r_off = EnvelopeSolver(lat, ref.copy(), _bare_initial(tw),
                           current=10.0).run()
    r_on = EnvelopeSolver(lat, ref.copy(), _bare_initial(tw),
                          current=10.0, phase_probe=True).run()
    assert r_on.s == r_off.s
    np.testing.assert_array_equal(np.asarray(r_on.sigma_matrix),
                                  np.asarray(r_off.sigma_matrix))
    assert len(r_on.element_maps_dep) == len(lat.elements)
    assert len(r_off.element_maps_dep) == 0


def test_probe_invariant_per_element():
    """M_dep·Σ_in·M_depᵀ == Σ_out for every element (deterministic maps)."""
    lat = _fodo()
    ref = _ref()
    tw = find_periodic_twiss(lat, ref)
    res = EnvelopeSolver(lat, ref.copy(), _bare_initial(tw),
                         current=10.0, phase_probe=True).run()
    for j in range(len(lat.elements)):
        r0, r1 = element_record_span(res, j, j + 1)
        S_in = np.asarray(res.sigma_matrix[r0])
        S_out = np.asarray(res.sigma_matrix[r1])
        M = res.element_maps_dep[j]
        np.testing.assert_allclose(M @ S_in @ M.T, S_out,
                                   rtol=1e-12, atol=1e-15)


def test_channel_i0_bare_equals_dep_and_structure():
    """At I=0: dep maps == bare maps exactly, and the channel tune
    equals the bare-matrix structure_phase_advance to 1e-6."""
    lat = _fodo()
    ref = _ref()
    tw = find_periodic_twiss(lat, ref)
    res = EnvelopeSolver(lat, ref.copy(), _bare_initial(tw),
                         current=0.0, phase_probe=True).run()
    period = next(p for p in detect_periods(lat) if p.n_repeats >= 3)
    ch = channel_phase_advance(res, period)
    assert not ch["coupled_xy"]
    np.testing.assert_array_equal(ch["mu_x_dep_deg"], ch["mu_x_bare_deg"])
    np.testing.assert_array_equal(ch["eta_x"][np.isfinite(ch["eta_x"])], 1.0)

    sigma0 = structure_phase_advance(lat, ref, period)
    for k in range(len(ch["cells"])):
        assert ch["mu_x_bare_deg"][k] == pytest.approx(
            sigma0["mu_x_deg"], abs=1e-6)
        assert ch["mu_y_bare_deg"][k] == pytest.approx(
            sigma0["mu_y_deg"], abs=1e-6)


def _matched_state(lat, ref, current):
    """Matched (α, β) of the first FODO cell WITH SC — periodicity
    residual solved with scipy root (PyORBIT matching.py style)."""
    from scipy.optimize import root

    cell = Lattice()
    for e in lat.elements[:4]:
        cell.add(e)

    def _out(state):
        # Continuous (DC) beam: 2-D SC kick, no longitudinal evolution —
        # a bunched test beam with no RF debunches and weakens its own
        # SC cell by cell, so no strictly periodic solution exists.
        init = dict(alpha_x=state[0], beta_x=state[1], emit_x=1.0,
                    alpha_y=state[2], beta_y=state[3], emit_y=1.0,
                    alpha_z=0.0, beta_z=1.0, emit_z=0.0, continuous=True)
        res = EnvelopeSolver(cell, ref.copy(), init, current=current).run()
        S = np.asarray(res.sigma_matrix[-1])
        out = []
        for i, j in ((0, 1), (2, 3)):
            eps = np.sqrt(max(S[i, i] * S[j, j] - S[i, j] ** 2, 1e-30))
            out += [-S[i, j] / eps, S[i, i] / eps]
        return np.asarray(out)

    tw = find_periodic_twiss(lat, ref)
    seed = np.asarray([tw["alpha_x"], tw["beta_x"],
                       tw["alpha_y"], tw["beta_y"]])
    sol = root(lambda v: _out(v) - v, seed, method="hybr", tol=1e-12)
    err = float(np.abs(_out(sol.x) - sol.x).max())
    return list(sol.x), err


def test_matched_fodo_channel_tune_vs_fine_integral():
    """THE acceptance test: on a converged matched FODO with SC, the
    probe channel tune must agree with the substep ∫ds/β to < 1 %
    (the legacy via-M approximation was ~9 % low here)."""
    lat = _fodo(n_cells=3)
    ref = _ref()
    current = 20.0
    state, err = _matched_state(lat, ref, current)
    assert err < 1e-8, f"matched fixed point did not converge ({err})"

    init = dict(alpha_x=state[0], beta_x=state[1], emit_x=1.0,
                alpha_y=state[2], beta_y=state[3], emit_y=1.0,
                alpha_z=0.0, beta_z=1.0, emit_z=0.0, continuous=True)
    res = EnvelopeSolver(lat, ref.copy(), init, current=current,
                         phase_probe=True, record_substeps=True).run()
    period = next(p for p in detect_periods(lat) if p.n_repeats >= 3)

    ch = channel_phase_advance(res, period)
    beam = beam_phase_advance(res, period)
    assert beam["mismatch_x"] < 1e-6 and beam["mismatch_y"] < 1e-6, beam

    # A DC-matched beam is periodic in EVERY cell: identical dep tunes.
    np.testing.assert_allclose(ch["mu_x_dep_deg"], ch["mu_x_dep_deg"][0],
                               rtol=1e-9)
    # Depression is real at 20 mA (η well below 1) …
    assert ch["eta_x"][0] < 0.95
    # … and the model tune matches the fine beam integral to < 1 %.
    assert ch["mu_x_dep_deg"][0] == pytest.approx(beam["mu_x_deg"], rel=0.01)
    assert ch["mu_y_dep_deg"][0] == pytest.approx(beam["mu_y_deg"], rel=0.01)


def _sol_lattice(n=2) -> Lattice:
    lat = Lattice()
    for _ in range(n):
        lat.add(Drift(name="D", length=100.0, aperture=20.0))
        lat.add(Solenoid(name="SOL", length=100.0, field=0.3, aperture=20.0))
        lat.add(Drift(name="D", length=100.0, aperture=20.0))
    return lat


def test_via_m_probe_path_i0_equals_structure_no_warning():
    """Probe-bearing results through the via-M API at I=0 == σ₀
    eigenmodes, with no deprecation warning."""
    import warnings
    lat = _sol_lattice()
    ref = _ref()
    period = PeriodicStructure(start=0, end=6, inner_period_length=3,
                               inner_slice_end=3, n_repeats=2,
                               label="sol", source="manual")
    initial = dict(alpha_x=0.0, beta_x=1.0, emit_x=0.5,
                   alpha_y=0.0, beta_y=1.0, emit_y=0.5,
                   alpha_z=0.0, beta_z=1.0, emit_z=0.05)
    res = EnvelopeSolver(lat, ref.copy(), initial, current=0.0,
                         phase_probe=True).run()
    cpc = coupled_phase_advance_per_cell(lat, ref, period)
    with warnings.catch_warnings():
        warnings.simplefilter("error")     # any warning fails the test
        bpc = coupled_beam_phase_advance_per_cell_via_M(
            lat, ref, period, res)
    for k in range(2):
        struct = sorted([float(cpc["mu_I_deg"][k]),
                         float(cpc["mu_II_deg"][k])])
        beam = sorted([float(bpc["mu_I_deg"][k]),
                       float(bpc["mu_II_deg"][k])])
        assert beam[0] == pytest.approx(struct[0], abs=1e-6)
        assert beam[1] == pytest.approx(struct[1], abs=1e-6)


def test_via_m_legacy_fallback_warns():
    lat = _sol_lattice()
    ref = _ref()
    period = PeriodicStructure(start=0, end=6, inner_period_length=3,
                               inner_slice_end=3, n_repeats=2,
                               label="sol", source="manual")
    initial = dict(alpha_x=0.0, beta_x=1.0, emit_x=0.5,
                   alpha_y=0.0, beta_y=1.0, emit_y=0.5,
                   alpha_z=0.0, beta_z=1.0, emit_z=0.05)
    res = EnvelopeSolver(lat, ref.copy(), initial, current=0.0).run()
    with pytest.warns(UserWarning, match="phase-probe"):
        coupled_beam_phase_advance_per_cell_via_M(lat, ref, period, res)


def test_run_phase_probe_helper():
    lat = _fodo()
    ref = _ref()
    tw = find_periodic_twiss(lat, ref)
    res = run_phase_probe(lat, ref, _bare_initial(tw), current=5.0)
    assert len(res.element_maps_dep) == len(lat.elements)
    assert len(res.element_maps_bare) == len(lat.elements)
    assert len(res.probe_M) >= len(lat.elements)
