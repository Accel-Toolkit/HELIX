"""Record-span mapping: element spans → record rows under substep recording.

``beam_phase_advance`` (and the coupled via-M σ lookup) used to equate
element indices with record-row indices.  That identity holds only for
one-record-per-element results; with ``record_substeps=True`` the
envelope solver inserts a variable number of interior rows per element
and every span silently landed on the wrong rows.  The fix routes all
translations through ``results.element_exit_idx`` /
``element_record_span``.
"""
from __future__ import annotations

import numpy as np
import pytest

from linac_gen.analysis.period_detect import detect_periods
from linac_gen.analysis.phase_advance import (
    beam_phase_advance, element_record_span,
)
from linac_gen.core.lattice import Lattice
from linac_gen.core.particle import PROTON
from linac_gen.core.reference import ReferenceParticle
from linac_gen.elements.drift import Drift
from linac_gen.elements.quadrupole import Quadrupole
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


def _matched_run(record_substeps: bool):
    lat = _fodo()
    ref = ReferenceParticle(species=PROTON, w_kin=2.5, frequency=162.5)
    tw = find_periodic_twiss(lat, ref)
    initial = dict(alpha_x=tw["alpha_x"], beta_x=tw["beta_x"], emit_x=1.0,
                   alpha_y=tw["alpha_y"], beta_y=tw["beta_y"], emit_y=1.0,
                   alpha_z=0.0, beta_z=1.0, emit_z=0.0)
    res = EnvelopeSolver(lat, ref.copy(), initial, current=0.0,
                         record_substeps=record_substeps).run()
    return lat, res


def test_element_exit_idx_populated_and_identity_without_substeps():
    lat, res = _matched_run(record_substeps=False)
    n_el = len(lat.elements)
    assert len(res.element_exit_idx) == n_el
    # One record per element: exit of element j is row j+1 (row 0=INPUT).
    assert res.element_exit_idx == list(range(1, n_el + 1))
    r0, r1 = element_record_span(res, 0, 4)
    assert (r0, r1) == (0, 4)


def test_span_endpoints_identical_s_with_and_without_substeps():
    lat, coarse = _matched_run(record_substeps=False)
    _, fine = _matched_run(record_substeps=True)
    n_el = len(lat.elements)
    assert len(fine.element_exit_idx) == n_el
    # Substeps inserted extra rows.
    assert len(fine.s) > len(coarse.s)
    # Every element-boundary s must agree between the two runs.
    for j_el in range(1, n_el + 1):
        rc = element_record_span(coarse, 0, j_el)[1]
        rf = element_record_span(fine, 0, j_el)[1]
        assert coarse.s[rc] == pytest.approx(fine.s[rf], abs=1e-9)


def test_beam_phase_advance_uses_mapped_span_under_substeps():
    lat, coarse = _matched_run(record_substeps=False)
    _, fine = _matched_run(record_substeps=True)
    period = next(p for p in detect_periods(lat) if p.n_repeats >= 3)

    mu_c = beam_phase_advance(coarse, period)
    mu_f = beam_phase_advance(fine, period)

    # Same physical span (the matched run makes both well-defined).
    assert mu_f["s_start"] == pytest.approx(mu_c["s_start"], abs=1e-9)
    assert mu_f["s_end"] == pytest.approx(mu_c["s_end"], abs=1e-9)

    # The fine-grid integral refines the coarse trapezoid: agreement at
    # the few-percent level (thick 50 mm quads), not equality.
    assert mu_f["mu_x_deg"] == pytest.approx(mu_c["mu_x_deg"], rel=0.05)
    assert mu_f["mu_y_deg"] == pytest.approx(mu_c["mu_y_deg"], rel=0.05)

    # Regression guard: an UNMAPPED span (the old identity indexing) on
    # the substep results lands on interior rows of the first elements
    # and yields a very different s_end — prove the mapping mattered.
    r0_raw, r1_raw = period.start, period.inner_slice_end
    assert fine.s[r1_raw] != pytest.approx(mu_f["s_end"], abs=1e-6)
