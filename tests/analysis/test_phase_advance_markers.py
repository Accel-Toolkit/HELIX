"""Regression: the coupled per-cell phase-advance family must stride the
RAW element list, not the significant-element count.

``PeriodicStructure.inner_period_length`` excludes Markers and
zero-length drifts; using it to stride ``lattice.elements`` walked the
wrong cell spans on any lattice with diagnostics inside the period —
i.e. most real TraceWin lattices.  Physics invariant: adding Markers
(zero length, identity transport) must not change any per-cell μ.
"""
from __future__ import annotations

import numpy as np
import pytest

from linac_gen.analysis.period_detect import detect_periods
from linac_gen.analysis.phase_advance import coupled_phase_advance_per_cell
from linac_gen.core.lattice import Lattice
from linac_gen.core.particle import PROTON
from linac_gen.core.reference import ReferenceParticle
from linac_gen.elements.drift import Drift
from linac_gen.elements.marker import Marker
from linac_gen.elements.quadrupole import Quadrupole


def _fodo(with_markers: bool) -> Lattice:
    lat = Lattice()
    for n in range(3):
        lat.add(Drift(name="D", length=100.0, aperture=10.0))
        lat.add(Quadrupole(name="QF", length=50.0, gradient=+10.0,
                           aperture=10.0))
        if with_markers:
            lat.add(Marker(name=f"BPM_{n}", is_bpm=True))
        lat.add(Drift(name="D", length=100.0, aperture=10.0))
        lat.add(Quadrupole(name="QD", length=50.0, gradient=-10.0,
                           aperture=10.0))
    return lat


def _ref() -> ReferenceParticle:
    return ReferenceParticle(species=PROTON, w_kin=2.5, frequency=162.5)


def _percell_mu(lat):
    periods = detect_periods(lat)
    cell = next(p for p in periods if p.n_repeats >= 3)
    res = coupled_phase_advance_per_cell(lat, _ref(), cell)
    return np.asarray(res["mu_I_deg"], float), \
        np.asarray(res["mu_II_deg"], float)


def test_markers_do_not_change_per_cell_mu():
    mu_I_plain, mu_II_plain = _percell_mu(_fodo(with_markers=False))
    mu_I_mark, mu_II_mark = _percell_mu(_fodo(with_markers=True))

    # Every cell must be resolved (finite) in both variants …
    assert np.all(np.isfinite(mu_I_plain)) and np.all(np.isfinite(mu_I_mark))
    # … and physically identical: a Marker is identity transport.
    np.testing.assert_allclose(mu_I_mark, mu_I_plain, rtol=1e-9)
    np.testing.assert_allclose(mu_II_mark, mu_II_plain, rtol=1e-9)

    # And the repeats of a perfectly periodic lattice must agree with
    # each other (the old stride made cell 2+ span wrong elements).
    assert np.allclose(mu_I_mark, mu_I_mark[0], rtol=1e-9)
    assert np.allclose(mu_II_mark, mu_II_mark[0], rtol=1e-9)
