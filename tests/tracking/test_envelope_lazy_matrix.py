"""The full-element matrix is fetched lazily (2026-07 reviewer #5a).

At nonzero current the SC split-operator walk rebuilds transport from
half-maps / sub-slice Jacobians, so requesting the full-element matrix
up front spent a computation (and, for surrogate-wrapped elements, an
MLP query plus its scope check) that was then discarded.  These tests
pin the branch structure:

* I > 0, extended elements  -> get_element_matrix NOT called;
* I > 0, zero-length element -> called (thin branch applies the map);
* I = 0                      -> called (pure-linear path).

The physics itself is covered by the bit-identical fnalscl envelope
comparison performed when the change shipped and by the existing
envelope suites — these tests guard the code path only.
"""
import numpy as np
import pytest

import linac_gen.tracking.envelope as envmod
from linac_gen.core.lattice import Lattice
from linac_gen.core.particle import PROTON
from linac_gen.core.reference import ReferenceParticle
from linac_gen.elements.drift import Drift
from linac_gen.elements.quadrupole import Quadrupole
from linac_gen.tracking.envelope import EnvelopeSolver

INITIAL = dict(beta_x=2.0, alpha_x=0.0, emit_x=0.25,
               beta_y=2.0, alpha_y=0.0, emit_y=0.25,
               beta_z=1.0, alpha_z=0.0, emit_z=0.30)


def _fodo(with_thin=False):
    lat = Lattice()
    lat.add(Quadrupole("QF", 50.0, gradient=5.0, n_steps=5))
    lat.add(Drift("D1", 200.0))
    lat.add(Quadrupole("QD", 50.0, gradient=-5.0, n_steps=5))
    lat.add(Drift("D2", 200.0))
    if with_thin:
        lat.add(Drift("D0", 0.0))          # zero-length -> thin branch
    return lat


def _run(lat, current, spy_calls, monkeypatch):
    orig = envmod.get_element_matrix

    def spy(element, ref):
        spy_calls.append(getattr(element, "name", type(element).__name__))
        return orig(element, ref)

    monkeypatch.setattr(envmod, "get_element_matrix", spy)
    ref = ReferenceParticle(species=PROTON, w_kin=3.0, frequency=352.21)
    res = EnvelopeSolver(lat, ref, dict(INITIAL), current=current).run()
    assert np.all(np.isfinite(np.asarray(res.sigma_x)))
    return res


def test_sc_path_skips_full_matrix_for_extended_elements(monkeypatch):
    calls = []
    _run(_fodo(), 20.0, calls, monkeypatch)
    assert calls == [], (
        f"full-element matrix requested on the SC path for {calls} — "
        "the split-operator walk never uses it")


def test_sc_path_still_queries_for_thin_elements(monkeypatch):
    calls = []
    _run(_fodo(with_thin=True), 20.0, calls, monkeypatch)
    assert calls == ["D0"]


def test_zero_current_path_queries_every_element(monkeypatch):
    calls = []
    _run(_fodo(), 0.0, calls, monkeypatch)
    assert calls == ["QF", "D1", "QD", "D2"]
