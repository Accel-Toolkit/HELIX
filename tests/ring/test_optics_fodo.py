"""Ring optics of a thin-lens FODO ring against the analytic answer (L1:
1e-12): the periodic Twiss functions and tunes, with the integer part, from
the 2x2 cell matrix; the linear chromaticity from the thin-lens formula."""
from __future__ import annotations

import math

import numpy as np
import pytest

from linac_gen.core.lattice import Lattice
from linac_gen.elements.drift import Drift
from linac_gen.elements.multipole import Multipole
from linac_gen.ring.compile import compile_ring
from linac_gen.ring.optics import twiss
from linac_gen.ring.reference import RingReference

N_CELLS, F, L = 12, 4.0, 3.0           # cells, focal length [m], drift [m]


def _ring(drift_model):
    lat = Lattice()
    for i in range(N_CELLS):
        lat.add(Multipole(name=f"qf{i}", knl=[0.0, 1.0 / F]))
        lat.add(Drift(name=f"d1_{i}", length=L * 1e3))
        lat.add(Multipole(name=f"qd{i}", knl=[0.0, -1.0 / F]))
        lat.add(Drift(name=f"d2_{i}", length=L * 1e3))
    ref = RingReference.from_kinetic("proton", 800.0)
    return twiss(compile_ring(lat, ref, drift_model=drift_model))


def _cell(k):
    """2x2 cell matrix from the start (before QF) for focusing sign k (+1 x, -1 y)."""
    q = lambda f: np.array([[1.0, 0.0], [-1.0 / f, 1.0]])      # noqa: E731
    d = np.array([[1.0, L], [0.0, 1.0]])
    return d @ q(-k * F) @ d @ q(k * F)


@pytest.mark.parametrize("drift_model", ["exact", "expanded"])
def test_thin_fodo_ring_matches_the_analytic_optics(drift_model):
    o = _ring(drift_model)
    for k, plane in ((1, "x"), (-1, "y")):
        M = _cell(k)
        mu = math.acos(0.5 * np.trace(M))
        if M[0, 1] < 0:
            mu = 2 * math.pi - mu
        beta = M[0, 1] / math.sin(mu)
        alpha = (M[0, 0] - M[1, 1]) / (2 * math.sin(mu))
        q = N_CELLS * mu / (2 * math.pi)
        assert getattr(o, f"q{plane}") == pytest.approx(q, abs=1e-12)
        assert o.start[f"bet{plane}"] == pytest.approx(beta, rel=1e-12)
        assert o.start[f"alf{plane}"] == pytest.approx(alpha, abs=1e-12)
    assert math.floor(o.qx) == math.floor(N_CELLS * math.acos(0.5 * np.trace(_cell(1))) / (2 * math.pi))
    # no bends: no dispersion, no closed-orbit offset
    assert np.abs(o.table["dx"]).max() < 1e-12 and np.abs(o.table["x"]).max() < 1e-15


def test_thin_fodo_chromaticity_is_the_thin_lens_sum():
    # dQ/d(delta) = -(1/4 pi) sum beta_i k_i  (k_i = +-1/f the thin-lens
    # strength, focusing in the plane positive)
    o = _ring("exact")
    names = o.table["name"]
    bx = np.asarray(o.table["betx"])
    by = np.asarray(o.table["bety"])
    kq = np.array([1.0 / F if n.startswith("qf") else (-1.0 / F if n.startswith("qd") else 0.0)
                   for n in names])
    # (a thin lens sits at its own exit position: beta there is beta at the lens)
    want_x = -np.sum(bx * kq) / (4 * math.pi)
    want_y = np.sum(by * kq) / (4 * math.pi)
    assert o.dqx == pytest.approx(want_x, rel=1e-6)
    assert o.dqy == pytest.approx(want_y, rel=1e-6)
