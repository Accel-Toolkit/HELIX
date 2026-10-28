"""The integer part of the tunes and the phase advance along the ring when
one element advances a mode's phase by more than pi (a long bend, a matrix).

The phase is read at element exits, in (-pi, pi]; unwrapping alone takes each
element's step in [-pi, pi), so such an element came out as a step back (the
tune one integer low or negative, every mux downstream wrong).  A mode
turning forward in its own plane — always, without coupling — never steps
back: its step is taken in [0, 2 pi).  Where no element advances a phase by
more than pi the tables are those of plain unwrapping, bit for bit; where
strong coupling turns a mode backward in a plane, its steps stay as they were.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

import linac_gen.ring.optics as optics
from linac_gen.core.lattice import Lattice
from linac_gen.elements.dipole import Dipole
from linac_gen.elements.drift import Drift
from linac_gen.elements.matrix_element import MatrixElement
from linac_gen.elements.multipole import Multipole
from linac_gen.io.madx_parser import parse_madx
from linac_gen.ring.compile import compile_ring
from linac_gen.ring.optics import twiss
from linac_gen.ring.reference import RingReference

DECK = Path(__file__).resolve().parent / "fixtures" / "simple_booster.madx"
REF = RingReference.from_kinetic("proton", 800.0)


def _plain_unwrap(phase, _forward):
    """The phase from the start as it was computed before (unwrapping only)."""
    return np.unwrap(np.concatenate([[0.0], phase]))[1:]


def _rotation(mu, beta, alpha=0.0):
    c, s = math.cos(mu), math.sin(mu)
    return np.array([[c + alpha * s, beta * s], [-(1 + alpha * alpha) / beta * s, c - alpha * s]])


def _analytic_tune(pieces):
    """Tune of a 2x2 transfer line closed on itself, integer part included:
    the phase advances of its pieces (each below pi) from the periodic Twiss
    functions, tan dmu = M12 / (beta M11 - alpha M12)."""
    M = np.eye(2)
    for P in pieces:
        M = P @ M
    mu = math.acos(0.5 * np.trace(M))
    if M[0, 1] < 0:
        mu = 2 * math.pi - mu
    beta = M[0, 1] / math.sin(mu)
    alpha = (M[0, 0] - M[1, 1]) / (2 * math.sin(mu))
    total = 0.0
    for P in pieces:
        b0 = P[0, 0] * beta - P[0, 1] * alpha
        step = math.atan2(P[0, 1], b0)
        assert step >= 0.0
        total += step
        a0 = P[1, 0] * beta - P[1, 1] * alpha
        beta, alpha = (b0 * b0 + P[0, 1] ** 2) / beta, -(b0 * a0 + P[0, 1] * P[1, 1]) / beta
    return total / (2 * math.pi)


def _focusing(k2, length):
    if k2 > 0:
        k = math.sqrt(k2)
        return np.array([[math.cos(k * length), math.sin(k * length) / k],
                         [-k * math.sin(k * length), math.cos(k * length)]])
    k = math.sqrt(-k2)
    return np.array([[math.cosh(k * length), math.sinh(k * length) / k],
                     [k * math.sinh(k * length), math.cosh(k * length)]])


def test_racetrack_whose_bends_advance_the_phase_by_more_than_pi():
    # two 180-degree sector bends of 1 m radius, field index -0.1: each
    # advances the horizontal phase by sqrt(1.1) pi (0.524 turns); thin
    # doublets in the straights.  Before: Qx = -0.229666 for 1.770334
    N, f1, f2 = -0.1, 0.6, 1.0
    lat = Lattice()
    for k in range(2):
        lat.add(Dipole(f"b{k}", 180.0, 1000.0, field_index=N))
        lat.add(Drift(f"da{k}", 500.0))
        lat.add(Multipole(f"qf{k}", knl=[0.0, 1.0 / f1]))
        lat.add(Drift(f"db{k}", 1000.0))
        lat.add(Multipole(f"qd{k}", knl=[0.0, -1.0 / f2]))
        lat.add(Drift(f"dc{k}", 500.0))
    o = twiss(compile_ring(lat, REF, edge_model="linear"), chromaticity=False)

    def drift(L):
        return np.array([[1.0, L], [0.0, 1.0]])

    def lens(f):
        return np.array([[1.0, 0.0], [-1.0 / f, 1.0]])

    for plane, k2, s in (("x", 1.0 - N, 1.0), ("y", N, -1.0)):
        half = [_focusing(k2, math.pi / 8)] * 8 + [drift(0.5), lens(s * f1), drift(1.0),
                                                   lens(-s * f2), drift(0.5)]
        q = _analytic_tune(half * 2)
        assert getattr(o, f"q{plane}") == pytest.approx(q, abs=1e-6), plane
    assert math.floor(o.qx) == 1
    # the first bend's own advance, from the periodic Twiss functions at the start
    mux = np.asarray(o.table["mux"])
    i = o.table["name"].index("b0")
    b0, a0 = o.start["betx"], o.start["alfx"]
    M = _focusing(1.0 - N, math.pi)
    want = (math.atan2(M[0, 1], M[0, 0] * b0 - M[0, 1] * a0) % (2 * math.pi)) / (2 * math.pi)
    assert want > 0.5 and mux[i] == pytest.approx(want, abs=1e-6)
    assert np.all(np.diff(mux) >= 0.0)
    assert mux[-1] == pytest.approx(o.qx, abs=1e-8)


@pytest.mark.parametrize("mu_arc", [0.3, 0.55, 0.7])
def test_matrix_arc_advancing_the_phase_by_more_than_pi(mu_arc):
    # four thin FODO cells and a zero-length matrix "arc" matched to their
    # periodic Twiss functions (before: Qx 0.039 and 0.189 for 1.039 and
    # 1.189, mux at the arc's exit -0.45 and -0.30)
    def fodo():
        lat = Lattice()
        for i in range(4):
            lat.add(Multipole(name=f"qf{i}", knl=[0.0, 0.25]))
            lat.add(Drift(name=f"d1_{i}", length=3000.0))
            lat.add(Multipole(name=f"qd{i}", knl=[0.0, -0.25]))
            lat.add(Drift(name=f"d2_{i}", length=3000.0))
        return lat

    base = twiss(compile_ring(fodo(), REF), chromaticity=False)
    st = base.start
    M = np.eye(6)
    M[:2, :2] = _rotation(2 * math.pi * mu_arc, st["betx"], st["alfx"])
    M[2:4, 2:4] = _rotation(2 * math.pi * 0.2, st["bety"], st["alfy"])
    lat = fodo()
    lat.elements.insert(0, MatrixElement(name="arc", matrix=M, length=0.0))
    o = twiss(compile_ring(lat, REF), chromaticity=False)
    assert o.qx == pytest.approx(base.qx + mu_arc, abs=1e-9)
    assert o.qy == pytest.approx(base.qy + 0.2, abs=1e-9)
    assert o.table["mux"][0] == pytest.approx(mu_arc, abs=1e-9)
    assert o.table["muy"][0] == pytest.approx(0.2, abs=1e-9)
    assert np.all(np.diff(o.table["mux"]) >= 0.0)


def _same_optics(a, b):
    assert a.qx == b.qx and a.qy == b.qy
    for k, v in a.table.items():
        if k in ("name", "type"):
            assert list(v) == list(b.table[k])
        else:
            assert np.array_equal(np.asarray(v), np.asarray(b.table[k])), k


def test_tables_are_unchanged_where_no_element_advances_more_than_pi(monkeypatch):
    lat, meta = parse_madx(str(DECK))
    w = meta["reference"].w_kin
    rl = compile_ring(lat, RingReference.from_kinetic("proton", w), lattice_energy=w)
    lat2 = Lattice()
    for i in range(12):
        lat2.add(Multipole(name=f"qf{i}", knl=[0.0, 0.25]))
        lat2.add(Drift(name=f"d1_{i}", length=3000.0))
        lat2.add(Multipole(name=f"qd{i}", knl=[0.0, -0.25]))
        lat2.add(Drift(name=f"d2_{i}", length=3000.0))
    rings = [rl, compile_ring(lat2, REF)]
    new = [twiss(r) for r in rings]
    monkeypatch.setattr(optics, "_phase_from_start", _plain_unwrap)
    for r, o in zip(rings, new, strict=True):
        _same_optics(o, twiss(r))


def _coupled_ring():
    """Thin normal and skew quadrupoles strong enough that mode I turns
    backward in x at places (its x part of the Lagrange bracket is negative)."""
    lat = Lattice()
    for i, (k, L, ks) in enumerate(((1.0242, 1.8735, -0.0518), (-1.3918, 1.3483, -0.8464))):
        lat.add(Multipole(name=f"q{i}", knl=[0.0, k]))
        lat.add(Drift(name=f"d{i}", length=L * 1e3))
        lat.add(Multipole(name=f"s{i}", knl=[0.0, 0.0], ksl=[0.0, ks]))
    return compile_ring(lat, REF)


def test_strong_coupling_keeps_its_backward_steps(monkeypatch):
    rl = _coupled_ring()
    o = twiss(rl, chromaticity=False)
    steps = np.diff(np.concatenate([[0.0], o.table["mux"]]))
    assert steps.min() < -0.1                     # a step back of more than 0.1 turn
    monkeypatch.setattr(optics, "_phase_from_start", _plain_unwrap)
    _same_optics(o, twiss(rl, chromaticity=False))
    # (taking every step back as a step forward would change them)
    monkeypatch.undo()
    monkeypatch.setattr(optics, "_forward", lambda W0, Wn, p: np.ones(Wn.shape[0] + 1, dtype=bool))
    assert twiss(rl, chromaticity=False).qx != o.qx


def test_phase_from_start():
    true_steps = np.array([0.5, 0.0, -1e-15, 3.5, 0.2, 6.0, 0.0, 1e-15, 2.0])
    phase = np.angle(np.exp(1j * np.cumsum(true_steps)))
    fwd = np.ones(true_steps.size + 1, dtype=bool)
    got = optics._phase_from_start(phase, fwd)
    assert got == pytest.approx(np.cumsum(true_steps), abs=1e-12)
    # no step beyond pi: plain unwrapping, the same array bit for bit
    small = np.array([0.5, 0.0, -1e-15, 3.0, 1e-15, 3.1, 0.2])
    phase = np.angle(np.exp(1j * np.cumsum(small)))
    assert np.array_equal(optics._phase_from_start(phase, np.ones(8, dtype=bool)),
                          _plain_unwrap(phase, None))
    # a step back where the mode turns backward (not forward at either end) stays
    back = np.array([0.5, -0.4, 0.3])
    phase = np.angle(np.exp(1j * np.cumsum(back)))
    assert optics._phase_from_start(phase, np.array([True, True, False, True])) == pytest.approx(
        np.cumsum(back), abs=1e-12)
