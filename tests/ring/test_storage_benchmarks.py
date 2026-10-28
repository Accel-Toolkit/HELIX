"""Storage-ring benchmarks of the turn loop (L2, fixed energy): a FODO ring
from reference code B's examples, against its published numbers and the
analytic cell."""
from __future__ import annotations

import math

import numpy as np
import pytest

from linac_gen.core.config import BeamConfig
from linac_gen.core.lattice import Lattice
from linac_gen.elements.drift import Drift
from linac_gen.elements.quadrupole import Quadrupole
from linac_gen.ring import RingConfig, RingSimulation
from linac_gen.ring.compile import compile_ring
from linac_gen.ring.optics import twiss
from linac_gen.ring.reference import RingReference

#: the reference's FODO example: drift 0.25 m, quadrupole 1 m k = +1 /m^2,
#: drift 0.5 m, quadrupole 1 m k = -1, drift 0.25 m, tracked 100 periods; its
#: analysis asserts these to rtol 1e-3
REF_FODO_TUNE = 0.1883
REF_ALPHA_X, REF_BETA_X = -1.59050035, 2.82161941


def _fodo_ring(w_mev=2000.0):
    ref = RingReference.from_kinetic("proton", w_mev)
    brho = ref.p0c / 299.792458                       # T m (p0c in MeV)
    lat = Lattice()
    lat.add(Drift(name="drift1", length=250.0))
    lat.add(Quadrupole(name="quad1", length=1000.0, gradient=1.0 * brho))
    lat.add(Drift(name="drift2", length=500.0))
    lat.add(Quadrupole(name="quad2", length=1000.0, gradient=-1.0 * brho))
    lat.add(Drift(name="drift3", length=250.0))
    return lat, compile_ring(lat, ref, lattice_energy=w_mev)


def _cell(k):
    """The thick-lens cell matrix (2x2) of the plane with focusing sign k."""
    def quad(kk, L):
        if kk > 0:
            s = math.sqrt(kk)
            return np.array([[math.cos(s * L), math.sin(s * L) / s],
                             [-s * math.sin(s * L), math.cos(s * L)]])
        s = math.sqrt(-kk)
        return np.array([[math.cosh(s * L), math.sinh(s * L) / s],
                         [s * math.sinh(s * L), math.cosh(s * L)]])

    def drift(L):
        return np.array([[1.0, L], [0.0, 1.0]])
    return drift(0.25) @ quad(-k, 1.0) @ drift(0.5) @ quad(k, 1.0) @ drift(0.25)


def test_fodo_ring_tune():
    lat, rl = _fodo_ring()
    o = twiss(rl, chromaticity=False)
    # the analytic thick-lens cell
    for k, plane in ((1.0, "x"), (-1.0, "y")):
        M = _cell(k)
        mu = math.acos(0.5 * np.trace(M))
        if M[0, 1] < 0:
            mu = 2 * math.pi - mu
        assert getattr(o, f"q{plane}") == pytest.approx(mu / (2 * math.pi), abs=1e-12)
        assert o.start[f"bet{plane}"] == pytest.approx(M[0, 1] / math.sin(mu), rel=1e-12)
    # the reference's numbers (its own tolerance), from the optics and from tracking
    assert o.start["alfx"] == pytest.approx(REF_ALPHA_X, rel=1e-3)
    assert o.start["betx"] == pytest.approx(REF_BETA_X, rel=1e-3)
    beam = BeamConfig(species="proton", energy=2000.0, frequency=100.0, current=0.0,
                      emit_nx=1.0, emit_ny=1.0, emit_z=1.0, beta_z=100.0)
    res = RingSimulation(lat, beam, RingConfig(n_turns=100, n_particles=8, source="matched",
                                               tune_probes=1, stats_every=100),
                         ring_lattice=rl).run()
    assert res.tunes["qx"] == pytest.approx(REF_FODO_TUNE, rel=1e-3)
    assert res.tunes["qy"] == pytest.approx(REF_FODO_TUNE, rel=1e-3)
    # (and far closer to the cell's own tune: NAFF over 100 periods)
    assert res.tunes["qx"] == pytest.approx(o.qx, abs=1e-6)
    assert res.tunes["qy"] == pytest.approx(o.qy, abs=1e-6)
