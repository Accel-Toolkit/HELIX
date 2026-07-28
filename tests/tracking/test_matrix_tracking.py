# tests/tracking/test_matrix_tracking.py
import numpy as np
import pytest
from linac_gen.core.particle import PROTON
from linac_gen.core.reference import ReferenceParticle
from linac_gen.core.lattice import Lattice
from linac_gen.elements.drift import Drift
from linac_gen.elements.quadrupole import Quadrupole
from linac_gen.tracking.matrix_tracking import compute_transfer_matrix, compute_twiss

def test_single_drift_matrix():
    lat = Lattice()
    lat.add(Drift("D1", 100.0))
    ref = ReferenceParticle(species=PROTON, w_kin=3.0, frequency=352.21)
    M = compute_transfer_matrix(lat, ref)
    assert M.shape == (6, 6)
    assert abs(M[0, 1] - 0.1) < 1e-10

def test_fodo_matrix():
    lat = Lattice()
    lat.add(Quadrupole("QF", 50.0, gradient=5.0))
    lat.add(Drift("D1", 200.0))
    lat.add(Quadrupole("QD", 50.0, gradient=-5.0))
    lat.add(Drift("D2", 200.0))
    ref = ReferenceParticle(species=PROTON, w_kin=3.0, frequency=352.21)
    M = compute_transfer_matrix(lat, ref)
    trace_x = M[0, 0] + M[1, 1]
    assert abs(trace_x) < 2.0, f"Unstable FODO: trace_x = {trace_x}"

def test_twiss_from_periodic():
    lat = Lattice()
    lat.add(Quadrupole("QF", 50.0, gradient=5.0))
    lat.add(Drift("D1", 200.0))
    lat.add(Quadrupole("QD", 50.0, gradient=-5.0))
    lat.add(Drift("D2", 200.0))
    ref = ReferenceParticle(species=PROTON, w_kin=3.0, frequency=352.21)
    M = compute_transfer_matrix(lat, ref)
    twiss_x = compute_twiss(M, plane="x")
    assert twiss_x["beta"] > 0
    assert 0 < twiss_x["mu"] < 180
    bg_check = twiss_x["beta"] * twiss_x["gamma_t"] - twiss_x["alpha"]**2
    assert abs(bg_check - 1.0) < 1e-10
