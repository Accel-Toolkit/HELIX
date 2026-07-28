# tests/elements/test_drift.py
import numpy as np
import pytest
from linac_gen.elements.drift import Drift
from linac_gen.core.reference import ReferenceParticle
from linac_gen.core.particle import PROTON
from linac_gen.core.constants import C_LIGHT, PI

def test_drift_transfer_matrix_shape():
    d = Drift(name="D1", length=100.0, aperture=20.0, n_steps=1)
    ref = ReferenceParticle(species=PROTON, w_kin=3.0, frequency=352.21)
    M = d.transfer_matrix(ref)
    assert M.shape == (6, 6)

def test_drift_identity_like():
    d = Drift(name="D0", length=0.0, aperture=0.0, n_steps=1)
    ref = ReferenceParticle(species=PROTON, w_kin=3.0, frequency=352.21)
    M = d.transfer_matrix(ref)
    np.testing.assert_array_almost_equal(M, np.eye(6))

def test_drift_x_transport():
    d = Drift(name="D1", length=100.0, aperture=20.0, n_steps=1)
    ref = ReferenceParticle(species=PROTON, w_kin=3.0, frequency=352.21)
    M = d.transfer_matrix(ref)
    # x(mm) += x'(mrad) * L(mm) * 1e-3 => M[0,1] = 100*1e-3 = 0.1
    assert abs(M[0, 1] - 0.1) < 1e-10

def test_drift_longitudinal_coupling():
    d = Drift(name="D1", length=100.0, aperture=20.0, n_steps=1)
    ref = ReferenceParticle(species=PROTON, w_kin=3.0, frequency=352.21)
    M = d.transfer_matrix(ref)
    assert M[4, 5] < 0  # negative phase slip coupling

def test_drift_slice():
    d = Drift(name="D1", length=100.0, aperture=20.0, n_steps=1)
    ref = ReferenceParticle(species=PROTON, w_kin=3.0, frequency=352.21)
    M_full = d.transfer_matrix(ref)
    M_half = d.transfer_matrix(ref, ds=50.0)
    assert abs(M_half[0, 1] - M_full[0, 1] / 2) < 1e-10
