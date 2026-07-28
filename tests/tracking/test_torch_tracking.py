"""Differentiable torch lattice composition vs the numpy matrix tracker."""
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from linac_gen.core.particle import PROTON
from linac_gen.core.reference import ReferenceParticle
from linac_gen.core.lattice import Lattice
from linac_gen.elements.drift import Drift
from linac_gen.elements.quadrupole import Quadrupole
from linac_gen.elements.solenoid import Solenoid
from linac_gen.elements.dipole import Dipole
from linac_gen.elements.edge import Edge
from linac_gen.elements.marker import Marker
from linac_gen.tracking.matrix_tracking import (
    compute_transfer_matrix, compute_twiss,
)
from linac_gen.tracking.torch_tracking import (
    compute_transfer_matrix_torch, compute_twiss_torch, track_beam_torch,
)

_ATOL = 1e-10


def _ref():
    return ReferenceParticle(species=PROTON, w_kin=3.0, frequency=352.21)


def _fodo():
    lat = Lattice()
    lat.add(Drift("D1", length=200.0))
    lat.add(Quadrupole("QF", length=100.0, gradient=8.0))
    lat.add(Drift("D2", length=400.0))
    lat.add(Quadrupole("QD", length=100.0, gradient=-8.0))
    lat.add(Drift("D3", length=200.0))
    return lat


def _dipole_cell():
    lat = Lattice()
    lat.add(Drift("D1", length=150.0))
    lat.add(Edge("E1", pole_rotation=10.0, rho=1000.0))
    lat.add(Dipole("B", angle=20.0, rho=1000.0, field_index=0.4))
    lat.add(Edge("E2", pole_rotation=10.0, rho=1000.0))
    lat.add(Drift("D2", length=150.0))
    return lat


def test_fodo_transfer_matrix_matches_numpy():
    ref, lat = _ref(), _fodo()
    M_np = compute_transfer_matrix(lat, ref)
    M_t = compute_transfer_matrix_torch(lat, ref)
    assert M_t.dtype == torch.float64
    np.testing.assert_allclose(M_t.detach().numpy(), M_np, rtol=0, atol=_ATOL)


def test_dipole_cell_transfer_matrix_matches_numpy():
    ref, lat = _ref(), _dipole_cell()
    M_np = compute_transfer_matrix(lat, ref)
    M_t = compute_transfer_matrix_torch(lat, ref)
    np.testing.assert_allclose(M_t.detach().numpy(), M_np, rtol=0, atol=_ATOL)


@pytest.mark.parametrize("plane", ["x", "y"])
def test_twiss_matches_numpy(plane):
    ref, lat = _ref(), _fodo()
    tw_np = compute_twiss(compute_transfer_matrix(lat, ref), plane)
    tw_t = compute_twiss_torch(compute_transfer_matrix_torch(lat, ref), plane)
    for k in tw_np:
        assert abs(tw_np[k] - float(tw_t[k])) < 1e-9


def test_track_beam_matches_numpy():
    ref, lat = _ref(), _fodo()
    M_np = compute_transfer_matrix(lat, ref)
    rng = np.random.default_rng(0)
    X = rng.normal(0.0, 1.0, size=(64, 6))
    out_np = (M_np @ X.T).T
    out_t = track_beam_torch(lat, ref, X)
    np.testing.assert_allclose(out_t.detach().numpy(), out_np,
                               rtol=0, atol=1e-9)


def test_coupled_lattice_raises_in_both_paths():
    ref = _ref()
    lat = Lattice()
    lat.add(Drift("D", length=100.0))
    lat.add(Solenoid("S", length=120.0, field=0.5))
    lat.add(Drift("D", length=100.0))
    M_np = compute_transfer_matrix(lat, ref)
    M_t = compute_transfer_matrix_torch(lat, ref)
    with pytest.raises(ValueError, match="coupled"):
        compute_twiss(M_np, "x")
    with pytest.raises(ValueError, match="coupled"):
        compute_twiss_torch(M_t, "x")


def test_nonlinear_element_identity_and_warning():
    """A non-linear element -> identity + a one-shot warning; the lattice
    still composes and matches numpy (Marker is identity in both paths)."""
    import linac_gen.tracking.torch_tracking as tt
    tt._warned_nonlinear.discard("Marker")
    ref = _ref()
    lat = Lattice()
    lat.add(Drift("D1", length=100.0))
    lat.add(Marker(name="MK"))
    lat.add(Drift("D2", length=100.0))
    with pytest.warns(UserWarning, match="not a differentiable linear element"):
        M_t = compute_transfer_matrix_torch(lat, ref)
    M_np = compute_transfer_matrix(lat, ref)
    np.testing.assert_allclose(M_t.detach().numpy(), M_np, rtol=0, atol=_ATOL)


def test_nonlinear_element_error_mode():
    ref = _ref()
    lat = Lattice()
    lat.add(Drift("D", length=100.0))
    lat.add(Marker(name="MK"))
    with pytest.raises(TypeError, match="not a differentiable linear element"):
        compute_transfer_matrix_torch(lat, ref, on_nonlinear="error")
