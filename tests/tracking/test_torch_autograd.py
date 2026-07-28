"""Autograd gradients of the differentiable tracker vs finite differences.

The autograd gradient (torch backward) must agree with a central
finite-difference taken through the numpy path.  Tolerance is
finite-difference-limited (~1e-5), not 1e-10.
"""
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
from linac_gen.tracking.matrix_tracking import (
    compute_transfer_matrix, compute_twiss,
)
from linac_gen.tracking.autograd_api import DifferentiableLattice

_FD_TOL = 1e-5


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


def _fd_grad(lattice, ref, element, attr, base, output_fn):
    """Central finite-difference d(output)/d(element.attr) via the numpy path."""
    h = max(1e-6, 1e-6 * abs(base))
    orig = getattr(element, attr)
    setattr(element, attr, base + h)
    fp = output_fn(compute_transfer_matrix(lattice, ref))
    setattr(element, attr, base - h)
    fm = output_fn(compute_transfer_matrix(lattice, ref))
    setattr(element, attr, orig)
    return (fp - fm) / (2.0 * h)


def _assert_close(g_auto, g_fd):
    assert abs(g_auto - g_fd) <= _FD_TOL * max(1.0, abs(g_fd)), \
        f"autograd {g_auto:.6e} vs finite-diff {g_fd:.6e}"


def test_autograd_quad_gradient_vs_fd_matrix_entry():
    ref, lat = _ref(), _fodo()
    qf = next(e for e in lat.elements if e.name == "QF")
    dl = DifferentiableLattice(lat, ref)
    p = dl.set_tunables([("QF", "gradient")])[0]
    dl.transfer_matrix()[0, 0].backward()
    g_fd = _fd_grad(lat, ref, qf, "gradient", 8.0, lambda M: M[0, 0])
    _assert_close(float(p.tensor.grad), g_fd)


def test_autograd_quad_gradient_vs_fd_twiss_beta():
    ref, lat = _ref(), _fodo()
    qf = next(e for e in lat.elements if e.name == "QF")
    dl = DifferentiableLattice(lat, ref)
    p = dl.set_tunables([("QF", "gradient")])[0]
    dl.twiss("x")["beta"].backward()
    g_fd = _fd_grad(lat, ref, qf, "gradient", 8.0,
                    lambda M: compute_twiss(M, "x")["beta"])
    _assert_close(float(p.tensor.grad), g_fd)


def test_autograd_solenoid_field_vs_fd():
    ref = _ref()
    lat = Lattice()
    lat.add(Drift("D1", length=100.0))
    lat.add(Solenoid("S", length=120.0, field=0.5))
    lat.add(Drift("D2", length=100.0))
    sol = next(e for e in lat.elements if e.name == "S")
    dl = DifferentiableLattice(lat, ref)
    p = dl.set_tunables([("S", "field")])[0]
    dl.transfer_matrix()[0, 1].backward()
    g_fd = _fd_grad(lat, ref, sol, "field", 0.5, lambda M: M[0, 1])
    _assert_close(float(p.tensor.grad), g_fd)


def test_autograd_dipole_angle_vs_fd():
    ref = _ref()
    lat = Lattice()
    lat.add(Drift("D1", length=150.0))
    lat.add(Dipole("B", angle=20.0, rho=1000.0))
    lat.add(Drift("D2", length=150.0))
    dip = next(e for e in lat.elements if e.name == "B")
    dl = DifferentiableLattice(lat, ref)
    p = dl.set_tunables([("B", "angle")])[0]
    dl.transfer_matrix()[0, 5].backward()
    g_fd = _fd_grad(lat, ref, dip, "angle", 20.0, lambda M: M[0, 5])
    _assert_close(float(p.tensor.grad), g_fd)


def test_autograd_tracked_coordinate_vs_fd():
    ref, lat = _ref(), _fodo()
    qf = next(e for e in lat.elements if e.name == "QF")
    X0 = np.array([[1.0, 0.2, -0.5, 0.1, 2.0, 0.01]])
    dl = DifferentiableLattice(lat, ref)
    p = dl.set_tunables([("QF", "gradient")])[0]
    dl.track(X0)[0, 0].backward()
    g_fd = _fd_grad(lat, ref, qf, "gradient", 8.0,
                    lambda M: (M @ X0.T).T[0, 0])
    _assert_close(float(p.tensor.grad), g_fd)


def test_autograd_sigma_entry_vs_fd():
    ref, lat = _ref(), _fodo()
    qf = next(e for e in lat.elements if e.name == "QF")
    S0 = np.eye(6) * 2.0
    dl = DifferentiableLattice(lat, ref)
    p = dl.set_tunables([("QF", "gradient")])[0]
    dl.sigma(S0)[0, 0].backward()
    g_fd = _fd_grad(lat, ref, qf, "gradient", 8.0,
                    lambda M: (M @ S0 @ M.T)[0, 0])
    _assert_close(float(p.tensor.grad), g_fd)


def test_multi_tunable_backward():
    ref, lat = _ref(), _fodo()
    dl = DifferentiableLattice(lat, ref)
    params = dl.set_tunables([("QF", "gradient"), ("QD", "gradient")])
    dl.twiss("x")["beta"].backward()
    assert all(p.tensor.grad is not None for p in params)
    assert all(np.isfinite(float(p.tensor.grad)) for p in params)
