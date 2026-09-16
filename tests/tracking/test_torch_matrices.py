"""Differentiable torch transfer matrices vs the numpy element matrices.

Every torch builder must reproduce the numpy ``transfer_matrix()`` to
~1e-10 across the full branch grid (quad cos/cosh, dipole sector vs
combined-function, hv swap, H-).
"""
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from linac_gen.core.particle import PROTON, H_MINUS
from linac_gen.core.reference import ReferenceParticle
from linac_gen.elements.drift import Drift
from linac_gen.elements.quadrupole import Quadrupole
from linac_gen.elements.solenoid import Solenoid
from linac_gen.elements.dipole import Dipole
from linac_gen.elements.edge import Edge
from linac_gen.tracking.torch_matrices import (
    RefKinematics, drift_matrix, quad_matrix, solenoid_matrix,
    dipole_matrix, edge_matrix,
)

_ATOL = 1e-10


def _proton_ref():
    return ReferenceParticle(species=PROTON, w_kin=3.0, frequency=352.21)


def _hminus_ref():
    return ReferenceParticle(species=H_MINUS, w_kin=10.0, frequency=352.21)


def _close(M_torch, M_numpy):
    assert M_torch.dtype == torch.float64
    # absolute for O(1) entries; the relative term only matters for the
    # strongly hyperbolic combined-function cases (entries ~1e5, where a
    # 1-ulp cosh difference between torch and libm is 1e-10 absolute)
    np.testing.assert_allclose(M_torch.detach().numpy(), np.asarray(M_numpy),
                               rtol=1e-13, atol=_ATOL)


# ── Drift ───────────────────────────────────────────────────────────────────
@pytest.mark.parametrize("L", [0.0, 50.0, 200.0])
def test_drift_matrix_matches_numpy(L):
    ref = _proton_ref()
    kin = RefKinematics.from_reference(ref)
    _close(drift_matrix(L, kin), Drift("D", length=L).transfer_matrix(ref))


# ── Quadrupole ──────────────────────────────────────────────────────────────
@pytest.mark.parametrize("g", [-12.0, -5.0, 5.0, 12.0])
def test_quad_matrix_matches_numpy(g):
    ref = _proton_ref()
    kin = RefKinematics.from_reference(ref)
    q = Quadrupole("Q", length=100.0, gradient=g)
    _close(quad_matrix(100.0, q.effective_gradient, q.skew_angle, kin),
           q.transfer_matrix(ref))


def test_quad_zero_gradient_is_drift_like():
    ref = _proton_ref()
    kin = RefKinematics.from_reference(ref)
    q = Quadrupole("Q", length=100.0, gradient=0.0)
    _close(quad_matrix(100.0, q.effective_gradient, q.skew_angle, kin),
           q.transfer_matrix(ref))


@pytest.mark.parametrize("skew", [0.0, 30.0, 45.0])
def test_quad_skew_matches_numpy(skew):
    ref = _proton_ref()
    kin = RefKinematics.from_reference(ref)
    q = Quadrupole("Q", length=100.0, gradient=6.0, skew_angle=skew)
    _close(quad_matrix(100.0, q.effective_gradient, q.skew_angle, kin),
           q.transfer_matrix(ref))


def test_quad_hminus_charge_sign():
    ref = _hminus_ref()
    kin = RefKinematics.from_reference(ref)
    q = Quadrupole("Q", length=100.0, gradient=5.0)
    _close(quad_matrix(100.0, q.effective_gradient, q.skew_angle, kin),
           q.transfer_matrix(ref))


# ── Solenoid ────────────────────────────────────────────────────────────────
@pytest.mark.parametrize("B", [0.0, 0.3, 0.7])
def test_solenoid_matrix_matches_numpy(B):
    ref = _proton_ref()
    kin = RefKinematics.from_reference(ref)
    s = Solenoid("S", length=80.0, field=B)
    _close(solenoid_matrix(80.0, s.effective_field, kin),
           s.transfer_matrix(ref))


# ── Dipole ──────────────────────────────────────────────────────────────────
@pytest.mark.parametrize("angle", [-45.0, -10.0, 10.0, 30.0, 90.0])
@pytest.mark.parametrize("N", [0.0, 0.5, 1.0, 1.5, -400.0])
@pytest.mark.parametrize("field_rel", [0.0, 0.02])
def test_dipole_matrix_matches_numpy(angle, N, field_rel):
    ref = _proton_ref()
    kin = RefKinematics.from_reference(ref)
    dip = Dipole("B", angle=angle, rho=1000.0, field_index=N, field_rel=field_rel)
    _close(dipole_matrix(dip.effective_angle, dip.rho, dip.length, dip.e1,
                         dip.e2, dip.field_index, dip.hv, kin,
                         design_angle_deg=dip.angle),
           dip.transfer_matrix(ref))


def test_dipole_angle_knob_keeps_the_magnet_length():
    """d(M23)/d(angle) must vanish: the tunable angle is the field, not the
    iron — the radius scales inversely and the vertical drift stays L."""
    ref = _proton_ref()
    kin = RefKinematics.from_reference(ref)
    dip = Dipole("B", angle=-12.0, rho=1500.0)          # n = 0: the other plane is a pure drift of L
    a = torch.tensor(float(dip.angle), dtype=torch.float64, requires_grad=True)
    M = dipole_matrix(a, dip.rho, dip.length, dip.e1, dip.e2, dip.field_index, dip.hv, kin,
                      design_angle_deg=dip.angle)
    (g23,) = torch.autograd.grad(M[2, 3], a, retain_graph=True)
    (g01,) = torch.autograd.grad(M[0, 1], a)
    assert abs(float(g23)) < 1e-12
    assert abs(float(g01)) > 1e-6            # the bend-plane block does depend on the knob


@pytest.mark.parametrize("e1,e2", [(0.0, 0.0), (15.0, 15.0), (10.0, 20.0)])
def test_dipole_with_edges_matches_numpy(e1, e2):
    ref = _proton_ref()
    kin = RefKinematics.from_reference(ref)
    dip = Dipole("B", angle=25.0, rho=1200.0, e1=e1, e2=e2)
    _close(dipole_matrix(dip.effective_angle, dip.rho, dip.length, dip.e1,
                         dip.e2, dip.field_index, dip.hv, kin),
           dip.transfer_matrix(ref))


@pytest.mark.parametrize("angle", [-20.0, 20.0])
@pytest.mark.parametrize("N", [0.0, 0.5, 1.5])
def test_dipole_vertical_matches_numpy(angle, N):
    ref = _proton_ref()
    kin = RefKinematics.from_reference(ref)
    dip = Dipole("B", angle=angle, rho=1000.0, hv=1, field_index=N)
    _close(dipole_matrix(dip.effective_angle, dip.rho, dip.length, dip.e1,
                         dip.e2, dip.field_index, dip.hv, kin),
           dip.transfer_matrix(ref))


def test_dipole_combined_function_cosh_branch():
    """field_index > 1 drives kx2 < 0 -> the hyperbolic horizontal branch."""
    ref = _proton_ref()
    kin = RefKinematics.from_reference(ref)
    dip = Dipole("B", angle=20.0, rho=1000.0, field_index=1.5)
    _close(dipole_matrix(dip.effective_angle, dip.rho, dip.length, dip.e1,
                         dip.e2, dip.field_index, dip.hv, kin),
           dip.transfer_matrix(ref))


# ── Edge ────────────────────────────────────────────────────────────────────
@pytest.mark.parametrize("pole,gap,hv", [
    (0.0, 0.0, 0), (15.0, 0.0, 0), (20.0, 50.0, 0), (20.0, 50.0, 1),
])
def test_edge_matrix_matches_numpy(pole, gap, hv):
    ref = _proton_ref()
    e = Edge("E", pole_rotation=pole, rho=1000.0, gap=gap, hv=hv)
    _close(edge_matrix(e.pole_rotation, e.rho, e.gap, e.k1, e.hv),
           e.transfer_matrix(ref))


# ── ds-slicing ──────────────────────────────────────────────────────────────
def test_quad_slice_composability():
    ref = _proton_ref()
    kin = RefKinematics.from_reference(ref)
    q = Quadrupole("Q", length=200.0, gradient=7.0)
    half = quad_matrix(100.0, q.effective_gradient, q.skew_angle, kin)
    full = quad_matrix(200.0, q.effective_gradient, q.skew_angle, kin)
    np.testing.assert_allclose((half @ half).detach().numpy(),
                               full.detach().numpy(), rtol=0, atol=_ATOL)
    _close(half, q.transfer_matrix(ref, ds=100.0))


def test_dipole_slice_matches_numpy():
    ref = _proton_ref()
    kin = RefKinematics.from_reference(ref)
    dip = Dipole("B", angle=30.0, rho=1000.0)
    half = dip.length / 2.0
    M_t = dipole_matrix(dip.effective_angle, dip.rho, dip.length, dip.e1,
                        dip.e2, dip.field_index, dip.hv, kin, ds_mm=half)
    _close(M_t, dip.transfer_matrix(ref, ds=half))


# ── dtype ───────────────────────────────────────────────────────────────────
def test_all_matrices_are_float64():
    ref = _proton_ref()
    kin = RefKinematics.from_reference(ref)
    dip = Dipole("B", angle=20.0, rho=1000.0)
    mats = [
        drift_matrix(100.0, kin),
        quad_matrix(100.0, 5.0, 0.0, kin),
        solenoid_matrix(80.0, 0.4, kin),
        dipole_matrix(dip.effective_angle, dip.rho, dip.length, dip.e1,
                      dip.e2, dip.field_index, dip.hv, kin),
        edge_matrix(15.0, 1000.0, 0.0, 0.45, 0),
    ]
    for M in mats:
        assert M.dtype == torch.float64


@pytest.mark.parametrize("field_index", [
    0.0,          # pure sector bend (the n = 0 early return)
    0.3,          # elliptic, kx2 > 0
    1.0,          # parabolic, kx2 == 0 exactly -- the J series
    1.0 + 1e-9,   # parabolic guard from above: |u| < 1e-3, series again
    1.7,          # hyperbolic, kx2 < 0
    -2.0,         # elliptic with strong defocusing in the other plane
])
@pytest.mark.parametrize("hv", [0, 1])
def test_dipole_path_length_row_autograd_matches_finite_difference(field_index, hv):
    """The compaction and the path-length coupling must be differentiable in
    the bend-angle knob: gradient-based matching optimises through them.

    Every branch of the body matrix is covered, including both sides of the
    ``|kx2 L^2| < 1e-3`` series threshold, where a closed-form J would make
    the gradient garbage even though the value still looks plausible.
    """
    ref = _proton_ref()
    kin = RefKinematics.from_reference(ref)
    dip = Dipole("B", angle=18.0, rho=1200.0, field_index=field_index, hv=hv)

    def M_of(angle_value):
        a = torch.tensor(float(angle_value), dtype=torch.float64, requires_grad=True)
        return a, dipole_matrix(a, dip.rho, dip.length, dip.e1, dip.e2, dip.field_index,
                                dip.hv, kin, design_angle_deg=dip.angle)

    row = 2 if hv == 1 else 0
    for entry in ((4, 5), (4, row), (4, row + 1)):
        a, M = M_of(dip.angle)
        (g,) = torch.autograd.grad(M[entry], a)
        h = 1e-6 * dip.angle
        _, Mp = M_of(dip.angle + h)
        _, Mm = M_of(dip.angle - h)
        fd = float((Mp[entry] - Mm[entry]).detach()) / (2.0 * h)
        assert abs(float(g)) > 0.0
        assert float(g) == pytest.approx(fd, rel=1e-6), f"entry {entry}"
