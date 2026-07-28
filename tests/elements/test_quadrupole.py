# tests/elements/test_quadrupole.py
import numpy as np
import math
import pytest
from linac_gen.elements.quadrupole import Quadrupole
from linac_gen.core.reference import ReferenceParticle
from linac_gen.core.particle import PROTON, H_MINUS

def test_quad_matrix_shape():
    q = Quadrupole(name="Q1", length=100.0, gradient=5.0, aperture=20.0, n_steps=5)
    ref = ReferenceParticle(species=PROTON, w_kin=3.0, frequency=352.21)
    M = q.transfer_matrix(ref)
    assert M.shape == (6, 6)

def test_quad_symplecticity():
    q = Quadrupole(name="Q1", length=100.0, gradient=5.0, aperture=20.0, n_steps=5)
    ref = ReferenceParticle(species=PROTON, w_kin=3.0, frequency=352.21)
    M = q.transfer_matrix(ref)
    det_x = M[0, 0] * M[1, 1] - M[0, 1] * M[1, 0]
    det_y = M[2, 2] * M[3, 3] - M[2, 3] * M[3, 2]
    assert abs(det_x - 1.0) < 1e-10
    assert abs(det_y - 1.0) < 1e-10

def test_quad_focusing_defocusing():
    q = Quadrupole(name="Q1", length=100.0, gradient=5.0, aperture=20.0, n_steps=5)
    ref = ReferenceParticle(species=PROTON, w_kin=3.0, frequency=352.21)
    M = q.transfer_matrix(ref)
    assert M[0, 0] < 1.0  # focusing in x: cos(kL) < 1
    assert M[2, 2] > 1.0  # defocusing in y: cosh(kL) > 1

def test_quad_negative_gradient():
    q = Quadrupole(name="Q1", length=100.0, gradient=-5.0, aperture=20.0, n_steps=5)
    ref = ReferenceParticle(species=PROTON, w_kin=3.0, frequency=352.21)
    M = q.transfer_matrix(ref)
    assert M[0, 0] > 1.0  # defocusing in x
    assert M[2, 2] < 1.0  # focusing in y

def test_quad_thin_lens_limit():
    L_mm = 1.0
    G = 5.0
    q = Quadrupole(name="Q1", length=L_mm, gradient=G, aperture=20.0, n_steps=1)
    ref = ReferenceParticle(species=PROTON, w_kin=3.0, frequency=352.21)
    M = q.transfer_matrix(ref)
    k2 = G / ref.brho
    L_m = L_mm * 1e-3
    expected_kick = -k2 * L_m
    assert abs(M[1, 0] - expected_kick) < abs(expected_kick) * 0.01

def test_quad_slice():
    q = Quadrupole(name="Q1", length=100.0, gradient=5.0, aperture=20.0, n_steps=5)
    ref = ReferenceParticle(species=PROTON, w_kin=3.0, frequency=352.21)
    M_full = q.transfer_matrix(ref)
    M_half = q.transfer_matrix(ref, ds=50.0)
    assert not np.allclose(M_full, M_half)
    M_product = M_half @ M_half
    np.testing.assert_array_almost_equal(M_product, M_full, decimal=10)

def test_quad_h_minus_reverses_focusing():
    """H- (charge=-1) should have reversed focusing compared to proton."""
    q = Quadrupole(name="Q1", length=100.0, gradient=5.0, aperture=20.0, n_steps=5)
    ref_proton = ReferenceParticle(species=PROTON, w_kin=3.0, frequency=352.21)
    ref_hminus = ReferenceParticle(species=H_MINUS, w_kin=3.0, frequency=352.21)
    M_p = q.transfer_matrix(ref_proton)
    M_h = q.transfer_matrix(ref_hminus)
    # Proton: G>0 focuses x (cos), defocuses y (cosh)
    assert M_p[0, 0] < 1.0  # cos(kL) < 1
    assert M_p[2, 2] > 1.0  # cosh(kL) > 1
    # H-minus: G>0 should DEFOCUS x (cosh), FOCUS y (cos) — reversed
    assert M_h[0, 0] > 1.0  # cosh(kL) > 1
    assert M_h[2, 2] < 1.0  # cos(kL) < 1

def test_quad_zero_gradient_acts_as_drift():
    """Zero gradient should behave like a drift, not crash."""
    from linac_gen.elements.drift import Drift
    q = Quadrupole(name="Q0", length=100.0, gradient=0.0, aperture=20.0, n_steps=1)
    d = Drift(name="D0", length=100.0, aperture=20.0, n_steps=1)
    ref = ReferenceParticle(species=PROTON, w_kin=3.0, frequency=352.21)
    M_q = q.transfer_matrix(ref)
    M_d = d.transfer_matrix(ref)
    np.testing.assert_array_almost_equal(M_q, M_d)


def test_skew_quad_couples_x_and_y():
    """A 45-degree skew quad must produce an M[2,0] != 0 (x -> y coupling)."""
    import numpy as np
    from linac_gen.core.particle import PROTON
    from linac_gen.core.reference import ReferenceParticle
    from linac_gen.elements.quadrupole import Quadrupole

    ref = ReferenceParticle(species=PROTON, w_kin=3.0, frequency=352.21)

    normal = Quadrupole("QN", length=100.0, gradient=5.0, skew_angle=0.0)
    skew45 = Quadrupole("QS", length=100.0, gradient=5.0, skew_angle=45.0)

    M_normal = normal.transfer_matrix(ref)
    M_skew = skew45.transfer_matrix(ref)

    # Normal quad is decoupled.
    assert abs(M_normal[2, 0]) < 1e-12
    assert abs(M_normal[2, 1]) < 1e-12
    # Skew couples x into y.
    assert abs(M_skew[2, 0]) > 0.001


def test_skew_zero_matches_normal():
    """skew_angle=0 must be bit-identical to the normal-quad matrix."""
    import numpy as np
    from linac_gen.core.particle import PROTON
    from linac_gen.core.reference import ReferenceParticle
    from linac_gen.elements.quadrupole import Quadrupole

    ref = ReferenceParticle(species=PROTON, w_kin=3.0, frequency=352.21)
    M_a = Quadrupole("Q", length=50.0, gradient=5.0, skew_angle=0.0).transfer_matrix(ref)
    M_b = Quadrupole("Q", length=50.0, gradient=5.0).transfer_matrix(ref)
    np.testing.assert_allclose(M_a, M_b, atol=0)


def test_skew_180_equals_normal():
    """Rotation by 180 degrees: R(180)·M·R(-180) == M  (rotations are inverse in R+R- order)."""
    import numpy as np
    from linac_gen.core.particle import PROTON
    from linac_gen.core.reference import ReferenceParticle
    from linac_gen.elements.quadrupole import Quadrupole

    ref = ReferenceParticle(species=PROTON, w_kin=3.0, frequency=352.21)
    M0 = Quadrupole("Q", length=50.0, gradient=5.0, skew_angle=0.0).transfer_matrix(ref)
    M180 = Quadrupole("Q", length=50.0, gradient=5.0, skew_angle=180.0).transfer_matrix(ref)
    np.testing.assert_allclose(M0, M180, atol=1e-12)


def test_skew_determinant_is_unity():
    """The transverse 4x4 block of any skew QUAD transfer matrix must be symplectic."""
    import numpy as np
    from linac_gen.core.particle import PROTON
    from linac_gen.core.reference import ReferenceParticle
    from linac_gen.elements.quadrupole import Quadrupole

    ref = ReferenceParticle(species=PROTON, w_kin=3.0, frequency=352.21)
    for theta in (0.0, 10.0, 33.3, 45.0, 90.0, 170.0):
        M = Quadrupole("Q", length=70.0, gradient=4.5,
                       skew_angle=theta).transfer_matrix(ref)
        assert abs(np.linalg.det(M[:4, :4]) - 1.0) < 1e-10, (
            f"skew_angle={theta} breaks symplecticity: det={np.linalg.det(M[:4,:4])}"
        )
