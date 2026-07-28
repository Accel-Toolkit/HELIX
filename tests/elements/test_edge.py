"""Standalone EDGE element -- pole-face rotation plus fringe correction."""
import math
import numpy as np
import pytest
from linac_gen.core.particle import PROTON
from linac_gen.core.reference import ReferenceParticle
from linac_gen.elements.edge import Edge


def test_edge_identity_when_zero_rotation():
    ref = ReferenceParticle(species=PROTON, w_kin=3.0, frequency=352.21)
    e = Edge("E", pole_rotation=0.0, rho=500.0)
    M = e.transfer_matrix(ref)
    np.testing.assert_allclose(M, np.eye(6), atol=1e-14)


def test_edge_horizontal_focusing_sign():
    """Positive pole-face rotation focuses horizontally, defocuses vertically."""
    ref = ReferenceParticle(species=PROTON, w_kin=3.0, frequency=352.21)
    rho_mm = 500.0
    beta_deg = 20.0
    e = Edge("E", pole_rotation=beta_deg, rho=rho_mm)
    M = e.transfer_matrix(ref)
    tan_b = math.tan(math.radians(beta_deg))
    rho_m = rho_mm * 1e-3
    # With K1=0.45 default + gap=0, the fringe psi=0, so the vertical entry
    # matches the raw -tan(beta)/rho formula exactly.
    assert M[1, 0] == pytest.approx(tan_b / rho_m, rel=1e-10)
    assert M[3, 2] == pytest.approx(-tan_b / rho_m, rel=1e-10)


def test_edge_fringe_correction_reduces_vertical_focusing_magnitude():
    """A positive K1 fringe reduces the |vertical defocusing|."""
    ref = ReferenceParticle(species=PROTON, w_kin=3.0, frequency=352.21)
    no_fringe = Edge("A", pole_rotation=20.0, rho=500.0, gap=0.0, k1=0.0)
    with_fringe = Edge("B", pole_rotation=20.0, rho=500.0, gap=50.0, k1=0.45)
    M0 = no_fringe.transfer_matrix(ref)
    M1 = with_fringe.transfer_matrix(ref)
    # With positive K1 and gap>0, psi>0, so tan(beta-psi) < tan(beta),
    # so |M[3,2]| decreases.
    assert abs(M1[3, 2]) < abs(M0[3, 2])


def test_edge_apply_kick_matches_transfer_matrix():
    """apply() on a beam should be equivalent to M @ particles."""
    ref = ReferenceParticle(species=PROTON, w_kin=3.0, frequency=352.21)
    from linac_gen.core.beam import Beam
    e = Edge("E", pole_rotation=20.0, rho=500.0, gap=50.0, k1=0.45)
    M = e.transfer_matrix(ref)

    beam_a = Beam(ref=ref, n_particles=5, current=0.0)
    beam_a.particles[:, 0] = [1.0, -1.0, 2.0, 0.0, 0.5]
    beam_a.particles[:, 2] = [0.0, 2.0, -1.0, 1.5, -0.5]
    expected = (M @ beam_a.particles.T).T

    e.apply(beam_a)
    np.testing.assert_allclose(beam_a.particles, expected, atol=1e-12)


def test_edge_vertical_bend_swaps_focus_planes():
    """hv=1 swaps the horizontal/vertical roles."""
    ref = ReferenceParticle(species=PROTON, w_kin=3.0, frequency=352.21)
    e_h = Edge("H", pole_rotation=20.0, rho=500.0, hv=0)
    e_v = Edge("V", pole_rotation=20.0, rho=500.0, hv=1)
    Mh = e_h.transfer_matrix(ref)
    Mv = e_v.transfer_matrix(ref)
    # For hv=1, M[3,2] gets the +tan/rho (focusing) and M[1,0] gets the -tan/rho.
    assert Mv[3, 2] == pytest.approx(Mh[1, 0], rel=1e-10)
    assert Mv[1, 0] == pytest.approx(Mh[3, 2], rel=1e-10)
