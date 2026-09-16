"""``compute_transfer_matrix`` honours ``tilt_deg`` the way every other mode
does (2026-09-06).  Before, the numpy matrix composition tracked a tilted
element untilted while the MP tracker, the envelope solver and the torch
matrix path all rotated it."""
import numpy as np
import pytest

from linac_gen.core.beam import Beam
from linac_gen.core.lattice import Lattice
from linac_gen.core.particle import PROTON
from linac_gen.core.reference import ReferenceParticle
from linac_gen.elements.drift import Drift
from linac_gen.elements.multipole import Multipole
from linac_gen.elements.quadrupole import Quadrupole
from linac_gen.tracking.matrix_tracking import compute_transfer_matrix, get_element_matrix


def _ref():
    return ReferenceParticle(species=PROTON, w_kin=3.0, frequency=352.21)


def _line(**quad_kw):
    lat = Lattice()
    lat.add(Drift("d1", length=100.0))
    lat.add(Quadrupole("q", length=80.0, gradient=6.0, **quad_kw))
    lat.add(Drift("d2", length=100.0))
    return lat


@pytest.mark.parametrize("tilt", [30.0, -17.5, 90.0])
def test_tilt_misalignment_equals_skew_angle(tilt):
    """A tilt misalignment of a normal quad IS a skew rotation of its axes:
    the two matrices are the same product of the same rotation matrices."""
    M_tilt = compute_transfer_matrix(_line(tilt_deg=tilt), _ref())
    M_skew = compute_transfer_matrix(_line(skew_angle=tilt), _ref())
    np.testing.assert_allclose(M_tilt, M_skew, rtol=0, atol=1e-15)
    assert abs(M_tilt[1, 2]) > 1e-3 or tilt == 90.0          # coupled


def test_matches_torch_matrix_path():
    torch = pytest.importorskip("torch")
    from linac_gen.tracking.torch_tracking import compute_transfer_matrix_torch
    lat = _line(tilt_deg=25.0)
    M_np = compute_transfer_matrix(lat, _ref())
    M_t = compute_transfer_matrix_torch(lat, _ref()).detach().numpy()
    np.testing.assert_allclose(M_np, M_t, rtol=1e-12, atol=1e-12)


def test_matches_multiparticle_jacobian():
    """Six unit-vector particles through the tilted quad give the columns of
    the matrix the MP tracker really applies."""
    from linac_gen.tracking.tracker import Tracker
    lat = _line(tilt_deg=25.0)
    M = compute_transfer_matrix(lat, _ref())
    beam = Beam(ref=_ref(), n_particles=4, current=0.0)
    beam.particles[:] = 0.0
    beam.particles[:4, :4] = np.eye(4) * 0.5           # 0.5 mm / 0.5 mrad probes
    Tracker(lat, beam).run()
    J = beam.particles[:4, :4].T / 0.5
    np.testing.assert_allclose(J, M[:4, :4], rtol=1e-9, atol=1e-9)


def test_untilted_lattice_is_the_plain_product():
    lat = _line()
    ref = _ref()
    M = compute_transfer_matrix(lat, ref)
    P = np.eye(6)
    rc = ref.copy()
    for el in lat.elements:
        P = get_element_matrix(el, rc.copy()) @ P
        rc.s += el.length
    assert np.array_equal(M, P)


def test_multipole_and_passive_elements_are_not_wrapped():
    """Multipole handles its own tilt inside apply_kick; wrapping it here
    would rotate it twice."""
    ref = _ref()
    lat = Lattice()
    m = Multipole("m", knl=[0.0, 0.4], tilt_deg=30.0)
    lat.add(m)
    M = compute_transfer_matrix(lat, ref)
    assert np.array_equal(M, m.kick_matrix(ref.copy()))
