"""MatrixElement: stores and applies a raw 6x6 map verbatim."""
from __future__ import annotations

import numpy as np
import pytest

from linac_gen.core.particle import PROTON
from linac_gen.core.reference import ReferenceParticle
from linac_gen.core.beam import Beam
from linac_gen.elements.matrix_element import MatrixElement


def _ref():
    return ReferenceParticle(species=PROTON, w_kin=100.0, frequency=352.21)


def test_rejects_non_6x6():
    with pytest.raises(ValueError, match="6x6"):
        MatrixElement("bad", np.eye(4))


def test_transfer_matrix_is_verbatim_and_energy_agnostic():
    rng = np.random.default_rng(0)
    M = rng.normal(size=(6, 6))
    el = MatrixElement("m", M, length=0.0)
    np.testing.assert_array_equal(el.transfer_matrix(_ref()), M)
    # different reference energy -> same matrix (Cheetah CustomTransferMap)
    other = ReferenceParticle(species=PROTON, w_kin=5.0, frequency=352.21)
    np.testing.assert_array_equal(el.transfer_matrix(other), M)


def test_track_applies_matrix_to_particles():
    rng = np.random.default_rng(1)
    M = np.eye(6) + 0.05 * rng.normal(size=(6, 6))
    el = MatrixElement("m", M)
    ref = _ref()
    beam = Beam(ref=ref, n_particles=200, current=0.0)
    beam.particles[:] = rng.normal(0, 1.0, (200, 6))
    expected = (M @ beam.particles.T).T
    el.track(beam)
    np.testing.assert_allclose(beam.particles, expected, rtol=1e-12,
                               atol=1e-12)


def test_affine_offset_applied():
    M = np.eye(6)
    c = np.array([0.1, 0.0, -0.2, 0.0, 0.0, 0.0])
    el = MatrixElement("m", M, offset=c)
    beam = Beam(ref=_ref(), n_particles=3, current=0.0)
    beam.particles[:] = 0.0
    el.track(beam)
    for row in beam.particles:
        np.testing.assert_allclose(row, c)


def test_length_advances_reference():
    el = MatrixElement("m", np.eye(6), length=300.0)  # 300 mm
    beam = Beam(ref=_ref(), n_particles=2, current=0.0)
    s0, phi0 = beam.ref.s, beam.ref.phi_s
    el.track(beam)
    assert beam.ref.s == pytest.approx(s0 + 300.0)
    assert beam.ref.phi_s > phi0            # drift-equivalent phase advance


def test_tracks_through_real_tracker():
    """End-to-end: a MatrixElement in a Lattice tracked by the production
    Tracker reproduces the hand-applied matrix."""
    from linac_gen.core.lattice import Lattice
    from linac_gen.tracking.tracker import Tracker
    rng = np.random.default_rng(2)
    M = np.eye(6) + 0.02 * rng.normal(size=(6, 6))
    lat = Lattice()
    lat.add(MatrixElement("m", M))
    beam = Beam(ref=_ref(), n_particles=50, current=0.0)
    beam.particles[:] = rng.normal(0, 0.5, (50, 6))
    expected = (M @ beam.particles.T).T
    Tracker(lat, beam).run()
    np.testing.assert_allclose(beam.particles, expected, rtol=1e-10,
                               atol=1e-12)
