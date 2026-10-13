"""linac_gen.core.beam.transport_alive applies a 6x6 matrix to the particles
still in the beam with the bits of the masked product
``P[alive] = (M @ P[alive].T).T`` -- on the array itself when none is lost
(the same BLAS call on the same layout), on the masked copy otherwise --
and the elements that use it (quadrupole, solenoid, dipole, edge, NCELLS
drifts) track as before."""
from __future__ import annotations

import numpy as np
import pytest

from linac_gen.core.beam import Beam, transport_alive
from linac_gen.core.particle import PROTON
from linac_gen.core.reference import ReferenceParticle


def _bits(a):
    return np.ascontiguousarray(a).view(np.uint64)


def _beam(n, seed=3, lost=0):
    rng = np.random.default_rng(seed)
    b = Beam(ref=ReferenceParticle(species=PROTON, w_kin=2.1, frequency=162.5),
             n_particles=n, current=0.0)
    b.particles[:] = rng.normal(size=(n, 6))
    if lost:
        b.lost[rng.choice(n, size=min(lost, n), replace=False)] = True
    return b


def _masked(beam, M):
    P = beam.particles.copy()
    alive = ~beam.lost
    P[alive] = (M @ P[alive].T).T
    return P


@pytest.mark.parametrize("n", [1, 2, 3, 5, 7, 8, 9, 16, 63, 64, 65, 1000, 5001,
                               50000])
@pytest.mark.parametrize("lost", [0, 1, 7])
def test_equals_the_masked_product(n, lost):
    rng = np.random.default_rng(n)
    M = rng.normal(size=(6, 6))
    M[np.abs(M) < 0.3] = 0.0                  # zeros, as transfer matrices have
    b = _beam(n, lost=lost)
    want = _masked(b, M)
    transport_alive(b, M)
    assert np.array_equal(_bits(b.particles), _bits(want))


def test_non_contiguous_and_malformed_masks_take_the_masked_form():
    M = np.random.default_rng(1).normal(size=(6, 6))
    b = _beam(100)
    b.particles = np.asfortranarray(b.particles)
    want = _masked(b, M)
    transport_alive(b, M)
    assert np.array_equal(_bits(b.particles), _bits(want))
    b = _beam(100)
    b.lost = np.zeros(100, dtype=np.int8)          # not a bool mask
    with np.errstate(all="ignore"):
        want = _masked(b, M)
    transport_alive(b, M)
    assert np.array_equal(_bits(b.particles), _bits(want))


def test_every_particle_lost_leaves_the_array_alone():
    M = np.random.default_rng(2).normal(size=(6, 6))
    b = _beam(50)
    b.lost[:] = True
    before = b.particles.copy()
    transport_alive(b, M)
    assert np.array_equal(_bits(b.particles), _bits(before))


def _elements():
    from linac_gen.elements.dipole import Dipole
    from linac_gen.elements.quadrupole import Quadrupole
    from linac_gen.elements.solenoid import Solenoid
    return {
        "quadrupole": Quadrupole(name="Q", length=80.0, gradient=12.0),
        "solenoid": Solenoid(name="S", length=150.0, field=0.4),
        "dipole": Dipole(name="D", angle=10.0, rho=1500.0),
    }


@pytest.mark.parametrize("name", ["quadrupole", "solenoid", "dipole"])
@pytest.mark.parametrize("lost", [0, 5])
def test_elements_track_as_with_the_masked_product(name, lost, monkeypatch):
    """Each element, tracked with transport_alive and with the masked
    product put back in its place: the same particles, bit for bit."""
    import linac_gen.core.beam as beam_mod
    elem = _elements()[name]
    a, b = _beam(3001, lost=lost), _beam(3001, lost=lost)
    elem.track(a)

    def masked(beam, M):
        alive = ~beam.lost
        beam.particles[alive] = (M @ beam.particles[alive].T).T
    mod = __import__(type(elem).__module__, fromlist=["transport_alive"])
    monkeypatch.setattr(mod, "transport_alive", masked)
    elem.track(b)
    assert np.array_equal(_bits(a.particles), _bits(b.particles))
    assert beam_mod.transport_alive is transport_alive
