"""Tests for PIC coordinate conversion (beam phase-space <-> spatial)."""
import numpy as np
import math
import pytest
from linac_gen.core.reference import ReferenceParticle
from linac_gen.core.particle import PROTON
from linac_gen.core.beam import Beam
from linac_gen.core.constants import C_LIGHT
from linac_gen.pic.coordinates import beam_to_spatial, spatial_z_to_dphi


@pytest.fixture
def ref():
    """Reference particle at 3 MeV, 352.21 MHz."""
    return ReferenceParticle(species=PROTON, w_kin=3.0, frequency=352.21)


@pytest.fixture
def beam(ref):
    """Small beam with 5 particles."""
    b = Beam(ref=ref, n_particles=5, current=20.0)
    # Set known deviations
    b.particles[0] = [1.0, 0.5, -0.3, 0.1, -10.0, 0.01]
    b.particles[1] = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    b.particles[2] = [-2.0, 1.0, 1.5, -0.5, 5.0, -0.02]
    b.particles[3] = [0.5, -0.2, 0.8, 0.3, -20.0, 0.05]
    b.particles[4] = [3.0, 0.0, -1.0, 0.0, 15.0, 0.0]
    return b


def test_dphi_zero_gives_z_zero(beam):
    """dphi=0 must map to z=0."""
    coords = beam_to_spatial(beam)
    # Particle 1 has dphi=0
    assert abs(coords[1, 2]) < 1e-15


def test_negative_dphi_gives_positive_z(beam):
    """Negative dphi (particle ahead in phase) -> positive z (ahead in space)."""
    coords = beam_to_spatial(beam)
    # Particle 0: dphi = -10 -> z > 0
    assert coords[0, 2] > 0.0
    # Particle 3: dphi = -20 -> z > 0
    assert coords[3, 2] > 0.0


def test_positive_dphi_gives_negative_z(beam):
    """Positive dphi (particle behind in phase) -> negative z (behind in space)."""
    coords = beam_to_spatial(beam)
    # Particle 2: dphi = +5 -> z < 0
    assert coords[2, 2] < 0.0
    # Particle 4: dphi = +15 -> z < 0
    assert coords[4, 2] < 0.0


def test_transverse_coordinates_unchanged(beam):
    """x and y pass through directly from beam deviations."""
    coords = beam_to_spatial(beam)
    alive = beam.alive_mask
    np.testing.assert_array_equal(coords[:, 0], beam.particles[alive, 0])
    np.testing.assert_array_equal(coords[:, 1], beam.particles[alive, 2])


def test_round_trip_dphi_z_dphi(ref):
    """dphi -> z -> dphi round-trip must preserve values to machine precision."""
    dphi_orig = np.array([-30.0, -10.0, 0.0, 5.0, 20.0, 45.0])
    z = -dphi_orig * ref.beta * ref.wavelength / 360.0
    dphi_back = spatial_z_to_dphi(z, ref)
    np.testing.assert_allclose(dphi_back, dphi_orig, atol=1e-12)


def test_known_value_dphi_to_z():
    """At 352.21 MHz, beta=0.08: dphi=-10 -> z ~ 1.89 mm."""
    # Create a reference at specific beta
    # beta=0.08 -> gamma = 1/sqrt(1-beta^2), w_kin = (gamma-1)*mass
    beta = 0.08
    gamma = 1.0 / math.sqrt(1.0 - beta**2)
    w_kin = (gamma - 1.0) * PROTON.mass
    ref = ReferenceParticle(species=PROTON, w_kin=w_kin, frequency=352.21)

    wavelength = C_LIGHT / (352.21e6) * 1000.0  # mm
    expected_z = 10.0 * beta * wavelength / 360.0

    b = Beam(ref=ref, n_particles=1, current=1.0)
    b.particles[0, 4] = -10.0  # dphi = -10 deg

    coords = beam_to_spatial(b)
    assert abs(coords[0, 2] - expected_z) < 0.01
    # Verify the approximate value (~1.89 mm)
    assert abs(expected_z - 1.89) < 0.1


def test_output_shape(beam):
    """beam_to_spatial returns (N_alive, 3) array."""
    coords = beam_to_spatial(beam)
    assert coords.shape == (beam.n_alive, 3)


def test_lost_particles_excluded(beam):
    """Lost particles should not appear in spatial output."""
    beam.record_loss(2, s=0.0, element_name="test")
    coords = beam_to_spatial(beam)
    assert coords.shape[0] == beam.n_alive
    assert coords.shape[0] == 4  # 5 - 1 lost


def test_spatial_z_to_dphi_scalar(ref):
    """spatial_z_to_dphi works with a scalar z."""
    z = 1.0  # mm
    dphi = spatial_z_to_dphi(z, ref)
    expected = -z * 360.0 / (ref.beta * ref.wavelength)
    assert abs(dphi - expected) < 1e-12


def test_spatial_z_to_dphi_array(ref):
    """spatial_z_to_dphi works with array input."""
    z = np.array([0.0, 1.0, -2.0, 5.0])
    dphi = spatial_z_to_dphi(z, ref)
    assert dphi.shape == (4,)
    assert abs(dphi[0]) < 1e-15  # z=0 -> dphi=0
