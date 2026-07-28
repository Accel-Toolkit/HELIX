import numpy as np
import pytest
from linac_gen.core.beam import Beam
from linac_gen.core.particle import PROTON
from linac_gen.core.reference import ReferenceParticle

def test_create_beam():
    ref = ReferenceParticle(species=PROTON, w_kin=3.0, frequency=352.21)
    beam = Beam(ref=ref, n_particles=1000, current=60.0)
    assert beam.particles.shape == (1000, 6)
    assert beam.n_particles == 1000
    assert beam.current == 60.0

def test_particles_initialized_to_zero():
    ref = ReferenceParticle(species=PROTON, w_kin=3.0, frequency=352.21)
    beam = Beam(ref=ref, n_particles=100, current=0.0)
    assert np.all(beam.particles == 0.0)

def test_lost_mask():
    ref = ReferenceParticle(species=PROTON, w_kin=3.0, frequency=352.21)
    beam = Beam(ref=ref, n_particles=100, current=0.0)
    assert beam.lost.shape == (100,)
    assert not np.any(beam.lost)
    beam.lost[5] = True
    assert beam.n_alive == 99

def test_alive_particles():
    ref = ReferenceParticle(species=PROTON, w_kin=3.0, frequency=352.21)
    beam = Beam(ref=ref, n_particles=10, current=0.0)
    beam.lost[3] = True
    beam.lost[7] = True
    alive = beam.alive_particles
    assert alive.shape == (8, 6)

def test_record_loss():
    ref = ReferenceParticle(species=PROTON, w_kin=3.0, frequency=352.21)
    beam = Beam(ref=ref, n_particles=10, current=0.0)
    beam.particles[3, 0] = 25.0
    beam.particles[3, 2] = 10.0
    beam.particles[3, 5] = 0.5
    beam.record_loss(particle_id=3, s=100.0, element_name="QUAD_01")
    assert beam.lost[3] == True
    assert len(beam.loss_table) == 1
    assert beam.loss_table[0]["s"] == 100.0
    assert beam.loss_table[0]["element_name"] == "QUAD_01"

def test_species_shortcut():
    ref = ReferenceParticle(species=PROTON, w_kin=3.0, frequency=352.21)
    beam = Beam(ref=ref, n_particles=10, current=0.0)
    assert beam.species is PROTON

def test_frequency_shortcut():
    ref = ReferenceParticle(species=PROTON, w_kin=3.0, frequency=352.21)
    beam = Beam(ref=ref, n_particles=10, current=0.0)
    assert beam.frequency == 352.21

def test_record_loss_no_double():
    """Recording the same particle twice should not duplicate entries."""
    ref = ReferenceParticle(species=PROTON, w_kin=3.0, frequency=352.21)
    beam = Beam(ref=ref, n_particles=10, current=0.0)
    beam.record_loss(particle_id=3, s=100.0, element_name="Q1")
    beam.record_loss(particle_id=3, s=200.0, element_name="Q2")  # should be no-op
    assert len(beam.loss_table) == 1
    assert beam.loss_table[0]["s"] == 100.0  # first loss recorded, second ignored
