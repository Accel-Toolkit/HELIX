# tests/tracking/test_fieldmap_substep_rows.py
"""Sub-step rows inside a field map (multiparticle, ``record_substeps``).

The tracker steps a field map in space-charge bundles: ``sc_every``
half-steps of ``ds/2``, the kick, ``sc_every`` more half-steps.  The rows
must fall every full integration step over the WHOLE map.  They used to be
written only during the second half of each bundle, so every run of rows
followed a gap of half a bundle (10 mm at 50 kicks/m) — straight segments
in every σ(s) curve inside solenoids and dipoles.  Recording only reads
the beam, so it must not change a single particle either.
"""
import numpy as np
import pytest

from linac_gen.core.beam import Beam
from linac_gen.core.config import SpaceChargeConfig
from linac_gen.core.lattice import Lattice
from linac_gen.core.particle import PROTON
from linac_gen.core.reference import ReferenceParticle
from linac_gen.core.simulation import Simulation
from linac_gen.elements.field_map import FieldMap
from linac_gen.io.field_map_reader import FieldMapData
from linac_gen.tracking.tracker import field_map_step_counts

L_MAP = 300.0                                             # mm


def _solenoid_lattice():
    z = np.linspace(0.0, L_MAP, 151)
    fd = FieldMapData(z=z, Bz=0.35 * (0.5 - 0.5 * np.cos(2 * np.pi * z / L_MAP)),
                      symmetry="1d")
    lat = Lattice()
    lat.add(FieldMap("SOL", length=L_MAP, field_data=fd, phase=0.0, frequency=162.5,
                     aperture=30.0, n_steps=120))
    return lat


def _dc_beam(n=400):
    b = Beam(ref=ReferenceParticle(species=PROTON, w_kin=0.03, frequency=162.5),
             n_particles=n, current=5.0)
    b.continuous = True
    rng = np.random.default_rng(3)
    for j, sd in enumerate((2.0, 3.0, 2.0, 3.0)):
        b.particles[:, j] = rng.normal(0.0, sd, n)
    b.particles[:, 4] = rng.uniform(-180.0, 180.0, n)
    return b


def _run(record_substeps):
    lat, beam = _solenoid_lattice(), _dc_beam()
    rec = Simulation(lat, beam, space_charge=SpaceChargeConfig(),
                     record_substeps=record_substeps).run()
    return lat, rec, beam


def test_rows_fall_every_integration_step_over_the_whole_map():
    lat, rec, _beam = _run(True)
    n_int, _n_sc, ds, sc_every, _n_bundles = field_map_step_counts(lat, lat.elements[0])
    assert sc_every > 1                    # several steps per bundle: the case that went wrong
    s = np.asarray(rec.s, float)
    assert len(s) == n_int + 1             # the input row, then one row per step
    np.testing.assert_allclose(np.diff(s), ds, rtol=1e-9)
    assert s[-1] == pytest.approx(L_MAP)   # the last row is the map's exit
    assert rec.element_exit_idx == [n_int]


def test_recording_leaves_the_beam_unchanged():
    _, _, off = _run(False)
    _, _, on = _run(True)
    np.testing.assert_array_equal(off.particles, on.particles)
    np.testing.assert_array_equal(off.lost, on.lost)
