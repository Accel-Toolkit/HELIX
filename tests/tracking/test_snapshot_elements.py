"""Targeted phase-space snapshots: snapshot_elements fires at the named
element (and only there), alive mask captured, tuple shape preserved."""
from __future__ import annotations

import numpy as np
import pytest

from linac_gen.core.particle import H_MINUS
from linac_gen.core.reference import ReferenceParticle
from linac_gen.core.beam import Beam
from linac_gen.core.lattice import Lattice
from linac_gen.core.simulation import Simulation
from linac_gen.elements.drift import Drift
from linac_gen.elements.quadrupole import Quadrupole


def _lattice():
    lat = Lattice()
    lat.add(Drift("D1", 100.0))
    lat.add(Quadrupole("Q1", 50.0, gradient=3.0))
    lat.add(Drift("D2", 100.0))
    lat.add(Quadrupole("Q2", 50.0, gradient=-3.0))
    lat.add(Drift("D3", 100.0))
    return lat


def _beam(n=300):
    ref = ReferenceParticle(species=H_MINUS, w_kin=2.12, frequency=162.5)
    b = Beam(ref=ref, n_particles=n, current=0.0)
    b.particles[:] = np.random.default_rng(0).normal(0, 0.3, (n, 6))
    return b


def test_snapshot_fires_only_at_named_elements():
    rec = Simulation(_lattice(), _beam(),
                     snapshot_elements={"Q1", "Q2"}).run()
    # exactly two snapshots, at the two quad exits:
    # Q1 exit = D1+Q1 = 150; Q2 exit = D1+Q1+D2+Q2 = 300
    assert len(rec._snapshots) == 2
    assert sorted(round(s, 3) for s in rec._snapshots) == [150.0, 300.0]


def test_no_snapshots_by_default():
    rec = Simulation(_lattice(), _beam()).run()
    assert rec._snapshots == {}


def test_alive_at_filters_lost_and_mask_recorded():
    rec = Simulation(_lattice(), _beam(), snapshot_elements={"Q1"}).run()
    s = next(iter(rec._snapshots))
    assert s in rec._snapshot_masks
    alive = rec.alive_at(s)
    full = rec.beam_at(s)[0]
    assert alive.shape[1] == 6 and len(alive) <= len(full)


def test_beam_at_stays_two_tuple():
    """hdf5/openpmd exporters unpack (particles, ref) — must not widen."""
    rec = Simulation(_lattice(), _beam(), snapshot_elements={"Q1"}).run()
    s = next(iter(rec._snapshots))
    val = rec.beam_at(s)
    assert isinstance(val, tuple) and len(val) == 2


def test_missing_snapshot_raises_keyerror():
    rec = Simulation(_lattice(), _beam()).run()
    with pytest.raises(KeyError):
        rec.beam_at(999.0)
