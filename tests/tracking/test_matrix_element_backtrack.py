"""The backtracker undoes a MatrixElement once, centroid offset included, as the tracker applies it once (a general
6x6 cannot be sliced, and its transfer_matrix ignores a slice length): it applied the inverse four times."""
import copy
import warnings

import numpy as np
import pytest

from linac_gen.core.beam import Beam
from linac_gen.core.lattice import Lattice
from linac_gen.core.particle import PROTON
from linac_gen.core.reference import ReferenceParticle
from linac_gen.elements.drift import Drift
from linac_gen.elements.matrix_element import MatrixElement
from linac_gen.tracking.backtrack import backtrack_distribution
from linac_gen.tracking.tracker import Tracker

F = 352.21
M = np.eye(6) + 0.05 * np.random.default_rng(1).normal(size=(6, 6))
OFFSET = np.array([0.1, 0.0, -0.05, 0.0, 0.2, 0.001])


def _ref():
    return ReferenceParticle(species=PROTON, w_kin=3.0, frequency=F)


def _line(*elements):
    lat = Lattice()
    for e in elements:
        lat.add(e)
    return lat


@pytest.mark.parametrize("length", [0.0, 120.0], ids=["thin", "long"])
@pytest.mark.parametrize("offset", [None, OFFSET], ids=["no offset", "offset"])
def test_the_backtracker_undoes_the_map_once(length, offset):
    """Forward through a MatrixElement, then back: the particles it started with (the map was undone four times,
    and its offset not at all)."""
    lat = _line(Drift("D1", 50.0), MatrixElement("M1", M, length=length, offset=offset), Drift("D2", 50.0))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        beam = Beam(_ref(), n_particles=20, current=0.0)
        beam.particles[:] = np.random.default_rng(2).normal(size=(20, 6)) * [0.5, 0.5, 0.5, 0.5, 3.0, 0.002]
        start = beam.particles.copy()
        Tracker(copy.deepcopy(lat), beam).run()
        backtrack_distribution(copy.deepcopy(lat), beam, _ref())
    np.testing.assert_allclose(beam.particles, start, rtol=0.0, atol=1e-9)


def test_the_backtracker_undoes_the_space_charge_kick_over_a_long_map():
    """With space charge the tracker kicks over the element's length after the map: the backtracker undoes that kick,
    then the map (the forward run's own field solver, so the kick is undone to round-off)."""
    from linac_gen.core.config import SpaceChargeConfig
    from linac_gen.pic.pic_solver import PicSolver
    lat = _line(Drift("D1", 50.0), MatrixElement("M1", M, length=120.0, offset=OFFSET), Drift("D2", 50.0))
    beam = Beam(_ref(), n_particles=600, current=20.0)
    beam.particles[:] = np.random.default_rng(3).normal(size=(600, 6)) * [1.0, 0.3, 1.0, 0.3, 4.0, 0.003]
    start = beam.particles.copy()
    solver = PicSolver(SpaceChargeConfig(nx=32, ny=32, nz=32))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        Tracker(copy.deepcopy(lat), beam, pic_solver=solver).run()
        moved = beam.particles.copy()
        backtrack_distribution(copy.deepcopy(lat), beam, _ref(), pic_solver=solver)
    assert np.max(np.abs(moved - start)) > 0.1
    np.testing.assert_allclose(beam.particles, start, rtol=1e-8, atol=1e-10)
