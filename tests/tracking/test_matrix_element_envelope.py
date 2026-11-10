"""The envelope applies a MatrixElement once, with sub-step recording and with space charge, then one kick over its
length, as the tracker does (a general 6x6 cannot be sliced, and its transfer_matrix ignores a slice length): it
applied the map once per slice."""
import copy
import warnings

import numpy as np
import pytest

from linac_gen.core.lattice import Lattice
from linac_gen.core.particle import PROTON
from linac_gen.core.reference import ReferenceParticle
from linac_gen.elements.matrix_element import MatrixElement
from linac_gen.tracking.envelope import EnvelopeSolver

F = 352.21
M = np.eye(6) + 0.05 * np.random.default_rng(1).normal(size=(6, 6))
INIT = dict(alpha_x=0.0, beta_x=1.0, emit_x=0.2, alpha_y=0.0, beta_y=1.0, emit_y=0.2, alpha_z=0.0, beta_z=1.0,
            emit_z=0.2)


def _ref():
    return ReferenceParticle(species=PROTON, w_kin=3.0, frequency=F)


def _line(*elements):
    lat = Lattice()
    for e in elements:
        lat.add(e)
    return lat


def _sigma_exit(lat, current, record_substeps):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = EnvelopeSolver(copy.deepcopy(lat), _ref(), dict(INIT), current=current,
                             record_substeps=record_substeps).run()
    return np.asarray(res.sigma_matrix[-1])


def test_the_envelope_applies_a_long_map_once_with_substeps():
    """Without space charge a long MatrixElement is its map: sub-step recording gives the thin map's sigma."""
    long = _sigma_exit(_line(MatrixElement("M1", M, length=120.0)), 0.0, True)
    thin = _sigma_exit(_line(MatrixElement("M1", M, length=0.0)), 0.0, True)
    np.testing.assert_array_equal(long, thin)


def test_the_envelope_applies_a_long_map_once_with_space_charge():
    """With space charge: the map once, then one kick over the length -- the map thin followed by an identity map of
    that length, which is the kick alone."""
    long = _sigma_exit(_line(MatrixElement("M1", M, length=120.0)), 5.0, False)
    split = _sigma_exit(_line(MatrixElement("M0", M, length=0.0), MatrixElement("I1", np.eye(6), length=120.0)),
                        5.0, False)
    np.testing.assert_allclose(long, split, rtol=1e-12, atol=1e-15)
    assert not np.allclose(long, _sigma_exit(_line(MatrixElement("M1", M, length=0.0)), 5.0, False))   # a kick


def test_the_kick_is_one_over_the_whole_length():
    """The kick is the envelope's space-charge kick over the element's length, at the sigma the map leaves."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        thin = EnvelopeSolver(_line(MatrixElement("M0", M, length=0.0)), _ref(), dict(INIT), current=5.0)
        after_map = np.asarray(thin.run().sigma_matrix[-1])
        kick = thin._sc_matrix(after_map, ds_mm=120.0)
    long = _sigma_exit(_line(MatrixElement("M1", M, length=120.0)), 5.0, False)
    np.testing.assert_allclose(long, kick @ after_map @ kick.T, rtol=1e-12, atol=1e-15)


def test_the_envelope_centroid_takes_the_maps_offset():
    """The map's constant term moves the centroid, as it moves every particle (c -> M c + o), with and without
    sub-steps and space charge (the envelope dropped it)."""
    offset = np.array([0.1, 0.0, -0.05, 0.0, 0.2, 0.001])
    c_in = np.array([0.3, -0.2, 0.1, 0.05, 1.5, 0.0004])
    for length, current, substeps in ((0.0, 0.0, False), (120.0, 0.0, True), (120.0, 5.0, False)):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = EnvelopeSolver(_line(MatrixElement("M1", M, length=length, offset=offset)), _ref(),
                                 dict(INIT, centroid=list(c_in)), current=current, record_substeps=substeps).run()
        np.testing.assert_allclose(np.asarray(res.centroid[-1]), M @ c_in + offset, rtol=1e-12, atol=1e-15,
                                   err_msg=f"L={length} I={current} substeps={substeps}")


def test_with_substeps_each_mode_records_the_long_maps_exit():
    """With sub-step recording the multiparticle tracker leaves an extended element's exit row to the element: a long
    MatrixElement now records it, as the envelope does, so both point at a row at its exit."""
    from linac_gen.core.beam import Beam
    from linac_gen.elements.drift import Drift
    from linac_gen.tracking.tracker import Tracker
    lat = _line(Drift("D1", 50.0), MatrixElement("M1", M, length=120.0), Drift("D2", 50.0))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        beam = Beam(_ref(), n_particles=20, current=0.0)
        rec = Tracker(copy.deepcopy(lat), beam, record_substeps=True).run()
        env = EnvelopeSolver(copy.deepcopy(lat), _ref(), dict(INIT), current=0.0, record_substeps=True).run()
    for res in (rec, env):
        idx = list(res.element_exit_idx)
        assert len(set(idx)) == 3 and np.asarray(res.s)[idx[1]] == pytest.approx(170.0)
