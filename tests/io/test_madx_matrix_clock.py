"""An imported MAD-X MATRIX holds its phase in degrees of the clock it was converted on (MAD-X's t holds no clock: the
first cavity's frequency, or a 352.21 MHz placeholder in a line with none), records that clock, and gives the same map
in the degrees of whatever frequency it is run at: M -> D M D^-1, D[4, 4] = f_run / clock."""
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
from linac_gen.io.madx_parser import parse_madx
from linac_gen.tracking.matrix_tracking import compute_transfer_matrix, get_element_matrix
from linac_gen.tracking.tracker import Tracker

W = 3.0                                   # MeV
M0 = np.eye(6) + 0.04 * np.random.default_rng(7).normal(size=(6, 6))


def _ref(f):
    return ReferenceParticle(species=PROTON, w_kin=W, frequency=f)


def test_the_map_at_another_clock_is_the_same_map_in_its_degrees():
    off = np.array([0.1, 0.0, -0.05, 0.0, 0.2, 0.001])
    e = MatrixElement("M", M0, offset=off, clock_mhz=352.21)
    M, o = e.map_at(_ref(162.5))
    D = np.diag([1, 1, 1, 1, 162.5 / 352.21, 1.0])
    np.testing.assert_allclose(M, D @ M0 @ np.linalg.inv(D), rtol=1e-14, atol=0.0)
    assert M[4, 4] == M0[4, 4]
    np.testing.assert_allclose(o, D @ off, rtol=1e-15, atol=0.0)
    assert e.map_at(_ref(352.21))[0] is e.matrix                       # at its clock: the stored map
    assert MatrixElement("N", M0).map_at(_ref(162.5))[0] is M0           # no clock: the run's degrees
    cache = {}
    a = get_element_matrix(e, _ref(162.5), cache=cache)
    e.clock_mhz = 162.5                                                  # an edit is a new map
    assert np.array_equal(get_element_matrix(e, _ref(162.5), cache=cache), M0) and not np.array_equal(a, M0)


def _drift_deck(length_m):
    """A MAD-X line with no cavity, holding one MATRIX that is a drift's map in MAD-X's (t, pt) basis:
    R12 = R34 = L, R56 = L / (beta^2 gamma^2)."""
    r = _ref(352.21)
    energy_gev = (W + PROTON.mass) / 1000.0
    r56 = length_m / (r.beta ** 2 * r.gamma ** 2)
    return (f"beam, particle=proton, energy={energy_gev!r};\n"
            f"d: matrix, l={length_m!r}, rm12={length_m!r}, rm34={length_m!r}, rm56={r56!r};\n"
            "line_d: line=(d);\nuse, period=line_d;\n")


@pytest.mark.parametrize("f", [162.5, 325.0, 352.21])
def test_an_imported_drift_matrix_is_the_drift_at_the_run_clock(tmp_path, f):
    """The external anchor is HELIX's own drift at the run's frequency (the import used the placeholder clock)."""
    deck = tmp_path / "d.madx"
    deck.write_text(_drift_deck(0.5))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        lat, _meta = parse_madx(str(deck))
    m = next(e for e in lat.elements if isinstance(e, MatrixElement))
    assert m.clock_mhz == pytest.approx(352.21)
    np.testing.assert_allclose(compute_transfer_matrix(lat, _ref(f)), Drift("D", 500.0).transfer_matrix(_ref(f)),
                               rtol=1e-12, atol=1e-13)


def test_a_line_on_one_clock_comes_back_from_madx_as_it_was(tmp_path):
    """A line at 162.5 MHz with no cavity, linearised to MAD-X and read back (on the placeholder clock): its matrix
    walk and a multiparticle run are the original's (R45 was 352.21 / 162.5 times too large)."""
    from linac_gen.io.madx_writer import write_madx
    lat = Lattice()
    for e in (Drift("D1", 200.0), MatrixElement("M1", M0), Drift("D2", 150.0)):
        lat.add(e)
    out = tmp_path / "line.madx"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        write_madx(lat, out, _ref(162.5), linearize=True)
        back, _meta = parse_madx(str(out))
    assert any(isinstance(e, MatrixElement) and e.clock_mhz > 0 for e in back.elements)
    np.testing.assert_allclose(compute_transfer_matrix(back, _ref(162.5)), compute_transfer_matrix(lat, _ref(162.5)),
                               rtol=1e-12, atol=1e-12)
    runs = []
    for line in (lat, back):
        beam = Beam(_ref(162.5), n_particles=10, current=0.0)
        beam.particles[:] = np.random.default_rng(3).normal(size=(10, 6)) * [0.5, 0.5, 0.5, 0.5, 3.0, 0.002]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            Tracker(copy.deepcopy(line), beam).run()
        runs.append(beam.particles.copy())
    np.testing.assert_allclose(runs[1], runs[0], rtol=1e-11, atol=1e-11)


def test_a_matrix_after_the_reference_is_lost_records_its_clock(tmp_path):
    """A MATRIX whose kick6 takes the reference below zero energy: the maps after it are converted about the beam
    energy, on the same clock, which they record too."""
    energy_gev = (W + PROTON.mass) / 1000.0
    deck = tmp_path / "lost.madx"
    deck.write_text(f"beam, particle=proton, energy={energy_gev!r};\n"
                    "a: matrix, l=0.1, kick6=-1.0;\nb: matrix, l=0.1, rm12=0.1;\n"
                    "line_l: line=(a, b);\nuse, period=line_l;\n")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        lat, _meta = parse_madx(str(deck))
    assert [e.clock_mhz for e in lat.elements if isinstance(e, MatrixElement)] == pytest.approx([352.21, 352.21])


_OFF = np.array([0.1, 0.0, -0.05, 0.0, 0.2, 0.001])


def test_every_mode_takes_the_map_at_the_run_clock():
    """A map recorded at 352.21 MHz, run at 162.5: the tracker, the backtracker (forward then back closes) and the
    envelope centroid all use D M D^-1 and D o."""
    from linac_gen.tracking.backtrack import backtrack_distribution
    from linac_gen.tracking.envelope import EnvelopeSolver
    e = MatrixElement("M1", M0, length=100.0, offset=_OFF, clock_mhz=352.21)
    M, o = e.map_at(_ref(162.5))
    lat = Lattice()
    lat.add(e)
    beam = Beam(_ref(162.5), n_particles=10, current=0.0)
    beam.particles[:] = np.random.default_rng(4).normal(size=(10, 6)) * [0.5, 0.5, 0.5, 0.5, 3.0, 0.002]
    start = beam.particles.copy()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        Tracker(copy.deepcopy(lat), beam).run()
        np.testing.assert_allclose(beam.particles, start @ M.T + o, rtol=1e-13, atol=1e-13)
        backtrack_distribution(copy.deepcopy(lat), beam, _ref(162.5))
        np.testing.assert_allclose(beam.particles, start, rtol=0.0, atol=1e-10)
        init = dict(alpha_x=0.0, beta_x=1.0, emit_x=0.2, alpha_y=0.0, beta_y=1.0, emit_y=0.2, alpha_z=0.0,
                    beta_z=1.0, emit_z=0.2, centroid=[0.3, -0.2, 0.1, 0.05, 1.5, 0.0004])
        res = EnvelopeSolver(copy.deepcopy(lat), _ref(162.5), init, current=0.0).run()
    np.testing.assert_allclose(np.asarray(res.centroid[-1]), M @ np.asarray(init["centroid"]) + o, rtol=1e-12,
                               atol=1e-15)


def test_a_map_on_another_clock_with_an_offset_comes_back_from_madx(tmp_path):
    """Linearised to MAD-X at the run's clock, offset included, and read back: the same particles.  (An energy offset
    comes back as the reference's gain, which moves the drift after it: the offset here has none.)"""
    from linac_gen.io.madx_writer import write_madx
    off = _OFF.copy()
    off[5] = 0.0
    lat = Lattice()
    for e in (Drift("D1", 200.0), MatrixElement("M1", M0, offset=off, clock_mhz=352.21), Drift("D2", 150.0)):
        lat.add(e)
    out = tmp_path / "off.madx"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        write_madx(lat, out, _ref(162.5), linearize=True)
        back, _meta = parse_madx(str(out))
    runs = []
    for line in (lat, back):
        beam = Beam(_ref(162.5), n_particles=10, current=0.0)
        beam.particles[:] = np.random.default_rng(5).normal(size=(10, 6)) * [0.5, 0.5, 0.5, 0.5, 3.0, 0.002]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            Tracker(copy.deepcopy(line), beam).run()
        runs.append(beam.particles.copy())
    np.testing.assert_allclose(runs[1], runs[0], rtol=1e-10, atol=1e-10)


def test_the_scan_pickers_do_not_offer_the_clock():
    pytest.importorskip("PyQt6")
    from linac_gen_gui.interphase.dialogs.parameter_scan import _scannable_params
    assert "clock_mhz" not in _scannable_params(MatrixElement("M", M0, clock_mhz=352.21))
