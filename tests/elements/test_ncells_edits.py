"""An NCELLS cavity edited after it was built is the cavity built with the new value.

Its cells, gaps and length derive from the mode, the cell count, βg, the field, the field and position corrections,
the frequency and the transit-time table.  These used to be derived once, in ``__init__``, so an edit through a
parameter scan, the assistant, a study or the API left them as they were: a scan of the field was flat.  Every engine
must now give the edited cavity exactly what it gives the cavity built fresh.
"""
import copy
import pickle
import warnings

import numpy as np
import pytest

from linac_gen.core.beam import Beam
from linac_gen.core.lattice import Lattice
from linac_gen.core.particle import PROTON
from linac_gen.core.reference import ReferenceParticle
from linac_gen.elements.drift import Drift
from linac_gen.elements.ncells import NCells, parse_ttf_tail
from linac_gen.tracking.envelope import EnvelopeSolver
from linac_gen.tracking.matrix_tracking import compute_transfer_matrix
from linac_gen.tracking.tracker import Tracker

BASE = dict(mode=1, n_cells=4, beta_g=0.2, eot_v_per_m=1.0e6, theta_s_deg=-30.0, aperture_mm=15.0, p_flag=2,
            frequency_mhz=325.0)
INIT = dict(alpha_x=0.0, beta_x=1.0, emit_x=0.2, alpha_y=0.0, beta_y=1.0, emit_y=0.2, alpha_z=0.0, beta_z=1.0,
            emit_z=0.2)
TTF = ["0.2", "0.8", "0.1", "0.05"]


def _cavity(**kw):
    kw = dict(BASE, **kw)
    if "_ttf" in kw:
        kw["ttf"] = kw.pop("_ttf")
    return NCells("NC", **kw)


def _ref():
    return ReferenceParticle(species=PROTON, w_kin=19.0, frequency=325.0)


def _engines(cavity):
    """What every engine gives a line of the cavity and a drift."""
    lat = Lattice()
    lat.add(cavity)
    lat.add(Drift("D1", 100.0))
    out = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        beam = Beam(_ref(), n_particles=50, current=0.0)
        beam.particles[:] = np.random.default_rng(5).normal(size=(50, 6)) * [0.5, 0.5, 0.5, 0.5, 3.0, 0.01]
        Tracker(lat, beam).run()
        out["tracker"] = np.concatenate([beam.particles.ravel(), [beam.ref.w_kin, beam.ref.phi_s, beam.ref.frequency]])
        res = EnvelopeSolver(lat, _ref(), dict(INIT), current=0.0).run()
        out["envelope"] = np.concatenate([np.asarray(res.sigma_matrix[-1]).ravel(), [res.ref_w_kin[-1]]])
        out["matrix"] = compute_transfer_matrix(lat, _ref())
    return out


EDITS = [({}, "mode", 0), ({}, "mode", 2), ({}, "n_cells", 5), ({}, "beta_g", 0.21), ({}, "beta_g", 0.0),
         ({}, "beta_g", -0.2), ({"beta_g": 0.0}, "beta_g", 0.2), ({}, "eot_v_per_m", 1.1e6), ({}, "k_eot_i", 0.1),
         ({}, "k_eot_o", -0.05), ({}, "dz_i_mm", 1.5), ({}, "dz_o_mm", -1.0), ({}, "frequency_mhz", 352.21)]


@pytest.mark.parametrize("base, name, value", EDITS, ids=[f"{n}={v}" + ("" if not b else " from βg 0")
                                                          for b, n, v in EDITS])
def test_an_edited_cavity_is_the_cavity_built_with_the_value(base, name, value):
    edited = _cavity(**base)
    setattr(edited, name, value)
    fresh = _cavity(**dict(base, **{name: value}))
    assert edited.length == fresh.length and edited.n_steps == fresh.n_steps
    a, b = _engines(edited), _engines(fresh)
    for engine in a:
        np.testing.assert_array_equal(a[engine], b[engine], err_msg=engine)


def test_an_edited_transit_time_table_is_the_cavity_built_with_it():
    edited = _cavity()
    edited._ttf = parse_ttf_tail(TTF)
    fresh = _cavity(_ttf=parse_ttf_tail(TTF))
    assert edited.beta_s == fresh.beta_s == 0.2
    a, b = _engines(edited), _engines(fresh)
    for engine in a:
        np.testing.assert_array_equal(a[engine], b[engine], err_msg=engine)


def test_an_edit_through_the_frequency_alias_is_the_cavity_built_with_it():
    edited = _cavity()
    edited.frequency = 352.21
    a, b = _engines(edited), _engines(_cavity(frequency_mhz=352.21))
    for engine in a:
        np.testing.assert_array_equal(a[engine], b[engine], err_msg=engine)


def test_a_whole_number_given_as_a_float_is_taken_and_a_fraction_refused():
    c = _cavity()
    c.n_cells = 6.0
    assert c.n_cells == 6 and isinstance(c.n_cells, int) and c.length == _cavity(n_cells=6).length
    for name in ("n_cells", "mode"):
        with pytest.raises(ValueError, match="whole number"):
            setattr(c, name, 1.5)
    assert c.n_cells == 6 and c.mode == 1


def test_the_length_follows_and_is_not_written_from_outside():
    c = _cavity()
    with pytest.raises(AttributeError, match="follows from"):
        c.length = 400.0
    c.beta_g = 0.25
    assert c.length == _cavity(beta_g=0.25).length > _cavity().length


def test_a_step_count_left_at_its_default_follows_the_cells():
    c = _cavity()
    c.n_cells = 6
    assert c.n_steps == _cavity(n_cells=6).n_steps == 12
    given = _cavity(n_steps=20)
    given.n_cells = 6
    assert given.n_steps == 20
    written = _cavity()
    written.n_steps = 30
    written.n_cells = 6
    assert written.n_steps == 30


def test_a_copy_and_a_pickle_keep_the_edit_and_edit_alone():
    c = _cavity()
    c.eot_v_per_m = 1.1e6
    fresh = _engines(_cavity(eot_v_per_m=1.1e6))
    for other in (copy.deepcopy(c), pickle.loads(pickle.dumps(c))):
        got = _engines(other)
        for engine in got:
            np.testing.assert_array_equal(got[engine], fresh[engine], err_msg=engine)
    d = copy.deepcopy(c)
    d.n_cells = 5
    assert d.length == _cavity(n_cells=5).length and c.length == _cavity().length


def test_the_assistant_edits_a_cavity_as_it_is_built():
    """Through the assistant's set_element_param, the path a chat edit takes: the edits take (a cell count arrives as
    a float), and the length is refused with the reason."""
    from linac_gen.assist.tools import TOOLS, WorkContext
    c = _cavity()
    lat = Lattice()
    lat.add(c)
    lat.add(Drift("D1", 100.0))
    ctx = WorkContext()
    ctx.lattice = lat
    set_param = TOOLS["set_element_param"].fn
    assert set_param(ctx, element_name="NC", param="eot_v_per_m", value=1.1e6)["status"] == "ok"
    assert set_param(ctx, element_name="NC", param="n_cells", value=5.0)["status"] == "ok"
    got, fresh = _engines(c), _engines(_cavity(eot_v_per_m=1.1e6, n_cells=5))
    for engine in got:
        np.testing.assert_array_equal(got[engine], fresh[engine], err_msg=engine)
    refused = set_param(ctx, element_name="NC", param="length", value=400.0)
    assert refused["status"] != "ok" and "follows from" in str(refused)
    assert c.length == _cavity(eot_v_per_m=1.1e6, n_cells=5).length


def test_a_refused_value_leaves_none_of_the_changes():
    """An edit of several knobs at once (a tuning plan) is all or nothing: when the cavity refuses one value, the knobs
    set before it go back."""
    from linac_gen.assist.tools import WorkContext
    c, d = _cavity(), Drift("D1", 100.0)
    lat = Lattice()
    lat.add(c)
    lat.add(d)
    ctx = WorkContext()
    ctx.lattice = lat
    for refused in (("length", 400.0), ("n_cells", 5.5), ("beta_s", 0.3)):
        with pytest.raises((AttributeError, ValueError)):
            ctx.apply_param_changes([(d, "length", 150.0), (c, "eot_v_per_m", 1.2e6), (c, *refused)])
        assert d.length == 100.0 and c.eot_v_per_m == 1.0e6 and c.length == _cavity().length


def test_the_transit_time_beta_follows_its_table():
    c = _cavity(_ttf=parse_ttf_tail(TTF))
    with pytest.raises(AttributeError, match="transit-time table"):
        c.beta_s = 0.3
    assert c.beta_s == 0.2


def test_a_command_line_override_refuses_a_fraction():
    """--set, --vary, scans and studies go through the same override, which used to truncate 6.5 to 6."""
    from linac_gen.cli.common import apply_element_override
    lat = Lattice()
    lat.add(_cavity())
    with pytest.raises(ValueError, match="whole number"):
        apply_element_override(lat, "NC.n_cells", "6.5")
    apply_element_override(lat, "NC.n_cells", "6")
    assert lat.elements[0].n_cells == 6 and lat.elements[0].length == _cavity(n_cells=6).length


def _fast_line(eot):
    """A βg=0 cavity of 6 cells for a 2 GeV proton (β 0.948), longer than cells sized for β 0.9, and a drift."""
    lat = Lattice()
    lat.add(NCells("NC", **dict(BASE, beta_g=0.0, n_cells=6, eot_v_per_m=eot, theta_s_deg=0.0)))
    lat.add(Drift("D1", 100.0))
    return lat, ReferenceParticle(species=PROTON, w_kin=2000.0, frequency=325.0)


def test_a_running_cavity_faster_than_beta_0_9_is_covered_whole():
    """With no field the cavity is a drift of its true length.  Cells sized for β 0.9 cut 133 mm off its end, in every
    engine alike, so the engines agreed with each other and not with the cavity."""
    lat, ref = _fast_line(0.0)
    c = lat.elements[0]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        M = compute_transfer_matrix(lat, ref.copy())
        res = EnvelopeSolver(lat, ref.copy(), dict(INIT), current=0.0).run()
    assert c.length > 6 * c._interior_cell_length(0.9) + 100.0
    assert M[0, 1] == pytest.approx((c.length + 100.0) * 1e-3, rel=1e-12)
    S0, S1 = np.asarray(res.sigma_matrix[0]), np.asarray(res.sigma_matrix[-1])
    np.testing.assert_allclose(S1, M @ S0 @ M.T, rtol=1e-9, atol=1e-12)


def test_a_running_cavity_faster_than_beta_0_9_is_the_same_in_every_engine():
    """With its field on: the tracker's linear map and the envelope's beam are the matrix walk's."""
    lat, ref = _fast_line(10.0e6)
    eps = np.array([1e-4, 1e-4, 1e-4, 1e-4, 1e-3, 1e-5])
    beam = Beam(ref.copy(), n_particles=12, current=0.0)
    beam.particles[:] = 0.0
    for j in range(6):
        beam.particles[2 * j, j], beam.particles[2 * j + 1, j] = eps[j], -eps[j]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        Tracker(_fast_line(10.0e6)[0], beam).run()
        M = compute_transfer_matrix(lat, ref.copy())
        res = EnvelopeSolver(_fast_line(10.0e6)[0], ref.copy(), dict(INIT), current=0.0).run()
    J = np.column_stack([(beam.particles[2 * j] - beam.particles[2 * j + 1])[:6] / (2 * eps[j]) for j in range(6)])
    np.testing.assert_allclose(J, M, rtol=1e-8, atol=1e-10 * np.abs(M).max())
    S0, S1 = np.asarray(res.sigma_matrix[0]), np.asarray(res.sigma_matrix[-1])
    np.testing.assert_allclose(S1, M @ S0 @ M.T, rtol=1e-9, atol=1e-12)


def test_a_field_scan_moves_the_energy():
    c = _cavity()
    w = []
    for eot in (0.8e6, 0.9e6, 1.0e6, 1.1e6, 1.2e6):
        c.eot_v_per_m = eot
        lat = Lattice()
        lat.add(c)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            w.append(EnvelopeSolver(lat, _ref(), dict(INIT), current=0.0).run().ref_w_kin[-1])
    assert np.all(np.diff(w) > 1e-3), w


class _Kicks:
    """A space-charge solver that records where it is asked to kick and does nothing."""
    config = None

    def __init__(self):
        self.ds = []

    def kick(self, beam, ds_mm):
        self.ds.append(ds_mm)


@pytest.mark.parametrize("beta_g", [0.0, -0.2])
def test_a_running_cavity_kicks_space_charge_over_its_own_length(beta_g):
    """A βg≤0 cavity's length is known once the beam arrives; its space-charge kicks must span that length, not the
    provisional over-estimate the slices used to be counted from."""
    c = _cavity(beta_g=beta_g)
    provisional = c.length
    lat = Lattice()
    lat.add(c)
    beam = Beam(_ref(), n_particles=20, current=1.0)
    beam.particles[:] = np.random.default_rng(1).normal(size=(20, 6)) * [0.5, 0.5, 0.5, 0.5, 3.0, 0.01]
    kicks = _Kicks()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        Tracker(lat, beam, pic_solver=kicks).run()
    assert c.length < 0.5 * provisional
    assert sum(kicks.ds) == pytest.approx(c.length, rel=1e-9)


def test_the_backtracker_slices_a_running_cavity_from_its_entrance(monkeypatch):
    """The backtracker counts a field map's slices after its replay of the line has built the cavity's cells for the
    entering beam, whatever length another run left on it."""
    import linac_gen.tracking.backtrack as bt
    c = _cavity(beta_g=0.0)
    lat = Lattice()
    lat.add(Drift("D0", 50.0))
    lat.add(c)
    lat.add(Drift("D1", 50.0))
    beam = Beam(_ref(), n_particles=20, current=0.0)
    beam.particles[:] = np.random.default_rng(2).normal(size=(20, 6)) * [0.5, 0.5, 0.5, 0.5, 3.0, 0.01]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        Tracker(lat, beam).run()
    true_length = c.length
    c.reset_run_state()                     # another run's reset leaves the provisional length
    seen = []
    counts = bt.field_map_step_counts

    def _recording(lattice, element):
        seen.append(element.length)
        return counts(lattice, element)

    monkeypatch.setattr(bt, "field_map_step_counts", _recording)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        bt.backtrack_distribution(lat, beam, _ref())
    assert seen and seen[0] == pytest.approx(true_length, rel=1e-12)
