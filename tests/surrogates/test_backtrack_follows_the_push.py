"""The backtracker undoes a push through surrogate weights exactly, as the forward run made it.

The multiparticle fast path (``SurrogateFieldMap.track_rk4``) pushes the particles with a matrix
slice per call, cut from the map the weights give at the element's entrance: a linear map with an
exact inverse.  The forward run records on the beam which calls the weights made and that map
(``surrogate_pushes``, per element of the lattice it went through; no call for a traversal the
native element made alone), and the backtracker undoes each such call with the inverse of its slice
and every other call with the native element's exact inverse, whatever the registry says when it
runs: a run with the fast path off, where every push is native, is undone natively throughout,
whatever weights are registered.  The record also holds the traversal's step schedule, the
reference at the element's entrance and the settings its push depends on (its structure among
them), and the walk takes the traversal from it, the weights' calls and the native element's alike,
only while all three are the same in the walk.  A beam with no record of the element in the lattice
walked (read from a file, tracked through another lattice or a copy, an element that appears
twice), or whose record those no longer match, is undone as the weights would push it now, with a
caveat unless the native element pushes it so and, as far as the walk can tell, pushed it in the
run.
"""
import copy
import warnings
from pathlib import Path

import numpy as np
import pytest

from linac_gen.core.step_config import StepConfig
from linac_gen.elements.drift import Drift
from linac_gen.elements.lattice_commands import SetBeamPhaseError
from linac_gen.elements.marker import Marker
from linac_gen.elements.quadrupole import Quadrupole
from linac_gen.surrogates import registry
from linac_gen.tracking.backtrack import backtrack_distribution, build_replay_table
from linac_gen.tracking.tracker import Tracker
from tests.surrogates.accuracy_helpers import W, exact, native, off, pinned
from tests.surrogates.test_frequency_jump import (
    F_NEW,
    F_OLD,
    _beam,
    _cavity,
    _cavity_1d,
    _lattice,
    _magnet,
    _ref,
)

MAPS = {
    "3-D cavity": lambda: _cavity(p_flag=1),
    "3-D magnet": _magnet,
    "1-D cavity": lambda: _cavity_1d(p_flag=1),
}
#: an energy range that holds the cavity's whole gain (the weights push every call), and one
#: only its entrance energy is in (they push its first call, the native element the rest)
WHOLE, ENTRANCE = (W - 0.1, W + 0.5), (W - 1e-9, W + 1e-9)
CLOSE = 1e-10


@pytest.fixture(autouse=True)
def _flags():
    mp, fast = registry.is_mp_enabled(), registry.is_fast_path_enabled()
    registry.clear()
    yield
    registry.set_mp_enabled(mp)
    registry.set_fast_path_enabled(fast)
    registry.clear()


def _forward(name, *, fast, copy_it=False, w=WHOLE, beam_f=F_NEW, edit=None):
    """A line with one map, its pinned weights registered: the native forward, and the forward
    with multiparticle surrogates on (the fast path as given)."""
    el = MAPS[name]()
    lat = _lattice(Drift("D0", 50.0), el, Drift("D1", 50.0))
    surr = pinned(el, native(el), w=w)
    registry.register(surr)
    reference = _beam(_ref(beam_f, W))
    Tracker(copy.deepcopy(lat), reference).run()        # multiparticle surrogates off: native
    registry.set_mp_enabled(True)
    registry.set_fast_path_enabled(fast)
    run = copy.deepcopy(lat) if copy_it else lat
    if edit is not None:
        edit(run.elements[1])
    beam = _beam(_ref(beam_f, W))
    start = beam.particles.copy()
    Tracker(run, beam).run()
    return run, beam, start, reference, surr


def _calls(beam, run, k=1):
    """The calls the weights made of element ``k`` in the beam's record."""
    assert beam.surrogate_pushes_lattice == id(run)
    name, rec = beam.surrogate_pushes[id(run.elements[k])]
    assert name == run.elements[k].name
    return rec["calls"]


def _back(run, beam, f=F_NEW, w=W):
    with warnings.catch_warnings(record=True) as seen:
        warnings.simplefilter("always")
        backtrack_distribution(run, beam, _ref(f, w))
    return [str(m.message) for m in seen]


@pytest.mark.parametrize("copy_it", [False, True], ids=["registered", "copy"])
@pytest.mark.parametrize("fast", [False, True], ids=["fast-off", "fast-on"])
@pytest.mark.parametrize("name", list(MAPS))
def test_every_push_is_undone_exactly(name, fast, copy_it):
    run, beam, start, reference, surr = _forward(name, fast=fast, copy_it=copy_it)
    pushed = not np.array_equal(beam.particles, reference.particles)
    # the fast path pushes a 3-D map whose weights answer, every call; the native element pushes
    # a 1-D map (the fast path samples 3-D channel interpolators), and every map with it off
    assert pushed == (fast and name != "1-D cavity")
    calls = _calls(beam, run)
    assert (calls == list(range(len(calls))) and len(calls) > 0) == pushed
    asked = surr.nn_calls
    said = _back(run, beam)
    assert not [m for m in said if "surrogate" in m]
    assert surr.nn_calls == asked                       # the backtracker asks the weights nothing
    np.testing.assert_allclose(beam.particles, start, rtol=0.0, atol=CLOSE)


def test_a_push_that_leaves_the_range_partway_is_undone_call_by_call():
    """Weights whose range holds only the entrance energy push the cavity's first call and the
    native element the others: each call is undone as it was made."""
    run, beam, start, _reference, _surr = _forward("3-D cavity", fast=True, w=ENTRANCE)
    assert _calls(beam, run) == [0]
    said = _back(run, beam)
    assert not [m for m in said if "surrogate" in m]
    np.testing.assert_allclose(beam.particles, start, rtol=0.0, atol=CLOSE)


@pytest.mark.parametrize("fast_forward", [True, False], ids=["run-on-back-off", "run-off-back-on"])
def test_the_forward_record_decides_not_the_settings_at_backtrack_time(fast_forward):
    run, beam, start, _reference, _surr = _forward("3-D cavity", fast=fast_forward)
    registry.set_fast_path_enabled(not fast_forward)
    said = _back(run, beam)
    assert not [m for m in said if "surrogate" in m]
    np.testing.assert_allclose(beam.particles, start, rtol=0.0, atol=CLOSE)


def test_a_later_native_pass_clears_the_record_of_an_earlier_one():
    """The record keeps a beam's latest pass over each place: a pass the native element made
    after one through the weights is undone natively, from the entrance of that pass."""
    run, beam, _start, _reference, _surr = _forward("3-D magnet", fast=True)
    assert _calls(beam, run)
    registry.set_mp_enabled(False)                      # the same beam, once more, natively
    after_first, second = beam.particles.copy(), beam.ref.copy()
    Tracker(run, beam).run()
    assert _calls(beam, run) == []
    registry.set_mp_enabled(True)
    registry.set_fast_path_enabled(True)                # what the weights would do now: not asked
    with warnings.catch_warnings(record=True) as seen:
        warnings.simplefilter("always")
        backtrack_distribution(run, beam, second)
    assert not [m for m in seen if "surrogate" in str(m.message)]
    np.testing.assert_allclose(beam.particles, after_first, rtol=0.0, atol=CLOSE)


@pytest.mark.parametrize("w", [WHOLE, ENTRANCE], ids=["whole", "partway"])
def test_a_beam_with_no_record_is_undone_as_the_weights_push_it_now(w):
    run, beam, start, _reference, _surr = _forward("3-D cavity", fast=True, w=w)
    del beam.surrogate_pushes                           # as a beam read from a file
    said = _back(run, beam)
    assert [m for m in said if "no record of its forward run" in m]
    np.testing.assert_allclose(beam.particles, start, rtol=0.0, atol=CLOSE)


@pytest.mark.parametrize("case", ["frequency jump", "out of range", "setting changed", "1-D map"])
def test_a_beam_with_no_record_where_the_weights_leave_the_push_native(case):
    """At the entrance the fast path leaves the element to the native one (a clock to jump, an
    input out of range, a setting the weights hold changed, a 1-D map): asked again for a beam
    with no record, the weights say so too, and the walk is exact with no caveat."""
    name, kw, back_f = "3-D cavity", {}, F_NEW
    if case == "frequency jump":
        kw["beam_f"] = back_f = F_OLD
    elif case == "out of range":
        kw["w"] = (W + 1.0, W + 2.0)
    elif case == "setting changed":
        name, kw["edit"] = "3-D magnet", (lambda el: setattr(el, "kb", 1.3))
    else:
        name = "1-D cavity"
    run, beam, start, _reference, _surr = _forward(name, fast=True, **kw)
    assert _calls(beam, run) == []
    del beam.surrogate_pushes
    said = _back(run, beam, f=back_f)
    assert not [m for m in said if "surrogate" in m]
    np.testing.assert_allclose(beam.particles, start, rtol=0.0, atol=CLOSE)


def test_the_record_holds_the_map_the_weights_pushed_with():
    run, beam, _start, _reference, surr = _forward("3-D magnet", fast=True)
    rec = beam.surrogate_pushes[id(run.elements[1])][1]
    np.testing.assert_allclose(rec["M_full"], surr.fitted_matrix(_ref(F_NEW, W)),
                               rtol=1e-14, atol=1e-14)
    assert rec["length"] == 200.0 and rec["calls"] == list(range(len(rec["calls"])))


@pytest.mark.parametrize("name, can_push", [("3-D cavity", True), ("3-D magnet", True),
                                            ("1-D cavity", False)])
def test_drives_mp_says_what_the_fast_path_pushes(name, can_push):
    """At an entrance where the weights answer: True only with multiparticle surrogates and the
    fast path on, and only for a map the fast path can push."""
    el = MAPS[name]()
    surr = pinned(el, native(el))
    ref = _ref(F_NEW, W)
    for mp, fast in ((False, False), (True, False), (False, True)):
        registry.set_mp_enabled(mp)
        registry.set_fast_path_enabled(fast)
        assert surr.drives_mp(ref) is False
    registry.set_mp_enabled(True)
    registry.set_fast_path_enabled(True)
    assert surr.drives_mp(ref) is can_push
    surr.fitted_matrix(ref)                             # the weights do answer there


def test_an_edit_that_leaves_the_element_and_its_entrance_unchanged_keeps_the_record():
    """The record goes with each element, not its place: a marker inserted upstream after the run
    (an edit through the Lattice tab keeps the results) changes neither the element's settings,
    its step schedule nor the reference at its entrance, and the push is undone exactly from the
    record."""
    run, beam, start, _reference, _surr = _forward("3-D cavity", fast=True)
    run.elements.insert(0, Marker("M_NEW"))
    said = _back(run, beam)
    assert not [m for m in said if "surrogate" in m]
    np.testing.assert_allclose(beam.particles, start, rtol=0.0, atol=CLOSE)


def _exact(el):
    """Weights whose network is the native map at each of their inputs: energy, phase, ke."""
    return exact(el, w=(4.0, 6.0), params=(("phase", -40.0, -20.0), ("ke", 0.5, 1.5)))


def _run_then_edit(edit, *, make=lambda: _cavity(p_flag=1), weights=_exact, step_config=None):
    """A line with a native cavity upstream, switched off (``UP``, no weights), and the map, its
    weights registered (the lattice's step settings ``step_config``, where given): the
    multiparticle run with the fast path on, the weights pushing the map, then
    ``edit(lattice)``.  Returns the lattice, the beam at the exit, its particles and the
    reference energy there."""
    up = _cavity(p_flag=1)
    up.name, up.ke = "UP", 0.0
    el = make()
    lat = _lattice(Drift("D0", 50.0), up, Drift("D1", 50.0), el, Drift("D2", 50.0))
    if step_config is not None:
        lat.step_config = step_config
    registry.register(weights(el))
    registry.set_mp_enabled(True)
    registry.set_fast_path_enabled(True)
    beam = _beam(_ref(F_NEW, W))
    Tracker(lat, beam).run()
    assert _calls(beam, lat, 3)
    exit_p, exit_w = beam.particles.copy(), beam.ref.w_kin
    edit(lat)
    return lat, beam, exit_p, exit_w


def _miss(lat, back, exit_p, exit_w, w=W, f=F_NEW):
    """How far the reconstruction ``back``, tracked forward again through the lattice as it
    stands, lands from the run's exit ``exit_p``: the largest difference per coordinate, the
    energy taken absolute (the walk keeps each particle's energy, on the exit reference of the
    lattice it walks)."""
    again = _beam(_ref(f, w))
    again.particles[:] = back
    Tracker(lat, again).run()
    got = again.particles.copy()
    got[:, 5] += again.ref.w_kin - exit_w
    return np.abs(got - exit_p).max(axis=0)


@pytest.mark.parametrize("edit, schedule, also", [
    ("n_steps", "40 half-steps of 5 mm then, 120 of 1.667 mm now", " and its settings (n_steps)"),
    ("step density", "40 half-steps of 5 mm then, 240 of 0.8333 mm now", ""),
    ("longer", "40 half-steps of 5 mm then, 42 of 5 mm now",          # more steps, as long
     " and its settings (length)"),
    ("shorter", "40 half-steps of 5 mm then, 40 of 4.975 mm now",     # as many, shorter
     " and its settings (length)"),
])
def test_a_changed_step_schedule_is_not_taken_from_the_record(edit, schedule, also):
    """The record holds the traversal's step schedule.  After a change of the element's steps
    (its own count, the lattice's step density, its length) the walk would cut the run's slices
    at the new steps and undo the rest natively; it undoes the element as the weights push it
    with the current settings, says why (the element's own step count and length are settings
    of its structure too), and inverts the lattice as it stands."""
    run, beam, _start, _reference, _surr = _forward("3-D cavity", fast=True)
    exit_p, exit_w = beam.particles.copy(), beam.ref.w_kin
    el = run.elements[1]
    if edit == "n_steps":
        el.n_steps = 60
    elif edit == "step density":
        run.step_config = StepConfig(integration_steps_per_metre=600.0)
    else:
        el.length = 210.0 if edit == "longer" else 199.0
    said = _back(run, beam)
    assert [m for m in said if f"'MAP': its step schedule ({schedule}){also} changed since the "
            "forward run, so the beam's record of that run does not cover it; undone " in m]
    assert _miss(run, beam.particles, exit_p, exit_w).max() < CLOSE


@pytest.mark.parametrize("name, setting, value", [
    ("3-D cavity", "phase", -25.0),             # an input of the weights
    ("3-D cavity", "ke", 1.0 + 1e-10),          # an input, by less than SETTING_TOLERANCE
    ("3-D cavity", "voltage_rel", 0.02),        # an error knob folded into their ke input
    ("3-D cavity", "phase_offset", 2.0),        # one folded into their phase input
    ("3-D cavity", "frequency_offset", 0.01),   # one they refuse: frequency is no input of theirs
    ("3-D cavity", "p_flag", 0),                # the phase reference: the weights refuse
    ("3-D magnet", "kb", 1.3),                  # a setting the weights hold: they refuse
])
def test_an_edit_of_the_pushed_element_is_not_taken_from_the_record(name, setting, value):
    """The record holds the settings the element's push depends on: after an edit of the element
    itself (what Matching -> Apply or the inspector does) the record's map is not the element's;
    the walk undoes it as the weights push it with the current settings, or natively where they
    refuse now, and says which settings changed."""
    make = {"3-D cavity": lambda: _cavity(p_flag=1), "3-D magnet": _magnet}[name]
    weights = (lambda el: pinned(el, native(el), w=WHOLE)) if setting == "kb" else _exact
    lat, beam, exit_p, exit_w = _run_then_edit(
        lambda lat: setattr(lat.elements[3], setting, value), make=make, weights=weights)
    said = _back(lat, beam)
    assert [m for m in said if f"'MAP': its settings ({setting}) changed since the forward run"
            in m]
    assert _miss(lat, beam.particles, exit_p, exit_w).max() < CLOSE


@pytest.mark.parametrize("field, value", [
    ("n_steps", 10),                    # under the lattice's step density: the same schedule
    ("aperture", 25.0),                 # an inspector field, as the step count is
    ("field_file", "another.edz"),      # what the card names, which no tab edits
    ("geom", 7),                        # the card's geometry code
    ("ki", 0.5),                        # its space-charge compensation factor
    ("ka", 0),                          # its aperture-shape flag
])
def test_an_edit_of_the_pushed_element_s_structure_is_not_taken_from_the_record(field, value):
    """The record holds the element's structure among its settings (what the registry
    recognises it by): weights refuse an element whose structure is no longer the one they were
    built on, so after an edit of it a forward run has the native element push it, also where
    its step schedule stays as it was (a step count under the lattice's step density, which
    still sets SET_SYNC_PHASE's calibration).  The walk undoes it natively, says what changed,
    and inverts the lattice as it stands."""
    lat, beam, exit_p, exit_w = _run_then_edit(
        lambda lat: setattr(lat.elements[3], field, value),
        step_config=StepConfig(integration_steps_per_metre=200.0))
    said = _back(lat, beam)
    assert [m for m in said if f"'MAP': its settings ({field}) changed since the forward run, so "
            "the beam's record of that run does not cover it; undone natively, as the surrogate "
            "weights would leave it with the current settings." in m]
    assert _miss(lat, beam.particles, exit_p, exit_w).max() < CLOSE


@pytest.mark.parametrize("edit, moved", [
    ("an upstream cavity switched on", "energy, phase"),
    ("an upstream drift lengthened", "phase"),
    ("an RF phase shift inserted upstream", "RF phase shift"),
    ("an RF phase correction inserted upstream", "RF phase correction"),
])
def test_an_upstream_edit_that_moves_the_entrance_is_not_taken_from_the_record(edit, moved):
    """The record holds the reference at the element's entrance, which an edit upstream moves:
    the weights' map there is another, and the walk's replay of the element is another walk.
    The walk undoes the element as the weights push it now, and says so."""
    def change(lat):
        if edit == "an upstream cavity switched on":
            lat.elements[1].ke = 1.0
        elif edit == "an upstream drift lengthened":
            lat.elements[2].length = 51.0
        elif edit == "an RF phase shift inserted upstream":
            lat.elements.insert(2, SetBeamPhaseError("SHIFT", 2.0))
        else:
            # a Dp = 0 card with an error study's correction stamped on it
            card = SetBeamPhaseError("RESTART", 0.0)
            card.rephase_deg, card.rephased = 2.0, 1
            lat.elements.insert(2, card)
    lat, beam, exit_p, exit_w = _run_then_edit(change)
    said = _back(lat, beam)
    assert [m for m in said if f"'MAP': its entrance reference ({moved}) changed since the "
            "forward run" in m]
    assert _miss(lat, beam.particles, exit_p, exit_w).max() < CLOSE


def test_a_changed_beam_entrance_energy_is_not_taken_from_the_record():
    """The walk starts from the entrance reference it is given (the GUI builds it from the beam
    settings when the backtrack starts): an energy changed since the run moves the reference at
    the element's entrance too."""
    lat, beam, exit_p, exit_w = _run_then_edit(lambda lat: None)
    said = _back(lat, beam, w=W + 0.01)
    assert [m for m in said if "'MAP': its entrance reference (energy, phase) changed since the "
            "forward run" in m]
    assert _miss(lat, beam.particles, exit_p, exit_w, w=W + 0.01).max() < CLOSE


def test_a_changed_beam_entrance_frequency_is_not_taken_from_the_record():
    """The frequency and the clock at the element's entrance are of the reference the record
    holds: the walk of a map first in the line started at another frequency than the run (the
    phase and the energy at the map the same) has the map jump to its own clock on entry, which
    the weights leave to the native element.  The walk undoes it natively, says so, and inverts
    the lattice as it stands."""
    el = _cavity(p_flag=1)
    lat = _lattice(el, Drift("D1", 50.0))
    registry.register(pinned(el, native(el), w=WHOLE))
    registry.set_mp_enabled(True)
    registry.set_fast_path_enabled(True)
    beam = _beam(_ref(F_NEW, W))
    Tracker(lat, beam).run()
    assert _calls(beam, lat, 0)
    exit_p, exit_w = beam.particles.copy(), beam.ref.w_kin
    said = _back(lat, beam, f=F_OLD)
    assert [m for m in said if "'MAP': its entrance reference (frequency, clock) changed since "
            "the forward run, so the beam's record of that run does not cover it; undone natively"
            in m]
    assert _miss(lat, beam.particles, exit_p, exit_w, f=F_OLD).max() < CLOSE


@pytest.mark.parametrize("card", ["RF phase shift", "RF phase correction"])
def test_a_run_through_an_rf_shift_is_taken_from_the_record(card):
    """A SET_BEAM_PHASE_ERROR card upstream of the map in the run, and nothing edited: the
    record's entrance holds the RF shift and its correction as the replay table does, and the
    push is undone from the record, exactly and with no caveat."""
    up = _cavity(p_flag=1)
    up.name, up.ke = "UP", 0.0
    if card == "RF phase shift":
        # a map that follows the running phase: the weights fold the shift into their phase
        # input (at p_flag 1 they refuse it), over a phase range that holds it
        el = _cavity(p_flag=0)
        weights = exact(el, w=(4.0, 6.0), params=(("phase", -179.0, 179.0), ("ke", 0.5, 1.5)))
        shift = SetBeamPhaseError("SHIFT", 2.0)
        held = "rf_shift"
    else:
        el = _cavity(p_flag=1)
        weights = _exact(el)
        shift = SetBeamPhaseError("RESTART", 0.0)
        shift.rephase_deg, shift.rephased = 2.0, 1
        held = "rf_rephase"
    lat = _lattice(Drift("D0", 50.0), up, shift, Drift("D1", 50.0), el, Drift("D2", 50.0))
    registry.register(weights)
    registry.set_mp_enabled(True)
    registry.set_fast_path_enabled(True)
    beam = _beam(_ref(F_NEW, W))
    start = beam.particles.copy()
    Tracker(lat, beam).run()
    assert _calls(beam, lat, 4)
    assert beam.surrogate_pushes[id(el)][1]["entry"][held] == 2.0
    said = _back(lat, beam)
    assert not [m for m in said if "surrogate" in m]
    np.testing.assert_allclose(beam.particles, start, rtol=0.0, atol=CLOSE)


@pytest.mark.parametrize("case", ["copied", "equal", "edited"])
def test_a_structure_value_that_is_not_a_scalar_is_compared_by_content(case):
    """The GUI's Simulate -> Backtrack walks a deep copy of the results' beam, and an element
    built in code may hold a structure value that is not a scalar (a field file given as a
    pathlib.Path; the parser stores a str).  The record compares it by content: the same in the
    copy, or held in a new, equal object, so the push is undone from the record with no caveat;
    another file, so it is not."""
    el = _cavity(p_flag=1)
    el.field_file = Path("cav.edz")
    lat = _lattice(Drift("D0", 50.0), el, Drift("D1", 50.0))
    registry.register(pinned(el, native(el), w=WHOLE))
    registry.set_mp_enabled(True)
    registry.set_fast_path_enabled(True)
    beam = _beam(_ref(F_NEW, W))
    start = beam.particles.copy()
    Tracker(lat, beam).run()
    assert _calls(beam, lat)
    if case == "edited":
        el.field_file = Path("another.edz")
        said = _back(lat, beam)
        assert [m for m in said if "'MAP': its settings (field_file) changed since the forward "
                "run" in m]
    else:
        if case == "equal":
            el.field_file = Path("cav.edz")             # another object, equal to the run's
        beam = copy.deepcopy(beam)
        said = _back(lat, beam)
        assert not [m for m in said if "surrogate" in m]
        np.testing.assert_allclose(beam.particles, start, rtol=0.0, atol=CLOSE)


def test_a_record_whose_entrance_the_walk_reaches_by_another_sum_is_still_taken():
    """The run steps a magnet in quarters and, with sub-step rows recorded, a drift in parts,
    where the replay table takes each whole: the phase at the map's entrance differs from the
    table's in its last bits, which is no change of the lattice, and the push is undone exactly
    from the record."""
    el = _cavity(p_flag=1)
    lat = _lattice(Drift("D0", 50.3), Quadrupole("Q", length=73.1, gradient=2.0),
                   Drift("D1", 41.7), el, Drift("D2", 50.0))
    registry.register(pinned(el, native(el), w=WHOLE))
    registry.set_mp_enabled(True)
    registry.set_fast_path_enabled(True)
    beam = _beam(_ref(F_NEW, W))
    start = beam.particles.copy()
    Tracker(lat, beam, record_substeps=True).run()
    assert _calls(beam, lat, 3)
    then = beam.surrogate_pushes[id(el)][1]["entry"]["phi_s"]
    now = build_replay_table(lat, _ref(F_NEW, W), end=len(lat.elements) - 1)[3].phi_s
    assert then != now and abs(then - now) < 1e-14 * abs(now)
    said = _back(lat, beam)
    assert not [m for m in said if "surrogate" in m]
    np.testing.assert_allclose(beam.particles, start, rtol=0.0, atol=CLOSE)


def test_an_entrance_phase_near_zero_reached_by_another_sum_is_still_taken():
    """Where the phase at the map's entrance is zero in the replay table, the run's sum over the
    drift before it in parts (sub-step rows recorded) lands a round-off away from zero: within
    1e-11 of 1, the scale of that round-off, it is the same reference, and the push is undone
    exactly from the record."""
    el = _cavity(p_flag=1)
    lat = _lattice(Drift("D0", 50.0), el, Drift("D1", 50.0))
    registry.register(pinned(el, native(el), w=WHOLE))
    registry.set_mp_enabled(True)
    registry.set_fast_path_enabled(True)
    ref = _ref(F_NEW, W)
    ref.phi_s = -360.0 * 50.0 / (ref.beta * ref.wavelength)     # the table's: 0 at the map
    beam = _beam(ref)
    start = beam.particles.copy()
    Tracker(lat, beam, record_substeps=True).run()
    assert _calls(beam, lat)
    then = beam.surrogate_pushes[id(el)][1]["entry"]["phi_s"]
    now = build_replay_table(lat, ref, end=len(lat.elements) - 1)[1].phi_s
    assert now == 0.0 and then != 0.0 and abs(then) < 1e-13
    with warnings.catch_warnings(record=True) as seen:
        warnings.simplefilter("always")
        backtrack_distribution(lat, beam, ref)
    assert not [m for m in seen if "surrogate" in str(m.message)]
    np.testing.assert_allclose(beam.particles, start, rtol=0.0, atol=CLOSE)


@pytest.mark.parametrize("case, edit, changed", [
    ("the fast path off", "phase", "settings (phase)"),
    ("out of the weights' range", "phase", "settings (phase)"),
    ("multiparticle surrogates off", "phase", "settings (phase)"),
    ("the fast path off", "an upstream drift lengthened", "entrance reference (phase)"),
    ("the fast path off", "step density",
     "step schedule (40 half-steps of 5 mm then, 240 of 0.8333 mm now)"),
], ids=["the fast path off", "out of the weights' range", "multiparticle surrogates off",
        "the fast path off, an upstream drift lengthened", "the fast path off, step density"])
def test_an_edit_of_an_element_the_native_element_pushed_is_not_taken_from_the_record(
        case, edit, changed):
    """The record of a traversal the native element made alone (the fast path off in the run, the
    map's phase out of its weights' range, multiparticle surrogates off) holds what the record of
    one through the weights holds, and the walk checks it the same way: after an edit that has
    the weights push the element now (of its settings, of the reference at its entrance or of
    its step schedule), it undoes the element as they push it with the current settings, says
    what changed, and inverts the lattice as it stands."""
    up = _cavity(p_flag=1)
    up.name, up.ke = "UP", 0.0
    el = _cavity(p_flag=1)
    if case == "out of the weights' range":
        el.phase = -10.0                                # their phase range is (-40, -20)
    lat = _lattice(Drift("D0", 50.0), up, Drift("D1", 50.0), el, Drift("D2", 50.0))
    surr = _exact(el)
    registry.register(surr)
    registry.set_mp_enabled(case != "multiparticle surrogates off")
    registry.set_fast_path_enabled(case != "the fast path off")
    beam = _beam(_ref(F_NEW, W))
    Tracker(lat, beam).run()
    assert surr.nn_calls == 0                           # the native element pushed it
    exit_p, exit_w = beam.particles.copy(), beam.ref.w_kin
    if edit == "phase":
        el.phase = -27.0
    elif edit == "an upstream drift lengthened":
        lat.elements[2].length = 51.0
    else:
        lat.step_config = StepConfig(integration_steps_per_metre=600.0)
    registry.set_mp_enabled(True)
    registry.set_fast_path_enabled(True)
    said = _back(lat, beam)
    assert [m for m in said if f"'MAP': its {changed} changed since the forward run, so the "
            "beam's record of that run does not cover it; undone as the surrogate weights push it "
            "with the current settings (the multiparticle fast path)." in m]
    assert _miss(lat, beam.particles, exit_p, exit_w).max() < CLOSE


@pytest.mark.parametrize("weights", [True, False], ids=["weights out of range", "no weights"])
def test_an_edit_where_the_native_element_pushes_then_and_now_gets_no_caveat(weights):
    """Where the native element pushed the element in the run and pushes it in the lattice as it
    stands (its weights refuse it then and now, or none are registered), the walk undoes it
    natively with the current settings, as it undoes every element, with no caveat."""
    up = _cavity(p_flag=1)
    up.name, up.ke = "UP", 0.0
    el = _cavity(p_flag=1)
    el.phase = -10.0                                    # out of the weights' phase range
    lat = _lattice(Drift("D0", 50.0), up, Drift("D1", 50.0), el, Drift("D2", 50.0))
    if weights:
        registry.register(_exact(el))
    registry.set_mp_enabled(True)
    registry.set_fast_path_enabled(True)
    beam = _beam(_ref(F_NEW, W))
    Tracker(lat, beam).run()
    exit_p, exit_w = beam.particles.copy(), beam.ref.w_kin
    el.phase = -5.0                                     # out of it still
    lat.elements[1].ke = 1.0                            # and the cavity upstream switched on
    said = _back(lat, beam)
    assert not [m for m in said if "surrogate" in m]
    assert _miss(lat, beam.particles, exit_p, exit_w).max() < CLOSE


def test_a_run_with_multiparticle_surrogates_off_is_undone_from_its_record():
    """A run with multiparticle surrogates off records the native element's traversals, which a
    walk with them and the fast path on takes as they stand: no caveat, and the run's entrance
    to round-off."""
    el = _cavity(p_flag=1)
    lat = _lattice(Drift("D0", 50.0), el, Drift("D1", 50.0))
    registry.register(pinned(el, native(el), w=WHOLE))
    registry.set_mp_enabled(False)
    beam = _beam(_ref(F_NEW, W))
    start = beam.particles.copy()
    Tracker(lat, beam).run()
    registry.set_mp_enabled(True)
    registry.set_fast_path_enabled(True)
    said = _back(lat, beam)
    assert not [m for m in said if "surrogate" in m]
    np.testing.assert_allclose(beam.particles, start, rtol=0.0, atol=CLOSE)


def test_a_beam_with_no_record_backtracked_with_the_fast_path_off_gets_a_caveat():
    """The weights pushed the cavity in the run (the fast path on); the beam, as read from a file,
    carries no record of it, and the fast path is off when it is backtracked.  The walk cannot
    tell whether the run had it on: it undoes the cavity natively and says so.  (The weights are
    2 % off the native map in one entry, as trained weights are off by something.)"""
    el = _cavity(p_flag=1)
    lat = _lattice(Drift("D0", 50.0), el, Drift("D1", 50.0))
    registry.register(pinned(el, off(native(el), 0, 0, 1.02), w=WHOLE))
    registry.set_mp_enabled(True)
    registry.set_fast_path_enabled(True)
    beam = _beam(_ref(F_NEW, W))
    start = beam.particles.copy()
    Tracker(lat, beam).run()
    assert _calls(beam, lat)
    del beam.surrogate_pushes
    registry.set_fast_path_enabled(False)
    said = _back(lat, beam)
    assert [m for m in said if "'MAP': the beam carries no record of its forward run; undone "
            "natively, as the multiparticle fast path is off, though a run with it on pushes it "
            "through the surrogate weights." in m]
    assert np.abs(beam.particles - start).max() > 1e-3        # what the caveat warns of


@pytest.mark.parametrize("case", ["1-D map", "out of range"])
def test_a_beam_with_no_record_the_weights_would_not_push_gets_no_caveat_with_the_fast_path_off(
        case):
    """Where the weights leave the element to the native one with the fast path on too (a 1-D
    map, an entrance out of their range), the native element pushed it whatever the run's fast
    path: the walk undoes it exactly, with no caveat."""
    name, kw = ("1-D cavity", {}) if case == "1-D map" else ("3-D cavity", {"w": (W + 1.0, W + 2.0)})
    run, beam, start, _reference, _surr = _forward(name, fast=True, **kw)
    assert _calls(beam, run) == []
    del beam.surrogate_pushes
    registry.set_fast_path_enabled(False)
    said = _back(run, beam)
    assert not [m for m in said if "surrogate" in m]
    np.testing.assert_allclose(beam.particles, start, rtol=0.0, atol=CLOSE)


def test_tracked_element_by_element_through_the_weights():
    el = MAPS["3-D cavity"]()
    lat = _lattice(Drift("D0", 50.0), el, Drift("D1", 50.0))
    registry.register(pinned(el, native(el), w=WHOLE))
    registry.set_mp_enabled(True)
    registry.set_fast_path_enabled(True)
    beam = _beam(_ref(F_NEW, W))
    start = beam.particles.copy()
    t = Tracker(lat, beam)
    for e in lat.elements:
        t._track_element(e)
    assert _calls(beam, lat)
    said = _back(lat, beam)
    assert not [m for m in said if "surrogate" in m]
    np.testing.assert_allclose(beam.particles, start, rtol=0.0, atol=CLOSE)


@pytest.mark.parametrize("case", ["a copy of the lattice", "one element at two places",
                                  "another lattice holding the same elements"])
def test_where_the_record_does_not_cover_the_element_the_current_settings_decide_with_a_caveat(case):
    """The walk cannot tell which traversal of the record an element is, or has no record of this
    lattice: it undoes each push as the weights would make it now, and says so.  A record is
    stamped with the lattice it was made on, so a walk of another lattice object does not trust
    it even where the element objects are the same ones (an id Python reuses after a lattice is
    freed could otherwise match another element)."""
    if case == "one element at two places":
        el = MAPS["3-D magnet"]()
        lat = _lattice(Drift("D0", 50.0), el, Drift("D1", 50.0), el, Drift("D2", 50.0))
        registry.register(pinned(el, native(el), w=WHOLE))
        registry.set_mp_enabled(True)
        registry.set_fast_path_enabled(True)
        beam = _beam(_ref(F_NEW, W))
        start = beam.particles.copy()
        Tracker(lat, beam).run()
        walked = lat
    else:
        run, beam, start, _reference, _surr = _forward("3-D magnet", fast=True)
        walked = (copy.deepcopy(run) if case == "a copy of the lattice"
                  else _lattice(*run.elements))
    said = _back(walked, beam)
    assert [m for m in said if "does not cover this element" in m]
    np.testing.assert_allclose(beam.particles, start, rtol=0.0, atol=CLOSE)


def test_an_element_renamed_since_the_run_is_not_taken_from_the_record():
    """The record names the element it was made for: an element renamed since (or another object
    at a reused id) is not undone from it.  No weights are registered under the new name, so it is
    undone natively, and the caveat says the record does not cover it."""
    run, beam, _start, _reference, _surr = _forward("3-D magnet", fast=True)
    assert _calls(beam, run)
    run.elements[1].name = "RENAMED"
    said = _back(run, beam)
    assert [m for m in said if "'RENAMED': the beam's record of its forward run does not cover this "
            "element in this lattice; undone natively" in m]


@pytest.mark.parametrize("mp", [False, True],
                         ids=["multiparticle surrogates off", "multiparticle surrogates on"])
def test_an_element_renamed_since_a_run_without_weights_gets_no_caveat(mp):
    """No weights are registered, so the native element pushed the map in the run and pushes it
    now: renamed since, the map is not undone from the record, but natively with the current
    settings, as every element is, with no caveat."""
    el = _cavity(p_flag=1)
    lat = _lattice(Drift("D0", 50.0), el, Drift("D1", 50.0))
    registry.set_mp_enabled(mp)
    registry.set_fast_path_enabled(True)
    beam = _beam(_ref(F_NEW, W))
    start = beam.particles.copy()
    Tracker(lat, beam).run()
    el.name = "RENAMED"
    said = _back(lat, beam)
    assert not [m for m in said if "surrogate" in m]
    np.testing.assert_allclose(beam.particles, start, rtol=0.0, atol=CLOSE)


@pytest.mark.parametrize("case", ["a copy of the lattice", "one element at two places"])
def test_a_caveat_where_weights_pushed_it_in_the_run_and_none_stand_for_it_now(case):
    """The weights (2 % off the native map) pushed the magnet in the run, and multiparticle
    surrogates are off when it is backtracked, so no weights stand for it now.  The record does
    not cover the element in the lattice walked, but holds the weights' calls under its name:
    the native undo may not be the push the run made, and the walk says so."""
    el = MAPS["3-D magnet"]()
    if case == "one element at two places":
        lat = _lattice(Drift("D0", 50.0), el, Drift("D1", 50.0), el, Drift("D2", 50.0))
    else:
        lat = _lattice(Drift("D0", 50.0), el, Drift("D1", 50.0))
    registry.register(pinned(el, off(native(el), 0, 0, 1.02), w=WHOLE))
    registry.set_mp_enabled(True)
    registry.set_fast_path_enabled(True)
    beam = _beam(_ref(F_NEW, W))
    Tracker(lat, beam).run()
    assert any(r["calls"] for _name, r in beam.surrogate_pushes.values())
    registry.set_mp_enabled(False)
    said = _back(lat if case == "one element at two places" else copy.deepcopy(lat), beam)
    assert [m for m in said if "'MAP': the beam's record of its forward run does not cover this "
            "element in this lattice; undone natively, as the surrogate weights would leave it "
            "with the current settings." in m]


def test_another_lattice_starts_the_record_over():
    run, beam, _start, _reference, _surr = _forward("3-D magnet", fast=True)
    assert _calls(beam, run)
    other = _lattice(Drift("X0", 10.0), MAPS["1-D cavity"](), Drift("X1", 10.0))
    Tracker(other, beam).run()
    assert beam.surrogate_pushes_lattice == id(other)
    assert list(beam.surrogate_pushes) == [id(other.elements[1])]
