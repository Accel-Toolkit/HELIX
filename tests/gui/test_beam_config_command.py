"""``BeamConfigCommand``: a change of the session beam as a step of the command bus.

A Matching Apply writes the matched beam with the matched element values; one Undo must put
back both.  The beam is otherwise set outside the bus (the Beam tab's Apply, a run's start, a
project), so undo and redo set back only the fields that still hold what the command left in
them, and never raise.
"""
from __future__ import annotations

import dataclasses

import pytest

pytest.importorskip("PyQt6")


@pytest.fixture
def session(qapp, mini_lattice):
    from linac_gen_gui.interphase.state import AppState
    from linac_gen_gui.interphase.tabs.beam_tab import BeamTab
    st = AppState()
    st.set_lattice(mini_lattice, None)
    form = BeamTab(st)
    form._npart.setValue(1000)
    form._apply(quiet=True)
    messages: list[str] = []
    st.status_message.connect(messages.append)
    yield st, form, messages
    form.deleteLater()


def _command(st, form, **values):
    from linac_gen_gui.interphase.commands import BeamConfigCommand
    return BeamConfigCommand(st, values, show=form.show_beam_fields, label="Matched beam")


def test_do_sets_the_fields_in_the_session_and_the_form_and_undo_redo_follow(session):
    st, form, _messages = session
    before = st.beam_config
    cmd = _command(st, form, alpha_x=0.5, beta_x=0.9)
    st.bus.do(cmd)
    assert (st.beam_config.alpha_x, st.beam_config.beta_x) == (0.5, 0.9)
    assert (form.get_beam_config().alpha_x, form.get_beam_config().beta_x) == (0.5, 0.9)
    assert dataclasses.replace(st.beam_config, alpha_x=before.alpha_x,
                               beta_x=before.beta_x) == before   # nothing else
    assert st.project_dirty and cmd.old_values == {"alpha_x": 1.228, "beta_x": 0.316}
    assert cmd.describe() == "Matched beam: alpha_x 1.228 → 0.5, beta_x 0.316 → 0.9"
    st.mark_project_clean()
    st.bus.undo()
    assert st.beam_config == before and form.get_beam_config() == before
    assert st.project_dirty                     # the beam differs from the saved one again
    st.bus.redo()
    assert (st.beam_config.alpha_x, st.beam_config.beta_x) == (0.5, 0.9)


def test_a_field_set_since_outside_the_bus_keeps_its_value(session):
    st, form, messages = session
    before = st.beam_config
    cmd = _command(st, form, alpha_x=0.5, beta_x=0.9)
    st.bus.do(cmd)
    # the Beam tab's Apply sets beta_x (and the current) after the step
    form._beta_x.setValue(0.7)
    form._current.setValue(3.0)
    form._apply()
    st.bus.undo()
    assert st.beam_config.alpha_x == before.alpha_x        # what the step set: back
    assert (st.beam_config.beta_x, st.beam_config.current) == (0.7, 3.0)   # kept
    assert cmd.kept == ["beta_x"]
    assert messages[-1] == "Matched beam: beta_x set since, kept as it is"
    # the field is the user's now: redo leaves it alone too
    st.bus.redo()
    assert (st.beam_config.alpha_x, st.beam_config.beta_x) == (0.5, 0.7)
    assert cmd.kept == []


def test_the_form_shows_the_fields_the_command_sets_and_nothing_else(session):
    """Do, undo and redo put in the form the fields they set; a value typed in another box
    and not applied stays there, and the form is not applied (the session beam does not
    get it)."""
    st, form, _messages = session
    before = st.beam_config
    st.bus.do(_command(st, form, alpha_x=0.5, beta_x=0.9))
    form._energy.setValue(2.5)                  # typed, the Beam tab's Apply not clicked
    st.bus.undo()
    assert (form._alpha_x.value(), form._beta_x.value(), form._energy.value()) == (
        before.alpha_x, before.beta_x, 2.5)
    assert st.beam_config == before
    st.bus.redo()
    assert (form._alpha_x.value(), form._beta_x.value(), form._energy.value()) == (
        0.5, 0.9, 2.5)
    assert st.beam_config == dataclasses.replace(before, alpha_x=0.5, beta_x=0.9)


def test_a_field_the_form_has_no_box_for_is_refused_before_any_box_changes(session):
    st, form, _messages = session
    before = st.beam_config
    with pytest.raises(ValueError, match="no box for species"):
        st.bus.do(_command(st, form, alpha_x=0.5, species="proton"))
    assert form._alpha_x.value() == before.alpha_x
    assert st.beam_config is before and not st.bus.can_undo


def test_a_run_start_committing_the_same_form_is_not_an_edit(session):
    """A run commits the form to the session (a new object, the same values): the
    step's fields still hold what it left, so undo sets them back."""
    st, form, _messages = session
    before = st.beam_config
    st.bus.do(_command(st, form, alpha_x=0.5))
    st.set_beam_config(form.get_beam_config())
    st.bus.undo()
    assert st.beam_config.alpha_x == before.alpha_x


def test_undo_never_raises_and_says_when_it_cannot_build_the_beam(session):
    st, form, messages = session
    cmd = _command(st, form, beta_x=0.9)
    st.bus.do(cmd)
    cmd.old_values = {"beta_x": -1.0}           # a beam BeamConfig refuses
    assert st.bus.undo()                        # no exception out of the bus
    assert st.beam_config.beta_x == 0.9
    assert messages[-1].startswith("Matched beam: the beam was left as it is (beta_x")


def test_a_refused_first_do_sets_nothing_and_aborts_the_whole_step(session):
    from linac_gen_gui.interphase.commands import MacroCommand, ParamChangeCommand
    st, form, _messages = session
    before = st.beam_config
    q = st.lattice.elements[1]
    step = MacroCommand([ParamChangeCommand(q, "gradient", q.gradient, 12.5),
                         _command(st, form, beta_x=-1.0)], label="Matching Apply")
    with pytest.raises(ValueError, match="beta_x"):
        st.bus.do(step)
    assert q.gradient == 10.0 and st.beam_config == before and not st.bus.can_undo


def test_a_form_that_fails_leaves_the_session_beam_as_it_was(session):
    from linac_gen_gui.interphase.commands import BeamConfigCommand

    def broken(_cfg):
        raise TypeError("the form refused it")

    st, _form, _messages = session
    before = st.beam_config
    with pytest.raises(TypeError, match="the form refused it"):
        st.bus.do(BeamConfigCommand(st, {"alpha_x": 0.5}, show=broken))
    assert st.beam_config is before and not st.bus.can_undo


def test_a_deleted_form_does_not_stop_the_step(session):
    """The Beam tab gone (teardown): the session beam is still set."""
    from linac_gen_gui.interphase.commands import BeamConfigCommand

    def gone(_cfg):
        raise RuntimeError("wrapped C/C++ object of type BeamTab has been deleted")

    st, _form, _messages = session
    st.bus.do(BeamConfigCommand(st, {"alpha_x": 0.5}, show=gone))
    assert st.beam_config.alpha_x == 0.5 and st.bus.can_undo


def test_without_a_form_it_sets_the_session_beam_only(qapp, mini_lattice):
    from linac_gen_gui.interphase.commands import BeamConfigCommand
    from linac_gen_gui.interphase.state import AppState

    from linac_gen.core.config import BeamConfig
    st = AppState()
    st.set_lattice(mini_lattice, None)
    st.set_beam_config(BeamConfig(species="proton", energy=3.0, frequency=162.5))
    st.bus.do(BeamConfigCommand(st, {"current": 2.5}))
    assert st.beam_config.current == 2.5
    st.bus.undo()
    assert st.beam_config.current == 0.0
