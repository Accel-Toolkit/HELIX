"""A bend edited through the GUI's command bus (the inspector's path): its arc length and every position after it
follow, and an undo restores them.  The inspector shows the length as following, not as an editor."""
from __future__ import annotations

import pytest

pytest.importorskip("PyQt6")


def test_the_inspector_shows_a_bends_length_as_it_follows(qapp):
    from linac_gen_gui.interphase.panels.element_inspector import ElementInspector
    from linac_gen_gui.interphase.state import AppState
    from PyQt6.QtCore import QEvent
    from PyQt6.QtWidgets import QApplication, QDoubleSpinBox, QLabel

    from linac_gen.core.lattice import Lattice
    from linac_gen.elements.dipole import Dipole
    from linac_gen.elements.drift import Drift
    lat = Lattice()
    for e in (Drift("D0", 100.0), Dipole("B", angle=10.0, rho=800.0), Drift("D1", 100.0)):
        lat.add(e)
    st = AppState()
    st.set_lattice(lat, None)
    insp = ElementInspector(st)
    bend = lat.elements[1]
    try:
        st.set_selected(bend)
        QApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
        spins = set(insp.findChildren(QDoubleSpinBox))
        length = dict(insp._geometry_editors)["length"]
        old = bend.length
        assert not length.isEnabled() and length.value() == pytest.approx(old, abs=1e-6)
        insp._commit(bend, "angle", 11.03)                                     # what the angle editor does
        assert set(insp.findChildren(QDoubleSpinBox)) == spins                 # the editor in use stays
        assert not length.isEnabled() and length.value() == pytest.approx(bend.length, abs=1e-6)
        shown = [lbl.text() for lbl in insp.findChildren(QLabel)]
        assert sum(t.startswith(f"{bend.length:g} ") for t in shown) == 1      # in Layout
        assert not any(t.startswith(f"{old:g} ") for t in shown)
    finally:
        insp.deleteLater()
        QApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)


def test_a_bend_set_to_no_angle_takes_a_length_in_the_inspector(qapp):
    """Without selecting it again: at angle 0 the length can be written (a pipe), with an angle it follows."""
    from linac_gen_gui.interphase.panels.element_inspector import ElementInspector
    from linac_gen_gui.interphase.state import AppState
    from PyQt6.QtCore import QEvent
    from PyQt6.QtWidgets import QApplication

    from linac_gen.core.lattice import Lattice
    from linac_gen.elements.dipole import Dipole
    bend = Dipole("B", angle=10.0, rho=800.0)
    lat = Lattice()
    lat.add(bend)
    st = AppState()
    st.set_lattice(lat, None)
    insp = ElementInspector(st)
    try:
        st.set_selected(bend)
        QApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
        length = dict(insp._geometry_editors)["length"]
        insp._commit(bend, "angle", 0.0)
        assert length.isEnabled() and length.value() == 0.0 and bend.length == 0.0
        length.setValue(300.0)                                                  # the user's entry
        assert _inputs(bend) == (0.0, 800.0, 300.0, 300.0)
        insp._commit(bend, "angle", 5.0)
        assert not length.isEnabled() and bend.length == Dipole("B", angle=5.0, rho=800.0).length
    finally:
        insp.deleteLater()
        QApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)


def test_the_inspector_shows_what_always_follows_as_values(qapp, monkeypatch):
    """A cavity's length and synchronous beta always follow from its cells.  The inspector has no form for a cavity
    yet; given one, they are values and its cell beta an editor."""
    from linac_gen_gui.interphase.panels import element_inspector
    from linac_gen_gui.interphase.state import AppState
    from PyQt6.QtCore import QEvent
    from PyQt6.QtWidgets import QApplication, QLabel

    from linac_gen.core.lattice import Lattice
    from linac_gen.elements.ncells import NCells
    monkeypatch.setitem(element_inspector._SCHEMA, "NCells", [
        ("length", "Length", "mm"), ("beta_s", "Synchronous beta", ""), ("beta_g", "Cell beta", "")])
    cav = NCells("NC", mode=0, n_cells=3, beta_g=0.5, eot_v_per_m=1.0e6, theta_s_deg=-30.0, frequency_mhz=650.0)
    lat = Lattice()
    lat.add(cav)
    st = AppState()
    st.set_lattice(lat, None)
    insp = element_inspector.ElementInspector(st)
    try:
        st.set_selected(cav)
        QApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
        editors = dict(insp._geometry_editors)
        assert list(editors) == ["beta_g"] and editors["beta_g"].isEnabled()
        values = [attr for attr, _v, _u in insp._derived_labels]
        assert values.count("length") == 2 and "beta_s" in values          # Properties and Layout
        insp._commit(cav, "beta_g", 0.6)
        assert any(lbl.text().startswith(f"{cav.length:g} ") for lbl in insp.findChildren(QLabel))
    finally:
        insp.deleteLater()
        QApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)


def test_the_add_element_dialog_makes_a_bend(qapp):
    """The dialog writes every field onto its template; a bend's length follows and is not written."""
    import math

    from linac_gen_gui.interphase.dialogs.add_element import AddElementDialog

    from linac_gen.elements.dipole import Dipole
    dlg = AddElementDialog(None)
    try:
        dlg._type_combo.setCurrentText("Dipole")
        assert not dlg._field_widgets["length"].isEnabled()
        dlg._accept()
        el = dlg.element()
        assert isinstance(el, Dipole)
        assert el.length == abs(el.rho) * abs(el.angle) * math.pi / 180.0
    finally:
        dlg.deleteLater()


@pytest.mark.parametrize("angles, length, want", [
    ((0.0,), None, dict(angle=0.0, length=0.0, pipe=0.0)),               # no arc, no length asked
    ((0.0,), 300.0, dict(angle=0.0, length=300.0, pipe=300.0)),          # a pipe
    ((0.0, 5.0), None, dict(angle=5.0, length=None, pipe=0.0)),          # turned again: the arc
], ids=["no angle", "a pipe", "an angle again"])
def test_the_add_element_dialog_follows_the_angle(qapp, angles, length, want):
    """The length field follows the angle as it is typed: at angle 0 it can be written, and only what was entered is
    written (it used to keep the arc of the default angle and make a pipe of it)."""
    from linac_gen_gui.interphase.dialogs.add_element import AddElementDialog

    from linac_gen.elements.dipole import Dipole
    dlg = AddElementDialog(None)
    try:
        dlg._type_combo.setCurrentText("Dipole")
        fields = dlg._field_widgets
        for a in angles:
            fields["angle"].setValue(a)
        rho = fields["rho"].value()
        if length is not None:
            assert fields["length"].isEnabled() and fields["length"].value() == 0.0
            fields["length"].setValue(length)
        dlg._accept()
        el = dlg.element()
        arc = Dipole("B", angle=want["angle"], rho=rho).length
        assert _inputs(el) == (want["angle"], rho, arc if want["length"] is None else want["length"], want["pipe"])
        assert fields["length"].isEnabled() == (want["angle"] == 0.0)
    finally:
        dlg.deleteLater()


def test_the_add_element_dialog_writes_no_field_that_could_not_be_entered(qapp):
    """Even when the form did not follow an edit (signals off), the length shown for the default angle is not taken
    as a pipe's length."""
    from linac_gen_gui.interphase.dialogs.add_element import AddElementDialog
    dlg = AddElementDialog(None)
    try:
        dlg._type_combo.setCurrentText("Dipole")
        angle = dlg._field_widgets["angle"]
        angle.blockSignals(True)
        angle.setValue(0.0)
        angle.blockSignals(False)
        dlg._accept()
        assert _inputs(dlg.element())[2:] == (0.0, 0.0)
    finally:
        dlg.deleteLater()


def test_the_inspector_follows_a_pipe_given_an_angle_and_back(qapp):
    from linac_gen_gui.interphase.panels.element_inspector import ElementInspector
    from linac_gen_gui.interphase.state import AppState
    from PyQt6.QtCore import QEvent
    from PyQt6.QtWidgets import QApplication

    from linac_gen.core.lattice import Lattice
    from linac_gen.elements.dipole import Dipole
    pipe = Dipole("P", angle=0.0, rho=1000.0)
    pipe.length = 300.0
    lat = Lattice()
    lat.add(pipe)
    st = AppState()
    st.set_lattice(lat, None)
    insp = ElementInspector(st)
    try:
        st.set_selected(pipe)
        QApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
        length = dict(insp._geometry_editors)["length"]
        assert length.isEnabled() and length.value() == pytest.approx(300.0)
        insp._commit(pipe, "angle", 10.0)
        assert not length.isEnabled() and length.value() == pytest.approx(pipe.length, abs=1e-6)
        insp._commit(pipe, "angle", 0.0)
        assert length.isEnabled() and length.value() == pytest.approx(300.0) and pipe.length == 300.0
    finally:
        insp.deleteLater()
        QApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)


def _inputs(b):
    """What a bend's behaviour follows from (an unset pipe length and a pipe of no length behave alike)."""
    return (b.angle, b.rho, b.length, getattr(b, "_pipe_length", None) or 0.0)


def test_the_assistants_rollback_puts_a_pipe_back_with_a_users_edit_on_top(qapp, tmp_path):
    """The assistant's plan cannot simply be undone when the user edited meanwhile: the compensating step puts the
    pipe back exactly and leaves the user's edit."""
    from linac_gen_gui.interphase.commands import ParamChangeCommand
    from linac_gen_gui.interphase.dialogs.assistant_panel import _make_context
    from linac_gen_gui.interphase.state import AppState

    from linac_gen.core.lattice import Lattice
    from linac_gen.elements.dipole import Dipole
    from linac_gen.elements.drift import Drift
    pipe = Dipole("P", angle=0.0, rho=1000.0)
    pipe.length = 300.0
    d = Drift("D", 100.0)
    lat = Lattice()
    lat.add(pipe)
    lat.add(d)
    st = AppState()
    st.set_lattice(lat, None)
    before = _inputs(pipe)
    ctx = _make_context(st, str(tmp_path))
    changes = [(pipe, "length", 500.0), (pipe, "angle", 10.0)]
    olds = [(e, a, getattr(e, a)) for e, a, _v in changes]
    handle = ctx.apply_param_changes(changes, label="Assistant tuning plan")
    st.bus.do(ParamChangeCommand(d, "length", 100.0, 120.0))         # the user's edit lands on top
    ctx.revert_param_changes(handle, olds, label="Assistant tuning plan")
    assert _inputs(pipe) == before and d.length == 120.0


def test_the_assistants_rollback_keeps_a_users_edit_of_the_same_bend(qapp, tmp_path):
    """The plan turned the bend, the user then changed its radius: the rollback turns it back and the user's radius
    stays, with the arc of the two."""
    from linac_gen_gui.interphase.commands import ParamChangeCommand
    from linac_gen_gui.interphase.dialogs.assistant_panel import _make_context
    from linac_gen_gui.interphase.state import AppState

    from linac_gen.core.lattice import Lattice
    from linac_gen.elements.dipole import Dipole
    bend = Dipole("B", angle=10.0, rho=800.0)
    lat = Lattice()
    lat.add(bend)
    st = AppState()
    st.set_lattice(lat, None)
    ctx = _make_context(st, str(tmp_path))
    changes = [(bend, "angle", 12.0)]
    olds = [(e, a, getattr(e, a)) for e, a, _v in changes]
    handle = ctx.apply_param_changes(changes, label="Assistant tuning plan")
    st.bus.do(ParamChangeCommand(bend, "rho", 800.0, 700.0))          # the user's edit of the same bend
    ctx.revert_param_changes(handle, olds, label="Assistant tuning plan")
    assert _inputs(bend) == _inputs(Dipole("B", angle=10.0, rho=700.0))
    st.bus.undo()                                                       # the rollback, then the user's edit
    assert _inputs(bend) == _inputs(Dipole("B", angle=12.0, rho=700.0))
    st.bus.undo()
    assert _inputs(bend) == _inputs(Dipole("B", angle=12.0, rho=800.0))


@pytest.mark.parametrize("user_edit_on_top", [False, True], ids=["undone", "compensated"])
def test_a_plan_that_makes_a_bend_a_pipe_rolls_back_exactly(qapp, tmp_path, user_edit_on_top):
    """The plan sets the angle to 0 and then gives the pipe a length.  The length the second change found is the one
    the first left (none), not the arc the plan's caller saw: rolled back, by an undo or by a compensating step, the
    bend is the bend before, and it has no pipe length to come back to at angle 0."""
    from linac_gen_gui.interphase.commands import ParamChangeCommand
    from linac_gen_gui.interphase.dialogs.assistant_panel import _make_context
    from linac_gen_gui.interphase.state import AppState

    from linac_gen.core.lattice import Lattice
    from linac_gen.elements.dipole import Dipole
    from linac_gen.elements.drift import Drift
    bend, twin = Dipole("B", angle=10.0, rho=800.0), Dipole("B", angle=10.0, rho=800.0)
    d = Drift("D", 100.0)
    lat = Lattice()
    lat.add(bend)
    lat.add(d)
    st = AppState()
    st.set_lattice(lat, None)
    before = _inputs(bend)
    ctx = _make_context(st, str(tmp_path))
    changes = [(bend, "angle", 0.0), (bend, "length", 500.0)]
    olds = [(e, a, getattr(e, a)) for e, a, _v in changes]
    handle = ctx.apply_param_changes(changes, label="Assistant tuning plan")
    assert _inputs(bend) == (0.0, 800.0, 500.0, 500.0)
    if user_edit_on_top:
        st.bus.do(ParamChangeCommand(d, "length", 100.0, 120.0))
    ctx.revert_param_changes(handle, olds, label="Assistant tuning plan")
    assert _inputs(bend) == before
    bend.angle, twin.angle = 0.0, 0.0
    assert _inputs(bend) == _inputs(twin)


def test_a_rollback_puts_back_what_it_can_and_says_what_it_cannot(qapp, tmp_path):
    """The plan resized a pipe and a drift; the user then gave the pipe an angle, so its length follows from the angle
    now: the drift goes back, the pipe keeps the user's angle, and the rollback says which value it could not put back
    (it used to fail as a whole, with a message about the GUI being busy)."""
    from linac_gen_gui.interphase.commands import ParamChangeCommand
    from linac_gen_gui.interphase.dialogs.assistant_panel import _make_context
    from linac_gen_gui.interphase.state import AppState

    from linac_gen.core.lattice import Lattice
    from linac_gen.elements.dipole import Dipole
    from linac_gen.elements.drift import Drift
    pipe = Dipole("P", angle=0.0, rho=1000.0)
    pipe.length = 300.0
    d = Drift("D", 100.0)
    lat = Lattice()
    lat.add(pipe)
    lat.add(d)
    st = AppState()
    st.set_lattice(lat, None)
    ctx = _make_context(st, str(tmp_path))
    changes = [(pipe, "length", 500.0), (d, "length", 150.0)]
    olds = [(e, a, getattr(e, a)) for e, a, _v in changes]
    handle = ctx.apply_param_changes(changes, label="Assistant tuning plan")
    st.bus.do(ParamChangeCommand(pipe, "angle", 0.0, 5.0))           # the user's edit
    with pytest.raises(RuntimeError, match=r"every change but P\.length"):
        ctx.revert_param_changes(handle, olds, label="Assistant tuning plan")
    assert d.length == 100.0
    assert (pipe.angle, pipe.length) == (5.0, Dipole("P", angle=5.0, rho=1000.0).length)


def test_a_refused_rollback_gives_its_reason(qapp, tmp_path, monkeypatch):
    """Whatever refuses the compensating step, the caller gets the reason (from a worker thread a raised exception
    used to arrive as 'the GUI did not apply the change (busy or shutting down)')."""
    from linac_gen_gui.interphase import commands
    from linac_gen_gui.interphase.commands import ParamChangeCommand
    from linac_gen_gui.interphase.dialogs.assistant_panel import _make_context
    from linac_gen_gui.interphase.state import AppState

    from linac_gen.core.lattice import Lattice
    from linac_gen.elements.drift import Drift
    d, e = Drift("D", 100.0), Drift("E", 100.0)
    lat = Lattice()
    lat.add(d)
    lat.add(e)
    st = AppState()
    st.set_lattice(lat, None)
    ctx = _make_context(st, str(tmp_path))
    handle = ctx.apply_param_changes([(d, "length", 150.0)], label="Assistant tuning plan")
    st.bus.do(ParamChangeCommand(e, "length", 100.0, 120.0))           # the user's edit on top

    def refuse(self, lattice):
        raise ValueError("refused by the element")
    monkeypatch.setattr(commands.MacroCommand, "do", refuse)
    with pytest.raises(RuntimeError, match="cannot roll back: refused by the element"):
        ctx.revert_param_changes(handle, [(d, "length", 100.0)], label="Assistant tuning plan")


def test_an_undo_after_a_change_made_outside_the_undo_stack_keeps_that_change(qapp):
    """A matched result applied outside the stack changes the radius after the user turned the bend: undoing the turn
    gives the bend of the old angle and the applied radius, and leaves the radius alone."""
    from linac_gen_gui.interphase.commands import ParamChangeCommand
    from linac_gen_gui.interphase.state import AppState

    from linac_gen.core.lattice import Lattice
    from linac_gen.elements.dipole import Dipole
    bend = Dipole("B", angle=10.0, rho=800.0)
    lat = Lattice()
    lat.add(bend)
    st = AppState()
    st.set_lattice(lat, None)
    st.bus.do(ParamChangeCommand(bend, "angle", 10.0, 11.0))
    bend.rho = 750.0                                                    # applied outside the stack
    st.bus.mark_dirty()
    st.bus.undo()
    assert _inputs(bend) == _inputs(Dipole("B", angle=10.0, rho=750.0))


def test_an_angle_edit_moves_what_follows_and_undoes(qapp):
    from linac_gen_gui.interphase.commands import ParamChangeCommand
    from linac_gen_gui.interphase.state import AppState

    from linac_gen.core.lattice import Lattice
    from linac_gen.elements.dipole import Dipole
    from linac_gen.elements.drift import Drift
    lat = Lattice()
    for e in (Drift("D0", 100.0), Dipole("B", angle=10.0, rho=800.0), Drift("D1", 100.0)):
        lat.add(e)
    st = AppState()
    st.set_lattice(lat, "<test>")
    bend = lat.elements[1]
    before = lat.get_s_positions()[1].copy()
    st.bus.do(ParamChangeCommand(bend, "angle", 10.0, 11.03))
    fresh = Dipole("B", angle=11.03, rho=800.0).length
    assert bend.length == fresh
    assert lat.get_s_positions()[1][-1] == pytest.approx(200.0 + fresh, rel=1e-15)
    st.bus.undo()
    assert bend.angle == 10.0 and bend.length == Dipole("B", angle=10.0, rho=800.0).length
    assert list(lat.get_s_positions()[1]) == list(before)
