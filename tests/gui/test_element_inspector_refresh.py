"""Regression tests: the element inspector must track external lattice
changes (undo/redo, edits from other views) without fighting its own
editors.

Three behaviours pinned here:

* an external ``bus.do`` on the selected element rebuilds the inspector
  so the displayed value matches the model;
* the inspector's OWN commits do NOT trigger a rebuild (that would
  destroy the editor under the user's cursor on every keystroke);
* when the selected element disappears from the lattice (delete /
  undo-of-insert), the selection is cleared instead of leaving a ghost
  editor bound to an orphaned element.
"""
from __future__ import annotations

import pytest

pytest.importorskip("PyQt6")

from PyQt6.QtCore import QEvent  # noqa: E402
from PyQt6.QtWidgets import QApplication, QDoubleSpinBox  # noqa: E402

from linac_gen_gui.interphase.commands import (  # noqa: E402
    DeleteCommand, ParamChangeCommand,
)
from linac_gen_gui.interphase.panels.element_inspector import (  # noqa: E402
    ElementInspector,
)
from linac_gen_gui.interphase.state import AppState  # noqa: E402


def _flush_deferred_deletes() -> None:
    QApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)


def _spin_values(inspector) -> set[float]:
    return {round(s.value(), 6) for s in inspector.findChildren(QDoubleSpinBox)}


@pytest.fixture()
def st_and_inspector(qapp, mini_lattice):
    st = AppState()
    st.set_lattice(mini_lattice, None)
    insp = ElementInspector(st)
    yield st, insp
    insp.deleteLater()
    _flush_deferred_deletes()


def test_external_change_refreshes_fields(st_and_inspector):
    st, insp = st_and_inspector
    quad = st.lattice.elements[1]          # gradient == 10.0
    st.set_selected(quad)
    _flush_deferred_deletes()
    assert 10.0 in _spin_values(insp)

    # Simulate an edit from another view (lattice table, undo, ...).
    st.bus.do(ParamChangeCommand(quad, "gradient", 10.0, 25.0))
    _flush_deferred_deletes()
    assert 25.0 in _spin_values(insp)

    # Undo must refresh back.
    st.bus.undo()
    _flush_deferred_deletes()
    assert 10.0 in _spin_values(insp)
    assert 25.0 not in _spin_values(insp)


def test_own_commit_does_not_rebuild(st_and_inspector):
    st, insp = st_and_inspector
    quad = st.lattice.elements[1]
    st.set_selected(quad)
    _flush_deferred_deletes()
    widgets_before = set(insp.findChildren(QDoubleSpinBox))

    insp._commit(quad, "gradient", 42.0)   # what an editor signal does

    assert quad.gradient == 42.0
    # No teardown/rebuild: the exact same editor widgets are still alive.
    assert set(insp.findChildren(QDoubleSpinBox)) == widgets_before


def test_deleted_selection_is_cleared(st_and_inspector):
    st, insp = st_and_inspector
    quad = st.lattice.elements[1]
    st.set_selected(quad)

    st.bus.do(DeleteCommand(quad))         # delete from "another view"

    assert st.selected is None
    assert insp._title.text() == "Inspector"
