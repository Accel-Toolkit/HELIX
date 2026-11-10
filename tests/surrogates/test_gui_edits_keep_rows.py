"""An edit of the lattice in hand keeps the Surrogates tab's rows and their Use ticks.

Every inspector edit, Matching Apply and undo re-broadcasts ``lattice_changed`` with the same
lattice.  The tab keeps its rows and the registry then, so no surrogate is unticked (the weights
check how their element is set when they are asked; the weights folder is not scanned again).  It
drops a row whose element left the lattice, was renamed or changed its structure, with whatever is
registered for it (and says so), and starts over for another lattice (a load) or another species'
weights folder.  The Matching Apply case: ``test_matching_apply_keeps_rows.py``.
"""
import os

import numpy as np
import pytest

from linac_gen.core.config import BeamConfig
from linac_gen.elements.drift import Drift
from linac_gen.surrogates import registry
from tests.surrogates.accuracy_helpers import W, native, pinned
from tests.surrogates.test_frequency_jump import F_NEW, _lattice, _magnet, _ref


@pytest.fixture(scope="module")
def qapp():
    pytest.importorskip("PyQt6")
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PyQt6.QtWidgets import QApplication
    return QApplication.instance() or QApplication([])


@pytest.fixture(autouse=True)
def _clean_registry():
    registry.clear()
    yield
    registry.clear()


def _tab(lattice, species="proton"):
    from linac_gen_gui.interphase.state import AppState
    from linac_gen_gui.interphase.tabs.surrogates_tab import SurrogatesTab
    state = AppState()
    state.set_beam_config(BeamConfig(species=species, energy=W, frequency=F_NEW))
    state.set_lattice(lattice, None)
    tab = SurrogatesTab(state)
    return state, tab


def _row(tab, element, use=True):
    """A row for ``element``, as the tab adds one from its weights folder, ticked."""
    from pathlib import Path
    surr = pinned(element, native(element))
    tab._trained[element.name] = (surr, Path("w"), surr.metadata)
    tab._refresh_table()
    if use:
        tab._on_use_toggled(element.name, True)
    return surr


def _line():
    return _lattice(Drift("D0", 100.0), _magnet(), Drift("D1", 100.0))


def test_an_edit_keeps_the_rows_and_their_ticks(qapp):
    lat = _line()
    state, tab = _tab(lat)
    try:
        el = lat.elements[1]
        surr = _row(tab, el)
        el.kb = 1.3                                    # an inspector edit of a held setting
        state.lattice_changed.emit(lat)
        assert tab._trained["MAP"][0] is surr
        assert registry.get(surr.metadata.lattice_hash, "MAP") is surr
        assert tab._table.cellWidget(0, 3).isChecked()
        # the weights decide at the query: this setting is not theirs
        from linac_gen.surrogates.base import OutOfScopeError
        with pytest.raises(OutOfScopeError, match="its kb is 1.3"):
            surr.fitted_matrix(_ref(F_NEW, W))
        el.kb = 1.0                                    # undone: they answer again
        state.lattice_changed.emit(lat)
        np.testing.assert_allclose(surr.fitted_matrix(_ref(F_NEW, W)), native(el))
    finally:
        tab.deleteLater()


@pytest.mark.parametrize("edit, why", [("length", "its length changed"),
                                       ("remove", "its element left the lattice")])
def test_a_row_whose_element_changed_its_structure_or_left_goes_and_is_named(qapp, edit, why):
    lat = _line()
    state, tab = _tab(lat)
    try:
        el = lat.elements[1]
        surr = _row(tab, el)
        if edit == "length":
            el.length = 150.0
        else:
            lat.elements.remove(el)
        state.lattice_changed.emit(lat)
        assert "MAP" not in tab._trained and registry.list_registered() == []
        assert tab._status.text() == (
            f"dropped the surrogate of MAP ({why}) — the element runs native until it is "
            "trained again (to load its weights again, undo the edit and use Train surrogate → "
            "Open, or re-open the project; both work only while the deck file holds the "
            "text it had at training, which File > Save Lattice can change)")
        assert tab._table.rowCount() == 0
        del surr
    finally:
        tab.deleteLater()


def test_an_edit_does_not_scan_the_weights_folder_again(qapp, monkeypatch):
    """A rescan would rebuild the rows, an old file then taking the edited settings as held."""
    from linac_gen_gui.interphase.tabs.surrogates_tab import SurrogatesTab
    lat = _line()
    state, tab = _tab(lat)
    try:
        surr = _row(tab, lat.elements[1])
        scans = []
        monkeypatch.setattr(SurrogatesTab, "_discover_cached_for_lattice",
                            lambda self, lattice: scans.append(lattice))
        lat.elements[1].kb = 1.3
        state.lattice_changed.emit(lat)
        assert scans == [] and tab._trained["MAP"][0] is surr
    finally:
        tab.deleteLater()


def test_a_rename_drops_the_row_and_its_registration(qapp):
    """The registry finds an element by its name: a renamed element's row would stay ticked,
    registered under the old name, while tracking never asks its weights."""
    from linac_gen_gui.interphase.commands import ParamChangeCommand
    lat = _line()
    state, tab = _tab(lat)
    try:
        el = lat.elements[1]
        _row(tab, el)
        state.bus.do(ParamChangeCommand(el, "name", "MAP", "CAV1"))
        assert "MAP" not in tab._trained and registry.list_registered() == []
        assert tab._status.text().startswith(
            "dropped the surrogate of MAP (its element was renamed)")
    finally:
        tab.deleteLater()


def test_an_element_replaced_by_a_copy_drops_its_row(qapp):
    import copy
    lat = _line()
    state, tab = _tab(lat)
    try:
        el = lat.elements[1]
        _row(tab, el)
        lat.elements[1] = copy.deepcopy(el)            # the same name, another object
        state.lattice_changed.emit(lat)
        assert "MAP" not in tab._trained and registry.list_registered() == []
        assert tab._status.text().startswith(
            "dropped the surrogate of MAP (its element left the lattice)")
    finally:
        tab.deleteLater()


def test_weights_opened_again_take_the_registration_and_a_dropped_row_leaves_none(qapp):
    """Train -> 'Cached surrogate found' -> Open gives an engaged element's row a new object: it
    takes the registration, so a delete that drops the row leaves nothing registered, its undo
    brings nothing back, and every tick shown is what tracking uses."""
    from pathlib import Path

    from linac_gen_gui.interphase.commands import DeleteCommand
    lat = _line()
    state, tab = _tab(lat)
    try:
        el = lat.elements[1]
        _row(tab, el)
        again = pinned(el, native(el))
        tab._load_cached_into_table(el, again.mlp, again.metadata, Path("w"))
        row = tab._trained["MAP"][0]
        assert registry.get(row.metadata.lattice_hash, "MAP") is row
        assert tab._table.cellWidget(0, 3).isChecked()
        state.bus.do(DeleteCommand(el))
        assert "MAP" not in tab._trained and registry.list_registered() == []
        state.bus.undo()
        assert registry.list_registered() == [] and registry.get_for_element(el) is None
    finally:
        tab.deleteLater()


def test_another_lattice_or_species_starts_over(qapp):
    import copy
    lat = _line()
    state, tab = _tab(lat)
    try:
        _row(tab, lat.elements[1])
        state.set_lattice(copy.deepcopy(lat), None)     # another lattice object (a load)
        assert tab._trained == {} and registry.list_registered() == []
        _row(tab, state.lattice.elements[1])
        state.set_beam_config(BeamConfig(species="H-", energy=W, frequency=F_NEW))
        assert tab._trained == {} and registry.list_registered() == []
    finally:
        tab.deleteLater()


def test_the_element_menu_follows_an_edit(qapp):
    lat = _line()
    state, tab = _tab(lat)
    try:
        assert tab._elem_combo.count() == 1
        lat.add(_magnet())
        state.lattice_changed.emit(lat)
        assert tab._elem_combo.count() == 2
    finally:
        tab.deleteLater()
