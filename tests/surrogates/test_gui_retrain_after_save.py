"""A retrain of an engaged element moves its tick to the fresh weights, after a Save too.

The element's registration goes, under whatever lattice hash it was made, and the fresh
weights take its place.  A Save Lattice that leaves other text in the deck file the tab reads
(the file it asks for after an Apply included) changes the hash after the element was
registered: kept under the old hash, the old weights would stay registered and tracking would
use them, while the row showed its Use box unticked and Deselect all could not reach them.  The
same holds for the batch path and for weights opened again from a folder.
"""
import os
import warnings
from pathlib import Path
from types import SimpleNamespace

import pytest

from linac_gen.surrogates import registry
from linac_gen.surrogates.training import save_surrogate
from tests.surrogates.accuracy_helpers import meta, mlp, native
from tests.surrogates.test_gui_edits_keep_rows import _line, _row, _tab


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


def _fresh(el, folder, lattice_hash="lh-after-save"):
    m = meta(el, native(el))
    m.lattice_hash = lattice_hash
    save_surrogate(mlp(m.architecture["input_dim"]), m, folder)
    return m


@pytest.mark.parametrize("path", ["single", "batch", "opened again"])
def test_the_fresh_weights_take_the_engaged_registration_whatever_its_hash(qapp, tmp_path, path):
    lat = _line()
    state, tab = _tab(lat)
    try:
        el = lat.elements[1]
        _row(tab, el)                                   # engaged, under the hash at load
        assert registry.list_registered() == [("lh", "MAP")]
        folder = tmp_path / "w"
        m = _fresh(el, folder)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if path == "single":
                tab._worker = SimpleNamespace(out_dir=folder)
                tab._on_finished(m)
            elif path == "batch":
                from linac_gen_gui.interphase.tabs.surrogates_tab import _BatchTrainController
                ctl = _BatchTrainController(elements=[el], ref_template=None, lattice=lat,
                                            lattice_hash="lh-after-save", shared_opts={}, tab=tab)
                ctl._out_dir = folder
                ctl._on_element_done(m)
            else:
                from linac_gen.surrogates.training import load_surrogate
                net, loaded = load_surrogate(folder)
                tab._load_cached_into_table(el, net, loaded, Path(folder))
        row = tab._trained["MAP"][0]
        assert registry.list_registered() == [("lh-after-save", "MAP")]
        assert registry.get_for_element(el) is row
        assert tab._table.cellWidget(0, 3).isChecked()
        tab._set_all_use(False)                         # and Deselect all reaches them
        assert registry.list_registered() == []
    finally:
        tab.deleteLater()


@pytest.mark.parametrize("path", ["single", "batch"])
def test_a_retrain_of_an_element_not_in_use_leaves_it_out(qapp, tmp_path, path):
    lat = _line()
    state, tab = _tab(lat)
    try:
        el = lat.elements[1]
        _row(tab, el, use=False)
        folder = tmp_path / "w"
        m = _fresh(el, folder)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if path == "single":
                tab._worker = SimpleNamespace(out_dir=folder)
                tab._on_finished(m)
            else:
                from linac_gen_gui.interphase.tabs.surrogates_tab import _BatchTrainController
                ctl = _BatchTrainController(elements=[el], ref_template=None, lattice=lat,
                                            lattice_hash="lh-after-save", shared_opts={}, tab=tab)
                ctl._out_dir = folder
                ctl._on_element_done(m)
        assert registry.list_registered() == [] and not tab._table.cellWidget(0, 3).isChecked()
    finally:
        tab.deleteLater()
