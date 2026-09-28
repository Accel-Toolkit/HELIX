"""Imported results fill the same Results tiles as a live run (GUI side
of tests/io/test_results_file_completeness.py)."""
from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("PyQt6")

from linac_gen.io.hdf5_output import load_results_hdf5, save_results_hdf5  # noqa: E402
from tests.io.test_results_file_completeness import _same, make_run      # noqa: E402


@pytest.fixture(scope="module")
def run():
    return make_run()


def test_loaded_run_resaves_unchanged(run, tmp_path, qapp):
    """GUI adapter: a loaded run keeps the loss table as a structured
    array (column access), serves density_array like a live run, and
    re-saves to a file that loads back identically."""
    from linac_gen_gui.interphase.tabs.results_tab import _LoadedResults
    p1, p2 = tmp_path / "a.h5", tmp_path / "b.h5"
    save_results_hdf5(run, str(p1))
    loaded = _LoadedResults(load_results_hdf5(str(p1)), p1)
    assert _same(loaded.loss_table["s"], run.loss_table["s"])
    assert _same(loaded.density_array("x"), run.density_array("x"))
    save_results_hdf5(loaded, str(p2))
    d1, d2 = load_results_hdf5(str(p1)), load_results_hdf5(str(p2))
    for key in ("sigma_matrix", "centroid", "emit_nz", "emit_4d",
                "element_names", "ref_frequency", "mass_mev", "tail",
                "loss_table"):
        a, b = d1[key], d2[key]
        if isinstance(a, dict):
            assert all(_same(a[k], b[k]) for k in a), key
        else:
            assert (a == b) if isinstance(a, (list, float)) and key in (
                "element_names", "mass_mev") else _same(a, b), key
    for axis in ("x", "y"):
        assert _same(d1["density"][axis], d2["density"][axis])


def test_imported_run_fills_the_emittance_tiles(run, tmp_path, qapp,
                                                monkeypatch):
    """The user-visible symptom: after Import Results… the normalised
    emittance popup lacked ε_nz, and 6-D emittance / dispersion /
    eigen-emittance plots were empty; loss popups raised."""
    import pyqtgraph as pg
    from linac_gen_gui.interphase.state import AppState
    from linac_gen_gui.interphase.tabs.results_tab import (
        ResultsTab, _LoadedResults)
    from PyQt6.QtWidgets import QMessageBox
    dialogs = []                    # a refresh failure pops a warning box
    for name in ("warning", "critical"):
        monkeypatch.setattr(QMessageBox, name, staticmethod(
            lambda *a, _n=name, **k: dialogs.append((_n, a[1:3]))))
    p = tmp_path / "r.h5"
    save_results_hdf5(run, str(p))
    noop = lambda *a, **k: None                              # noqa: E731

    def points(res, key):
        state = AppState()
        state.set_results(res)
        tab = ResultsTab(state, noop, noop, noop)
        tab._open_popup(key)
        pop = tab._popups[key]
        n = 0
        for pw in pop.findChildren(pg.PlotWidget):
            for it in pw.getPlotItem().listDataItems():
                _x, y = it.getData()
                if y is not None:
                    n += int(np.isfinite(np.asarray(y, float)).sum())
        pop.close()
        return n

    loaded = _LoadedResults(load_results_hdf5(str(p)), p)
    for key in ("emit_n", "emit6d", "dispersion", "eigenemit", "emit4d",
                "long_twiss", "divergence", "aperture_loss"):
        live, imp = points(run, key), points(loaded, key)
        assert live > 0 and imp == live, (key, live, imp)
    # the loss-power popup (0 mA here, so nothing to plot either way) read
    # the loss table by column and raised on the list the adapter made
    assert points(loaded, "loss_power") == points(run, "loss_power")
    assert dialogs == [], dialogs
