"""A Matching Apply keeps the Surrogates tab's rows, their Use ticks and the registry.

Apply writes the matched values into the lattice in hand, an edit like an inspector change,
not another lattice object, for which the Surrogates tab would start over (every surrogate
unticked, the registry cleared, and a rescan taking an old file's held settings from the
matched elements).  The rows, ticks and registrations stay, and the weights decide at the
query — a setting they hold that Apply changed has them refuse (``its <setting> changed``), an
input of theirs moved inside their range has them answer.  Undo puts the setting back, and
they answer again.
"""
import copy
import os

import numpy as np
import pytest

from linac_gen.core.config import BeamConfig
from linac_gen.surrogates import registry
from linac_gen.surrogates.base import OutOfScopeError
from tests.surrogates.accuracy_helpers import W, native, pinned
from tests.surrogates.test_frequency_jump import F_NEW, _ref
from tests.surrogates.test_gui_edits_keep_rows import _line


@pytest.fixture(scope="module")
def qapp():
    pytest.importorskip("PyQt6")
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PyQt6.QtWidgets import QApplication
    return QApplication.instance() or QApplication([])


@pytest.fixture(autouse=True)
def _no_modal_boxes(monkeypatch):
    from PyQt6.QtWidgets import QMessageBox
    shown = []
    for kind in ("warning", "critical", "information", "question"):
        monkeypatch.setattr(QMessageBox, kind, staticmethod(
            lambda *a, _k=kind, **k: shown.append(_k) or QMessageBox.StandardButton.Yes))
    yield shown


def _tabs(lattice):
    """The session as the app builds it: a proton beam in the Beam tab, then the Surrogates
    and Matching tabs on the same state."""
    from linac_gen_gui.interphase.state import AppState
    from linac_gen_gui.interphase.tabs.beam_tab import BeamTab
    from linac_gen_gui.interphase.tabs.matching_tab import MatchingTab
    from linac_gen_gui.interphase.tabs.surrogates_tab import SurrogatesTab
    state = AppState()
    state.set_lattice(lattice, None)
    beam_tab = BeamTab(state)
    beam_tab.set_beam_config(BeamConfig(species="proton", energy=W, frequency=F_NEW,
                                        n_particles=500))
    surrogates = SurrogatesTab(state)
    return state, beam_tab, surrogates, MatchingTab(state, beam_tab)


def _row(tab, element, **kw):
    """A row for ``element`` (pinned weights, its native map), ticked."""
    from pathlib import Path
    surr = pinned(element, native(element), **kw)
    tab._trained[element.name] = (surr, Path("w"), surr.metadata)
    tab._refresh_table()
    tab._on_use_toggled(element.name, True)
    return surr


def _matched_kb(lattice, kb):
    """A match result that set the map's kb to ``kb`` in its copy of the lattice."""
    from linac_gen.matching.engine import MatchResult
    from linac_gen.matching.variables import Variable
    lat2 = copy.deepcopy(lattice)
    target = lat2.elements[1]
    x0 = float(target.kb)
    target.kb = kb
    var = Variable(target=target, attr="kb", vmin=0.5, vmax=1.5, x0=x0, label="MAP.kb")
    res = MatchResult(success=True, message="", n_iter=1, elapsed_s=0.0,
                      x0=np.array([x0]), x_final=np.array([kb]), residuals=np.zeros(1),
                      cost=0.0, variables=[var], constraints=[])
    return lat2, res


def _close(*widgets):
    for w in widgets:
        w.deleteLater()


def test_apply_keeps_the_row_its_tick_and_its_registration(qapp):
    lat = _line()
    state, beam_tab, surrogates, matching = _tabs(lat)
    try:
        el = lat.elements[1]
        surr = _row(surrogates, el)                    # kb is a setting these weights hold
        lat2, res = _matched_kb(lat, 1.3)
        matching._aa_result = (lat2, copy.deepcopy(state.beam_config), res)

        matching._on_match_apply()

        assert surrogates._trained.get("MAP", (None,))[0] is surr       # the row
        assert surrogates._table.rowCount() == 1
        assert surrogates._table.cellWidget(0, 3).isChecked()            # its tick
        assert registry.get(surr.metadata.lattice_hash, "MAP") is surr  # its registration
        assert state.lattice is lat and el.kb == 1.3
        # the weights decide at the query: the setting they hold changed, as after an
        # inspector edit
        with pytest.raises(OutOfScopeError, match="its kb is 1.3"):
            surr.fitted_matrix(_ref(F_NEW, W))
        assert surr.refusals == {"its kb changed": 1}

        state.bus.undo()                               # the setting back: they answer
        assert el.kb == 1.0 and registry.get(surr.metadata.lattice_hash, "MAP") is surr
        np.testing.assert_allclose(surr.fitted_matrix(_ref(F_NEW, W)), native(el))
        state.bus.redo()
        assert el.kb == 1.3 and surrogates._table.cellWidget(0, 3).isChecked()
    finally:
        _close(matching, surrogates, beam_tab)


def test_apply_of_an_input_of_the_weights_keeps_them_answering(qapp):
    lat = _line()
    state, beam_tab, surrogates, matching = _tabs(lat)
    try:
        el = lat.elements[1]
        M = native(el)
        surr = _row(surrogates, el, params=(("kb", 0.5, 1.5),))   # kb an input of theirs
        lat2, res = _matched_kb(lat, 1.3)
        matching._aa_result = (lat2, copy.deepcopy(state.beam_config), res)

        matching._on_match_apply()

        assert surrogates._trained.get("MAP", (None,))[0] is surr
        assert surrogates._table.cellWidget(0, 3).isChecked()
        assert registry.get(surr.metadata.lattice_hash, "MAP") is surr
        assert el.kb == 1.3
        np.testing.assert_allclose(surr.fitted_matrix(_ref(F_NEW, W)), M)
        assert surr.refusals == {}
    finally:
        _close(matching, surrogates, beam_tab)


@pytest.mark.parametrize("compare", ["envelope", "mp"])
def test_a_stopped_comparison_refuses_the_apply_until_its_worker_ends(qapp, monkeypatch,
                                                                     compare):
    """The Surrogates tab's comparisons set the running flag too and walk the lattice in
    hand.  The toolbar's Stop clears the flag and does not stop them: the Apply is refused
    until the comparison's worker has ended."""
    from linac_gen_gui.interphase.tabs import surrogates_tab as mod

    from tests.gui.test_matching_apply import _Walker
    walkers = []

    def walker(*_args, **_kwargs):
        walkers.append(_Walker())
        return walkers[-1]

    monkeypatch.setattr(mod, "_EnvCompareWorker" if compare == "envelope"
                        else "_MpCompareWorker", walker)
    lat = _line()
    state, beam_tab, surrogates, matching = _tabs(lat)
    try:
        el = lat.elements[1]
        _row(surrogates, el)
        lat2, res = _matched_kb(lat, 1.3)
        matching._aa_result = (lat2, copy.deepcopy(state.beam_config), res)
        if compare == "envelope":
            surrogates._on_compare_clicked(el.name)
        else:
            surrogates._on_mp_compare_clicked()
        assert state.running and len(walkers) == 1
        state.set_running(False)                       # what the toolbar's Stop does to it

        matching._on_match_apply()

        assert el.kb == 1.0
        assert matching._aa_status.text().startswith("Not applied — a run is going.")
        walkers[0].end()
        matching._on_match_apply()
        assert el.kb == 1.3
    finally:
        for w in walkers:
            w.end()
        _close(matching, surrogates, beam_tab)
