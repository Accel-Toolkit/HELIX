"""The Matching tab's Apply is an edit of the lattice in hand.

``_on_match_apply`` writes the values the match's variables varied into the
lattice in hand, and the beam fields its ``ADJUST_BEAM_*`` variables varied
into the session beam and the Beam tab, as ONE step of the command bus: one
Undo puts back both, Redo applies them again.  The lattice object stays the
same, so its undo history goes on and every tab that keeps state for the
lattice it is bound to (the Surrogates tab's rows and Use ticks) keeps it.  The
path is kept, both dirty flags are set, plain Save reroutes to Save-As (fitted
values), and the matched copy stays the result, so Save matched .dat still
writes it.  A lattice replaced, or edited in its structure, while the match
ran is refused; so is a result for a lattice or a session beam changed after
Match was clicked, and an Apply while a run is going (the result kept); the
Apply's Undo and Redo are refused while a run is going too, and a run stopped
with the toolbar's Stop is going until its worker has ended.  The Beam tab form
shows the matched fields only: a value typed there and not applied stays
through the Apply, its Undo and its Redo.
"""
from __future__ import annotations

import copy
import threading

import numpy as np
import pytest

pytest.importorskip("PyQt6")

from PyQt6.QtCore import QThread, pyqtSignal  # noqa: E402


class _Walker(QThread):
    """A run's worker whose solver has no stop check (the Sacherer envelope's): it
    goes on after the toolbar's Stop until the test ends it.  It has the signals the
    app and the Surrogates tab connect, and emits none of them."""

    finished_ok = pyqtSignal(object)
    failed = pyqtSignal(str)
    progress = pyqtSignal(int)
    progress_s = pyqtSignal(float, float)
    progress_bunch = pyqtSignal(int, int)
    aborted = pyqtSignal()
    cancelled = pyqtSignal()
    status = pyqtSignal(str)

    def __init__(self, *_args, **_kwargs):
        super().__init__()
        self._end = threading.Event()

    def request_stop(self) -> None:           # no stop check: it walks on
        pass

    def run(self) -> None:
        self._end.wait(60)

    def end(self) -> None:
        self._end.set()
        assert self.wait(10_000), "the worker did not end"


def _tabs(lattice, path="/tmp/matched_source.dat"):
    from linac_gen_gui.interphase.state import AppState
    from linac_gen_gui.interphase.tabs.beam_tab import BeamTab
    from linac_gen_gui.interphase.tabs.matching_tab import MatchingTab

    st = AppState()
    st.set_lattice(lattice, path)
    beam_tab = BeamTab(st)
    beam_tab._npart.setValue(1000)          # a light preview for every push
    beam_tab._apply(quiet=True)
    return st, beam_tab, MatchingTab(st, beam_tab)


def _result(*variables):
    """A match result whose variables are ``variables`` (as the engine leaves
    them: their targets in the match's copies, holding the matched values)."""
    from linac_gen.matching.engine import MatchResult
    x = np.array([v.read() for v in variables], dtype=float)
    return MatchResult(success=True, message="", n_iter=1, elapsed_s=0.0,
                       x0=x.copy(), x_final=x, residuals=np.zeros(1), cost=0.0,
                       variables=list(variables), constraints=[])


def _var(target, attr, label=None):
    from linac_gen.matching.variables import Variable
    return Variable(target=target, attr=attr, vmin=-1e3, vmax=1e3,
                    x0=float(getattr(target, attr)), label=label or attr)


def _matched(lattice, beam, **beam_values):
    """What a match leaves: copies of the lattice and the beam, with Q1's
    gradient and G1's phase matched (and ``beam_values``), and the result."""
    lat2, beam2 = copy.deepcopy(lattice), copy.deepcopy(beam)
    lat2.elements[1].gradient = 12.5
    lat2.elements[3].phase = -25.0
    for k, v in beam_values.items():
        setattr(beam2, k, v)
    variables = [_var(lat2.elements[1], "gradient", "Q1.gradient"),
                 _var(lat2.elements[3], "phase", "G1.phase")]
    variables += [_var(beam2, k, f"beam.{k}") for k in beam_values]
    return lat2, beam2, _result(*variables)


def _close(*widgets):
    for w in widgets:
        w.deleteLater()


def test_match_apply_preserves_lattice_path(qapp, mini_lattice):
    st, beam_tab, tab = _tabs(mini_lattice)
    try:
        matched = copy.deepcopy(mini_lattice)
        matched.elements[1].gradient = 12.5           # what the match wrote into its copy
        tab._aa_result = (matched, beam_tab.get_beam_config(),
                          _result(_var(matched.elements[1], "gradient")))

        tab._on_match_apply()

        # the lattice in hand is kept, and holds the matched value
        assert st.lattice is mini_lattice
        assert mini_lattice.elements[1].gradient == 12.5
        # THE regression: path must survive the apply.
        assert st.lattice_path == "/tmp/matched_source.dat"
        # Apply re-flags both dirty bits (lattice diverges from disk).
        assert st.bus.dirty
        assert st.project_dirty
    finally:
        _close(tab, beam_tab)


def test_apply_writes_the_matched_values_into_the_lattice_in_hand(qapp, mini_lattice):
    from linac_gen_gui.interphase.commands import MacroCommand, ParamChangeCommand
    st, beam_tab, tab = _tabs(mini_lattice)
    try:
        st.set_selected(mini_lattice.elements[1])
        lat2, beam2, res = _matched(mini_lattice, st.beam_config)
        lat2.elements[0].length = 999.0      # not a variable: Apply leaves it as it is
        tab._on_match_finished(lat2, beam2, res)       # the worker's finished_with slot
        assert tab._aa_apply.isEnabled()

        tab._aa_apply.click()

        assert st.lattice is mini_lattice
        q, d, gap = (mini_lattice.elements[i] for i in (1, 0, 3))
        assert (q.gradient, gap.phase, d.length) == (12.5, -25.0, 100.0)
        step = st.bus.peek_undo()            # ONE step of the bus …
        assert isinstance(step, MacroCommand) and step.label == "Matching Apply"
        assert [(type(c), c.element, c.attr, c.old_value, c.new_value)
                for c in step.commands] == [
            (ParamChangeCommand, q, "gradient", 10.0, 12.5),
            (ParamChangeCommand, gap, "phase", -30.0, -25.0)]
        assert st.bus.describe_changes_since_clean() == [step.describe()]
        # … which kept what an edit keeps: the selection, the path
        assert st.selected is q and st.lattice_path == "/tmp/matched_source.dat"
        assert st.lattice_fitted and st.bus.dirty and st.project_dirty
        # the result stays valid: Save matched .dat is still offered
        assert tab._aa_result[0] is lat2 and tab._aa_save.isEnabled()
        assert tab._aa_status.text().startswith("Applied — lattice + beam updated.")
    finally:
        _close(tab, beam_tab)


def test_undo_restores_the_values_and_the_beam_and_redo_applies_them_again(
        qapp, mini_lattice):
    st, beam_tab, tab = _tabs(mini_lattice)
    try:
        before = copy.deepcopy(st.beam_config)
        lat2, beam2, res = _matched(mini_lattice, st.beam_config,
                                    alpha_x=0.75, beta_x=0.42)
        tab._aa_result = (lat2, beam2, res)
        tab._on_match_apply()
        q, gap = mini_lattice.elements[1], mini_lattice.elements[3]

        def state():
            cfg, form = st.beam_config, beam_tab.get_beam_config()
            return (q.gradient, gap.phase, cfg.alpha_x, cfg.beta_x,
                    form.alpha_x, form.beta_x)

        matched = (12.5, -25.0, 0.75, 0.42, 0.75, 0.42)
        assert state() == matched
        assert st.beam_config.emit_nx == before.emit_nx      # only the varied fields
        assert "Matched beam: alpha_x 1.228 → 0.75, beta_x 0.316 → 0.42" in (
            st.bus.peek_undo().describe())

        assert st.bus.undo()
        assert state() == (10.0, -30.0, before.alpha_x, before.beta_x,
                           before.alpha_x, before.beta_x)
        assert st.beam_config == before and not st.bus.dirty
        # an edit after the Apply — its Undo among them — clears the result
        assert tab._aa_result is None and not tab._aa_apply.isEnabled()

        assert st.bus.redo()
        assert state() == matched and st.bus.dirty
    finally:
        _close(tab, beam_tab)


def test_save_after_apply_writes_the_matched_values(qapp, mini_lattice, tmp_path,
                                                     monkeypatch):
    from linac_gen_gui.interphase.tabs import matching_tab as mt

    from linac_gen.io.tracewin_parser import parse_tracewin
    st, beam_tab, tab = _tabs(mini_lattice)
    try:
        tab._on_match_finished(*_matched(mini_lattice, st.beam_config))
        tab._aa_apply.click()
        assert st.lattice is mini_lattice and tab._aa_save.isEnabled()
        out = tmp_path / "matched.dat"
        monkeypatch.setattr(mt.QFileDialog, "getSaveFileName",
                            staticmethod(lambda *a, **k: (str(out), "")))

        tab._aa_save.click()

        saved, _meta = parse_tracewin(str(out))
        quad = next(e for e in saved.elements if type(e).__name__ == "Quadrupole")
        gap = next(e for e in saved.elements if type(e).__name__ == "RFGap")
        assert (quad.gradient, gap.phase) == (12.5, -25.0)
        assert (quad.gradient, gap.phase) == (mini_lattice.elements[1].gradient,
                                              mini_lattice.elements[3].phase)
        assert tab._aa_status.text() == f"Wrote {out}"
    finally:
        _close(tab, beam_tab)


def test_a_lattice_edited_while_the_match_ran_is_refused(qapp, mini_lattice,
                                                         gui_message_boxes):
    from linac_gen_gui.interphase.commands import DeleteCommand
    st, beam_tab, tab = _tabs(mini_lattice)
    try:
        matched = _matched(mini_lattice, st.beam_config)    # the match's copies …
        st.bus.do(DeleteCommand(mini_lattice.elements[4]))  # … and an edit meanwhile
        tab._aa_result = matched                            # then the match finishes
        depth = len(st.bus._undo)

        tab._on_match_apply()

        assert mini_lattice.elements[1].gradient == 10.0
        assert mini_lattice.elements[3].phase == -30.0
        assert len(st.bus._undo) == depth and not st.lattice_fitted
        assert tab._aa_status.text().startswith(
            "Not applied — the lattice was edited while the match ran")
        assert gui_message_boxes[-1][:2] == ("warning", "Apply")
    finally:
        _close(tab, beam_tab)


def test_another_lattice_loaded_while_the_match_ran_is_refused(qapp, mini_lattice,
                                                               gui_message_boxes):
    st, beam_tab, tab = _tabs(mini_lattice)
    try:
        tab._aa_source = mini_lattice                 # as Match records it
        matched = _matched(mini_lattice, st.beam_config)
        reloaded = copy.deepcopy(mini_lattice)        # the same deck opened again
        st.set_lattice(reloaded, "/tmp/matched_source.dat")
        tab._aa_result = matched

        tab._on_match_apply()

        assert reloaded.elements[1].gradient == 10.0 and not st.bus.can_undo
        assert tab._aa_status.text().startswith(
            "Not applied — another lattice was loaded after the match started")
        assert gui_message_boxes[-1][:2] == ("warning", "Apply")
    finally:
        _close(tab, beam_tab)


def test_apply_while_a_run_is_going_changes_nothing_and_keeps_the_result(qapp, mini_lattice):
    """An envelope or multiparticle run walks the elements of the lattice in hand (its
    worker gets state.lattice itself): Apply refuses while one is going and keeps the
    result for when the run has ended.  The app marks a run with state.set_running(True)
    right before its worker starts and set_running(False) when the run ends, fails or is
    stopped; the test sets that flag as the app does (a worker that goes on after a Stop:
    see test_after_a_stop_the_apply_its_undo_and_its_redo_are_refused_until_its_worker_ends)."""
    st, beam_tab, tab = _tabs(mini_lattice)
    try:
        q, gap = mini_lattice.elements[1], mini_lattice.elements[3]
        lat2, beam2, res = _matched(mini_lattice, st.beam_config, alpha_x=0.75)
        tab._on_match_finished(lat2, beam2, res)
        beam0 = st.beam_config
        st.set_running(True)                      # as _run_envelope / _run_mp do

        tab._aa_apply.click()

        assert (q.gradient, gap.phase, st.beam_config) == (10.0, -30.0, beam0)
        assert not st.bus.can_undo and not st.lattice_fitted and not st.project_dirty
        assert tab._aa_status.text() == ("Not applied — a run is going.  The result is "
                                         "kept: click Apply when the run has ended.")
        assert tab._aa_result[0] is lat2 and tab._aa_apply.isEnabled()
        st.set_running(False)                     # as _env_done / _mp_done do
        tab._aa_apply.click()
        assert (q.gradient, gap.phase, st.beam_config.alpha_x) == (12.5, -25.0, 0.75)
        assert tab._aa_status.text().startswith("Applied — lattice + beam updated.")
    finally:
        _close(tab, beam_tab)


def test_a_run_is_going_until_its_worker_has_ended(qapp):
    """state.run_going: the running flag, or the worker of a run that has not ended (the
    toolbar's Stop clears the flag at once).  A worker that has ended, or whose object is
    gone, does not count, and the record keeps only the workers still going."""
    import gc

    from linac_gen_gui.interphase.state import AppState
    from PyQt6 import sip
    from PyQt6.QtCore import QObject
    st = AppState()
    walker = _Walker()
    try:
        assert not st.run_going
        st.set_running(True)                      # a run starts …
        assert st.run_going
        walker.start()
        st.add_run_worker(walker)                 # … the app records its worker …
        st.set_running(False)                     # … and the toolbar's Stop clears the flag
        assert st.run_going                       # the worker walks on
        walker.end()
        assert not st.run_going
    finally:
        walker.end()
    idle = _Walker()
    st.add_run_worker(idle)                       # not started: not going
    assert not st.run_going and len(st._run_workers) == 1     # the ended one dropped
    del idle
    gc.collect()
    assert not st.run_going                       # its object is gone
    owner = QObject()
    thread = QThread(owner)
    st.add_run_worker(thread)
    sip.delete(owner)                             # its C++ object goes with its owner
    assert sip.isdeleted(thread) and not st.run_going


@pytest.mark.parametrize("ends_first", ["later", "earlier"])
def test_a_run_is_going_until_every_recorded_worker_has_ended(qapp, ends_first):
    """Two runs, each with its worker recorded: the first one stopped with the toolbar's
    Stop while its worker walks on, then a second one started.  One worker ends (the later
    one at its own Stop, or the earlier one, whose result clears the running flag while the
    later one walks), and the other keeps a run going until it has ended too."""
    from linac_gen_gui.interphase.state import AppState
    st = AppState()
    earlier, later = _Walker(), _Walker()
    try:
        st.set_running(True)                      # a run starts …
        earlier.start()
        st.add_run_worker(earlier)
        st.set_running(False)                     # … and is stopped: its worker walks on
        st.set_running(True)                      # a second run starts
        later.start()
        st.add_run_worker(later)
        ended, walking = (later, earlier) if ends_first == "later" else (earlier, later)
        ended.end()
        st.set_running(False)                     # that Stop, or the earlier run's result
        assert st.run_going                       # the other worker walks on
        walking.end()
        assert not st.run_going
    finally:
        earlier.end()
        later.end()


def test_the_undo_and_the_redo_of_the_apply_are_refused_while_a_run_is_going(
        qapp, mini_lattice):
    """A run walks the elements of the lattice in hand: while one is going, the Undo and
    the Redo of the Apply change nothing, leave both stacks as they are and say why on
    the status line.  The Lattice tab's buttons and its shortcuts reach them through the
    bus; each one is tried."""
    from linac_gen_gui.interphase.tabs.lattice_tab import LatticeTab
    from PyQt6.QtGui import QShortcut
    st, beam_tab, tab = _tabs(mini_lattice)
    lattice_tab = LatticeTab(st)
    messages: list[str] = []
    st.status_message.connect(messages.append)
    try:
        q, gap = mini_lattice.elements[1], mini_lattice.elements[3]
        tab._aa_result = _matched(mini_lattice, st.beam_config, alpha_x=0.75)
        tab._on_match_apply()
        step = st.bus.peek_undo()
        keys = {s.key().toString(): s.activated.emit
                for s in lattice_tab.findChildren(QShortcut)}

        def in_hand():
            return ((q.gradient, gap.phase, st.beam_config.alpha_x),
                    list(st.bus._undo), list(st.bus._redo))

        def tried(actions, done):
            for act in actions:
                messages.clear()
                act()
                assert in_hand() == held
                assert messages == [f"Matching Apply not {done} — a run is going.  Try "
                                    "again when the run has ended."]

        st.set_running(True)                      # as _run_envelope / _run_mp do
        held = in_hand()
        assert held == ((12.5, -25.0, 0.75), [step], [])
        tried([st.bus.undo, lattice_tab._btn_undo.click, keys["Ctrl+Z"]], "undone")
        st.set_running(False)
        assert st.bus.undo()                      # the run has ended
        st.set_running(True)
        held = in_hand()
        assert held == ((10.0, -30.0, 1.228), [], [step])
        tried([st.bus.redo, lattice_tab._btn_redo.click, keys["Ctrl+Y"],
               keys["Ctrl+Shift+Z"]], "redone")
        st.set_running(False)
        assert st.bus.redo() and in_hand()[0] == (12.5, -25.0, 0.75)
    finally:
        st.set_running(False)
        _close(lattice_tab, tab, beam_tab)


def test_the_undo_and_the_redo_of_another_edit_go_through_while_a_run_is_going(
        qapp, mini_lattice):
    """While a run is going only the Apply's Undo and Redo are refused: an edit made then
    (an inspector edit, or a grouped one such as an orbit correction) is undone and redone,
    both while the running flag is set and after the toolbar's Stop while the run's worker
    walks on, and the status line gets nothing.  The Undo under it, the Apply's, is refused,
    and the status line says why."""
    from linac_gen_gui.interphase.commands import MacroCommand, ParamChangeCommand
    st, beam_tab, tab = _tabs(mini_lattice)
    messages: list[str] = []
    st.status_message.connect(messages.append)
    walker = _Walker()
    try:
        drift = mini_lattice.elements[0]
        tab._aa_result = _matched(mini_lattice, st.beam_config)
        tab._on_match_apply()
        step = st.bus.peek_undo()

        def edit_then_undo_redo_undo():
            for edit in (ParamChangeCommand(drift, "length", 100.0, 120.0),
                         MacroCommand([ParamChangeCommand(drift, "length", 100.0, 120.0)],
                                      label="Orbit correction")):
                st.bus.do(edit)
                messages.clear()
                assert st.bus.undo() and drift.length == 100.0
                assert st.bus.redo() and drift.length == 120.0
                assert st.bus.undo() and drift.length == 100.0
                assert messages == []
            assert not st.bus.undo() and st.bus.peek_undo() is step
            assert messages == ["Matching Apply not undone — a run is going.  Try again "
                                "when the run has ended."]

        st.set_running(True)                      # as the app starts a run …
        walker.start()
        st.add_run_worker(walker)                 # … and records its worker
        edit_then_undo_redo_undo()
        st.set_running(False)                     # the toolbar's Stop: the worker walks on
        edit_then_undo_redo_undo()
    finally:
        walker.end()
        st.set_running(False)
        _close(tab, beam_tab)


def test_after_a_stop_the_apply_its_undo_and_its_redo_are_refused_until_its_worker_ends(
        qapp, mini_lattice):
    """The toolbar's Stop clears state.running at once, and a solver without a stop
    check walks the lattice on to its end: the Apply, its Undo and its Redo are refused
    until the run's worker has ended."""
    st, beam_tab, tab = _tabs(mini_lattice)
    messages: list[str] = []
    st.status_message.connect(messages.append)
    walkers: list = []

    def run_then_stop():
        st.set_running(True)                      # as the app starts a run …
        walkers.append(_Walker())
        walkers[-1].start()
        st.add_run_worker(walkers[-1])            # … and records its worker
        st.set_running(False)                     # the toolbar's Stop

    try:
        q = mini_lattice.elements[1]
        tab._on_match_finished(*_matched(mini_lattice, st.beam_config))
        run_then_stop()
        tab._aa_apply.click()
        assert q.gradient == 10.0 and not st.bus.can_undo
        assert tab._aa_status.text() == ("Not applied — a run is going.  The result is "
                                         "kept: click Apply when the run has ended.")
        walkers[-1].end()
        tab._aa_apply.click()
        assert q.gradient == 12.5
        run_then_stop()
        assert not st.bus.undo() and q.gradient == 12.5
        assert messages[-1].startswith("Matching Apply not undone — a run is going.")
        walkers[-1].end()
        assert st.bus.undo() and q.gradient == 10.0
        run_then_stop()
        assert not st.bus.redo() and q.gradient == 10.0
        assert messages[-1].startswith("Matching Apply not redone — a run is going.")
        walkers[-1].end()
        assert st.bus.redo() and q.gradient == 12.5
    finally:
        for w in walkers:
            w.end()
        _close(tab, beam_tab)


def test_a_refused_undo_or_redo_of_the_apply_is_not_done_when_the_run_ends(
        qapp, mini_lattice):
    """A refused Undo or Redo of the Apply is not kept for later: when the run has ended
    as the app ends it (its worker ended, the running flag cleared, its results stored, the
    events processed) the lattice and both stacks are as they were, until the Undo or the
    Redo is clicked again."""
    st, beam_tab, tab = _tabs(mini_lattice)
    walkers: list = []
    try:
        q = mini_lattice.elements[1]
        tab._aa_result = _matched(mini_lattice, st.beam_config)
        tab._on_match_apply()
        step = st.bus.peek_undo()
        for click, gradient, stacks in ((st.bus.undo, 12.5, ([step], [])),
                                        (st.bus.redo, 10.0, ([], [step]))):
            st.set_running(True)                  # a run starts …
            walkers.append(_Walker())
            walkers[-1].start()
            st.add_run_worker(walkers[-1])        # … and its worker is recorded
            assert not click()
            walkers[-1].end()                     # the run ends …
            st.set_running(False)
            st.set_results(None)                  # … and its results are stored (_env_done)
            qapp.processEvents()
            assert q.gradient == gradient
            assert (list(st.bus._undo), list(st.bus._redo)) == stacks
            assert click()                        # clicked again
    finally:
        for w in walkers:
            w.end()
        st.set_running(False)
        _close(tab, beam_tab)


@pytest.mark.parametrize("run", ["envelope", "mp", "backtrack", "train"])
def test_a_stopped_run_of_the_window_refuses_the_apply_until_its_worker_ends(
        qapp, win, mini_lattice, monkeypatch, run):
    """Each run the window starts records its worker: after the toolbar's Stop the
    Matching tab's Apply is refused until that worker has ended."""
    from linac_gen_gui.interphase import app as appmod
    from linac_gen_gui.interphase.dialogs import backtrack_dialog as bd
    from linac_gen_gui.interphase.dialogs import train_dialog as td

    from linac_gen.core.config import BeamConfig
    from linac_gen.train import PulsePattern, TrainConfig
    walkers: list = []

    def walker(*_args, **_kwargs):
        walkers.append(_Walker())
        return walkers[-1]

    worker_class = {"envelope": "EnvelopeWorker", "mp": "MultiparticleWorker",
                    "backtrack": "BacktrackWorker", "train": "TrainWorker"}[run]
    monkeypatch.setattr(appmod, worker_class, walker)
    monkeypatch.setattr(bd.BacktrackDialog, "exec", lambda self: True)
    monkeypatch.setattr(bd.BacktrackDialog, "get_settings", lambda self: {
        "source": "results", "dst_path": None, "start": 0, "end": 4,
        "field_map_mode": "rk4", "space_charge": False, "write_dst": None})

    def train_ok(dlg):
        dlg.train_config = TrainConfig(bunch_frequency_MHz=162.5, mode="fast",
                                       pattern=PulsePattern.from_rle("1*2"))
        dlg.space_charge_enabled = False
        return True

    monkeypatch.setattr(td.TrainConfigDialog, "exec", train_ok)
    monkeypatch.setenv("LINAC_GEN_FIELDMAP_KERNEL", "1")     # the runs set it
    st, tab = win.state, win.matching_tab
    st.set_lattice(mini_lattice, "/tmp/matched_source.dat")
    cfg = BeamConfig(species="proton", energy=3.0, frequency=162.5, current=0.0,
                     n_particles=300)
    win.beam_tab.set_beam_config(cfg)
    st.set_beam_config(cfg)
    q = mini_lattice.elements[1]
    start = {"envelope": win._run_envelope, "mp": win._run_mp,
             "backtrack": win._run_backtrack, "train": win._open_train_study}[run]
    try:
        tab._aa_changed = set()                   # as Match clears it when it starts
        tab._on_match_finished(*_matched(mini_lattice, st.beam_config))
        start()
        assert st.running and len(walkers) == 1
        win._toolbar._stop_btn.click()            # the flag cleared at once
        assert not st.running and tab._aa_result is not None

        tab._aa_apply.click()

        assert q.gradient == 10.0
        assert tab._aa_status.text().startswith("Not applied — a run is going.")
        walkers[0].end()
        tab._aa_apply.click()
        assert q.gradient == 12.5
    finally:
        for w in walkers:
            w.end()


@pytest.mark.parametrize("meanwhile", ["value", "beam"])
def test_a_value_edit_or_a_beam_change_while_the_match_ran_is_refused(
        qapp, mini_lattice, meanwhile, gui_message_boxes):
    from linac_gen_gui.interphase.commands import ParamChangeCommand
    st, beam_tab, tab = _tabs(mini_lattice)
    try:
        tab._aa_source = mini_lattice                 # as Match records it …
        tab._aa_beam = copy.deepcopy(st.beam_config)
        matched = _matched(mini_lattice, st.beam_config, alpha_x=0.75)
        q, gap = mini_lattice.elements[1], mini_lattice.elements[3]
        if meanwhile == "value":                      # … an inspector edit meanwhile
            st.bus.do(ParamChangeCommand(mini_lattice.elements[0], "length", 100.0, 120.0))
        else:                                         # … or the Beam tab's Apply
            beam_tab._energy.setValue(2.5)
            beam_tab._apply_btn.click()
        tab._on_match_finished(*matched)              # then the result arrives
        what = "the lattice" if meanwhile == "value" else "the beam"
        assert tab._aa_status.text().endswith(
            f".  It no longer holds: {what} changed after Match was clicked — click "
            "Match again.")
        beam0, depth = st.beam_config, len(st.bus._undo)

        tab._aa_apply.click()

        assert (q.gradient, gap.phase, st.beam_config) == (10.0, -30.0, beam0)
        assert len(st.bus._undo) == depth and not st.lattice_fitted
        assert tab._aa_status.text() == (f"Not applied — {what} changed after Match was "
                                         "clicked: click Match again.")
        assert gui_message_boxes[-1][:2] == ("warning", "Apply")
    finally:
        _close(tab, beam_tab)


def test_while_a_run_is_going_apply_refuses_before_any_other_check(qapp, mini_lattice,
                                                                  gui_message_boxes):
    """A run going comes first: Apply changes nothing and keeps the result without
    looking at it (no warning box), even one that no longer holds; once the run has
    ended it says why that result cannot be applied."""
    from linac_gen_gui.interphase.commands import ParamChangeCommand
    st, beam_tab, tab = _tabs(mini_lattice)
    try:
        tab._aa_source = mini_lattice                 # as Match records it
        matched = _matched(mini_lattice, st.beam_config)
        st.bus.do(ParamChangeCommand(mini_lattice.elements[0], "length", 100.0, 120.0))
        tab._on_match_finished(*matched)              # it arrives no longer holding
        boxes = len(gui_message_boxes)
        st.set_running(True)

        tab._aa_apply.click()

        assert tab._aa_status.text() == ("Not applied — a run is going.  The result is "
                                         "kept: click Apply when the run has ended.")
        assert len(gui_message_boxes) == boxes and tab._aa_result is not None
        st.set_running(False)
        tab._aa_apply.click()
        assert tab._aa_status.text() == ("Not applied — the lattice changed after Match "
                                         "was clicked: click Match again.")
        assert mini_lattice.elements[1].gradient == 10.0
    finally:
        _close(tab, beam_tab)


def test_a_beam_change_with_no_result_in_hand_is_recorded_and_the_status_line_kept(
        qapp, mini_lattice):
    """No result to drop: the change is recorded, for a match still running to be refused
    when its result arrives, and the status line keeps what it says."""
    st, beam_tab, tab = _tabs(mini_lattice)
    try:
        tab._aa_beam = copy.deepcopy(st.beam_config)  # as Match records it
        tab._aa_status.setText("Matching — iter 3")
        beam_tab._energy.setValue(2.5)
        beam_tab._apply_btn.click()
        assert tab._aa_changed == {"beam"} and tab._aa_result is None
        assert tab._aa_status.text() == "Matching — iter 3"
    finally:
        _close(tab, beam_tab)


def test_the_beam_set_of_the_apply_itself_keeps_the_result(qapp, mini_lattice):
    """Apply's own step sets the session beam: that is no change after Match, even where
    the beam it leaves differs from the matched copy (here the copy also holds another
    current, a field the match did not vary and Apply does not write)."""
    st, beam_tab, tab = _tabs(mini_lattice)
    try:
        lat2, beam2, res = _matched(mini_lattice, st.beam_config, alpha_x=0.75)
        beam2.current = 9.0
        tab._on_match_finished(lat2, beam2, res)

        tab._aa_apply.click()

        assert (st.beam_config.alpha_x, st.beam_config.current) == (0.75, 5.0)
        assert tab._aa_result[0] is lat2 and tab._aa_save.isEnabled()
        assert tab._aa_status.text().startswith("Applied — lattice + beam updated.")
    finally:
        _close(tab, beam_tab)


def test_undo_and_redo_leave_a_value_typed_in_the_beam_tab_alone(qapp, mini_lattice):
    """The Apply, its Undo and its Redo put the matched beam fields in the Beam tab form,
    and nothing else: a value typed there and not applied stays, and does not reach the
    session beam (the form is not applied)."""
    st, beam_tab, tab = _tabs(mini_lattice)
    try:
        before = copy.deepcopy(st.beam_config)
        tab._aa_result = _matched(mini_lattice, st.beam_config, alpha_x=0.75, beta_x=0.42)
        beam_tab._energy.setValue(2.5)            # typed, the Beam tab's Apply not clicked
        tab._on_match_apply()
        beam_tab._emit_nx.setValue(0.33)          # typed after the Apply

        def form():
            return (beam_tab._energy.value(), beam_tab._emit_nx.value(),
                    beam_tab._alpha_x.value(), beam_tab._beta_x.value())

        assert form() == (2.5, 0.33, 0.75, 0.42)
        assert st.bus.undo()
        assert form() == (2.5, 0.33, before.alpha_x, before.beta_x)
        assert st.beam_config == before               # nothing typed reached the session
        assert st.bus.redo()
        assert form() == (2.5, 0.33, 0.75, 0.42)
        assert (st.beam_config.alpha_x, st.beam_config.beta_x) == (0.75, 0.42)
        assert (st.beam_config.energy, st.beam_config.emit_nx) == (before.energy,
                                                                   before.emit_nx)
    finally:
        _close(tab, beam_tab)


def test_a_beam_field_the_form_has_no_box_for_keeps_its_value(qapp, mini_lattice):
    """The Apply, its Undo and its Redo do not apply the form: a field of the session
    beam that the form has no box for (the bunch frequency, set in the session beam
    before the Apply) keeps its value."""
    import dataclasses
    st, beam_tab, tab = _tabs(mini_lattice)
    try:
        st.set_beam_config(dataclasses.replace(st.beam_config, bunch_frequency_MHz=81.25))
        tab._aa_beam = copy.deepcopy(st.beam_config)  # as Match records it
        tab._aa_changed = set()
        tab._aa_result = _matched(mini_lattice, st.beam_config, alpha_x=0.75)
        tab._on_match_apply()
        assert (st.beam_config.alpha_x, st.beam_config.bunch_frequency_MHz) == (0.75, 81.25)
        assert st.bus.undo()
        assert (st.beam_config.alpha_x, st.beam_config.bunch_frequency_MHz) == (1.228, 81.25)
        assert st.bus.redo()
        assert (st.beam_config.alpha_x, st.beam_config.bunch_frequency_MHz) == (0.75, 81.25)
    finally:
        _close(tab, beam_tab)


def test_nothing_to_apply_changes_nothing(qapp, mini_lattice):
    st, beam_tab, tab = _tabs(mini_lattice)
    try:
        lat2 = copy.deepcopy(mini_lattice)            # a match that moved nothing
        tab._aa_result = (lat2, copy.deepcopy(st.beam_config),
                          _result(_var(lat2.elements[1], "gradient")))

        tab._on_match_apply()

        assert not st.bus.can_undo and not st.bus.dirty
        assert not st.project_dirty and not st.lattice_fitted
        assert tab._aa_status.text() == ("Nothing to apply — the lattice and beam "
                                         "already hold the matched values.")
    finally:
        _close(tab, beam_tab)


# A real match from the tab's own controls: a quad gradient and the beam's
# alpha_x / beta_x matched to an exit Twiss.
_DECK = """\
ADJUST_BEAM_TWISS 1 1 1 0 0 0 0
DRIFT 200 30
ADJUST QUAD 2 0 -30 30 0.5 0
QUAD 80 8 20
DRIFT 400 30
QUAD 80 -8 20
DRIFT 200 30
SET_TWISS END 0 1.5 0 1.5 0 0 1 1 1 1 0 0
END
"""


def _real_match(qapp, tab, timeout_ms=120_000):
    tab._aa_iter.setValue(20)
    tab._aa_run.click()                               # Match
    worker = tab._aa_worker
    assert worker.wait(timeout_ms), "the match did not finish"
    for _ in range(20):                               # deliver its finished_with
        qapp.processEvents()
        if tab._aa_result is not None:
            break
    dlg = tab._aa_convergence_dlg
    if dlg is not None:
        dlg.close()
        dlg.deleteLater()


def test_a_real_match_from_the_tab_applies_undoes_and_redoes(qapp, tmp_path):
    from linac_gen.io.tracewin_parser import parse_tracewin
    deck = tmp_path / "match.dat"
    deck.write_text(_DECK)
    lat, meta = parse_tracewin(str(deck))
    assert meta["warnings"] == []
    st, beam_tab, tab = _tabs(lat, str(deck))
    try:
        quad = next(e for e in lat.elements if type(e).__name__ == "Quadrupole")
        g0, beam0 = quad.gradient, copy.deepcopy(st.beam_config)
        _real_match(qapp, tab)
        lat2, beam2, res = tab._aa_result
        assert [v.label for v in res.variables] == [
            "beam.alpha_x", "beam.beta_x", f"{quad.name}.gradient"]
        q2 = lat2.elements[next(i for i, e in enumerate(lat.elements) if e is quad)]
        matched = (q2.gradient, beam2.alpha_x, beam2.beta_x)
        assert matched != (g0, beam0.alpha_x, beam0.beta_x)

        tab._aa_apply.click()

        assert st.lattice is lat
        assert (quad.gradient, st.beam_config.alpha_x, st.beam_config.beta_x) == matched
        assert beam_tab.get_beam_config().beta_x == beam2.beta_x
        assert st.bus.undo()
        assert (quad.gradient, st.beam_config) == (g0, beam0)
        assert st.bus.redo()
        assert (quad.gradient, st.beam_config.alpha_x, st.beam_config.beta_x) == matched
    finally:
        _close(tab, beam_tab)


# Two quads ganged by one link group: one optimiser column, two variables.
_LINKED = """\
ADJUST QUAD 2 1 -30 30 0.5 0
DRIFT 100 30
QUAD 80 5 20
ADJUST QUAD 2 1 -30 30 0.5 0
DRIFT 200 30
QUAD 80 -5 20
DRIFT 100 30
SET_SIZE 1 4 0 0 0
END
"""


def test_a_real_match_of_linked_knobs_writes_every_member(qapp, tmp_path):
    from linac_gen.io.tracewin_parser import parse_tracewin
    deck = tmp_path / "linked.dat"
    deck.write_text(_LINKED)
    lat, _meta = parse_tracewin(str(deck))
    st, beam_tab, tab = _tabs(lat, str(deck))
    try:
        quads = [e for e in lat.elements if type(e).__name__ == "Quadrupole"]
        _real_match(qapp, tab)
        lat2, _beam2, res = tab._aa_result
        assert len(res.variables) == 2 and res.x_final.shape == (1,)
        at = [next(i for i, e in enumerate(lat.elements) if e is q) for q in quads]
        matched = [lat2.elements[i].gradient for i in at]
        assert matched == [float(res.x_final[0])] * 2

        tab._aa_apply.click()

        assert [q.gradient for q in quads] == matched
        assert len(st.bus.peek_undo().commands) == 2
        assert st.bus.undo()
        assert [q.gradient for q in quads] == [5.0, -5.0]
    finally:
        _close(tab, beam_tab)


@pytest.mark.parametrize("meanwhile", ["load", "delete"])
def test_a_real_match_refuses_a_lattice_changed_while_it_ran(qapp, tmp_path, meanwhile,
                                                             gui_message_boxes):
    """Match records the lattice it started from: a load, or a structural edit,
    before its result arrives is refused (the result arrives queued)."""
    from linac_gen_gui.interphase.commands import DeleteCommand

    from linac_gen.io.tracewin_parser import parse_tracewin
    deck = tmp_path / "match.dat"
    deck.write_text(_DECK)
    lat, _meta = parse_tracewin(str(deck))
    st, beam_tab, tab = _tabs(lat, str(deck))
    try:
        tab._aa_iter.setValue(10)
        tab._aa_run.click()
        if meanwhile == "load":
            st.set_lattice(copy.deepcopy(lat), str(deck))
        else:
            st.bus.do(DeleteCommand(st.lattice.elements[-1]))

        def in_hand():
            return ([getattr(e, "gradient", None) for e in st.lattice.elements],
                    st.beam_config, len(st.bus._undo))

        before = copy.deepcopy(in_hand())
        assert tab._aa_worker.wait(120_000)
        for _ in range(20):
            qapp.processEvents()
            if tab._aa_result is not None:
                break
        tab._aa_convergence_dlg.close()
        tab._aa_convergence_dlg.deleteLater()
        assert tab._aa_apply.isEnabled()

        tab._aa_apply.click()

        why = ("another lattice was loaded after the match started" if meanwhile == "load"
               else "the lattice was edited while the match ran")
        assert tab._aa_status.text().startswith(f"Not applied — {why}")
        assert in_hand() == before
        assert gui_message_boxes[-1][:2] == ("warning", "Apply")
    finally:
        _close(tab, beam_tab)


def _match_meanwhile(qapp, tab, meanwhile=None, iters=10):
    """Click Match, run ``meanwhile`` before the result arrives (the worker hands it to
    the tab queued, through the event loop), then let the result in."""
    tab._aa_iter.setValue(iters)
    tab._aa_run.click()
    if meanwhile is not None:
        meanwhile()
    assert tab._aa_worker.wait(120_000), "the match did not finish"
    for _ in range(20):                               # deliver its finished_with
        qapp.processEvents()
        if tab._aa_run.isEnabled():
            break
    tab._aa_convergence_dlg.close()
    tab._aa_convergence_dlg.deleteLater()


def _deck_tabs(tmp_path):
    from linac_gen.io.tracewin_parser import parse_tracewin
    deck = tmp_path / "match.dat"
    deck.write_text(_DECK)
    lat, _meta = parse_tracewin(str(deck))
    return (lat, *_tabs(lat, str(deck)))


def _in_hand(st):
    return ([(e.name, getattr(e, "gradient", None)) for e in st.lattice.elements],
            st.beam_config, len(st.bus._undo))


@pytest.mark.parametrize("meanwhile", ["move", "rename"])
def test_a_real_match_refuses_a_move_or_a_rename_made_while_it_ran(qapp, tmp_path, meanwhile,
                                                                   gui_message_boxes):
    """The same number of elements, each of a class the copy has at some place: only
    the place, class and name comparison sees the lattice is not the one matched."""
    from linac_gen_gui.interphase.commands import MoveCommand, ParamChangeCommand
    lat, st, beam_tab, tab = _deck_tabs(tmp_path)
    try:
        def edit():
            if meanwhile == "move":
                st.bus.do(MoveCommand(1, 4))      # the first drift after the first quad
            else:
                quad = [e for e in lat.elements if type(e).__name__ == "Quadrupole"][1]
                st.bus.do(ParamChangeCommand(quad, "name", quad.name, "QX"))

        _match_meanwhile(qapp, tab, edit)
        before = copy.deepcopy(_in_hand(st))
        assert tab._aa_apply.isEnabled()

        tab._aa_apply.click()

        assert tab._aa_status.text().startswith(
            "Not applied — the lattice was edited while the match ran")
        assert _in_hand(st) == before
        assert gui_message_boxes[-1][:2] == ("warning", "Apply")
    finally:
        _close(tab, beam_tab)


@pytest.mark.parametrize("meanwhile", ["value", "beam"])
def test_a_real_match_refuses_a_value_edit_or_a_beam_change_made_while_it_ran(
        qapp, tmp_path, meanwhile, gui_message_boxes):
    from linac_gen_gui.interphase.commands import ParamChangeCommand
    lat, st, beam_tab, tab = _deck_tabs(tmp_path)
    try:
        def edit():
            if meanwhile == "value":              # the quad the match does not vary
                quad = [e for e in lat.elements if type(e).__name__ == "Quadrupole"][1]
                st.bus.do(ParamChangeCommand(quad, "gradient", quad.gradient, -6.0))
            else:                                 # an energy, applied in the Beam tab
                beam_tab._energy.setValue(2.5)
                beam_tab._apply_btn.click()

        _match_meanwhile(qapp, tab, edit)
        what = "the lattice" if meanwhile == "value" else "the beam"
        assert tab._aa_status.text().endswith(
            f"It no longer holds: {what} changed after Match was clicked — click Match "
            "again.")
        before = copy.deepcopy(_in_hand(st))

        tab._aa_apply.click()

        assert tab._aa_status.text() == (f"Not applied — {what} changed after Match was "
                                         "clicked: click Match again.")
        assert _in_hand(st) == before
        assert gui_message_boxes[-1][:2] == ("warning", "Apply")
    finally:
        _close(tab, beam_tab)


@pytest.mark.parametrize("meanwhile", ["value", "beam"])
def test_a_change_after_the_match_finished_drops_the_result(qapp, tmp_path, meanwhile):
    from linac_gen_gui.interphase.commands import ParamChangeCommand
    lat, st, beam_tab, tab = _deck_tabs(tmp_path)
    try:
        _match_meanwhile(qapp, tab)
        assert tab._aa_apply.isEnabled()
        if meanwhile == "value":
            quad = [e for e in lat.elements if type(e).__name__ == "Quadrupole"][1]
            st.bus.do(ParamChangeCommand(quad, "gradient", quad.gradient, -6.0))
        else:
            beam_tab._energy.setValue(2.5)
            beam_tab._apply_btn.click()

        assert tab._aa_result is None
        assert not tab._aa_apply.isEnabled() and not tab._aa_save.isEnabled()
        assert tab._aa_status.text() == (
            f"{'Lattice' if meanwhile == 'value' else 'Beam'} changed since last match — "
            "click Match again.")
    finally:
        _close(tab, beam_tab)


def test_the_match_clicks_beam_and_a_beam_set_that_changes_nothing_keep_the_result(
        qapp, tmp_path):
    """Match puts the Beam tab form in the session beam: that beam is the one matched.  A
    later beam set that leaves the session beam as it is (the Beam tab's Apply with
    nothing changed) keeps the result."""
    lat, st, beam_tab, tab = _deck_tabs(tmp_path)
    try:
        quad = next(e for e in lat.elements if type(e).__name__ == "Quadrupole")
        beam_tab._energy.setValue(2.5)            # typed, the Beam tab's Apply not clicked
        _match_meanwhile(qapp, tab)
        assert st.beam_config.energy == 2.5      # the click set it
        lat2, beam2, _res = tab._aa_result
        assert beam2.energy == 2.5 and "no longer holds" not in tab._aa_status.text()
        beam_tab._apply_btn.click()               # the same form again
        assert tab._aa_result is not None and tab._aa_apply.isEnabled()

        tab._aa_apply.click()

        assert tab._aa_status.text().startswith("Applied — lattice + beam updated.")
        q2 = lat2.elements[lat.elements.index(quad)]
        assert (quad.gradient, st.beam_config.beta_x) == (q2.gradient, beam2.beta_x)
        beam_tab._apply_btn.click()               # the matched beam, as the form shows it
        assert tab._aa_result is not None and tab._aa_save.isEnabled()
    finally:
        _close(tab, beam_tab)


def test_an_edit_while_a_second_match_runs_refuses_its_result(qapp, tmp_path,
                                                              gui_message_boxes):
    from linac_gen_gui.interphase.commands import ParamChangeCommand
    lat, st, beam_tab, tab = _deck_tabs(tmp_path)
    try:
        _match_meanwhile(qapp, tab)               # a first result, not applied
        first = tab._aa_result
        quad = [e for e in lat.elements if type(e).__name__ == "Quadrupole"][1]
        seen = {}

        def edit():
            st.bus.do(ParamChangeCommand(quad, "gradient", quad.gradient, -6.0))
            seen["status"], seen["result"] = tab._aa_status.text(), tab._aa_result

        _match_meanwhile(qapp, tab, edit)
        assert first is not None and seen["result"] is None     # the first one dropped
        assert seen["status"] == "Lattice changed since last match — click Match again."
        assert tab._aa_result is not None and tab._aa_result is not first
        assert "It no longer holds: the lattice changed after Match was clicked" in (
            tab._aa_status.text())
        before = copy.deepcopy(_in_hand(st))

        tab._aa_apply.click()

        assert tab._aa_status.text() == ("Not applied — the lattice changed after Match "
                                         "was clicked: click Match again.")
        assert _in_hand(st) == before
    finally:
        _close(tab, beam_tab)


def test_the_apply_its_undo_and_its_redo_leave_the_next_match_free(qapp, tmp_path):
    """Apply's own edit keeps its result; its Undo clears it, as any edit does; the next
    Match starts over: its result holds."""
    lat, st, beam_tab, tab = _deck_tabs(tmp_path)
    try:
        _match_meanwhile(qapp, tab)
        tab._aa_apply.click()
        assert tab._aa_status.text().startswith("Applied — lattice + beam updated.")
        assert tab._aa_result is not None and tab._aa_save.isEnabled()
        assert st.bus.undo()
        assert tab._aa_result is None
        assert st.bus.redo()

        _match_meanwhile(qapp, tab)

        assert tab._aa_result is not None
        assert "no longer holds" not in tab._aa_status.text()
        tab._aa_apply.click()
        assert not tab._aa_status.text().startswith("Not applied")
    finally:
        _close(tab, beam_tab)
