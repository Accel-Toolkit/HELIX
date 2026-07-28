"""Fitted-lattice save guard: plain Save must never silently overwrite
the opened deck with optimizer output (the HWR-fixture clobber).
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from linac_gen_gui.interphase.app import InterphaseWindow
from linac_gen_gui.interphase.state import AppState


def _fake_app(state, calls):
    return SimpleNamespace(
        state=state,
        _save_lattice_as=lambda suggest=None: calls.append(
            ("save_as", suggest)),
        _write_lattice=lambda path: calls.append(("write", path)),
    )


def _state(path="examples/foo.dat"):
    st = AppState()
    st.set_lattice(object(), path)
    return st


def test_flag_reset_by_set_lattice(qapp):
    st = _state()
    st.lattice_fitted = True
    st.set_lattice(object(), "examples/other.dat")
    assert st.lattice_fitted is False


def test_plain_save_writes_in_place_when_not_fitted(qapp):
    st, calls = _state(), []
    InterphaseWindow._save_lattice(_fake_app(st, calls))
    assert calls == [("write", "examples/foo.dat")]


def test_fitted_lattice_reroutes_to_save_as_with_suggestion(qapp):
    st, calls = _state(), []
    st.lattice_fitted = True
    InterphaseWindow._save_lattice(_fake_app(st, calls))
    assert len(calls) == 1
    kind, suggest = calls[0]
    assert kind == "save_as"
    assert suggest.endswith("foo.matched.dat")
    assert not any(k == "write" for k, _ in calls)   # source untouched


def test_rematched_deck_suggestion_does_not_stack_suffix(qapp):
    """Re-fitting a previously saved *.matched.dat must suggest the same
    name, not foo.matched.matched.dat."""
    st, calls = _state("examples/foo.matched.dat"), []
    st.lattice_fitted = True
    InterphaseWindow._save_lattice(_fake_app(st, calls))
    assert calls[0][1].endswith("foo.matched.dat")
    assert ".matched.matched" not in calls[0][1]


def test_apply_knee_point_sets_fitted_flag(qapp, monkeypatch):
    """Review CONFIRMED-BUG closure: the Pareto dialog's Apply writes
    optimizer output into the LIVE lattice — it must arm the save guard
    exactly like the matcher's Apply, or open deck → mo run → Apply →
    Ctrl+S silently clobbers the source (the original footgun)."""
    import numpy as np
    from linac_gen.core.lattice import Lattice
    from linac_gen_gui.interphase.dialogs import multiobjective_dialog as mod

    # Any QMessageBox here means the apply path errored (and a real one
    # would hang the offscreen run) — fail fast instead.
    monkeypatch.setattr(
        mod.QMessageBox, "critical",
        staticmethod(lambda *a, **k: pytest.fail(
            f"QMessageBox.critical called: {a}")))

    st = AppState()
    st.set_lattice(Lattice(), "examples/foo.dat")   # empty: 0 variables
    assert st.lattice_fitted is False
    fake = SimpleNamespace(
        _result=SimpleNamespace(pareto_x=[np.zeros(0)],
                                pareto_F=np.zeros((1, 2))),
        _table=SimpleNamespace(currentRow=lambda: 0),
        _state=st,
        _status=SimpleNamespace(setText=lambda t: None),
    )
    mod.MultiObjectiveDialog._on_apply_knee(fake)
    assert st.lattice_fitted is True


def test_save_as_failed_write_leaves_state_untouched(qapp, monkeypatch,
                                                     tmp_path):
    """A failed Save-As write must not repoint the lattice path or
    disarm the guard — state would otherwise claim a file that was
    never written."""
    from linac_gen_gui.interphase import app as app_mod
    st = _state()
    st.lattice_fitted = True
    monkeypatch.setattr(
        app_mod.QFileDialog, "getSaveFileName",
        staticmethod(lambda *a, **k: (str(tmp_path / "new.dat"), "")))
    win = SimpleNamespace(state=st, _write_lattice=lambda path: False)
    app_mod.InterphaseWindow._save_lattice_as(win)
    assert st.lattice_fitted is True
    assert st.lattice_path == "examples/foo.dat"


def test_save_as_success_clears_flag_and_repoints(qapp, monkeypatch,
                                                  tmp_path):
    from linac_gen_gui.interphase import app as app_mod
    st = _state()
    st.lattice_fitted = True
    new = str(tmp_path / "new.dat")
    monkeypatch.setattr(
        app_mod.QFileDialog, "getSaveFileName",
        staticmethod(lambda *a, **k: (new, "")))
    win = SimpleNamespace(state=st, _write_lattice=lambda path: True)
    app_mod.InterphaseWindow._save_lattice_as(win)
    assert st.lattice_fitted is False
    assert st.lattice_path == new
