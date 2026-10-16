"""App Nap control (services/app_nap.py): one reference-counted activity,
held while the app computes, never raising, a no-op off macOS."""
from __future__ import annotations

import pytest

pytest.importorskip("PyQt6")


class _Backend:
    available = True

    def __init__(self, fail=False):
        self.begun = 0
        self.ended = []
        self.fail = fail

    def begin(self, reason):
        if self.fail:
            raise OSError("boom")
        self.begun += 1
        return f"token{self.begun}"

    def end(self, token):
        self.ended.append(token)


def test_blocker_counts_holds():
    from linac_gen_gui.interphase.services.app_nap import AppNapBlocker
    be = _Backend()
    b = AppNapBlocker(be)
    b.hold()
    b.hold()
    assert be.begun == 1 and b.held
    b.release()
    assert be.ended == [] and b.held
    b.release()
    assert be.ended == ["token1"] and not b.held
    b.release()                                 # unmatched: nothing
    assert be.ended == ["token1"]
    b.hold()
    b.release()
    assert be.begun == 2 and be.ended == ["token1", "token2"]


def test_blocker_survives_a_failing_backend():
    from linac_gen_gui.interphase.services.app_nap import AppNapBlocker
    b = AppNapBlocker(_Backend(fail=True))
    b.hold()                                    # logged, never raised
    b.release()
    b.hold()
    b.release()
    assert not b.held


def test_no_backend_off_macos(monkeypatch):
    from linac_gen_gui.interphase.services import app_nap
    monkeypatch.setattr(app_nap.sys, "platform", "linux")
    be = app_nap._make_backend()
    assert be.available is False
    b = app_nap.AppNapBlocker(be)
    b.hold()
    b.release()


class _Clocks:
    def __init__(self):
        self.cpu = 0.0
        self.wall = 0.0

    def step(self, wall_s, cores):
        self.wall += wall_s
        self.cpu += wall_s * cores


def _activity(qapp, state=None):
    from linac_gen_gui.interphase.services.app_nap import (
        AppNapBlocker, ComputeActivity)
    from linac_gen_gui.interphase.state import AppState
    be = _Backend()
    clk = _Clocks()
    st = state or AppState()
    act = ComputeActivity(st, blocker=AppNapBlocker(be),
                          cpu_clock=lambda: clk.cpu,
                          wall_clock=lambda: clk.wall)
    return act, be, clk, st


def test_held_while_the_process_computes(qapp):
    """Half a core over an interval counts as work; two idle intervals end
    the hold."""
    act, be, clk, _st = _activity(qapp)
    try:
        clk.step(2.0, 0.1)
        act._tick()
        assert not act.held and be.begun == 0   # an idle GUI naps freely
        clk.step(2.0, 2.7)
        act._tick()
        assert act.held and be.begun == 1
        clk.step(2.0, 3.0)
        act._tick()
        assert be.begun == 1                    # one activity, not one per tick
        clk.step(2.0, 0.0)
        act._tick()
        assert act.held                         # one idle interval: kept
        clk.step(2.0, 0.0)
        act._tick()
        assert not act.held and be.ended == ["token1"]
    finally:
        act.stop()
        act.deleteLater()


def test_held_while_a_run_is_flagged(qapp):
    """``AppState.running`` holds at once (no wait for the poll) and for as
    long as it is set, whatever the CPU use; ``stop()`` releases."""
    act, be, clk, st = _activity(qapp)
    try:
        st.set_running(True)
        assert act.held and be.begun == 1
        for _ in range(3):
            clk.step(2.0, 0.0)
            act._tick()
        assert act.held
        st.set_running(False)
        for _ in range(2):
            clk.step(2.0, 0.0)
            act._tick()
        assert not act.held and be.ended == ["token1"]
        st.set_running(True)
        act.stop()
        assert not act.held and be.ended == ["token1", "token2"]
        assert not act._timer.isActive()
        st.set_running(False)
        st.set_running(True)                    # after stop(): no new hold
        clk.step(2.0, 3.0)
        act._tick()
        assert not act.held and be.begun == 2
    finally:
        act.deleteLater()


def test_the_window_holds_while_running_and_releases_on_close(win):
    from linac_gen_gui.interphase.services.app_nap import ComputeActivity
    act = win._compute_activity
    assert isinstance(act, ComputeActivity)
    win.state.set_running(True)
    assert act.held
    win.state.set_running(False)
    win._shutdown_workers()
    assert not act.held
