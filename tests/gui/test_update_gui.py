"""Offscreen GUI plumbing of the update feature (no network, no modal)."""
from __future__ import annotations

import time

import pytest

pytest.importorskip("PyQt6")


@pytest.fixture()
def win(qapp):
    from linac_gen_gui.interphase.app import InterphaseWindow
    w = InterphaseWindow()
    yield w
    w.close()
    w.deleteLater()


def _pump(qapp, seconds=1.5):
    from PyQt6.QtCore import QCoreApplication
    end = time.time() + seconds
    while time.time() < end:
        QCoreApplication.processEvents()
        time.sleep(0.02)


class TestMenuAndNotify:
    def test_help_entries_exist(self, win):
        tb = win._toolbar
        assert tb._update_action.text().startswith("Check for Updates")
        assert tb._update_toggle.isCheckable()

    def test_set_update_available_retitles_bold(self, win):
        win._toolbar.set_update_available("v9.9")
        act = win._toolbar._update_action
        assert "v9.9 available" in act.text()
        assert act.font().bold()

    def test_statusbar_segment_hidden_then_shown(self, win):
        seg = win._statusbar._update_seg
        assert not seg.isVisibleTo(win._statusbar)
        win._statusbar.show_update_available("v9.9")
        assert seg.isVisibleTo(win._statusbar)
        assert "v9.9" in seg.text()
        win._statusbar.clear_update_notice()
        assert not seg.isVisibleTo(win._statusbar)

    def test_statusbar_click_routes_to_flow(self, win):
        called = []
        win._open_update_flow = lambda: called.append(True)
        win._statusbar.update_clicked.connect(win._open_update_flow)
        win._statusbar._update_seg.click()
        assert called


class TestSettingsToggle:
    def test_round_trip(self, win, qapp):
        from linac_gen_gui.interphase.app import (_SETTINGS_UPDATE_CHECK,
                                                  _settings)
        win._toolbar._update_toggle.setChecked(False)
        assert _settings().value(_SETTINGS_UPDATE_CHECK, True,
                                 type=bool) is False
        win._toolbar._update_toggle.setChecked(True)
        assert _settings().value(_SETTINGS_UPDATE_CHECK, True,
                                 type=bool) is True


class TestCheckPaths:
    def test_fetch_failure_is_totally_silent(self, win, qapp):
        msgs = []
        win.state.status_message.connect(msgs.append)

        def boom():
            raise RuntimeError("no network")
        win._release_fetcher = boom
        win._start_update_check(manual=False)
        _pump(qapp)
        assert msgs == []
        assert not win._statusbar._update_seg.isVisibleTo(win._statusbar)

    def test_dev_launch_check_is_silent(self, win, qapp, monkeypatch):
        # Force the DEV classification (environment-independent: the
        # public release tree has no .git and would classify NOT_GIT) —
        # the launch path must short-circuit before calling the fetcher.
        from pathlib import Path

        from linac_gen_gui.interphase.services import update_check as uc
        monkeypatch.setattr(uc, "find_repo_root", lambda: Path("."))
        monkeypatch.setattr(uc, "classify_install",
                            lambda *_a, **_k: uc.InstallState.DEV)
        calls = []
        win._release_fetcher = lambda: calls.append(1) or None
        win._start_update_check(manual=False)
        _pump(qapp)
        assert calls == []

    def test_newer_release_raises_the_notice(self, win, qapp):
        from linac_gen_gui.interphase.services.update_check import (
            InstallState)
        win._on_update_check_done({
            "state": InstallState.PUBLIC_CLEAN, "root": None,
            "manual": False, "tag": "v99.0", "url": "https://x"})
        assert win._statusbar._update_seg.isVisibleTo(win._statusbar)
        assert "v99.0" in win._toolbar._update_action.text()

    def test_non_clean_install_opens_browser(self, win, qapp,
                                             monkeypatch):
        from linac_gen_gui.interphase.services.update_check import (
            InstallState)
        opened = []
        from PyQt6.QtGui import QDesktopServices
        monkeypatch.setattr(QDesktopServices, "openUrl",
                            staticmethod(lambda u: opened.append(
                                u.toString()) or True))
        win._pending_update = {
            "state": InstallState.PUBLIC_DIRTY, "root": None,
            "manual": True, "tag": "v99.0", "url": "https://x/rel"}
        win._open_update_flow()
        assert opened == ["https://x/rel"]


class TestRestartSeam:
    def test_restart_spawns_helper_and_skips_second_confirm(
            self, win, qapp, monkeypatch, tmp_path):
        import subprocess as sp
        launcher = tmp_path / "run_gui.sh"
        launcher.write_text("#!/bin/sh\n")
        win._pending_update = {"root": tmp_path, "tag": "v9.9"}
        spawned = []
        monkeypatch.setattr(
            sp, "Popen",
            lambda *a, **k: spawned.append((a, k)) or None)
        confirms = []
        win._confirm_discard = lambda *_a: confirms.append(1) or True
        win._restart_for_update()
        assert spawned, "detached helper was not spawned"
        argv = spawned[0][0][0]
        assert "run_gui.sh" in argv[-1]
        assert spawned[0][1].get("start_new_session") is True
        assert getattr(win, "_closing_for_restart", False) is True
        # closeEvent must NOT have asked again (flag suppressed it)
        assert confirms == []
