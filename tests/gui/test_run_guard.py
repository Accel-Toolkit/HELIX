"""Regression: Ctrl+R / menu run paths bypass the disabled toolbar
buttons — _run_envelope/_run_mp must refuse to double-start while a run
is already in flight (the offscreen probe showed a second live worker
being created)."""
from __future__ import annotations

import pytest

pytest.importorskip("PyQt6")


def test_run_buttons_track_lattice_presence(qapp, mini_lattice):
    """Run env/MP are disabled until a lattice loads (they used to be
    enabled at startup and the click bounced off a warning dialog)."""
    from linac_gen_gui.interphase.app import InterphaseWindow

    win = InterphaseWindow()
    try:
        assert not win._toolbar._run_env_btn.isEnabled()
        assert not win._toolbar._run_mp_btn.isEnabled()

        win.state.set_lattice(mini_lattice, None)
        assert win._toolbar._run_env_btn.isEnabled()
        assert win._toolbar._run_mp_btn.isEnabled()

        win.state.set_running(True)
        assert not win._toolbar._run_env_btn.isEnabled()
        win.state.set_running(False)
        assert win._toolbar._run_env_btn.isEnabled()
    finally:
        win.state.set_running(False)
        win.close()
        win.deleteLater()


def test_run_refused_while_running(qapp, mini_lattice):
    from linac_gen_gui.interphase.app import InterphaseWindow

    win = InterphaseWindow()
    try:
        win.state.set_lattice(mini_lattice, None)
        win.state.set_running(True)          # a run is in flight

        win._run_envelope()
        assert win._envelope_worker is None  # no second worker spawned

        win._run_mp()
        assert win._mp_worker is None
    finally:
        win.state.set_running(False)
        win.close()
        win.deleteLater()
