"""Session beam persistence: a GUI restart must reproduce the beam the
user was actually running, not silently reset to widget defaults.

Root cause of the 2026-07-16 BTL incident: the startup restore's
bare-lattice fallback restored the lattice but left the beam at defaults
(~2 MeV), so an 800 MeV transfer line ran "completely off" while the
Lattice tab looked correct."""
from __future__ import annotations

import pytest

pytest.importorskip("PyQt6")

from linac_gen.core.config import BeamConfig


@pytest.fixture()
def win(qapp):
    from linac_gen_gui.interphase.app import InterphaseWindow
    w = InterphaseWindow()
    yield w
    w.deleteLater()


def _btl_beam() -> BeamConfig:
    return BeamConfig(species="H-", energy=800.0, frequency=162.5,
                      current=4.84, n_particles=1000,
                      distribution="gaussian",
                      alpha_x=1.918, beta_x=13.814,
                      alpha_y=-0.916, beta_y=5.387)


def test_beam_config_persisted_on_state_change(win):
    from linac_gen_gui.interphase.app import (_settings,
                                              _SETTINGS_SESSION_BEAM)
    import json
    win.state.set_beam_config(_btl_beam())
    raw = _settings().value(_SETTINGS_SESSION_BEAM)
    assert raw, "beam not persisted on beam_config_changed"
    d = json.loads(raw)
    assert d["energy"] == 800.0 and d["species"] == "H-"


def test_fresh_window_restores_session_beam(win, qapp):
    """Window #1 uses an 800 MeV beam; a NEW window (fresh session, no
    project to restore) must come up with that beam, not defaults."""
    from linac_gen_gui.interphase.app import (InterphaseWindow, _settings,
                                              _SETTINGS_LAST_PROJECT)
    win.state.set_beam_config(_btl_beam())
    _settings().remove(_SETTINGS_LAST_PROJECT)   # force non-project path

    w2 = InterphaseWindow()
    try:
        w2._restore_last_session()               # what the startup timer runs
        cfg = w2.beam_tab.get_beam_config()
        assert cfg.energy == pytest.approx(800.0)
        assert cfg.species == "H-"
        assert cfg.beta_x == pytest.approx(13.814)
    finally:
        w2.deleteLater()


def test_malformed_session_beam_is_dropped(win):
    from linac_gen_gui.interphase.app import (_settings,
                                              _SETTINGS_SESSION_BEAM)
    _settings().setValue(_SETTINGS_SESSION_BEAM, "{not json")
    win._restore_session_beam()                  # must not raise
    assert not _settings().value(_SETTINGS_SESSION_BEAM)
