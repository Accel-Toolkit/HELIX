"""The autouse ``_settings_hygiene`` fixture must leave the process-wide
(sandboxed) QSettings store clean of session keys between tests.

Regression for the 2026-09-02 full-suite hang: ``test_assistant_panel``
opened ``examples/fodo_cell.dat`` through the real slot, which persisted
``lastLatticePath`` into the process-wide sandbox store; a later
window's startup restore timer then silently swapped ``state.lattice``
to the fodo cell mid-flow in ``test_backtrack_end_to_end`` and the
backtrack energy audit (correctly) refused the mismatched pair.

The two tests below are order-dependent BY DESIGN (pytest runs a file
top-down): test_a plants the pollution, test_b proves the fixture
boundary scrubbed it.
"""
from __future__ import annotations

import pytest

pytest.importorskip("PyQt6")


def test_a_plants_session_keys():
    from linac_gen_gui.interphase.app import (
        _SETTINGS_LAST_LATTICE, _SETTINGS_SESSION_BEAM, _settings)
    s = _settings()
    s.setValue(_SETTINGS_LAST_LATTICE, "/tmp/hygiene_pollution.dat")
    s.setValue(_SETTINGS_SESSION_BEAM, '{"energy": 1.0}')
    # sanity: the keys really are in the store within this test
    assert s.value(_SETTINGS_LAST_LATTICE) == "/tmp/hygiene_pollution.dat"
    assert s.value(_SETTINGS_SESSION_BEAM)


def test_b_store_is_clean_again():
    from linac_gen_gui.interphase.app import (
        _SETTINGS_LAST_LATTICE, _SETTINGS_SESSION_BEAM, _settings)
    s = _settings()
    assert s.value(_SETTINGS_LAST_LATTICE) is None, \
        "lastLatticePath survived the test boundary — hygiene failed"
    assert s.value(_SETTINGS_SESSION_BEAM) is None, \
        "sessionBeamConfig survived the test boundary — hygiene failed"
