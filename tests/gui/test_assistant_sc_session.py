"""The assistant panel's GUI adapter gives the compute tools the Numerics
tab's space-charge settings, and says so when the GUI does not answer
(linac_gen.assist.tools._tool_sc_config)."""
from __future__ import annotations

from types import SimpleNamespace

import pytest

pytest.importorskip("PyQt6")


class _Tab:
    def __init__(self, cfg):
        self.cfg = cfg
        self.calls = []

    def current_sc_config(self, current, continuous=False):
        self.calls.append((current, continuous))
        return self.cfg


class _Nav:
    """The panel's run_on_gui: the call, or None (timeout, closing)."""

    def __init__(self, tab, answer=True):
        self._app = SimpleNamespace(convergence_tab=tab)
        self._answer = answer

    def run_on_gui(self, fn, timeout=3.0):
        return fn() if self._answer else None


def _context(tmp_path, nav):
    from linac_gen_gui.interphase.dialogs import assistant_panel as ap
    from linac_gen_gui.interphase.state import AppState
    return ap._make_context(AppState(), str(tmp_path), nav=nav)


def test_the_adapter_reads_the_numerics_tab(qapp, tmp_path):
    from linac_gen.assist.tools import _tool_sc_config
    from linac_gen.core.config import SpaceChargeConfig
    cfg = SpaceChargeConfig(method="tsc_adaptive", nx=40, ny=40, nz=40)
    tab = _Tab(cfg)
    ctx = _context(tmp_path, _Nav(tab))
    assert ctx.session_sc_config(5.0, continuous=True) is cfg
    assert tab.calls == [(5.0, True)]
    got, info, warns = _tool_sc_config({}, ctx, current=5.0)
    assert got is cfg and info["source"] == "Numerics tab" and warns == []


def test_a_gui_that_does_not_answer_is_reported(qapp, tmp_path):
    from linac_gen.assist.tools import _tool_sc_config
    from linac_gen.core.config import SpaceChargeConfig
    ctx = _context(tmp_path, _Nav(_Tab(SpaceChargeConfig()), answer=False))
    with pytest.raises(RuntimeError, match="Numerics tab"):
        ctx.session_sc_config(5.0)
    cfg, info, warns = _tool_sc_config({}, ctx, current=5.0)
    assert (cfg.nx, info["source"]) == (32, "fallback")
    assert len(warns) == 1 and "Numerics tab" in warns[0]
