"""Matching tab: the "Allow inert constraints" escape hatch (2026-07-11).

The engine's pre-run constraint audit refuses to run when an active
constraint would be silently ignored (stub card with weight, or
MIN_TRANSMISSION under the envelope cost-solver).  The CLI grew
``--allow-inert-constraints``; this locks the GUI counterpart: a
checkbox (default OFF) whose state reaches ``match()`` via the worker.
"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("PyQt6")


def _fake_result():
    return SimpleNamespace(
        success=True, cost=0.0, n_iter=1, elapsed_s=0.0,
        x0=np.array([]), x_final=np.array([]),
        variables=[], constraints=[], per_constraint_residuals={},
        baseline_cost=None, message="",
    )


def _beam_cfg():
    from linac_gen.core.config import BeamConfig
    return BeamConfig(species="proton", energy=3.0, frequency=162.5,
                      current=0.0, n_particles=100)


def test_checkbox_exists_and_defaults_off(qapp, mini_lattice):
    from linac_gen_gui.interphase.state import AppState
    from linac_gen_gui.interphase.tabs.beam_tab import BeamTab
    from linac_gen_gui.interphase.tabs.matching_tab import MatchingTab

    st = AppState()
    st.set_lattice(mini_lattice, "/tmp/allow_inert.dat")
    beam_tab = BeamTab(st)
    tab = MatchingTab(st, beam_tab)
    try:
        assert hasattr(tab, "_aa_allow_inert")
        # Default MUST be off — the audit's refusal is the safe behavior.
        assert not tab._aa_allow_inert.isChecked()
        assert "inert" in tab._aa_allow_inert.text().lower()
    finally:
        tab.deleteLater()
        beam_tab.deleteLater()


@pytest.mark.parametrize("flag", [False, True])
def test_worker_forwards_allow_inert_to_match(qapp, mini_lattice,
                                              monkeypatch, flag):
    """The worker must pass its allow_inert_constraints through to
    match() — run synchronously with match() stubbed out."""
    import linac_gen.matching as matching_mod
    from linac_gen_gui.interphase.tabs.matching_tab import _MatchWorker

    captured: dict = {}

    def fake_match(lattice, beam_cfg, **kwargs):
        captured.update(kwargs)
        return _fake_result()

    monkeypatch.setattr(matching_mod, "match", fake_match)

    worker = _MatchWorker(
        mini_lattice, _beam_cfg(),
        space_charge=False, max_iter=2,
        allow_inert_constraints=flag,
    )
    worker.run()          # synchronous — QThread.run is a plain method here
    assert captured.get("allow_inert_constraints") is flag


def test_worker_default_is_false(qapp, mini_lattice, monkeypatch):
    """Omitting the kwarg keeps the audit armed (False), matching the
    engine default."""
    import linac_gen.matching as matching_mod
    from linac_gen_gui.interphase.tabs.matching_tab import _MatchWorker

    captured: dict = {}

    def fake_match(lattice, beam_cfg, **kwargs):
        captured.update(kwargs)
        return _fake_result()

    monkeypatch.setattr(matching_mod, "match", fake_match)

    worker = _MatchWorker(mini_lattice, _beam_cfg(),
                          space_charge=False, max_iter=2)
    worker.run()
    assert captured.get("allow_inert_constraints") is False
