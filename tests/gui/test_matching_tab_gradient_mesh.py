"""Gradient+SC matching receives the Numerics-tab PIC mesh config.

2026-07-25 review, claim 8: `_on_match_clicked` built ``mp_sc_config``
only for ``cost_solver == "mp"`` — but the 'gradient' algorithm ignores
that selector and still tracks a macro-particle bunch through PIC space
charge when SC is on, so it silently fell back to the engine's
32^3/±4σ defaults while every other MP path honoured the tab.

These tests drive the REAL click path (InterphaseWindow → MatchingTab →
_on_match_clicked) offscreen, with only the worker's ``start`` stubbed
so no matcher thread actually runs.
"""
from __future__ import annotations


def _click_match(mini_lattice, monkeypatch, *, algo: str, sc: bool):
    from linac_gen_gui.interphase.app import InterphaseWindow
    from linac_gen_gui.interphase.tabs import matching_tab as mt

    # Real worker object (it stores the kwargs we assert on), but its
    # QThread must never run — the click is what's under test.
    monkeypatch.setattr(mt._MatchWorker, "start", lambda self: None)

    win = InterphaseWindow()
    try:
        win.state.set_lattice(mini_lattice, "mini.dat")
        tab = win.matching_tab
        items = [tab._aa_algorithm.itemText(i)
                 for i in range(tab._aa_algorithm.count())]
        assert algo in items, f"{algo!r} not in algorithm combo: {items}"
        tab._aa_algorithm.setCurrentText(algo)
        tab._aa_sc.setChecked(sc)
        assert tab._aa_cost_solver.currentText() == "envelope"

        tab._on_match_clicked()
        cfg = tab._aa_worker._mp_sc_config
        tab._aa_tick.stop()
        return cfg
    finally:
        win.close()
        win.deleteLater()


def test_gradient_sc_builds_mesh_config(qapp, mini_lattice, monkeypatch):
    cfg = _click_match(mini_lattice, monkeypatch, algo="gradient", sc=True)
    assert cfg is not None, \
        "gradient+SC must receive the Numerics-tab SC config"
    assert cfg.nx > 0 and cfg.grid_extent > 0


def test_least_squares_envelope_mesh_stays_none(qapp, mini_lattice,
                                                monkeypatch):
    """The envelope cost solver needs no PIC mesh — unchanged behaviour."""
    cfg = _click_match(mini_lattice, monkeypatch,
                       algo="least_squares", sc=True)
    assert cfg is None
