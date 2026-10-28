"""Results tab tile search: the box above the grid filters the result
tiles by label / section / unit / synonyms (Greek letters typed as words,
small typos tolerated), re-packs each section from the left, hides empty
sections, opens the first match on Enter, clears on Esc, never touches a
tile's enabled state, and leaves ``open_plot`` usable for hidden tiles.
"""
from __future__ import annotations

import pytest

pytest.importorskip("PyQt6")


def _noop(*_a, **_k):
    return None


def _keys():
    import linac_gen_gui.interphase.tabs.results_tab as rt
    return [c[0] for _sec, cards in rt._SECTIONS for c in cards]


def _no_lattice_keys():
    """The tiles shown without a lattice: all but the RFQ section, which
    shows only for a lattice with RFQ cells, and the RING RUN section, which
    shows only while a ring run is shown."""
    return [k for k in _keys() if not k.startswith(("rfq_", "ring_"))]


def _rfq_only(cards):
    """A section hidden here: the RFQ one (no RFQ cells) or RING RUN (no ring
    run shown)."""
    return bool(cards) and all(c._key.startswith(("rfq_", "ring_")) for c in cards)


def _hay(key):
    import linac_gen_gui.interphase.tabs.results_tab as rt
    for sec, cards in rt._SECTIONS:
        for c in cards:
            if c[0] == key:
                return rt._tile_haystack(key, c[1], sec, c[4], c[3])
    raise KeyError(key)


def _matching(query):
    import linac_gen_gui.interphase.tabs.results_tab as rt
    return [k for k in _keys() if rt._tile_matches(query, _hay(k))]


# ------------------------------------------------------------- unit (no Qt)
def test_keywords_cover_every_tile_and_no_value_word():
    import linac_gen_gui.interphase.tabs.results_tab as rt
    keys = _keys()
    assert set(keys) <= set(rt._TILE_KEYWORDS), sorted(set(keys) - set(rt._TILE_KEYWORDS))
    assert len(keys) == len(set(keys))
    assert not any("value" in _hay(k) for k in keys)


def test_transliteration_and_synonyms():
    m = _matching("sigma")
    assert "rms" in m and "phase_adv" in m and "sigma" in m
    assert "emit" in _matching("epsilon") and "emit" in _matching("emittance")
    assert set(_matching("bmag")) >= {"bmag_s", "phase"}
    assert _matching("eta") and "tune_depr" in _matching("eta") and "dispersion" in _matching("eta")
    assert set(_matching("h-")) >= {"ibs", "magstrip"}
    assert "long_twiss" in _matching("longitudinal twiss")
    assert _matching("twiss longitudinal") == ["long_twiss"]         # AND of terms
    assert _matching("TWISS") == _matching("twiss")                  # case-insensitive


def test_typo_tolerance_and_nonsense():
    assert set(_matching("emitance")) >= {"emit", "emit_n"}         # fuzzy ≥ 0.8
    assert "dispersion" in _matching("dispersoin")
    assert _matching("qqqqzz") == []
    assert _matching("xy") == []                                     # short terms: substring only
    assert len(_matching("")) == len(_keys())


# --------------------------------------------------------------------- GUI
def _tab(qapp, species="H-"):
    from linac_gen.core.config import BeamConfig
    from linac_gen_gui.interphase.state import AppState
    from linac_gen_gui.interphase.tabs.results_tab import ResultsTab
    st = AppState()
    st.set_beam_config(BeamConfig(species=species, energy=2.12, frequency=162.5, n_particles=10))
    return st, ResultsTab(st, _noop, _noop, _noop)


def _visible(tab):
    return [c._key for c in tab._cards if not c.isHidden()]


def _pos(grid, card):
    for i in range(grid.count()):
        it = grid.itemAt(i)
        if it is not None and it.widget() is card:
            r, c, _rs, _cs = grid.getItemPosition(i)
            return r, c
    return None


def test_filter_hides_repacks_and_restores(qapp):
    import linac_gen_gui.interphase.tabs.results_tab as rt
    st, tab = _tab(qapp)
    try:
        all_keys = _visible(tab)
        assert sorted(all_keys) == sorted(_no_lattice_keys())
        before = {c._key: _pos(g, c) for _h, g, cards in tab._tile_sections for c in cards}
        tab._search.setText("bmag")
        vis = _visible(tab)
        assert set(vis) == set(_matching("bmag")) and "bmag_s" in vis
        for hdr, grid, cards in tab._tile_sections:
            shown = [c for c in cards if not c.isHidden()]
            assert hdr.isHidden() == (not shown)
            for i, c in enumerate(shown):                        # packed from the left
                assert _pos(grid, c) == (i // 4, i % 4)
        assert tab._no_match.isHidden()
        tab._search.setText("qqqqzz")
        assert _visible(tab) == []
        assert all(hdr.isHidden() for hdr, _g, _c in tab._tile_sections)
        assert not tab._no_match.isHidden()
        tab._search.clear()
        assert _visible(tab) == all_keys
        assert all(hdr.isHidden() == _rfq_only(cards)
                   for hdr, _g, cards in tab._tile_sections)
        assert tab._no_match.isHidden()
        assert {c._key: _pos(g, c) for _h, g, cards in tab._tile_sections for c in cards} == before
    finally:
        tab.deleteLater()


def test_enter_opens_first_match_and_hidden_tiles_stay_openable(qapp, gui_message_boxes):
    st, tab = _tab(qapp)
    try:
        tab._search.setText("bmag along")
        assert _visible(tab) == ["bmag_s"]
        tab._search.returnPressed.emit()
        assert "bmag_s" in tab._popups
        assert tab.open_plot("rms")                                  # hidden tile: still opens
        assert "rms" in tab._popups
        assert gui_message_boxes == []
    finally:
        for d in tab._popups.values():
            d.close()
        tab.deleteLater()


def test_filter_never_changes_enabled_state(qapp):
    st, tab = _tab(qapp, species="proton")                           # IBS gated off
    try:
        ibs = tab._ibs_card
        assert not ibs.isEnabled() and not ibs.isHidden()
        tab._search.setText("stripping")
        assert not ibs.isHidden() and not ibs.isEnabled()
        tab._search.setText("rms")
        assert ibs.isHidden() and not ibs.isEnabled()
        tab._search.clear()
        assert not ibs.isHidden() and not ibs.isEnabled()
        # Enter with the disabled IBS tile as the only match opens nothing
        tab._search.setText("intra-beam")
        assert _visible(tab) == ["ibs"]
        tab._search.returnPressed.emit()
        assert "ibs" not in tab._popups
    finally:
        tab.deleteLater()


def test_refresh_while_filtered_updates_hidden_cards(qapp):
    import numpy as np
    from linac_gen.tracking.envelope import EnvelopeResults
    st, tab = _tab(qapp)
    try:
        tab._search.setText("bmag")
        env = EnvelopeResults()
        env.s = [0.0, 100.0, 200.0]
        env.sigma_x = [1.0, 1.5, 2.0]; env.sigma_y = [1.0, 1.0, 1.0]
        env.sigma_phi = [1.0, 1.0, 1.0]; env.sigma_w = [0.1, 0.1, 0.1]
        env.emit_x = [1.0] * 3; env.emit_y = [1.0] * 3
        st.set_results(env)                                          # → tab._refresh
        rms = next(c for c in tab._cards if c._key == "rms")
        assert rms.isHidden()
        assert "2" in rms._value_lbl.text()                          # updated although hidden
        assert np.isfinite(float(rms._value_lbl.text().replace(",", ".")))
    finally:
        tab.deleteLater()


def test_escape_clears_and_shortcut_exists(qapp):
    from PyQt6.QtGui import QKeySequence, QShortcut
    st, tab = _tab(qapp)
    try:
        tab._search.setText("rms")
        tab._clear_search()
        assert tab._search.text() == "" and \
            len(_visible(tab)) == len(_no_lattice_keys())
        seqs = {sc.key().toString() for sc in tab.findChildren(QShortcut)}
        assert QKeySequence(QKeySequence.StandardKey.Find).toString() in seqs
        assert QKeySequence("Esc").toString() in seqs
    finally:
        tab.deleteLater()


def test_focus_search_in_real_window(win):
    win._tabs.setCurrentWidget(win.results_tab)
    win.show(); win.pump(0.3)
    win.results_tab._focus_search()
    win.pump(0.2)
    assert win.results_tab._search.hasFocus()
    win.results_tab._search.setText("phase space")
    assert "phase" in _visible(win.results_tab)
    win.results_tab._search.clear()
