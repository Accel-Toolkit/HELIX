"""Regression test: Matching-tab Apply must keep the lattice's on-disk path.

``_on_match_apply`` used to call ``state.set_lattice(lattice_after)``
without the path argument, so Apply turned Save into Save-As, the
titlebar showed "(no lattice loaded)", and a subsequent Save wrote a
brand-new file instead of updating the opened .dat.
"""
from __future__ import annotations

import copy

import pytest

pytest.importorskip("PyQt6")


def test_match_apply_preserves_lattice_path(qapp, mini_lattice):
    from linac_gen_gui.interphase.state import AppState
    from linac_gen_gui.interphase.tabs.beam_tab import BeamTab
    from linac_gen_gui.interphase.tabs.matching_tab import MatchingTab

    st = AppState()
    st.set_lattice(mini_lattice, "/tmp/matched_source.dat")
    beam_tab = BeamTab(st)
    tab = MatchingTab(st, beam_tab)
    try:
        matched = copy.deepcopy(mini_lattice)
        tab._aa_result = (matched, beam_tab.get_beam_config(), None)

        tab._on_match_apply()

        assert st.lattice is matched
        # THE regression: path must survive the apply.
        assert st.lattice_path == "/tmp/matched_source.dat"
        # Apply re-flags both dirty bits (lattice diverges from disk).
        assert st.bus.dirty
        assert st.project_dirty
    finally:
        tab.deleteLater()
        beam_tab.deleteLater()
