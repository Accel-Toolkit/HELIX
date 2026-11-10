"""VaneRFQ is retired: the Lattice tab no longer offers the RfqCell → VaneRFQ
swap, even with an RFQ deck open (its tracking has no phase slip — 57 %
transmission where the RfqCell chain transmits 100 %)."""
from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("PyQt6")

_DEMO = Path(__file__).resolve().parents[2] / "examples" / "rfq_demo" / "rfq_demo.dat"


def test_lattice_tab_offers_no_vane_swap_with_an_rfq_open(win):
    from PyQt6.QtWidgets import QPushButton

    from linac_gen.elements.rfq_cell import RfqCell
    win.open_lattice(str(_DEMO))
    assert sum(isinstance(e, RfqCell) for e in win.state.lattice.elements) == 199
    lt = win.lattice_tab
    labels = [b.text() for b in lt.findChildren(QPushButton)]
    assert "Correct orbit" in labels and "BPM targets…" in labels
    assert not [t for t in labels if "vane" in t.lower()]
    assert not hasattr(lt, "_btn_to_vane")
    assert not hasattr(lt, "_on_replace_with_vane_rfq")
