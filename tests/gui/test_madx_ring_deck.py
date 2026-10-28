"""A MAD-X ring deck through the GUI's own Open Lattice path.

The importer's LINE expansion, inherited magnets and HARMON frequencies reach
the window the way a user's click does (only the file dialog is stubbed), and
an envelope run starts and finishes on the imported machine.
"""
from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("PyQt6")

FIXTURE = Path(__file__).resolve().parents[1] / "ring" / "fixtures" / "simple_booster.madx"


def test_ring_deck_opens_and_runs_through_the_gui(win):
    from linac_gen.core.constants import C_LIGHT
    from linac_gen.elements.rf_gap import RFGap
    from linac_gen.io.madx_parser import parse_madx

    win.open_lattice(FIXTURE)
    lat = win.state.lattice
    assert lat is not None and len(lat.elements) == 976
    assert sum(e.length for e in lat.elements) / 1000 == pytest.approx(474.202752, abs=1e-9)
    gaps = [e for e in lat.elements if isinstance(e, RFGap)]
    assert len(gaps) == 22 and {g.harmonic for g in gaps} == {84}
    beta = parse_madx(str(FIXTURE))[1]["reference"].beta
    f = 84 * beta * C_LIGHT / 474.202752 / 1e6                   # h·βc/C
    assert all(g.frequency == pytest.approx(f, rel=1e-12) for g in gaps)

    win._run_envelope()
    win.wait_worker(win._envelope_worker)
    res = win.state.results
    assert res is not None and len(res.s) > 0
    assert res.s[-1] == pytest.approx(474202.752, abs=1e-6)       # mm: the whole ring
