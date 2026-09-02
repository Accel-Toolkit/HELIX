"""Aperture-loss popup: watts column + loss-table sourcing.

The popup must read the results-level ``loss_table`` (present on live
MP runs AND runs reloaded from HDF5) — the legacy ``results.beam``
reference exists only on live runs, so reloaded runs used to show
"No losses recorded" despite carrying the record.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from linac_gen.core.beam import LOSS_DTYPE
from linac_gen_gui.interphase.state import AppState
from linac_gen_gui.interphase.tabs.results_tab import _ApertureLossPopup


@dataclass
class _MockBeamConfig:
    species: str = "H-"
    current: float = 5.0
    frequency: float = 162.5
    duty_cycle: float = 100.0


class _MockResults:
    pass


def _results_with_losses():
    res = _MockResults()
    res.s = np.linspace(0.0, 3000.0, 10)
    res.transmission = np.linspace(100.0, 99.0, 10)
    # results-level table only — NO res.beam (the reloaded-run case)
    res.loss_table = np.array([
        (0, 500.0, 1.9, 0.0, 2.0, "SCRAPER1"),
        (1, 510.0, -2.0, 0.1, 2.0, "SCRAPER1"),
        (2, 2500.0, 0.0, 2.1, 10.0, "APER7"),
    ], dtype=LOSS_DTYPE)
    res.n_macro = 100
    return res


def test_popup_shows_watts_for_reloaded_results(qapp):
    state = AppState()
    state.set_beam_config(_MockBeamConfig())
    state.set_results(_results_with_losses())

    pop = _ApertureLossPopup(parent=None, state=state)
    pop.refresh(state.results)

    rows = [pop._summary.item(i).text()
            for i in range(pop._summary.count())]
    # 5 mA CW, N=100: 100 W per macro at 2 MeV, 500 W at 10 MeV —
    # APER7 (500 W) sorts above SCRAPER1 (2 x 100 = 200 W)
    assert rows and "APER7" in rows[0] and "500 W" in rows[0]
    assert "SCRAPER1" in rows[1] and "200 W" in rows[1]
    status = pop._status.text()
    assert "3 particle losses" in status
    assert "lost 700 W" in status
    assert "W/m" in status
    pop.close()


def test_popup_counts_only_when_current_is_zero(qapp):
    state = AppState()
    state.set_beam_config(_MockBeamConfig(current=0.0))
    state.set_results(_results_with_losses())

    pop = _ApertureLossPopup(parent=None, state=state)
    pop.refresh(state.results)
    rows = [pop._summary.item(i).text()
            for i in range(pop._summary.count())]
    assert rows and all("W" not in r.split("%")[-1] for r in rows)
    pop.close()
