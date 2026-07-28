"""Smoke test for the IBS popup in the Results tab."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

from linac_gen_gui.interphase.state import AppState
from linac_gen_gui.interphase.tabs.results_tab import _IbsPopup


@dataclass
class _MockBeamConfig:
    species: str = "H-"
    current: float = 5.0
    frequency: float = 162.5
    duty_cycle: float = 100.0


class _MockResults:
    pass


def _make_synthetic_results(n: int = 20):
    res = _MockResults()
    s = np.linspace(0.0, 5000.0, n)
    res.s = s
    res.sigma_x = np.full(n, 1.5)
    res.sigma_y = np.full(n, 1.5)
    res.sigma_phi = np.full(n, 5.0)
    res.sigma_w = np.full(n, 0.001)
    res.ref_beta = np.linspace(0.5, 0.85, n)
    res.ref_gamma = 1.0 / np.sqrt(1.0 - res.ref_beta ** 2)
    res.ref_w_kin = (res.ref_gamma - 1.0) * 939.294308
    res.ref_frequency = np.full(n, 162.5)
    sm = np.zeros((n, 6, 6))
    sm[:, 1, 1] = 0.5 ** 2
    sm[:, 3, 3] = 0.5 ** 2
    sm[:, 5, 5] = 0.001 ** 2
    res.sigma_matrix = sm
    res.mass_mev = 939.294308
    res.transmission = np.full(n, 100.0)
    res.beta_x = np.full(n, 1.0)
    return res


def test_ibs_popup_opens_and_refreshes(qapp):
    state = AppState()
    state.set_beam_config(_MockBeamConfig())
    state.set_results(_make_synthetic_results())

    pop = _IbsPopup(parent=None, state=state)
    pop.refresh(state.results)
    # Each curve should now hold something.
    for c in (pop._c_rate, pop._c_pwr, pop._c_iL, pop._c_iP):
        x, y = c.getData()
        assert x is not None and len(x) > 0

    # Toggling the non-constant xs checkbox should not crash and should
    # leave the curves populated.
    pop._chk_nonconst.setChecked(True)
    qapp.processEvents()
    for c in (pop._c_rate, pop._c_pwr, pop._c_iL, pop._c_iP):
        x, y = c.getData()
        assert x is not None and len(x) > 0


def test_ibs_popup_on_proton_beam_shows_message(qapp):
    state = AppState()
    state.set_beam_config(_MockBeamConfig(species="proton"))
    state.set_results(_make_synthetic_results())

    pop = _IbsPopup(parent=None, state=state)
    pop.refresh(state.results)
    assert "H⁻" in pop._summary.text()
    for c in (pop._c_rate, pop._c_pwr, pop._c_iL, pop._c_iP):
        x, y = c.getData()
        assert x is None or len(x) == 0


def test_ibs_popup_independent_duty_factor(qapp):
    """Changing BeamConfig.duty_cycle should not move the popup spinner."""
    state = AppState()
    state.set_beam_config(_MockBeamConfig(duty_cycle=100.0))
    state.set_results(_make_synthetic_results())

    pop = _IbsPopup(parent=None, state=state)
    initial = pop._duty_spin.value()
    # Mutate the BeamConfig duty cycle — popup must NOT mirror it.
    state.beam_config.duty_cycle = 5.0
    assert pop._duty_spin.value() == initial
