"""Loss-power popup (Results tab): W/m profile + exit-plane W/cm² map.

Covers the four states the popup must handle: a live MP run (both
plots), a run reloaded from HDF5 (profile only — no beam reference),
an envelope run (no loss record at all) and a zero-current run.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from linac_gen.core.beam import LOSS_DTYPE, Beam
from linac_gen.core.particle import H_MINUS
from linac_gen.core.reference import ReferenceParticle
from linac_gen_gui.interphase.state import AppState
from linac_gen_gui.interphase.tabs.results_tab import _LossPowerPopup


@dataclass
class _MockBeamConfig:
    species: str = "H-"
    current: float = 5.0
    frequency: float = 162.5
    duty_cycle: float = 100.0


class _MockResults:
    pass


def _loss_table():
    return np.array([
        (0, 500.0, 1.9, 0.0, 2.0, "SCRAPER1"),
        (1, 510.0, -2.0, 0.1, 2.0, "SCRAPER1"),
        (2, 2500.0, 0.0, 2.1, 10.0, "APER7"),
    ], dtype=LOSS_DTYPE)


def _base_results():
    res = _MockResults()
    res.s = np.linspace(0.0, 3000.0, 10)
    res.transmission = np.linspace(100.0, 97.0, 10)
    res.ref_w_kin = np.full(10, 800.0)
    res.loss_table = _loss_table()
    res.n_macro = 100
    return res


def _with_beam(res, n=200):
    ref = ReferenceParticle(species=H_MINUS, w_kin=800.0, frequency=650.0)
    beam = Beam(ref=ref, n_particles=n, current=5.0)
    rng = np.random.default_rng(3)
    beam.particles[:, 0] = rng.normal(0.0, 3.0, n)
    beam.particles[:, 2] = rng.normal(0.0, 2.0, n)
    # deliberate energy offset + spread: the heat map must use
    # PER-PARTICLE energies (W_ref + dW), not the reference alone
    beam.particles[:, 5] = 5.0 + rng.normal(0.0, 1.0, n)
    res.beam = beam
    return res


def test_live_run_fills_profile_and_plane_map(qapp):
    state = AppState()
    state.set_beam_config(_MockBeamConfig())
    state.set_results(_with_beam(_base_results()))

    pop = _LossPowerPopup(parent=None, state=state)
    pop.refresh(state.results)

    xs, ys = pop._curve.getData()
    assert xs is not None and len(xs) == len(ys) + 1   # step edges
    assert ys.max() > 0                                # losses drawn
    assert pop._image.image is not None                # heat map drawn
    txt = pop._status.text()
    # 5 mA CW, N=100 -> 2 MeV macro = 100 W, 10 MeV macro = 500 W
    assert "lost 700 W" in txt
    assert "W/m" in txt and "W/cm²" in txt
    assert "I_avg = 5 mA" in txt

    # the map must be built from PER-PARTICLE energies (W_ref + dW):
    # the +5 MeV offset makes a reference-energy-only map differ
    from linac_gen.analysis.loss_power import plane_power_density
    alive = state.results.beam.alive_particles
    want, _xe, _ye = plane_power_density(
        alive[:, 0], alive[:, 2], energy_mev=800.0 + alive[:, 5],
        current_mA=5.0, duty_pct=100.0, n_macro=100, bins=64)
    assert np.allclose(pop._image.image, want.T)
    ref_only, _a, _b = plane_power_density(
        alive[:, 0], alive[:, 2], energy_mev=800.0,
        current_mA=5.0, duty_pct=100.0, n_macro=100, bins=64)
    assert not np.allclose(want, ref_only)      # the offset is visible
    pop.close()


def test_reloaded_run_keeps_profile_without_beam(qapp):
    state = AppState()
    state.set_beam_config(_MockBeamConfig())
    state.set_results(_base_results())            # no .beam attribute

    pop = _LossPowerPopup(parent=None, state=state)
    pop.refresh(state.results)

    _xs, ys = pop._curve.getData()
    assert ys is not None and ys.max() > 0
    assert pop._image.image is None
    assert "exit-plane map needs the final distribution" in pop._status.text()
    pop.close()


def test_losses_beyond_last_recorder_step_are_binned(qapp):
    """The histogram range must cover every recorded loss: a loss past
    the last recorder step would otherwise vanish from the profile."""
    state = AppState()
    state.set_beam_config(_MockBeamConfig())
    res = _base_results()
    res.s = np.linspace(0.0, 1000.0, 5)           # ends BEFORE the 2500 mm loss
    state.set_results(res)

    pop = _LossPowerPopup(parent=None, state=state)
    pop.refresh(state.results)
    _xs, ys = pop._curve.getData()
    # all three losses accounted: 2x100 W in bin 0, 500 W near 2.5 m
    assert float(np.sum(ys)) == 700.0              # W/m x 1 m bins
    assert "lost 700 W" in pop._status.text()
    pop.close()


def test_envelope_results_explain_themselves(qapp):
    state = AppState()
    state.set_beam_config(_MockBeamConfig())
    res = _MockResults()
    res.s = np.linspace(0.0, 3000.0, 10)
    res.ref_w_kin = np.full(10, 800.0)
    state.set_results(res)                         # no loss_table/n_macro

    pop = _LossPowerPopup(parent=None, state=state)
    pop.refresh(state.results)
    txt = pop._status.text()
    assert "multi-particle" in txt
    # older result files predate loss recording — say so,
    # rather than implying the user never ran MP
    assert "re-run" in txt
    pop.close()


def test_zero_current_is_refused_clearly(qapp):
    state = AppState()
    state.set_beam_config(_MockBeamConfig(current=0.0))
    state.set_results(_with_beam(_base_results()))

    pop = _LossPowerPopup(parent=None, state=state)
    pop.refresh(state.results)
    assert "current is zero" in pop._status.text()
    pop.close()


def test_card_is_registered_and_opens_via_the_real_click_path(qapp):
    """The tile must exist in the catalogue AND route through the same
    click path a user (or the assistant's open_plot) takes — direct
    construction alone would not prove the card is wired up."""
    from PyQt6.QtCore import QObject, pyqtSignal
    from linac_gen_gui.interphase.tabs.results_tab import ResultsTab

    catalog = dict(ResultsTab.plot_catalog(None))
    assert catalog.get("loss_power") == "Loss power (W/m · W/cm²)"

    class _Card(QObject):
        clicked = pyqtSignal(str)

        def __init__(self, key):
            super().__init__()
            self._key = key

        def isEnabled(self):
            return True

    fake = type("F", (), {})()
    clicks = []
    card = _Card("loss_power")
    card.clicked.connect(clicks.append)
    fake._cards = [card]
    assert ResultsTab.open_plot(fake, "loss_power") is True
    assert clicks == ["loss_power"]


def test_tile_series_provider_totals_lost_power(qapp):
    """The tile's footer value is the cumulative lost power, so the card
    reads the run's TOTAL loss in watts at a glance."""
    from linac_gen_gui.interphase.tabs.results_tab import _build_series_fns

    state = AppState()
    state.set_beam_config(_MockBeamConfig())
    res = _base_results()
    fn = _build_series_fns(state)["loss_power"]
    xs, ys = fn(res)
    assert len(xs) == len(ys)
    assert ys[-1] == pytest.approx(700.0)      # 2x100 W + 500 W
    assert np.all(np.diff(ys) >= 0.0)          # cumulative

    # envelope / zero-current / no-loss-record -> placeholder, never 0 W
    assert fn(None) is None
    state.set_beam_config(_MockBeamConfig(current=0.0))
    assert _build_series_fns(state)["loss_power"](res) is None


def test_provider_arrays_work_for_reloaded_numpy_results(qapp):
    """Regression: the `_arr` helper used `getattr(...) or []`, whose
    truth test RAISES on a numpy array — so every provider-based tile
    silently showed its "—" placeholder for runs reloaded from HDF5
    (numpy fields) while working for live runs (list fields)."""
    from linac_gen_gui.interphase.tabs.results_tab import _build_series_fns

    class _R:
        pass

    state = AppState()
    state.set_beam_config(_MockBeamConfig())
    fns = _build_series_fns(state)

    numpy_res, list_res = _R(), _R()
    numpy_res.s = np.linspace(0.0, 1000.0, 5)
    numpy_res.transmission = np.linspace(100.0, 98.0, 5)
    numpy_res.ref_w_kin = np.full(5, 800.0)
    list_res.s = list(numpy_res.s)
    list_res.transmission = list(numpy_res.transmission)
    list_res.ref_w_kin = list(numpy_res.ref_w_kin)

    for key in ("aperture_loss", "power"):
        a = np.asarray(fns[key](numpy_res), dtype=float)
        b = np.asarray(fns[key](list_res), dtype=float)
        assert a.size == 5 and np.allclose(a, b)   # identical either way
