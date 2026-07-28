# tests/gui/test_results_popups.py
"""Results-tab popup seams (module-level functions, no window)."""
import pytest


def test_beam_inputs_from_config_applies_mismatch():
    """2026-07 external review: the envelope OVERLAY seed dropped
    mismatch, producing an artifactual MP/envelope discrepancy in the
    comparison figure.  The seam now routes through the shared
    geometric_emittances helper."""
    from linac_gen.core.config import BeamConfig
    from linac_gen_gui.interphase.tabs.results_tab import (
        _beam_inputs_from_config)
    cfg0 = BeamConfig(species="proton", energy=3.0, frequency=352.21,
                      n_particles=10, emit_nx=0.25, emit_ny=0.25,
                      emit_z=0.3, beta_x=2.0, beta_y=2.0, beta_z=10.0)
    cfgm = BeamConfig(species="proton", energy=3.0, frequency=352.21,
                      n_particles=10, emit_nx=0.25, emit_ny=0.25,
                      emit_z=0.3, beta_x=2.0, beta_y=2.0, beta_z=10.0,
                      mismatch_x=50.0, mismatch_z=-20.0)
    _, i0 = _beam_inputs_from_config(cfg0)
    _, im = _beam_inputs_from_config(cfgm)
    assert im["emit_x"] / i0["emit_x"] == pytest.approx(1.5, rel=1e-12)
    assert im["emit_z"] / i0["emit_z"] == pytest.approx(0.8, rel=1e-12)
    assert im["emit_y"] == pytest.approx(i0["emit_y"], rel=1e-12)


def test_refresh_accepts_ndarray_results_kpis(qapp):
    """Regression: loading a saved result (assistant `load_results` or
    File->Load) yields numpy arrays, and `_refresh` did `if ex:` on them
    -> "truth value of an array is ambiguous" ValueError at the KPI block.
    A live run yields lists (never hit it); an HDF5 load yields ndarrays.
    _refresh must handle both and populate the growth/transmission KPIs."""
    import numpy as np
    from linac_gen_gui.interphase.state import AppState
    from linac_gen_gui.interphase.tabs.results_tab import ResultsTab

    class _R:                       # ndarray-valued result, as HDF5 load returns
        sigma_x = np.linspace(1.0, 2.0, 5)
        sigma_y = np.linspace(1.0, 1.5, 5)
        sigma_phi = np.linspace(3.0, 4.0, 5)
        emit_x = np.array([0.20, 0.21, 0.22, 0.23, 0.25])
        transmission = np.array([100.0, 99.5, 99.0, 98.5, 98.0])
        ref_beta = np.linspace(0.8, 0.9, 5)
        ref_frequency = 162.5

    _noop = lambda *a, **k: None
    tab = ResultsTab(AppState(), _noop, _noop, _noop)
    tab._refresh(_R())              # must not raise
    # emit-growth KPI computed from the ndarray (0.25/0.20 = 1.25x)
    assert "1.25" in _kpi_val(tab._k_ex)
    assert "98.00" in _kpi_val(tab._k_trans)


def test_refresh_via_set_results_signal_ndarray(qapp):
    """The real dispatch path the assistant uses: AppState.set_results ->
    results_changed -> ResultsTab._refresh, with ndarray-valued results."""
    import numpy as np
    from linac_gen_gui.interphase.state import AppState
    from linac_gen_gui.interphase.tabs.results_tab import ResultsTab

    class _R:
        sigma_x = np.linspace(1.0, 2.0, 4)
        sigma_y = np.linspace(1.0, 1.5, 4)
        sigma_phi = np.linspace(3.0, 4.0, 4)
        emit_x = np.array([0.20, 0.21, 0.22, 0.24])
        transmission = np.array([100.0, 99.0, 98.0, 97.0])

    _noop = lambda *a, **k: None
    state = AppState()
    tab = ResultsTab(state, _noop, _noop, _noop)
    state.set_results(_R())         # emits results_changed -> _refresh; must not raise
    assert _kpi_val(tab._k_sx) != "—"


def _kpi_val(card):
    """Read the value label written by kpi_set (child QLabel name='value')."""
    from PyQt6.QtWidgets import QLabel
    for c in card.findChildren(QLabel):
        if c.objectName() == "value":
            return c.text()
    return None
