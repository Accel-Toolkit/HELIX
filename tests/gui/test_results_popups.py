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


def test_emittance_growth_kpi_is_the_normalised_growth(qapp):
    """Through an accelerating line (an RFQ: 30 keV -> 2.1 MeV) the
    geometric emittance shrinks (x0.14 here) while the normalised one
    grows (x1.17): the tile shows the normalised growth.  Results without
    a normalised emittance keep the geometric ratio."""
    import numpy as np
    from linac_gen_gui.interphase.state import AppState
    from linac_gen_gui.interphase.tabs.results_tab import ResultsTab

    class _R:
        sigma_x = np.array([1.0, 0.8])
        sigma_y = np.array([1.0, 0.6])
        emit_x = np.array([17.14, 2.405])
        emit_nx = np.array([0.137, 0.161])
        transmission = np.array([100.0, 77.35])

    class _NoNormalised(_R):
        emit_nx = np.zeros(2)

    _noop = lambda *a, **k: None
    tab = ResultsTab(AppState(), _noop, _noop, _noop)
    tab._refresh(_R())
    assert _kpi_val(tab._k_ex).startswith("1.18")       # 0.161 / 0.137
    tab._refresh(_NoNormalised())
    assert _kpi_val(tab._k_ex).startswith("0.14")       # 2.405 / 17.14


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


def _energy_results(dw_end_mev):
    """Three rows: the reference particle accelerated to 2.0985 MeV, the
    beam's centroid ΔW ending at ``dw_end_mev``."""
    import numpy as np

    class _R:
        s = [0.0, 100.0, 200.0]
        ref_w_kin = [0.03, 1.0, 2.0985]
        ref_gamma = [1.000032, 1.001065, 1.002234]
        transmission = [100.0, 99.0, 98.0]
    r = _R()
    r.centroid = np.zeros((3, 6))
    r.centroid[:, 5] = [0.0, 0.5 * dw_end_mev, dw_end_mev]
    return r


def _legend_names(plot):
    return [label.text for _sample, label in plot.getPlotItem().legend.items]


def test_energy_popup_draws_the_beam_mean_beside_the_reference(qapp):
    """The Energy tile's 2.0985 MeV is the reference particle's energy; the
    popup draws the beam's mean energy (reference + centroid ΔW) beside it,
    and lists it in the legend only while it has data."""
    import numpy as np
    from linac_gen_gui.interphase.tabs.results_tab import _EnergyPopup
    dlg = _EnergyPopup(None)
    try:
        r = _energy_results(0.0253)
        dlg.refresh(r)
        assert np.allclose(dlg._cw.getData()[1], r.ref_w_kin)
        x, y = dlg._cm.getData()
        assert np.allclose(x, r.s)
        assert np.allclose(y, np.asarray(r.ref_w_kin) + r.centroid[:, 5])
        assert _legend_names(dlg._pw) == ["reference particle", "beam mean"]
        # ΔW = 0 at every row (no RF, or no centroid recorded): the mean is
        # the reference itself, so it is not drawn a second time
        dlg.refresh(_energy_results(0.0))
        x, _y = dlg._cm.getData()
        assert x is None or len(x) == 0
        assert _legend_names(dlg._pw) == ["reference particle"]
        dlg.refresh(r)
        assert _legend_names(dlg._pw) == ["reference particle", "beam mean"]
        dlg.refresh(None)
        assert _legend_names(dlg._pw) == ["reference particle"]
    finally:
        dlg.deleteLater()


def test_energy_tile_tooltip_gives_the_reference_and_the_beam_mean(qapp):
    from linac_gen_gui.interphase.state import AppState
    from linac_gen_gui.interphase.tabs.results_tab import (
        ResultsTab, _energy_tooltip)
    tip = _energy_tooltip(_energy_results(0.0253))
    assert tip.startswith("reference particle  2.0985 MeV\n"
                          "beam mean  2.1238 MeV  (+25.3 keV)\n\n")
    for r in (None, _energy_results(0.0)):       # nothing to add
        assert "beam mean" not in _energy_tooltip(r)
    _noop = lambda *a, **k: None
    tab = ResultsTab(AppState(), _noop, _noop, _noop)
    try:
        card = next(c for c in tab._cards if c._key == "energy")
        tab._refresh(_energy_results(0.0253))
        assert card.toolTip() == tip
    finally:
        tab.deleteLater()


def test_result_card_tooltip_fn(qapp):
    """``tooltip_fn`` sets the tile's tooltip on every set_results, also
    with no results; one that raises leaves no tooltip."""
    from linac_gen_gui.interphase.panels.result_card import ResultCard
    card = ResultCard("energy", "Energy", "gauge", series_attr="ref_w_kin",
                      tooltip_fn=lambda r: "none" if r is None
                      else f"{len(r.s)} rows")
    card.set_results(None)
    assert card.toolTip() == "none"
    card.set_results(_energy_results(0.0253))
    assert card.toolTip() == "3 rows"

    def _raises(_r):
        raise RuntimeError("boom")
    bad = ResultCard("x", "X", "gauge", tooltip_fn=_raises)
    bad.setToolTip("stale")
    bad.set_results(None)
    assert bad.toolTip() == ""
    for c in (card, bad):
        c.deleteLater()
