"""Results tab "BMAG along s" tile: the Courant–Snyder mismatch factor of the
run's own Twiss at every record against the design (error-free envelope of
the current lattice and Beam-tab config), per plane, along s.

Pins: BMAG equals ``mismatch_factor`` row by row (raw and dispersion-
corrected); records with no design row within 0.5 mm and DC records
(longitudinal) are NaN; the popup opens through the real card, follows
the Raw / Dispersion-corrected choice, exports BMAG_x/y/z, reports the
design status, starts the background probe on a cache miss and shows a
clear message without a lattice; a real FODO run gives BMAG ≈ 1 at the
entrance.
"""
from __future__ import annotations

import os

import numpy as np
import pytest

pytest.importorskip("PyQt6")

FODO = os.path.abspath(os.path.join(
    os.path.dirname(__file__), "..", "..", "examples", "fodo_cell.dat"))


@pytest.fixture(autouse=True)
def _clean_cache():
    import linac_gen_gui.interphase.tabs.results_tab as rt
    rt._COMPANION_PROBE.clear()
    yield
    rt._COMPANION_PROBE.clear()


def _noop(*_a, **_k):
    return None


def _sig(ax, bx, ay=0.0, by=1.0, az=0.0, bz=2.0, disp_x=0.0):
    from linac_gen.tracking.envelope import _build_sigma_matrix
    return _build_sigma_matrix(ax, bx, 1.0, ay, by, 1.0, az, bz, 0.1, disp_x=disp_x)


def _results(rows, continuous_at=None):
    """A DiagnosticRecorder-like run: rows = [(s_mm, Σ)]."""
    from linac_gen.diagnostics.recorder import DiagnosticRecorder
    rec = DiagnosticRecorder()
    rec.s = [float(s) for s, _ in rows]
    rec.sigma_matrix = [S for _, S in rows]
    rec.element_names = [f"E{i}" for i in range(len(rows))]
    if continuous_at is not None:
        rec.continuous_at = list(continuous_at)
    return rec


def _env(rows, n_elements):
    from linac_gen.tracking.envelope import EnvelopeResults
    env = EnvelopeResults()
    for s, S in rows:
        env.s.append(float(s)); env.sigma_matrix.append(S)
    env.element_exit_idx = list(range(1, n_elements + 1))
    return env


def _fodo(n_cells=3):
    from linac_gen.core.lattice import Lattice
    from linac_gen.elements.drift import Drift
    from linac_gen.elements.quadrupole import Quadrupole
    lat = Lattice()
    for _ in range(n_cells):
        lat.add(Drift(name="D", length=100.0, aperture=10.0))
        lat.add(Quadrupole(name="QF", length=50.0, gradient=+10.0, aperture=10.0))
        lat.add(Drift(name="D", length=100.0, aperture=10.0))
        lat.add(Quadrupole(name="QD", length=50.0, gradient=-10.0, aperture=10.0))
    return lat


def _cfg(**kw):
    from linac_gen.core.config import BeamConfig
    base = dict(species="H-", energy=2.12, frequency=162.5, n_particles=10,
                emit_nx=0.25, emit_ny=0.25, emit_z=0.3, beta_x=1.0, beta_y=1.0, beta_z=2.0)
    base.update(kw)
    return BeamConfig(**base)


# ------------------------------------------------------------- unit (no Qt)
def test_bmag_along_s_matches_mismatch_factor_and_gaps():
    import linac_gen_gui.interphase.tabs.results_tab as rt
    from linac_gen.analysis.phase_advance import mismatch_factor
    beam = [(0.0, _sig(0.0, 1.0, 0.0, 1.0, 0.0, 2.0)), (100.0, _sig(0.5, 1.5, -0.3, 0.8, 1.0, 4.0)),
            (200.0, _sig(-1.0, 3.0, 0.2, 1.2, 0.0, 2.0)), (300.0, _sig(0.0, 2.0))]
    design = [(0.0, _sig(0.0, 1.0, 0.0, 1.0, 0.0, 2.0)), (100.0, _sig(0.0, 1.0, 0.0, 1.0, 0.0, 2.0)),
              (200.3, _sig(0.3, 2.0, 0.0, 1.0, 0.5, 3.0)), (450.0, _sig(0.0, 1.0))]
    out = rt._bmag_along_s(_results(beam), _env(design, 3))
    assert out["s_mm"].tolist() == [0.0, 100.0, 200.0, 300.0]
    want_x = [mismatch_factor(0.0, 1.0, 0.0, 1.0), mismatch_factor(0.5, 1.5, 0.0, 1.0),
              mismatch_factor(-1.0, 3.0, 0.3, 2.0)]
    want_y = [1.0, mismatch_factor(-0.3, 0.8, 0.0, 1.0), mismatch_factor(0.2, 1.2, 0.0, 1.0)]
    want_z = [1.0, mismatch_factor(1.0, 4.0, 0.0, 2.0), mismatch_factor(0.0, 2.0, 0.5, 3.0)]
    assert out["x"][:3] == pytest.approx(want_x, rel=1e-12)
    assert out["y"][:3] == pytest.approx(want_y, rel=1e-12)
    assert out["z"][:3] == pytest.approx(want_z, rel=1e-12)
    assert np.isnan(out["x"][3]) and np.isnan(out["y"][3]) and np.isnan(out["z"][3])   # 300 vs 450: no row
    assert out["x"][0] == 1.0 and out["x"][1] > 1.0
    # DC records: no longitudinal value, transverse unchanged
    out_dc = rt._bmag_along_s(_results(beam, continuous_at=[True] * 4), _env(design, 3))
    assert np.isnan(out_dc["z"][:3]).all()
    assert out_dc["x"][:3] == pytest.approx(want_x, rel=1e-12)


def test_bmag_along_s_dispersion_corrected_mode():
    """A dispersive record: the raw projected Twiss is inflated by D·σ_δ so
    raw BMAG > 1 against a dispersion-free design, while the betatron
    (dispersion-corrected) BMAG is exactly 1."""
    import linac_gen_gui.interphase.tabs.results_tab as rt
    beam = [(0.0, _sig(0.3, 2.0, disp_x=0.0)), (100.0, _sig(0.3, 2.0, disp_x=5.0))]
    design = [(0.0, _sig(0.3, 2.0)), (100.0, _sig(0.3, 2.0))]
    raw = rt._bmag_along_s(_results(beam), _env(design, 2), betatron=False)
    bet = rt._bmag_along_s(_results(beam), _env(design, 2), betatron=True)
    assert raw["x"][0] == pytest.approx(1.0, abs=1e-12) and bet["x"][0] == pytest.approx(1.0, abs=1e-12)
    assert raw["x"][1] > 1.001
    assert bet["x"][1] == pytest.approx(1.0, abs=1e-9)
    assert rt._bmag_along_s(_results([]), _env(design, 2)) is None


# ------------------------------------------------------------------- popup
def _tab_with(rec, lat, cfg, env=None):
    import linac_gen_gui.interphase.tabs.results_tab as rt
    from linac_gen_gui.interphase.state import AppState
    from linac_gen_gui.interphase.tabs.results_tab import ResultsTab
    st = AppState()
    if lat is not None:
        st.set_lattice(lat)
    if cfg is not None:
        st.set_beam_config(cfg)
    st.set_results(rec)
    if env is not None:
        rt._COMPANION_PROBE.clear()
        rt._COMPANION_PROBE.update({"sig": rt._probe_sig(lat, cfg), "results": env})
    tab = ResultsTab(st, _noop, _noop, _noop)
    return st, tab


def test_card_exists_and_popup_through_real_card(qapp, gui_message_boxes):
    import linac_gen_gui.interphase.tabs.results_tab as rt
    lat, cfg = _fodo(1), _cfg()
    beam = [(0.0, _sig(0.0, 1.0)), (100.0, _sig(0.4, 1.3, disp_x=4.0)), (300.0, _sig(-0.2, 0.9))]
    design = [(0.0, _sig(0.0, 1.0)), (100.0, _sig(0.0, 1.0)), (150.0, _sig(0.1, 1.1)), (300.0, _sig(0.0, 1.0))]
    st, tab = _tab_with(_results(beam), lat, cfg, _env(design, 4))
    try:
        assert "bmag_s" in {getattr(c, "_key", None) for c in tab._cards}
        assert tab.open_plot("bmag_s")
        dlg = tab._popups["bmag_s"]
        assert gui_message_boxes == []
        want = rt._bmag_along_s(st.results, rt._COMPANION_PROBE["results"])
        x, y = dlg._cx.getData()
        assert np.asarray(x).tolist() == [0.0, 100.0, 300.0]
        np.testing.assert_allclose(y, want["x"], rtol=1e-12)
        assert y[1] > 1.001                                      # raw: dispersive record
        assert "design" in dlg._info.text() and "3 of 3" in dlg._info.text()
        dlg._mode_combo.setCurrentIndex(1)                       # dispersion-corrected
        _, yb = dlg._cx.getData()
        from linac_gen.analysis.phase_advance import mismatch_factor
        assert yb[1] == pytest.approx(mismatch_factor(0.4, 1.3, 0.0, 1.0), rel=1e-9)   # optics-only mismatch
        assert y[1] > yb[1]                                      # raw includes the dispersive part
        names = {str(c.name) for pn in dlg._collect_panels() for c in (getattr(pn, "curves", None) or [])}
        assert {"BMAG_x", "BMAG_y", "BMAG_z"} <= names
        assert dlg._probe_worker is None
    finally:
        tab.deleteLater()


def test_no_lattice_message_and_probe_on_cache_miss(qapp):
    import linac_gen_gui.interphase.tabs.results_tab as rt
    from linac_gen_gui.interphase.tabs.results_tab import _BmagPopup
    beam = [(0.0, _sig(0.0, 1.0)), (300.0, _sig(0.2, 1.4))]
    st, tab = _tab_with(_results(beam), None, _cfg())
    dlg = _BmagPopup(tab, st)
    try:
        dlg.refresh(st.results)
        assert "no lattice" in dlg._info.text()
        assert dlg._probe_worker is None
        assert dlg._cx.getData()[0] is None or len(dlg._cx.getData()[0]) == 0
    finally:
        dlg.close(); tab.deleteLater()
    lat = _fodo(1)                                               # 300 mm, 4 elements
    st, tab = _tab_with(_results(beam), lat, _cfg(current=0.0))
    dlg = _BmagPopup(tab, st)
    try:
        dlg.refresh(st.results)
        assert dlg._probe_worker is not None
        assert "computing" in dlg._info.text()
        assert dlg._probe_worker.wait(120000)
        for _ in range(3):
            qapp.processEvents()
        assert rt._COMPANION_PROBE.get("sig") == rt._probe_sig(lat, st.beam_config)
        x, y = dlg._cx.getData()
        assert len(x) == 2 and np.isfinite(y).all() and (y >= 1.0 - 1e-12).all()
        assert y[0] == pytest.approx(1.0, abs=1e-9)             # entrance: design = input Twiss
    finally:
        dlg.close(); tab.deleteLater()


def test_real_window_fodo_run(win):
    win.open_lattice(FODO)
    win.beam_tab._npart.setValue(300)
    win.beam_tab._apply(); win.pump()
    win._run_mp(); win.wait_worker(win._mp_worker)
    assert win.results_tab.open_plot("bmag_s")
    dlg = win.results_tab._popups["bmag_s"]
    win.wait_popup_idle(dlg); win.pump(0.3)
    x, y = dlg._cx.getData()
    fin = np.isfinite(np.asarray(y, float))
    assert fin.sum() >= 2
    assert (np.asarray(y, float)[fin] >= 1.0 - 1e-9).all()
    assert np.asarray(y, float)[fin][0] < 1.1                     # starts matched (300-particle sampling)
    assert "design" in dlg._info.text()
    assert win.message_boxes == []
