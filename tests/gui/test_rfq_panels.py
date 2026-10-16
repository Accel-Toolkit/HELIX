"""The Results tab's RFQ section: ten panels for a lattice with RFQ cells
(vane geometry, cross-section, modulation, synchronous phase, focusing,
surface field, fields, losses, bunching, cell table).  The section shows
only for such a lattice; the panels draw from the lattice alone, add the
run at every cell exit when the results belong to the lattice, never build
a field map, and say why when they cannot show something."""
from __future__ import annotations

import dataclasses
import importlib.util
import warnings
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("PyQt6")

ROOT = Path(__file__).resolve().parents[2]
DEMO = ROOT / "examples" / "rfq_demo" / "rfq_demo.dat"
RFQ_KEYS = ["rfq_geometry", "rfq_section", "rfq_modulation", "rfq_phase",
            "rfq_focusing", "rfq_surface", "rfq_fields", "rfq_losses",
            "rfq_bunching", "rfq_table"]


def _noop(*_a, **_k):
    return None


def _parse(path):
    from linac_gen.io.tracewin_parser import parse_tracewin
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        lat, _ = parse_tracewin(str(path))
    return lat


def _demo_beam(n=300, current=0.0):
    from linac_gen.core.config import BeamConfig
    return BeamConfig(
        species="proton", energy=0.075, frequency=352.21, current=current,
        n_particles=n, distribution="gaussian", cutoff=4.0, emit_nx=0.20,
        alpha_x=0.0, beta_x=0.01, emit_ny=0.20, alpha_y=0.0, beta_y=0.01,
        emit_z=0.0, alpha_z=0.0, beta_z=1.0, continuous=True,
        dc_energy_spread_keV=0.0, periodic_phase=True)


def _run(lattice, cfg, solver):
    from linac_gen.core.simulation import Simulation
    from linac_gen.distributions.factory import create_beam
    lattice.step_config = dataclasses.replace(lattice.step_config,
                                              rfq_solver=solver)
    beam = create_beam(cfg, seed=11)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        rec = Simulation(lattice, beam, space_charge="off",
                         rfq_geometry="off").run()
    rec.beam = beam
    return rec


def _tab(lattice=None, cfg=None, results=None):
    from linac_gen_gui.interphase.state import AppState
    from linac_gen_gui.interphase.tabs.results_tab import ResultsTab
    st = AppState()
    if lattice is not None:
        st.set_lattice(lattice)
    if cfg is not None:
        st.set_beam_config(cfg)
    if results is not None:
        st.set_results(results)
    return st, ResultsTab(st, _noop, _noop, _noop)


def _open(tab, key):
    tab._open_popup(key)
    return tab._popups[key]


def _close(tab):
    for d in list(tab._popups.values()):
        d.close()
    tab.deleteLater()


def _curves(plot):
    """name -> (x, y) of the named curves in a plot."""
    out = {}
    for it in plot.getPlotItem().listDataItems():
        name = it.name()
        if name:
            x, y = it.getData()
            out[name] = (np.asarray(x, float), np.asarray(y, float))
    return out


def _bars(plot):
    import pyqtgraph as pg
    return [it for it in plot.getPlotItem().items
            if isinstance(it, pg.BarGraphItem)]


def _table_column(dlg, header):
    t = dlg._table
    for j in range(t.columnCount()):
        if t.horizontalHeaderItem(j).text() == header:
            return [t.item(i, j).text() for i in range(t.rowCount())]
    return None


@pytest.fixture
def no_map_cache(tmp_path, monkeypatch):
    """An empty field-map cache: no time-domain map exists."""
    from linac_gen.rfq_td import build
    monkeypatch.setenv("HELIX_CACHE_DIR", str(tmp_path / "cache"))
    build._MEMORY.clear()
    yield
    build._MEMORY.clear()


@pytest.fixture(scope="module")
def demo_run():
    """One per-cell run of the demo RFQ (no space charge, 300 particles)."""
    lat = _parse(DEMO)
    cfg = _demo_beam()
    return lat, cfg, _run(lat, cfg, "cells")


# ---------------------------------------------------------------- the section
def _visible(tab):
    return [c._key for c in tab._cards if not c.isHidden()]


def _rfq_header(tab):
    for hdr, _grid, cards in tab._tile_sections:
        if cards and all(c._key.startswith("rfq_") for c in cards):
            return hdr
    raise AssertionError("no RFQ section")


def test_rfq_section_follows_the_lattice(qapp, rfq_lattice, mini_lattice):
    st, tab = _tab()
    try:
        assert not any(k.startswith("rfq_") for k in _visible(tab))
        assert _rfq_header(tab).isHidden()
        st.set_lattice(rfq_lattice)
        assert [k for k in _visible(tab) if k.startswith("rfq_")] == RFQ_KEYS
        assert not _rfq_header(tab).isHidden()
        tab._search.setText("kilpatrick")
        assert _visible(tab) == ["rfq_surface"]
        tab._search.setText("vane")
        assert set(_visible(tab)) >= {"rfq_geometry", "rfq_section"}
        tab._search.clear()
        st.set_lattice(mini_lattice)
        assert not any(k.startswith("rfq_") for k in _visible(tab))
        assert _rfq_header(tab).isHidden()
        assert "rf_volt" in _visible(tab)          # other tiles untouched
    finally:
        _close(tab)


def test_rfq_tiles_are_in_the_plot_catalogue(qapp):
    st, tab = _tab()
    try:
        cat = dict(tab.plot_catalog())
        assert all(k in cat and cat[k].startswith("RFQ ") for k in RFQ_KEYS)
        labels = [lab for _k, lab in tab.plot_catalog()]
        assert len(labels) == len(set(labels))     # names resolve uniquely
        # opened without an RFQ (the assistant can do that): an empty state
        assert tab.open_plot("rfq_geometry")
        dlg = tab._popups["rfq_geometry"]
        assert not dlg._empty.isHidden()
        assert "No RFQ cells" in dlg._empty.text()
    finally:
        _close(tab)


# ------------------------------------------------------ lattice only, no run
def test_every_panel_draws_from_the_lattice_alone(qapp, no_map_cache,
                                                  gui_message_boxes):
    from linac_gen.analysis import rfq_design as rd
    lat = _parse(DEMO)
    st, tab = _tab(lat, _demo_beam())
    try:
        dlgs = {k: _open(tab, k) for k in RFQ_KEYS}
        assert gui_message_boxes == []

        g = dlgs["rfq_geometry"]
        top = _curves(g._top)
        assert {"+x vane tip", "−x vane tip", "r0"} <= set(top)
        assert "beam: largest |x|" not in top
        assert "RFQ_CELL cards" in g._note.text()
        assert "No run values yet." in g._note.text()

        cells = rd.find_rfq(lat).cells
        m = _curves(dlgs["rfq_modulation"]._m)["m"][1]
        np.testing.assert_allclose(m, [c.modulation for c in cells])
        assert "βλ/2 of the reference particle" not in \
            _curves(dlgs["rfq_modulation"]._len)

        f = dlgs["rfq_focusing"]
        assert "B = qVλ²/(mc²r0²) = 5.68" in f._note.text()
        assert "design β" in f._note.text()
        assert {"σ0 transverse", "σ0 longitudinal"} <= set(_curves(f._s))

        ph = dlgs["rfq_phase"]
        assert {"φ_s (cards)", "φ_s from the cell lengths",
                "design (β = 2L/λ)"} <= \
            set(_curves(ph._phi)) | set(_curves(ph._w))

        s = dlgs["rfq_surface"]                     # no map: says so
        assert not s._empty.isHidden() and "No map is cached" in s._empty.text()

        fl = dlgs["rfq_fields"]
        assert "cards' two-term field" in _curves(fl._ez)
        assert "not cached" in fl._note.text()

        for key in ("rfq_losses", "rfq_bunching"):
            assert not dlgs[key]._empty.isHidden(), key
        assert "run a multiparticle simulation" in dlgs["rfq_losses"]._empty.text()
        assert "No run values yet" in dlgs["rfq_bunching"]._empty.text()

        t = dlgs["rfq_table"]
        assert t._table.rowCount() == 199
        assert _table_column(t, "transmission [%]") is None
        assert _table_column(t, "name") == [c.name for c in cells]
        assert "No run values yet." in t._note.text()
    finally:
        _close(tab)


def test_panels_without_an_rf_frequency(qapp, rfq_lattice, no_map_cache,
                                        gui_message_boxes):
    """No beam configuration and no FREQ card: the frequency-dependent
    figures are left blank and say why; nothing raises."""
    st, tab = _tab(rfq_lattice)
    try:
        for key in RFQ_KEYS:
            _open(tab, key)
        assert gui_message_boxes == []
        f = tab._popups["rfq_focusing"]
        assert "B = qVλ²/(mc²r0²) = unknown (no RF frequency)" in f._note.text()
        assert _table_column(tab._popups["rfq_table"], "B") == [""] * 3
    finally:
        _close(tab)


def test_cross_section_slider_and_cell_box(qapp, no_map_cache):
    lat = _parse(DEMO)
    st, tab = _tab(lat, _demo_beam())
    try:
        d = _open(tab, "rfq_section")
        assert d._busy() is False          # the popup base's method intact
        d._cell.setValue(120)
        qapp.processEvents()
        from linac_gen.analysis import rfq_design as rd
        ch = rd.find_rfq(lat)
        z = d._slider.value() * d._STEP_MM
        assert ch.edges[119] <= z <= ch.edges[120]
        assert d._cell.value() == 120
        assert ch.names[119] in d._pos.text() and "cell 120" in d._pos.text()
        img = d._img.image
        assert img is not None and img.max() == 255 and img.min() == 0
        circles = _curves(d._plot)
        a = min(rd.tips_along(rd.cards_electrodes(lat, ch), [z])[k][0]
                for k in ("R_x", "R_y", "R_mx", "R_my"))
        assert f"smallest tip distance {a:.3f} mm" in circles
    finally:
        _close(tab)


def test_plots_switch_to_the_cell_number(qapp, no_map_cache, demo_run):
    from linac_gen.analysis import rfq_design as rd
    lat, cfg, rec = demo_run
    st, tab = _tab(lat, cfg, rec)
    try:
        ch = rd.find_rfq(lat)
        g = _open(tab, "rfq_geometry")
        lo = _open(tab, "rfq_losses")
        s_tip = _curves(g._top)["+x vane tip"]
        for d in (g, lo):
            assert not d._cell_axis.isHidden()
            d._cell_axis.setChecked(True)
        x, y = _curves(g._top)["+x vane tip"]
        np.testing.assert_array_equal(y, s_tip[1])          # same values
        np.testing.assert_allclose(
            x, np.interp(s_tip[0], np.r_[ch.s0[:1], ch.s1],
                         np.arange(len(ch.cells) + 1) + 0.5))
        r0 = _curves(g._top)["r0"][0]
        np.testing.assert_allclose(r0, np.arange(len(ch.cells) + 1) + 0.5)
        assert "cell" in g._top.getAxis("bottom").labelText
        assert g._lattice_strip.isHidden()                  # the strip is in s
        bar = _bars(lo._n)[0]
        np.testing.assert_allclose(bar.opts["x"], np.arange(1, len(ch.cells) + 1))
        assert not _curves(lo._xy)["loss point"][0].max() > 50   # x–y untouched
        st.set_results(rec)                                 # a refresh keeps it
        assert "cell" in g._top.getAxis("bottom").labelText
        g._cell_axis.setChecked(False)
        for _ in range(3):
            st.set_results(rec)
        # ticks in mm: pyqtgraph's SI scale untouched by the relabelling
        for p in (g._top, g._side, lo._n):
            assert p.getAxis("bottom").autoSIPrefixScale == 1.0
        assert g._top.getAxis("bottom").labelUnits == "mm"
        assert lo._n.getAxis("bottom").labelUnits == ""     # still in cells
        np.testing.assert_array_equal(_curves(g._top)["+x vane tip"][0],
                                      s_tip[0])
        assert not g._lattice_strip.isHidden()
        # no s axis, no switch
        assert _open(tab, "rfq_table")._cell_axis.isHidden()
        assert _open(tab, "rfq_section")._cell_axis.isHidden()
    finally:
        _close(tab)


def test_rf_voltage_and_synchronous_phase_include_rfq_cells(qapp):
    from linac_gen.analysis import rfq_design as rd
    from linac_gen_gui.interphase.tabs.results_tab import _build_series_fns
    lat = _parse(DEMO)
    st, tab = _tab(lat, _demo_beam())
    try:
        fns = _build_series_fns(st)
        x, v = fns["rf_volt"](None)
        assert len(v) == 199 and np.allclose(v, 0.085)        # 85 kV in MV
        _x, ph = fns["sync_phase"](None)
        np.testing.assert_allclose(ph, [c.phi_s_deg for c in
                                        rd.find_rfq(lat).cells])
        for key in ("rf_volt", "sync_phase"):
            d = _open(tab, key)
            items = d._plot.getPlotItem().listDataItems()
            assert items, key
    finally:
        _close(tab)


def test_an_edited_cell_redraws_an_open_panel(qapp, no_map_cache):
    from linac_gen.analysis import rfq_design as rd
    from linac_gen_gui.interphase.commands import ParamChangeCommand
    lat = _parse(DEMO)
    st, tab = _tab(lat, _demo_beam())
    try:
        d = _open(tab, "rfq_modulation")
        cell = rd.find_rfq(lat).cells[150]
        st.bus.do(ParamChangeCommand(cell, "modulation", cell.modulation, 2.5))
        import time
        t0 = time.time()
        while time.time() - t0 < 2.0:
            qapp.processEvents()
            if _curves(d._m)["m"][1][150] == 2.5:
                break
            time.sleep(0.02)
        assert _curves(d._m)["m"][1][150] == 2.5
    finally:
        _close(tab)


# ------------------------------------------------------------- after a run
def test_panels_after_a_per_cell_run(qapp, no_map_cache, demo_run,
                                     gui_message_boxes):
    from linac_gen.analysis import rfq_design as rd
    lat, cfg, rec = demo_run
    st, tab = _tab(lat, cfg, rec)
    try:
        dlgs = {k: _open(tab, k) for k in RFQ_KEYS}
        assert gui_message_boxes == []
        ch = rd.find_rfq(lat)
        rows = np.asarray(rec.element_exit_idx)[ch.index]

        top = _curves(dlgs["rfq_geometry"]._top)
        np.testing.assert_allclose(top["beam: largest |x|"][1],
                                   np.asarray(rec.x_max)[rows])
        assert "Run values at each cell exit." in dlgs["rfq_geometry"]._note.text()

        bl = _curves(dlgs["rfq_modulation"]._len)
        assert "βλ/2 of the reference particle" in bl

        trans = _table_column(dlgs["rfq_table"], "transmission [%]")
        np.testing.assert_allclose([float(v) for v in trans],
                                   np.asarray(rec.transmission)[rows],
                                   atol=0.006)
        assert "tracked by the per-cell model" in dlgs["rfq_table"]._note.text()

        lo = dlgs["rfq_losses"]
        n_lost = int(np.isin(rec.loss_table["element_name"], ch.names).sum())
        assert n_lost > 0
        assert lo._empty.isHidden()
        assert _bars(lo._n)[0].opts["height"].sum() == n_lost
        assert f"{n_lost} macroparticles lost in the RFQ" in lo._note.text()
        assert "per-cell model" in lo._note.text()

        bu = dlgs["rfq_bunching"]
        assert bu._empty.isHidden() and not bu._ps.isHidden()
        n_alive = int(rec.beam.alive_mask.sum())
        assert f"{n_alive} particles" in _curves(bu._ps)

        # the same run read back from its results file: same values
        from linac_gen.io.hdf5_output import (load_results_hdf5,
                                              save_results_hdf5)
        from linac_gen_gui.interphase.tabs.results_tab import _LoadedResults
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            p = str(Path(td) / "run.h5")
            save_results_hdf5(rec, p, beam_config=cfg)
            st.set_results(_LoadedResults(load_results_hdf5(p), p))
        assert gui_message_boxes == []
        trans2 = _table_column(dlgs["rfq_table"], "transmission [%]")
        assert trans2 == trans
        assert _bars(lo._n)[0].opts["height"].sum() == n_lost
    finally:
        _close(tab)


def test_results_of_another_lattice_are_not_mixed_in(qapp, no_map_cache,
                                                     demo_run, tmp_path):
    _lat, _cfg, rec = demo_run
    spec = importlib.util.spec_from_file_location(
        "tc", ROOT / "tests" / "rfq_td" / "test_chain.py")
    tc = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tc)
    other = _parse(tc._write_rfq(tmp_path))
    st, tab = _tab(other, _demo_beam())
    try:
        for key in ("rfq_geometry", "rfq_table", "rfq_bunching", "rfq_losses"):
            d = _open(tab, key)
            d.refresh(rec)                      # the demo's run, not this one
            text = d._note.text() + d._empty.text()
            assert "do not belong to this lattice" in text, key
    finally:
        _close(tab)


def test_envelope_results_leave_the_particle_panels_empty(qapp, no_map_cache):
    from linac_gen.tracking.envelope import EnvelopeResults
    lat = _parse(DEMO)
    env = EnvelopeResults()
    env.s = [0.0, 100.0]
    env.sigma_x = [1.0, 1.0]
    st, tab = _tab(lat, _demo_beam(), env)
    try:
        for key in RFQ_KEYS:
            _open(tab, key)
        assert not tab._popups["rfq_losses"]._empty.isHidden()
        assert tab._popups["rfq_table"]._table.rowCount() == 199
    finally:
        _close(tab)


# ------------------------------------- the time-domain solver and its map
@pytest.fixture
def coarse_map(tmp_path, monkeypatch):
    """A coarse field map in a per-test cache, built in-process (as the
    time-domain tests do: a worker process would import linac_gen afresh,
    which a developer checkout may resolve elsewhere)."""
    from linac_gen.rfq_td import build
    monkeypatch.setenv("HELIX_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setenv("HELIX_RFQTD_WORKERS", "1")
    for k, v in dict(h=0.2, x_max=10.0, pad_mm=10.0, upstream_mm=12.0,
                     downstream_mm=6.0).items():
        monkeypatch.setitem(build.DEFAULT_SETTINGS, k, v)
    build._MEMORY.clear()
    yield
    build._MEMORY.clear()


def test_surface_and_fields_from_the_time_domain_map(qapp, tmp_path,
                                                     coarse_map,
                                                     gui_message_boxes):
    pytest.importorskip("pyamg")
    from linac_gen.analysis import rfq_design as rd
    spec = importlib.util.spec_from_file_location(
        "tc", ROOT / "tests" / "rfq_td" / "test_chain.py")
    tc = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tc)
    lat = _parse(tc._write_rfq(tmp_path))
    from linac_gen.core.config import BeamConfig
    cfg = BeamConfig(species="proton", energy=0.050, frequency=162.5,
                     current=0.0, n_particles=400, distribution="gaussian",
                     cutoff=3.0, emit_nx=0.02, alpha_x=0.0, beta_x=0.05,
                     emit_ny=0.02, alpha_y=0.0, beta_y=0.05, emit_z=0.0,
                     alpha_z=0.0, beta_z=1.0, continuous=True,
                     dc_energy_spread_keV=0.0)
    st, tab = _tab(lat, cfg)
    try:
        # before any time-domain run: no map, none built by the panels
        s = _open(tab, "rfq_surface")
        _open(tab, "rfq_fields")
        assert "No map is cached" in s._empty.text()
        assert not list((tmp_path / "cache").rglob("map-*.npz"))
        assert not getattr(lat, "_rfq_td_notes", None)   # notes left to runs

        lat.step_config = dataclasses.replace(lat.step_config,
                                              rfq_td_steps_per_period=24)
        rec = _run(lat, cfg, "time_domain")
        assert rec.rfq_solver_used == "time_domain"
        st.set_results(rec)
        assert gui_message_boxes == []

        assert s._empty.isHidden()
        e = _curves(s._e)
        v_over_r0 = tc.V / tc.R0 * 1e-3                    # MV/m
        for name in ("±x vane tips", "±y vane tips"):
            peak = np.nanmax(e[name][1])
            assert 1.1 < peak / v_over_r0 < 1.5, (name, peak)
        ek = rd.kilpatrick_field(162.5)
        assert f"Kilpatrick {ek:.2f} MV/m at 162.5 MHz" in e
        assert "Peak" in s._note.text()

        fl = tab._popups["rfq_fields"]
        ez = _curves(fl._ez)
        assert "time-domain field map" in ez
        assert ("Solid: the time-domain solver's field map of the vane file's "
                "vanes") in fl._note.text()
        assert fl._prof.isHidden()                 # no vane-profile cache

        lo = _open(tab, "rfq_losses")
        if lo._empty.isHidden():
            assert "time-domain solver" in lo._note.text()
        bu = _open(tab, "rfq_bunching")       # a 5 mm drift after the RFQ
        assert not bu._ps.isHidden() and "5.0 mm of drift" in bu._note.text()
        tb = _open(tab, "rfq_table")
        assert "tracked by the time-domain solver" in tb._note.text()
        assert "Time domain: 400 particles" in tb._note.text()
    finally:
        _close(tab)
