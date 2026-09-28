"""Results-tab "Halo action scan" window (tabs/halo_scan_popup.py).

Driven through the real seams: ResultsTab._open_popup / open_plot, a live
MP recorder, the HDF5 writer + loader + _LoadedResults adapter, and — end
to end — the Numerics-tab checkbox -> win._run_mp -> auto-dumped file ->
Results > Import.
"""
from __future__ import annotations

import os
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("PyQt6")
h5py = pytest.importorskip("h5py")

from linac_gen.core.beam import Beam                          # noqa: E402
from linac_gen.core.config import SpaceChargeConfig           # noqa: E402
from linac_gen.core.lattice import Lattice                    # noqa: E402
from linac_gen.core.particle import PROTON                    # noqa: E402
from linac_gen.core.reference import ReferenceParticle        # noqa: E402
from linac_gen.core.simulation import Simulation              # noqa: E402
from linac_gen.elements.drift import Drift                    # noqa: E402
from linac_gen.elements.quadrupole import Quadrupole          # noqa: E402
from linac_gen.elements.rf_gap import RFGap                   # noqa: E402

FODO = os.path.abspath(os.path.join(
    os.path.dirname(__file__), "..", "..", "examples", "fodo_cell.dat"))
_noop = lambda *a, **k: None                                  # noqa: E731


def _lattice():
    lat = Lattice()
    for i in range(4):
        lat.add(Drift(f"D{i}a", length=120.0, aperture=8.0))
        lat.add(Quadrupole(f"QF{i}", length=60.0, gradient=14.0,
                           aperture=8.0))
        lat.add(Drift(f"D{i}b", length=120.0, aperture=8.0))
        lat.add(Quadrupole(f"QD{i}", length=60.0, gradient=-14.0,
                           aperture=8.0))
    return lat


def _run(scan=True, n=3000, seed=2):
    ref = ReferenceParticle(species=PROTON, w_kin=3.0, frequency=162.5)
    beam = Beam(ref=ref, n_particles=n, current=3.0)
    rng = np.random.default_rng(seed)
    p = rng.standard_normal((n, 6)) * np.array([1.4, 1.1, 1.4, 1.1, 8.0,
                                                 0.004])
    p[:, 0] += 60.0 * p[:, 5]                     # some dispersion
    beam.particles[:] = p
    lat = _lattice()
    rec = Simulation(lat, beam, space_charge=SpaceChargeConfig(
        nx=12, ny=12, nz=12), record_action_scan=scan).run()
    rec.beam = beam
    return lat, rec


def _dc_run():
    lat = Lattice()
    for i in range(3):
        lat.add(Drift(f"DA{i}", length=100.0, aperture=50.0))
    lat.add(RFGap("GAP", voltage=0.01, phase=-90.0, frequency=162.5))
    for i in range(3):
        lat.add(Drift(f"DB{i}", length=100.0, aperture=50.0))
    ref = ReferenceParticle(species=PROTON, w_kin=0.03, frequency=162.5)
    beam = Beam(ref=ref, n_particles=1500, current=0.0)
    beam.continuous = True
    rng = np.random.default_rng(7)
    for c, s in ((0, 2.0), (1, 5.0), (2, 2.0), (3, 5.0)):
        beam.particles[:, c] = rng.normal(0, s, 1500)
    beam.particles[:, 4] = rng.uniform(-180.0, 180.0, 1500)
    beam.particles[:, 5] = rng.normal(0, 1e-5, 1500)
    return lat, Simulation(lat, beam, space_charge="off",
                           record_action_scan=True).run()


def _tab(lat, res):
    from linac_gen_gui.interphase.state import AppState
    from linac_gen_gui.interphase.tabs.results_tab import ResultsTab
    state = AppState()
    state.set_lattice(lat, path=None)
    state.set_results(res)
    return ResultsTab(state, _noop, _noop, _noop)


def _open(tab):
    from linac_gen_gui.interphase.tabs.halo_scan_popup import (
        HaloActionScanPopup)
    tab._open_popup("halo_scan")
    pop = tab._popups["halo_scan"]
    assert isinstance(pop, HaloActionScanPopup)
    return pop


# --------------------------------------------------------------------------
def test_live_run_through_the_results_tab(qapp, gui_message_boxes):
    lat, rec = _run()
    tab = _tab(lat, rec)
    assert ("halo_scan", "Halo action scan (n·ε_rms)") in tab.plot_catalog()
    pop = _open(tab)
    assert not pop._hint.isVisible()
    img = pop._image.image
    assert img is not None and img.ndim == 2 and np.isfinite(img).any()
    s_lvl, n_lvl = pop._level_curves[1.0].getData()
    assert len(s_lvl) == len(rec.s) and np.isfinite(n_lvl).any()
    assert np.isfinite(pop._eps_curve.getData()[1]).all()
    assert pop._slice_curve.getData()[0] is not None
    info = pop._info.text()
    assert "ε_rms" in info and "fewer than 1 % outside" in info
    assert "dispersion-corrected" in info

    pop._plane_cb.setCurrentIndex(2)                         # φ–W
    assert "deg·MeV" in pop._info.text()
    assert not pop._mode_cb.isEnabled()
    pop._plane_cb.setCurrentIndex(0)
    pop._mode_cb.setCurrentIndex(0)                          # raw
    assert pop._plane() == "x_raw" and "(raw)" in pop._info.text()

    first = pop._step_label.text()
    pop._slider.setValue(len(rec.s) - 1)
    assert pop._step_label.text() != first
    assert pop._cursor.value() == pytest.approx(rec.s[-1])

    txt = pop._hover_text(float(rec.s[3]), 9.3)
    assert "outside n = 9.25" in txt and "ε_rms,n" in txt
    assert "outside the recorded range" in pop._hover_text(0.0, 1e4)

    assert tab.open_plot("halo_scan") is True
    assert gui_message_boxes == []
    pop.close()


def test_empty_states_say_why(qapp, gui_message_boxes, tmp_path):
    from types import SimpleNamespace

    from linac_gen.io.hdf5_output import load_results_hdf5, save_results_hdf5
    from linac_gen_gui.interphase.tabs.results_tab import _LoadedResults
    lat, rec_no = _run(scan=False, n=500)
    tab = _tab(lat, None)
    pop = _open(tab)
    assert pop._hint.isVisible() and "No results yet" in pop._hint.text()

    pop.refresh(rec_no)
    assert "did not record the action scan" in pop._hint.text()
    fp = tmp_path / "run_mp.h5"
    save_results_hdf5(rec_no, str(fp))
    pop.refresh(_LoadedResults(load_results_hdf5(str(fp)), str(fp)))
    assert "results file has no action scan" in pop._hint.text()
    pop.refresh(SimpleNamespace(s=[0.0, 1.0]))            # envelope-like
    assert "multi-particle runs only" in pop._hint.text()
    pop.refresh(SimpleNamespace(s=[0.0], n_macro=10,
                                source_path=str(tmp_path / "a.opmd.h5")))
    assert "openPMD" in pop._hint.text()
    pop.refresh(SimpleNamespace(s=[0.0], n_macro=10, direction="backward"))
    assert "Backtracked" in pop._hint.text()
    assert pop._image.image is None or not np.isfinite(
        pop._image.image).any()
    assert gui_message_boxes == []
    pop.close()


def test_loaded_file_shows_the_same_map_as_the_live_run(qapp, tmp_path):
    from linac_gen.io.hdf5_output import load_results_hdf5, save_results_hdf5
    from linac_gen_gui.interphase.tabs.results_tab import _LoadedResults
    lat, rec = _run()
    tab = _tab(lat, rec)
    pop = _open(tab)
    live = {p: np.array(pop._plane_view(p)["img"]) for p in ("x", "z")}
    fp = tmp_path / "run_mp.h5"
    save_results_hdf5(rec, str(fp))
    pop.refresh(_LoadedResults(load_results_hdf5(str(fp)), str(fp)))
    assert not pop._hint.isVisible()
    for p, img in live.items():
        np.testing.assert_array_equal(pop._plane_view(p)["img"], img)
    pop.close()


def test_dc_records_masked_in_the_longitudinal_plane(qapp):
    lat, rec = _dc_run()
    tab = _tab(lat, rec)
    pop = _open(tab)
    pop._plane_cb.setCurrentIndex(2)
    view = pop._plane_view("z")
    dc = np.asarray(rec.continuous_at)
    assert dc.any() and (~dc).any()
    assert np.isnan(view["eps_n"][dc]).all()
    assert np.isfinite(view["eps_n"][~dc]).any()
    assert np.isnan(view["n_max"][dc]).all()
    pop._slider.setValue(0)                                   # a DC record
    assert "DC (unbunched)" in pop._info.text()
    xview = pop._plane_view("x")                              # x unaffected
    assert np.isfinite(xview["eps_n"][dc]).all()
    pop.close()


def test_betatron_emittance_matches_the_results_tab_schur(qapp):
    from linac_gen_gui.interphase.tabs.results_tab import (
        _betatron_sigma_blocks)
    _lat, rec = _run()
    sxx, sxxp, sxpxp, *_ = _betatron_sigma_blocks(np.asarray(rec.sigma_matrix))
    np.testing.assert_allclose(rec.action_scan["eps_x"],
                               np.sqrt(sxx * sxpxp - sxxp ** 2), rtol=1e-12)


def test_3d_view_renders_offscreen_and_only_when_visible(qapp, monkeypatch):
    lat, rec = _run()
    tab = _tab(lat, rec)
    pop = _open(tab)
    pop.show()
    assert pop._ax3d is None                     # nothing built until viewed
    pop._tabs.setCurrentWidget(pop._page3d)
    qapp.processEvents()
    assert pop._ax3d is not None and not pop._dirty_3d
    assert len(pop._ax3d.collections) == 1       # one surface
    labels = [t.get_text() for t in pop._ax3d.get_zticklabels()]
    assert "100 %" in labels and "1 %" in labels
    draws = []
    monkeypatch.setattr(pop._canvas, "draw", lambda: draws.append(1))
    pop._tabs.setCurrentWidget(pop._map_page)
    pop._plane_cb.setCurrentIndex(1)
    assert pop._dirty_3d and draws == []         # hidden: marked, not drawn
    pop._tabs.setCurrentWidget(pop._page3d)
    assert draws == [1] and not pop._dirty_3d
    pop.close()


def test_ctrl_s_exports_the_exact_counts(qapp, tmp_path):
    lat, rec = _run()
    tab = _tab(lat, rec)
    pop = _open(tab)
    panels = {p.label: p for p in pop._collect_panels()}
    exact = panels["halo_action_scan_x"]
    counts = [i for i in exact.images if i.name.startswith("count_outside")][0]
    np.testing.assert_array_equal(counts.data,
                                  np.asarray(rec.action_scan["count_x"]))
    names = {c.name for c in exact.curves}
    assert {"eps_rms_normalized_mm_mrad", "n_max_outermost", "n_alive",
            "n_grid"} <= names
    out = tmp_path / "scan.npz"
    pop._write_npz(out, list(panels.values()))
    keys = np.load(out).files
    assert any(k.startswith("halo_action_scan_x__count_outside") for k in keys)
    pop.close()


def test_map_columns_pool_steps_and_hold_the_last_one(qapp):
    """Steps sharing a display column are max-pooled; the empty columns
    after them hold the LAST of those steps, not the pooled maximum (a
    one-step halo spike must not smear over the following drift)."""
    from types import SimpleNamespace

    from linac_gen.diagnostics.action_scan import PLANES
    from linac_gen_gui.interphase.tabs.halo_scan_popup import (
        HaloActionScanPopup)
    counts = np.array([[1000, 50, 5, 0],          # s = 0.00
                       [1000, 500, 200, 100],     # s = 0.01: the spike
                       [1000, 100, 10, 0],        # s = 0.02
                       [1000, 60, 6, 1]], np.int32)   # s = 100 mm
    s = [0.0, 0.01, 0.02, 100.0]
    scan = {"n": np.array([0.0, 1.0, 2.0, 3.0]),
            "n_alive": np.full(4, 1000), "continuous": np.zeros(4, bool)}
    for p in PLANES:
        scan[f"count_{p}"] = counts
        for key, val in (("eps", 1.0), ("eps_n", 1.0), ("n_max", 3.0)):
            scan[f"{key}_{p}"] = np.full(4, val)
    pop = HaloActionScanPopup(None, None)
    pop.refresh(SimpleNamespace(s=s, action_scan=scan, n_macro=1000))
    view = pop._plane_view("x")
    img, cap = view["img"], view["cap"]
    row = int(2.5 / cap * img.shape[0])           # n = 2.5 -> grid 2
    pct = lambda c: np.log10(100.0 * c / 1000.0)  # noqa: E731
    assert img[row, 0] == pytest.approx(pct(200), abs=1e-6)     # pooled
    assert img[row, 500] == pytest.approx(pct(10), abs=1e-6)    # held
    assert img[row, -1] == pytest.approx(pct(6), abs=1e-6)      # last step
    pop.close()


def test_empty_map_cells_render_transparent(qapp):
    """Cells with no particle outside are NaN and must render with alpha 0
    (not the lowest colour) — pins the pyqtgraph behaviour relied on."""
    lat, rec = _run()
    tab = _tab(lat, rec)
    pop = _open(tab)
    img = pop._image
    img.render()
    q = img.qimage
    data = img.image
    rows, cols = data.shape
    r_nan, c_nan = np.argwhere(np.isnan(data))[0]
    r_ok, c_ok = np.argwhere(np.isfinite(data))[0]
    assert (q.width(), q.height()) == (cols, rows)
    assert q.pixelColor(int(c_nan), int(r_nan)).alpha() == 0
    assert q.pixelColor(int(c_ok), int(r_ok)).alpha() == 255
    pop.close()


def test_mp_worker_forwards_the_flag(qapp, monkeypatch):
    from linac_gen.core import simulation as sim_mod
    from linac_gen_gui.interphase.workers import MultiparticleWorker

    seen = {}

    class _Fake:
        def __init__(self, *a, **k):
            seen.update(k)

        def run(self):
            return type("R", (), {})()

    monkeypatch.setattr(sim_mod, "Simulation", _Fake)
    w = MultiparticleWorker(None, type("B", (), {"ref": None})(), None,
                            record_action_scan=True)
    assert w.stackSize() == 16 * 1024 * 1024
    w._run_locked()
    assert seen.get("record_action_scan") is True


def test_end_to_end_checkbox_run_dump_and_reimport(qapp, win, tmp_path,
                                                   monkeypatch):
    """Numerics checkbox -> _run_mp -> the auto-dumped HELIX file carries
    action_scan/ -> the window shows it -> Results > Import of that file
    shows it again."""
    from PyQt6.QtWidgets import QFileDialog

    from linac_gen_gui.interphase.tabs.results_tab import _LoadedResults
    win.open_lattice(FODO)
    win.beam_tab._npart.setValue(400)
    win.beam_tab._apply()
    win.pump()
    win.convergence_tab._record_action_scan.setChecked(True)
    win._run_mp()
    win.wait_worker(win._mp_worker)
    res = win.state.results
    assert type(res).__name__ == "DiagnosticRecorder"
    assert len(res.action_scan["count_x"]) == len(res.s)

    dumps = [p for p in tmp_path.glob("*_mp.h5")]
    assert len(dumps) == 1
    with h5py.File(dumps[0], "r") as f:
        assert f["action_scan"]["count_x"].shape[0] == len(res.s)

    assert win.results_tab.open_plot("halo_scan")
    pop = win.results_tab._popups["halo_scan"]
    assert not pop._hint.isVisible() and pop._scan is not None

    monkeypatch.setattr(QFileDialog, "getOpenFileName",
                        staticmethod(lambda *a, **k: (str(dumps[0]), "")))
    win.results_tab._import_results()
    win.pump()
    assert isinstance(win.state.results, _LoadedResults)
    assert pop._scan is not None and not pop._hint.isVisible()
    assert pop._scan.s.size == len(res.s)
    assert win.message_boxes == []
    pop.close()


def _fake(counts, n_alive, s, *, n_max=3.0, planes=None, **extra):
    """Dict-form scan (what load_results_hdf5 returns) on a tiny grid."""
    from types import SimpleNamespace

    from linac_gen.diagnostics.action_scan import PLANES
    grid = np.arange(np.asarray(counts).shape[1], dtype=float)
    k = len(s)
    scan = {"n": grid, "n_alive": np.asarray(n_alive),
            "continuous": np.zeros(k, bool)}
    for p in planes or PLANES:
        scan[f"count_{p}"] = np.asarray(counts, dtype=np.int32)
        for key, val in (("eps", 1.0), ("eps_n", 1.0), ("n_max", n_max)):
            scan[f"{key}_{p}"] = np.full(k, val)
    return SimpleNamespace(s=list(s), action_scan=scan,
                           n_macro=int(max(n_alive)), **extra)


def test_auto_n_max_rule(qapp):
    """auto = clip(1.1 * p95(outermost n), 20, grid max)."""
    lat, rec = _run()
    tab = _tab(lat, rec)
    pop = _open(tab)
    n_max = np.asarray(rec.action_scan["n_max_x"], float)
    want = float(np.clip(1.1 * np.percentile(n_max, 95.0), 20.0, 400.0))
    assert pop._plane_view("x")["cap"] == pytest.approx(want)
    pop._nmax.setValue(35.0)
    assert pop._plane_view("x")["cap"] == 35.0
    pop.close()


def test_readout_sentences_per_case(qapp):
    from linac_gen_gui.interphase.tabs.halo_scan_popup import (
        HaloActionScanPopup)
    # 500 particles: 0.01 % is below one particle -> "not resolved";
    # 30 % stay outside the whole grid -> "more than ... even at n = 3"
    pop = HaloActionScanPopup(None, None)
    pop.refresh(_fake([[500, 400, 300, 150]], [500], [0.0]))
    txt = pop._info.text()
    assert "0.01 % level: not resolved with 500 particles" in txt
    assert "more than 1 % outside even at n = 3" in txt
    assert "beyond n = beyond" not in txt
    assert "beyond n = not resolved" not in txt
    pop.refresh(_fake([[5000, 40, 3, 0]], [5000], [0.0]))
    assert "fewer than 1 % outside beyond n = 1" in pop._info.text()
    pop.close()


def test_empty_state_hides_every_overlay(qapp):
    lat, rec = _run()
    _lat2, rec_no = _run(scan=False, n=500)
    tab = _tab(lat, rec)
    pop = _open(tab)
    assert pop._cursor.isVisible() or not pop.isVisible()
    pop.refresh(rec_no)
    assert pop._hint.isVisible()
    for item in (pop._cursor, pop._eps_cursor, pop._slice_nmax):
        assert not item.isVisible()
    pop.refresh(rec)                                   # and back
    assert not pop._hint.isVisible()
    assert pop._slider.maximum() == len(rec.s) - 1
    pop.close()


def test_view_cache_is_bounded_and_released_on_close(qapp):
    lat, rec = _run()
    tab = _tab(lat, rec)
    pop = _open(tab)
    pop._slider.setValue(5)
    s_kept = pop._step_label.text()
    for cap in (10.0, 15.0, 20.0, 25.0, 30.0):
        pop._nmax.setValue(cap)
    pop._plane_cb.setCurrentIndex(2)
    assert len(pop._views) <= 3                        # one per plane
    assert pop._nmax.keyboardTracking() is False
    pop.close()
    assert pop._scan is None and pop._views == {} and pop._results is None
    tab._open_popup("halo_scan")                       # reopen: data back
    assert pop._scan is not None and pop._step_label.text() == s_kept
    pop.close()


def test_readouts_never_force_the_window_wider(qapp):
    from PyQt6.QtWidgets import QSizePolicy
    lat, rec = _run()
    tab = _tab(lat, rec)                     # keep the parent alive
    pop = _open(tab)
    for label in (pop._hover, pop._step_label):
        assert label.sizePolicy().horizontalPolicy() == \
            QSizePolicy.Policy.Ignored
    pop.close()


def test_missing_plane_is_reported_not_raised(qapp):
    from linac_gen_gui.interphase.tabs.halo_scan_popup import (
        HaloActionScanPopup)
    pop = HaloActionScanPopup(None, None)
    pop.refresh(_fake([[100, 50, 5, 0]] * 3, [100] * 3, [0.0, 1.0, 2.0],
                      planes=("x", "y", "z")))
    # never shown: isVisible() is False for every child, so ask isHidden()
    assert pop._hint.isHidden()
    pop._mode_cb.setCurrentIndex(0)                    # raw: not recorded
    assert not pop._hint.isHidden() and "has no x – x′ (raw)" in \
        pop._hint.text()
    pop._on_step_changed(2)                            # slots stay quiet
    assert "no x – x′ (raw) data" in pop._hover_text(1.0, 1.0)
    pop._mode_cb.setCurrentIndex(1)
    assert pop._hint.isHidden() and pop._slider.maximum() == 2
    pop.close()


def test_periodic_phase_note_only_when_known_false(qapp, tmp_path):
    from linac_gen.io.hdf5_output import load_results_hdf5, save_results_hdf5
    from linac_gen_gui.interphase.tabs.results_tab import _LoadedResults
    lat, rec = _dc_run()
    tab = _tab(lat, rec)
    pop = _open(tab)
    pop._plane_cb.setCurrentIndex(2)
    pop._slider.setValue(len(rec.s) - 1)               # a bunched record
    note = "without periodic phase coordinates"
    assert note in pop._info.text()                    # live: False
    fp = tmp_path / "dc.h5"
    save_results_hdf5(rec, str(fp))
    pop.refresh(_LoadedResults(load_results_hdf5(str(fp)), str(fp)))
    pop._slider.setValue(len(rec.s) - 1)
    assert note in pop._info.text()                    # stored in the file
    rec.periodic_phase = True
    save_results_hdf5(rec, str(fp))
    pop.refresh(_LoadedResults(load_results_hdf5(str(fp)), str(fp)))
    pop._slider.setValue(len(rec.s) - 1)
    assert note not in pop._info.text()
    unknown = _fake([[100, 50, 5, 0]] * 2, [100] * 2, [0.0, 1.0])
    unknown.action_scan["continuous"] = np.array([True, False])
    pop.refresh(unknown)                               # old file: unknown
    pop._slider.setValue(1)
    assert note not in pop._info.text()
    pop.close()


def test_click_selects_the_nearest_record(qapp):
    from linac_gen_gui.interphase.tabs.halo_scan_popup import (
        HaloActionScanPopup)
    pop = HaloActionScanPopup(None, None)
    pop.refresh(_fake([[100, 50, 5, 0]] * 4, [100] * 4,
                      [0.0, 10.0, 10.0, 20.0]))
    assert pop._nearest_record(9.99) == 2              # last at s = 10
    assert pop._nearest_record(14.0) == 2
    assert pop._nearest_record(16.0) == 3
    assert pop._record_at(19.99) == 2                  # hover: at/before
    pop.close()


def test_export_carries_the_dc_and_undefined_flags(qapp):
    lat, rec = _dc_run()
    tab = _tab(lat, rec)                     # keep the parent alive
    pop = _open(tab)
    pop._plane_cb.setCurrentIndex(2)
    panel = [p for p in pop._collect_panels()
             if p.label == "halo_action_scan_z"][0]
    curves = {c.name: c.y for c in panel.curves}
    np.testing.assert_array_equal(curves["continuous_dc"],
                                  np.asarray(rec.continuous_at, float))
    assert (curves["no_rms_ellipse"] >= curves["continuous_dc"]).all()
    pop.close()


def test_unreadable_scan_is_reported_as_such(qapp, tmp_path):
    """A file whose scan cannot be read must not claim the run was made
    without it."""
    from linac_gen.io.hdf5_output import load_results_hdf5, save_results_hdf5
    from linac_gen_gui.interphase.tabs.halo_scan_popup import (
        HaloActionScanPopup)
    from linac_gen_gui.interphase.tabs.results_tab import _LoadedResults
    _lat, rec = _run(n=500)
    fp = tmp_path / "newer.h5"
    save_results_hdf5(rec, str(fp))
    with h5py.File(fp, "a") as f:
        f["action_scan"].attrs["schema_version"] = 2
    with pytest.warns(RuntimeWarning):
        loaded = _LoadedResults(load_results_hdf5(str(fp)), str(fp))
    pop = HaloActionScanPopup(None, None)
    pop.refresh(loaded)
    txt = pop._hint.text()
    assert "could not be read" in txt and "schema_version 2" in txt
    assert "made without" not in txt
    pop.close()


def test_position_and_colour_bar_through_an_empty_state(qapp):
    lat, rec = _run()
    tab = _tab(lat, rec)
    pop = _open(tab)
    pop._slider.setValue(9)
    kept = pop._step_label.text()
    pop.refresh(SimpleNamespace(s=[0.0, 1.0]))          # e.g. a preview
    assert not pop._cbar.isVisible()
    pop.refresh(rec)
    assert pop._step_label.text() == kept and pop._cbar.isVisible()
    pop.close()


def test_map_and_emittance_strip_share_the_s_axis(qapp):
    """Same left/right pixel edges and the same s range (the colour bar
    column is reserved on the strip too)."""
    lat, rec = _run()
    tab = _tab(lat, rec)
    pop = _open(tab)
    for size in ((1200, 860), (1000, 700)):
        pop.resize(*size)
        for _ in range(10):
            qapp.processEvents()

        def edges(pw):
            r = pw.getPlotItem().vb.sceneBoundingRect()
            a = pw.mapTo(pop, pw.mapFromScene(r.topLeft()))
            b = pw.mapTo(pop, pw.mapFromScene(r.bottomRight()))
            return a.x(), b.x()
        (ml, mr), (el, er) = edges(pop._map), edges(pop._eps_plot)
        assert abs(ml - el) <= 1 and abs(mr - er) <= 1, (size, ml, mr, el, er)
        np.testing.assert_allclose(pop._map.getPlotItem().vb.viewRange()[0],
                                   pop._eps_plot.getPlotItem().vb.viewRange()[0])
    pop.close()
