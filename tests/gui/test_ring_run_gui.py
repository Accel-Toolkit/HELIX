"""Ring mode in the GUI, milestone M2: a run through the Ring tab's Run
button with the ring kernels — the live strip, the run in the Results tab
(phase space with the separatrix, losses, tunes), live statistics, the
results file written to the calc directory, and the run settings in the
project."""
from __future__ import annotations

import copy
import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("PyQt6")

ROOT = Path(__file__).resolve().parents[2]
FIXTURE = ROOT / "tests" / "ring" / "fixtures" / "simple_booster.madx"


def _n(curve) -> int:
    """Points of a curve (all of them, not the clipped, thinned display)."""
    return 0 if curve.xData is None else len(curve.xData)


def _ring_on(win, particles=300, matched=True):
    """Ring mode on, the Beam tab's beam of ``particles`` (a run takes it)
    and, with ``matched``, its Twiss those of the beam matched to the ring
    (Ring & Optics → Use the matched beam in the Beam tab)."""
    act = win._toolbar._ring_action
    if not act.isChecked():
        act.trigger()
    win.pump(0.1)
    tab = win.ring_tab
    win.beam_tab._npart.setValue(particles)
    if matched:
        tab._to_beam_btn.click()
        win.wait_worker(tab._matched_worker)
        assert tab._optics_status.text().startswith("Beam tab: the beam matched"), \
            tab._optics_status.text()
    return tab


def test_a_run_fills_the_strip_and_the_results_tab_and_writes_its_file(win, tmp_path):
    # (the GUI tests' calc directory is the test's tmp_path)
    from linac_gen.ring.results import load_ring_hdf5
    win.open_lattice(FIXTURE)
    tab = _ring_on(win)
    tab._turns.setValue(40)
    tab._snapshots.setText("0, 20, last")
    assert not tab._to_results_btn.isEnabled()
    tab._run_btn.click()
    win.wait_worker(tab._worker)
    res = win.state.ring_results
    assert res is not None and res.turns_done == 40 and res.meta["device"] == "cpu"
    assert sorted(res.snapshots) == [0, 20, 40]
    strip = tab._live_strip
    # (xData: the whole curve — what is drawn is clipped to the view and thinned)
    assert len(strip.c_surv.xData) == 41 and len(strip.c_ex.xData) == 41
    assert "Survival 100 % (none lost)" in tab._summary.text()
    # the Beam tab's beam, matched to the ring
    assert res.meta["n_particles"] == 300 and res.meta["config"]["source"] == "beam"
    assert res.meta["start_mismatch"] == pytest.approx([1.0, 1.0, 1.0], abs=1e-3)
    assert "At the start, BMAG (1 when matched) x 1.0000, y 1.0000, z 1.0000." \
        in tab._summary.text()
    # the run is shown in the Results tab, which "Show in the Results tab" brings up
    rt = win.results_tab
    assert rt._source == "ring" and not rt._run_combo.isHidden()
    assert tab._to_results_btn.isEnabled()
    tab._to_results_btn.click()
    assert win._tabs.currentWidget() is rt
    assert rt.open_plot("ring_phase")
    d = rt._popups["ring_phase"]
    assert [d.snap_combo.itemText(i) for i in range(d.snap_combo.count())] == ["0", "20", "40"]
    assert d.phase_note.text().startswith("Turn 40: 300 particles")
    assert len(d.c_sep_up.getData()[0]) > 0           # (the RF bucket's separatrix)
    d.mode.setCurrentIndex(d.mode.findData("dots"))
    assert len(d.d_z._scatter.getData()[0]) == 300
    assert rt.open_plot("ring_tunes")
    t = rt._popups["ring_tunes"].tunes_text.text()
    assert "NAFF" in t and "Qx" in t
    assert not [m for m in win.message_boxes if "refresh failed" in str(m)]
    # the run is written to the calc directory, ring schema
    files = sorted(Path(tmp_path).glob("*_ring.h5"))
    assert len(files) == 1 and str(files[0]) in tab._summary.text()
    back = load_ring_hdf5(files[0])
    assert np.array_equal(back.cov, res.cov) and back.meta["config"]["n_turns"] == 40


def test_live_statistics_and_stop(win, tmp_path):
    win.open_lattice(FIXTURE)
    tab = _ring_on(win, particles=2000)
    tab._turns.setValue(1_000_000)
    tab._probes.setValue(0)
    tab._run_btn.click()
    assert win.pump_until(lambda: _n(tab._live_strip.c_surv) > 3, 120_000)
    assert tab._worker.isRunning()                    # (live, not the final results)
    live_ex0 = float(tab._live_strip.c_ex.yData[0])
    tab._stop_btn.click()
    win.wait_worker(tab._worker)
    res = win.state.ring_results
    assert res is not None and res.truncated and 0 < res.turns_done < 1_000_000
    assert "stopped after" in tab._summary.text()
    # (the live emittance had the run's reference: the results' own value)
    assert live_ex0 == res.emit_normalized[0][0] * 1e6
    # (a stopped run is written too)
    assert len(sorted(Path(tmp_path).glob("*_ring.h5"))) == 1


def test_the_run_settings_go_into_the_project(win):
    win.open_lattice(FIXTURE)
    tab = _ring_on(win, matched=False)
    assert [tab._source.itemData(i) for i in range(tab._source.count())] == ["beam", "empty"]
    tab._probes.setValue(3)
    tab._snapshots.setText("0, 7")
    tab._snapshots.editingFinished.emit()
    data = win._collect_project_dict()
    cfg = data["ring"]["config"]
    # (the particles are the Beam tab's: the project's beam holds them)
    assert cfg["source"] == "beam" and "n_particles" not in cfg
    assert cfg["tune_probes"] == 3 and cfg["snapshot_turns"] == [0, 7]
    assert data["beam"]["n_particles"] == 300
    tab._probes.setValue(1)
    win._apply_project_dict(data, silent=True)
    t = win.ring_tab
    assert t._source.currentData() == "beam" and t._probes.value() == 3
    assert t._snapshots.text() == "0, 7"
    assert t._beam_line.text().startswith("300 particles of the Beam tab: gaussian")


def test_a_project_whose_run_started_matched_takes_the_beam_tabs_beam(win):
    # an earlier version's run could start from a Gaussian matched to the ring
    # with its own number of particles: the run now takes the Beam tab's beam,
    # and the Run page says so until the next run
    win.open_lattice(FIXTURE)
    tab = _ring_on(win, particles=200, matched=False)
    data = copy.deepcopy(win._collect_project_dict())
    data["ring"]["config"].update({"source": "matched", "n_particles": 5000, "n_turns": 3})
    win._apply_project_dict(copy.deepcopy(data), silent=True)
    t = win.ring_tab
    cfg = win.state.ring_config
    assert cfg["source"] == "beam" and "n_particles" not in cfg
    assert t._source.currentData() == "beam"
    note = "This project's ring run started from a beam of 5000 particles matched to the ring."
    assert note in t._beam_line.text()
    assert "n_particles" not in win._collect_project_dict()["ring"]["config"]
    # (no source then meant the matched one — with or without its particles;
    # a project of this version always holds the start)
    data["ring"]["config"].pop("source")
    win._apply_project_dict(copy.deepcopy(data), silent=True)
    assert note in win.ring_tab._beam_line.text()
    data["ring"]["config"].pop("n_particles")
    win._apply_project_dict(copy.deepcopy(data), silent=True)
    assert "This project's ring run started from a beam matched to the ring." \
        in win.ring_tab._beam_line.text()
    assert win._collect_project_dict()["ring"]["config"]["source"] == "beam"
    win._apply_project_dict(win._collect_project_dict(), silent=True)
    assert "started from a beam" not in win.ring_tab._beam_line.text()
    # the note goes with the next run
    data["ring"]["config"].update({"source": "matched", "n_particles": 5000})
    win._apply_project_dict(copy.deepcopy(data), silent=True)
    t = win.ring_tab
    t._probes.setValue(0)
    t._run_btn.click()
    assert "started from a beam" not in t._beam_line.text()
    win.wait_worker(t._worker)
    assert win.state.ring_results.meta["n_particles"] == 200


def test_a_matched_projects_note_reaches_a_ring_tab_made_after_it(win):
    # a project loaded before ring mode made its tab: the tab says it when made
    win.open_lattice(FIXTURE)
    data = copy.deepcopy(win._collect_project_dict())
    data["ring"] = {"enabled": True, "config": {"source": "matched", "n_particles": 700}}
    assert win.ring_tab is None
    win._apply_project_dict(data, silent=True)
    win.pump(0.1)
    assert win.ring_tab is not None and win._ring_load_note is None
    assert ("This project's ring run started from a beam of 700 particles matched to the "
            "ring.") in win.ring_tab._beam_line.text()
    # (a linac project next: the note goes)
    plain = {k: v for k, v in data.items() if k != "ring"}
    win._apply_project_dict(plain, silent=True)
    assert "started from a beam" not in win.ring_tab._beam_line.text()


def test_the_losses_window(win):
    # a run with a tight aperture (through the engine), shown in the Results
    # tab's "Ring losses"
    from linac_gen.cli.common import load_lattice
    from linac_gen.core.config import BeamConfig
    from linac_gen.elements.aperture import Aperture
    from linac_gen.ring import RingConfig, RingSimulation
    lat = load_lattice(str(FIXTURE))
    lat.elements.insert(20, Aperture(name="tight", dx=3.0, dy=3.0,
                                     aperture_type=Aperture.CIRCULAR))
    b = BeamConfig(species="proton", energy=800.0, frequency=44.7044096, current=0.0,
                   emit_nx=1.5, emit_ny=1.5, emit_z=5.0, beta_z=100.0, cutoff=4.0)
    res = RingSimulation(copy.deepcopy(lat), b, RingConfig(n_turns=4, n_particles=1000,
                                                           source="matched",
                                                           tune_probes=0)).run()
    win.open_lattice(FIXTURE)
    _ring_on(win)
    win.state.set_ring_results(res)
    rt = win.results_tab
    assert rt.open_plot("ring_losses")
    d = rt._popups["ring_losses"]
    n_lost = res.losses["pid"].size
    assert n_lost > 0
    assert int(np.sum(d.b_loss_s.opts["height"])) == n_lost
    h, x = np.asarray(d.b_loss_t.opts["height"]), np.asarray(d.b_loss_t.opts["x"])
    assert int(h.sum()) == n_lost
    # (a loss in turn t — counted from 1 — is drawn at t, where survival drops)
    drops = np.flatnonzero(np.diff(res.n_alive)) + 1
    assert set(x[h > 0].astype(int)) == set(res.turns[drops].astype(int))
    assert "tight" in d.loss_note.text()
    # (along s in m, under the ring's lattice strip)
    assert d._lattice_strip is not None and not d._lattice_strip.isHidden()
    assert float(np.max(d.b_loss_s.opts["x"])) < float(res.meta["length_m"])
    assert not [m for m in win.message_boxes if "refresh failed" in str(m)]


@pytest.mark.skipif(sys.platform != "darwin", reason="the CUDA item is greyed on macOS only")
def test_cuda_is_greyed_on_macos(win):
    win.open_lattice(FIXTURE)
    tab = _ring_on(win)
    item = tab._device.model().item(1)
    assert not item.isEnabled() and "macOS" in item.toolTip()


def test_live_rows_are_appended_with_their_reference(qapp):
    # the worker sends only the rows added since its last update, with the
    # reference their emittances need: a fresh strip draws them right
    from linac_gen_gui.interphase.panels.ring_plots import RingLiveStrip
    d = RingLiveStrip()

    def rows(start, k, n_alive):
        cov = np.zeros((k, 6, 6))
        cov[:, 0, 0], cov[:, 1, 1] = 4e-6, 1e-6         # sqrt(det) = 2e-6 m rad
        cov[:, 2, 2], cov[:, 3, 3] = 1e-6, 1e-6
        cov[:, 4, 4], cov[:, 5, 5] = 1e-2, 1e-8
        return {"start": start, "turns": np.arange(start, start + k),
                "n_alive": np.asarray(n_alive), "cov": cov, "n_turns": 10,
                "beta0gamma0": 1.6, "p0c_mev": 1500.0}
    d.show_live(rows(0, 3, [10, 10, 9]))
    assert np.allclose(d.c_ex.yData, 2e-6 * 1.6 * 1e6) and len(d.c_ex.xData) == 3
    d.show_live(rows(3, 2, [9, 8]))
    assert list(d.c_surv.xData) == [0, 1, 2, 3, 4] and d.c_surv.yData[-1] == pytest.approx(80.0)
    # a new run (start 0) starts over; clear() forgets the reference
    d.show_live(rows(0, 2, [5, 5]))
    assert len(d.c_surv.xData) == 2
    d.clear()
    assert d._meta == {} and _n(d.c_surv) == 0
    d.deleteLater()


def test_last_stays_the_last_turn_and_absent_settings_reset(win):
    win.open_lattice(FIXTURE)
    tab = _ring_on(win)
    tab._turns.setValue(10)
    tab._snapshots.setText("0, last")
    tab._snapshots.editingFinished.emit()
    assert win._collect_project_dict()["ring"]["config"]["snapshot_turns"] == [0, "last"]
    tab._turns.setValue(1000)
    data = win._collect_project_dict()
    win._apply_project_dict(copy.deepcopy(data), silent=True)
    t = win.ring_tab
    assert t._snapshots.text() == "0, last" and t._turns.value() == 1000
    # a project without the run settings: the Run page's defaults, nothing
    # kept from the project before
    t._probes.setValue(7)
    t._source.setCurrentIndex(1)                      # (empty)
    bare = copy.deepcopy(data)
    bare["ring"]["config"] = {"n_turns": 20}
    win._apply_project_dict(bare, silent=True)
    t = win.ring_tab
    assert (t._source.currentData(), t._probes.value(), t._snapshots.text()) == (
        "beam", 1, "0, last")
    # an empty list is no snapshot (one reading on every path)
    empty = copy.deepcopy(data)
    empty["ring"]["config"] = {"snapshot_turns": []}
    win._apply_project_dict(empty, silent=True)
    assert win.ring_tab._snapshots.text() == ""
    # a linac project in between: the Ring tab goes back to its defaults
    win.ring_tab._probes.setValue(5)
    linac = copy.deepcopy(data)
    linac.pop("ring")
    win._apply_project_dict(linac, silent=True)
    assert win.ring_tab._probes.value() == 1


def test_the_strip_keeps_the_results_it_describes(win, monkeypatch):
    # a run that fails, or is stopped before its first turn, leaves the last
    # results on the strip; a new lattice takes them away (from the Results
    # tab too)
    win.open_lattice(FIXTURE)
    tab = _ring_on(win, particles=100)
    tab._turns.setValue(3)
    tab._probes.setValue(0)
    tab._run_btn.click()
    win.wait_worker(tab._worker)
    assert _n(tab._live_strip.c_surv) == 4
    done = tab._summary.text()
    from linac_gen.ring import engine as E

    def boom(*_a, **_k):
        raise RuntimeError("boom")
    with monkeypatch.context() as m:
        m.setattr(E.RingSimulation, "run", boom)
        tab._run_btn.click()
        win.wait_worker(tab._worker)
    assert "boom" in tab._turn_label.text()
    assert _n(tab._live_strip.c_surv) == 4 and tab._summary.text() == done
    from linac_gen_gui.interphase import workers as W
    with monkeypatch.context() as m:
        m.setattr(W, "acquire_walk_lock", lambda _stop: False)
        tab._run_btn.click()
        win.wait_worker(tab._worker)
    assert "before the first turn" in tab._turn_label.text()
    assert _n(tab._live_strip.c_surv) == 4 and tab._summary.text() == done
    assert win.results_tab._source == "ring"
    # a long run, then a new lattice: its live curves go with it
    tab._turns.setValue(1_000_000)
    tab._run_btn.click()
    assert win.pump_until(lambda: _n(tab._live_strip.c_surv) > 2, 120_000)
    win.state.mark_project_clean()        # (no "unsaved changes" prompt in the way)
    win.open_lattice(FIXTURE)
    assert win.ring_tab._summary.text() == "No ring run yet."
    assert _n(win.ring_tab._live_strip.c_surv) == 0
    assert not win.ring_tab._to_results_btn.isEnabled()
    rt = win.results_tab
    assert rt._source == "linac" and rt._run_combo.isHidden()


def test_saved_files_are_named_apart_and_a_failed_save_is_said(win, monkeypatch):
    import h5py

    win.open_lattice(FIXTURE)
    tab = _ring_on(win, particles=100)
    tab._turns.setValue(2)
    tab._probes.setValue(0)
    paths = []
    for _ in range(2):
        tab._run_btn.click()
        win.wait_worker(tab._worker)
        paths.append(win.state.ring_results.meta["dump_path"])
    assert paths[0] != paths[1] and all(Path(q).is_file() for q in paths)
    with h5py.File(paths[1], "r") as f:
        assert f["provenance"].attrs["lattice_path"].endswith("simple_booster.madx")
        assert "lattice_sha256" in f["provenance"].attrs
        assert f["beam_config"].attrs["species"] in ("proton", "H-")
    from linac_gen.ring import results as R

    def full(*_a, **_k):
        raise OSError("disk full")
    monkeypatch.setattr(R, "save_ring_hdf5", full)
    tab._run_btn.click()
    win.wait_worker(tab._worker)
    text = tab._summary.text()
    assert "Not saved: OSError: disk full" in text and paths[1] not in text


def test_a_run_too_big_for_the_memory_is_refused(win, monkeypatch):
    from linac_gen_gui.interphase.tabs import ring_tab as RT
    win.open_lattice(FIXTURE)
    tab = _ring_on(win)
    monkeypatch.setattr(RT, "_physical_memory", lambda: 2 ** 30)
    win.beam_tab._npart.setValue(2_000_000)           # (the Beam tab's particles)
    tab._run_btn.click()
    assert tab._worker is None
    assert "GB of the machine's 1 GB" in tab._turn_label.text()


def test_the_tunes_window_says_why_it_shows_no_tune(qapp):
    # no probe asked for, or probes whose motion gave NAFF no tune (a run
    # with a tune shift beyond half an integer, for example)
    from types import SimpleNamespace

    from linac_gen_gui.interphase.panels.ring_plots import tunes_text
    from linac_gen_gui.interphase.tabs.ring_popups import RingTunesPopup
    d = RingTunesPopup(None)
    for probes, text in ((0, "No tune probe (Tune probes 0)."),
                         (3, "NAFF found no tune in the probes' motion")):
        ring = SimpleNamespace(tunes={}, optics={}, probes={}, ramp=None,
                               meta={"config": {"tune_probes": probes}})
        assert text in "\n".join(tunes_text(ring)[0]), probes
        d.refresh(SimpleNamespace(is_ring_view=True, ring=ring))
        assert text in d.tunes_text.text(), probes
    d.refresh(None)
    assert "No ring run is shown" in d.tunes_text.text()
    d.deleteLater()



def test_a_mismatched_start_is_said_once_and_points_to_the_button(win):
    # the Beam tab's linac default is far from the ring's matched beam: the
    # summary gives its BMAG once (not the engine's note again) and the button
    win.open_lattice(FIXTURE)
    tab = _ring_on(win, particles=100, matched=False)
    tab._turns.setValue(2)
    tab._probes.setValue(0)
    tab._run_btn.click()
    win.wait_worker(tab._worker)
    text = tab._summary.text()
    assert text.count("BMAG") == 1 and "At the start, BMAG (1 when matched) x " in text
    assert "the beam at the start is mismatched" not in text
    assert "Ring & Optics → Use the matched beam in the Beam tab gives the Beam tab the " \
           "matched beam." in text
    # (the results keep the engine's note)
    assert any(n.startswith("the beam at the start is mismatched")
               for n in win.state.ring_results.meta["notes"])


def test_an_earlier_projects_note_goes_with_a_bare_lattice(win):
    win.open_lattice(FIXTURE)
    tab = _ring_on(win, matched=False)
    data = copy.deepcopy(win._collect_project_dict())
    data["ring"]["config"].update({"source": "matched", "n_particles": 900})
    win._apply_project_dict(copy.deepcopy(data), silent=True)
    assert "started from a beam of 900 particles" in win.ring_tab._beam_line.text()
    win.state.mark_project_clean()
    win.open_lattice(FIXTURE)                             # (a bare Open Lattice)
    assert "started from a beam" not in win.ring_tab._beam_line.text()


def test_save_takes_snapshot_turns_still_being_typed(win, tmp_path):
    # the field is taken when its edit ends; a Save does not end it, and the
    # file written holds what was typed (a run would take it too)
    import json

    from PyQt6.QtTest import QTest
    win.open_lattice(FIXTURE)
    tab = _ring_on(win, matched=False)
    tab._snapshots.selectAll()
    QTest.keyClicks(tab._snapshots, "0, 5, 9")
    assert tab._snapshots.isModified()
    assert tab.current_settings()["snapshot_turns"] == [0, 5, 9]
    path = tmp_path / "typed.lgproj"
    assert win._write_project_file(str(path))
    saved = json.loads(path.read_text(encoding="utf-8"))
    assert saved["ring"]["config"]["snapshot_turns"] == [0, 5, 9]
    assert not tab._snapshots.isModified() and not win.state.project_dirty
    # nothing typed since: a save marks nothing changed
    win._collect_project_dict()
    assert not win.state.project_dirty
