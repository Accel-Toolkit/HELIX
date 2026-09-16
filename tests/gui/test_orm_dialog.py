"""Orbit-response calibration dialog through the real seams.

Tools → Orbit-Response Calibration (LOCO)… (toolbar signal) on the
orm_demo FODO with a synthetic FORMA export: load both folders, mapping
table, compare (worker on a snapshot), fit, undoable apply, export with
re-parse verification, stale-lattice discard, single instance, close
during a fit."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("PyQt6")

REPO = Path(__file__).resolve().parents[2]
DEMO = REPO / "examples" / "orm_demo"


@pytest.fixture(scope="module")
def synthetic(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("orm_gui")
    spec = importlib.util.spec_from_file_location("make_synthetic_forma", DEMO / "make_synthetic_forma.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert mod.main(["--out", str(tmp / "synthetic"), "--g-sigma", "0.02", "--noise", "0.005", "--seed", "3"]) == 0
    return tmp / "synthetic/20260101_000000_H", tmp / "synthetic/20260101_000000_V"


def _quads(lat):
    from linac_gen.elements.quadrupole import Quadrupole
    return [q for q in lat.elements if isinstance(q, Quadrupole)]


def _open(win, monkeypatch, synthetic):
    from linac_gen.io.project import load_project
    from linac_gen_gui.interphase.dialogs import orm_dialog as mod
    win.open_lattice(DEMO / "fodo_orm.dat")
    win.state.set_beam_config(load_project(DEMO / "fodo_orm.lgproj").beam)
    win._toolbar.open_orm_requested.emit()                       # the real Tools-menu seam
    dlg = win._orm_dlg
    assert dlg.isVisible()
    for folder in synthetic:
        monkeypatch.setattr(mod.QFileDialog, "getExistingDirectory", staticmethod(lambda *a, _f=folder, **k: str(_f)))
        dlg._load_folder_btn.click()
    return dlg


def test_load_map_compare_fit_apply_export(win, monkeypatch, tmp_path, synthetic):
    from linac_gen.orm import verify_exported_deck
    from linac_gen_gui.interphase.dialogs import orm_dialog as mod
    dlg = _open(win, monkeypatch, synthetic)
    assert set(dlg._measured) == {"x", "y"} and dlg._sel.n_bpm == 12 and dlg._sel.n_trim == 6
    assert dlg._map_table.rowCount() == 18                        # 12 BPM devices + 6 trims, unique
    assert all(dlg._map_table.item(r, 5).text().startswith("matched") for r in range(18))
    assert win.state.orm_session["paths"] == [str(p) for p in synthetic]

    # a mapping edit: leave one BPM out (trims are listed first — pick the first BPM row)
    r_bpm = next(r for r in range(18) if dlg._map_table.item(r, 1).text() == "BPM")
    dlg._map_table.item(r_bpm, 7).setCheckState(mod.Qt.CheckState.Unchecked)
    dlg._apply_map_btn.click()
    dev0 = dlg._map_table.item(r_bpm, 0).text()
    assert dev0 in dlg._map.exclude_bpms and dlg._map_table.item(r_bpm, 5).text() == "excluded"
    dlg._map_table.item(r_bpm, 7).setCheckState(mod.Qt.CheckState.Checked)
    dlg._apply_map_btn.click()
    assert dev0 not in dlg._map.exclude_bpms and dlg._map_table.item(r_bpm, 5).text().startswith("matched")

    # compare on a snapshot
    dlg._tracking_chk.setChecked(True)
    dlg._compare_btn.click()
    win.wait_worker(dlg._worker)
    assert dlg._compare is not None and dlg._metrics_table.rowCount() == 6
    assert dlg._heat_img["meas"].image is not None and dlg._heat_img["meas"].image.shape == (12, 6)
    assert "cross-check" in dlg._compare_summary.text()
    dlg._plane_combo.setCurrentIndex(1)
    assert dlg._metrics_table.rowCount() == 6 and dlg._trim_combo.count() == 6

    # fit
    before = [(q.gradient, q.gradient_rel) for q in _quads(win.state.lattice)]
    dlg._fit_btn.click()
    win.wait_worker(dlg._worker)
    assert dlg._fit is not None and dlg._fit.converged and dlg._quad_table.rowCount() == 12
    assert dlg._cal["__kind__"] == "helix_orm_calibration" and win.state.orm_session["calibration"] is dlg._cal
    assert [(q.gradient, q.gradient_rel) for q in _quads(win.state.lattice)] == before      # worker never touched the live lattice

    # apply: ONE undoable command, gradient_rel set, Save reroutes to Save-As
    dlg._apply_btn.click()
    quads = _quads(win.state.lattice)
    assert sum(1 for q in quads if q.gradient_rel != 0.0) >= 8
    assert all(q.gradient == g for q, (g, _r) in zip(quads, before))
    assert win.state.lattice_fitted
    assert win.state.bus.undo()
    assert all(q.gradient_rel == 0.0 for q in quads)
    assert win.state.bus.redo()
    assert sum(1 for q in quads if q.gradient_rel != 0.0) >= 8
    win.state.bus.undo()

    # bake mode
    dlg._bake_chk.setChecked(True)
    dlg._apply_btn.click()
    assert sum(1 for q, (g, _r) in zip(quads, before) if q.gradient != g) >= 8 and all(q.gradient_rel == 0.0 for q in quads)
    win.state.bus.undo()
    assert all(q.gradient == g for q, (g, _r) in zip(quads, before))

    # export (text surgery on the session deck, verified by re-parse)
    out = tmp_path / "fodo_orm_fit.dat"
    monkeypatch.setattr(mod.QFileDialog, "getSaveFileName", staticmethod(lambda *a, **k: (str(out), "")))
    dlg._export_deck_btn.click()
    assert out.exists()
    assert verify_exported_deck(DEMO / "fodo_orm.dat", out, dlg._cal)["max_rel_dev"] < 1e-9
    cal_out = tmp_path / "cal.json"
    monkeypatch.setattr(mod.QFileDialog, "getSaveFileName", staticmethod(lambda *a, **k: (str(cal_out), "")))
    dlg._save_cal_btn.click()
    assert cal_out.exists()

    # a stale result (lattice edited while computing) is discarded
    out_dict = {"task": "compare", "selection": dlg._sel, "model": dlg._model, "compare": dlg._compare}
    dlg._fp_at_launch = ("stale",)
    dlg._lattice_at_launch = win.state.lattice
    dlg._on_done(out_dict)
    assert dlg._fit_status.text().startswith("discarded")

    # single instance
    win._toolbar.open_orm_requested.emit()
    assert win._orm_dlg is dlg


def test_close_during_fit_stops_worker_and_session_survives(win, monkeypatch, synthetic):
    dlg = _open(win, monkeypatch, synthetic)
    dlg._max_iter.setValue(200)
    dlg._fit_btn.click()
    assert dlg._worker is not None and dlg._worker.isRunning()
    dlg.close()
    assert not dlg._worker.isRunning()
    win.pump(0.3)
    # reopening restores the loaded measurement from state.orm_session
    win._toolbar.open_orm_requested.emit()
    dlg2 = win._orm_dlg
    assert dlg2.isVisible() and set(dlg2._measured) == {"x", "y"} and dlg2._map_table.rowCount() == 18


def test_apply_refuses_when_design_gradients_drifted(win, monkeypatch, synthetic):
    dlg = _open(win, monkeypatch, synthetic)
    dlg._fit_btn.click()
    win.wait_worker(dlg._worker)
    assert dlg._cal is not None
    q0 = _quads(win.state.lattice)[0]
    q0.gradient *= 1.01                                          # the deck was edited after the fit
    dlg._apply_btn.click()
    assert all(q.gradient_rel == 0.0 for q in _quads(win.state.lattice))
    assert any("not applied" in (t or "").lower() or "differs" in (x or "") for _k, t, x in win.message_boxes)


# ---------------------------------------------------------------------------
# Review findings (2026-09-14 adversarial pass), each pinned
# ---------------------------------------------------------------------------
@pytest.fixture()
def capture_slot_exceptions():
    """Record exceptions PyQt routes to sys.excepthook from slot calls."""
    import sys
    caught: list = []
    saved = sys.excepthook
    sys.excepthook = lambda _t, v, _tb: caught.append(v)
    try:
        yield caught
    finally:
        sys.excepthook = saved


def test_app_shutdown_stops_a_running_orm_worker(win, monkeypatch, synthetic):
    """Finding 1: quitting the app with a fit running destroyed a live QThread (SIGABRT)."""
    dlg = _open(win, monkeypatch, synthetic)
    dlg._max_iter.setValue(200)
    dlg._fit_btn.click()
    assert dlg._worker.isRunning()
    win._shutdown_workers()
    assert not dlg._worker.isRunning()


def test_import_map_after_compare_invalidates_results(win, monkeypatch, tmp_path, synthetic, capture_slot_exceptions):
    """Finding 3: a re-resolved selection with a stale compare broadcast (11,) against (12,) in a slot."""
    from linac_gen_gui.interphase.dialogs import orm_dialog as mod
    dlg = _open(win, monkeypatch, synthetic)
    dlg._compare_btn.click(); win.wait_worker(dlg._worker)
    assert dlg._compare is not None and dlg._sel.n_bpm == 12
    d = dlg._map.to_dict(); d["bpms"].pop("D011BPM")
    mp = tmp_path / "map.json"; mp.write_text(__import__("json").dumps(d), encoding="utf-8")
    monkeypatch.setattr(mod.QFileDialog, "getOpenFileName", staticmethod(lambda *a, **k: (str(mp), "")))
    dlg._import_map_btn.click()
    assert dlg._compare is None and dlg._sel.n_bpm == 11 and dlg._compare_summary.text() == "—"
    dlg._plane_combo.setCurrentIndex(1); dlg._plane_combo.setCurrentIndex(0)
    assert capture_slot_exceptions == []
    # Auto-map after that restores the 12 and also invalidates
    dlg._compare_btn.click(); win.wait_worker(dlg._worker)
    assert dlg._compare is not None
    dlg._automap_btn.click()
    assert dlg._compare is None and dlg._sel.n_bpm == 12


def test_apply_twice_is_a_noop_and_one_undo_step(win, monkeypatch, synthetic):
    """Finding 2: the second Apply compounded the scale factors (scale² − 1) and pushed a second undo step."""
    dlg = _open(win, monkeypatch, synthetic)
    dlg._fit_btn.click(); win.wait_worker(dlg._worker)
    dlg._apply_btn.click()
    rel = [q.gradient_rel for q in _quads(win.state.lattice)]
    assert sum(1 for r in rel if r != 0.0) >= 8
    dlg._apply_btn.click()
    assert [q.gradient_rel for q in _quads(win.state.lattice)] == rel
    assert "nothing to apply" in dlg._fit_status.text()
    # bake on top of the applied calibration is refused
    dlg._bake_chk.setChecked(True); dlg._apply_btn.click()
    assert [q.gradient_rel for q in _quads(win.state.lattice)] == rel
    assert any("undo the applied calibration" in (x or "") for _k, _t, x in win.message_boxes)
    assert win.state.bus.undo() and all(q.gradient_rel == 0.0 for q in _quads(win.state.lattice))
    assert not win.state.bus.undo()                          # exactly one command was pushed


def test_clear_during_compute_discards_the_result(win, monkeypatch, synthetic):
    """Finding 4: results landed on a cleared session; the load/clear buttons are disabled while busy anyway."""
    dlg = _open(win, monkeypatch, synthetic)
    dlg._fit_btn.click()
    assert not dlg._clear_btn.isEnabled() and not dlg._load_folder_btn.isEnabled()
    dlg._clear()                                             # what an Esc-hidden dialog could still reach
    win.wait_worker(dlg._worker)
    assert dlg._fit is None and dlg._cal is None and dlg._measured == {}
    assert win.state.orm_session["calibration"] is None
    assert dlg._fit_status.text().startswith("discarded")
    assert dlg._clear_btn.isEnabled()


def test_optics_edit_or_beam_change_while_computing_discards(win, monkeypatch, synthetic):
    """Finding 5: only quad gradients were fingerprinted; a drift/beam edit slipped through."""
    from linac_gen.elements.drift import Drift
    from linac_gen_gui.interphase.commands import ParamChangeCommand
    dlg = _open(win, monkeypatch, synthetic)
    dlg._compare_btn.click()
    drift = next(e for e in win.state.lattice.elements if isinstance(e, Drift))
    win.state.bus.do(ParamChangeCommand(drift, "length", float(drift.length), float(drift.length) * 1.5))
    win.wait_worker(dlg._worker)
    assert dlg._compare is None and dlg._fit_status.text().startswith("discarded: the lattice or beam")
    win.state.bus.undo()
    dlg._compare_btn.click(); win.wait_worker(dlg._worker)
    assert dlg._compare is not None
    cfg = win.state.beam_config
    cfg.energy = cfg.energy * 1.3
    win.state.set_beam_config(cfg)
    assert dlg._compare is None and dlg._fit_status.text().startswith("beam changed")


def test_stale_session_path_is_dropped_not_poisoning(win, monkeypatch, tmp_path, synthetic):
    """Finding 6/7: a vanished folder in the session kept every later load failing."""
    from linac_gen.io.project import load_project
    win.open_lattice(DEMO / "fodo_orm.dat")
    win.state.set_beam_config(load_project(DEMO / "fodo_orm.lgproj").beam)
    gone = tmp_path / "gone_H"
    win.state.orm_session = {"paths": [str(gone), str(synthetic[0])], "map": None, "calibration": None}
    win._toolbar.open_orm_requested.emit()
    dlg = win._orm_dlg
    assert dlg._paths == [str(synthetic[0])] and set(dlg._measured) == {"x"}
    assert any("gone_H" in (x or "") for _k, _t, x in win.message_boxes)
    from linac_gen_gui.interphase.dialogs import orm_dialog as mod
    monkeypatch.setattr(mod.QFileDialog, "getExistingDirectory", staticmethod(lambda *a, **k: str(synthetic[1])))
    dlg._load_folder_btn.click()
    assert set(dlg._measured) == {"x", "y"} and win.state.orm_session["paths"] == [str(p) for p in synthetic]


def test_bad_calibration_json_is_refused_and_keeps_the_old_one(win, monkeypatch, tmp_path, synthetic):
    """Finding 8: a null scale raised in a slot and left _cal half-set."""
    import json
    from linac_gen_gui.interphase.dialogs import orm_dialog as mod
    dlg = _open(win, monkeypatch, synthetic)
    dlg._fit_btn.click(); win.wait_worker(dlg._worker)
    good = dlg._cal
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"__kind__": "helix_orm_calibration", "__version__": 1,
                               "quads": [{"label": "Q011", "scale": None, "gradient_design": 12.0, "gradient_rel": 0.0}]}), encoding="utf-8")
    monkeypatch.setattr(mod.QFileDialog, "getOpenFileName", staticmethod(lambda *a, **k: (str(bad), "")))
    dlg._load_cal_btn.click()
    assert dlg._cal is good
    assert any("finite number" in (x or "") for _k, _t, x in win.message_boxes)


def test_invalid_sign_is_flagged_and_unicode_minus_accepted(win, monkeypatch, synthetic):
    """Finding 9: '0', '−1' (U+2212) and '' silently became +1."""
    dlg = _open(win, monkeypatch, synthetic)
    r = next(r for r in range(dlg._map_table.rowCount()) if dlg._map_table.item(r, 1).text() == "BPM")
    dev = dlg._map_table.item(r, 0).text()
    dlg._map_table.item(r, 6).setText("0"); dlg._apply_map_btn.click()
    assert dlg._map_table.item(r, 5).text().startswith("invalid sign") and dev not in dlg._map.bpm_sign
    dlg._map_table.item(r, 6).setText("−1"); dlg._apply_map_btn.click()
    assert dlg._map.bpm_sign[dev] == -1 and dlg._map_table.item(r, 5).text().startswith("matched")
    assert dlg._map_table.item(r, 6).text() == "-1"


def test_loading_another_lattice_resets_the_dialog(win, monkeypatch, synthetic, capture_slot_exceptions):
    """Finding 14: after a new deck the tabs kept the previous lattice's content."""
    dlg = _open(win, monkeypatch, synthetic)
    dlg._compare_btn.click(); win.wait_worker(dlg._worker)
    win.open_lattice(REPO / "examples" / "halo_fodo.dat")
    assert dlg._compare is None and dlg._fit_status.text().startswith("lattice replaced")
    assert dlg._sel.n_bpm == 0 and "unmatched" in dlg._map_notes.toPlainText()
    assert capture_slot_exceptions == []
    dlg.close()
    win._toolbar.open_orm_requested.emit()                       # discard + rebuild path disconnects the old slots
    win.open_lattice(DEMO / "fodo_orm.dat")
    assert capture_slot_exceptions == []
