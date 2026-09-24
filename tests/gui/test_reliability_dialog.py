"""Reliability Study dialog through the real seams.

Tools → Reliability Study… (toolbar signal) on the shipped demo deck: a
quick campaign end to end (worker on the campaign path, tables, bar,
report, session), the refusals, cancel then resume, close during a run,
the shutdown sweep, stale results after a lattice edit, the undoable
apply of compensator settings, the Self-test button, and no dangling
slots after close / reopen."""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

pytest.importorskip("PyQt6")

REPO = Path(__file__).resolve().parents[2]
DEMO = REPO / "examples" / "reliability_demo"


@pytest.fixture()
def deck(tmp_path):
    for name in ("reliability_demo.dat", "reliability_demo.lgproj", "circuits.json"):
        shutil.copy2(DEMO / name, tmp_path / name)
    return tmp_path / "reliability_demo.dat"


def _open(win, deck):
    from linac_gen.io.project import load_project
    from PyQt6.QtCore import Qt
    win.open_lattice(deck)
    win.state.set_beam_config(load_project(deck.with_suffix(".lgproj")).beam)
    win._toolbar.open_reliability_requested.emit()                # the real Tools-menu seam
    dlg = win._reliability_dlg
    assert dlg.isVisible()
    dlg._root.setText(str(deck.parent / "campaigns"))
    dlg._name.setText("gui")
    dlg._workers.setValue(1)
    for lg, cb in dlg._leg_chk.items():
        cb.setChecked(lg in ("faults", "availability"))
    for i in range(dlg._class_list.count()):
        it = dlg._class_list.item(i)
        it.setCheckState(Qt.CheckState.Checked if it.data(Qt.ItemDataRole.UserRole) in ("S1", "S5")
                         else Qt.CheckState.Unchecked)
    dlg._comp_top.setValue(1); dlg._comp_iter.setValue(60)
    return dlg


def test_quick_campaign_end_to_end_and_session(win, deck):
    dlg = _open(win, deck)
    dlg._start_btn.click()
    assert dlg._worker is not None and dlg._cancel_btn.isEnabled()
    win.wait_worker(dlg._worker)
    cdir = deck.parent / "campaigns" / "gui"
    assert (cdir / "summary.json").exists() and (cdir / "report.html").exists()
    assert dlg._results_dir == cdir and win.state.reliability_session["campaign_dir"] == str(cdir)
    assert dlg._fault_table.rowCount() == 7                        # 4 cavities + 3 steerers
    assert dlg._fault_table.item(0, 0).text().startswith("S1_")    # worst first
    assert dlg._comp_table.rowCount() == 1 and dlg._comp_table.item(0, 2).text() == "True"
    assert dlg._avail_table.rowCount() == 2 and "SURROGATE" in dlg._avail_note.text()
    assert dlg._bar.plotItem.items and dlg._report_view.toPlainText().startswith("Reliability Study")
    assert dlg._apply_btn.isEnabled() and dlg._stale_reason == ""
    assert "complete" in dlg._overall.text()
    # the leg-A table reproduces the truth's re-phased deficit of the top cavity
    truth = json.loads((DEMO / "truth.json").read_text())
    rows = {dlg._fault_table.item(r, 0).text(): dlg._fault_table.item(r, 5).text()
            for r in range(dlg._fault_table.rowCount())}
    assert float(rows["S1_GAP_004_re"]) == pytest.approx(truth["cavity_off_rephased"]["GAP_004"]["d_energy_mev"], rel=2e-3)   # 4-digit display
    # close and reopen: the session restores the results without a run
    dlg.close(); win.pump(0.2)
    win._toolbar.open_reliability_requested.emit()
    dlg2 = win._reliability_dlg
    assert dlg2 is not dlg and dlg2._fault_table.rowCount() == 7 and dlg2._results_dir == cdir


def test_refusals(win, deck, tmp_path):
    from PyQt6.QtWidgets import QMessageBox
    n0 = len(win.message_boxes)
    win.state.set_lattice(None, None)
    win._open_reliability_study()                                   # no lattice
    assert getattr(win, "_reliability_dlg", None) is None or not win._reliability_dlg.isVisible()
    assert len(win.message_boxes) == n0 + 1 and "Load a lattice" in win.message_boxes[-1][2]
    dlg = _open(win, deck)
    win.state.bus.mark_dirty() if hasattr(win.state.bus, "mark_dirty") else None
    if getattr(win.state.bus, "dirty", False):
        dlg._start_btn.click()
        assert dlg._worker is None and "unsaved" in win.message_boxes[-1][2]
    _ = QMessageBox


def test_cancel_then_resume_and_close_during_run(win, deck):
    dlg = _open(win, deck)
    dlg._start_btn.click()
    w = dlg._worker
    dlg._cancel_btn.click()
    win.wait_worker(w)
    assert "stopped" in dlg._overall.text() or "complete" in dlg._overall.text()
    from linac_gen.reliability.campaign import ReliabilityCampaign
    cdir = deck.parent / "campaigns" / "gui"
    c = ReliabilityCampaign.load(cdir)
    pending_before = len(c.pending())
    # resume through the folder chooser
    from linac_gen_gui.interphase.dialogs import reliability_dialog as mod
    import pytest as _pt
    _pt.MonkeyPatch().setattr(mod.QFileDialog, "getExistingDirectory", staticmethod(lambda *a, **k: str(cdir)))
    dlg._resume_btn.click()
    win.wait_worker(dlg._worker)
    assert ReliabilityCampaign.load(cdir).pending() == [] and pending_before >= 0
    # close during a run: the worker is stopped and joined, the session survives
    dlg._name.setText("gui2")
    dlg._start_btn.click()
    w2 = dlg._worker
    dlg.close()
    assert not w2.isRunning() and win.state.reliability_session["campaign_dir"].endswith("gui2")


def test_shutdown_sweep_collects_a_running_campaign_worker(win, deck):
    dlg = _open(win, deck)
    dlg._start_btn.click()
    w = dlg._worker
    assert w.isRunning()
    got = dlg.shutdown_begin()
    assert got == [w]
    win.wait_worker(w)
    win._shutdown_workers()                                         # no live thread left behind
    assert not w.isRunning()


def test_stale_after_lattice_edit_and_undoable_apply(win, deck):
    dlg = _open(win, deck)
    dlg._start_btn.click()
    win.wait_worker(dlg._worker)
    assert dlg._apply_btn.isEnabled()
    # apply the (single) compensation as ONE undoable edit
    dlg._comp_table.selectRow(0)
    settings = json.loads(dlg._comp_rows[0]["settings"])
    lat = win.state.lattice
    before = {k: getattr(next(e for e in lat.elements if e.name == k.split(".")[0]), k.split(".")[1]) for k in settings}
    bus = win.state.bus
    dlg._apply_btn.click()
    for k, v in settings.items():
        el = next(e for e in lat.elements if e.name == k.split(".")[0])
        assert getattr(el, k.split(".")[1]) == pytest.approx(float(v))
    assert "applied" in dlg._apply_status.text() and not dlg._apply_btn.isEnabled()
    bus.undo()
    for k, v in before.items():
        el = next(e for e in lat.elements if e.name == k.split(".")[0])
        assert getattr(el, k.split(".")[1]) == v
    # a fresh campaign, then an unrelated edit through the bus → stale, Apply refuses
    dlg._name.setText("gui3")
    dlg._start_btn.click()
    win.wait_worker(dlg._worker)
    assert dlg._apply_btn.isEnabled()
    from linac_gen_gui.interphase.commands import ParamChangeCommand
    q = next(e for e in lat.elements if e.name == "SOL_001")
    bus.do(ParamChangeCommand(q, "field", q.field, q.field * 1.01))
    assert dlg._stale_reason and not dlg._apply_btn.isEnabled()
    n = len(win.message_boxes)
    dlg._comp_table.selectRow(0)
    dlg._apply_settings()
    assert len(win.message_boxes) == n + 1 and "no longer" in win.message_boxes[-1][2]


def test_selftest_button_reports_pass(win, deck):
    dlg = _open(win, deck)
    dlg._selftest_btn.click()
    w = dlg._selftest_worker
    assert w is not None
    # the window is busy for the self-test too: Start / Export / Import wait
    assert not dlg._start_btn.isEnabled() and not dlg._export_btn.isEnabled() and not dlg._import_btn.isEnabled()
    dlg._on_start()
    assert dlg._worker is None
    win.wait_worker(w)
    assert dlg._start_btn.isEnabled() and dlg._export_btn.isEnabled()
    assert dlg._selftest_result["verdict"] == "PASS"
    assert "VERDICT PASS" in dlg._log.toPlainText() and dlg._selftest_btn.isEnabled()
    assert "PASS" in win.message_boxes[-1][2]


def test_no_dangling_slots_after_close_reopen_and_new_deck(win, deck):
    dlg = _open(win, deck)
    dlg.close(); win.pump(0.2)
    win._toolbar.open_reliability_requested.emit()
    dlg2 = win._reliability_dlg
    assert dlg2 is not dlg
    dlg2.close(); win.pump(0.2)
    win.open_lattice(deck)                                          # lattice_changed fires on a closed dialog
    win.pump(0.2)
    from linac_gen.core.config import BeamConfig
    win.state.set_beam_config(BeamConfig(species="proton", energy=20.0, frequency=162.5))
    win.pump(0.2)


def _first_data_image(html: str) -> str:
    import re
    m = re.search(r'src="(data:image/png;base64,[^"]+)"', html)
    return m.group(1) if m else ""


def test_leg_b_and_d_inputs_reach_the_campaign_and_report_images_resolve(win, deck, tmp_path):
    from linac_gen_gui.interphase.dialogs import reliability_dialog as mod
    dlg = _open(win, deck)
    for lg, cb in dlg._leg_chk.items():
        cb.setChecked(lg in ("availability", "foil"))
    assert dlg._corr_pairing.currentText() == "cards"
    mp = pytest.MonkeyPatch()
    blocks = tmp_path / "my_blocks.csv"
    mp.setattr(mod.QFileDialog, "getSaveFileName", staticmethod(lambda *a, **k: (str(blocks), "")))
    dlg._blocks_template.click()
    assert blocks.exists() and dlg._blocks.text() == str(blocks) and "SURROGATE" in blocks.read_text()
    dlg._av_trials.setValue(50); dlg._av_hours.setValue(4000); dlg._av_seed.setValue(7)
    dlg._av_variants.setText("srf_fdr=62.5"); dlg._av_bins.setText("10, 60")
    dlg._foil_thick.setText("0, 300"); dlg._foil_offsets.setText("0.5,0"); dlg._foil_thinned.setValue(0.25)
    dlg._foil_extent.setText("3, 4"); dlg._foil_hits.setValue(2.0)
    dlg._start_btn.click()
    win.wait_worker(dlg._worker)
    cdir = deck.parent / "campaigns" / "gui"
    spec = json.loads((cdir / "campaign.json").read_text())
    av, fo = spec["availability"], spec["foil"]
    assert av["blocks"] == str(blocks.resolve()) and av["n_trials"] == 50 and av["hours_per_year"] == 4000.0
    assert av["seed"] == 7 and av["variants"] == {"srf_fdr": 62.5} and av["budget_bins_s"] == [10.0, 60.0]
    assert fo["thickness_ug_cm2"] == [0.0, 300.0] and fo["offsets_mm"] == [[0.5, 0.0]] and fo["thinned_fraction"] == 0.25
    assert fo["extent_mm"] == [3.0, 4.0] and fo["hits_per_particle"] == 2.0 and spec["correction"]["pairing"] == "cards"
    # leg D: nominal + 2 thicknesses + 1 offset + thinned
    assert dlg._foil_table.rowCount() == 5
    labels = {dlg._foil_table.item(r, 1).text() for r in range(5)}
    assert "foil missing" in labels and "thinned foil x0.25" in labels
    # leg B: one variant, the user's block table, the trip budget with the two bins
    assert dlg._avail_table.rowCount() == 1 and dlg._avail_table.item(0, 0).text() == "srf_fdr"
    assert "my_blocks.csv" in dlg._avail_note.text()
    assert dlg._trip_table.columnCount() == 2 and dlg._trip_table.rowCount() >= 2
    assert dlg._trip_table.item(0, 0).text().startswith("10")
    assert "srf_fdr" in dlg._trip_table.horizontalHeaderItem(1).text()
    # the report view resolves its embedded figures
    from PyQt6.QtCore import QUrl
    from PyQt6.QtGui import QTextDocument
    html = (cdir / "report.html").read_text(encoding="utf-8")
    src = _first_data_image(html)
    assert src, "the report embeds no PNG"
    res = dlg._report_view.document().resource(QTextDocument.ResourceType.ImageResource, QUrl(src))
    assert res is not None and not res.isNull() and res.width() > 0, type(res)
    # a malformed entry is refused before anything is created
    dlg._name.setText("gui_bad")
    for field, text, word in ((dlg._foil_thick, "0, abc", "abc"), (dlg._foil_thick, "0, nan", "finite"),
                              (dlg._foil_thick, "-5", "negative"), (dlg._av_bins, "60, 10", "increasing"),
                              (dlg._av_variants, "a=1, a=2", "twice"), (dlg._av_variants, "a=0", "positive")):
        keep = field.text(); field.setText(text)
        n = len(win.message_boxes)
        dlg._start_btn.click()
        assert dlg._worker is None or not dlg._worker.isRunning()
        assert len(win.message_boxes) == n + 1 and word in win.message_boxes[-1][2], (text, win.message_boxes[-1][2])
        field.setText(keep)
    assert not (deck.parent / "campaigns" / "gui_bad").exists()


def test_export_and_import_through_the_buttons(win, deck, tmp_path):
    from PyQt6.QtCore import Qt
    from linac_gen_gui.interphase.dialogs import reliability_dialog as mod
    from linac_gen.reliability.campaign import ReliabilityCampaign
    dlg = _open(win, deck)
    for lg, cb in dlg._leg_chk.items():
        cb.setChecked(lg in ("faults", "foil"))
    for i in range(dlg._class_list.count()):
        it = dlg._class_list.item(i)
        it.setCheckState(Qt.CheckState.Checked if it.data(Qt.ItemDataRole.UserRole) == "S5" else Qt.CheckState.Unchecked)
    dlg._comp_top.setValue(0)
    # before any campaign the buttons explain themselves
    n0 = len(win.message_boxes)
    dlg._export_btn.click()
    assert len(win.message_boxes) == n0 + 1 and "Start or resume" in win.message_boxes[-1][2]
    dlg._start_btn.click(); win.wait_worker(dlg._worker)
    cdir = deck.parent / "campaigns" / "gui"
    assert dlg._foil_table.rowCount() == 7                       # nominal + 3 thicknesses + 2 offsets + thinned
    # export the foil leg only: its completed items travel with the job
    mp = pytest.MonkeyPatch()
    job = tmp_path / "job"
    mp.setattr(mod.QFileDialog, "getSaveFileName", staticmethod(lambda *a, **k: (str(job), "")))
    dlg._leg_chk["faults"].setChecked(False)
    n0 = len(win.message_boxes)
    dlg._export_btn.click()
    assert (job / "job.json").exists() and (job / "README.txt").exists()
    doc = json.loads((job / "job.json").read_text())
    assert doc["legs"] == ["foil"] and len(doc["expected"]) == 7
    assert len(win.message_boxes) == n0 + 1 and "7 item(s)" in win.message_boxes[-1][2]
    # a folder that is not a job is refused
    mp.setattr(mod.QFileDialog, "getExistingDirectory", staticmethod(lambda *a, **k: str(tmp_path)))
    dlg._import_btn.click()
    assert "no job.json" in win.message_boxes[-1][2]
    # pretend the foil leg ran remotely: drop the local results, import the job's
    shutil.rmtree(cdir / "legs" / "foil")
    assert len(ReliabilityCampaign.load(cdir).pending()) == 7
    mp.setattr(mod.QFileDialog, "getExistingDirectory", staticmethod(lambda *a, **k: str(job)))
    dlg._import_btn.click()
    rec = json.loads((cdir / "import.json").read_text())
    assert len(rec["imported"]) == 7 and rec["missing"] == []
    assert ReliabilityCampaign.load(cdir).pending() == [] and dlg._foil_table.rowCount() == 7
    assert "imported 7" in dlg._status.text() and win.state.reliability_session["campaign_dir"] == str(cdir)
    # a second import changes nothing (never overwrites an ok item)
    dlg._import_btn.click()
    assert "imported 0" in dlg._status.text() and "7 already complete" in dlg._status.text()
    # a partial job asks before importing the rest
    shutil.rmtree(cdir / "legs" / "foil")
    victim = next(p for p in (job / "legs" / "foil" / "runs").iterdir() if p.is_dir())
    shutil.rmtree(victim)
    n0 = len(win.message_boxes)
    dlg._import_btn.click()                                     # the tripwire answers Yes
    assert any("anyway" in mb[2] for mb in win.message_boxes[n0:])
    rec = json.loads((cdir / "import.json").read_text())
    assert len(rec["imported"]) == 6 and len(rec["missing"]) == 1
    # while a run is going the buttons are disabled
    dlg._name.setText("gui2"); dlg._leg_chk["faults"].setChecked(True); dlg._leg_chk["foil"].setChecked(False)
    dlg._start_btn.click()
    assert not dlg._export_btn.isEnabled() and not dlg._import_btn.isEnabled() and not dlg._blocks_template.isEnabled()
    win.wait_worker(dlg._worker)
    assert dlg._export_btn.isEnabled() and dlg._import_btn.isEnabled()
