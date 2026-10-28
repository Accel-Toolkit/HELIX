"""Ring mode in the GUI, milestone M3: the Programs page (the ramp's energy,
voltage and phase programs, transition, paraphasing; tables from and to CSV
files; the preview of the reference alone) and a run through a ramp from
the Run button — the Results tab's "Ring ramp", the separatrix of the
turn's bucket, the programs in the project."""
from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("PyQt6")

ROOT = Path(__file__).resolve().parents[2]
FIXTURE = ROOT / "tests" / "ring" / "fixtures" / "simple_booster.madx"
GAMMA_T = 5.449167323


def _n(curve) -> int:
    return 0 if curve.xData is None else len(curve.xData)


def _ring_on(win, particles=100):
    act = win._toolbar._ring_action
    if not act.isChecked():
        act.trigger()
    win.pump(0.1)
    tab = win.ring_tab
    win.beam_tab._npart.setValue(particles)           # (a run takes the Beam tab's beam)
    return tab


def _use_matched(win, tab):
    """Ring & Optics → Use the matched beam in the Beam tab (at the start of
    the run as the pages stand)."""
    tab._to_beam_btn.click()
    win.wait_worker(tab._matched_worker)
    assert tab._optics_status.text().startswith("Beam tab: the beam matched"), \
        tab._optics_status.text()
    return win.beam_tab.get_beam_config()


def _set(edit, text):
    edit.setText(text)
    edit.editingFinished.emit()


def _sine(page):
    page._e_kind.setCurrentIndex(2)                       # resonant cycle
    _set(page._e_start, "800")
    _set(page._e_top, "6000")
    _set(page._e_freq, "15")
    _set(page._e_tmin, "0.5")
    _set(page._e_tstart, "0.8")
    _set(page._gamma_t, str(GAMMA_T))
    _set(page._t0, "4")


def test_the_programs_go_into_the_project_and_come_back(win):
    from linac_gen.ring.programs import RampConfig
    win.open_lattice(FIXTURE)
    tab = _ring_on(win)
    page = tab._programs
    assert page.ramp_config() is None and "fixed" in tab._ramp_line.text()
    _sine(page)
    page._v_kind.setCurrentIndex(1)                       # a voltage table
    for t, v in (("0", "0.4"), ("3", "1.2"), ("30", "1.2")):
        page._v_table.add_row()
        r = page._v_table.table.rowCount() - 1
        page._v_table.table.item(r, 0).setText(t)
        page._v_table.table.item(r, 1).setText(v)
    want = RampConfig.from_dict({
        "energy": {"kind": "sine", "start_kinetic_mev": 800.0, "end_kinetic_mev": 6000.0,
                   "frequency_hz": 15.0, "t_min_s": 0.0005, "t_start_s": 0.0008},
        "voltage": {"kind": "table", "times_s": [0.0, 0.003, 0.03],
                    "values_mv": [0.4, 1.2, 1.2]},
        "gamma_t": GAMMA_T, "t_start_s": 0.004}).to_dict()
    # (times typed in ms: '0.8' is 0.0008 s exactly as typed in seconds)
    assert win._collect_project_dict()["ring"]["config"]["ramp"] == want
    assert "sine from 800 MeV to 6000 MeV at 15 Hz" in tab._ramp_line.text()
    # (the project's file holds it; loading it back fills the page)
    data = copy.deepcopy(win._collect_project_dict())
    page._e_kind.setCurrentIndex(0)
    assert "ramp" not in win.state.ring_config
    win._apply_project_dict(data, silent=True)
    page = win.ring_tab._programs
    assert page.ramp_config() == data["ring"]["config"]["ramp"]
    assert page._e_top.text() == "6000" and page._v_table.table.rowCount() == 3
    # an entry that is not valid is reported; the project keeps the last valid
    _set(page._e_freq, "fifteen")
    assert "Not valid" in page._status.text() and "cycle frequency" in page._status.text()
    assert win.state.ring_config["ramp"]["energy"]["frequency_hz"] == 15.0
    tab._turns.setValue(3)
    tab._run_btn.click()
    assert "Programs page" in tab._turn_label.text()
    assert tab._worker is None


def test_tables_from_and_to_csv_files(win, tmp_path):
    win.open_lattice(FIXTURE)
    page = _ring_on(win)._programs
    page._e_kind.setCurrentIndex(3)                       # a table
    f = tmp_path / "energy.csv"
    f.write_text("t_s,momentum_mevc\n0.0,1463.3\n0.0005,1500.0\n")
    assert page._e_table.import_csv(str(f)) == "momentum_mevc"
    # (the file's column sets the quantity, before the change is saved)
    assert page._e_quantity.currentData() == "momentum_mevc"
    assert win.state.ring_config["ramp"]["energy"]["quantity"] == "momentum_mevc"
    d = page.ramp_config()
    assert d["energy"]["times_s"] == [0.0, 0.0005] and d["energy"]["values"] == [1463.3, 1500.0]
    assert d["energy"]["quantity"] == "momentum_mevc"
    assert page._e_table.table.item(1, 0).text() == "0.5"          # (ms on the page)
    out = tmp_path / "back.csv"
    page._e_table.export_csv(str(out))
    from linac_gen.ring.programs import read_table_csv
    assert read_table_csv(out, "energy") == ((0.0, 0.0005), (1463.3, 1500.0), "momentum_mevc")
    page._e_quantity.setCurrentIndex(2)                   # total energy names the column
    page._e_table.export_csv(str(out))
    assert out.read_text().splitlines()[0] == "t_s,total_mev"
    # a file that cannot be read: said in the page, no exception from the button
    page._e_table.failed.emit("Import: test")
    assert page._status.text() == "Import: test"
    # paraphasing groups by number and by name pattern (commas, semicolons or
    # spaces between them; only plain digits are numbers)
    page._add_group()
    page._para.item(0, 0).setText("0 2;rf*, 007")
    page._para.item(0, 1).setText("10")
    assert page.ramp_config()["paraphasing"] == [{"cavities": [0, 2, "rf*", 7],
                                                  "angle_deg": 10.0}]
    page._para.item(0, 0).setText("1\u00b2")
    assert page.ramp_config()["paraphasing"][0]["cavities"] == ["1\u00b2"]
    # times that are not finite numbers are refused with the field's name
    from linac_gen_gui.interphase.tabs.ring_programs import _ms_to_s
    for bad in ("1e999999999", "1e400", "nan", "abc"):
        with pytest.raises(ValueError, match="clock"):
            _ms_to_s(bad, "clock")


def test_preview_and_a_run_through_the_ramp(win, tmp_path):
    from linac_gen.ring.results import load_ring_hdf5
    win.open_lattice(FIXTURE)
    tab = _ring_on(win, particles=100)
    page = tab._programs
    page._e_kind.setCurrentIndex(3)
    page._e_table.set_points([0.0, 0.0001], [800.0, 806.0])   # about 54 turns
    page._e_table.changed.emit()
    page._v_kind.setCurrentIndex(1)
    page._v_table.set_points([0.0, 1.0], [1.2, 1.2])
    page._v_table.changed.emit()
    # the beam matched to the ring at the start of the ramp (its RF at turn 0)
    cfg = _use_matched(win, tab)
    assert "turn 0 of the ramp" in tab._optics_status.text()
    assert cfg.energy == pytest.approx(800.0, rel=1e-12) and cfg.beta_z > 0.0
    page.preview()
    win.wait_worker(page._worker)
    assert "turns to the end of the energy program" in page._status.text()
    assert page._worker is not None
    n_end = page._preview["turns_to_end"]
    assert 40 < n_end < 70 and _n(page._plots.c_energy) == n_end
    assert page._to_run_btn.isEnabled()
    page._to_run_btn.click()
    assert tab._turns.value() == n_end
    tab._snapshots.setText("0, last")
    tab._snapshots.editingFinished.emit()
    tab._run_btn.click()
    win.wait_worker(tab._worker)
    res = win.state.ring_results
    assert res is not None and res.turns_done == n_end and res.ramp
    assert res.ramp["kinetic_mev"][0] == pytest.approx(800.0, rel=1e-15)
    # the Beam tab's beam, matched: BMAG 1 in every mode (no mismatch note)
    assert res.meta["n_particles"] == 100
    assert res.meta["start_mismatch"] == pytest.approx([1.0, 1.0, 1.0], abs=1e-3)
    assert "At the start, BMAG (1 when matched) x 1.0000" in tab._summary.text()
    assert "mismatched" not in tab._summary.text()
    rt = win.results_tab
    assert rt._source == "ring" and rt.open_plot("ring_ramp")
    rp = rt._popups["ring_ramp"].ramp_plots
    assert _n(rp.c_energy) == n_end and _n(rp.c_freq) == n_end
    assert "800 -> 806 MeV" in rp.note.text()
    # the separatrix of the turn's bucket at the last snapshot
    assert rt.open_plot("ring_phase")
    d = rt._popups["ring_phase"]
    d.snap_combo.setCurrentIndex(0)
    d.snap_combo.setCurrentIndex(d.snap_combo.count() - 1)
    assert "the separatrix of the turn's bucket" in d.phase_note.text()
    z, up = d.c_sep_up.getData()
    assert len(z) > 0 and np.nanmax(up) > 0
    # emittances normalized with each row's reference: the live strip's are the
    # run's own; the Results tab's (x' = px/pz) agree with them
    strip = tab._live_strip
    assert np.allclose(strip.c_ey.yData, res.emit_normalized[:, 1] * 1e6, rtol=0, atol=0)
    assert rt.open_plot("emit_n")
    cy = rt._popups["emit_n"]._cy
    assert np.allclose(cy.yData, res.emit_normalized[:, 1] * 1e6, rtol=1e-4, atol=0)
    assert list(cy.xData) == [float(t) for t in res.turns]
    assert not [m for m in win.message_boxes if "refresh failed" in str(m)]
    back = load_ring_hdf5(sorted(Path(tmp_path).glob("*_ring.h5"))[-1])
    assert back.ramp["kinetic_end_mev"] == res.ramp["kinetic_end_mev"]
    assert back.meta["config"]["ramp"]["energy"]["values"] == [800.0, 806.0]


def test_loading_a_project_resets_the_page(win):
    win.open_lattice(FIXTURE)
    tab = _ring_on(win)
    page = tab._programs
    page._e_kind.setCurrentIndex(3)
    page._e_table.set_points([0.0, 0.001], [800.0, 810.0])
    page._e_table.changed.emit()
    _set(page._gamma_t, "5.4")
    _set(page._t0, "2")
    page._p_excess.setCurrentIndex(1)
    page._add_group()
    page._para.item(0, 0).setText("0")
    page._para.item(0, 1).setText("5")
    assert page.ramp_config()["paraphasing"]
    data = win._collect_project_dict()
    data["ring"]["config"].pop("ramp")                    # (project B: no ramp)
    win._apply_project_dict(data, silent=True)
    page = win.ring_tab._programs
    assert page.ramp_config() is None
    assert page._gamma_t.text() == "" and page._t0.text() == "0"
    assert page._e_table.table.rowCount() == 0 and page._para.rowCount() == 0
    assert page._p_excess.currentData() == "cap"
    page._e_kind.setCurrentIndex(3)                       # nothing of project A comes back
    assert "ramp" not in win.state.ring_config or not win.state.ring_config["ramp"]


def test_steerers_belong_to_the_ramp(win):
    win.open_lattice(FIXTURE)
    tab = _ring_on(win, particles=20)
    page = tab._programs
    # saved even while another entry is not valid
    _sine(page)
    _set(page._gamma_t, "0.5")
    page._corr.setCurrentIndex(1)
    assert win.state.ring_config["corrector_mode"] == "fixed_field"
    # at fixed energy the run keeps the steerers' kicks
    page._e_kind.setCurrentIndex(0)
    tab._turns.setValue(1)
    tab._probes.setValue(0)
    tab._run_btn.click()
    assert tab._worker.ring_config.corrector_mode == "follow"
    win.wait_worker(tab._worker)
    # (the Run page's turns do not change during a run)
    tab._turns.setEnabled(False)
    assert tab._set_turns(5) is False and tab._turns.value() == 1
    tab._turns.setEnabled(True)


def test_a_new_lattice_drops_the_preview(win, tmp_path):
    win.open_lattice(FIXTURE)
    page = _ring_on(win)._programs
    page._e_kind.setCurrentIndex(1)                       # flat
    page.preview()
    w = page._worker
    win.wait_worker(w)
    assert page._preview is not None and not page._to_run_btn.isEnabled()   # (no end)
    # (the finished worker is kept until it is replaced or detached: its
    # thread may still be ending when its result arrives)
    assert page._worker is w
    assert _n(page._plots.c_energy) == 10                 # (the Run page's turns)
    other = tmp_path / "other.madx"
    other.write_text(FIXTURE.read_text())
    win.state.mark_project_clean()                        # (no "discard changes?" question)
    win.open_lattice(other)
    win.pump(0.2)
    assert page._preview is None and _n(page._plots.c_energy) == 0


def _type_in_cell(win, table, row, col, text):
    """The cell's editor opened and ``text`` typed into it (the edit not
    ended); returns the editor."""
    from PyQt6.QtTest import QTest
    item = table.item(row, col)
    table.setCurrentItem(item)
    table.editItem(item)
    win.pump(0.05)
    editor = table.indexWidget(table.model().index(row, col))
    assert editor is not None
    editor.selectAll()
    QTest.keyClicks(editor, text)
    return editor


def test_a_save_takes_what_is_being_typed(win):
    # a Save does not end an edit: a line edit's text and a table cell's open
    # editor go into the project (Run reads the page as it stands), and the
    # project is marked changed; nothing typed, a save rewrites nothing
    from PyQt6.QtTest import QTest
    from PyQt6.QtWidgets import QAbstractItemView
    win.open_lattice(FIXTURE)
    tab = _ring_on(win)
    page = tab._programs
    _sine(page)
    page._v_kind.setCurrentIndex(1)                       # a voltage table
    for t, v in (("0", "0.4"), ("3", "1.2"), ("30", "1.2")):
        page._v_table.add_row()
        r = page._v_table.table.rowCount() - 1
        page._v_table.table.item(r, 0).setText(t)
        page._v_table.table.item(r, 1).setText(v)
    page._add_group()
    page._para.item(0, 0).setText("0")
    win.state.mark_project_clean()
    win._collect_project_dict()
    assert not win.state.project_dirty
    # a line edit
    page._e_top.selectAll()
    QTest.keyClicks(page._e_top, "1200")
    assert page._e_top.isModified()
    ramp = win._collect_project_dict()["ring"]["config"]["ramp"]
    assert ramp["energy"]["end_kinetic_mev"] == 1200.0
    assert win.state.project_dirty and not page._e_top.isModified()
    # a cell of the voltage table, and one of the paraphasing groups
    for table, row, col, text in ((page._v_table.table, 1, 1, "0.9"), (page._para, 0, 1, "15")):
        win.state.mark_project_clean()
        _type_in_cell(win, table, row, col, text)
        assert table.state() == QAbstractItemView.State.EditingState
        ramp = win._collect_project_dict()["ring"]["config"]["ramp"]
        assert table.state() != QAbstractItemView.State.EditingState     # (the edit ended)
        assert table.item(row, col).text() == text and win.state.project_dirty
    assert ramp["voltage"]["values_mv"] == [0.4, 0.9, 1.2]
    assert ramp["paraphasing"] == [{"cavities": [0], "angle_deg": 15.0}]
    assert page.ramp_config() == ramp                     # (what a run takes)
    win.state.mark_project_clean()
    win._collect_project_dict()
    assert not win.state.project_dirty
