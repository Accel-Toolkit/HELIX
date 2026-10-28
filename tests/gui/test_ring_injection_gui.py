"""Ring mode in the GUI, milestone M5: the Injection page (the turns, the
macroparticles and the current, the foil, the injected beam, the chopper,
the bump and painting magnets) and a run with injection from the Run button
— the settings in the project and in the results file, the Results tab's
"Ring injection"."""
from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("PyQt6")

ROOT = Path(__file__).resolve().parents[2]
FIXTURE = ROOT / "tests" / "ring" / "fixtures" / "simple_booster.madx"


@pytest.fixture
def foil_deck(tmp_path):
    """The simple Booster with a marker 'foilm' at its start."""
    text = FIXTURE.read_text()
    line = "cell01 : line = (sa, "
    assert text.count(line) == 1
    p = tmp_path / "booster_foil.madx"
    p.write_text(text.replace("sa: drift, l=0.176", "foilm: marker;\nsa: drift, l=0.176")
                 .replace(line, "cell01 : line = (foilm, sa, "))
    return p


def _ring_on(win):
    act = win._toolbar._ring_action
    if not act.isChecked():
        act.trigger()
    win.pump(0.1)
    return win.ring_tab


def _set(edit, text):
    edit.setText(text)
    edit.editingFinished.emit()


def _fill(page):
    """A small valid injection: a Gaussian H- beam into the foil at foilm."""
    page._on.setCurrentIndex(1)
    page._turns.setValue(3)
    page._per_turn.setValue(30)
    _set(page._current, "2")
    page._at.setEditText("foilm")
    _set(page._thick, "600")
    for w, v in ((page._xmin, "-10"), (page._xmax, "10"), (page._ymin, "-10"),
                 (page._ymax, "10")):
        _set(w, v)
    page._kind.setCurrentIndex(2)                         # a Gaussian at the foil
    page._species.setCurrentIndex(1)                      # H-
    _set(page._kinetic, "800")
    _set(page._freq, "162.5")
    for w, v in ((page._enx, "0.3"), (page._eny, "0.3"), (page._bx, "30"), (page._by, "5")):
        _set(w, v)


def test_the_waveforms_typed_on_the_page(tmp_path):
    import csv

    from linac_gen_gui.interphase.tabs.ring_injection import parse_waveform, waveform_text
    assert parse_waveform("1") == {"kind": "constant", "value": 1.0}
    w = parse_waveform("sqrt 0 0.55 1 0")
    assert w == {"kind": "sqrt", "t_start_s": 0.0, "t_end_s": 0.00055, "start": 1.0, "end": 0.0}
    assert parse_waveform(waveform_text(w)) == w
    t = parse_waveform("table 0:1 0.1:0.5 0.2:0")
    assert t["times_s"] == [0.0, 0.0001, 0.0002] and t["values"] == [1.0, 0.5, 0.0]
    assert parse_waveform(waveform_text(t)) == t
    for bad in ("linear 0 1", "wiggle 1 2", "table 1", "1 2"):
        with pytest.raises(ValueError):
            parse_waveform(bad)
    with pytest.raises(ValueError, match="one number"):
        parse_waveform("constant 1 2")
    # a file: its relative name from the lattice file's folder, what is wrong said
    (tmp_path / "w.csv").write_text("t_s,value\n0,1\n0.001,0\n")
    assert parse_waveform("file w.csv", base_dir=str(tmp_path))["values"] == [1.0, 0.0]
    with pytest.raises(ValueError, match="cannot read"):
        parse_waveform("file missing.csv", base_dir=str(tmp_path))
    (tmp_path / "bad.csv").write_text("time,value\n0,1\n")
    with pytest.raises(ValueError, match="header"):
        parse_waveform("file bad.csv", base_dir=str(tmp_path))
    (tmp_path / "huge.csv").write_text('t_s,value\n"' + "x" * (csv.field_size_limit() + 10)
                                       + '",1\n')
    with pytest.raises(ValueError, match="not a CSV table"):
        parse_waveform("file huge.csv", base_dir=str(tmp_path))


def test_the_settings_go_into_the_project_and_come_back(win, foil_deck):
    win.open_lattice(foil_deck)
    tab = _ring_on(win)
    page = tab._injection
    assert page.injection_config() is None
    assert "Injection: off" in tab._ramp_line.text()
    assert "foilm" in [page._at.itemText(k) for k in range(page._at.count())]
    _fill(page)
    page._on_add_group()
    r = page._mags.rowCount() - 1
    page._mags.item(r, 1).setText("qsxx")                 # (a quadrupole: not a kicker)
    assert "Not valid" not in page._status.text()        # (checked when the run compiles)
    page._mags.item(r, 5).setText("linear 0 0.05 1 0")
    d = win._collect_project_dict()["ring"]["config"]["injection"]
    assert d["turns"] == 3 and d["particles_per_turn"] == 30 and d["current_ma"] == 2.0
    assert d["foil"]["at"] == "foilm" and d["foil"]["x_min_m"] == -0.01
    assert d["source"]["kind"] == "gaussian" and d["source"]["species"] == "H-"
    assert d["magnets"][0]["waveform"]["t_end_s"] == 5e-5
    assert "Injection: 3 turn(s) from turn 0, 30 macroparticles a turn from a Gaussian" \
        in tab._ramp_line.text()
    data = copy.deepcopy(win._collect_project_dict())
    page._on.setCurrentIndex(0)
    assert "injection" not in win.state.ring_config
    win._apply_project_dict(data, silent=True)
    page = win.ring_tab._injection
    assert page.injection_config() == data["ring"]["config"]["injection"]
    assert page._xmin.text() == "-10" and page._mags.rowCount() == 1
    # an entry that is not valid: reported, the project keeps the last valid settings
    _set(page._current, "-1")
    assert "Not valid" in page._status.text() and "current_ma" in page._status.text()
    assert win.state.ring_config["injection"]["current_ma"] == 2.0


def test_a_run_with_injection_from_the_run_button(win, foil_deck, tmp_path):
    from linac_gen.ring.results import load_ring_hdf5
    win.open_lattice(foil_deck)
    tab = _ring_on(win)
    _fill(tab._injection)
    assert tab._source.currentData() == "empty"           # (injection on: the ring starts empty)
    tab._probes.setValue(0)
    tab._turns.setValue(4)
    tab._run_btn.click()
    assert tab._worker.ring_config.injection["turns"] == 3
    win.wait_worker(tab._worker)
    res = win.state.ring_results
    assert res is not None and res.turns_done == 4
    assert res.ledger["injected"] == 90
    text = tab._summary.text()
    assert "Injection: 90 macroparticles" in text and "Foil foilm:" in text
    assert tab._live_strip.p_surv.getAxis("left").labelText == "particles circulating"
    rt = win.results_tab
    assert rt._source == "ring" and rt.open_plot("ring_injection")
    d = rt._popups["ring_injection"]
    # (a long run's turns: drawn at the screen's resolution)
    assert d.c_inj.opts["autoDownsample"] and d.c_inj.opts["clipToView"]
    # (each turn's injection at that turn counted from 1, where the
    # circulating count rises)
    assert list(d.c_inj.xData) == [1.0, 2.0, 3.0, 4.0]
    assert list(d.c_inj.yData) == [30.0, 30.0, 30.0, 0.0]
    assert list(d.c_circ.xData) == [0.0, 1.0, 2.0, 3.0, 4.0]
    assert list(d.c_circ.yData) == [float(v) for v in res.n_alive]
    assert d.i_map.image is not None and d.i_map.image.sum() > 0
    assert "Foil foilm:" in d.inj_note.text()
    assert not [m for m in win.message_boxes if "refresh failed" in str(m)]
    back = load_ring_hdf5(sorted(Path(tmp_path).glob("*_ring.h5"))[-1])
    assert back.injection["settings"]["turns"] == 3
    # an injection page that is not valid stops the run before it starts
    _set(tab._injection._thick, "abc")
    tab._run_btn.click()
    assert "Injection page" in tab._turn_label.text()


def test_a_projects_injection_that_is_not_valid_is_left_out(win, foil_deck):
    win.open_lattice(foil_deck)
    tab = _ring_on(win)
    _fill(tab._injection)
    assert tab._source.currentData() == "empty"
    data = copy.deepcopy(win._collect_project_dict())
    good = copy.deepcopy(data["ring"]["config"]["injection"])
    assert data["ring"]["config"]["source"] == "empty"
    # (an injection that is not valid: left out with its warning; the empty start
    # cannot stand without it — the Beam tab's beam)
    data["ring"]["config"]["injection"] = dict(good, current_ma=-1.0)
    win.message_boxes.clear()
    win._apply_project_dict(copy.deepcopy(data), silent=True)
    cfg = win.state.ring_config
    assert "injection" not in cfg and cfg["source"] == "beam"
    assert win.ring_tab._injection.injection_config() is None
    # (each valid on its own, not together: a space-charge intensity with injection)
    data["ring"]["config"]["injection"] = good
    data["ring"]["config"]["space_charge"] = {"intensity": 1e10, "kicks_per_turn": 12}
    win._apply_project_dict(copy.deepcopy(data), silent=True)
    cfg = win.state.ring_config
    assert "injection" not in cfg and cfg["space_charge"]["intensity"] == 1e10
    # (both valid: kept, with the empty start)
    data["ring"]["config"].pop("space_charge")
    win._apply_project_dict(copy.deepcopy(data), silent=True)
    cfg = win.state.ring_config
    assert cfg["injection"] == good and cfg["source"] == "empty"


def test_turning_injection_on_starts_the_ring_empty(win, foil_deck):
    win.open_lattice(foil_deck)
    tab = _ring_on(win)
    page = tab._injection
    assert tab._source.currentData() == "beam"
    page._on.setCurrentIndex(1)
    assert tab._source.currentData() == "empty"
    assert win.state.ring_config["source"] == "empty"
    assert "No beam at the start" in tab._beam_line.text()
    page._on.setCurrentIndex(0)
    assert tab._source.currentData() == "beam"
    assert win.state.ring_config["source"] == "beam"
    assert "particles of the Beam tab" in tab._beam_line.text()


def test_the_survival_with_injection_is_of_the_particles_received(win, foil_deck):
    from linac_gen_gui.interphase.tabs.ring_tab import _SOURCES
    win.open_lattice(foil_deck)
    tab = _ring_on(win)
    _fill(tab._injection)
    # (a beam at the start as well — the Beam tab's, matched to the ring: 100
    # particles, and 30 injected a turn)
    tab._source.setCurrentIndex([k for _l, k in _SOURCES].index("beam"))
    win.beam_tab._npart.setValue(100)
    tab._to_beam_btn.click()
    win.wait_worker(tab._matched_worker)
    tab._probes.setValue(0)
    tab._turns.setValue(4)
    tab._run_btn.click()
    win.wait_worker(tab._worker)
    res = win.state.ring_results
    assert res.ledger["injected"] == 190
    d = tab._live_strip
    assert d.p_surv.getAxis("left").labelText == "survival (% received)"
    y = np.asarray(d.c_surv.yData)
    assert y[0] == 100.0 and np.all(y <= 100.0) and np.all(y > 90.0)
    # (the Results tab's transmission: the same survival, turn by turn)
    view = win.results_tab._shown_results()
    assert np.array_equal(np.asarray(view.transmission), y)
    assert "particles into the ring" in tab._summary.text()


def test_the_page_keeps_what_it_does_not_show(win, foil_deck):
    win.open_lattice(foil_deck)
    tab = _ring_on(win)
    _fill(tab._injection)
    data = copy.deepcopy(win._collect_project_dict())
    inj = data["ring"]["config"]["injection"]
    inj["foil"]["map_bins"] = [20, 30]
    inj["source"]["draw"] = "in_order"                    # (a Gaussian's pool, in its order)
    inj["source"]["bunch_frequency_mhz"] = 40.625
    win._apply_project_dict(copy.deepcopy(data), silent=True)
    page = win.ring_tab._injection
    d = page.injection_config()
    assert d["foil"]["map_bins"] == [20, 30] and d["source"]["draw"] == "in_order"
    assert d["source"]["bunch_frequency_mhz"] == 40.625 and page._bunch.text() == "40.625"
    # (an entry finished with nothing changed: the project as it was)
    win.state.mark_project_clean()
    page._current.editingFinished.emit()
    assert not win.state.project_dirty
    assert win.state.ring_config["injection"]["foil"]["map_bins"] == [20, 30]


def test_the_magnet_table_reads_the_fringe_and_refuses_it_elsewhere(win, foil_deck):
    win.open_lattice(foil_deck)
    tab = _ring_on(win)
    page = tab._injection
    _fill(page)
    page._on_add_group()
    r = page._mags.rowCount() - 1
    page._mags.item(r, 1).setText("kb1")
    page._mags.item(r, 2).setText("dipole")
    page._mags.item(r, 6).setText("0.5, 11.3")
    page._mags.item(r, 7).setText("54.7, 0")
    g = page.injection_config()["magnets"][0]
    assert g["fint"] == 0.5 and g["hgap_m"] == 0.0113 and g["e1_rad"] == 0.0547
    page._mags.item(r, 2).setText("kick")
    with pytest.raises(ValueError, match="dipole"):
        page.injection_config()


def test_the_foil_choices_are_the_unique_markers(win, tmp_path):
    text = FIXTURE.read_text()
    p = tmp_path / "two_markers.madx"
    p.write_text(text.replace("sa: drift, l=0.176", "dupm: marker;\nonem: marker;\nsa: drift, "
                                                    "l=0.176")
                 .replace("cell01 : line = (sa, ", "cell01 : line = (dupm, onem, sa, ")
                 .replace("cell02 : line = (sa, ", "cell02 : line = (dupm, sa, "))
    win.open_lattice(p)
    tab = _ring_on(win)
    names = [tab._injection._at.itemText(k) for k in range(tab._injection._at.count())]
    assert "onem" in names and "dupm" not in names


def test_the_estimate_follows_the_optics(win, foil_deck):
    win.open_lattice(foil_deck)
    tab = _ring_on(win)
    page = tab._injection
    _fill(page)
    assert "compute the optics" in page._estimate.text()
    tab._optics_btn.click()
    win.wait_worker(tab._optics_worker)
    assert tab._optics is not None and tab._optics["rf_buckets"] == 84
    text = page._estimate.text()
    assert "over 84 filled bucket(s)" in text and "theta0" in text


def test_the_estimate_takes_the_foil_elements_settings():
    from types import SimpleNamespace

    from linac_gen_gui.interphase.tabs.ring_injection import _estimate_text

    from linac_gen.elements.foil import Foil
    res = {"optics": SimpleNamespace(length=474.202752, beta0=0.841810706), "rf_on": True,
           "rf_buckets": 84, "kinetic_mev": 800.0}
    d = {"notch_buckets": 0, "current_ma": 2.0, "turns": 10, "particles_per_turn": 20,
         "foil": {"at": "f1", "thickness_ug_cm2": None}}
    lat = SimpleNamespace(elements=[Foil(name="f1", thickness_ug_cm2=380.0)])
    assert "theta0 18.81 urad" in _estimate_text(res, d, lat)
    assert "give its thickness" in _estimate_text(res, d, SimpleNamespace(elements=[]))
    assert "theta0 24.36 urad" in _estimate_text(res, dict(d, foil={"at": "f1",
                                                               "thickness_ug_cm2": 600.0}), lat)


def test_the_run_memory_counts_the_injections_record():
    from linac_gen_gui.interphase.tabs.ring_tab import _injection_record_memory
    assert _injection_record_memory({"magnets": [{}, {}]}, 1000) == 2 * 8 * 16 * 1001


def test_a_project_with_a_null_injection_loads(win, foil_deck):
    win.open_lattice(foil_deck)
    tab = _ring_on(win)
    _fill(tab._injection)
    data = copy.deepcopy(win._collect_project_dict())
    assert data["ring"]["config"]["source"] == "empty"
    data["ring"]["config"]["injection"] = None             # (a hand-edited project)
    win._apply_project_dict(copy.deepcopy(data), silent=True)
    cfg = win.state.ring_config
    assert cfg["source"] == "beam" and "injection" not in cfg
    assert win.ring_tab._source.currentData() == "beam"


def test_a_save_takes_a_magnet_cell_being_edited(win, foil_deck):
    # a Save does not end an edit: the cell's open editor goes into the
    # project (what a run would take), the edit ends, the project is changed
    from PyQt6.QtTest import QTest
    from PyQt6.QtWidgets import QAbstractItemView
    win.open_lattice(foil_deck)
    tab = _ring_on(win)
    page = tab._injection
    _fill(page)
    page._on_add_group()
    r = page._mags.rowCount() - 1
    page._mags.item(r, 1).setText("foilm")
    win.state.mark_project_clean()
    item = page._mags.item(r, 5)
    page._mags.setCurrentItem(item)
    page._mags.editItem(item)
    win.pump(0.05)
    editor = page._mags.indexWidget(page._mags.model().index(r, 5))
    assert editor is not None
    editor.selectAll()
    QTest.keyClicks(editor, "0.5")
    assert page._mags.state() == QAbstractItemView.State.EditingState
    wave = win._collect_project_dict()["ring"]["config"]["injection"]["magnets"][0]["waveform"]
    assert wave["kind"] == "constant" and wave["value"] == 0.5
    assert page._mags.state() != QAbstractItemView.State.EditingState
    assert page._mags.item(r, 5).text() == "0.5" and win.state.project_dirty
    assert page.injection_config()["magnets"][0]["waveform"] == wave


def test_a_loaded_projects_injection_reaches_the_space_charge_estimate(win, foil_deck):
    # (the Space charge page is loaded before the Injection page: its
    # estimate follows the injection the project holds)
    win.open_lattice(foil_deck)
    tab = _ring_on(win)
    _fill(tab._injection)
    s = tab._space_charge
    s._model.setCurrentIndex(s._model.findData("pic"))
    data = copy.deepcopy(win._collect_project_dict())
    assert data["ring"]["config"]["injection"]
    assert data["ring"]["config"]["space_charge"]["intensity"] is None
    assert s._estimate.text().startswith("With injection")
    tab._injection._on.setCurrentIndex(0)                 # off: the Beam tab's charge
    assert not s._estimate.text().startswith("With injection")
    win._apply_project_dict(data, silent=True)            # the project's: on
    assert win.ring_tab._space_charge._estimate.text().startswith("With injection")
    assert "the circulating charge the injection builds" in win.ring_tab._ramp_line.text()
