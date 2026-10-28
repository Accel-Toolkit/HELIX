"""Ring mode in the GUI, milestone M4: the Space charge page (the model, the
intensity, the kicks, the grid, the pipe, the longitudinal kick; the
analytic tune shift with the optics of the Ring & Optics page) and a run
with space charge from the Run button — the settings in the project and in
the results file."""
from __future__ import annotations

import copy
from pathlib import Path

import pytest

pytest.importorskip("PyQt6")

ROOT = Path(__file__).resolve().parents[2]
FIXTURE = ROOT / "tests" / "ring" / "fixtures" / "simple_booster.madx"


def _ring_on(win, particles=100):
    act = win._toolbar._ring_action
    if not act.isChecked():
        act.trigger()
    win.pump(0.1)
    tab = win.ring_tab
    win.beam_tab._npart.setValue(particles)           # (a run takes the Beam tab's beam)
    return tab


def _use_matched(win, tab):
    """Ring & Optics → Use the matched beam in the Beam tab."""
    tab._to_beam_btn.click()
    win.wait_worker(tab._matched_worker)
    assert tab._optics_status.text().startswith("Beam tab: the beam matched"), \
        tab._optics_status.text()


def _set(edit, text):
    edit.setText(text)
    edit.editingFinished.emit()


def test_the_settings_go_into_the_project_and_come_back(win):
    win.open_lattice(FIXTURE)
    tab = _ring_on(win)
    page = tab._space_charge
    assert page.space_charge_config() is None
    assert "space_charge" not in (win.state.ring_config or {})
    assert "Space charge: off" in tab._ramp_line.text()
    page._model.setCurrentIndex(1)                        # particle-in-cell
    _set(page._intensity, "5e10")
    page._kicks.setValue(48)
    page._nx.setValue(32)
    page._pipe.setCurrentIndex(1)                         # round: no radius yet
    assert "Not valid" in page._status.text() and "pipe_a_m" in page._status.text()
    assert win.state.ring_config["space_charge"]["pipe"] == "open"   # (the last valid)
    _set(page._pipe_a, "35")
    d = win._collect_project_dict()["ring"]["config"]["space_charge"]
    assert d["model"] == "pic" and d["intensity"] == 5e10 and d["kicks_per_turn"] == 48
    assert d["grid"] == [32, 64] and d["pipe"] == "round" and d["pipe_a_m"] == pytest.approx(0.035)
    assert "particle-in-cell, 5e+10 particles" in tab._ramp_line.text()
    # loading it back fills the page; a project without it turns it off
    data = copy.deepcopy(win._collect_project_dict())
    page._model.setCurrentIndex(0)
    assert "space_charge" not in win.state.ring_config
    win._apply_project_dict(data, silent=True)
    page = win.ring_tab._space_charge
    assert page.space_charge_config() == data["ring"]["config"]["space_charge"]
    assert page._pipe_a.text() == "35" and page._nx.value() == 32
    data["ring"]["config"].pop("space_charge")
    win._apply_project_dict(data, silent=True)
    assert win.ring_tab._space_charge.space_charge_config() is None


def test_the_analytic_tune_shift_comes_with_the_optics(win):
    win.open_lattice(FIXTURE)
    tab = _ring_on(win)
    page = tab._space_charge
    page._model.setCurrentIndex(2)                        # frozen
    _set(page._intensity, "2e10")
    assert "compute the optics" in page._estimate.text()
    tab._six_d.setChecked(True)
    tab.compute_optics()
    win.wait_worker(tab._optics_worker)
    assert "Analytic tune shift at the centre of a Gaussian bunch" in page._estimate.text()
    assert "dQx -" in page._estimate.text()
    assert "kicks per betatron wavelength (Q 6.81" in page._estimate.text()
    page._kicks.setValue(40)
    assert "fewer than ten" in page._estimate.text()
    _set(page._intensity, "0")
    assert "Not valid" in page._status.text() and "leave it out" in page._status.text()
    assert win.state.ring_config["space_charge"]["intensity"] == 2e10   # (the last valid)


def test_a_run_with_space_charge_from_the_run_button(win, tmp_path):
    from linac_gen.ring.results import load_ring_hdf5
    win.open_lattice(FIXTURE)
    tab = _ring_on(win, particles=400)
    _use_matched(win, tab)
    page = tab._space_charge
    page._model.setCurrentIndex(1)
    _set(page._intensity, "1e10")
    page._kicks.setValue(24)
    page._nx.setValue(16)
    page._ny.setValue(16)
    tab._turns.setValue(3)
    tab._run_btn.click()
    assert tab._worker.ring_config.space_charge["kicks_per_turn"] == 24
    win.wait_worker(tab._worker)
    res = win.state.ring_results
    assert res is not None and res.turns_done == 3
    assert res.meta["space_charge"]["kicks"] == 24
    assert "Space charge: 2.5-D particle-in-cell, 1e+10 particles" in tab._summary.text()
    # (the line density's bins: "auto", those the run used)
    used = dict((int(t), int(b)) for t, b in res.meta["space_charge"]["z_bins_used"])
    lo, hi = min(used.values()), max(used.values())     # (at a turn, the last bins set)
    assert "line-density bins following the bunch, " + (
        f"{lo}" if lo == hi else f"{lo} to {hi}") in tab._summary.text()
    # "Ring tunes": the probe's shift from the bare working point (marked),
    # and the analytic shift from it
    rt = win.results_tab
    assert rt.open_plot("ring_tunes")
    d = rt._popups["ring_tunes"]
    assert "Analytic, at the centre of a Gaussian bunch" in d.tunes_text.text()
    q6 = res.optics["eigen_tunes_6d"]
    assert len(d.s_bare.data) == 1 and d.s_bare.data["x"][0] == q6[0]
    # (this beam's shift is beyond half an integer: not marked — the
    # fractional tunes cannot show it — and the text says so)
    q = res.meta["space_charge"]["laslett_gaussian"]
    assert max(abs(q[0]), abs(q[1])) > 0.5 and len(d.s_shift.data) == 0
    assert "beyond half an integer" in d.tunes_text.text()
    # "Ring bunch": the line-density bins the run used
    assert rt.open_plot("ring_long")
    b = rt._popups["ring_long"]
    assert not b.p_bins.isHidden() and set(b.c_bins.yData) <= set(used.values())
    x, y = b.c_bins.xData, b.c_bins.yData
    assert len(x) == len(y) + 1 and x[-1] == res.turns_done      # (edges: to the end)
    assert not [m for m in win.message_boxes if "refresh failed" in str(m)]
    back = load_ring_hdf5(sorted(Path(tmp_path).glob("*_ring.h5"))[-1])
    assert back.meta["space_charge"]["intensity"] == 1e10
    # an entry that is not valid stops the run before it starts
    page._long.setCurrentIndex(1)                         # g-factor, no g
    tab._run_btn.click()
    assert "Space charge page" in tab._turn_label.text()
    assert tab._worker is None or not tab._worker.isRunning()


def test_a_greyed_field_never_blocks_the_page(win):
    win.open_lattice(FIXTURE)
    page = _ring_on(win)._space_charge
    page._model.setCurrentIndex(1)
    _set(page._intensity, "1e10")
    page._pipe.setCurrentIndex(2)                         # rectangle
    _set(page._pipe_a, "60")
    _set(page._pipe_b, "abc")
    assert "half height" in page._status.text()
    page._pipe.setCurrentIndex(1)                         # round: the height does not apply
    assert not page._pipe_b.isEnabled() and page._status.text() == ""
    assert win.state.ring_config["space_charge"]["pipe"] == "round"
    _set(page._r_ref, "-1")                               # (greyed: a pipe is set)
    assert page._status.text() == ""
    page._pipe.setCurrentIndex(0)                         # open: now it applies
    assert "reference_radius_m" in page._status.text()


def test_the_page_keeps_a_projects_exact_values(win):
    win.open_lattice(FIXTURE)
    tab = _ring_on(win)
    data = win._collect_project_dict()
    sc = {"model": "pic", "intensity": 1.2345678901234e10, "kicks_per_turn": 1_000_000,
          "box_sigma": 4.125, "z_smoothing_bins": 1500.0, "pipe": "round", "pipe_a_m": 0.0547}
    data["ring"]["config"]["space_charge"] = sc
    win._apply_project_dict(copy.deepcopy(data), silent=True)
    page = win.ring_tab._space_charge
    win.state.mark_project_clean()
    got = page.space_charge_config()
    for k, v in sc.items():
        assert got[k] == v, k
    # a focus leaving a field with nothing typed changes nothing
    for w in (page._intensity, page._pipe_a, page._r_ref):
        w.editingFinished.emit()
    assert not win.state.project_dirty
    assert win.state.ring_config["space_charge"]["pipe_a_m"] == 0.0547
    del tab


def test_an_invalid_space_charge_section_keeps_the_other_ring_settings(win, capsys):
    win.open_lattice(FIXTURE)
    tab = _ring_on(win)
    tab._turns.setValue(77)
    data = win._collect_project_dict()
    data["ring"]["config"].update({"n_turns": 77, "model": "curvilinear",
                                   "space_charge": {"grid": [4, 4]}})
    capsys.readouterr()
    win._apply_project_dict(copy.deepcopy(data), silent=True)
    cfg = win.state.ring_config
    assert cfg["n_turns"] == 77 and cfg["model"] == "curvilinear" and "space_charge" not in cfg
    # (a silent load prints its warnings)
    assert "space-charge settings are not valid" in capsys.readouterr().out


def test_the_estimate_asks_for_6d_optics_on_a_ring_with_rf(win):
    win.open_lattice(FIXTURE)
    tab = _ring_on(win)
    page = tab._space_charge
    page._model.setCurrentIndex(1)
    _set(page._intensity, "1e10")
    tab._six_d.setChecked(False)
    tab.compute_optics()
    win.wait_worker(tab._optics_worker)
    assert "6-D" in page._estimate.text() and "dQx" not in page._estimate.text()
    # the kicks a request places, not the request
    page._kicks.setValue(5000)
    tab._six_d.setChecked(True)
    tab.compute_optics()
    win.wait_worker(tab._optics_worker)
    assert "of the 5000 asked: element boundaries" in page._estimate.text()


def test_an_invalid_ramp_keeps_the_valid_space_charge(win, capsys):
    win.open_lattice(FIXTURE)
    _ring_on(win)
    data = win._collect_project_dict()
    sc = {"model": "frozen", "intensity": 3e10, "kicks_per_turn": 40}
    data["ring"]["config"].update({"n_turns": 55, "space_charge": sc,
                                   "ramp": {"energy": {"kind": "nonsense"}}})
    capsys.readouterr()
    win._apply_project_dict(copy.deepcopy(data), silent=True)
    cfg = win.state.ring_config
    assert cfg["n_turns"] == 55 and "ramp" not in cfg
    assert cfg["space_charge"]["model"] == "frozen" and cfg["space_charge"]["intensity"] == 3e10
    out = capsys.readouterr().out
    assert "ramp settings are not valid" in out and "space-charge settings" not in out
    # both not valid: both left out, each with its own warning; the rest kept
    data["ring"]["config"]["space_charge"] = {"grid": [4, 4]}
    win._apply_project_dict(copy.deepcopy(data), silent=True)
    cfg = win.state.ring_config
    assert cfg["n_turns"] == 55 and "ramp" not in cfg and "space_charge" not in cfg
    out = capsys.readouterr().out
    assert "ramp settings are not valid" in out and "space-charge settings are not valid" in out
    # the ring settings themselves not valid: the defaults
    data["ring"]["config"]["n_turns"] = -3
    win._apply_project_dict(copy.deepcopy(data), silent=True)
    assert "the defaults are used" in capsys.readouterr().out


def test_a_save_rewrites_nothing_unless_an_entry_was_typed(win):
    from PyQt6.QtTest import QTest
    win.open_lattice(FIXTURE)
    tab = _ring_on(win)
    data = win._collect_project_dict()
    # (settings the page does not show as they are: a greyed half height and
    # g, more decimals than the spin box keeps)
    sc = {"model": "pic", "intensity": 1e10, "box_sigma": 4.1234567891, "pipe": "round",
          "pipe_a_m": 0.05, "pipe_b_m": 0.02, "longitudinal": "potential", "g_factor": 1.7}
    data["ring"]["config"]["space_charge"] = sc
    win._apply_project_dict(copy.deepcopy(data), silent=True)
    win.state.mark_project_clean()
    page = win.ring_tab._space_charge
    saved = win._collect_project_dict()["ring"]["config"]["space_charge"]
    for k, v in sc.items():
        assert saved[k] == v, k
    assert not win.state.project_dirty
    # an entry being typed when the project is saved is taken
    page._intensity.selectAll()
    QTest.keyClicks(page._intensity, "2e10")
    assert page._intensity.isModified()
    saved = win._collect_project_dict()["ring"]["config"]["space_charge"]
    assert saved["intensity"] == 2e10 and not page._intensity.isModified()
    assert win.state.project_dirty
    del tab


def test_the_line_density_bins_are_auto_unless_a_number_is_given(win):
    win.open_lattice(FIXTURE)
    tab = _ring_on(win)
    page = tab._space_charge
    page._model.setCurrentIndex(1)                        # particle-in-cell
    _set(page._intensity, "5e10")
    assert page._z_auto.isChecked() and not page._z_bins.isEnabled()
    assert page.space_charge_config()["z_bins"] == "auto"
    assert win.state.ring_config["space_charge"]["z_bins"] == "auto"
    page._z_auto.setChecked(False)
    assert page._z_bins.isEnabled()
    page._z_bins.setValue(512)
    assert page.space_charge_config()["z_bins"] == 512
    assert win.state.ring_config["space_charge"]["z_bins"] == 512
    # a project's number comes back unticked with it; "auto" ticked
    data = copy.deepcopy(win._collect_project_dict())
    page._z_auto.setChecked(True)
    win._apply_project_dict(data, silent=True)
    page = win.ring_tab._space_charge
    assert not page._z_auto.isChecked() and page._z_bins.value() == 512
    data["ring"]["config"]["space_charge"]["z_bins"] = "auto"
    win._apply_project_dict(data, silent=True)
    page = win.ring_tab._space_charge
    assert page._z_auto.isChecked() and not page._z_bins.isEnabled()
    assert page.space_charge_config()["z_bins"] == "auto"
    # off: both greyed
    page._model.setCurrentIndex(0)
    assert not page._z_auto.isEnabled() and not page._z_bins.isEnabled()


def test_the_memory_estimate_counts_the_frozen_models_line_densities():
    from linac_gen_gui.interphase.tabs.ring_tab import _space_charge_memory

    from linac_gen.ring.sc import solver
    from linac_gen.ring.sc.config import SpaceChargeConfig
    pic = SpaceChargeConfig.from_dict({"model": "pic", "kicks_per_turn": 1000,
                                       "z_bins": 65536}).to_dict()
    frozen = dict(pic, model="frozen")
    extra = _space_charge_memory(frozen) - _space_charge_memory(pic)
    assert extra == solver.FROZEN_CACHE_BYTES + 1000 * 2 * 8 * 65536
    assert _space_charge_memory(None) == 0
    # (of the kicks a ring of that many elements can place)
    extra = _space_charge_memory(frozen, n_starts=714) - _space_charge_memory(pic, n_starts=714)
    assert extra == solver.FROZEN_CACHE_BYTES + 714 * 2 * 8 * 65536
    # ("auto": the estimate takes the most bins they may become)
    auto = dict(frozen, z_bins="auto")
    assert _space_charge_memory(auto) - _space_charge_memory(pic) == \
        solver.FROZEN_CACHE_BYTES + 1000 * 2 * 8 * solver.AUTO_NZ_MAX


def test_a_project_value_beyond_the_pages_range_is_left_out_whole(win, capsys):
    # (a value no spin box holds: the section is left out with its warning —
    # never a half-done load)
    win.open_lattice(FIXTURE)
    _ring_on(win)
    data = win._collect_project_dict()
    for bad in ({"kicks_per_turn": 3_000_000_000}, {"refresh_turns": 2 ** 31}):
        data["ring"]["config"].update({"n_turns": 44, "space_charge": dict(
            {"model": "pic", "intensity": 2e10}, **bad)})
        capsys.readouterr()
        win._apply_project_dict(copy.deepcopy(data), silent=True)
        cfg = win.state.ring_config
        assert cfg["n_turns"] == 44 and "space_charge" not in cfg, bad
        assert "space-charge settings are not valid" in capsys.readouterr().out
        assert win.ring_tab._space_charge.space_charge_config() is None


def test_typing_one_entry_keeps_the_others_exact(win):
    from PyQt6.QtTest import QTest
    win.open_lattice(FIXTURE)
    _ring_on(win)
    data = win._collect_project_dict()
    sc = {"model": "pic", "intensity": 1e10, "box_sigma": 4.1234567891,
          "z_smoothing_bins": 2.123456789}
    data["ring"]["config"]["space_charge"] = sc
    win._apply_project_dict(copy.deepcopy(data), silent=True)
    page = win.ring_tab._space_charge
    page._intensity.selectAll()
    QTest.keyClicks(page._intensity, "3e10")
    page._intensity.editingFinished.emit()
    saved = win._collect_project_dict()["ring"]["config"]["space_charge"]
    assert saved["intensity"] == 3e10
    assert saved["box_sigma"] == 4.1234567891 and saved["z_smoothing_bins"] == 2.123456789
    # a box changed takes its own value
    page._box_sigma.setValue(5.5)
    assert win.state.ring_config["space_charge"]["box_sigma"] == 5.5


def test_a_reference_radius_typed_with_an_open_pipe_stays_when_a_pipe_is_chosen(win):
    # (greyed while a pipe is chosen, it is kept for the open pipe: in the
    # project, through a save and a load, and back on the page)
    win.open_lattice(FIXTURE)
    tab = _ring_on(win)
    s = tab._space_charge
    s._model.setCurrentIndex(s._model.findData("pic"))
    _set(s._r_ref, "2.5")
    assert win.state.ring_config["space_charge"]["reference_radius_m"] == 2.5
    s._pipe.setCurrentIndex(s._pipe.findData("round"))
    _set(s._pipe_a, "30")
    assert s._status.text() == ""
    assert win.state.ring_config["space_charge"]["reference_radius_m"] == 2.5
    data = copy.deepcopy(win._collect_project_dict())
    win._apply_project_dict(data, silent=True)
    s = win.ring_tab._space_charge
    assert s._r_ref.text() == "2.5"
    s._pipe.setCurrentIndex(s._pipe.findData("open"))
    assert s._r_ref.text() == "2.5"
    assert win.state.ring_config["space_charge"]["reference_radius_m"] == 2.5
    # one that is not valid is not kept: a pipe chosen after it takes the last
    # valid one, and the page is valid
    _set(s._r_ref, "-1")
    assert "reference_radius_m" in s._status.text()
    s._pipe.setCurrentIndex(s._pipe.findData("round"))
    assert s._status.text() == ""
    assert win.state.ring_config["space_charge"]["reference_radius_m"] == 2.5


def test_with_injection_the_charge_is_the_circulating_charge_it_builds(win):
    # (a run with injection takes the injected current's charge, not the Beam
    # tab's, and gives the tune shift at its end: the Run page's line and the
    # page's estimate say so; an intensity given is refused by such a run)
    win.open_lattice(FIXTURE)
    tab = _ring_on(win)
    s = tab._space_charge
    s._model.setCurrentIndex(s._model.findData("pic"))
    assert "the Beam tab's intensity" in tab._ramp_line.text()
    assert s._estimate.text().startswith("The analytic tune shift: compute the optics")
    tab._injection._on.setCurrentIndex(1)                 # injection on
    line = tab._ramp_line.text()
    assert ("particle-in-cell, the circulating charge the injection builds (its tune shift is "
            "reported at the end of the run)") in line
    assert "Beam tab's intensity" not in line
    assert s._estimate.text() == ("With injection, the space charge is that of the "
                                  "circulating charge the injection builds: a run reports "
                                  "its analytic tune shift at the end.")
    _set(s._intensity, "5e10")
    assert "5e+10 particles in the bunch, which a run with injection refuses" \
        in tab._ramp_line.text()
    tab._injection._on.setCurrentIndex(0)                 # injection off again
    assert "particle-in-cell, 5e+10 particles in the bunch, about" in tab._ramp_line.text()
    assert s._estimate.text().startswith("The analytic tune shift: compute the optics")
