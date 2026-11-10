"""An error study with the envelope, and the surrogates in it.

``ErrorStudy(mode="envelope")`` runs each seed's envelope from the same draws as the
multiparticle study.  It uses the registered surrogates when its design lattice with them agrees
with the native run, and says per seed, counted in its own thread, how many times the network
answered and which traversals it left to the native element, and why (warning when more than
10 % of the traversals the network would answer fell outside the trained range).  Its orbit
correction runs native.  Its first three seeds run again natively, and it reports their largest
exit difference.  Each seed's exit values are given as run and, for a seed the network answered
as it answered the design lattice, as the native design value plus the network's change from
its own design value.  A stop ends the design check and skips the native check.  A
multiparticle study runs native, without loading the surrogate package.
"""
import contextlib
import copy
import warnings

import numpy as np
import pytest

from linac_gen.core.config import BeamConfig
from linac_gen.elements.drift import Drift
from linac_gen.errors.error_model import ErrorStudy
from linac_gen.surrogates import registry
from linac_gen.surrogates.base import OutOfScopeError
from linac_gen.surrogates.study_check import EXIT_KEYS
from linac_gen.tracking.envelope import EnvelopeSolver
from tests.surrogates.accuracy_helpers import W, exact, native, off, pinned
from tests.surrogates.test_frequency_jump import F_NEW, _cavity, _lattice, _magnet, _ref


@pytest.fixture(autouse=True)
def _clean_registry():
    registry.clear()
    yield
    registry.clear()


def _line():
    return _lattice(Drift("D0", 100.0), _cavity(p_flag=0), Drift("D1", 100.0))


def _beam(**kw):
    kw = {"species": "proton", "energy": W, "frequency": F_NEW, "current": 0.0, "n_particles": 32,
          "emit_nx": 0.2, "alpha_x": 0.0, "beta_x": 1.0, "emit_ny": 0.2, "alpha_y": 0.0,
          "beta_y": 1.0, "emit_z": 0.2, "alpha_z": 0.0, "beta_z": 1.0, **kw}
    return BeamConfig(**kw)


def _study(lat, mode="envelope", n=4, phase_sigma=1.0):
    study = ErrorStudy(copy.deepcopy(lat), _beam(), n_seeds=n, mode=mode)
    study.add_error("MAP", "phase_offset", sigma=phase_sigma, cutoff=3.0)
    study.add_error("MAP", "voltage_rel", sigma=0.01, cutoff=3.0)
    study.add_error("MAP", "dx", sigma=0.2, cutoff=3.0)
    return study


def _run(study):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return study.run()


def test_the_mode_and_the_envelope_solver_are_each_one_of_two():
    with pytest.raises(ValueError, match="mode must be 'mp' or 'envelope'"):
        ErrorStudy(_line(), _beam(), mode="matrix")
    with pytest.raises(ValueError, match="env_solver must be 'matrix' or 'sacherer', not 'pic'"):
        ErrorStudy(_line(), _beam(), mode="envelope", env_solver="pic")
    assert ErrorStudy(_line(), _beam(), mode="envelope", env_solver="Sacherer").env_solver == \
        "sacherer"


def test_an_envelope_study_draws_what_a_multiparticle_study_draws():
    env, mp = _study(_line()), _study(_line(), mode="mp", n=2)
    for study in (env, mp):
        study.add_beam_error("centroid_x", sigma=0.1)
        study.add_beam_error("emit_nx_rel", sigma=0.05)
    _run(env)
    _run(mp)
    for seed in (0, 1):
        assert env.applied_log[seed] == mp.applied_log[seed]
        assert env.beam_applied_log[seed] and env.beam_applied_log[seed] == mp.beam_applied_log[seed]


def test_each_seed_is_its_envelope():
    study = _study(_line())
    results = _run(study)
    assert results.mode == "envelope" and results.n_seeds == 4
    probe = _study(_line())
    for seed, res in enumerate(results._recorders):
        lat, cfg = probe._apply_errors(seed), probe._apply_beam_errors(seed)
        from linac_gen.cli.common import _envelope_initial, build_ref
        ref = build_ref(cfg)
        want = EnvelopeSolver(lat, ref, _envelope_initial(cfg, ref), current=cfg.current).run()
        np.testing.assert_array_equal(np.asarray(res.sigma_matrix), np.asarray(want.sigma_matrix))
        assert list(res.transmission) == [100.0] * len(res.s)        # an envelope loses nothing
    assert results.mean("sigma_x").shape == np.asarray(results._recorders[0].sigma_x).shape


def test_surrogates_that_agree_are_used_and_three_seeds_are_checked_natively():
    lat = _line()
    el = lat.elements[1]
    surr = exact(el, params=(("phase", -60.0, 0.0), ("scale", 0.9, 1.1)))
    registry.register(surr)
    results = _run(_study(lat))
    assert results.surrogates.use
    assert results.surrogate_calls == [1, 1, 1, 1]        # each seed's cavity, errors folded in
    assert results.native_traversals == [{}, {}, {}, {}]
    check = results.native_check
    assert check["seeds"] == [0, 1, 2] and check["largest"] < 1e-9
    assert check["seed"] in (0, 1, 2) and check["key"] in EXIT_KEYS      # named even at 0
    for raw, corrected in zip(results.exit_raw, results.exit_corrected, strict=True):
        for k in EXIT_KEYS:                               # an exact network: no design offset
            assert corrected[k] == pytest.approx(raw[k], rel=1e-9)


def test_the_corrected_value_is_the_native_design_plus_the_networks_change():
    """(A magnet: its map follows no running phase, so weights with no input answer.)"""
    lat = _lattice(Drift("D0", 100.0), _magnet(), Drift("D1", 100.0))
    el = lat.elements[1]
    registry.register(pinned(el, off(native(el), 2, 2, 1.0005)))   # passes the 0.5 % check
    study = ErrorStudy(copy.deepcopy(lat), _beam(), n_seeds=2, mode="envelope")
    study.add_error("MAP", "dx", sigma=0.2, cutoff=3.0)              # what pinned weights follow
    results = _run(study)
    design = results.surrogates
    assert design.use and results.surrogate_calls == [1, 1]
    # the design values are the design lattice's, natively and with the network, and differ
    from linac_gen.cli.common import run_envelope_sim
    from linac_gen.surrogates.study_check import exit_values
    with registry.suspended():
        nat = exit_values(run_envelope_sim(copy.deepcopy(lat), _beam()))
    sur = exit_values(run_envelope_sim(copy.deepcopy(lat), _beam()))
    assert design.native == pytest.approx(nat, rel=1e-12)
    assert design.surrogate == pytest.approx(sur, rel=1e-12)
    assert abs(design.surrogate["sigma_y"] / design.native["sigma_y"] - 1.0) > 1e-5
    for raw, corrected in zip(results.exit_raw, results.exit_corrected, strict=True):
        for k in EXIT_KEYS:
            assert corrected[k] == pytest.approx(design.native[k] + raw[k] - design.surrogate[k],
                                                 rel=1e-12)
    # the seeds run again natively see the network's difference, and name where it is largest
    check = results.native_check
    assert check["seeds"] == [0, 1] and check["largest"] > 1e-5
    assert check["seed"] in (0, 1) and check["key"] in EXIT_KEYS


def test_surrogates_that_disagree_leave_the_study_native():
    lat = _line()
    native_run = _run(_study(lat))
    el = lat.elements[1]
    surr = pinned(el, off(native(el), 0, 0, 1.05))
    registry.register(surr)
    results = _run(_study(lat))
    assert not results.surrogates.use and results.surrogates.note.startswith("surrogates not used")
    assert surr.nn_calls == results.surrogates.nn_calls      # the check's answers only
    assert results.surrogate_calls == [] and results.native_check is None
    for a, b in zip(results._recorders, native_run._recorders, strict=True):
        np.testing.assert_array_equal(np.asarray(a.sigma_matrix), np.asarray(b.sigma_matrix))


def test_seeds_outside_the_trained_range_warn():
    """(The cavity first in the line: its phase input is its phase, the running phase at its
    entrance being the training's.)"""
    lat = _lattice(_cavity(p_flag=0), Drift("D1", 100.0))
    el = lat.elements[0]
    # a phase range of ±0.5° around the design: most seeds' phase errors (σ 1°) leave it
    registry.register(exact(el, params=(("phase", el.phase - 0.5, el.phase + 0.5),
                                        ("scale", 0.9, 1.1))))
    with pytest.warns(UserWarning, match="more than 10 % of the traversals the surrogates would "
                                         "answer fell outside their trained range") as seen:
        results = _study(lat, n=6).run()
    assert sum(r.get("outside its range", 0) for r in results.native_traversals) >= 3
    assert max(results.outside_range) == 1.0
    assert [w.filename for w in seen if "trained range" in str(w.message)] == [__file__]


def test_the_out_of_range_part_counts_what_the_network_would_answer():
    """Its answers and the traversals outside its range: traversals it refused for another
    reason (an accuracy check failed, a setting changed) are not ones it would answer."""
    from linac_gen.errors.error_model import OUT_OF_RANGE_WARN, outside_fractions
    got = outside_fractions([4, 19, 0, 3], [{"outside its range": 1, "accuracy check": 30},
                                            {"outside its range": 1},
                                            {},
                                            {"its kb changed": 5}])
    assert got == [0.2, 0.05, 0.0, 0.0]
    assert [f > OUT_OF_RANGE_WARN for f in got] == [True, False, False, False]


@pytest.fixture
def mp_fast_path():
    mp, fast = registry.is_mp_enabled(), registry.is_fast_path_enabled()
    registry.set_mp_enabled(True)
    registry.set_fast_path_enabled(True)
    yield
    registry.set_mp_enabled(mp)
    registry.set_fast_path_enabled(fast)


def test_a_multiparticle_study_runs_native(mp_fast_path):
    """Even with the multiparticle surrogates switched on, which a toolbar run would use."""
    lat = _line()
    el = lat.elements[1]
    surr = exact(el, params=(("phase", -60.0, 0.0), ("scale", 0.9, 1.1)))
    registry.register(surr)
    results = _run(_study(lat, mode="mp", n=2))
    assert not results.surrogates.use and "multiparticle" in results.surrogates.note
    assert surr.nn_calls == 0 and results.surrogate_calls == []


def test_without_surrogates_an_envelope_study_says_nothing_of_them():
    results = _run(_study(_line(), n=2))
    assert results.surrogates.note == "" and results.native_check is None


def test_a_seed_the_network_answered_otherwise_has_no_corrected_value():
    """A scale error the weights do not take: each seed runs the magnet native, so the
    network's design offset is not in its exit value and is not taken out of it."""
    lat = _lattice(Drift("D0", 100.0), _magnet(), Drift("D1", 100.0))
    el = lat.elements[1]
    registry.register(pinned(el, off(native(el), 2, 2, 1.0005)))
    study = ErrorStudy(copy.deepcopy(lat), _beam(), n_seeds=3, mode="envelope")
    study.add_error("MAP", "scale", sigma=0.01, cutoff=3.0)
    results = _run(study)
    assert results.surrogates.use and results.surrogate_calls == [0, 0, 0]
    assert results.native_traversals == [{"its scale changed": 1}] * 3
    assert results.exit_corrected == [None, None, None] and len(results.exit_raw) == 3
    # native both times: no difference at all, and still a seed and a quantity named
    check = results.native_check
    assert check["largest"] == 0.0 and check["seed"] == 0 and check["key"] in EXIT_KEYS


def test_a_sacherer_study_runs_native_with_surrogates_registered():
    lat = _line()
    el = lat.elements[1]
    surr = exact(el, params=(("phase", -60.0, 0.0), ("scale", 0.9, 1.1)))
    registry.register(surr)
    study = ErrorStudy(copy.deepcopy(lat), _beam(), n_seeds=2, mode="envelope",
                       env_solver="sacherer")
    study.add_error("MAP", "dx", sigma=0.2, cutoff=3.0)
    results = _run(study)
    assert results.n_seeds == 2 and not results.surrogates.use
    assert results.surrogates.note == "surrogates not used: the Sacherer envelope does not use them"
    assert surr.nn_calls == 0 and results.surrogate_calls == []


def test_the_seed_counts_are_the_studys_own():
    """Another thread running the same weights meanwhile (a toolbar run) lands in none of the
    seeds' counts.  (Pinned weights: the exact test network is not thread-safe.)"""
    import threading
    lat = _lattice(Drift("D0", 100.0), _magnet(), Drift("D1", 100.0))
    el = lat.elements[1]
    surr = pinned(el, native(el))
    registry.register(surr)
    done = threading.Event()
    other_el = copy.deepcopy(el)
    other_el.kb = 1.3                                    # its queries refuse: a setting changed

    def toolbar():
        while not done.is_set():
            surr.fitted_matrix(_ref(F_NEW, W))
            with contextlib.suppress(OutOfScopeError):
                surr.bound_to(other_el).fitted_matrix(_ref(F_NEW, W))
    other = threading.Thread(target=toolbar)
    study = ErrorStudy(copy.deepcopy(lat), _beam(), n_seeds=5, mode="envelope")
    study.add_error("MAP", "dx", sigma=0.2, cutoff=3.0)
    other.start()
    try:
        results = _run(study)
    finally:
        done.set()
        other.join()
    assert results.surrogate_calls == [1] * 5 and results.native_traversals == [{}] * 5
    assert results.outside_range == [0.0] * 5
    # the other thread did answer and refuse meanwhile, on the same weights
    assert surr.nn_calls > 5 + results.surrogates.nn_calls and surr.refusals["its kb changed"] > 0


def test_the_orbit_correction_runs_native(monkeypatch):
    from linac_gen.errors import correction
    seen = []

    def fake(lattice, factory, **kw):
        seen.append(registry.is_suspended())
        return {"status": "ok"}
    monkeypatch.setattr(correction, "run_correction_from_lattice", fake)
    lat = _line()
    el = lat.elements[1]
    registry.register(exact(el, params=(("phase", -60.0, 0.0), ("scale", 0.9, 1.1))))
    study = _study(copy.deepcopy(lat), n=2)
    study.enable_correction(reading_backend="envelope")
    results = _run(study)
    assert results.surrogates.use and seen == [True, True]
    assert results.surrogate_calls == [1, 1]


def test_a_stop_ends_the_design_check_and_skips_the_native_check(monkeypatch):
    from linac_gen.cli import common
    lat = _line()
    el = lat.elements[1]
    surr = exact(el, params=(("phase", -60.0, 0.0), ("scale", 0.9, 1.1)))
    registry.register(surr)
    runs = []
    real = common.run_envelope_sim
    monkeypatch.setattr(common, "run_envelope_sim",
                        lambda *a, **k: runs.append(1) or real(*a, **k))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        early = _study(copy.deepcopy(lat), n=3).run(should_stop=lambda: True)
    assert early.n_seeds == 0 and surr.nn_calls == 0 and runs == []     # not even begun
    assert early.surrogates.note == "surrogates not used: the study was stopped"
    flag = {"stop": False}

    def progress(i, n):
        flag["stop"] = i == n                            # after the last seed
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        late = _study(copy.deepcopy(lat), n=3).run(should_stop=lambda: flag["stop"],
                                                   progress_cb=progress)
    assert late.n_seeds == 3 and late.surrogates.use and late.surrogate_calls == [1, 1, 1]
    assert late.native_check is None and len(late.exit_raw) == 3


def test_an_envelope_study_with_space_charge_runs_native_and_says_why():
    """With a beam current the envelope steps through the map in slices: the whole-element
    weights are never asked, and each seed is the native envelope with space charge."""
    lat = _line()
    el = lat.elements[1]
    registry.register(exact(el, params=(("phase", -60.0, 0.0), ("scale", 0.9, 1.1))))
    study = ErrorStudy(copy.deepcopy(lat), _beam(current=5.0), n_seeds=2, mode="envelope")
    study.add_error("MAP", "dx", sigma=0.2, cutoff=3.0)
    results = _run(study)
    assert not results.surrogates.use
    assert results.surrogates.note == ("surrogates not used: none answered on the design lattice "
                                       "(none was asked: with space charge the envelope steps "
                                       "through each field map in slices)")
    from linac_gen.cli.common import _envelope_initial, build_ref
    probe = ErrorStudy(copy.deepcopy(lat), _beam(current=5.0), n_seeds=2, mode="envelope")
    probe.add_error("MAP", "dx", sigma=0.2, cutoff=3.0)
    for seed, res in enumerate(results._recorders):
        seed_lat, cfg = probe._apply_errors(seed), probe._apply_beam_errors(seed)
        ref = build_ref(cfg)
        want = EnvelopeSolver(seed_lat, ref, _envelope_initial(cfg, ref), current=5.0).run()
        np.testing.assert_array_equal(np.asarray(res.sigma_matrix), np.asarray(want.sigma_matrix))


def test_a_multiparticle_study_leaves_the_surrogate_package_unloaded():
    """Nothing can be registered before the registry is imported: a multiparticle study of a
    line with no field map leaves it (and torch) unloaded."""
    import subprocess
    import sys
    from pathlib import Path
    code = (
        "import sys\n"
        "from linac_gen.core.config import BeamConfig\n"
        "from linac_gen.core.lattice import Lattice\n"
        "from linac_gen.elements.drift import Drift\n"
        "from linac_gen.elements.quadrupole import Quadrupole\n"
        "from linac_gen.errors.error_model import ErrorStudy\n"
        "lat = Lattice()\n"
        "for e in (Drift('D0', 100.0), Quadrupole('Q1', 50.0, 5.0, 20.0), Drift('D1', 100.0)):\n"
        "    lat.add(e)\n"
        "cfg = BeamConfig(species='proton', energy=5.0, frequency=325.0, current=0.0,\n"
        "                 n_particles=16)\n"
        "study = ErrorStudy(lat, cfg, n_seeds=1)\n"
        "study.add_error('Q1', 'dx', sigma=0.1)\n"
        "before = set(sys.modules)\n"
        "res = study.run()\n"
        "new = sorted(m for m in set(sys.modules) - before\n"
        "             if m == 'torch' or m.startswith('linac_gen.surrogates'))\n"
        "print('NEW', new, res.surrogates.use, repr(res.surrogates.note))\n")
    repo = Path(__file__).resolve().parents[2]
    env = {k: v for k, v in __import__("os").environ.items() if k != "HELIX_QSETTINGS_DIR"}
    env["PYTHONPATH"] = f"{repo}{__import__('os').pathsep}{repo / 'gui'}"
    out = subprocess.run([sys.executable, "-c", code], cwd=repo, env=env, capture_output=True,
                         text=True, timeout=300)
    assert "NEW [] False ''" in out.stdout, out.stdout + out.stderr[-2000:]


# ---------------------------------------------------------------------------
# the Error Study tab
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def qapp():
    pytest.importorskip("PyQt6")
    from PyQt6.QtWidgets import QApplication
    return QApplication.instance() or QApplication([])


def _tab(monkeypatch):
    from linac_gen_gui.interphase.state import AppState
    from linac_gen_gui.interphase.tabs import error_study_tab
    state = AppState()
    state.set_lattice(_line(), None)
    state.set_beam_config(_beam())
    tab = error_study_tab.ErrorStudyTab(state)
    tab._element_errors.append({"pattern": "MAP", "parameter": "phase_offset",
                                "distribution": "gaussian", "sigma": 1.0, "half_width": 0.0,
                                "cutoff": 3.0})
    started = []
    monkeypatch.setattr(error_study_tab._StudyWorker, "start", lambda self: started.append(self))
    return tab, started


def test_the_status_line_has_the_run_box_width(qapp, monkeypatch):
    """It can be long: below the controls, across the box, not a narrow column beside them."""
    from PyQt6.QtWidgets import QVBoxLayout
    tab, _started = _tab(monkeypatch)
    layout = tab._status.parentWidget().layout()
    assert isinstance(layout, QVBoxLayout)
    assert layout.itemAt(layout.count() - 1).widget() is tab._status
    tab.deleteLater()


def test_the_tab_runs_the_forward_model_chosen(qapp, monkeypatch):
    tab, started = _tab(monkeypatch)
    assert tab._forward.currentText() == "multiparticle"
    tab._on_run()
    tab._forward.setCurrentText("envelope")
    tab._on_run()
    assert [w._study.mode for w in started] == ["mp", "envelope"]
    tab.deleteLater()


def test_the_tab_shows_what_a_real_envelope_study_did_with_the_surrogates(qapp, monkeypatch):
    from linac_gen_gui.interphase.tabs.error_study_tab import surrogates_line
    tab, _started = _tab(monkeypatch)
    lat = _line()
    el = lat.elements[1]
    registry.register(exact(el, params=(("phase", -60.0, 0.0), ("scale", 0.9, 1.1))))
    results = _run(_study(lat, n=3))
    tab._on_done(results, stopped=False)
    line = surrogates_line(results)
    assert line.startswith("surrogates used: ") and "1 network answers per seed" in line
    assert tab._status.text().endswith("; " + line)
    tab._on_done(results, stopped=True)                 # a stop after the last seed
    assert tab._status.text().startswith("done — 3 seeds")
    tab.deleteLater()


def test_the_tab_says_what_the_study_did_with_the_surrogates(qapp):
    from types import SimpleNamespace

    from linac_gen_gui.interphase.tabs.error_study_tab import surrogates_line
    check = SimpleNamespace(use=True, note="surrogates used: ...")
    results = SimpleNamespace(surrogates=check, surrogate_calls=[11, 11, 10],
                              native_traversals=[{"accuracy check": 5},
                                                 {"accuracy check": 5, "outside its range": 1},
                                                 {"accuracy check": 5}],
                              outside_range=[0.0, 1 / 12, 0.0],
                              native_check={"seeds": [0, 1, 2], "largest": 1.04e-3,
                                            "key": "eps_y", "seed": 2})
    line = surrogates_line(results)
    assert line == ("surrogates used: ...; 10–11 network answers per seed; left native over the "
                    "3 seeds: accuracy check 15, outside its range 1; 3 seeds run again natively: "
                    "largest exit difference 0.001 (eps_y, seed 2)")
    results.outside_range = [0.0, 0.25, 0.5]
    results.native_check = {"seeds": [0, 1, 2], "largest": 0.0, "key": None, "seed": None}
    assert surrogates_line(results).endswith(
        "; outside the trained range in 2 of 3 seeds (up to 50 %); 3 seeds run again natively: "
        "largest exit difference 0")
    assert surrogates_line(SimpleNamespace(surrogates=SimpleNamespace(use=False, note="x"))) == "x"


def test_the_ensemble_popup_shows_no_transmission_for_an_envelope_study(qapp):
    """An envelope loses no particle: the popup says so instead of a 100 % histogram."""
    from linac_gen_gui.interphase.state import AppState
    from linac_gen_gui.interphase.tabs.results_tab import _EnsemblePopup
    state = AppState()
    state.error_study_results = _run(_study(_line(), n=2))
    popup = _EnsemblePopup(None, state)
    popup.refresh(None)
    assert popup._summary.text() == ("n_seeds = 2   envelope study: particle losses are not "
                                     "modelled (no transmission)")
    popup.deleteLater()
