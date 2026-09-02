"""STALE-banner correctness in the tune-depression and phase-advance popups.

Both popups compare the current a result was produced at against the
live Beam-tab config and flag drift ("STALE: σ run at X mA...").  Before
the run-current fix only ``EnvelopeResults`` carried ``current_mA``:
every fresh MP run at I > 0 and every reloaded MP file (0.0 sentinel on
disk) showed a FALSE banner (defect logs: tune-stale repro scenario D,
skeptic S1/S2), while results with a genuinely unknown current (openPMD
imports) would banner forever.

Regimes covered (through the REAL slots — _open_lattice, BeamTab._apply,
_run_mp/_run_envelope, ResultsTab.open_plot, _open_study_run_results,
ResultsTab._import_results):

  * MP fresh at 5 mA           -> no banner (failed before: S1)
  * MP + config drift to 6 mA  -> TRUE banner with the run's 5.000 mA
  * envelope 5 -> 6 -> re-run  -> unchanged behaviour (regime guard,
                                  passes before AND after by design)
  * reloaded LEGACY MP file    -> resolved via beam_config (failed: S2)
  * openPMD import (unknown)   -> never banners (contract 7)
"""
import os

import pytest

pytest.importorskip("PyQt6")
h5py = pytest.importorskip("h5py")

FODO = os.path.abspath(os.path.join(
    os.path.dirname(__file__), "..", "..", "examples", "fodo_cell.dat"))
POPUPS = ("tune_depr", "phase_adv")
#: text that proves the popup actually rendered (so "no STALE" can never
#: pass on an empty popup)
PRECONDITION = {"tune_depr": "η_med", "phase_adv": "per-cell σ₀"}


def _open_popups(win):
    dlgs = {}
    for key in POPUPS:
        assert win.results_tab.open_plot(key), f"open_plot({key!r}) failed"
        dlg = win.results_tab._popups[key]
        win.wait_popup_idle(dlg)
        dlgs[key] = dlg
    return dlgs


def _texts(win, dlgs):
    out = {}
    for key, dlg in dlgs.items():
        win.wait_popup_idle(dlg)
        out[key] = dlg._info.text()
    return out


def _apply_beam(win, current, n_particles=None):
    win.beam_tab._current.setValue(current)
    if n_particles is not None:
        win.beam_tab._npart.setValue(n_particles)
    win.beam_tab._apply()
    win.pump()


def _mp_run(win, current=5.0):
    win.open_lattice(FODO)
    _apply_beam(win, current, n_particles=300)
    win._run_mp()
    win.wait_worker(win._mp_worker)
    return _open_popups(win)


def test_mp_run_at_5mA_shows_no_stale_banner_both_popups(qapp, win):
    dlgs = _mp_run(win, 5.0)
    assert type(win.state.results).__name__ == "DiagnosticRecorder"
    assert win.state.results.current_mA == 5.0
    for key, txt in _texts(win, dlgs).items():
        assert PRECONDITION[key] in txt, f"[{key}] popup did not render: {txt!r}"
        assert "STALE" not in txt, f"[{key}] false banner: {txt!r}"


def test_mp_then_config_change_shows_true_banner_with_run_current(qapp, win):
    dlgs = _mp_run(win, 5.0)
    _apply_beam(win, 6.0)
    for key, txt in _texts(win, dlgs).items():
        assert "STALE: σ run at 5.000 mA, beam_config now 6.000 mA" in txt, \
            f"[{key}] wrong or missing banner: {txt!r}"


def test_envelope_run_banner_both_states_unchanged(qapp, win):
    """Envelope-regime guard: worked before the fix, must keep working."""
    win.open_lattice(FODO)
    _apply_beam(win, 5.0)
    win._run_envelope()
    win.wait_worker(win._envelope_worker)
    dlgs = _open_popups(win)
    for key, txt in _texts(win, dlgs).items():
        assert PRECONDITION[key] in txt
        assert "STALE" not in txt, f"[{key}] {txt!r}"
    _apply_beam(win, 6.0)
    for key, txt in _texts(win, dlgs).items():
        assert "STALE: σ run at 5.000 mA, beam_config now 6.000 mA" in txt, \
            f"[{key}] {txt!r}"
    win._run_envelope()
    win.wait_worker(win._envelope_worker)
    for key, txt in _texts(win, dlgs).items():
        assert "STALE" not in txt, f"[{key}] banner did not clear: {txt!r}"


def test_reloaded_legacy_mp_file_shows_no_false_banner(qapp, win, tmp_path):
    """A pre-fix MP file (0.0 sentinel, no marker) resolves through the
    file's own beam_config on load (skeptic S2)."""
    dlgs = _mp_run(win, 5.0)
    # the autouse _sandbox_calc_dir points the auto-dump at tmp_path
    mp_files = sorted(tmp_path.glob("*_mp.h5"))
    assert mp_files, "MP run did not auto-dump a results file"
    path = str(mp_files[-1])
    with h5py.File(path, "a") as f:
        env = f["envelope"]
        if "run_current_known" in env.attrs:
            del env.attrs["run_current_known"]
        env.attrs["current_mA"] = 0.0          # the pre-fix sentinel
    win._open_study_run_results(path)
    win.pump(0.5)
    assert type(win.state.results).__name__ == "_LoadedResults"
    assert win.state.results.current_mA == 5.0
    for key, txt in _texts(win, dlgs).items():
        assert PRECONDITION[key] in txt
        assert "STALE" not in txt, f"[{key}] false banner on reload: {txt!r}"
    _apply_beam(win, 6.0)
    for key, txt in _texts(win, dlgs).items():
        assert "STALE: σ run at 5.000 mA, beam_config now 6.000 mA" in txt, \
            f"[{key}] {txt!r}"


def test_openpmd_import_unknown_current_shows_no_banner(qapp, win, tmp_path,
                                                        monkeypatch):
    """openPMD files carry no run current: unknown must NEVER banner
    (contract 7) — at the matching config or after config drift."""
    from linac_gen.io.openpmd_output import save_results_openpmd
    from linac_gen_gui.interphase.tabs import results_tab as rt_mod

    dlgs = _mp_run(win, 5.0)
    path = str(tmp_path / "run.opmd.h5")
    save_results_openpmd(win.state.results, path,
                         beam_config=win.state.beam_config)
    monkeypatch.setattr(rt_mod.QFileDialog, "getOpenFileName",
                        staticmethod(lambda *a, **k: (path, "")))
    win.results_tab._import_results()
    win.pump(0.5)
    assert type(win.state.results).__name__ == "_LoadedResults"
    assert getattr(win.state.results, "current_mA", None) is None
    for key, txt in _texts(win, dlgs).items():
        assert "STALE" not in txt, f"[{key}] banner on unknown current: {txt!r}"
    _apply_beam(win, 6.0)
    for key, txt in _texts(win, dlgs).items():
        assert "STALE" not in txt, \
            f"[{key}] unknown current must never banner: {txt!r}"
