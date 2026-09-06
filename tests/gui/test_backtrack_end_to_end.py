"""App-level end-to-end backtracking — the exact user click path.

Post-mortem test (2026-07-11): the Backtrack feature shipped with two
thread bugs that unit tests missed because they called ``worker.run()``
synchronously and never exercised the seam the GUI actually uses
(``_run_backtrack`` slot → real ``QThread.start()`` → queued signals →
``_backtrack_done`` → ``state.set_results``).  This test drives the REAL
``InterphaseWindow`` through the whole scenario — load lattice, run a
real MP simulation, invoke the backtrack slot — with only the human
interactions (the dialog's OK click) monkeypatched.
"""
from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("PyQt6")


def _wait_worker(qapp, worker, timeout_ms=60_000):
    assert worker is not None, "worker was never constructed"
    assert worker.wait(timeout_ms), "worker thread did not finish"
    qapp.processEvents()               # deliver queued signals


def test_full_user_scenario_mp_then_backtrack(qapp, win, mini_lattice,
                                              monkeypatch):
    from linac_gen.core.config import BeamConfig
    from linac_gen_gui.interphase.dialogs import backtrack_dialog as bd

    # --- the user's steps 1-2: load a lattice, configure a small beam ---
    win.state.set_lattice(mini_lattice, "/tmp/e2e_backtrack.dat")
    cfg = BeamConfig(species="proton", energy=3.0, frequency=162.5,
                     current=0.0, n_particles=300)
    win.beam_tab.set_beam_config(cfg)
    win.state.set_beam_config(cfg)

    # --- step 3: Run Multi-particle (the REAL slot + worker) -------------
    win._run_mp()
    _wait_worker(qapp, win._mp_worker)
    fwd = win.state.results
    assert fwd is not None and getattr(fwd, "beam", None) is not None, \
        "MP run did not attach its beam — backtrack source unavailable"
    exit_particles = fwd.beam.particles.copy()

    # --- step 4: Simulate → Backtrack Distribution… → OK ----------------
    # Only the human click is mocked: exec() "presses OK", get_settings
    # returns what the dialog defaults would produce.
    #
    # ``_backtrack_done`` pops a modal "Backtrack caveats" QMessageBox
    # when the run emits BacktrackWarnings — a real backtrack (survivors-
    # only reconstruction, etc.) ALWAYS does, and a modal exec() blocks
    # forever under the offscreen platform.  Record the box instead of
    # showing it: this both unblocks the run AND asserts the new caveat-
    # surfacing path fired (a GUI user's only window onto the warnings).
    from PyQt6.QtWidgets import QMessageBox
    warn_boxes = []
    monkeypatch.setattr(QMessageBox, "warning",
                        staticmethod(lambda *a, **k: warn_boxes.append(a)))
    n_last = len(mini_lattice.elements) - 1
    monkeypatch.setattr(bd.BacktrackDialog, "exec", lambda self: True)
    monkeypatch.setattr(
        bd.BacktrackDialog, "get_settings",
        lambda self: {"source": "results", "dst_path": None,
                      "start": 0, "end": n_last,
                      "field_map_mode": "rk4", "space_charge": False,
                      "write_dst": None})
    win._run_backtrack()
    _wait_worker(qapp, win._backtrack_worker)

    # --- step 5: results landed, backward, monotone s --------------------
    res = win.state.results
    assert res is not fwd, "backtrack results never replaced the MP results"
    assert getattr(res, "direction", None) == "backward"
    s = np.asarray(res.s)
    assert (np.diff(s) >= 0).all()
    assert not win.state.running

    # The forward run's beam must be UNTOUCHED (the worker deep-copies);
    # the reconstructed beam differs from the exit state.
    np.testing.assert_array_equal(fwd.beam.particles, exit_particles)
    assert getattr(res, "beam", None) is not None
    assert not np.array_equal(res.beam.particles, exit_particles)

    # --- step 6: physics caveats reached the user -----------------------
    # Whatever BacktrackWarnings the worker captured must have been
    # surfaced through the "Backtrack caveats" box (never swallowed).
    caveats = list(getattr(res, "backtrack_warnings", ()) or ())
    if caveats:
        assert any(len(a) >= 2 and "caveat" in str(a[1]).lower()
                   for a in warn_boxes), \
            "backtrack emitted caveats but no 'Backtrack caveats' box shown"


def test_leftover_last_lattice_key_cannot_swap_lattice_mid_flow(
        qapp, win, mini_lattice, monkeypatch):
    """Regression (2026-09-02 full-suite hang): a ``lastLatticePath``
    left in the process-wide sandboxed QSettings by an EARLIER test
    module armed the window's startup restore timer, which silently
    swapped ``state.lattice`` to fodo_cell.dat at the first
    ``processEvents()``; the backtrack energy audit then (correctly)
    refused the mismatched lattice/beam pair and the failure raised a
    modal critical box.  Plant the key INSIDE the test body — after the
    hygiene fixture has run, before the first event-loop pump — and
    prove the mini-lattice MP → backtrack flow is immune."""
    import os

    from linac_gen.core.config import BeamConfig
    from linac_gen_gui.interphase.app import (_SETTINGS_LAST_LATTICE,
                                              _settings)
    from linac_gen_gui.interphase.dialogs import backtrack_dialog as bd

    fodo = os.path.abspath(os.path.join(
        os.path.dirname(__file__), "..", "..", "examples", "fodo_cell.dat"))
    assert os.path.exists(fodo), "fixture lattice missing"
    _settings().setValue(_SETTINGS_LAST_LATTICE, fodo)   # the poison key

    lattice_path = "/tmp/e2e_backtrack_leftover.dat"
    win.state.set_lattice(mini_lattice, lattice_path)
    cfg = BeamConfig(species="proton", energy=3.0, frequency=162.5,
                     current=0.0, n_particles=300)
    win.beam_tab.set_beam_config(cfg)
    win.state.set_beam_config(cfg)

    win._run_mp()
    _wait_worker(qapp, win._mp_worker)
    assert win.state.results is not None

    n_last = len(mini_lattice.elements) - 1
    monkeypatch.setattr(bd.BacktrackDialog, "exec", lambda self: True)
    monkeypatch.setattr(
        bd.BacktrackDialog, "get_settings",
        lambda self: {"source": "results", "dst_path": None,
                      "start": 0, "end": n_last,
                      "field_map_mode": "rk4", "space_charge": False,
                      "write_dst": None})
    win._run_backtrack()
    _wait_worker(qapp, win._backtrack_worker)

    # The leftover key must not have swapped the working lattice…
    assert win.state.lattice_path == lattice_path, \
        "stale lastLatticePath resurrected a different lattice mid-flow"
    # …the backtrack must have succeeded…
    assert getattr(win.state.results, "direction", None) == "backward"
    # …and no failure modal was raised (caveat *warnings* are fine).
    criticals = [b for b in win.message_boxes if b[0] == "critical"]
    assert not criticals, f"failure box(es) recorded: {criticals}"


def test_backtrack_slot_guards(qapp, win, monkeypatch):
    """No lattice loaded → a warning box, no crash, no worker."""
    from PyQt6.QtWidgets import QMessageBox
    seen = []
    monkeypatch.setattr(QMessageBox, "warning",
                        staticmethod(lambda *a, **k: seen.append(a)))
    win._run_backtrack()
    assert seen, "expected the 'No lattice' warning"
    assert win._backtrack_worker is None
