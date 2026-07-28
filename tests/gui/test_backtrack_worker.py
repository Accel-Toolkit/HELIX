"""BacktrackWorker + BacktrackDialog contract (GUI backtracking, 2026-07-11).

The worker mirrors MultiparticleWorker's five-signal contract but drives
``Simulation.run_backtrack``; the Results tab consumes the reversed-s
recorder unchanged.
"""
from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("PyQt6")


def _mkref():
    from linac_gen.core.particle import PROTON
    from linac_gen.core.reference import ReferenceParticle
    return ReferenceParticle(species=PROTON, w_kin=3.0, frequency=162.5)


def _exit_beam(mini_lattice):
    """Forward-track a beam to the lattice exit (the worker's input)."""
    from linac_gen.core.beam import Beam
    from linac_gen.tracking.tracker import Tracker
    beam = Beam(ref=_mkref(), n_particles=200, current=0.0)
    rng = np.random.default_rng(4)
    for j, s in enumerate([0.5, 0.3, 0.5, 0.3, 3.0, 0.002]):
        beam.particles[:, j] = rng.normal(0, s, 200)
    entrance = beam.particles.copy()
    t = Tracker(mini_lattice, beam)
    for e in mini_lattice.elements:
        t._track_element(e)
    return beam, entrance


def test_worker_finished_ok_with_backward_recorder(qapp, mini_lattice):
    """Runs through the REAL QThread path (``.start()`` + wait) — a
    synchronous ``worker.run()`` call missed the live bug where an
    attribute named ``start`` shadowed ``QThread.start()`` and clicking
    OK in the dialog raised "'int' object is not callable"."""
    from linac_gen_gui.interphase.workers import BacktrackWorker

    beam, entrance = _exit_beam(mini_lattice)
    worker = BacktrackWorker(mini_lattice, beam, None, _mkref(),
                             start=0, end=len(mini_lattice.elements) - 1)
    # The regression itself: QThread.start must still be the method.
    assert callable(worker.start), \
        "an attribute shadows QThread.start() — the GUI cannot launch it"

    got: list = []
    worker.finished_ok.connect(got.append)
    worker.start()                     # the real thread launch
    assert worker.wait(30_000), "worker thread did not finish"
    qapp.processEvents()               # deliver the queued signal

    assert len(got) == 1
    rec = got[0]
    assert getattr(rec, "direction", None) == "backward"
    s = np.asarray(rec.s)
    assert (np.diff(s) >= 0).all()    # reversed to increasing s
    # The walk reconstructed the entrance distribution on the beam.
    np.testing.assert_allclose(beam.particles, entrance,
                               rtol=1e-9, atol=1e-11)


def test_worker_request_stop_emits_aborted(qapp, mini_lattice):
    from linac_gen_gui.interphase.workers import BacktrackWorker

    beam, _ = _exit_beam(mini_lattice)
    worker = BacktrackWorker(mini_lattice, beam, None, _mkref(),
                             start=0, end=len(mini_lattice.elements) - 1)
    done: list = []
    aborted: list = []
    worker.finished_ok.connect(done.append)
    worker.aborted.connect(lambda: aborted.append(True))
    worker.request_stop()             # stop BEFORE running
    worker.run()
    assert aborted and not done


def test_dialog_settings_round_trip(qapp):
    from linac_gen_gui.interphase.dialogs.backtrack_dialog import (
        BacktrackDialog,
    )
    dlg = BacktrackDialog(n_elements=100, has_results_beam=True)
    try:
        dlg._start_spin.setValue(5)
        dlg._end_spin.setValue(80)
        dlg._inverse_combo.setCurrentText("linear")
        dlg._sc_check.setChecked(True)
        s = dlg.get_settings()
        assert s["source"] == "results"
        assert (s["start"], s["end"]) == (5, 80)
        assert s["field_map_mode"] == "linear"
        assert s["space_charge"] is True
        assert s["write_dst"] is None
        # without a results beam the .dst radio must be pre-selected
        dlg2 = BacktrackDialog(n_elements=10, has_results_beam=False)
        try:
            assert not dlg2._src_results.isEnabled()
            assert dlg2._src_dst.isChecked()
        finally:
            dlg2.deleteLater()
    finally:
        dlg.deleteLater()
