"""Cooperative-stop regressions for the Convergence scan.

The old _stop_scan called QThread.terminate() — undefined behaviour
inside numpy/PIC code — and re-enabled Run immediately, so a second
scan could start while the first still mutated the live lattice.
"""
from __future__ import annotations

import pytest

pytest.importorskip("PyQt6")

from linac_gen.core.config import BeamConfig  # noqa: E402
from linac_gen.core.step_config import StepConfig  # noqa: E402
from linac_gen_gui.interphase.state import AppState  # noqa: E402
from linac_gen_gui.interphase.tabs.convergence_tab import (  # noqa: E402
    AXIS_GRID, ConvergenceTab, _ScanWorker,
)


def _beam_cfg() -> BeamConfig:
    return BeamConfig(
        species="proton", energy=2.5, frequency=162.5, current=0.0,
        n_particles=80, distribution="waterbag", emit_nx=0.25, beta_x=1.0,
        emit_ny=0.25, beta_y=1.0, emit_z=0.3, beta_z=10.0)


def _collect(worker):
    out = {"rows": [], "done": [], "failed": []}
    worker.row_done.connect(lambda i, d: out["rows"].append((i, d)))
    worker.done.connect(lambda stopped: out["done"].append(stopped))
    worker.failed.connect(out["failed"].append)
    return out


def test_prestopped_scan_exits_before_first_point(qapp, mini_lattice):
    original = StepConfig(integration_steps_per_metre=77.0,
                          sc_steps_per_metre=33.0)
    mini_lattice.step_config = original
    w = _ScanWorker(mini_lattice, _beam_cfg(), AXIS_GRID, [16, 24],
                    fixed_nx=16, fixed_extent=5.0,
                    fixed_step1=20.0, fixed_step2=10.0,
                    scan_n_particles=80, parallel_workers=0)
    out = _collect(w)
    w.request_stop()
    w.run()                              # synchronous — house pattern

    assert out["done"] == [True]         # stopped, not failed
    assert out["failed"] == []
    assert out["rows"] == []             # nothing half-computed
    assert mini_lattice.step_config is original   # live lattice untouched


def test_unstopped_scan_completes_all_points(qapp, mini_lattice):
    original = StepConfig(integration_steps_per_metre=77.0,
                          sc_steps_per_metre=33.0)
    mini_lattice.step_config = original
    w = _ScanWorker(mini_lattice, _beam_cfg(), AXIS_GRID, [16, 24],
                    fixed_nx=16, fixed_extent=5.0,
                    fixed_step1=20.0, fixed_step2=10.0,
                    scan_n_particles=80, parallel_workers=0)
    out = _collect(w)
    w.run()

    assert out["done"] == [False]
    assert out["failed"] == []
    assert [i for i, _ in out["rows"]] == [0, 1]
    # step_config restored after the sweep mutated it per point.
    assert mini_lattice.step_config is original


def test_tab_stop_state_machine(qapp):
    tab = ConvergenceTab(AppState())
    try:
        # Idle: nothing to stop.
        assert not tab._stop_btn.isEnabled()
        tab._stop_scan()                 # no worker — must be a no-op

        # A user stop lands as done(stopped=True): buttons recover, the
        # Run-All queue is cleared so the next axis does NOT auto-start.
        tab._rows = [dict(value=16), None, None]
        tab._auto_mode = True
        tab._auto_queue = ["x", "y"]
        tab._run_btn.setEnabled(False)
        tab._run_all_btn.setEnabled(False)
        tab._on_done(stopped=True)

        assert tab._run_btn.isEnabled()
        assert tab._run_all_btn.isEnabled()
        assert not tab._stop_btn.isEnabled()
        assert tab._auto_mode is False
        assert tab._auto_queue == []
        assert "stopped" in tab._badge.text()
        assert "1/3" in tab._badge.text()
    finally:
        tab.deleteLater()
