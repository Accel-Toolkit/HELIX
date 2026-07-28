"""Regression: parallel scan-point pool == serial, bit-for-bit.

Runs the same 3-point mini-scan (grid sweep) via both the
``ProcessPoolExecutor`` path and the in-process serial path, and asserts
every output scalar matches to machine precision.  If this ever fails,
the parallel scheduler has introduced non-determinism and must be fixed
before the GUI scan workers adopt it.
"""
from __future__ import annotations

from dataclasses import asdict

import numpy as np
import pytest

from linac_gen.core.config import BeamConfig
from linac_gen.parallel import (
    ScanPoint, run_scan_points, run_scan_points_serial,
)


def _beam_cfg_dict() -> dict:
    cfg = BeamConfig(
        species="H-", energy=2.1226695, frequency=162.5,
        current=5.0, duty_cycle=100.0,
        n_particles=2000, distribution="gaussian", cutoff=4.0,
        emit_nx=0.21, alpha_x=1.228, beta_x=0.316,
        emit_ny=0.21, alpha_y=-0.095394, beta_y=0.113,
        emit_z=0.06231832, alpha_z=0.0, beta_z=819.05492,
    )
    return asdict(cfg)


def _mini_scan_points() -> list[ScanPoint]:
    beam = _beam_cfg_dict()
    return [
        ScanPoint(
            lattice_path="examples/fodo_cell.dat",
            beam_config=beam, nx=nx, grid_extent=5.0, step1=100.0, step2=50.0,
        )
        for nx in (32, 48, 64)
    ]


def _sort_keys(rows: list[dict], values: list) -> list[dict]:
    """Return rows sorted by the nx they ran at (so serial/parallel align)."""
    return [r for _, r in sorted(zip(values, rows), key=lambda kv: kv[0])]


def test_parallel_vs_serial_bit_for_bit():
    """Every end-of-lattice scalar must match to 1e-10 rtol across executors."""
    points = _mini_scan_points()
    vals = [p.nx for p in points]

    serial = run_scan_points_serial(points)
    parallel = run_scan_points(points, max_workers=2)

    # Serial is already in input order; parallel runner returns in input order too.
    for (pt, s, p) in zip(points, serial, parallel):
        for key in ("sigma_x", "sigma_y", "sigma_phi", "emit_x", "emit_y"):
            np.testing.assert_allclose(
                s[key], p[key], rtol=1e-10,
                err_msg=f"parallel drift at nx={pt.nx}, key={key}",
            )


def test_on_done_callback_fires_once_per_point():
    """GUI integration depends on on_done; make sure it's fully invoked."""
    points = _mini_scan_points()
    seen: list[tuple[int, float]] = []

    def cb(idx: int, row: dict) -> None:
        seen.append((idx, row["sigma_x"]))

    run_scan_points_serial(points, on_done=cb)
    assert len(seen) == len(points)
    assert sorted(i for i, _ in seen) == list(range(len(points)))


# ---------------------------------------------------------------------------
# Cooperative stop (should_stop)
# ---------------------------------------------------------------------------

def test_serial_stop_before_first_point():
    points = _mini_scan_points()
    out = run_scan_points_serial(points, should_stop=lambda: True)
    assert out == []


def test_serial_stop_after_first_point():
    points = _mini_scan_points()
    polls = {"n": 0}

    def stop_after_first():
        polls["n"] += 1
        return polls["n"] > 1          # False for point 0, True afterwards

    out = run_scan_points_serial(points, should_stop=stop_after_first)
    assert len(out) == 1               # exactly one completed row kept


def test_pool_immediate_stop_returns_promptly():
    """The kill path: queued futures cancelled, children terminated —
    a stop must not block on running points finishing (previously the
    pool always ran to completion; worst case minutes)."""
    import time

    points = _mini_scan_points()
    t0 = time.time()
    out = run_scan_points(points, max_workers=2, should_stop=lambda: True)
    elapsed = time.time() - t0

    assert out == []                   # nothing completed before the stop
    # Generous ceiling: spawn-context startup only. A broken kill path
    # (waiting for the running points) would take a couple of minutes.
    assert elapsed < 30.0
