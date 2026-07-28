"""HALO-PIC M1: mismatched-FODO halo testbed — physics sanity + pinned
baseline.

The baseline pins the full trajectory of core and tail observables of a
CI-sized run (N=2000, 24^3, seed 0) so any accidental change to the PIC,
tracker cadence, tail diagnostics, or beam sampling shows up loudly.
Regen (deliberate changes only): tests/analysis/regen_halo_testbed_baseline.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

FIXTURE = Path(__file__).parent / "fixtures" / "halo_testbed_baseline.npz"
CFG = dict(n=2000, grid=24, step1=50, step2=20, seed=0)   # + mismatch below


@pytest.fixture(scope="module")
def testbed_run():
    from scripts.halo_testbed import run_testbed
    return run_testbed(mismatch=1.4, **CFG)


def test_baseline_pinned(testbed_run):
    assert FIXTURE.exists(), "run regen_halo_testbed_baseline.py first"
    ref = np.load(FIXTURE, allow_pickle=False)
    for key in ("emit_x", "sigma_x", "tail_emit_x_q999", "tail_r_q999",
                "halo_x"):
        got = testbed_run[key] if not key.startswith("tail_") \
            else testbed_run[key]
        np.testing.assert_allclose(
            got, ref[key], rtol=1e-10, atol=1e-12,
            err_msg=f"{key} drifted from the pinned baseline")


def test_halo_physics_ladder():
    """Mismatch drives tail growth monotonically; matched beam is quiet.
    External physics anchor (Gluckstern/Wangler): the 99.9% emittance must
    grow FASTER than the rms emittance under mismatch — halo formation, the
    signature this whole benchmark exists to resolve."""
    from scripts.halo_testbed import run_testbed, matched_twiss  # noqa: F401
    growth = {}
    for m in (1.0, 1.4):
        out = run_testbed(mismatch=m, **CFG)
        growth[m] = (out["emit_x"][-1] / out["emit_x"][0],
                     out["tail_emit_x_q999"][-1] / out["tail_emit_x_q999"][0])
    # matched: near-quiescent
    assert growth[1.0][0] < 1.10
    assert growth[1.0][1] < 1.10
    # mismatched: real growth, and tail outgrows core
    assert growth[1.4][0] > 1.25
    assert growth[1.4][1] > 1.5
    assert growth[1.4][1] > 1.2 * growth[1.4][0]


def test_tail_arrays_aligned(testbed_run):
    n = len(testbed_run["s"])
    for k in ("tail_emit_x_q99", "tail_emit_x_q999", "tail_emit_y_q99",
              "tail_r_q99", "tail_r_q999"):
        assert len(testbed_run[k]) == n, k
    # quantile ordering holds at every step
    assert np.all(testbed_run["tail_emit_x_q999"]
                  >= testbed_run["tail_emit_x_q99"])
