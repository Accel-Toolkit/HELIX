"""Regenerate the CI-sized halo-testbed baseline fixture.

Run from the repo root after any DELIBERATE physics change to the PIC,
tracker, or testbed defaults:

    PYTHONPATH=. python3 tests/analysis/regen_halo_testbed_baseline.py

Keep the config in sync with tests/analysis/test_halo_testbed.py.
"""
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from scripts.halo_testbed import run_testbed  # noqa: E402

CFG = dict(n=2000, grid=24, step1=50, step2=20, mismatch=1.4, seed=0)

out = run_testbed(**CFG)
fix = Path(__file__).parent / "fixtures" / "halo_testbed_baseline.npz"
fix.parent.mkdir(exist_ok=True)
np.savez_compressed(
    fix,
    emit_x=out["emit_x"], sigma_x=out["sigma_x"],
    tail_emit_x_q999=out["tail_emit_x_q999"],
    tail_r_q999=out["tail_r_q999"],
    halo_x=out["halo_x"], meta=out["meta"],
)
print(f"wrote {fix}")
print(f"eps_x growth x{out['emit_x'][-1]/out['emit_x'][0]:.4f}, "
      f"eps99.9 x{out['tail_emit_x_q999'][-1]/out['tail_emit_x_q999'][0]:.4f}")
