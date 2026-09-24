"""Regenerate ``fixtures/reliability_selftest_baseline.npz`` — the values the
Reliability Study self-test pins (fault-leg metrics per case and bracket,
the top compensation settings, seed-0 draws and correction kicks, the foil
spot, the availability Monte Carlo of seed 7).

Run from the tree that DEFINES the reference behaviour (the BEFORE tree of
the change you are about to make), then commit the fixture with the change::

    PYTHONPATH=.:gui python tests/reliability/regen_reliability_selftest_baseline.py

The metadata (``__machine__``, ``__deck_sha256__``, ``__helix_version__``,
``__git_rev__``) is recorded for the report; ``HELIX_BASELINE_EXACT=1`` asks
the test for bit identity, the default allows one platform-libm ulp.
"""
from __future__ import annotations

import hashlib
import platform
import subprocess
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "gui"))


def main() -> int:
    from linac_gen import __version__
    from linac_gen.reliability.selftest import BASELINE_PATH, DEMO, baseline_values
    values = baseline_values()
    rev = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True,
                         cwd=REPO).stdout.strip() or "unknown"
    meta = {"__machine__": np.array(platform.machine()),
            "__deck_sha256__": np.array(hashlib.sha256((DEMO / "reliability_demo.dat").read_bytes()).hexdigest()),
            "__helix_version__": np.array(str(__version__)), "__git_rev__": np.array(rev)}
    BASELINE_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(BASELINE_PATH, **values, **meta)
    print(f"wrote {BASELINE_PATH} ({len(values)} arrays; {platform.machine()}, HELIX {__version__}, {rev})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
