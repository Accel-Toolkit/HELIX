"""Field-map Jacobians pinned bit for bit against a committed baseline.

The fixture was written by ``regen_fieldmap_matrix_baseline.py`` from the
tree BEFORE the 2026-09-08 batched-probe change (commit 837d2c3), so a
later edit to ``fitted_matrix`` / ``fitted_matrix_slice`` in any of the
three field-map elements that moves a single bit — or leaves different
walk state behind — fails here with the offending keys listed.

Exact equality is asserted only with ``HELIX_BASELINE_EXACT=1`` — the
developer's check after a core-physics change, on the machine that wrote
the fixture (its architecture is stored under ``__machine__`` for the
record).  "Same architecture" is NOT enough: GitHub's arm64 macOS runner
moved 3 of the 338 arrays by 1e-18..1e-21 in the v1.11.0 CI (a different
numpy build / libm), so the default comparison everywhere allows one
platform-libm ulp: ``rtol=1e-14`` plus an absolute floor of ``1e-12``
times the array's scale, because the fixture holds thousands of exact
zeros and cancellation residues (1e-13..1e-17, from ±x probes landing in
different trilinear cells) that a purely relative test would compare
exactly.  The batched-vs-single invariant that must hold on every
platform lives in ``test_fieldmap_jacobian_batched.py``.
"""
import os
from pathlib import Path

import numpy as np
import pytest

from tests.elements.regen_fieldmap_matrix_baseline import matrices

FIXTURE = Path(__file__).parent / "fixtures" / "fieldmap_matrix_baseline.npz"


@pytest.mark.slow
def test_fieldmap_matrix_baseline_bit_identical():
    base = np.load(FIXTURE)
    machine = str(base["__machine__"])
    now = matrices()
    keys = [k for k in base.files if k != "__machine__"]
    assert set(keys) == set(now), (
        f"key set changed: -{sorted(set(keys) - set(now))} +{sorted(set(now) - set(keys))}")
    exact = os.environ.get("HELIX_BASELINE_EXACT") == "1"
    bad = []
    for k in keys:
        a, b = base[k], now[k]
        if a.shape != b.shape:
            bad.append((k, "shape")); continue
        if exact:
            if not np.array_equal(a, b, equal_nan=True):
                bad.append((k, float(np.nanmax(np.abs(a - b)))))
        else:
            atol = 1e-12 * max(1.0, float(np.nanmax(np.abs(a))))
            if not np.allclose(a, b, rtol=1e-14, atol=atol, equal_nan=True):
                bad.append((k, float(np.nanmax(np.abs(a - b)))))
    assert not bad, (f"{len(bad)} of {len(keys)} arrays moved "
                     f"({'exact' if exact else 'one-ulp'}; fixture from {machine}): {bad[:8]}")
