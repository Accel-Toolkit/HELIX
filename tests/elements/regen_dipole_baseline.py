"""Regenerate the Dipole transfer-matrix baseline used by
``test_dipole.py::test_dipole_baseline_bit_identical``.

The baseline pins the configurations whose matrices must NOT move when
the body matrix is edited: positive bend angles with rho > 0 (both
planes, 0 <= N < 1 and N < 0, with and without pole-face angles, full
element and a half slice) and the Elegant importer's signed-rho
convention (angle < 0, rho < 0, N = 0, horizontal).  Run it from the
tree whose behaviour is the reference — e.g. a git worktree at the
commit BEFORE a change — and commit the fixture together with the
change::

    PYTHONPATH=/path/to/reference/tree python3 tests/elements/regen_dipole_baseline.py

Writes ``tests/elements/fixtures/dipole_matrix_baseline.npz``; the test
compares with ``np.array_equal`` (bit-identical, no tolerance).
"""
from __future__ import annotations

import itertools
from pathlib import Path

import numpy as np

from linac_gen.core.particle import H_MINUS, PROTON
from linac_gen.core.reference import ReferenceParticle
from linac_gen.elements.dipole import Dipole

REFS = {
    "p100": dict(species=PROTON, w_kin=100.0, frequency=352.21),
    "h800": dict(species=H_MINUS, w_kin=800.0, frequency=162.5),
}
ANGLES = (5.0, 10.0, 30.0, 90.0, 180.0)
RHOS = (500.0, 1000.0, 5000.0)
FIELD_INDICES = (0.0, 0.3, 0.5, -0.5, -2.0, -400.0)
EDGES = ((0.0, 0.0), (10.0, 20.0))
ELEGANT = ((-5.0, -1000.0), (-10.0, -5000.0), (-45.0, -1000.0))


def configurations():
    for rn, ang, rho, n, (e1, e2), hv, frac in itertools.product(
            REFS, ANGLES, RHOS, FIELD_INDICES, EDGES, (0, 1), (None, 0.5)):
        yield f"{rn}|{ang}|{rho}|{n}|{e1},{e2}|{hv}|{frac}", rn, ang, rho, n, e1, e2, hv, frac
    for rn, (ang, rho), (e1, e2), frac in itertools.product(
            REFS, ELEGANT, EDGES, (None, 0.5)):
        yield f"{rn}|{ang}|{rho}|0.0|{e1},{e2}|0|{frac}", rn, ang, rho, 0.0, e1, e2, 0, frac


def matrices() -> dict[str, np.ndarray]:
    out = {}
    for key, rn, ang, rho, n, e1, e2, hv, frac in configurations():
        ref = ReferenceParticle(**REFS[rn])
        d = Dipole("b", angle=ang, rho=rho, e1=e1, e2=e2, field_index=n, hv=hv)
        ds = None if frac is None else d.length * frac
        out[key] = d.transfer_matrix(ref, ds=ds)
    return out


def main() -> None:
    out_dir = Path(__file__).parent / "fixtures"
    out_dir.mkdir(exist_ok=True)
    path = out_dir / "dipole_matrix_baseline.npz"
    arrs = matrices()
    np.savez_compressed(path, **arrs)
    print(f"wrote {path}  ({len(arrs)} configurations)")


if __name__ == "__main__":
    main()
