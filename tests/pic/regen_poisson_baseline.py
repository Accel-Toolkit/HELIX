"""Regenerate the Poisson-solver CPU baselines used by
``test_poisson_solver.py::test_cpu_backend_reproduces_baseline``.

Run ONLY when the solver is intentionally changed:

    python tests/pic/regen_poisson_baseline.py

Writes ``tests/pic/fixtures/poisson_baseline_{igf,point}.npz`` from the
forced-CPU (float64) path.  The regression test compares at rtol=1e-12,
so baselines generated on any platform are valid on every other one
(cross-platform float64 FFT differences are ~1e-15 relative).
"""
from pathlib import Path

import numpy as np

from linac_gen.pic.poisson_solver import PoissonSolverFFT


def main() -> None:
    rng = np.random.default_rng(11)
    rho = rng.standard_normal((16, 16, 16))
    gmin = np.array([-1., -1., -1.])
    gmax = np.array([+1., +1., +1.])
    ng = np.array([16, 16, 16])
    out_dir = Path(__file__).parent / "fixtures"
    out_dir.mkdir(exist_ok=True)
    for kind in ("igf", "point"):
        Ex, Ey, Ez = PoissonSolverFFT(
            gmin, gmax, ng, use_gpu="cpu", green_kind=kind
        ).solve(rho)
        path = out_dir / f"poisson_baseline_{kind}.npz"
        np.savez_compressed(path, Ex=Ex, Ey=Ey, Ez=Ez)
        print(f"wrote {path}  (|E|max = {max(np.abs(a).max() for a in (Ex, Ey, Ez)):.6e})")


if __name__ == "__main__":
    main()
