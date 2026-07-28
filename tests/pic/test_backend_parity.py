"""End-to-end parity between the C++ and Python PIC backends.

The individual kernels (``deposit_cic``, ``interpolate_cic``) are exercised
in ``test_pic_kernels_cpp.py``; this file exercises the full
``PicSolver.kick`` cycle so that an ABI or dispatch bug would surface even
if the individual kernels look correct.
"""
import numpy as np
import pytest

import linac_gen.pic.pic_solver as pic_solver_mod
from linac_gen.core.beam import Beam
from linac_gen.core.config import SpaceChargeConfig
from linac_gen.core.particle import PROTON
from linac_gen.core.reference import ReferenceParticle
from linac_gen.pic.pic_solver import PicSolver


def _make_beam(seed=1):
    ref = ReferenceParticle(species=PROTON, w_kin=3.0, frequency=352.21)
    beam = Beam(ref=ref, n_particles=200, current=20.0)
    rng = np.random.default_rng(seed)
    beam.particles[:, 0] = rng.normal(0.0, 1.0, 200)
    beam.particles[:, 1] = rng.normal(0.0, 0.3, 200)
    beam.particles[:, 2] = rng.normal(0.0, 1.0, 200)
    beam.particles[:, 3] = rng.normal(0.0, 0.3, 200)
    beam.particles[:, 4] = rng.normal(0.0, 4.0, 200)
    beam.particles[:, 5] = rng.normal(0.0, 0.01, 200)
    return beam


@pytest.mark.skipif(not pic_solver_mod._USE_CPP,
                    reason="C++ PIC kernels not available")
def test_pic_kick_cpp_matches_python(monkeypatch):
    """PicSolver.kick must produce the same beam with both backends."""
    sc = SpaceChargeConfig(nx=32, ny=32, nz=32, grid_extent=3.0)

    # Run #1: C++ (current module-level bindings)
    beam_cpp = _make_beam(seed=42)
    PicSolver(sc).kick(beam_cpp, ds=50.0)

    # Run #2: Python fallback. Replace the module-level names the solver
    # looks up at call time.
    monkeypatch.setattr(pic_solver_mod, "deposit_cic",
                        pic_solver_mod._py_deposit_cic)
    monkeypatch.setattr(pic_solver_mod, "interpolate_cic",
                        pic_solver_mod._py_interpolate_cic)

    beam_py = _make_beam(seed=42)
    PicSolver(sc).kick(beam_py, ds=50.0)

    np.testing.assert_allclose(
        beam_cpp.particles, beam_py.particles,
        rtol=1e-10, atol=1e-14,
        err_msg="C++ and Python PIC backends diverged in PicSolver.kick",
    )
