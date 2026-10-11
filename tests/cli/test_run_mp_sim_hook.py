"""``run_mp_sim(..., pic_setup_hook=)`` hands the built space-charge solver
to the caller (the ``Simulation`` hook), so studies can instrument kicks on
the real entry path without patching classes."""
from __future__ import annotations

from linac_gen.cli import common
from linac_gen.core.config import BeamConfig, SpaceChargeConfig
from linac_gen.core.lattice import Lattice
from linac_gen.core.step_config import StepConfig
from linac_gen.elements.drift import Drift
from linac_gen.pic.pic_solver import PicSolver


def _run(hook=None, current=2.0):
    lat = Lattice()
    lat.add(Drift("D1", length=40.0))
    cfg = BeamConfig(species="H-", energy=2.1, frequency=162.5,
                     current=current, n_particles=500)
    sc = (SpaceChargeConfig(nx=16, ny=16, nz=16, use_gpu="cpu")
          if current > 0 else None)
    return common.run_mp_sim(lat, cfg, sc, StepConfig(), seed=1,
                             pic_setup_hook=hook)


def test_hook_receives_the_solver_and_its_kicks_can_be_counted():
    seen = []

    def hook(solver):
        seen.append(solver)
        kick = solver.kick

        def counted(beam, ds):
            solver.n_kicks = getattr(solver, "n_kicks", 0) + 1
            return kick(beam, ds)
        solver.kick = counted

    rec, beam = _run(hook)
    assert len(seen) == 1 and isinstance(seen[0], PicSolver)
    assert seen[0].n_kicks > 0
    assert rec.sc_config is seen[0].config


def test_hook_not_called_without_space_charge():
    seen = []
    _run(seen.append, current=0.0)
    assert seen == []


def test_hook_does_not_change_results():
    rec_a, beam_a = _run()
    rec_b, beam_b = _run(lambda s: None)
    assert (beam_a.particles == beam_b.particles).all()
