"""Space charge in ring mode: the 2.5-D model.

The bunch's transverse distribution, projected over the whole bunch, gives a
two-dimensional field at each kick; a particle feels it in proportion to the
line density where it is (the model of a bunch long compared with its width),
and, with the slope of the line density, a longitudinal field.  Settings in
:mod:`.config`; the field solve in :mod:`.greens`; the pipe in
:mod:`.boundary`; where the kicks go in :mod:`.placement`; the particle
kernels in :mod:`.kernels`; a run's kicks in :mod:`.solver`; the analytic
tune shift in :mod:`.laslett`.
"""
from linac_gen.ring.sc.config import SpaceChargeConfig

__all__ = ["SpaceChargeConfig"]
