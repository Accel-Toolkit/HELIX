"""Injection in ring mode: multi-turn H⁻ charge-exchange injection through a
stripping foil, with bump and painting magnets.

Settings in :mod:`.config`; the injected beam (its particles in ring
coordinates at the foil) in :mod:`.source`; the linac bunches the chopper
keeps in :mod:`.chopper`; the foil (scattering, energy loss, stripping, its
hit and deposit maps) in :mod:`.foil`; the magnets' waveforms written into
the compiled ring turn by turn in :mod:`.programs`; a run's injection, turn
by turn, in :mod:`.injector`.
"""
from linac_gen.ring.injection.config import InjectionConfig, InjectionError

__all__ = ["InjectionConfig", "InjectionError"]
