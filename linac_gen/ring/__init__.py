"""Ring mode: multi-turn tracking of a synchrotron (opt-in).

Ring mode is a separate engine.  It reads HELIX's ``Lattice`` / ``Element``
objects as its description, so every importer and the lattice editor apply,
but it never changes the linac tracker or any linac result.  Nothing here is
imported unless ring mode is used: the public names below load on first
access, and the compiled kernels (numba, an optional dependency) load only
when a ring run starts.
"""
from __future__ import annotations

import importlib

_EXPORTS = {
    "ConstantsProfile": "linac_gen.ring.constants",
    "PROFILES": "linac_gen.ring.constants",
    "RingReference": "linac_gen.ring.reference",
    "RingParticles": "linac_gen.ring.particles",
    "RingConfig": "linac_gen.ring.config",
    "RingVerdict": "linac_gen.ring.detect",
    "detect_ring": "linac_gen.ring.detect",
    "RingSimulation": "linac_gen.ring.engine",
    "RingResults": "linac_gen.ring.engine",
    "RingLattice": "linac_gen.ring.compile",
    "RingCompileError": "linac_gen.ring.compile",
    "compile_ring": "linac_gen.ring.compile",
    "compile_from_config": "linac_gen.ring.compile",
    "RingOptics": "linac_gen.ring.optics",
    "twiss": "linac_gen.ring.optics",
    "ring_optics": "linac_gen.ring.optics",
    "closed_orbit": "linac_gen.ring.optics",
    "rf_bucket": "linac_gen.ring.optics",
}

__all__ = sorted(_EXPORTS)


def __getattr__(name: str):
    mod = _EXPORTS.get(name)
    if mod is None:
        raise AttributeError(f"module 'linac_gen.ring' has no attribute {name!r}")
    return getattr(importlib.import_module(mod), name)
