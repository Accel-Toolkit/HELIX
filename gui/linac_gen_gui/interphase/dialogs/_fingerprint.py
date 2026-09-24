"""Optics fingerprints shared by the tool dialogs: a hashable snapshot of
every scalar of every element plus the beam scalars, so a result computed
on a snapshot can be discarded (or marked stale) when the live optics
changed underneath it."""
from __future__ import annotations


def scalars(obj) -> tuple:
    """Hashable snapshot of the scalar attributes of an element / config (floats by repr: NaN-safe)."""
    out = []
    for k, v in sorted(vars(obj).items()):
        if isinstance(v, float):
            out.append((k, repr(v)))
        elif isinstance(v, (int, str, bool, type(None))):
            out.append((k, v))
    return tuple(out)


def optics_fingerprint(lattice, beam_cfg) -> tuple:
    """Every scalar of every element plus the beam scalars — any optics edit while a worker runs invalidates its result."""
    els = tuple((type(e).__name__, scalars(e)) for e in lattice.elements) if lattice is not None else ()
    return els, (scalars(beam_cfg) if beam_cfg is not None else ())
