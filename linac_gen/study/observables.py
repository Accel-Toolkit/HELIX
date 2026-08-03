"""Pure-h5py readers for per-run study observables.

Deliberately lattice-free: element-position observables are resolved
to ``s_m`` by the engine at validation time, so this module only ever
interpolates arrays out of a results file.  ``load_results_hdf5``
omits ``provenance/`` — :func:`read_provenance` reads it directly.
"""
from __future__ import annotations

import numpy as np


def evaluate(results_h5: str, observables: list) -> dict:
    """Return {name: float | None} for each ObservableSpec.

    ``None`` marks an unavailable quantity (e.g. ``transmission`` from
    an envelope run) rather than raising — a study row must never die
    on a missing column.
    """
    out: dict = {}
    if not observables:
        return out
    import h5py
    with h5py.File(results_h5, "r") as f:
        env = f.get("envelope")
        s = np.asarray(env["s"]) if env is not None and "s" in env \
            else None
        for ob in observables:
            val = None
            if env is not None and ob.quantity in env:
                arr = np.asarray(env[ob.quantity], dtype=float)
                if arr.size:
                    if ob.at == "end" or ob.s_m is None:
                        val = float(arr[-1])
                    elif s is not None and s.size == arr.size:
                        # EnvelopeResults.s is in mm (house gotcha) —
                        # pinned by tests/study/test_engine_e2e.py
                        val = float(np.interp(ob.s_m * 1e3, s, arr))
            out[ob.name] = val
    return out


def read_provenance(results_h5: str) -> dict:
    import h5py
    with h5py.File(results_h5, "r") as f:
        grp = f.get("provenance")
        if grp is None:
            return {}
        return {k: (v.decode() if isinstance(v, bytes) else v)
                for k, v in grp.attrs.items()}
