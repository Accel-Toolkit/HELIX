"""The decorator every ring-mode map carries.

With numba installed, ``jitable`` is ``numba.extending.register_jitable``: the
function stays an ordinary Python function when called from Python, and is
compiled when it is called from numba code — the multi-core CPU kernels and
the CUDA kernels (and numba's CUDA simulator) alike, from this one source.
Without numba it is the identity, so the maps still run (slowly) as plain
Python.
"""
from __future__ import annotations

try:
    from numba.extending import register_jitable as jitable
except ImportError:                  # numba is optional (the [ring] extra)
    def jitable(fn=None, **_kw):
        if fn is None:
            return lambda f: f
        return fn

__all__ = ["jitable"]
