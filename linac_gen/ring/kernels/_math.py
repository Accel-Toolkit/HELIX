"""The math functions of the ring kernels.

Compiled by numba (the CPU kernels and the CUDA kernels), the kernels use the
math module's own functions, whose compiled forms follow IEEE arithmetic: the
square root of a negative number, an arcsine beyond ±1, the logarithm of a
negative number give NaN, and the node loop then loses the particle as
invalid (:data:`.nodes.LOSS_INVALID`) at that node.

Run as plain Python because numba is not installed, the math module raises
instead.  The versions here return what the compiled functions give — NaN,
or ±inf — so the plain-Python kernels lose the same particle at the same
node; for any other argument they return exactly the math module's value.
(A division by exactly zero gives inf or NaN in the compiled kernels — they
use numpy's error model — but still raises in plain Python; for a particle
that is a set of measure zero.  In numba's CUDA simulator, a test tool, the
kernels run as Python with the math module's functions.)
"""
from __future__ import annotations

import math

try:
    import numba  # noqa: F401
except ImportError:
    _COMPILED = False
else:
    _COMPILED = True

if _COMPILED:
    from math import asin, atan, cos, cosh, fabs, isfinite, log1p, nan, sin, sinh, sqrt, tan
else:
    from math import atan, fabs, isfinite, nan

    def sqrt(v):
        return math.sqrt(v) if v >= 0.0 else nan

    def asin(v):
        return math.asin(v) if -1.0 <= v <= 1.0 else nan

    def log1p(v):
        if v > -1.0:
            return math.log1p(v)
        return -math.inf if v == -1.0 else nan

    def sin(v):
        return math.sin(v) if isfinite(v) else nan

    def cos(v):
        return math.cos(v) if isfinite(v) else nan

    def tan(v):
        return math.tan(v) if isfinite(v) else nan

    def sinh(v):
        try:
            return math.sinh(v)
        except OverflowError:
            return math.copysign(math.inf, v)

    def cosh(v):
        try:
            return math.cosh(v)
        except OverflowError:
            return math.inf

__all__ = ["asin", "atan", "cos", "cosh", "fabs", "isfinite", "log1p", "nan", "sin", "sinh",
           "sqrt", "tan"]
