"""The ring kernels in arbitrary precision (tests only).

The kernel sources are executed again with mpmath's functions in place of
the math module's, so a map can be differentiated to 30 digits and its
symplecticity checked far below float64 rounding.  The sources are the
shipped ones, unchanged except for their import lines.
"""
from __future__ import annotations

import types
from pathlib import Path

import mpmath as mp

_KDIR = Path(__file__).resolve().parents[2] / "linac_gen" / "ring" / "kernels"
_MATH_NAMES = ("asin", "atan", "atan2", "cos", "cosh", "fabs", "isfinite", "log", "log1p", "nan",
               "sin", "sinh", "sqrt", "tan")


def _load(name: str, extra: dict) -> types.ModuleType:
    src = (_KDIR / f"{name}.py").read_text()
    out = []
    for line in src.splitlines():
        s = line.strip()
        if s.startswith("from math import"):
            continue
        if s.startswith(("from linac_gen.ring.kernels._jit import",
                         "from linac_gen.ring.kernels._math import")):
            continue
        if s.startswith("from linac_gen.ring.kernels import"):
            continue
        out.append(line)
    mod = types.ModuleType(f"mp_{name}")
    ns = mod.__dict__
    for nm in _MATH_NAMES:
        ns[nm] = getattr(mp, nm)
    ns["jitable"] = lambda f=None, **_k: f
    ns.update(extra)
    exec(compile("\n".join(out), f"<mp {name}>", "exec"), ns)
    return mod


def load(dps: int = 50):
    """(cartesian, curvilinear, nodes) modules computing in mpmath at ``dps`` digits."""
    mp.mp.dps = dps
    cart = _load("cartesian", {})
    curv = _load("curvilinear", {})
    nd = _load("nodes", {"_cart": cart, "_curv": curv})
    return cart, curv, nd
