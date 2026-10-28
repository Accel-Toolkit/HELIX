"""The ring aperture shapes: which points are outside."""
from __future__ import annotations

import numpy as np
import pytest

from linac_gen.ring.compile import blocks as B
from linac_gen.ring.kernels.nodes import outside_aperture


def _outside(blk, x, y):
    p = np.array(blk, dtype=float)
    out = bool(outside_aperture(p, 0, x, y))
    if _compiled is not None:
        assert bool(_compiled(p, x, y)) is out
    return out


try:
    import numba

    @numba.njit(cache=False)
    def _compiled(p, x, y):
        return outside_aperture(p, 0, x, y)
except ImportError:                              # pragma: no cover
    _compiled = None


@pytest.mark.parametrize("x, y, out", [(0.0, 0.0, False), (2.9e-3, 0.0, False),
                                       (3.1e-3, 0.0, True), (2.2e-3, 2.2e-3, True),
                                       (2.1e-3, -2.1e-3, False)])
def test_circle(x, y, out):
    assert _outside(B.aperture("circle", 3e-3), x, y) is out


@pytest.mark.parametrize("x, y, out", [(4.9e-3, 1.9e-3, False), (5.1e-3, 0.0, True),
                                       (0.0, -2.1e-3, True), (-4.9e-3, -1.9e-3, False)])
def test_rectangle(x, y, out):
    assert _outside(B.aperture("rectangle", 5e-3, 2e-3), x, y) is out


@pytest.mark.parametrize("x, y, out", [(4.9e-3, 0.0, False), (0.0, 1.9e-3, False),
                                       (3.5e-3, 1.4e-3, False), (3.6e-3, 1.5e-3, True)])
def test_ellipse(x, y, out):
    # (3.5/5)^2 + (1.4/2)^2 = 0.98 inside; (3.6/5)^2 + (1.5/2)^2 = 1.0809 outside
    assert _outside(B.aperture("ellipse", 5e-3, 2e-3), x, y) is out


def test_rectangle_and_ellipse_and_an_offset_centre():
    blk = B.aperture("rectellipse", 4e-3, 4e-3, 5e-3, 5e-3)
    assert not _outside(blk, 3.9e-3, 0.0)
    assert _outside(blk, 4.1e-3, 0.0)                 # (the rectangle)
    assert _outside(blk, 3.9e-3, 3.9e-3)              # (the circle of 5 mm)
    off = B.aperture("circle", 1e-3, x0=2e-3, y0=-1e-3)
    assert not _outside(off, 2.5e-3, -1.5e-3) and _outside(off, 0.0, 0.0)


def test_polygon_even_odd_rule():
    # an L-shaped (concave) aperture
    verts = [(0.0, 0.0), (4e-3, 0.0), (4e-3, 1e-3), (1e-3, 1e-3), (1e-3, 3e-3), (0.0, 3e-3)]
    blk = B.aperture("polygon", vertices=verts)
    assert not _outside(blk, 0.5e-3, 0.5e-3)
    assert not _outside(blk, 3.5e-3, 0.5e-3)
    assert not _outside(blk, 0.5e-3, 2.5e-3)
    assert _outside(blk, 2.5e-3, 2.5e-3)              # (in the notch)
    assert _outside(blk, -0.5e-3, 0.5e-3) and _outside(blk, 5e-3, 0.5e-3)
