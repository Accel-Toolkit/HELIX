"""DC space-charge kernels must scale with the SURVIVING current.

The macrocharge convention (pic/macrocharge.py) — each LAUNCHED
macroparticle carries a fixed share of the configured current, so the
transported current decays with transmission.  The bunched 3-D PIC
always followed it; the three DC kernels historically drove the field
from the configured ``beam.current`` outright, so a beam that had lost
particles kept pushing at full strength (measured on the PXIE LEBT:
σ_x +16 % at the SOL2 exit at 77 % transmission vs TraceWin partran;
loss-scaled the deviation is ~1 %).

The test construction: two beams with IDENTICAL alive phase-space, but
beam B launched twice the particles and lost half.  Every kernel must
kick B's survivors with exactly half of A's field.
"""
from __future__ import annotations

import numpy as np
import pytest

from linac_gen.core.beam import Beam
from linac_gen.core.particle import H_MINUS
from linac_gen.core.reference import ReferenceParticle
from linac_gen.pic.pic_solver import (
    kick_continuous_2d,
    kick_continuous_2d_gauss,
    kick_continuous_2d_pic,
)

N = 400
DS_MM = 10.0
CURRENT = 5.0


def _positions(asym: float = 1.0):
    rng = np.random.default_rng(11)
    q = np.zeros((N, 6))
    q[:, 0] = rng.normal(0.0, 2.0, N)          # x  (mm)
    q[:, 2] = rng.normal(0.0, 2.0 * asym, N)   # y  (mm)
    return q


def _beam(q_alive, n_lost_extra: int):
    n = len(q_alive) + n_lost_extra
    b = Beam(ref=ReferenceParticle(species=H_MINUS, w_kin=0.030,
                                   frequency=162.5),
             n_particles=n, current=CURRENT)
    b.particles[:len(q_alive)] = q_alive
    b.particles[len(q_alive):] = 0.0
    b.lost[len(q_alive):] = True               # dead on arrival
    b.continuous = True
    return b


def _kick_of(kernel, q_alive, n_lost_extra, **kw):
    b = _beam(q_alive, n_lost_extra)
    before = b.particles[:len(q_alive), [1, 3]].copy()
    kernel(b, DS_MM, **kw)
    return b.particles[:len(q_alive), [1, 3]] - before


# asym=1.6 exercises the Bassetti-Erskine branch of the gauss kernel and
# the a≠b branch of the uniform one; asym=1.0 the round-beam limits.
@pytest.mark.parametrize("asym", [1.0, 1.6])
@pytest.mark.parametrize("kernel,kw", [
    (kick_continuous_2d, {}),
    (kick_continuous_2d_gauss, {}),
    (kick_continuous_2d_pic, dict(nx=64, ny=64, grid_extent=6.0)),
])
def test_half_transmission_halves_the_kick(kernel, kw, asym):
    q = _positions(asym)
    full = _kick_of(kernel, q, 0, **kw)
    half = _kick_of(kernel, q, N, **kw)        # launched 2N, half lost
    assert np.abs(full).max() > 0.0            # the kick genuinely fired
    np.testing.assert_allclose(half, 0.5 * full, rtol=1e-12, atol=1e-15)


def test_lossless_beam_scaling_factor_is_exactly_one():
    """n_alive == n_particles ⇒ the scale is exactly 1.0, so lossless
    results are bit-identical to the pre-fix behaviour."""
    b = _beam(_positions(), 0)
    assert b.n_alive == b.n_particles
    assert (b.n_alive / b.n_particles) == 1.0  # exact, not approx


def test_transported_current_decays_with_transmission():
    """Kick amplitude tracks n_alive/n_launched across a sweep."""
    q = _positions()
    ref = np.abs(_kick_of(kick_continuous_2d, q, 0)).max()
    for extra, frac in ((N // 4, 0.8), (N, 0.5), (3 * N, 0.25)):
        got = np.abs(_kick_of(kick_continuous_2d, q, extra)).max()
        assert got == pytest.approx(frac * ref, rel=1e-12)
