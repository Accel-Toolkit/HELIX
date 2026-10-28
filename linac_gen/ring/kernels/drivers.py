"""Loops over particles and nodes — one source for every backend.

These functions are compiled by :mod:`.cpu` (numba, serial and multi-core) and
run unchanged as plain Python when numba is absent.  Particle arrays follow
:class:`linac_gen.ring.particles.RingParticles`: ``coords`` (6, n), the
energy helpers ``ptau``/``rpp``/``rvv`` (n) kept consistent with delta by the
kernels themselves, ``state`` (1 = circulating; only those move) and the loss
records.
"""
from __future__ import annotations

from math import floor

from linac_gen.ring.kernels._jit import jitable
from linac_gen.ring.kernels.nodes import track_node, track_particle

try:
    from numba import prange
except ImportError:          # plain Python
    prange = range


@jitable
def wrap_zeta(zeta, length):
    """zeta into [-length/2, length/2) (the period of the RF maps); no change
    when length is 0."""
    if length > 0.0:
        zeta -= length * floor(zeta / length + 0.5)
    return zeta


@jitable
def advance_particle(i, coords, ptau, rpp, rvv, state, lost_turn, lost_node, lost_cause,
                     kinds, offsets, p, ref_beta0, ref_p0c, ref_bg2, j0, j1, n_turns, turn0,
                     zeta_wrap):
    """Advance particle ``i`` (if circulating) ``n_turns`` times through nodes
    j0..j1-1, zeta wrapped into [-zeta_wrap/2, zeta_wrap/2) after each turn
    (0: not wrapped).  Lost at node j of turn k: state 0, lost_turn =
    turn0 + k, lost_node = j and its cause, coordinates those at the entrance
    of node j.  Shared by the CPU loop and the CUDA kernel."""
    if state[i] != 1:
        return
    x = coords[0, i]
    px = coords[1, i]
    y = coords[2, i]
    py = coords[3, i]
    zeta = coords[4, i]
    delta = coords[5, i]
    pt = ptau[i]
    rp = rpp[i]
    rv = rvv[i]
    for k in range(n_turns):
        (x, px, y, py, zeta, delta, pt, rp, rv, jl, cause) = track_particle(
            kinds, offsets, p, ref_beta0, ref_p0c, ref_bg2, j0, j1,
            x, px, y, py, zeta, delta, pt, rp, rv)
        if jl >= 0:
            state[i] = 0
            lost_turn[i] = turn0 + k
            lost_node[i] = jl
            lost_cause[i] = cause
            break
        zeta = wrap_zeta(zeta, zeta_wrap)
    coords[0, i] = x
    coords[1, i] = px
    coords[2, i] = y
    coords[3, i] = py
    coords[4, i] = zeta
    coords[5, i] = delta
    ptau[i] = pt
    rpp[i] = rp
    rvv[i] = rv


def track_turns(coords, ptau, rpp, rvv, state, lost_turn, lost_node, lost_cause,
                kinds, offsets, p, ref_beta0, ref_p0c, ref_bg2, j0, j1, n_turns, turn0,
                zeta_wrap):
    """Advance the circulating particles ``n_turns`` times through nodes
    j0..j1-1 (a whole turn when j0 = 0, j1 = n_nodes); see
    :func:`advance_particle`."""
    n = coords.shape[1]
    for i in prange(n):
        advance_particle(i, coords, ptau, rpp, rvv, state, lost_turn, lost_node, lost_cause,
                         kinds, offsets, p, ref_beta0, ref_p0c, ref_bg2, j0, j1, n_turns, turn0,
                         zeta_wrap)


def track_turns_tbt(coords, ptau, rpp, rvv, state, lost_turn, lost_node, lost_cause,
                    kinds, offsets, p, ref_beta0, ref_p0c, ref_bg2, n_turns, turn0, zeta_wrap,
                    tbt):
    """Whole turns of a few particles (tune probes), keeping each one's
    coordinates after every turn: tbt[k, :, i] after turn turn0 + k + 1
    (left as given — NaN — once the particle is lost)."""
    n = coords.shape[1]
    n_nodes = kinds.shape[0]
    for k in range(n_turns):
        for i in range(n):
            if state[i] != 1:
                continue
            advance_particle(i, coords, ptau, rpp, rvv, state, lost_turn, lost_node,
                             lost_cause, kinds, offsets, p, ref_beta0, ref_p0c, ref_bg2, 0,
                             n_nodes, 1, turn0 + k, zeta_wrap)
            if state[i] == 1:
                for c in range(6):
                    tbt[k, c, i] = coords[c, i]


def track_probe(coords, ptau, rpp, rvv, kinds, offsets, p, ref_beta0, ref_p0c, ref_bg2,
                j0, j1, out, record):
    """Advance every particle once through nodes j0..j1-1 with no apertures
    (an optics probe); with ``record`` keep its coordinates after each node,
    out[k, :, i] after node j0 + k.  A non-finite value shows where a map was
    not defined for the particle."""
    n = coords.shape[1]
    for i in range(n):
        x = coords[0, i]
        px = coords[1, i]
        y = coords[2, i]
        py = coords[3, i]
        zeta = coords[4, i]
        delta = coords[5, i]
        pt = ptau[i]
        rp = rpp[i]
        rv = rvv[i]
        for j in range(j0, j1):
            (x, px, y, py, zeta, delta, pt, rp, rv, _lost) = track_node(
                kinds[j], p, offsets[j], ref_beta0[j], ref_p0c[j], ref_bg2[j],
                x, px, y, py, zeta, delta, pt, rp, rv)
            if record:
                k = j - j0
                out[k, 0, i] = x
                out[k, 1, i] = px
                out[k, 2, i] = y
                out[k, 3, i] = py
                out[k, 4, i] = zeta
                out[k, 5, i] = delta
        coords[0, i] = x
        coords[1, i] = px
        coords[2, i] = y
        coords[3, i] = py
        coords[4, i] = zeta
        coords[5, i] = delta
        ptau[i] = pt
        rpp[i] = rp
        rvv[i] = rv
