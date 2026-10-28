"""Where the space-charge kicks go: about ``n`` per turn, each at an element
boundary (never inside an element's nodes: between a misalignment's shift
and its body, or inside a magnet), the nearest boundary to equally spaced
targets from the start of the ring.  Each kick stands for the ring from
halfway to the kick before to halfway to the next (lengths summing to the
circumference)."""
from __future__ import annotations

import numpy as np

#: targets looked at a time (bounds the memory a large request takes)
CHUNK = 1 << 16


def element_starts(rl) -> tuple[np.ndarray, np.ndarray]:
    """(node index, s [m]) of the first node of every element (where a kick
    may go: before that node).  A start at the very end of the ring (a
    zero-length element there) is the start of the ring, node 0's place: it
    is left out."""
    ne = np.asarray(rl.node_element)
    if ne.size == 0:
        return np.zeros(0, dtype=np.int64), np.zeros(0)
    s_all = np.asarray(rl.node_s, dtype=float)
    if np.any(np.diff(s_all) < 0.0):
        raise ValueError("space charge: the ring's element positions decrease along it (an "
                         "element of negative length?) — the kicks cannot be placed")
    first = np.flatnonzero(np.concatenate([[True], ne[1:] != ne[:-1]]))
    s = s_all[first]
    keep = s < float(rl.length) * (1.0 - 1e-12)
    keep[0] = True
    return first[keep].astype(np.int64), s[keep]


def _nearest(s: np.ndarray, length: float, n: int) -> np.ndarray:
    """The first index of each element start nearest (periodically) to one of
    the n equally spaced targets k C/n (k = 0 .. n-1: the first at the start
    of the ring, so that a ring of identical cells gets the same kicks in
    every cell), sorted and without repeats; of two starts at the same
    distance, the one with the smaller index.  (s is sorted: the nearest
    start to a target is the one before it or the one after it, around the
    ring.)  A request denser than the starts — targets closer than half the
    smallest gap between distinct starts, so that every start is the nearest
    to one — gets every distinct start without its targets being made; the
    others are looked at a chunk at a time."""
    C = float(length)
    m = s.size
    n = max(1, int(n))
    distinct = np.flatnonzero(np.concatenate([[True], s[1:] != s[:-1]]))
    if distinct.size == 1:
        return distinct
    su = s[distinct]
    gap = min(float(np.min(np.diff(su))), C - float(su[-1]) + float(su[0]))
    if C / n <= 0.5 * gap:
        return distinct
    hit = np.zeros(m, dtype=bool)
    for k0 in range(0, n, CHUNK):
        t = np.arange(k0, min(n, k0 + CHUNK)) * C / n
        i = np.searchsorted(s, t, side="left")
        a = (i - 1) % m
        b = i % m
        da = np.abs(s[a] - t)
        da = np.minimum(da, C - da)
        db = np.abs(s[b] - t)
        db = np.minimum(db, C - db)
        hit[np.where(db < da, b, np.where(db == da, np.minimum(a, b), a))] = True
    return np.searchsorted(s, np.unique(s[hit]), side="left")


def placed_count(starts_s, length: float, n: int) -> int:
    """How many kicks a request for ``n`` places on a ring of ``length`` whose
    elements start at ``starts_s`` (the rule of :func:`kick_positions`)."""
    s = np.asarray(starts_s, dtype=float)
    return 0 if s.size == 0 else int(_nearest(s, length, n).size)


def kick_positions(rl, n: int):
    """(nodes, s, ds): the node each kick comes before (sorted), its position
    and the length it stands for."""
    C = float(rl.length)
    nodes, s = element_starts(rl)
    if nodes.size == 0:
        raise ValueError("space charge: the ring has no elements")
    idx = _nearest(s, C, n)
    pos = s[idx]
    nodes_k = nodes[idx].astype(np.int64)
    m = len(pos)
    if m == 1:
        ds = np.array([C])
    else:
        gap = np.empty(m)                       # to the next kick, around the ring
        gap[:-1] = pos[1:] - pos[:-1]
        gap[-1] = C - pos[-1] + pos[0]
        ds = 0.5 * (gap + np.roll(gap, 1))
    return nodes_k, pos, ds
