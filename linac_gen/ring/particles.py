"""Ring-mode particle container (structure of arrays, fixed capacity).

All arrays are allocated once at ``capacity`` so injection over hundreds of
turns never reallocates (on a GPU the arrays stay resident).  A slot is
``EMPTY`` until a particle is injected into it.  The energy helpers ``ptau``,
``rpp`` and ``rvv`` are kept consistent with ``delta`` by :meth:`sync_energy`,
the only place they are recomputed.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from linac_gen.ring import coords as _co
from linac_gen.ring.reference import RingReference

# particle states
EMPTY = -1          # slot not yet used
LOST = 0            # hit an aperture or left the bucket for good
CIRCULATING = 1     # alive in the ring
WASTE = 2           # unstripped (H⁻ / H⁰) or missed the foil: on its way to the absorber

LOSS_CAUSES = {0: "none", 1: "aperture", 2: "longitudinal", 3: "absorber", 4: "unstripped",
               5: "user", 6: "invalid", 7: "missed the foil"}


@dataclass
class RingParticles:
    coords: np.ndarray          # (6, cap): x, px, y, py, zeta, delta
    ptau: np.ndarray
    rpp: np.ndarray
    rvv: np.ndarray
    weight: np.ndarray          # real particles per macroparticle
    state: np.ndarray           # int8, see the constants above
    pid: np.ndarray             # int64 particle id (-1 for empty slots)
    inj_turn: np.ndarray        # int32 turn of injection
    foil_hits: np.ndarray       # int32
    lost_turn: np.ndarray       # int32, -1 while alive
    lost_node: np.ndarray       # int32, -1 while alive
    lost_s: np.ndarray          # float64, s of the loss within the turn [m]
    lost_cause: np.ndarray      # int8, key of LOSS_CAUSES
    n_used: int = 0             # slots handed out so far
    next_pid: int = 0

    @classmethod
    def allocate(cls, capacity: int) -> RingParticles:
        if capacity <= 0:
            raise ValueError("ring particles: capacity must be positive")
        n = int(capacity)
        return cls(
            coords=np.zeros((6, n)), ptau=np.zeros(n), rpp=np.ones(n), rvv=np.ones(n),
            weight=np.zeros(n), state=np.full(n, EMPTY, dtype=np.int8),
            pid=np.full(n, -1, dtype=np.int64), inj_turn=np.full(n, -1, dtype=np.int32),
            foil_hits=np.zeros(n, dtype=np.int32), lost_turn=np.full(n, -1, dtype=np.int32),
            lost_node=np.full(n, -1, dtype=np.int32), lost_s=np.full(n, np.nan),
            lost_cause=np.zeros(n, dtype=np.int8))

    @property
    def capacity(self) -> int:
        return self.coords.shape[1]

    def add(self, coords: np.ndarray, weight, *, ref: RingReference, turn: int = 0,
            state: int = CIRCULATING) -> slice:
        """Put particles (6, k) into the next free slots; returns their slice.
        Refuses (rather than reallocating) when the capacity is exhausted."""
        X = np.asarray(coords, dtype=float)
        if X.ndim != 2 or X.shape[0] != 6:
            raise ValueError("ring particles: coords must have shape (6, k)")
        k = X.shape[1]
        if self.n_used + k > self.capacity:
            raise ValueError(f"ring particles: capacity {self.capacity} exceeded "
                             f"({self.n_used} used + {k} new) — raise the capacity")
        sl = slice(self.n_used, self.n_used + k)
        self.coords[:, sl] = X
        self.weight[sl] = np.broadcast_to(np.asarray(weight, dtype=float), (k,))
        self.state[sl] = state
        self.pid[sl] = np.arange(self.next_pid, self.next_pid + k, dtype=np.int64)
        self.inj_turn[sl] = turn
        self.n_used += k
        self.next_pid += k
        self.sync_energy(ref, sl)
        return sl

    def sync_energy(self, ref: RingReference, sl: slice | None = None) -> None:
        """Recompute from delta the energy deviation ptau = (E - E0)/(P0 c),
        rpp = 1/(1 + delta) and rvv = beta/beta0 = (1 + delta)/(1 + beta0 ptau)."""
        sl = slice(0, self.n_used) if sl is None else sl
        d = self.coords[5, sl]
        b0 = ref.beta0
        self.ptau[sl] = _co.ptau_from_delta(d, b0)
        self.rpp[sl] = 1.0 / (1.0 + d)
        self.rvv[sl] = (1.0 + d) / (1.0 + b0 * self.ptau[sl])

    def change_reference(self, old: RingReference, new: RingReference) -> None:
        """New reference momentum (a ramp step): each particle keeps its
        momentum and its arrival time, so delta, px, py and zeta are
        re-expressed relative to the new reference (see
        :func:`linac_gen.ring.coords.change_reference`)."""
        sl = slice(0, self.n_used)
        self.coords[:, sl] = _co.change_reference(self.coords[:, sl], old=old, new=new)
        self.sync_energy(new, sl)

    def mark_lost(self, mask: np.ndarray, *, turn: int, node: int, s: float, cause: int) -> int:
        """Flag circulating particles in ``mask`` (length n_used) as lost."""
        m = np.asarray(mask, dtype=bool) & (self.state[: self.n_used] == CIRCULATING)
        idx = np.flatnonzero(m)
        self.state[idx] = LOST
        self.lost_turn[idx] = turn
        self.lost_node[idx] = node
        self.lost_s[idx] = s
        self.lost_cause[idx] = cause
        return int(idx.size)

    @property
    def alive(self) -> np.ndarray:
        return self.state[: self.n_used] == CIRCULATING

    def ledger(self) -> dict:
        """Exact bookkeeping: injected = circulating + lost + waste (macroparticles and charge)."""
        st = self.state[: self.n_used]
        w = self.weight[: self.n_used]
        out = {"injected": int(self.n_used), "injected_charge": float(w.sum())}
        for name, code in (("circulating", CIRCULATING), ("lost", LOST), ("waste", WASTE)):
            m = st == code
            out[name] = int(m.sum())
            out[name + "_charge"] = float(w[m].sum())
        return out
