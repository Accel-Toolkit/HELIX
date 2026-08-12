"""Direct bunch-to-bunch space charge (multibunch M5) — driver machinery.

M5a (pattern-aware image charge factors) lives on the PIC solver itself
(``PicSolver.train_image_factors`` / ``train_force_engage``); the
TrainRunner computes per-bunch factors from the pulse pattern and hands
them to each bunch's freshly built solver through the Simulation
``pic_setup_hook`` seam.

This module provides the M5b OPT-IN "distinct neighbours" piece: a
bounded ring buffer that records the CURRENT bunch's state at engaged
SC-kick cadence, so the NEXT bunch can deposit the real (loss-scaled,
subsampled) previous bunch as its leading image instead of an exact
self-copy.  With identical bunches the two are equal within deposition
noise — the payoff appears only once bunches genuinely differ (losses,
beam-loading-shifted refs), which is why "images" stays the default.
"""
from __future__ import annotations

import warnings

import numpy as np


class NeighborSnapshotBuffer:
    """Subsampled (x, y, dphi) snapshots keyed by the per-pass SC-kick
    ordinal (``PicSolver._train_kick_ordinal`` — counted on every kick()
    call, so consecutive bunches align by call sequence).

    One buffer records one bunch's pass; the TrainRunner swaps buffers
    between bunches so at most the PREVIOUS bunch is retained (bounded
    memory: <= n_recorded_kicks x n_sub x 3 doubles per buffer, times
    two live buffers).  Subsampling is a deterministic stride over the
    alive set (no RNG — reruns bit-reproduce), and the snapshot stores
    the alive COUNT so the consumer can restore the neighbour's full
    loss-scaled charge (n_alive/n_sub macro shares per particle).
    """

    #: refuse to grow past this many recorded kicks (~ safety net; at the
    #: default n_sub=1024 this caps one buffer near 500 MB — far beyond
    #: any realistic engaged-RFQ region, so hitting it means something is
    #: wrong and the fallback-to-self-copy path takes over, loudly).
    MAX_KICKS = 20_000

    def __init__(self, n_sub: int = 1024):
        self.n_sub = int(n_sub)
        if self.n_sub < 8:
            raise ValueError("n_sub must be >= 8")
        self._snaps: dict[int, dict] = {}
        self._warned_full = False
        self._warned_miss = False

    # -- PicSolver.train_snapshot_recorder -----------------------------
    def record(self, ordinal: int, beam) -> None:
        if len(self._snaps) >= self.MAX_KICKS:
            if not self._warned_full:
                self._warned_full = True
                warnings.warn(
                    f"distinct-neighbour snapshot buffer hit its "
                    f"{self.MAX_KICKS}-kick cap — later kicks fall back "
                    "to self-copy leading images", stacklevel=2)
            return
        al = beam.alive_mask
        n_alive = int(np.count_nonzero(al))
        if n_alive == 0:
            return
        cols = beam.particles[al][:, (0, 2, 4)]
        if n_alive > self.n_sub:
            # deterministic stride: strictly increasing, hence unique,
            # because spacing (n_alive-1)/(n_sub-1) >= 1 here
            sel = np.linspace(0.0, n_alive - 1, self.n_sub)
            cols = cols[sel.astype(np.int64)]
        self._snaps[int(ordinal)] = {
            "xyphi": np.array(cols, dtype=np.float64),
            "n_alive": float(n_alive),
            "ref_phi_s": float(beam.ref.phi_s),
            "ref_w_kin": float(beam.ref.w_kin),
        }

    # -- PicSolver.train_neighbor_provider ------------------------------
    def snapshot(self, ordinal: int):
        s = self._snaps.get(int(ordinal))
        if s is None and not self._warned_miss:
            # The previous bunch never recorded this ordinal (different
            # engagement window, early loss, or it never engaged at
            # all).  The solver then uses the exact self-copy leading
            # image — physical and safe, but say so once instead of
            # silently degrading the study.
            self._warned_miss = True
            warnings.warn(
                f"distinct-neighbour snapshot missing for SC kick "
                f"#{int(ordinal)} (previous bunch recorded "
                f"{len(self._snaps)} kicks) — falling back to the "
                "self-copy leading image", stacklevel=2)
        return s

    @property
    def n_recorded(self) -> int:
        return len(self._snaps)
