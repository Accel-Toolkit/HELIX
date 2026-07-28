"""Background pre-population of the per-element transfer-matrix cache.

Phase-advance and matching tabs share a single ``state.matrix_cache``
dict; the heavy cost is the first walk through the lattice (FieldMap3D
elements rebuild their fitted 6×6 matrix via 6 RK4-tracked jacobian
columns — ~500 s on full PIP-II).  This service runs that walk in a
QThread the moment a lattice (and beam config) become available, so by
the time the user opens the Phase Advance popup the cache is already
warm and the first call returns in milliseconds.

Lifecycle:

* Started by :class:`MatrixCacheWarmer.maybe_start` whenever either the
  lattice or the beam config changes.  The previous warm-up (if any)
  is asked to stop and disowned — the freshly-replaced ``matrix_cache``
  dict on AppState is what the next warmer fills.
* Emits ``state.matrix_cache_warmed(n_entries)`` on success so any UI
  element interested in showing a "cache ready" indicator can react.
* Silent on failure (it's a pure optimisation; the on-demand path will
  recompute correctly if the warmer aborts).
"""
from __future__ import annotations

import math
from typing import Any

from PyQt6.QtCore import QThread, pyqtSignal

from linac_gen.core.particle import PROTON, DEUTERON, H_MINUS
from linac_gen.core.reference import ReferenceParticle
from linac_gen.tracking.matrix_tracking import compute_transfer_matrix


_SPECIES_MAP = {"proton": PROTON, "deuteron": DEUTERON, "H-": H_MINUS}


def _build_ref(beam_config: Any) -> ReferenceParticle:
    sp = _SPECIES_MAP.get(getattr(beam_config, "species", "proton"), PROTON)
    return ReferenceParticle(
        species=sp,
        w_kin=float(beam_config.energy),
        frequency=float(beam_config.frequency),
    )


class _WarmWorker(QThread):
    done = pyqtSignal(int)   # n_entries in cache after the walk

    def __init__(self, lattice, ref, cache: dict) -> None:
        super().__init__()
        self._lattice = lattice
        self._ref = ref
        self._cache = cache

    def run(self) -> None:
        try:
            compute_transfer_matrix(self._lattice, self._ref, cache=self._cache)
        except Exception:
            # Pure optimisation; on-demand path will recompute correctly.
            return
        self.done.emit(len(self._cache))


class MatrixCacheWarmer:
    """Owns at most one in-flight :class:`_WarmWorker`.

    Re-entrant: calling :meth:`maybe_start` while a previous warm-up is
    running disowns the old thread (it keeps writing to its now-stale
    cache dict, harmless) and starts a fresh one against the current
    ``state.matrix_cache``.
    """

    def __init__(self, state) -> None:
        self._state = state
        self._worker: _WarmWorker | None = None
        state.lattice_changed.connect(self._maybe_start_from_signal)
        state.beam_config_changed.connect(self._maybe_start_from_signal)
        # When a real run starts, get out of its way; when it ends,
        # try warming again so the next Phase Advance click is fast.
        state.running_changed.connect(self._on_running_changed)

    def _maybe_start_from_signal(self, _obj: Any) -> None:
        self.maybe_start()

    def _on_running_changed(self, running: bool) -> None:
        if running:
            # Run started: disown any in-flight warmer.  Python can't
            # cleanly kill a QThread mid-RK4, but disowning lets the OS
            # scheduler de-prioritise it and its target dict is the
            # current state.matrix_cache (which is fine to keep filling).
            self._worker = None
        else:
            # Run ended: try warming again.
            self.maybe_start()

    def maybe_start(self) -> None:
        if self._state.running:
            return
        lattice = self._state.lattice
        beam = self._state.beam_config
        if lattice is None or beam is None:
            return
        try:
            ref = _build_ref(beam)
        except Exception:
            return
        # Disown the old worker — its target dict is no longer the
        # state's matrix_cache (set_lattice / set_beam_config swapped it
        # out wholesale), so its writes won't pollute anyone.
        self._worker = _WarmWorker(lattice, ref, self._state.matrix_cache)
        self._worker.done.connect(self._state.matrix_cache_warmed.emit)
        self._worker.start()
