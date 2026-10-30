# linac_gen/elements/base.py
"""Abstract base classes for lattice elements (capability-based)."""
from abc import ABC, abstractmethod
import numpy as np


def angle_nearest(angle_deg: float, ref_deg: float) -> float:
    """``angle_deg`` moved by whole turns to within 180 deg of ``ref_deg``
    (returned unchanged when it already is)."""
    while angle_deg - ref_deg > 180.0:
        angle_deg -= 360.0
    while angle_deg - ref_deg < -180.0:
        angle_deg += 360.0
    return angle_deg


def wrap_deg(angle_deg: float) -> float:
    """``angle_deg`` moved by whole turns into [-180, 180] (returned
    unchanged when it already is)."""
    while angle_deg > 180.0:
        angle_deg -= 360.0
    while angle_deg < -180.0:
        angle_deg += 360.0
    return angle_deg

class Element(ABC):
    """Base class for all lattice elements."""
    def __init__(self, name: str, length: float, aperture: float, n_steps: int):
        self.name = name
        self.length = length
        self.aperture = aperture
        self.n_steps = n_steps
        # Deck label (``Q01: QUAD …`` → ``"Q01"``) when the source file
        # carried one; ``None`` otherwise.  Never a key for HELIX
        # internals — ``name`` stays the generated unique identifier
        # (QUAD_001, STEER_001, …).  Set by the TraceWin parser, emitted
        # again by the writer, used to talk to the control system
        # (orbit-response calibration maps device names to labels).
        self.label: str | None = None

class TransferMapElement(Element):
    """Elements with a linear 6x6 transfer matrix (drift, quad, solenoid, dipole)."""
    @abstractmethod
    def transfer_matrix(self, ref, ds: float = None) -> np.ndarray:
        ...

    @abstractmethod
    def track(self, beam, ds: float = None) -> None:
        ...

class ThinKickElement(Element):
    """Zero-length elements with instantaneous kicks (RF gap, multipole, steerer)."""
    def __init__(self, name: str, aperture: float = 0.0):
        super().__init__(name=name, length=0.0, aperture=aperture, n_steps=0)

    @abstractmethod
    def apply_kick(self, beam) -> None:
        ...

    @abstractmethod
    def kick_matrix(self, ref) -> np.ndarray:
        ...

    def advance_ref(self, ref) -> None:
        """Advance reference particle state. Override in RF gaps. Default: no-op."""
        pass

    def inverse_kick(self, beam, ref_entry) -> None:
        """Exactly undo :meth:`apply_kick` (backward tracking).

        Called by the backtracker with ``beam.ref`` still holding the
        element's EXIT reference state and ``ref_entry`` the recorded
        ENTRANCE state (from the forward replay table).  Thin kicks
        leave positions unchanged, so the forward kick's inputs are
        recoverable from the current coordinates — subclasses subtract
        the recomputed kick.  Elements without an implemented inverse
        raise so the backtracker fails loudly instead of silently
        mis-propagating.
        """
        raise NotImplementedError(
            f"{type(self).__name__} has no inverse_kick — this element "
            "cannot be backtracked yet.")

class FieldMapElement(Element):
    """Elements tracked via RK4 through imported field data."""

    # Position cursor of the step-by-step walk (see _advance_z_cursor).  Class-level
    # defaults on purpose: instances only carry these once they have been walked, so
    # lattices dumped attribute by attribute, deep copies and pickles of elements that
    # have never been walked are unchanged.
    _z_sync_idx = -1          # the _step_idx the cursor below is valid for
    _z_base = 0.0             # map-local position (mm) where the current step length began
    _z_unit = 0.0             # the current step length (mm)
    _z_n = 0                  # steps of _z_unit taken since _z_base
    _last_z_entry = None      # entry position (mm) of the last step (backtrack replay)
    _Z_REBASE_RTOL = 1e-9     # step lengths closer than this are the same length

    @abstractmethod
    def track_rk4(self, beam, ds: float) -> None:
        ...

    def _advance_z_cursor(self, ds: float) -> float:
        """Map-local position (mm) where this ``ds`` step begins; counts the step.

        The envelope slices a field map with sub-steps of different lengths (a half
        bundle and the short last bundle rarely share one), so the position cannot be
        ``_step_idx * ds``.  While the
        step length stays the same the result is exactly ``_step_idx * ds`` (the
        operations of the old formula, so constant-step walks keep their bits); when
        it changes, the position reached so far becomes the new origin.  A
        ``_step_idx`` set from outside (0 at element entry, a slice restarting its
        probes, a test) restarts the cursor at ``_step_idx * ds``.  Code that saves
        and restores the position around a walk of its own must use
        ``_z_cursor_state`` / ``_restore_z_cursor_state``: restoring ``_step_idx``
        alone is not seen as a restart when it equals the value the cursor reached.
        """
        idx = self._step_idx
        if idx == 0 or idx != self._z_sync_idx:
            self._z_base, self._z_unit, self._z_n = 0.0, ds, idx
        elif (ds != self._z_unit
              and abs(ds - self._z_unit) > self._Z_REBASE_RTOL * abs(self._z_unit)):
            self._z_base = self._z_base + self._z_n * self._z_unit
            self._z_unit, self._z_n = ds, 0
        z_entry = self._z_n * ds if self._z_base == 0.0 else self._z_base + self._z_n * ds
        self._z_n += 1
        self._step_idx = self._z_sync_idx = idx + 1
        self._last_z_entry = z_entry
        return z_entry

    def _z_cursor_state(self) -> tuple:
        return (self._step_idx, self._z_sync_idx, self._z_base, self._z_unit,
                self._z_n, self._last_z_entry)

    def _restore_z_cursor_state(self, state: tuple) -> None:
        (self._step_idx, self._z_sync_idx, self._z_base, self._z_unit,
         self._z_n, self._last_z_entry) = state

    def fitted_matrix(self, ref) -> np.ndarray:
        """Linearized 6x6 matrix for envelope mode. Placeholder until Phase 7."""
        return np.eye(6)

    def fitted_matrix_slice(self, ref, ds_mm: float) -> np.ndarray:
        """Linearised 6x6 map for a *ds_mm* slice starting at the element's
        current position (``_step_idx`` and the position cursor).

        Advances ``_step_idx`` by the number of sub-steps consumed so that
        successive calls in the envelope SC loop cover the full element in
        order.  Concrete subclasses must override this method.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not implement fitted_matrix_slice"
        )

    def advance_ref(self, ref) -> None:
        """Advance ref through field map. Placeholder until Phase 7."""
        pass

    def reset_run_state(self) -> None:
        """Clear per-run integrator state so a fresh tracker pass over a
        re-used lattice does not inherit stale calibration or sub-step
        cursors.  Subclasses extend this to clear additional caches.
        """
        self._step_idx = 0

class PassiveElement(Element):
    """Zero-length elements with no dynamics (aperture, marker, diag, SC comp)."""
    def __init__(self, name: str):
        super().__init__(name=name, length=0.0, aperture=0.0, n_steps=0)

    @abstractmethod
    def apply(self, beam) -> None:
        ...
