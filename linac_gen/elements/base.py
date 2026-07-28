# linac_gen/elements/base.py
"""Abstract base classes for lattice elements (capability-based)."""
from abc import ABC, abstractmethod
import numpy as np

class Element(ABC):
    """Base class for all lattice elements."""
    def __init__(self, name: str, length: float, aperture: float, n_steps: int):
        self.name = name
        self.length = length
        self.aperture = aperture
        self.n_steps = n_steps

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
    @abstractmethod
    def track_rk4(self, beam, ds: float) -> None:
        ...

    def fitted_matrix(self, ref) -> np.ndarray:
        """Linearized 6x6 matrix for envelope mode. Placeholder until Phase 7."""
        return np.eye(6)

    def fitted_matrix_slice(self, ref, ds_mm: float) -> np.ndarray:
        """Linearised 6x6 map for a *ds_mm* slice starting at the element's
        current ``_step_idx``.

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
