"""Thin lens element: thin-kick approximation for any focusing lens."""
import numpy as np
from linac_gen.elements.base import ThinKickElement


class ThinLens(ThinKickElement):
    """Thin converging/diverging lens with separate horizontal and vertical focal lengths.

    Applies the paraxial thin-lens kick:
        Δx'[mrad] = -x[mm] / f_x[m]
        Δy'[mrad] = -y[mm] / f_y[m]

    Unit derivation:
        Δx'[rad] = -x[m] / f_x[m]
        Δx'[mrad] = -x[m]/f_x[m] * 1e3 = -x[mm]*1e-3 / f_x[m] * 1e3 = -x[mm] / f_x[m]

    Parameters
    ----------
    fx : float
        Horizontal focal length in metres.  Use ``float('inf')`` (default) for
        no horizontal focusing.
    fy : float
        Vertical focal length in metres.  Use ``float('inf')`` (default) for
        no vertical focusing.
    """

    # Matrix-affecting params for the opt-in `get_element_matrix(cache=)`
    # path.  ThinLens kick matrix uses fx, fy.
    _cache_keys: tuple[str, ...] = ("fx", "fy")

    def __init__(self, name: str, fx: float = float('inf'), fy: float = float('inf'),
                 aperture: float = 0.0):
        super().__init__(name=name, aperture=aperture)
        self.fx = fx
        self.fy = fy

    # ------------------------------------------------------------------
    # ThinKickElement interface
    # ------------------------------------------------------------------

    def apply_kick(self, beam) -> None:
        """Apply thin-lens angular kicks to alive particles."""
        alive = beam.alive_mask
        if not np.any(alive):
            return
        if self.fx != float('inf'):
            # Δx'[mrad] = -x[mm] / fx[m]
            beam.particles[alive, 1] += -beam.particles[alive, 0] / self.fx
        if self.fy != float('inf'):
            # Δy'[mrad] = -y[mm] / fy[m]
            beam.particles[alive, 3] += -beam.particles[alive, 2] / self.fy

    def inverse_kick(self, beam, ref_entry) -> None:
        """Exactly undo apply_kick — positions are unchanged by the kick,
        so the same Δ is recomputable from the current coordinates."""
        alive = beam.alive_mask
        if not np.any(alive):
            return
        if self.fx != float('inf'):
            beam.particles[alive, 1] -= -beam.particles[alive, 0] / self.fx
        if self.fy != float('inf'):
            beam.particles[alive, 3] -= -beam.particles[alive, 2] / self.fy

    def kick_matrix(self, ref) -> np.ndarray:
        """Linearised 6x6 transfer matrix for the thin lens."""
        M = np.eye(6)
        if self.fx != float('inf'):
            M[1, 0] = -1.0 / self.fx   # mrad/mm
        if self.fy != float('inf'):
            M[3, 2] = -1.0 / self.fy
        return M
