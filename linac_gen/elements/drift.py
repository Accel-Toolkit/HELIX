"""Drift space element."""
import numpy as np
from linac_gen.elements.base import TransferMapElement
from linac_gen.elements.mixins import Misalignment
from linac_gen.core.reference import ReferenceParticle
from linac_gen.core.beam import Beam


class Drift(TransferMapElement, Misalignment):
    """Field-free drift space."""

    # Matrix-affecting params for the opt-in `get_element_matrix(cache=)`
    # path.  Drift's transfer matrix depends only on `length` (ref.beta /
    # gamma / mass come from the ref, captured separately by the cache).
    _cache_keys: tuple[str, ...] = ("length",)

    def __init__(self, name: str, length: float, aperture: float = 0.0,
                 aperture_y: float | None = None,
                 x_shift: float = 0.0, y_shift: float = 0.0,
                 dx: float = 0.0, dy: float = 0.0, dz: float = 0.0,
                 tilt_deg: float = 0.0,
                 pitch_deg: float = 0.0, yaw_deg: float = 0.0,
                 n_steps: int = 1):
        super().__init__(name=name, length=length, aperture=aperture, n_steps=n_steps)
        self.aperture_y = aperture_y
        # x_shift / y_shift are TraceWin DRIFT card fields (mm) describing
        # static beam-frame offsets accumulated upstream — they fold into
        # the misalignment dx / dy by adding directly.
        self.x_shift = x_shift
        self.y_shift = y_shift
        self._init_misalignment(dx=dx + x_shift, dy=dy + y_shift,
                                dz=dz, tilt_deg=tilt_deg,
                                pitch_deg=pitch_deg, yaw_deg=yaw_deg)

    def transfer_matrix(self, ref: ReferenceParticle, ds: float = None) -> np.ndarray:
        L = ds if ds is not None else self.length  # mm
        M = np.eye(6)
        L_m = L * 1e-3  # mm -> m for mrad coupling
        M[0, 1] = L_m
        M[2, 3] = L_m
        beta = ref.beta
        gamma = ref.gamma
        mass = ref.species.mass
        wl = ref.wavelength
        # Longitudinal phase slip for a drift of length L.
        # Derivation: Δt = -Δs·Δβ/(β²·c), Δβ = ΔW/(β·γ³·m) → Δt = -Δs·ΔW/(β³·γ³·m·c)
        # Δφ[deg] = -360·f·Δt ⇒ dΔφ/dΔW = -360·Δs/(β³·γ³·m·λ).
        # Matches TraceWin's R_zz = [[1, Δs/γ²], [0, 1]] after converting (z, δ) to (Δφ, ΔW).
        M[4, 5] = -360.0 * L / (beta**3 * gamma**3 * mass * wl)
        return M

    def track(self, beam: Beam, ds: float = None) -> None:
        L = ds if ds is not None else self.length
        beam.ref.s += L
        beam.ref.phi_s += 360.0 * L / (beam.ref.beta * beam.ref.wavelength)
        M = self.transfer_matrix(beam.ref, ds=L)
        alive = beam.alive_mask
        beam.particles[alive] = (M @ beam.particles[alive].T).T
