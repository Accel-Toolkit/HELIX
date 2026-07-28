# linac_gen/elements/solenoid.py
"""Solenoid magnet element: couples x-y motion through Larmor rotation."""
import numpy as np
import math
from linac_gen.elements.base import TransferMapElement
from linac_gen.elements.mixins import Misalignment, FieldError
from linac_gen.core.reference import ReferenceParticle
from linac_gen.core.beam import Beam
from linac_gen.units import FIELD_ZERO_TOL


class Solenoid(TransferMapElement, Misalignment, FieldError):
    """Hard-edge solenoid with coupled x-y transfer matrix.

    Parameters
    ----------
    name : str
        Element name.
    length : float
        Physical length (mm).
    field : float
        On-axis magnetic field B0 (T).
    aperture : float
        Aperture radius (mm), default 0.0 (no aperture check).
    n_steps : int
        Number of integration steps for tracking, default 5.
    """

    # Matrix-affecting params for the opt-in `get_element_matrix(cache=)`
    # path.  Solenoid matrix uses length and effective_field
    # (= field × (1 + field_rel)).
    _cache_keys: tuple[str, ...] = ("length", "field", "field_rel")

    def __init__(self, name: str, length: float, field: float,
                 aperture: float = 0.0,
                 dx: float = 0.0, dy: float = 0.0, dz: float = 0.0,
                 tilt_deg: float = 0.0,
                 pitch_deg: float = 0.0, yaw_deg: float = 0.0,
                 field_rel: float = 0.0,
                 n_steps: int = 5):
        super().__init__(name=name, length=length, aperture=aperture, n_steps=n_steps)
        self.field = field  # B0 on-axis design field (T)
        self._init_misalignment(dx=dx, dy=dy, dz=dz, tilt_deg=tilt_deg,
                                pitch_deg=pitch_deg, yaw_deg=yaw_deg)
        self._init_field_error(field_rel=field_rel)

    @property
    def effective_field(self) -> float:
        """Design field with the per-seed relative error folded in."""
        return self.field * (1.0 + self.field_rel)

    def transfer_matrix(self, ref: ReferenceParticle, ds: float = None) -> np.ndarray:
        """6x6 transfer matrix for a solenoid section.

        The 4x4 transverse block uses the standard hard-edge solenoid model
        with Larmor rotation. The longitudinal part is drift-like (phase slip).

        Parameters
        ----------
        ref : ReferenceParticle
            Reference particle state.
        ds : float, optional
            Step length (mm). If None, uses self.length.
        """
        L_mm = ds if ds is not None else self.length
        L_m = L_mm * 1e-3
        M = np.eye(6)

        # Longitudinal: same as drift (always present)
        beta = ref.beta
        gamma = ref.gamma
        mass = ref.species.mass
        wl = ref.wavelength
        # Longitudinal phase slip; see drift.py for the β³ derivation.
        M[4, 5] = -360.0 * L_mm / (beta**3 * gamma**3 * mass * wl)

        eff_B = self.effective_field
        if L_m == 0.0 or abs(eff_B) < FIELD_ZERO_TOL:
            # Zero field or zero length: drift-like in transverse
            M[0, 1] = L_m
            M[2, 3] = L_m
            return M

        # Solenoid focusing parameter: k_s = charge * B0 / (2 * brho)
        # brho is always positive (|p/q|); charge_sign handles the direction.
        charge_sign = 1 if ref.species.charge > 0 else -1
        k_s = charge_sign * eff_B / (2 * ref.brho)  # 1/m (signed)

        phi = k_s * L_m  # Larmor angle (rad, signed)
        C = math.cos(phi)
        S = math.sin(phi)

        # 4x4 coupled transverse matrix in lab frame
        # Units: positions in mm, angles in mrad
        # k_s is in 1/m; CS/k_s is in m (mm/mrad); kSC is in 1/m (mrad/mm)
        M[0, 0] = C * C
        M[0, 1] = C * S / k_s      # m  (mm per mrad)
        M[0, 2] = S * C
        M[0, 3] = S * S / k_s      # m

        M[1, 0] = -k_s * S * C     # 1/m (mrad per mm)
        M[1, 1] = C * C
        M[1, 2] = -k_s * S * S     # 1/m
        M[1, 3] = S * C

        M[2, 0] = -S * C
        M[2, 1] = -S * S / k_s     # m
        M[2, 2] = C * C
        M[2, 3] = C * S / k_s      # m

        M[3, 0] = k_s * S * S      # 1/m
        M[3, 1] = -S * C
        M[3, 2] = -k_s * S * C     # 1/m
        M[3, 3] = C * C

        return M

    def track(self, beam: Beam, ds: float = None) -> None:
        """Track beam through solenoid, updating reference and particle coordinates."""
        L = ds if ds is not None else self.length
        beam.ref.s += L
        beam.ref.phi_s += 360.0 * L / (beam.ref.beta * beam.ref.wavelength)
        M = self.transfer_matrix(beam.ref, ds=L)
        alive = beam.alive_mask
        beam.particles[alive] = (M @ beam.particles[alive].T).T
