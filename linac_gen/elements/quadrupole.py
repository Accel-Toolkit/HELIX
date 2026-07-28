"""Quadrupole magnet element."""
import numpy as np
import math
from linac_gen.elements.base import TransferMapElement
from linac_gen.elements.mixins import Misalignment, FieldError
from linac_gen.core.reference import ReferenceParticle
from linac_gen.core.beam import Beam


def _transverse_rotation(theta_rad: float) -> np.ndarray:
    """6x6 rotation around the longitudinal s-axis by *theta_rad* radians.

    Applies the same 2D rotation to (x, y) positions and to (x', y') angles,
    leaving the longitudinal (phi, dW) block identity.
    """
    c, s = math.cos(theta_rad), math.sin(theta_rad)
    R = np.eye(6)
    R[0, 0] = c;  R[0, 2] = -s
    R[1, 1] = c;  R[1, 3] = -s
    R[2, 0] = s;  R[2, 2] = c
    R[3, 1] = s;  R[3, 3] = c
    return R


class Quadrupole(TransferMapElement, Misalignment, FieldError):
    """Magnetic quadrupole with hard-edge model."""

    # Matrix-affecting params for the opt-in `get_element_matrix(cache=)`
    # path.  The transfer matrix uses length, effective_gradient
    # (= gradient × (1 + gradient_rel)), and skew_angle.  Higher-order
    # multipoles (g3..g6, gfr) are applied as ThinKicks around the
    # matrix, NOT inside it, so they are NOT in this fingerprint.
    _cache_keys: tuple[str, ...] = (
        "length", "gradient", "gradient_rel", "skew_angle",
    )

    def __init__(self, name: str, length: float, gradient: float,
                 aperture: float = 0.0,
                 skew_angle: float = 0.0,
                 g3: float = 0.0, g4: float = 0.0,
                 g5: float = 0.0, g6: float = 0.0,
                 gfr: float = 0.0,
                 dx: float = 0.0, dy: float = 0.0, dz: float = 0.0,
                 tilt_deg: float = 0.0,
                 pitch_deg: float = 0.0, yaw_deg: float = 0.0,
                 gradient_rel: float = 0.0,
                 n_steps: int = 5):
        super().__init__(name=name, length=length, aperture=aperture, n_steps=n_steps)
        self.gradient = gradient  # T/m design value
        self.skew_angle = skew_angle     # degrees; existing TraceWin field — matrix rotation in transfer_matrix
        self.g3, self.g4, self.g5, self.g6 = g3, g4, g5, g6
        self.gfr = gfr
        self._init_misalignment(dx=dx, dy=dy, dz=dz, tilt_deg=tilt_deg,
                                pitch_deg=pitch_deg, yaw_deg=yaw_deg)
        self._init_field_error(gradient_rel=gradient_rel)

    @property
    def effective_gradient(self) -> float:
        """Design gradient with the per-seed relative error folded in."""
        return self.gradient * (1.0 + self.gradient_rel)

    def transfer_matrix(self, ref: ReferenceParticle, ds: float = None) -> np.ndarray:
        M = self._normal_transfer_matrix(ref, ds)
        if self.skew_angle == 0.0:
            return M
        theta = math.radians(self.skew_angle)
        R = _transverse_rotation(theta)
        R_inv = _transverse_rotation(-theta)
        return R @ M @ R_inv

    def _normal_transfer_matrix(self, ref: ReferenceParticle, ds: float = None) -> np.ndarray:
        L_mm = ds if ds is not None else self.length
        L_m = L_mm * 1e-3
        M = np.eye(6)
        eff_G = self.effective_gradient
        if L_m == 0.0 or eff_G == 0.0:
            # Zero gradient = drift-like (identity in transverse, phase slip in longitudinal)
            L_m_drift = L_mm * 1e-3
            M[0, 1] = L_m_drift
            M[2, 3] = L_m_drift
            # Longitudinal phase slip; see drift.py for the β³ derivation.
            M[4, 5] = -360.0 * L_mm / (ref.beta**3 * ref.gamma**3 * ref.species.mass * ref.wavelength)
            return M
        # Effective focusing: k1 = q*G / p. For protons (q>0), positive G focuses in x.
        # For H- (q<0), positive G defocuses in x. brho = |p/q|, so k1 = charge_sign * G / brho.
        charge_sign = 1 if ref.species.charge > 0 else -1
        k1 = charge_sign * eff_G / ref.brho  # signed, 1/m^2
        k2 = abs(k1)
        k = math.sqrt(k2)
        kL = k * L_m
        if k1 > 0:  # focusing in x (positive k1)
            cos_kL = math.cos(kL)
            sin_kL = math.sin(kL)
            cosh_kL = math.cosh(kL)
            sinh_kL = math.sinh(kL)
            M[0, 0] = cos_kL
            M[0, 1] = sin_kL / k
            M[1, 0] = -k * sin_kL
            M[1, 1] = cos_kL
            M[2, 2] = cosh_kL
            M[2, 3] = sinh_kL / k
            M[3, 2] = k * sinh_kL
            M[3, 3] = cosh_kL
        else:
            cosh_kL = math.cosh(kL)
            sinh_kL = math.sinh(kL)
            cos_kL = math.cos(kL)
            sin_kL = math.sin(kL)
            M[0, 0] = cosh_kL
            M[0, 1] = sinh_kL / k
            M[1, 0] = k * sinh_kL
            M[1, 1] = cosh_kL
            M[2, 2] = cos_kL
            M[2, 3] = sin_kL / k
            M[3, 2] = -k * sin_kL
            M[3, 3] = cos_kL
        beta = ref.beta
        gamma = ref.gamma
        mass = ref.species.mass
        wl = ref.wavelength
        # Longitudinal phase slip; see drift.py for the β³ derivation.
        M[4, 5] = -360.0 * L_mm / (beta**3 * gamma**3 * mass * wl)
        return M

    def track(self, beam: Beam, ds: float = None) -> None:
        L = ds if ds is not None else self.length
        beam.ref.s += L
        beam.ref.phi_s += 360.0 * L / (beam.ref.beta * beam.ref.wavelength)
        # Split-operator: half of the higher-order multipole kick, then the
        # linear matrix transport, then the other half.  Distributes the
        # element's total g3/g4/g5/g6 content uniformly across substeps —
        # the integrated kick over the full magnet equals the design value.
        # No-op when all of g3..g6 are zero (the common case).
        self._apply_higher_multipole_kick(beam, ds_slice=L, fraction=0.5)
        M = self.transfer_matrix(beam.ref, ds=L)
        alive = beam.alive_mask
        beam.particles[alive] = (M @ beam.particles[alive].T).T
        self._apply_higher_multipole_kick(beam, ds_slice=L, fraction=0.5)

    def _apply_higher_multipole_kick(self, beam: Beam,
                                      ds_slice: float, fraction: float) -> None:
        """Apply ``fraction`` of the slice's share of g3/g4/g5/g6 as a
        thin Multipole kick.

        Mapping (MAD-X integrated-strength convention):
            knl[2] = g3 * Δs / Bρ        (1/m²,  sextupole)
            knl[3] = g4 * Δs / Bρ        (1/m³,  octupole)
            knl[4] = g5 * Δs / Bρ        (1/m⁴,  decapole)
            knl[5] = g6 * Δs / Bρ        (1/m⁵,  dodecapole)

        ``Δs`` here is the substep length in metres (mm·1e-3).  The
        ``fraction`` factor (0.5 in the split-operator scheme above)
        halves the kick for the entrance/exit half-applications.
        """
        if (self.g3 == 0.0 and self.g4 == 0.0
                and self.g5 == 0.0 and self.g6 == 0.0):
            return
        from linac_gen.elements.multipole import Multipole
        L_m = ds_slice * 1e-3
        brho = beam.ref.brho
        if brho == 0.0:
            return
        # MAD-X ordering in `knl`: index 0 = dipole, 1 = quad, 2 = sext, ...
        # We zero indices 0 and 1 (the dipole / linear-quad slots) since
        # those are handled by the closed-orbit + transfer-matrix paths.
        knl = [0.0, 0.0,
               fraction * self.g3 * L_m / brho,
               fraction * self.g4 * L_m / brho,
               fraction * self.g5 * L_m / brho,
               fraction * self.g6 * L_m / brho]
        # Trim trailing zeros to keep the inner loop short.
        while knl and knl[-1] == 0.0:
            knl.pop()
        if not knl:
            return
        kick = Multipole(self.name + "__multipole", knl=knl)
        kick.apply_kick(beam)
