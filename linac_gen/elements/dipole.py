"""Sector-bend dipole element with edge focusing."""
import numpy as np
import math
from linac_gen.elements.base import TransferMapElement
from linac_gen.elements.mixins import Misalignment, FieldError
from linac_gen.core.reference import ReferenceParticle
from linac_gen.core.beam import Beam


class Dipole(TransferMapElement, Misalignment, FieldError):
    """Sector-bend dipole magnet with optional edge focusing.

    Uses the standard sector-bend transfer matrix.  Edge angles *e1* and *e2*
    apply thin-lens edge-focusing matrices at entrance and exit respectively.

    Parameters
    ----------
    angle : float
        Total bend angle in degrees.
    rho : float
        Bending radius in mm.  The arc length is ``|rho| * |angle| * pi/180``.
    e1, e2 : float
        Entrance / exit pole-face (edge) angles in degrees.
    field_rel : float
        Relative magnet-strength error: ``B → B·(1 + field_rel)`` is
        equivalent to ``angle → angle·(1 + field_rel)`` since arc length is
        fixed by geometry.  Drives orbit and dispersion errors.
    """

    # Matrix-affecting params for the opt-in `get_element_matrix(cache=)`
    # path.  Sector-bend matrix reads angle, rho, e1, e2, field_index, hv,
    # plus field_rel (folded into effective_angle).
    _cache_keys: tuple[str, ...] = (
        "angle", "rho", "e1", "e2", "field_index", "hv", "field_rel",
    )

    def __init__(self, name: str, angle: float, rho: float,
                 e1: float = 0.0, e2: float = 0.0,
                 field_index: float = 0.0,
                 aperture: float = 0.0,
                 hv: int = 0,
                 dx: float = 0.0, dy: float = 0.0, dz: float = 0.0,
                 tilt_deg: float = 0.0,
                 pitch_deg: float = 0.0, yaw_deg: float = 0.0,
                 field_rel: float = 0.0,
                 n_steps: int = 5):
        length = abs(rho) * abs(angle) * math.pi / 180.0  # arc length in mm
        super().__init__(name=name, length=length, aperture=aperture, n_steps=n_steps)
        self.angle = angle   # design bend angle (deg)
        self.rho = rho       # bending radius (mm)
        self.e1 = e1         # entrance edge angle (deg)
        self.e2 = e2         # exit edge angle (deg)
        self.field_index = field_index  # combined-function field index N
        self.hv = hv         # 0=horizontal bend, 1=vertical bend (TraceWin)
        self._init_misalignment(dx=dx, dy=dy, dz=dz, tilt_deg=tilt_deg,
                                pitch_deg=pitch_deg, yaw_deg=yaw_deg)
        self._init_field_error(field_rel=field_rel)

    @property
    def effective_angle(self) -> float:
        """Design bend angle with the per-seed magnet-strength error folded in."""
        return self.angle * (1.0 + self.field_rel)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _edge_matrix(e_deg: float, rho_m: float) -> np.ndarray:
        """6x6 thin-lens edge-focusing matrix for a single edge.

        The horizontal plane is focusing (positive rho > 0):
            M_edge_x = [[1, 0], [tan(e)/rho, 1]]
        The vertical plane is defocusing:
            M_edge_y = [[1, 0], [-tan(e)/rho, 1]]
        """
        M = np.eye(6)
        if e_deg == 0.0 or abs(rho_m) < 1e-12:
            return M
        tan_e = math.tan(math.radians(e_deg))
        M[1, 0] = tan_e / rho_m    # horizontal focusing  (1/m  → mrad/mm: factor = 1)
        M[3, 2] = -tan_e / rho_m   # vertical defocusing
        return M

    def _body_matrix(self, theta_deg: float, rho_mm: float,
                     ref: ReferenceParticle) -> np.ndarray:
        """6x6 sector-bend body matrix for bend angle *theta_deg* and
        bending radius *rho_mm*, with optional combined-function field
        index ``self.field_index``.

        Focusing strengths (with ``N = field_index``):

            k_x^2 = (1 - N) / rho^2        (horizontal)
            k_y^2 = N / rho^2              (vertical)

        For ``N = 0`` this reduces to the classical pure sector bend
        (horizontal focusing from curvature, vertical pure drift) and the
        dispersion / phase-slip terms are kept compatible with the legacy
        implementation (beam-energy units of mm/MeV and mrad/MeV).

        Coordinate conventions (same as the rest of the code):
          * transverse positions in mm, angles in mrad
          * rho in mm internally; converted to metres where SI physics needed
        """
        M = np.eye(6)
        if theta_deg == 0.0:
            return M

        theta = math.radians(theta_deg)
        rho_m = rho_mm * 1e-3               # bending radius in metres
        L_m = abs(rho_m) * abs(theta)       # arc length in metres

        N = self.field_index

        if N == 0.0:
            # ---- Pure sector bend (legacy behaviour) ----
            cos_t = math.cos(theta)
            sin_t = math.sin(theta)

            # Horizontal (bending) plane
            M[0, 0] = cos_t
            M[0, 1] = rho_m * sin_t          # mm/mrad  (= m)
            M[1, 0] = -sin_t / rho_m         # mrad/mm  (= 1/m)
            M[1, 1] = cos_t

            # Dispersion coupling to dW (MeV): delta = dW / (beta^2 gamma m)
            beta2gm = ref.beta ** 2 * ref.gamma * ref.species.mass
            # Sigma matrix is in (mm, mrad, mm, mrad, deg, MeV) units.
            # rho_m is in metres → ×1000 for x [mm].
            # 1/(beta2gm) yields rad/MeV → ×1000 for x' [mrad].
            M[0, 5] = 1000.0 * rho_m * (1.0 - cos_t) / beta2gm   # mm / MeV
            M[1, 5] = 1000.0 * sin_t / beta2gm                   # mrad / MeV

            # Vertical plane: pure drift
            M[2, 3] = L_m                    # mm/mrad coupling (= m)

            # Longitudinal: phase slip (same as drift of length L_m)
            beta = ref.beta
            gamma = ref.gamma
            mass = ref.species.mass
            wl = ref.wavelength
            L_mm = L_m * 1e3
            # Longitudinal phase slip; see drift.py for the β³ derivation.
            M[4, 5] = -360.0 * L_mm / (beta ** 3 * gamma ** 3 * mass * wl)
            return M

        # ---- Combined-function bend (N != 0) ----
        # Use the unit-consistent forms that emerge from integrating the
        # linearised equations of motion.  Note: sign of the bend angle only
        # matters for the dispersion / longitudinal-coupling signs; the body
        # focusing is independent of direction.  We mirror the legacy
        # convention by using the signed theta for horizontal trig.
        kx2 = (1.0 - N) / (rho_m * rho_m)
        ky2 = N / (rho_m * rho_m)
        L = L_m
        sign = 1.0 if theta_deg >= 0.0 else -1.0

        # --- Horizontal plane ---
        if kx2 > 1e-30:
            kx = math.sqrt(kx2)
            cx = math.cos(kx * L)
            sx = math.sin(kx * L)
            M[0, 0] = cx
            M[0, 1] = sx / kx                # mm/mrad (= m)
            M[1, 0] = -kx * sx               # mrad/mm (= 1/m)
            M[1, 1] = cx

            # Dispersion to dW (MeV): delta = dW / (beta^2 gamma m)
            beta2gm = ref.beta ** 2 * ref.gamma * ref.species.mass
            # Same unit-conversion ×1000 as the pure-sector branch above
            # (m → mm for x, rad → mrad for x').
            M[0, 5] = 1000.0 * (1.0 - cx) / (rho_m * kx2) / beta2gm   # mm/MeV
            M[1, 5] = 1000.0 * sign * sx / (rho_m * kx) / beta2gm     # mrad/MeV
        elif kx2 < -1e-30:
            kx = math.sqrt(-kx2)
            ch = math.cosh(kx * L)
            sh = math.sinh(kx * L)
            M[0, 0] = ch
            M[0, 1] = sh / kx
            M[1, 0] = kx * sh
            M[1, 1] = ch
            beta2gm = ref.beta ** 2 * ref.gamma * ref.species.mass
            # Dispersion (hyperbolic branch): 1/(rho * kx^2) still works since
            # kx^2 is negative; use (ch - 1)/(rho_m * kx2).
            M[0, 5] = (ch - 1.0) / (rho_m * kx2) / beta2gm
            M[1, 5] = sign * sh / (rho_m * kx) / beta2gm
        else:
            # kx2 ≈ 0: horizontal is a pure drift over length L.
            M[0, 1] = L

        # --- Vertical plane ---
        if ky2 > 1e-30:
            ky = math.sqrt(ky2)
            cy = math.cos(ky * L)
            sy = math.sin(ky * L)
            M[2, 2] = cy
            M[2, 3] = sy / ky
            M[3, 2] = -ky * sy
            M[3, 3] = cy
        elif ky2 < -1e-30:
            ky = math.sqrt(-ky2)
            ch = math.cosh(ky * L)
            sh = math.sinh(ky * L)
            M[2, 2] = ch
            M[2, 3] = sh / ky
            M[3, 2] = ky * sh
            M[3, 3] = ch
        else:
            # N = 0 exactly (unreachable here — handled above) or numerically.
            M[2, 3] = L

        # --- Longitudinal: phase slip (treat as drift of length L_m) ---
        beta = ref.beta
        gamma = ref.gamma
        mass = ref.species.mass
        wl = ref.wavelength
        L_mm = L_m * 1e3
        # Longitudinal phase slip; see drift.py for the β³ derivation.
        M[4, 5] = -360.0 * L_mm / (beta ** 3 * gamma ** 3 * mass * wl)

        return M

    # ------------------------------------------------------------------
    # TransferMapElement interface
    # ------------------------------------------------------------------

    def transfer_matrix(self, ref: ReferenceParticle, ds: float = None) -> np.ndarray:
        """Return the 6x6 transfer matrix for the full element or a slice *ds*.

        When *ds* is provided the bend angle is scaled proportionally.
        Edge angles are applied only for the full element (ds=None).
        For hv=1 the bend curves in the y plane: roles of (x, x') and
        (y, y') are swapped (and the dispersion couples to y, not x).
        """
        use_edges = ds is None
        L_full = self.length  # mm
        # Field-error scaling: B(1+δ) ≡ angle·(1+δ) for fixed arc length.
        eff_angle = self.effective_angle

        if ds is not None:
            # Scale angle proportionally to the requested slice length
            theta_deg = eff_angle * (ds / L_full) if L_full != 0.0 else 0.0
            rho_mm = self.rho
        else:
            theta_deg = eff_angle
            rho_mm = self.rho

        # TraceWin convention for hv=1: body matrix uses |angle|; angle sign
        # encodes the bend direction (up vs down) which only flips the
        # dispersion entries M[2,5], M[3,5] post-swap.
        body_theta = abs(theta_deg) if self.hv == 1 else theta_deg
        M_body = self._body_matrix(body_theta, rho_mm, ref)

        if use_edges:
            rho_m = rho_mm * 1e-3
            M_ent = self._edge_matrix(self.e1, rho_m)
            M_ext = self._edge_matrix(self.e2, rho_m)
            M = M_ext @ M_body @ M_ent
        else:
            M = M_body

        if self.hv == 1:
            P = np.eye(6)
            P[[0, 1, 2, 3]] = P[[2, 3, 0, 1]]
            M = P @ M @ P
            if theta_deg < 0:
                # Bend goes "down": dispersion in y flips sign
                M[2, 5] *= -1
                M[3, 5] *= -1
        return M

    def track(self, beam: Beam, ds: float = None) -> None:
        L = ds if ds is not None else self.length
        beam.ref.s += L
        beam.ref.phi_s += 360.0 * L / (beam.ref.beta * beam.ref.wavelength)
        M = self.transfer_matrix(beam.ref, ds=L)
        alive = beam.alive_mask
        beam.particles[alive] = (M @ beam.particles[alive].T).T
