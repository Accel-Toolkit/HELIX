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
        """6x6 thin-lens edge matrix for a single pole-face rotation *e*.

        A positive pole-face angle DEFOCUSES in the bend plane and
        focuses in the other plane (rho > 0):
            M_edge_x = [[1, 0], [ tan(e)/rho, 1]]
            M_edge_y = [[1, 0], [-tan(e)/rho, 1]]
        so a rectangular magnet (e1 = e2 = |theta|/2, either bend
        direction) has ~zero net focusing in the bend plane and focuses
        in the other plane.  Same convention as the standalone Edge
        element and TraceWin's EDGE card.
        """
        M = np.eye(6)
        if e_deg == 0.0 or abs(rho_m) < 1e-12:
            return M
        tan_e = math.tan(math.radians(e_deg))
        M[1, 0] = tan_e / rho_m    # bend plane: defocusing for e > 0  (1/m → mrad/mm: factor = 1)
        M[3, 2] = -tan_e / rho_m   # other plane: focusing for e > 0
        return M

    def _body_matrix(self, theta_deg: float, rho_mm: float,
                     ref: ReferenceParticle, length_mm: float = 0.0) -> np.ndarray:
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
        if theta_deg == 0.0 or rho_mm == 0.0:
            # No bending (a zero-angle card, or a field error that zeroes
            # the field): the magnet is still a pipe of its own length.
            if length_mm:
                L_m = length_mm * 1e-3
                M[0, 1] = L_m
                M[2, 3] = L_m
                M[4, 5] = -360.0 * length_mm / (ref.beta ** 3 * ref.gamma ** 3
                                                * ref.species.mass * ref.wavelength)
            return M

        # The bend direction is carried by sign(theta_deg) alone.  The
        # focusing trig sees |theta| and |rho| (the mirror image of a
        # magnet focuses identically) and only the dispersion column
        # D_x, D_x' changes sign:  M(-theta) = S M(+theta) S  with
        # S = diag(-1, -1, 1, 1, 1, 1).  A signed rho (Elegant importer)
        # therefore adds nothing to the BODY that the angle sign does not
        # already say; the element's own e1/e2 edges do see the sign of rho
        # (h·tan e with the signed curvature, see transfer_matrix).
        # Pinned against MAD-X (cpymad) in tests/io/test_madx_conventions.py
        # for both angle signs, tilt = 0 / +-pi/2 and N = 0, 0 < N < 1,
        # N = 1, N > 1, N < 0.
        theta = math.radians(abs(theta_deg))
        rho_m = abs(rho_mm) * 1e-3          # bending radius magnitude (m)
        L_m = rho_m * theta                 # arc length in metres
        sign = 1.0 if theta_deg >= 0.0 else -1.0
        # Dispersion coupling to dW (MeV): delta = dW / (beta^2 gamma m).
        # Sigma matrix is in (mm, mrad, mm, mrad, deg, MeV) units:
        # rho_m is in metres -> x1000 for x [mm]; 1/beta2gm yields rad/MeV
        # -> x1000 for x' [mrad].
        beta2gm = ref.beta ** 2 * ref.gamma * ref.species.mass

        def _path_length_row(kx2: float, s1: float) -> None:
            """Fill M[4,0], M[4,1] and M[4,5] — the path length of the orbit.

            A particle at x0 with angle x0' travels an extra
            ``dL = c1 x0 + c2 x0' + c3 delta`` through the magnet, with
            ``c1 = h S1``, ``c2 = h C1``, ``c3 = h^2 J`` where ``h`` is the
            signed curvature, ``S1 = sin(kL)/k``, ``C1 = (1-cos kL)/k^2``
            and ``J = (L - S1)/k^2`` (hyperbolic continuation for k^2 < 0).
            A longer path arrives later, so dphi = 360 dL/(beta lambda).

            In HELIX's units that row is a fixed multiple of the dispersion
            column this branch has just written — the symplectic relation —
            so it is built from M[0,5] and M[1,5] instead of re-deriving the
            trigonometry: the two rows then cannot drift apart, and the
            stabilised small-angle forms carry over for free.  ``s1`` is
            passed in rather than read back out of M[0,1] (the same number)
            so each branch states the dependency at its call site.

            Pinned against MAD-X and against TraceWin's own exported matrices
            (tests/io/test_madx_conventions.py, test_dipole_tracewin_anchor.py).
            """
            L = L_m
            u = kx2 * L * L
            # (L - S1)/k^2 subtracts two nearly equal numbers and divides by
            # k^2: at N ~ 1 every digit is lost.  The series is the same for
            # both signs (the hyperbolic branch is the elliptic one at
            # imaginary argument); worst relative error of this rule is 7e-13.
            J = (L ** 3 * (1.0 / 6.0 - u / 120.0 + u * u / 5040.0)
                 if abs(u) < 1e-3 else (L - s1) / kx2)
            k_phi = 0.36 * ref.beta * ref.gamma * ref.species.mass / ref.wavelength
            M[4, 0] = k_phi * M[1, 5]        # deg/mm
            M[4, 1] = k_phi * M[0, 5]        # deg/mrad
            # The drift-of-arc-length velocity slip, byte-for-byte the
            # expression this branch carried before (and the one in
            # drift.py), plus the momentum compaction, which does not
            # depend on the bend direction (c3 goes as h^2).  Grouped
            # exactly as torch_matrices._path_length_entries: that makes
            # row 4 bit-identical between the mirrors for 974 of 1024
            # sampled configurations, the rest differing by 2e-16 where
            # torch.sin and math.sin disagree in the last bit -- the same
            # floor the transverse block has always had (2.2e-16).
            M[4, 5] = (-360.0 * (L * 1e3)
                       / (ref.beta ** 3 * ref.gamma ** 3 * ref.species.mass
                          * ref.wavelength)
                       + 360000.0 * (J / (rho_m * rho_m))
                       / (ref.beta ** 3 * ref.gamma * ref.species.mass
                          * ref.wavelength))

        N = self.field_index

        if N == 0.0:
            # ---- Pure sector bend ----
            cos_t = math.cos(theta)
            sin_t = math.sin(theta)

            # Bending plane
            s1 = rho_m * sin_t               # S1 = sin(kL)/k with k = 1/rho
            M[0, 0] = cos_t
            M[0, 1] = s1                     # mm/mrad  (= m)
            M[1, 0] = -sin_t / rho_m         # mrad/mm  (= 1/m)
            M[1, 1] = cos_t

            # 2 sin^2(theta/2) is (1 - cos theta) without the cancellation:
            # the naive form loses 3.5e-9 relative at 0.01 deg and 27 % at
            # 1e-6 deg.  The combined-function branches below already use it.
            M[0, 5] = 2000.0 * sign * rho_m * math.sin(0.5 * theta) ** 2 / beta2gm   # mm / MeV
            M[1, 5] = 1000.0 * sign * sin_t / beta2gm                   # mrad / MeV

            # Other plane: pure drift
            M[2, 3] = L_m                    # mm/mrad coupling (= m)

            # Longitudinal: path length and phase slip
            _path_length_row(1.0 / (rho_m * rho_m), s1)
            return M

        # ---- Combined-function bend (N != 0) ----
        # Unit-consistent forms from integrating the linearised equations
        # of motion with k_x^2 = (1 - N)/rho^2 and k_y^2 = N/rho^2.  The
        # body focusing is independent of the bend direction; `sign`
        # enters the dispersion column only.
        kx2 = (1.0 - N) / (rho_m * rho_m)
        ky2 = N / (rho_m * rho_m)
        L = L_m

        # --- Bending plane ---
        # The 2x2 block always comes from the trig (cos/sin and cosh/sinh
        # are exact to rounding for any k L, including the R21 = -k^2 L
        # term a thin slice must keep).  Only the dispersion divides by
        # kx2: for k L below ~3e-3 the plain (1 - cos) loses digits, so
        # the equivalent 2 sin^2(k L / 2) is used there, and the exact
        # kx2 = 0 case (N = 1) takes the parabolic limit.
        kL2 = kx2 * L * L
        if kx2 > 0.0:
            kx = math.sqrt(kx2)
            cx = math.cos(kx * L)
            sx = math.sin(kx * L)
            s1 = sx / kx                     # S1 = sin(kL)/k
            M[0, 0] = cx
            M[0, 1] = s1                     # mm/mrad (= m)
            M[1, 0] = -kx * sx               # mrad/mm (= 1/m)
            M[1, 1] = cx
            one_minus_cx = (1.0 - cx) if kL2 >= 1e-5 else 2.0 * math.sin(0.5 * kx * L) ** 2
            M[0, 5] = 1000.0 * sign * one_minus_cx / (rho_m * kx2) / beta2gm     # mm/MeV
            M[1, 5] = 1000.0 * sign * sx / (rho_m * kx) / beta2gm                # mrad/MeV
        elif kx2 < 0.0:
            kx = math.sqrt(-kx2)
            ch = math.cosh(kx * L)
            sh = math.sinh(kx * L)
            s1 = sh / kx                     # S1 = sinh(|k|L)/|k|
            M[0, 0] = ch
            M[0, 1] = s1
            M[1, 0] = kx * sh
            M[1, 1] = ch
            # Hyperbolic branch: (1 - ch)/kx2 with kx2 < 0 is the analytic
            # continuation of (1 - cx)/kx2 (both positive).
            one_minus_ch = (1.0 - ch) if kL2 <= -1e-5 else -2.0 * math.sinh(0.5 * kx * L) ** 2
            M[0, 5] = 1000.0 * sign * one_minus_ch / (rho_m * kx2) / beta2gm     # mm/MeV
            M[1, 5] = 1000.0 * sign * sh / (rho_m * kx) / beta2gm                # mrad/MeV
        else:
            # kx2 == 0 (N = 1 exactly): the bend plane is a drift of length L
            # with the parabolic dispersion limit  D = L^2/(2 rho),  D' = L/rho.
            s1 = L                           # S1 -> L as k -> 0
            M[0, 1] = s1
            M[0, 5] = 1000.0 * sign * L * L / (2.0 * rho_m) / beta2gm      # mm/MeV
            M[1, 5] = 1000.0 * sign * L / rho_m / beta2gm                  # mrad/MeV

        # --- Other plane ---
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

        # --- Longitudinal: path length and phase slip ---
        _path_length_row(kx2, s1)

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
        # Field error: B(1+δ) in a magnet of FIXED length — the angle
        # scales by (1+δ) and the bending radius by 1/(1+δ); L = ρθ is
        # unchanged (until 2026-09-06 only the angle scaled, so a field
        # error stretched the magnet).  The element's own e1/e2 edges use
        # the same scaled radius (h·tan e with h = 1/ρ_eff), with the sign
        # of ``self.rho`` (a signed rho flips the edge sense, as in MAD).
        eff_angle = self.effective_angle
        if eff_angle == self.angle:
            rho_mm = self.rho
        elif eff_angle == 0.0:
            rho_mm = 0.0                       # zero field: a pipe of length L
        else:
            # same expression as the torch mirror (design/effective, then
            # multiply) so the two paths agree to the last bit
            rho_mm = self.rho * (self.angle / eff_angle)
        L_full = self.length
        if ds is not None:
            theta_deg = eff_angle * (ds / L_full) if L_full != 0.0 else 0.0
            L_part = ds
        else:
            theta_deg = eff_angle
            L_part = L_full

        # TraceWin convention for hv=1: body matrix uses |angle|; angle sign
        # encodes the bend direction (up vs down) which only flips the
        # dispersion entries M[2,5], M[3,5] post-swap.
        body_theta = abs(theta_deg) if self.hv == 1 else theta_deg
        M_body = self._body_matrix(body_theta, rho_mm, ref, length_mm=L_part)

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
                # Bend goes "down": the body was built from |theta|, so the
                # dispersion in y and the path length's dependence on y flip
                # sign.  M[4,5] does NOT: the compaction goes as h^2.
                M[2, 5] *= -1
                M[3, 5] *= -1
                M[4, 2] *= -1
                M[4, 3] *= -1
        return M

    def track(self, beam: Beam, ds: float = None) -> None:
        L = ds if ds is not None else self.length
        beam.ref.s += L
        beam.ref.phi_s += 360.0 * L / (beam.ref.beta * beam.ref.wavelength)
        M = self.transfer_matrix(beam.ref, ds=L)
        alive = beam.alive_mask
        beam.particles[alive] = (M @ beam.particles[alive].T).T
