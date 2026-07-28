"""General thin 2n-pole kick element (Multipole)."""
import math
import numpy as np
from linac_gen.elements.base import ThinKickElement


class Multipole(ThinKickElement):
    """Thin multipole element providing normal and skew kicks of arbitrary order.

    The kick uses the complex-coordinate convention:

        Δx' + i·Δy' = -Σ_{n=1}^{N} (b_n + i·a_n) / (n-1)! · (x + i·y)^{n-1}

    where positions are in metres and angles in radians.  Conversion from/to
    the code's mm / mrad convention is handled internally.

    Parameters
    ----------
    knl : list of float
        Integrated normal strengths [k0L, k1L, k2L, ...] in SI units
        (1/m^(n-2) for order n, i.e. k0L in T·m/Bρ, k1L in 1/m, k2L in 1/m², …).
        Index 0 → dipole (n=1), index 1 → quadrupole (n=2), index 2 → sextupole (n=3), …
    ksl : list of float
        Integrated skew strengths, same indexing as *knl*.
    dx, dy : float, default 0.0
        Element transverse offset in **mm**.  The kick is evaluated at
        ``(x − dx, y − dy)`` so a non-zero offset converts a pure
        multipole into an effective lower-order feed-down (e.g. a
        misaligned sextupole acquires a quadrupole + dipole component).
        Matches IMPACT-X's ``dx``/``dy`` parameters.
    tilt_deg : float, default 0.0
        Element tilt angle about the z-axis in **degrees**.  Applied
        as a 2-D rotation on the (x, y) frame before the kick and an
        inverse rotation on (x', y') after.  ``tilt_deg = 90 / n`` for
        an order-``n`` multipole rotates a pure normal element into a
        pure skew one (e.g. 45° tilts a normal quad into a skew quad).
    """

    def __init__(self, name: str, knl=None, ksl=None, aperture: float = 0.0,
                 dx: float = 0.0, dy: float = 0.0, tilt_deg: float = 0.0):
        super().__init__(name=name, aperture=aperture)
        self.knl = list(knl) if knl is not None else []
        self.ksl = list(ksl) if ksl is not None else []
        self.dx = float(dx)
        self.dy = float(dy)
        self.tilt_deg = float(tilt_deg)

    # ------------------------------------------------------------------
    # ThinKickElement interface
    # ------------------------------------------------------------------

    def _kick_mrad(self, beam):
        """Compute the multipole kick from CURRENT positions.

        Returns ``(alive_mask, dxp_mrad, dyp_mrad)`` — shared by
        :meth:`apply_kick` (adds) and :meth:`inverse_kick` (subtracts);
        the kick depends only on positions, which it never modifies, so
        the same computation serves both directions exactly.
        """
        alive = beam.alive_mask
        if not np.any(alive):
            return alive, None, None

        # Positions in metres (code stores mm).  Apply offset *before*
        # rotation so the tilt is taken about the element's centre.
        x_m = (beam.particles[alive, 0] - self.dx) * 1e-3
        y_m = (beam.particles[alive, 2] - self.dy) * 1e-3

        if self.tilt_deg != 0.0:
            theta = math.radians(self.tilt_deg)
            c, s = math.cos(theta), math.sin(theta)
            x_rot = c * x_m + s * y_m
            y_rot = -s * x_m + c * y_m
            x_m, y_m = x_rot, y_rot

        z = x_m + 1j * y_m

        dxp = np.zeros(len(x_m))  # rad
        dyp = np.zeros(len(x_m))  # rad

        max_order = max(len(self.knl), len(self.ksl))
        for n in range(1, max_order + 1):
            bn = self.knl[n - 1] if n <= len(self.knl) else 0.0
            an = self.ksl[n - 1] if n <= len(self.ksl) else 0.0
            if bn == 0.0 and an == 0.0:
                continue
            factor = 1.0 / math.factorial(n - 1)
            # Convention (MAD-X / standard accelerator physics):
            #   Δx' - i·Δy' = -(b_n + i·a_n)/(n-1)! · (x + iy)^{n-1}
            # Therefore:
            #   Δx' = +Re{ kick }
            #   Δy' = -Im{ kick }    ← note the sign flip on y
            kick = -(bn + 1j * an) * factor * (z ** (n - 1))
            dxp += kick.real
            dyp -= kick.imag  # sign flip: defocusing in y for normal focusing quad

        if self.tilt_deg != 0.0:
            theta = math.radians(self.tilt_deg)
            c, s = math.cos(theta), math.sin(theta)
            dxp_rot = c * dxp - s * dyp
            dyp_rot = s * dxp + c * dyp
            dxp, dyp = dxp_rot, dyp_rot

        return alive, dxp * 1e3, dyp * 1e3   # mrad

    def apply_kick(self, beam) -> None:
        """Apply the nonlinear multipole kick to all alive particles."""
        alive, dxp_mrad, dyp_mrad = self._kick_mrad(beam)
        if dxp_mrad is None:
            return
        beam.particles[alive, 1] += dxp_mrad
        beam.particles[alive, 3] += dyp_mrad

    def inverse_kick(self, beam, ref_entry) -> None:
        """Exactly undo apply_kick — the kick is position-only and the
        positions are untouched by it, so recomputing from the current
        coordinates recovers the forward kick precisely."""
        alive, dxp_mrad, dyp_mrad = self._kick_mrad(beam)
        if dxp_mrad is None:
            return
        beam.particles[alive, 1] -= dxp_mrad
        beam.particles[alive, 3] -= dyp_mrad

    def kick_matrix(self, ref) -> np.ndarray:
        """Linearised 6x6 transfer matrix.

        Only the quadrupole (n=2) term contributes to the linear optics.
        Tilt rotation is folded in by promoting the normal quad k1L to a
        rotated combination of normal + skew components.  Element offsets
        ``dx, dy`` produce a constant kick (not part of the *linear*
        matrix — they belong to the closed-orbit part) and are ignored
        here.
        """
        M = np.eye(6)
        # Effective normal/skew k1L after tilt: a rotation by θ converts
        # (k1L_n, k1L_s) → (k1L_n·cos2θ - k1L_s·sin2θ,  k1L_n·sin2θ + k1L_s·cos2θ).
        # The factor 2 in the angle reflects that the n-th multipole
        # rotates by n·θ under a frame rotation by θ.
        k1L_n = self.knl[1] if len(self.knl) >= 2 else 0.0
        k1L_s = self.ksl[1] if len(self.ksl) >= 2 else 0.0
        if self.tilt_deg != 0.0 and (k1L_n != 0.0 or k1L_s != 0.0):
            two_theta = math.radians(2.0 * self.tilt_deg)
            c2, s2 = math.cos(two_theta), math.sin(two_theta)
            k1L_eff_n = k1L_n * c2 - k1L_s * s2
            k1L_eff_s = k1L_n * s2 + k1L_s * c2
        else:
            k1L_eff_n, k1L_eff_s = k1L_n, k1L_s

        if k1L_eff_n != 0.0:
            # dx'[rad] = -k1L * x[m]  →  dx'[mrad] = -k1L * x[mm]
            M[1, 0] = -k1L_eff_n
            M[3, 2] = k1L_eff_n
        if k1L_eff_s != 0.0:
            M[1, 2] = k1L_eff_s
            M[3, 0] = k1L_eff_s
        return M


# ---------------------------------------------------------------------------
def Sextupole(name: str, k2L: float, skew: bool = False,
              aperture: float = 0.0, dx: float = 0.0, dy: float = 0.0,
              tilt_deg: float = 0.0) -> Multipole:
    """Convenience constructor for a thin sextupole.

    Parameters
    ----------
    k2L : float
        Integrated sextupole strength in 1/m² (MAD-X convention).
    skew : bool, default False
        If True, the strength goes to the skew slot.
    """
    knl = [0.0, 0.0, k2L] if not skew else [0.0, 0.0, 0.0]
    ksl = [0.0, 0.0, 0.0] if not skew else [0.0, 0.0, k2L]
    return Multipole(name, knl=knl, ksl=ksl, aperture=aperture,
                     dx=dx, dy=dy, tilt_deg=tilt_deg)


def Octupole(name: str, k3L: float, skew: bool = False,
             aperture: float = 0.0, dx: float = 0.0, dy: float = 0.0,
             tilt_deg: float = 0.0) -> Multipole:
    """Convenience constructor for a thin octupole.

    Parameters
    ----------
    k3L : float
        Integrated octupole strength in 1/m³ (MAD-X convention).
    skew : bool, default False
        If True, the strength goes to the skew slot.
    """
    knl = [0.0, 0.0, 0.0, k3L] if not skew else [0.0, 0.0, 0.0, 0.0]
    ksl = [0.0, 0.0, 0.0, 0.0] if not skew else [0.0, 0.0, 0.0, k3L]
    return Multipole(name, knl=knl, ksl=ksl, aperture=aperture,
                     dx=dx, dy=dy, tilt_deg=tilt_deg)
