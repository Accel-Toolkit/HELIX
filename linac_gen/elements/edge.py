"""EDGE: pole-face rotation thin element used on both sides of a BEND.

Standard linear edge matrix (Brown / SLAC convention):

    M_edge[1, 0] = +tan(beta) / rho
    M_edge[3, 2] = -tan(beta - psi) / rho

where the fringe correction psi (dimensionless) is:

    psi = K1 * gap * (1 + sin^2(beta)) / (rho * cos(beta))

``gap`` is the full magnetic gap (mm), ``K1`` the fringe factor
(TraceWin default 0.45).  All lengths convert to metres internally so
the (mrad/mm) units of the matrix entries come out right.
"""
import math

import numpy as np

from linac_gen.elements.base import PassiveElement
from linac_gen.core.reference import ReferenceParticle


class Edge(PassiveElement):
    # Matrix-affecting params for the opt-in `get_element_matrix(cache=)`
    # path.  Edge transfer matrix reads pole_rotation, rho, gap, k1, k2, hv.
    _cache_keys: tuple[str, ...] = (
        "pole_rotation", "rho", "gap", "k1", "k2", "hv",
    )

    def __init__(self, name: str,
                 pole_rotation: float,
                 rho: float,
                 gap: float = 0.0,
                 k1: float = 0.45,
                 k2: float = 2.80,
                 aperture: float = 0.0,
                 hv: int = 0):
        super().__init__(name=name)
        self.pole_rotation = pole_rotation  # deg
        self.rho = rho                      # mm
        self.gap = gap                      # mm
        self.k1 = k1
        self.k2 = k2
        self.aperture_radius = aperture     # stored but not used at this stage
        self.hv = hv

    def apply(self, beam) -> None:
        """Apply the linear edge kick to alive particles."""
        M = self.transfer_matrix(beam.ref)
        alive = beam.alive_mask
        if np.any(alive):
            beam.particles[alive] = (M @ beam.particles[alive].T).T

    def transfer_matrix(self, ref: ReferenceParticle) -> np.ndarray:
        M = np.eye(6)
        beta_deg = self.pole_rotation
        if beta_deg == 0.0 or self.rho == 0.0:
            return M
        beta_rad = math.radians(beta_deg)
        rho_m = self.rho * 1e-3
        tan_b = math.tan(beta_rad)
        # Fringe correction psi (dimensionless).
        if self.gap > 0.0 and self.k1 != 0.0:
            gap_m = self.gap * 1e-3
            psi = (self.k1 * gap_m * (1.0 + math.sin(beta_rad) ** 2)
                   / (rho_m * max(math.cos(beta_rad), 1e-12)))
        else:
            psi = 0.0
        if self.hv == 0:          # horizontal bend plane
            M[1, 0] = +tan_b / rho_m
            M[3, 2] = -math.tan(beta_rad - psi) / rho_m
        else:                     # vertical bend plane
            M[3, 2] = +tan_b / rho_m
            M[1, 0] = -math.tan(beta_rad - psi) / rho_m
        return M
