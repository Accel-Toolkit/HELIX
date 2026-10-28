"""The incoherent space-charge tune shift of a ring, analytically (L. J.
Laslett, "On intensity limitations imposed by transverse space-charge
effects in circular particle accelerators", BNL-7534, 1963, p. 324).

For a Gaussian beam at its centre (a KV beam of the same rms sizes: half of
it, the same for every particle),

    dQ_x = -(r_c N_b f / (2 pi beta0^2 gamma0^3)) ∮ beta_x / (sigma_x (sigma_x + sigma_y)) ds,

f the normalized line density where the particle is (1/C for a coasting
beam; 1/(sqrt(2 pi) sigma_zeta) at the centre of a Gaussian bunch) and
sigma_x = sqrt(eps_x beta_x + (D_x sigma_delta)^2).  The integral is taken
over the optics table (the start and every element's exit, trapezoids) — an
estimate within a few per cent for a ring sampled every few metres.
"""
from __future__ import annotations

import math

import numpy as np

from linac_gen.ring.sc.solver import classical_radius


def tune_shift(optics, *, emit_x: float, emit_y: float, intensity: float,
               line_density: float, sigma_delta: float = 0.0, mass_mev: float,
               gaussian: bool = True) -> tuple[float, float]:
    """(dQ_x, dQ_y) for geometric rms emittances ``emit_x``, ``emit_y`` [m],
    ``intensity`` real particles and the normalized line density
    ``line_density`` [1/m] at the particle."""
    tab = optics.table
    s = np.concatenate([[0.0], np.asarray(tab["s"], dtype=float)])
    st = optics.start

    def col(k, k0):
        return np.concatenate([[float(st[k0])], np.asarray(tab[k], dtype=float)])
    bx, by = col("betx", "betx"), col("bety", "bety")
    dx, dy = col("dx", "dx"), col("dy", "dy")
    sx = np.sqrt(np.maximum(emit_x * bx + (dx * sigma_delta) ** 2, 0.0))
    sy = np.sqrt(np.maximum(emit_y * by + (dy * sigma_delta) ** 2, 0.0))
    trap = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        gx = bx / (sx * (sx + sy))
        gy = by / (sy * (sx + sy))
        ix = float(trap(gx, s))
        iy = float(trap(gy, s))
    b0, g0 = float(optics.beta0), float(optics.gamma0)
    k = classical_radius(mass_mev) * float(intensity) * float(line_density) / (
        2.0 * math.pi * b0 * b0 * g0 ** 3)
    if not gaussian:
        k *= 0.5
    return -k * ix, -k * iy
