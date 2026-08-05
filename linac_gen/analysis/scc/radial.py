"""Axisymmetric Poisson-Boltzmann by damped Newton with a tridiagonal solve.

WHY THIS EXISTS. The 2-D solve in `poisson2d.py` is correct for the potential of a
GIVEN charge distribution, and it is validated against closed-form answers. But
the self-consistent compensation problem is nonlinear -- the trapped density
depends on the potential it creates -- and evaluating exp(-q phi / kT) explicitly
and feeding it back (damped Gummel) diverges: the potential ran away to ~1e17 V.

The cure is to linearise the exponential about the current iterate and fold it
into the OPERATOR, not the right-hand side. Doing that on the 2-D FFT solver would
need a screened (Helmholtz) Green's function with a spatially varying screening
length, which the cached Poisson kernel cannot represent.

For a round beam in a round pipe the problem is axisymmetric, so nothing is lost
by reducing it to one radial dimension -- and in 1-D the Newton Jacobian is
tridiagonal, which makes the whole thing a few dozen lines, exactly solvable per
iteration, and quadratically convergent. That is the right trade.

DISCRETISATION. Finite volume on r_i = (i + 1/2) dr, i = 0..N-1, so no node sits
on the axis singularity:

    (1/(r_i dr)) [ r_{i+1/2} (phi_{i+1} - phi_i)/dr
                 - r_{i-1/2} (phi_i - phi_{i-1})/dr ]  =  -rho_i / eps0

with a zero-flux (Neumann) condition at r = 0 -- enforced structurally by
r_{-1/2} = 0, which is why the flux form is used rather than a naive
second-difference -- and phi = 0 at the grounded wall.

Newton on F(phi) = L phi + rho(phi)/eps0:

    J = L - diag( q_t^2 n_t(phi) / (eps0 kT) )

The subtracted diagonal is exactly 1/lambda_Debye^2, i.e. Newton on this problem
IS the screened-Poisson operator. That is the whole reason it is stable where the
explicit iteration is not.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .constants import EPS0, Q_E


@dataclass(frozen=True)
class RadialGrid:
    n: int
    r_pipe_m: float

    @property
    def dr(self) -> float:
        return self.r_pipe_m / self.n

    @property
    def r(self) -> np.ndarray:
        """Cell centres; none lies on the axis."""
        return (np.arange(self.n) + 0.5) * self.dr

    @property
    def cell_area(self) -> np.ndarray:
        """Annulus area per unit length, for integrating a density to a charge."""
        edges = np.arange(self.n + 1) * self.dr
        return math.pi * (edges[1:] ** 2 - edges[:-1] ** 2)


def _laplacian_bands(grid: RadialGrid) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sub-, main- and super-diagonal of the cylindrical Laplacian.

    Face radii carry the r weighting; r_{-1/2} = 0 is what imposes the on-axis
    regularity condition without a special case.
    """
    n, dr = grid.n, grid.dr
    r = grid.r
    r_minus = r - 0.5 * dr           # inner face; zero for i = 0
    r_plus = r + 0.5 * dr            # outer face
    inv = 1.0 / (r * dr * dr)
    lower = inv * r_minus
    upper = inv * r_plus
    main = -(lower + upper)
    # Dirichlet phi = 0 at the wall: the outer face of the last cell couples to a
    # ghost node held at zero, so `upper[-1]` simply drops out of the system while
    # its contribution stays in `main`.
    upper_out = upper.copy()
    upper_out[-1] = 0.0
    return lower, main, upper_out


def _thomas(lower: np.ndarray, main: np.ndarray, upper: np.ndarray,
            rhs: np.ndarray) -> np.ndarray:
    """Tridiagonal solve (Thomas). O(n), no pivoting needed: the matrix is
    diagonally dominant because the Newton term only makes `main` more negative."""
    n = rhs.size
    c = np.empty(n)
    d = np.empty(n)
    beta = main[0]
    c[0] = upper[0] / beta
    d[0] = rhs[0] / beta
    for i in range(1, n):
        beta = main[i] - lower[i] * c[i - 1]
        c[i] = upper[i] / beta if i < n - 1 else 0.0
        d[i] = (rhs[i] - lower[i] * d[i - 1]) / beta
    x = np.empty(n)
    x[-1] = d[-1]
    for i in range(n - 2, -1, -1):
        x[i] = d[i] - c[i] * x[i + 1]
    return x


def solve_poisson_radial(grid: RadialGrid, rho: np.ndarray) -> np.ndarray:
    """Linear solve: potential of a given radial charge density, phi(R) = 0."""
    lower, main, upper = _laplacian_bands(grid)
    return _thomas(lower, main, upper, -rho / EPS0)


def solve_poisson_boltzmann_radial(
        grid: RadialGrid, rho_beam: np.ndarray, *,
        n_t0: float, q_t: float, kT_J: float,
        max_newton: int = 60, tol: float = 1e-10,
) -> tuple[np.ndarray, bool, int]:
    """Damped Newton for  L phi = -(rho_beam + q_t n_t0 exp(-q_t phi/kT)) / eps0.

    Returns (phi, converged, iterations).
    """
    lower, main, upper = _laplacian_bands(grid)
    phi_uncomp = solve_poisson_radial(grid, rho_beam)
    scale = max(float(np.max(np.abs(phi_uncomp))), 1e-30)

    # QUASI-NEUTRAL INITIAL GUESS. Starting Newton from the uncompensated
    # potential does not work: a 946 V well against a 3 eV population makes the
    # Boltzmann exponent ~315, so the first residual is astronomically large and
    # every step saturates the damping cap. The iterate then just walks downhill
    # at a fixed rate and never converges -- which is what made every n_t0 return
    # the same number.
    #
    # The standard cure is to start from local charge neutrality, where the
    # trapped density exactly cancels the beam:
    #     q_t n_t0 exp(-q_t phi / kT) = -rho_beam
    #  => phi = -(kT/q_t) ln( -rho_beam / (q_t n_t0) )
    # Newton then starts within a few kT of the root and converges quadratically.
    # Outside the beam (rho_beam = 0) neutrality gives no constraint, so fall back
    # to the wall value of zero and let the solve fill it in.
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(rho_beam != 0.0, -rho_beam / (q_t * n_t0), np.nan)
        phi = np.where(np.isfinite(ratio) & (ratio > 0.0),
                       -(kT_J / q_t) * np.log(np.maximum(ratio, 1e-300)), 0.0)
    # Never start deeper than the uncompensated well; that is a hard bound.
    phi = np.clip(phi, -np.abs(phi_uncomp), np.abs(phi_uncomp))

    for it in range(1, max_newton + 1):
        arg = np.clip(-q_t * phi / kT_J, -200.0, 200.0)
        n_t = n_t0 * np.exp(arg)
        rho = rho_beam + q_t * n_t

        # Residual F = L phi + rho/eps0
        Lphi = (lower * np.concatenate(([0.0], phi[:-1]))
                + main * phi
                + upper * np.concatenate((phi[1:], [0.0])))
        F = Lphi + rho / EPS0

        # Jacobian: the Boltzmann term contributes -q^2 n /(eps0 kT) on the
        # diagonal, which is -1/lambda_Debye^2. Folding it into the OPERATOR is
        # what makes this stable; leaving it on the RHS is what diverged.
        main_j = main - (q_t * q_t) * n_t / (EPS0 * kT_J)
        delta = _thomas(lower, main_j, upper, -F)

        # Damp the step so the exponential cannot be driven into overflow while
        # the iterate is still far from the root.
        step_cap = 4.0 * kT_J / abs(q_t)
        big = np.max(np.abs(delta))
        if big > step_cap:
            delta *= step_cap / big
        phi = phi + delta

        if np.max(np.abs(delta)) < tol * scale:
            return phi, True, it
    return phi, False, max_newton
