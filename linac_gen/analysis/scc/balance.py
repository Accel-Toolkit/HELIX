"""Self-consistent space-charge compensation: f_c as an OUTPUT.

This is the module the whole rewrite exists for. The previous model relaxed f_c
toward a number the user typed into a spin box, so pressure changed only how FAST
compensation arrived, never how MUCH: 1e-8 mbar and 1e-4 mbar both settled at
mean f_c = 0.8150. Every "which gas compensates better" answer was the user's own
assumption read back to them.

THE MODEL

The beam digs an electrostatic well. Secondaries of the opposite sign are trapped
in it; those of the same sign are expelled promptly. The trapped population grows
until production balances loss, and the depth of the RESIDUAL well is what sets
f_c.

Two coupled statements, solved to a fixed point:

1. SHAPE. Trapped secondaries thermalise fast compared with the accumulation
   time, so their density follows Boltzmann in the self-consistent potential,
   n_t(r) = n_t0 exp(-q_t phi(r)/kT_t). That makes Poisson nonlinear; it is
   solved by damped Newton with the exponential folded into the operator (see
   `radial.py` -- doing it explicitly diverges).

2. AMOUNT. Production is gas ionisation along the line; loss is axial escape.
   While the well confines, only the exp(-W/kT) tail of the population clears it;
   once the population over-compensates and the residual potential flips, the
   same species is ACCELERATED out ballistically. That sign change is what makes
   the fixed point stable and unique, and it is found by bisection on n_t0.

   Getting the sign wrong here is not cosmetic: using |phi_axis| for the barrier
   makes over-neutralisation look exactly as confining as under-neutralisation,
   the loss rate never rises to meet production, and the root cannot be
   bracketed at all.

Raising the pressure raises production, which deepens compensation. f_c therefore
responds to pressure, gas species, beam current and beam energy -- which is the
entire point.

HONEST LIMITATIONS -- read before quoting a number from this.

* `T_trapped_eV` is a PARAMETER, not a computed quantity, and it is the largest
  free knob in the model: f_c depends on it logarithmically through the escape
  barrier. It stands in for the secondary birth spectrum plus whatever heating
  the population undergoes, neither of which is modelled.
* Axial escape is a 0-D barrier estimate, not a solved axial transport problem.
* Secondary-secondary collisions and recombination are neglected.
* The trapped species is assumed to be in thermal equilibrium with itself. For a
  well hundreds of volts deep against a few-eV population that is a strong
  assumption; it is most defensible near the well-compensated limit the code
  actually operates in.
* ABOVE ROUGHLY 1e-5 mbar THE MODEL OVER-NEUTRALISES. The balance lands past
  neutrality, the residual potential flips sign, and `over_neutralised` is set;
  f_c is clamped to 1.0. Real LEBTs saturate near 0.95-0.99 instead, because
  channels this model omits (resonant charge exchange with the gas, end-electrode
  fields, the fact that ions are born at rest at the LOCAL potential rather than
  drawn from a thermal bath) all cap the population. Treat any run with
  `over_neutralised` set as "essentially fully compensated", not as a number.
* There is no in-repo or literature reference that pins the absolute answer. What
  IS checked (tests/physics/test_scc.py): no gas gives f_c = 0 exactly; f_c rises
  monotonically with pressure and saturates below 1; the residual well settles to
  a few times kT; heavier gases compensate faster at equal pressure; and the
  build-up time tracks 1/(n sigma v).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .constants import Q_E
from .radial import (
    RadialGrid,
    solve_poisson_boltzmann_radial,
    solve_poisson_radial,
)


class SCCNotConverged(RuntimeError):
    """The self-consistent solve failed.

    Raised rather than returning a diverged number: a solver that quietly hands
    back nonsense is precisely the class of defect this rewrite is removing.
    """


@dataclass
class SCCResult:
    """Outcome of one self-consistent solve at a single z-slice."""
    fc: float                     # compensation fraction -- THE OUTPUT
    phi_axis_V: float             # residual on-axis potential (signed)
    phi_uncompensated_V: float    # what it would be with no secondaries
    well_depth_V: float           # |phi_axis|, the residual trapping depth
    trapped_density_axis_m3: float
    trapped_line_density_C_per_m: float
    converged: bool
    iterations: int
    # True when the balance lands past neutrality (residual potential flips sign
    # relative to the beam). f_c is clamped to 1.0 there, but the run is outside
    # the model's validity -- see the module docstring.
    over_neutralised: bool = False


class SelfConsistentSCC:
    """Solve the trapped-secondary balance on one transverse slice."""

    def __init__(self, grid: RadialGrid, T_trapped_eV: float = 3.0,
                 max_outer: int = 80, tol: float = 1e-4):
        self.grid = grid
        self.T_trapped_eV = float(T_trapped_eV)
        self.max_outer = int(max_outer)
        self.tol = float(tol)

    def solve(self, rho_beam: np.ndarray, *,
              beam_sign: float,
              production_per_m_per_s: float,
              lebt_length_m: float,
              trapped_mass_kg: float) -> SCCResult:
        """Fixed point of (Boltzmann shape) x (production/loss amount).

        Parameters
        ----------
        rho_beam
            Beam charge density on the radial grid, SIGNED, C/m^3.
        beam_sign
            Sign of the beam charge; the trapped species has the opposite sign.
        production_per_m_per_s
            Ionisation events per metre per second, n_gas * sigma_ion * I / |q|.
        lebt_length_m
            Axial escape path length.
        trapped_mass_kg
            Mass of the trapped species, for its thermal speed.
        """
        g = self.grid
        q_t = -math.copysign(Q_E, beam_sign)
        kT_J = self.T_trapped_eV * Q_E
        v_th = math.sqrt(2.0 * kT_J / max(trapped_mass_kg, 1e-40))
        transit = 0.5 * lebt_length_m / max(v_th, 1e-9)

        phi_uncomp = solve_poisson_radial(g, rho_beam)
        phi0_uncomp = float(phi_uncomp[0])

        if production_per_m_per_s <= 0.0 or phi0_uncomp == 0.0:
            # No gas -> no secondaries -> no compensation. This limit is a test.
            return SCCResult(0.0, phi0_uncomp, phi0_uncomp, abs(phi0_uncomp),
                             0.0, 0.0, True, 0)

        def balance(n_t0: float):
            """Return (loss_rate - production, phi) at a trial n_t0."""
            phi, ok, _ = solve_poisson_boltzmann_radial(
                g, rho_beam, n_t0=n_t0, q_t=q_t, kT_J=kT_J)
            if not ok:
                return None, phi
            # Escape energy SEEN BY THE TRAPPED SPECIES, axis to grounded wall:
            #     W = q_t * (phi_wall - phi_axis) = -q_t * phi_axis
            # W > 0 is a confining barrier; W < 0 means the residual potential
            # has flipped and now EXPELS the trapped species.
            #
            # Using |phi_axis| here is wrong twice over. It breaks the bracket
            # (over-neutralisation looks as confining as under-neutralisation, so
            # the loss rate never rises to meet production), and it misses the
            # self-limiting mechanism below.
            W = -q_t * float(phi[0])
            if W >= 0.0:
                # Confined: only the exp(-W/kT) tail of a thermal population
                # clears the barrier, over a thermal axial transit.
                tau_escape = transit * math.exp(min(W / kT_J, 200.0))
            else:
                # Repulsive: ions are ACCELERATED out, so the transit is
                # ballistic at sqrt(v_th^2 + 2|W|/m) rather than thermal. This is
                # what stops the population running away past neutrality -- the
                # more it over-compensates, the faster it is expelled, which
                # pins the fixed point just above f_c = 1 instead of letting the
                # well grow to tens of volts positive.
                v_out = math.sqrt(v_th * v_th
                                  + 2.0 * abs(W) / max(trapped_mass_kg, 1e-40))
                tau_escape = 0.5 * lebt_length_m / v_out
            n_t = n_t0 * np.exp(np.clip(-q_t * phi / kT_J, -200.0, 200.0))
            N_line = float((n_t * g.cell_area).sum())      # particles per metre
            return N_line / max(tau_escape, 1e-300) - production_per_m_per_s, phi

        # Bracket. Loss rate increases with n_t0 (more particles, shallower well,
        # exponentially easier escape), so the root is unique.
        lo, hi = self._bracket(balance)
        if lo is None:
            raise SCCNotConverged(
                "could not bracket the production/loss balance; production "
                f"{production_per_m_per_s:.3e} /m/s is outside the solvable range")

        phi = phi_uncomp
        n_t0 = lo
        converged = False
        it = 0
        for it in range(1, self.max_outer + 1):
            n_t0 = math.sqrt(lo * hi)            # geometric: n_t0 spans decades
            resid, phi = balance(n_t0)
            if resid is None:
                raise SCCNotConverged(f"inner Newton failed at n_t0={n_t0:.3e}")
            if resid < 0.0:
                lo = n_t0
            else:
                hi = n_t0
            if hi / lo < 1.0 + self.tol:
                converged = True
                break

        phi0 = float(phi[0])
        fc = 1.0 - phi0 / phi0_uncomp
        over = (phi0 * phi0_uncomp) < 0.0
        n_t = n_t0 * np.exp(np.clip(-q_t * phi / kT_J, -200.0, 200.0))
        return SCCResult(
            fc=float(min(max(fc, 0.0), 1.0)),
            phi_axis_V=phi0,
            phi_uncompensated_V=phi0_uncomp,
            well_depth_V=abs(phi0),
            trapped_density_axis_m3=float(n_t[0]),
            trapped_line_density_C_per_m=float(abs(q_t) * (n_t * g.cell_area).sum()),
            converged=converged, iterations=it, over_neutralised=over,
        )

    def _bracket(self, balance, n_lo: float = 1e4, n_hi: float = 1e22):
        """Expand outward until the balance changes sign."""
        r_lo, _ = balance(n_lo)
        r_hi, _ = balance(n_hi)
        if r_lo is None or r_hi is None or r_lo > 0.0 or r_hi < 0.0:
            return (None, None)
        return (n_lo, n_hi)
