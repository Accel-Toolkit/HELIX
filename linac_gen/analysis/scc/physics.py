"""Core physics of the LEBT space-charge-compensation simulator.

Supports any singly-charged beam species (H±, D±, He⁺, ...) transiting a
round-pipe LEBT with solenoid focusing and residual-gas neutralisation.

Conventions
-----------
* All SI units.
* z is the longitudinal beam coordinate.
* Beam kinetic energy T [J], momentum p = gamma*m*v.
* beta = v/c, gamma = 1/sqrt(1 - beta^2).

Key equations (references in constants.py docstring):

  Generalized perveance:
      K = |q| I / (2 pi eps0 m c^3 beta^3 gamma^3)

  Characteristic current:
      I0 = 4 pi eps0 m c^3 / |q|

  Solenoid focusing (linear KV-type thin-lens limit on a round beam):
      k^2(z) = (q B(z) / (2 m c beta gamma))^2

  RMS envelope with space-charge compensation (Valerio-Lizarraga Eq 3.13):
      r''(z) + k0^2(z) r(z) - eps_g^2 / r^3 - K (1 - f_c(t,z)) / r = 0

  SCC time constant (Chauvin Eq 66 generalised to mixture):
      tau_scc = 1 / ( sum_i n_i sigma_i * v_b )

  Build-up (simple linear ramp, Valerio-Lizarraga Eq 3.15):
      f_c(t) = eta * min(1, t / tau_scc)

  Stripping loss per unit length (only for species with `can_strip=True`):
      dN/dz / N = -sum_i n_i sigma_strip_i
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .constants import (
    C_LIGHT, EPS0, K_BOLTZ, MBAR_TO_PA, Q_E,
    Species,
    sigma_ionization, sigma_stripping, sigma_capture,
    characteristic_current,
)


# ---------------------------------------------------------------------------
# Kinematics
# ---------------------------------------------------------------------------
def kinematic(species: Species, T_keV: float) -> tuple[float, float, float]:
    """Return (beta, gamma, momentum p [kg m/s]) for given kinetic energy."""
    T = T_keV * 1e3 * Q_E
    mc2 = species.rest_energy
    gamma = 1.0 + T / mc2
    beta = math.sqrt(max(0.0, 1.0 - 1.0 / (gamma * gamma)))
    p = gamma * beta * species.mass * C_LIGHT
    return beta, gamma, p


def velocity(species: Species, T_keV: float) -> float:
    beta, _, _ = kinematic(species, T_keV)
    return beta * C_LIGHT


# ---------------------------------------------------------------------------
# Perveance
# ---------------------------------------------------------------------------
def generalized_perveance(species: Species, T_keV: float, current_A: float) -> float:
    beta, gamma, _ = kinematic(species, T_keV)
    I0 = characteristic_current(species)
    # sign-agnostic: |q| * |I| / (2 pi eps0 m c^3 beta^3 gamma^3)
    return 2.0 * abs(current_A) / (I0 * beta ** 3 * gamma ** 3)


# ---------------------------------------------------------------------------
# Gas / compensation
# ---------------------------------------------------------------------------
@dataclass
class GasMix:
    """A homogeneous mixture of residual gases in the LEBT."""
    components: dict[str, float]   # gas-name -> partial pressure in mbar
    temperature_K: float = 300.0

    def number_densities(self) -> dict[str, float]:
        """Pa / (kT) for each component -> m^-3."""
        kT = K_BOLTZ * self.temperature_K
        return {g: p * MBAR_TO_PA / kT for g, p in self.components.items()}

    def total_pressure_mbar(self) -> float:
        return sum(self.components.values())


def compensation_time(mix: GasMix, species: Species, T_keV: float) -> float:
    """Steady-state SCC transient time tau_scc = 1/(sum n_i sigma_i v_b).

    Uses tabulated ionisation cross sections (Tabata-Shirai 2000 / Rudd 1992)
    interpolated at the projectile velocity.
    """
    vb = velocity(species, T_keV)
    nd = mix.number_densities()
    sum_ns = 0.0
    for name, n in nd.items():
        sigma = sigma_ionization(name, species, T_keV)
        sum_ns += n * sigma
    if sum_ns <= 0.0:
        return math.inf
    return 1.0 / (sum_ns * vb)


def beam_loss_rate_per_length(mix: GasMix, species: Species,
                              T_keV: float) -> float:
    """Inverse mean free path for the beam (1/m). Combines:

    - electron stripping  (negative-ion beams): H− + X → H⁰ + X + e⁻
    - electron capture    (positive-ion beams): H⁺ + X → H⁰ + X⁺

    For positive beams the capture cross-section is comparable to
    ionisation in the 5–50 keV range and produces real beam loss
    (neutral H is no longer transported by the magnetic LEBT).
    """
    nd = mix.number_densities()
    rate = 0.0
    for name, n in nd.items():
        rate += n * (sigma_stripping(name, species, T_keV)
                      + sigma_capture(name, species, T_keV))
    return rate


# legacy alias kept for the old test names
def stripping_rate_per_length(mix: GasMix, species: Species,
                               T_keV: float) -> float:
    return beam_loss_rate_per_length(mix, species, T_keV)


def compensation_fraction(t_s: float, tau_s: float, eta_ss: float) -> float:
    """Linear ramp to the steady-state neutralisation factor eta_ss."""
    if tau_s == math.inf or tau_s <= 0:
        return 0.0
    ratio = t_s / tau_s
    return eta_ss * min(1.0, ratio)


# ---------------------------------------------------------------------------
# Beam potential (Chauvin Eq 64)
# ---------------------------------------------------------------------------
def rms_to_edge_radius(sigma_rms: float) -> float:
    """Edge radius `a` of the uniform beam with the same rms size: a = 2 sigma.

    The uniform (KV) beam of edge radius a has <r^2> = a^2/2 in 2-D, i.e. a
    per-plane rms of sigma = a/2. Every formula below that says "edge radius"
    means `a`, and every caller holding a `sigma` must come through here.

    This conversion was previously missing: callers passed sigma straight into
    beam_potential_axis as if it were the edge radius, inflating the log bracket
    by 2*ln(2) = 1.386 -- about +16% on the default case (1203 V vs 1036 V).
    """
    return 2.0 * sigma_rms


def beam_potential_axis(current_A: float, beta: float,
                        r_edge: float, r_pipe: float,
                        charge_sign: float = -1.0) -> float:
    """On-axis electrostatic potential of a uniform round beam in a grounded pipe.

    phi(0) = sign * |I| / (4 pi eps0 beta c) * [1 + 2 ln(r_pipe / r_edge)]

    This is the exact solution for a uniform cylinder of edge radius `r_edge`
    carrying line charge lambda = I/(beta c) inside a grounded pipe of radius
    `r_pipe` (Chauvin, CERN-2013-007 Eq. 64). It is used as the analytic oracle
    the Poisson solver is tested against, so the conventions matter:

    * `r_edge` is the EDGE radius, not the rms size. Use `rms_to_edge_radius`.
    * The result is SIGNED. `charge_sign` is the sign of the beam charge, so an
      H- beam correctly reports a negative on-axis potential. Callers that want a
      magnitude (secondary-particle speeds, display) must take abs() explicitly.
    * When the beam fills or exceeds the pipe the log term goes to zero and the
      result tends smoothly to sign*|I|/(4 pi eps0 beta c). It previously returned
      a silent 0.0 in exactly that case -- the most space-charge-dominated one --
      which read as "no potential at all".
    """
    if r_edge <= 0.0 or r_pipe <= 0.0 or beta <= 0.0:
        return 0.0
    lam_over_4pieps0 = abs(current_A) / (4.0 * math.pi * EPS0 * beta * C_LIGHT)
    # Beam wider than the pipe is unphysical (it would be scraped); clamp the log
    # to its r_edge == r_pipe limit rather than going negative or returning zero.
    log_term = math.log(r_pipe / r_edge) if r_edge < r_pipe else 0.0
    return math.copysign(1.0, charge_sign) * lam_over_4pieps0 * (1.0 + 2.0 * log_term)


# ---------------------------------------------------------------------------
# Envelope equation - RHS for an RK4 integrator (Reiser / Valerio-Lizarraga)
# ---------------------------------------------------------------------------
def envelope_rhs(z: float, state: np.ndarray, *,
                 k0_sq_of_z,
                 emittance_geom: float,
                 perveance: float,
                 fc_of_z: float) -> np.ndarray:
    """Return d/dz of [r, r'] for the RMS envelope equation.

    Parameters
    ----------
    z                 : longitudinal position (m).
    state             : [r (m), r' (dimensionless slope)].
    k0_sq_of_z        : callable returning k0^2(z) in 1/m^2 (focusing).
    emittance_geom    : geometric rms emittance eps_g (m·rad).
    perveance         : generalized perveance K (dimensionless).
    fc_of_z           : local space-charge compensation fraction, in [0, 1].

    Returns
    -------
    [r', r''] where r'' = -k0^2 r + eps_g^2/r^3 + K (1 - fc) / (4 r).

    THE FACTOR OF 4. This equation previously read `+ K (1-fc) / r`, mixing two
    conventions: the emittance term `+eps^2/r^3` is the rms (Sacherer) form, while
    `K/r` is the edge/KV form. They cannot both be right for the same `r`.

    Starting from the KV envelope for the EDGE radius a,

        a'' + k0^2 a - K/a - eps_KV^2/a^3 = 0 ,   eps_KV = 4 eps_rms

    and substituting a = 2 sigma (the rms radius) gives, after dividing by 2,

        sigma'' + k0^2 sigma - K/(4 sigma) - eps_rms^2/sigma^3 = 0

    i.e. r'' = -k0^2 r + eps_rms^2/r^3 + K/(4r). The callers pin `r` as the rms
    radius (simulation.py `r_initial_mm` is documented as sigma_x and is used as
    the Gaussian ensemble's sigma) and `emittance_geom` as the 1x-rms geometric
    emittance, so the missing 1/4 made space-charge defocusing 4x too strong on
    every run. At package defaults that drove the envelope onto the pipe wall,
    where a reflecting clamp then destroyed grid convergence entirely.
    """
    r, rp = state
    r = max(r, 1e-12)
    k2 = k0_sq_of_z(z)
    fc = float(fc_of_z(z)) if callable(fc_of_z) else float(fc_of_z)
    rpp = (-k2 * r
           + emittance_geom ** 2 / (r ** 3)
           + perveance * (1.0 - fc) / (4.0 * r))
    return np.array([rp, rpp])


def rk4_step(f, z, y, dz, **kwargs):
    k1 = f(z, y, **kwargs)
    k2 = f(z + 0.5 * dz, y + 0.5 * dz * k1, **kwargs)
    k3 = f(z + 0.5 * dz, y + 0.5 * dz * k2, **kwargs)
    k4 = f(z + dz, y + dz * k3, **kwargs)
    return y + (dz / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


# ---------------------------------------------------------------------------
# Solenoid field profile
# ---------------------------------------------------------------------------
def solenoid_field(z: np.ndarray, z_center: float, length: float,
                   peak_B: float, edge_soft: float = 0.03) -> np.ndarray:
    """Enge-like hard-edge-with-cosine-fringes axial field of a solenoid."""
    dz = z - z_center
    half = 0.5 * length
    # smooth core plateau with tanh edges
    edge = max(1e-4, edge_soft)
    rise = 0.5 * (np.tanh((dz + half) / edge) - np.tanh((dz - half) / edge))
    return peak_B * rise


def solenoid_focusing(B_z: np.ndarray, species: Species,
                      T_keV: float) -> np.ndarray:
    """k0^2 = (qB / (2 p))^2 for a thin solenoid (KV round beam)."""
    _, _, p = kinematic(species, T_keV)
    return (abs(species.charge) * B_z / (2.0 * p)) ** 2


# ---------------------------------------------------------------------------
# Emittance helpers
# ---------------------------------------------------------------------------
def normalized_to_geometric_emit(eps_n_rms: float, species: Species,
                                 T_keV: float) -> float:
    beta, gamma, _ = kinematic(species, T_keV)
    return eps_n_rms / (beta * gamma)


def rms_from_particles(x: np.ndarray, xp: np.ndarray) -> tuple[float, float, float]:
    """Return (x_rms, xp_rms, eps_geom_rms) from particle coordinates.

    Moments are taken about the ENSEMBLE CENTROID, which is the standard rms
    (Sacherer) definition:

        eps = sqrt( <dx^2><dx'^2> - <dx dx'>^2 ),   dx = x - <x>

    They were previously taken about the origin. That is harmless for a beam
    already centred on the axis, but it silently reports a displaced beam as
    having larger emittance -- a pure centroid offset would masquerade as
    emittance growth. Since this becomes the PIC's primary diagnostic, and since
    the centroid/steerer study is one of the tool's stated uses, the distinction
    has to be right.
    """
    if x.size == 0:
        return 0.0, 0.0, 0.0
    dx = x - np.mean(x)
    dxp = xp - np.mean(xp)
    x2 = float(np.mean(dx * dx))
    xp2 = float(np.mean(dxp * dxp))
    xxp = float(np.mean(dx * dxp))
    eps = math.sqrt(max(0.0, x2 * xp2 - xxp * xxp))
    return math.sqrt(x2), math.sqrt(xp2), eps


# ---------------------------------------------------------------------------
# Non-linear space-charge emittance growth (Reiser, "Theory and Design of
# Charged Particle Beams" §4.5; Wangler §9; Hofmann (1981))
# ---------------------------------------------------------------------------
# Coefficient of the non-linear space-charge emittance-growth closure. Reiser
# quotes ~0.05-0.15 for a bounded free-energy RATIO; using it as a per-metre RATE
# is a different quantity, so the value has to be fitted rather than quoted.
# Disabled (0.0) until Phase 10 fits it against the PIC -- see the docstring.
HOFMANN_ALPHA = 0.0


def hofmann_emittance_growth(eps_n_in: float, K: float, fc: float,
                              dz: float, sigma_x: float) -> float:
    """Increment ΔεN over a slice dz from non-linear SCC.

    Reiser §4.5 / Wangler §9.6.1: a beam transported through space-charge
    fields with a non-uniform density acquires emittance growth driven by
    the free energy U_field stored in the local non-linearity. For a
    near-Gaussian beam the *rate* of normalised-emittance growth is

        d(ε_n²)/ds  ≈  α · K_eff² · σ_x²

    with K_eff = K(1 − f_c) the effective perveance and α a coefficient
    of order 0.05–0.15 depending on density profile (≈ 0.10 for a
    Gaussian → uniform thermalisation). Integrated step-wise:

        ε_n(s+ds)²  =  ε_n(s)²  +  α · K_eff² · σ_x² · ds

    UNITS. `eps_n_in` and the return value are NORMALISED emittance in SI
    (m*rad), not pi*mm*mrad. This is the fix for a defect that made the whole
    term inert: the caller carried `emit_z` in pi*mm*mrad, O(0.2), while the
    increment `alpha (K_eff sigma_x)^2 dz` was built from sigma_x and dz in
    metres, O(1e-12). Adding them suppressed the growth by ~1e12 -- measured
    total growth over a 2 m line was 3.6e-8 relative, i.e. exactly zero. Every
    emittance readout in the GUI was therefore a flat line equal to the user's
    own input, while the tracked particles were growing 12.5x.

    ALPHA IS NOT YET CALIBRATED. With consistent units the nominal alpha = 0.10
    makes the term far too strong (a single 5 mm slice takes 2.5e-7 -> 1.9e-6
    m*rad). Reiser gives a BOUNDED free-energy ratio, not a per-metre rate, so
    the constant cannot simply be read off. `HOFMANN_ALPHA` is therefore set to
    0.0 until it is fitted against the PIC in Phase 10: a term that is honestly
    off is better than one that is silently wrong in either direction.
    Set `HOFMANN_ALPHA` explicitly to experiment.
    """
    if sigma_x <= 0 or HOFMANN_ALPHA <= 0.0:
        return eps_n_in
    K_eff = K * max(0.0, 1.0 - fc)
    d_eps2 = HOFMANN_ALPHA * (K_eff * sigma_x) ** 2 * dz
    return math.sqrt(max(0.0, eps_n_in ** 2 + d_eps2))
