"""Sacherer / Kapchinsky-Vladimirsky RMS envelope ODE solver.

Continuous-beam envelope equations integrated with scipy ``solve_ivp``,
following the textbook formulation (Reiser §5; Sacherer 1971; KENV's
``solver.py``).  For an unbunched (DC) beam:

    σ_x'' + κ_x(s) σ_x − ε_x²/σ_x³ − K/[2(σ_x + σ_y)] = 0
    σ_y'' + κ_y(s) σ_y − ε_y²/σ_y³ − K/[2(σ_x + σ_y)] = 0

in rms quantities (σ rms size, ε rms emittance; the equivalent
edge-radius form X'' + κX − ε_X²/X³ − 2K/(X + Y) = 0 uses X = 2σ and
total emittance ε_X = 4ε),
with generalised perveance ``K = q·I / (2π ε₀ m₀ c³ (βγ)³)`` and a
focusing function κ(s) read off from the lattice element by element.

Scope of this implementation
----------------------------
Designed as a textbook benchmark for the existing matrix-plus-discrete-
SC EnvelopeSolver, not a replacement for general-purpose tracking.

Supported elements:

* **Drift** — κ = 0.
* **Quadrupole** (hard-edge) — κ_x = ±k₁, κ_y = ∓k₁ from
  ``element.gradient`` and the reference momentum.
* **Solenoid** (hard-edge SOLENOID card) — constant Larmor focusing
  κ = (B/(2Bρ))² in both planes from ``element.field``.
* **FieldMap / FieldMap3D — solenoid only** (1-D B_z(s) on-axis): the
  Larmor focusing κ(s) = (q B_z / (2 p))² is extracted from the on-axis
  z-component of the static-B channel.
* **Markers / Apertures / Steerers / SpaceChargeComp / RF gaps** — pass
  through with κ = 0 (no transverse focus contribution).  RF gaps are
  treated as drifts (DC-beam approximation: no synchronous-phase focus).

The reference particle's β, γ are held constant per call (DC beam
through low-energy structures); for accelerated beams the existing
matrix tracker remains the right tool.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List, Optional

import numpy as np
from scipy.integrate import solve_ivp

from linac_gen.core.constants import EPSILON_0, PI, E_CHARGE, C_LIGHT
from linac_gen.core.reference import ReferenceParticle
from linac_gen.elements.drift import Drift
from linac_gen.elements.quadrupole import Quadrupole
from linac_gen.elements.solenoid import Solenoid
from linac_gen.elements.field_map import FieldMap as FieldMap1D
from linac_gen.elements.field_map_3d import FieldMap3D
from linac_gen.elements.space_charge_comp import SpaceChargeComp
from linac_gen.io.tracewin_geom import Channel


# ---------------------------------------------------------------------------
# Data container
# ---------------------------------------------------------------------------
@dataclass
class SachererResults:
    """Output of the Sacherer ODE solver."""
    s: List[float] = field(default_factory=list)            # mm
    sigma_x: List[float] = field(default_factory=list)      # mm
    sigma_y: List[float] = field(default_factory=list)
    sigma_xp: List[float] = field(default_factory=list)     # mrad
    sigma_yp: List[float] = field(default_factory=list)
    element_names: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _perveance(current_mA: float, ref: ReferenceParticle) -> float:
    """Generalised perveance K = qI / (2π ε₀ m c³ (βγ)³).

    K is dimensionless when σ is in metres and σ' in radians; we then
    rescale below to the mm/mrad system used internally.
    """
    if current_mA == 0:
        return 0.0
    q = abs(ref.species.charge) * E_CHARGE  # |q| in Coulombs
    I_A = abs(current_mA) * 1e-3
    m_kg = ref.species.mass * 1e6 * E_CHARGE / C_LIGHT ** 2
    bg = ref.beta * ref.gamma
    return q * I_A / (
        2.0 * PI * EPSILON_0 * m_kg * C_LIGHT ** 3 * bg ** 3
    )


def _solenoid_kappa_profile(elem) -> Optional[Callable[[float], float]]:
    """Return a callable κ(z_local_mm) for a 1-D / 3-D solenoid map.

    κ(z) = (q B_z(z) / (2 p))² in 1/m².  Returns None if the element has
    no static-B channel we can read on-axis.  ``z_local_mm`` is measured
    from the element's start.
    """
    fdata = getattr(elem, "field_data", None)
    if fdata is None or Channel.STAT_B not in fdata.channels:
        return None
    ch = fdata.channels[Channel.STAT_B]
    z_axis = np.asarray(ch.z, dtype=np.float64)  # mm, element-local
    Fz = np.asarray(ch.Fz, dtype=np.float64)
    # Pick on-axis B_z.  For 3-D maps this is Fz[ix0, iy0, :]; for 1-D
    # there is only a z axis.
    if Fz.ndim == 3:
        ix = Fz.shape[0] // 2
        iy = Fz.shape[1] // 2
        bz_on_axis = Fz[ix, iy, :]
    elif Fz.ndim == 1:
        bz_on_axis = Fz
    else:
        return None
    kb = float(getattr(elem, "kb", 1.0)) or 1.0
    bz_on_axis = bz_on_axis * kb  # apply scaling

    z0 = float(z_axis[0])

    def kappa_of_z(z_local_mm: float, _z=z_axis, _bz=bz_on_axis,
                   _z0=z0) -> float:
        # Numpy interp clamps at the endpoints — outside the element we
        # would not call it, so the clamp is just defensive.
        return float(np.interp(z_local_mm + _z0, _z, _bz))

    return kappa_of_z


# ---------------------------------------------------------------------------
# Main solver
# ---------------------------------------------------------------------------
class SachererSolver:
    """Integrate the KV/Sacherer envelope ODE element by element.

    Parameters
    ----------
    lattice : Lattice
        Linac_Gen lattice.
    ref : ReferenceParticle
        Synchronous particle (β, γ held constant for DC operation).
    init : dict
        Initial Twiss parameters with keys ``alpha_x, beta_x, emit_x``
        and the y-counterparts — the SAME ``initial`` dialect
        ``EnvelopeSolver`` consumes (``cli/common.py:_envelope_initial``):
        ``emit_*`` GEOMETRIC in **mm·mrad**, ``beta_*`` in m (numerically
        ≡ mm/mrad).  The two solvers are dispatch-interchangeable behind
        ``run_envelope_sim``, so they must share one dict contract.
        (Before 2026-07-25 this boundary expected m·rad, so every
        production caller was 1e6 off in ε and σ came out 1000× large.)
    current : float
        Beam current in mA.
    n_record_per_element : int
        Number of σ samples written between element entry and exit (≥ 2).
    """

    def __init__(self, lattice, ref: ReferenceParticle, init: dict,
                 current: float = 0.0, n_record_per_element: int = 21):
        self.lattice = lattice
        self.ref = ref
        self.init = init
        self.current = float(current)
        self.n_record = max(2, int(n_record_per_element))
        # Base perveance at full current; the effective K used inside each
        # element is scaled by (1 − SCC.factor) to mirror the Tracker's
        # SPACE_CHARGE_COMP convention.
        self._K_full = _perveance(self.current, ref)
        self._sc_factor = 1.0  # current effective SC fraction (1 − comp)

        # Convert input twiss → σ at element 0 entrance.  The dict
        # carries ε in mm·mrad (ecosystem dialect); the ODE runs in
        # m/rad, so scale by 1e-6 here — and ONLY here, so the ε²/σ³
        # pressure term and the σ seed stay consistent.
        ax = float(init["alpha_x"]); bx = float(init["beta_x"])
        ay = float(init["alpha_y"]); by_ = float(init["beta_y"])
        ex = float(init["emit_x"]) * 1e-6      # mm·mrad → m·rad
        ey = float(init["emit_y"]) * 1e-6
        # σ_x in metres, σ_x' in radians.  Sacherer ODE uses m/rad
        # internally; we record in mm/mrad for parity with EnvelopeSolver.
        self.sx0 = float(np.sqrt(bx * ex))
        self.sxp0 = -ax * np.sqrt(ex / bx) if bx > 0 else 0.0
        self.sy0 = float(np.sqrt(by_ * ey))
        self.syp0 = -ay * np.sqrt(ey / by_) if by_ > 0 else 0.0
        self.ex_g = float(ex)
        self.ey_g = float(ey)

    # ------------------------------------------------------------------
    def _kappa_of_element(self, elem):
        """Return (κ_x(z), κ_y(z), L_m) for the given element.

        z is measured from element entrance, in metres.  Functions return
        focusing strength in 1/m².
        """
        L_mm = float(getattr(elem, "length", 0) or 0)
        L_m = L_mm * 1e-3

        if isinstance(elem, Drift) or L_mm <= 0:
            return (lambda z: 0.0), (lambda z: 0.0), L_m

        if isinstance(elem, Quadrupole):
            # Hard-edge: use element.gradient (T/m) and reference Bρ.
            G = float(getattr(elem, "gradient", 0.0))
            p_SI = self.ref.beta * self.ref.gamma * \
                self.ref.species.mass * 1e6 * E_CHARGE / C_LIGHT
            Brho = p_SI / E_CHARGE  # T·m for unit charge
            k1 = G / Brho  # 1/m²
            return (lambda z, k=k1: +k), (lambda z, k=k1: -k), L_m

        if isinstance(elem, Solenoid):
            # Hard-edge SOLENOID card: constant Larmor focusing
            # κ = (B / (2 Bρ))² in BOTH planes.  Was silently treated
            # as a drift before 2026-07-25 (review claim 1) — for a
            # solver whose stated niche is DC LEBT benchmarking, the
            # solenoid is the element that matters most.
            B = float(getattr(elem, "effective_field", None)
                      or getattr(elem, "field", 0.0) or 0.0)
            p_SI = self.ref.beta * self.ref.gamma * \
                self.ref.species.mass * 1e6 * E_CHARGE / C_LIGHT
            Brho = p_SI / E_CHARGE  # T·m
            k_sol = (B / (2.0 * Brho)) ** 2
            return (lambda z, k=k_sol: k), (lambda z, k=k_sol: k), L_m

        if isinstance(elem, (FieldMap1D, FieldMap3D)):
            bz_fn = _solenoid_kappa_profile(elem)
            if bz_fn is not None:
                # Larmor focusing: κ = (q Bz / (2 p))² ≡ (Bz / (2 Bρ))²
                p_SI = self.ref.beta * self.ref.gamma * \
                    self.ref.species.mass * 1e6 * E_CHARGE / C_LIGHT
                Brho = p_SI / E_CHARGE  # T·m
                def kappa(z_m, _bz=bz_fn, _Brho=Brho):
                    bz = _bz(z_m * 1e3)  # mm
                    return (bz / (2.0 * _Brho)) ** 2
                return kappa, kappa, L_m
            # No static-B channel — treat as drift (RF cavities).
            return (lambda z: 0.0), (lambda z: 0.0), L_m

        # Anything else (markers, apertures, steerers, SpaceChargeComp):
        # zero focusing, just consume length (which is 0 for thin
        # markers / kicks anyway).
        return (lambda z: 0.0), (lambda z: 0.0), L_m

    # ------------------------------------------------------------------
    def _rhs(self, z, X, kx_fn, ky_fn):
        """Right-hand side of the coupled KV envelope ODE."""
        sx, sxp, sy, syp = X
        # Avoid σ → 0 singularities.
        sx_safe = sx if abs(sx) > 1e-12 else 1e-12
        sy_safe = sy if abs(sy) > 1e-12 else 1e-12
        kx = float(kx_fn(z))
        ky = float(ky_fn(z))
        K = self._K_full * self._sc_factor
        # σ in m, σ' in rad, K dimensionless, ε² in (m·rad)² → 1/m³
        # rms form: K/[2(σx+σy)] (edge-radius 2K/(X+Y) with X=2σ, ε_X=4ε)
        sc_term = 0.5 * K / (sx_safe + sy_safe) if K else 0.0
        d_sx = sxp
        d_sxp = -kx * sx + self.ex_g ** 2 / sx_safe ** 3 + sc_term
        d_sy = syp
        d_syp = -ky * sy + self.ey_g ** 2 / sy_safe ** 3 + sc_term
        return [d_sx, d_sxp, d_sy, d_syp]

    # ------------------------------------------------------------------
    def run(self) -> SachererResults:
        res = SachererResults()

        s_global_mm = 0.0  # accumulated path length in mm
        sx, sxp, sy, syp = self.sx0, self.sxp0, self.sy0, self.syp0

        # Record initial state.
        res.s.append(s_global_mm)
        res.sigma_x.append(sx * 1e3)        # mm
        res.sigma_y.append(sy * 1e3)
        res.sigma_xp.append(sxp * 1e3)      # mrad
        res.sigma_yp.append(syp * 1e3)
        res.element_names.append("ENTRANCE")

        for elem in self.lattice.elements:
            # SPACE_CHARGE_COMP markers update the effective K factor.
            if isinstance(elem, SpaceChargeComp):
                self._sc_factor = max(0.0, 1.0 - float(elem.factor))
                # Still record so element_names line up with envelope solver.
                res.s.append(s_global_mm)
                res.sigma_x.append(sx * 1e3)
                res.sigma_y.append(sy * 1e3)
                res.sigma_xp.append(sxp * 1e3)
                res.sigma_yp.append(syp * 1e3)
                res.element_names.append(elem.name)
                continue

            kx_fn, ky_fn, L_m = self._kappa_of_element(elem)
            if L_m <= 0:
                # Thin element — record current state and continue.
                res.s.append(s_global_mm)
                res.sigma_x.append(sx * 1e3)
                res.sigma_y.append(sy * 1e3)
                res.sigma_xp.append(sxp * 1e3)
                res.sigma_yp.append(syp * 1e3)
                res.element_names.append(elem.name)
                continue

            t_eval = np.linspace(0.0, L_m, self.n_record)
            sol = solve_ivp(
                self._rhs,
                (0.0, L_m),
                [sx, sxp, sy, syp],
                args=(kx_fn, ky_fn),
                t_eval=t_eval,
                method="RK45",
                rtol=1e-8,
                atol=1e-12,
                max_step=L_m / 50.0,
            )
            if not sol.success:
                raise RuntimeError(
                    f"Sacherer ODE failed in {elem.name}: {sol.message}"
                )
            # Append all but the first point (which equals the previous
            # element's exit, already recorded).
            for i, t in enumerate(sol.t[1:], start=1):
                res.s.append(s_global_mm + t * 1e3)
                res.sigma_x.append(sol.y[0, i] * 1e3)
                res.sigma_y.append(sol.y[2, i] * 1e3)
                res.sigma_xp.append(sol.y[1, i] * 1e3)
                res.sigma_yp.append(sol.y[3, i] * 1e3)
                res.element_names.append(elem.name)
            sx, sxp, sy, syp = sol.y[:, -1]
            s_global_mm += L_m * 1e3

        return res
