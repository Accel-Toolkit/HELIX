"""External analytic benchmark: KV/rms envelope ODE vs channel tunes.

Independent validation of the depressed channel tune: integrate the rms
envelope equation with EXACTLY the coefficients HELIX's 2-D DC kick
uses (envelope._sc_kick_matrix_2d_dc — uniform elliptic cylinder,
semi-axes a = 2σ), find the matched periodic solution by shooting, and
compare

    μ_analytic = ∫ ε ds / σ_x²      (the PyORBIT matching.py recipe)

against the phase-probe channel tune.  The I=0 limit of the same ODE
must reproduce the bare-matrix σ₀ — validating the ODE construction
itself before it is trusted as a benchmark.
"""
from __future__ import annotations

import math

import numpy as np
from scipy.integrate import trapezoid
import pytest

from linac_gen.analysis.period_detect import detect_periods
from linac_gen.analysis.phase_advance import channel_phase_advance
from linac_gen.core.lattice import Lattice
from linac_gen.core.particle import PROTON
from linac_gen.core.reference import ReferenceParticle
from linac_gen.elements.drift import Drift
from linac_gen.elements.quadrupole import Quadrupole
from linac_gen.tracking.envelope import EnvelopeSolver
from linac_gen.matching.periodic import find_periodic_twiss

# Cell geometry (metres) — mirrors the FODO used across these tests.
_L_D = 0.100
_L_Q = 0.050
_GRAD = 10.0          # T/m
_EPS = 1.0e-6         # 1 mm·mrad geometric, both planes
_CURRENT_MA = 20.0

_E_CHARGE = 1.602176634e-19
_EPS0 = 8.8541878128e-12
_C = 299792458.0


def _fodo(n_cells: int = 3) -> Lattice:
    lat = Lattice()
    for _ in range(n_cells):
        lat.add(Drift(name="D", length=_L_D * 1e3, aperture=10.0))
        lat.add(Quadrupole(name="QF", length=_L_Q * 1e3, gradient=+_GRAD,
                           aperture=10.0))
        lat.add(Drift(name="D", length=_L_D * 1e3, aperture=10.0))
        lat.add(Quadrupole(name="QD", length=_L_Q * 1e3, gradient=-_GRAD,
                           aperture=10.0))
    return lat


def _ref() -> ReferenceParticle:
    return ReferenceParticle(species=PROTON, w_kin=2.5, frequency=162.5)


def _k_of_s(s: float, kq: float) -> tuple:
    """(k_x, k_y) [1/m²] at cell position s [m] — D QF D QD."""
    if s < _L_D:
        return 0.0, 0.0
    if s < _L_D + _L_Q:
        return kq, -kq
    if s < 2 * _L_D + _L_Q:
        return 0.0, 0.0
    return -kq, kq


def _envelope_ode(ref, current_mA):
    """Return f(s, y) for y = (σx, σx', σy, σy') [m, rad] with HELIX's
    exact 2-D DC kick coefficient."""
    kq = _GRAD / ref.brho                      # quad k [1/m²]
    beta, gamma = ref.beta, ref.gamma
    mc2_J = ref.species.mass * 1e6 * _E_CHARGE
    q = abs(ref.species.charge) * _E_CHARGE
    I = abs(current_mA) * 1e-3
    # Single-particle: x'' = pre·k_x·x with pre = q/(β²γ·mc²)·(per ds),
    # k_x = I/(π ε0 v (a+b) a), a = 2σx — rms envelope picks up
    # C/(4(σx+σy)) with C = q·I/(π ε0 β³ γ m c³).
    C = q * I / (math.pi * _EPS0 * beta ** 3 * gamma * mc2_J * _C)

    def f(s, y):
        sx, sxp, sy, syp = y
        kx, ky = _k_of_s(s % (2 * (_L_D + _L_Q)), kq)
        sc = C / (4.0 * (sx + sy)) if (sx > 0 and sy > 0) else 0.0
        return [
            sxp,
            -kx * sx + _EPS ** 2 / sx ** 3 + sc,
            syp,
            -ky * sy + _EPS ** 2 / sy ** 3 + sc,
        ]
    return f


def _matched_ode_solution(ref, current_mA):
    """Shooting: periodic (σx, σx', σy, σy') over one cell."""
    from scipy.integrate import solve_ivp
    from scipy.optimize import root
    L = 2 * (_L_D + _L_Q)
    f = _envelope_ode(ref, current_mA)

    def prop(y0):
        sol = solve_ivp(f, (0.0, L), y0, rtol=1e-11, atol=1e-14,
                        dense_output=True, max_step=_L_Q / 10)
        return sol

    def resid(y0):
        return np.asarray(prop(y0).y[:, -1]) - np.asarray(y0)

    # Seed from the bare matched Twiss (σ = sqrt(εβ), σ' = -αε/σ).
    tw = find_periodic_twiss(_fodo(), ref)
    sx0 = math.sqrt(_EPS * tw["beta_x"])       # β in mm/mrad ≡ m/rad
    sy0 = math.sqrt(_EPS * tw["beta_y"])
    y0 = np.asarray([sx0, -tw["alpha_x"] * _EPS / sx0,
                     sy0, -tw["alpha_y"] * _EPS / sy0])
    sol = root(resid, y0, method="hybr", tol=1e-12)
    # Judge by the residual, not sol.success — hybr reports "no
    # progress" when the seed is already (numerically) the fixed point.
    best = min((y0, np.asarray(sol.x)),
               key=lambda v: np.abs(resid(v)).max())
    rel = np.abs(resid(best)).max() / np.abs(best).max()
    assert rel < 1e-6, f"ODE shooting residual {rel:.2e}"
    return prop(best)


def _mu_from_ode(sol) -> tuple:
    """(μ_x, μ_y) [deg] = ∫ ε ds / σ² over the matched cell."""
    s = np.linspace(sol.t[0], sol.t[-1], 4001)
    y = sol.sol(s)
    mu_x = trapezoid(_EPS / y[0] ** 2, s)
    mu_y = trapezoid(_EPS / y[2] ** 2, s)
    return math.degrees(mu_x), math.degrees(mu_y)


def _helix_channel(current_mA):
    from scipy.optimize import root
    lat = _fodo()
    ref = _ref()
    cell = Lattice()
    for e in lat.elements[:4]:
        cell.add(e)

    def _out(state):
        init = dict(alpha_x=state[0], beta_x=state[1], emit_x=1.0,
                    alpha_y=state[2], beta_y=state[3], emit_y=1.0,
                    alpha_z=0.0, beta_z=1.0, emit_z=0.0, continuous=True)
        res = EnvelopeSolver(cell, ref.copy(), init,
                             current=current_mA).run()
        S = np.asarray(res.sigma_matrix[-1])
        out = []
        for i, j in ((0, 1), (2, 3)):
            eps = np.sqrt(max(S[i, i] * S[j, j] - S[i, j] ** 2, 1e-30))
            out += [-S[i, j] / eps, S[i, i] / eps]
        return np.asarray(out)

    tw = find_periodic_twiss(lat, ref)
    seed = np.asarray([tw["alpha_x"], tw["beta_x"],
                       tw["alpha_y"], tw["beta_y"]])
    sol = root(lambda v: _out(v) - v, seed, method="hybr", tol=1e-12)
    assert sol.success
    init = dict(alpha_x=sol.x[0], beta_x=sol.x[1], emit_x=1.0,
                alpha_y=sol.x[2], beta_y=sol.x[3], emit_y=1.0,
                alpha_z=0.0, beta_z=1.0, emit_z=0.0, continuous=True)
    res = EnvelopeSolver(lat, ref.copy(), init, current=current_mA,
                         phase_probe=True).run()
    period = next(p for p in detect_periods(lat) if p.n_repeats >= 3)
    return channel_phase_advance(res, period)


def test_ode_reproduces_bare_tune_at_zero_current():
    """Validates the in-test ODE itself: its I=0 matched cell must
    reproduce the bare-matrix σ₀ to <0.5 %."""
    ref = _ref()
    sol = _matched_ode_solution(ref, 0.0)
    mu_x, mu_y = _mu_from_ode(sol)
    lat = _fodo()
    period = next(p for p in detect_periods(lat) if p.n_repeats >= 3)
    from linac_gen.analysis.phase_advance import structure_phase_advance
    sigma0 = structure_phase_advance(lat, ref, period)
    assert mu_x == pytest.approx(sigma0["mu_x_deg"], rel=5e-3)
    assert mu_y == pytest.approx(sigma0["mu_y_deg"], rel=5e-3)


def test_channel_tune_matches_kv_ode_benchmark():
    """Depressed channel tune vs the external KV envelope-ODE benchmark
    (identical SC coefficients): agreement to 1.5 %, with a real
    depression at 20 mA."""
    ref = _ref()
    sol = _matched_ode_solution(ref, _CURRENT_MA)
    mu_x_ode, mu_y_ode = _mu_from_ode(sol)

    ch = _helix_channel(_CURRENT_MA)
    assert not ch["coupled_xy"]
    mu_x = float(ch["mu_x_dep_deg"][0])
    mu_y = float(ch["mu_y_dep_deg"][0])
    # Depression is significant (η < 0.9) …
    assert float(ch["eta_x"][0]) < 0.9
    # … and both planes agree with the external benchmark.
    assert mu_x == pytest.approx(mu_x_ode, rel=0.015)
    assert mu_y == pytest.approx(mu_y_ode, rel=0.015)
