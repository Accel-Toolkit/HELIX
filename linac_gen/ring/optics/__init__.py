"""Ring optics computed with the ring kernels themselves.

* :func:`closed_orbit` — Newton's method on the one-turn map: 4-D (momentum
  deviation fixed, RF off) or 6-D (RF on, the whole phase space periodic);
* :func:`one_turn_matrix` — the linear one-turn map at an orbit, by central
  differences;
* :func:`twiss` — tunes with their integer part, chromaticities, momentum
  compaction, slip factor and transition gamma, and the periodic Twiss
  functions, dispersion and closed orbit at every element's exit; with
  ``six_d=True`` also the synchrotron tune from the 6-D map;
* :func:`rf_bucket` — the stationary RF bucket: small-amplitude synchrotron
  tune, bucket half-height, separatrix.

Nothing here is a separate model of the ring: every number comes from
tracking particles through the same nodes a ring run uses, so the optics are
those of the tracking (the chromaticity includes every nonlinear map, the
pole faces of the chosen edge model, etc.).  Units: metres, radians, the ring
coordinates of :mod:`linac_gen.ring.coords`; chromaticity and dispersion per
unit delta (multiply a MAD-X per-PT value by beta0 to compare).
"""
from __future__ import annotations

import math
import threading
import time
from dataclasses import dataclass, field

import numpy as np

from linac_gen.core.cancelled import OperationCancelled
from linac_gen.ring.optics.normal_form import UnstableError, normal_form, twiss_from_W

C_LIGHT = 299792458.0

#: central-difference steps of the one-turn matrix (x, px, y, py, zeta, delta)
R_STEPS = np.array([1e-7, 1e-8, 1e-7, 1e-8, 1e-6, 1e-7])
#: steps of the Newton Jacobian
NEWTON_STEPS = np.array([1e-8, 1e-9, 1e-8, 1e-9, 1e-7, 1e-8])


class ClosedOrbitError(RuntimeError):
    """Newton's method did not find the closed orbit."""


# the stop hook of the optics computation running in this thread (see
# :func:`twiss`): polled before every pass of particles around the ring
_STOP = threading.local()


def _energy(delta, beta0):
    """(ptau, rpp, rvv) — the same expressions as the kernels'
    :func:`~linac_gen.ring.kernels.cartesian.energy_from_delta`."""
    d = np.asarray(delta, dtype=float)
    inv_b0 = 1.0 / beta0
    opd = 1.0 + d
    ptau = d * (2.0 + d) / (np.sqrt(opd * opd + inv_b0 * inv_b0 - 1.0) + inv_b0)
    return ptau, 1.0 / opd, opd / (1.0 + beta0 * ptau)


def track_once(rl, X, *, record: bool = False):
    """Track the particles X (6, n) once around ``rl`` with no apertures
    (an optics probe) → X after one turn, and with ``record`` the
    coordinates after every node (n_nodes, 6, n)."""
    from linac_gen.ring.kernels.cpu import backend
    stop = getattr(_STOP, "fn", None)
    if stop is not None and stop():
        raise OperationCancelled("ring optics: stopped")
    be = backend()
    X = np.array(X, dtype=np.float64, copy=True, order="C")
    ptau, rpp, rvv = _energy(X[5], float(rl.ref_beta0[0]))
    ptau = np.ascontiguousarray(ptau)
    rpp = np.ascontiguousarray(rpp)
    rvv = np.ascontiguousarray(rvv)
    n_nodes = rl.n_nodes
    out = np.empty((n_nodes if record else 0, 6, X.shape[1]))
    be.track_probe(X, ptau, rpp, rvv, *rl.node_table(), 0, n_nodes, out, record)
    return (X, out) if record else X


def _jacobian(rl, z, dims, steps):
    """(T(z), dT/dz) with central differences over the first ``dims`` coordinates."""
    n = 1 + 2 * dims
    X = np.repeat(np.asarray(z, dtype=float)[:, None], n, axis=1)
    for k in range(dims):
        X[k, 1 + 2 * k] += steps[k]
        X[k, 2 + 2 * k] -= steps[k]
    Y = track_once(rl, X)
    J = np.empty((6, dims))
    for k in range(dims):
        J[:, k] = (Y[:, 1 + 2 * k] - Y[:, 2 + 2 * k]) / (2.0 * steps[k])
    return Y[:, 0], J


def closed_orbit(rl, *, delta: float = 0.0, six_d: bool = False, guess=None,
                 tol: float = 1e-14, max_iter: int = 40) -> np.ndarray:
    """The closed orbit (6,) at the start of ``rl``.

    4-D (default): delta is fixed at ``delta`` and the transverse coordinates
    are periodic; zeta is left free (``rl`` should have its RF off, see
    :meth:`RingLattice.with_rf`).  6-D: all six coordinates periodic.
    """
    dims = 6 if six_d else 4
    z = np.zeros(6) if guess is None else np.array(guess, dtype=float)
    if not six_d:
        z[4] = 0.0
        z[5] = float(delta)
    best = None
    for _it in range(max_iter):
        Tz, J = _jacobian(rl, z, dims, NEWTON_STEPS)
        if not np.all(np.isfinite(Tz)) or not np.all(np.isfinite(J)):
            raise ClosedOrbitError("a particle near the closed orbit was not trackable "
                                   "(beyond 90 degrees, or outside a map's domain)")
        F = Tz[:dims] - z[:dims]
        err = float(np.abs(F).max())
        if best is not None and err >= best and err <= 1e3 * tol:
            break                         # (rounding floor reached)
        best = err if best is None else min(best, err)
        if err <= tol:
            break
        dz = np.linalg.solve(J[:dims, :dims] - np.eye(dims), -F)
        z[:dims] += dz
    else:
        if best is None or best > 1e-9:
            raise ClosedOrbitError(f"no closed orbit after {max_iter} Newton steps "
                                   f"(residual {best:.3g})")
    if best is not None and best > 1e-9:
        raise ClosedOrbitError(f"no closed orbit (residual {best:.3g})")
    return z


def one_turn_matrix(rl, orbit, *, steps=R_STEPS) -> np.ndarray:
    """The 6×6 linear one-turn map at ``orbit`` (central differences)."""
    _Tz, J = _jacobian(rl, orbit, 6, steps)
    return J


@dataclass
class RingOptics:
    """Optics of a ring at its start (s = 0) and along it.

    ``table`` holds per-element arrays at each element's exit (``s`` [m]):
    ``betx, alfx, bety, alfy`` [m, 1], ``mux, muy`` [tune units, from the
    start], ``dx, dpx, dy, dpy`` (per delta), closed orbit ``x, px, y, py``.
    """
    mode: str
    length: float
    gamma0: float
    beta0: float
    qx: float
    qy: float
    dqx: float | None
    dqy: float | None
    alpha_c: float
    eta: float
    gamma_t: float | None
    qs: float | None
    closed_orbit: np.ndarray
    R: np.ndarray
    W: np.ndarray
    start: dict
    table: dict
    options: dict
    warnings: list[str] = field(default_factory=list)
    wall_s: float = 0.0

    def summary(self) -> dict:
        out = {"length_m": self.length, "qx": self.qx, "qy": self.qy, "dqx": self.dqx,
               "dqy": self.dqy, "alpha_c": self.alpha_c, "eta": self.eta,
               "gamma_t": self.gamma_t, "qs": self.qs, "gamma0": self.gamma0,
               "beta0": self.beta0, "mode": self.mode}
        out.update({f"{k}_start": float(v) for k, v in self.start.items()})
        out["closed_orbit_start"] = [float(v) for v in self.closed_orbit]
        return out


def _tune_fraction(mu):
    return (np.asarray(mu) / (2.0 * math.pi)) % 1.0


#: [rad] a step of the unwrapped phase below minus this is an element
#: advancing the phase by more than pi (rounding of a zero step is far smaller)
PHASE_STEP_TOL = 1e-9


def _phase_from_start(phase, forward):
    """The phase advance from the start at each element exit, from the mode
    phases in (-pi, pi] there.  ``forward`` (one more entry, the start
    first): whether the mode turns forward in its own plane there — the
    plane's part of the mode's Lagrange bracket is positive; without coupling
    it is 1, strong coupling can make it negative and the phase then
    decreases.  Unwrapping takes each element's step in [-pi, pi); a
    negative step between two forward points is an element advancing the
    phase by more than pi (a long bend, a matrix): 2 pi is added from there
    on."""
    ph = np.unwrap(np.concatenate([[0.0], phase]))
    wrap = (np.diff(ph) < -PHASE_STEP_TOL) & forward[:-1] & forward[1:]
    ph = ph[1:]
    if wrap.any():
        ph = ph + 2.0 * math.pi * np.cumsum(wrap)
    return ph


def _forward(W0, Wn, p):
    """Whether mode p turns forward in its plane p at the start (W0) and at
    each element exit (Wn): the plane's part of the mode's Lagrange bracket,
    W[2p,2p] W[2p+1,2p+1] - W[2p,2p+1] W[2p+1,2p], is positive."""
    W = np.concatenate([W0[None], Wn])
    i, j = 2 * p, 2 * p + 1
    return W[:, i, i] * W[:, j, j] - W[:, i, j] * W[:, j, i] > 0.0


def twiss(rl, *, delta0: float = 0.0, six_d: bool = False, chromaticity: bool = True,
          d_delta: float = 1e-6, d_delta_chrom: float = 1e-5, eps: float = 1e-6,
          should_stop=None) -> RingOptics:
    """Periodic optics of the compiled ring ``rl`` (4-D with the RF off;
    ``six_d`` adds the 6-D closed orbit and synchrotron tune with the RF on).

    ``should_stop`` (a callable) is polled before every pass of particles
    around the ring; when it returns True the computation stops with
    :class:`~linac_gen.core.cancelled.OperationCancelled`."""
    prev = getattr(_STOP, "fn", None)
    _STOP.fn = should_stop
    try:
        return _twiss(rl, delta0=delta0, six_d=six_d, chromaticity=chromaticity,
                      d_delta=d_delta, d_delta_chrom=d_delta_chrom, eps=eps)
    finally:
        _STOP.fn = prev


def _twiss(rl, *, delta0, six_d, chromaticity, d_delta, d_delta_chrom, eps) -> RingOptics:
    t0 = time.perf_counter()
    rl4 = rl.with_rf(False)
    ref = rl.reference
    co = closed_orbit(rl4, delta=delta0)
    R = one_turn_matrix(rl4, co)
    try:
        W4, mu = normal_form(R[:4, :4])
    except UnstableError as exc:
        raise UnstableError(f"the ring is unstable at {ref.kinetic0:.6g} MeV: {exc}") from None
    co_p = closed_orbit(rl4, delta=delta0 + d_delta, guess=co)
    co_m = closed_orbit(rl4, delta=delta0 - d_delta, guess=co)

    # probes: the closed orbit, ±eps along each normal-mode vector, and the
    # off-momentum closed orbits (dispersion)
    cols = [co]
    for k in range(4):
        w = np.zeros(6)
        w[:4] = W4[:, k]
        cols += [co + eps * w, co - eps * w]
    cols += [co_p, co_m]
    X = np.array(cols).T
    Y, rec = track_once(rl4, X, record=True)
    # everything is read at element exits: inside an element the frame may be
    # rotated (a vertical bend, a tilted magnet), where x and y are not the
    # ring's.  An element with no node of its own reports the node before it.
    last = np.asarray(rl.element_last_node).copy()
    for i in range(last.size):
        if last[i] < 0:
            last[i] = last[i - 1] if i > 0 else -1
    at0 = last < 0
    sel = np.where(at0, 0, last)
    Wn = np.empty((sel.size, 4, 4))
    for k in range(4):
        Wn[:, :, k] = (rec[sel, :4, 1 + 2 * k] - rec[sel, :4, 2 + 2 * k]) / (2.0 * eps)
    Wn[at0] = W4
    tw = twiss_from_W(Wn)
    tw0 = twiss_from_W(W4)
    # phase advance from the start, unwrapped from element exit to element exit
    phx = _phase_from_start(tw["phase_x"], _forward(W4, Wn, 0))
    phy = _phase_from_start(tw["phase_y"], _forward(W4, Wn, 1))
    disp = (rec[sel, :4, 9] - rec[sel, :4, 10]) / (2.0 * d_delta)
    D0 = (co_p[:4] - co_m[:4]) / (2.0 * d_delta)
    disp[at0] = D0
    orbit = rec[sel, :4, 0]
    orbit[at0] = co[:4]
    if last[-1] != rl.n_nodes - 1:
        raise RuntimeError("internal: the last element does not end the turn")

    frac = _tune_fraction(mu)
    nx = round((phx[-1] - mu[0]) / (2.0 * math.pi))
    ny = round((phy[-1] - mu[1]) / (2.0 * math.pi))
    qx = nx + frac[0]
    qy = ny + frac[1]

    # momentum compaction from the path of the off-momentum closed orbits
    dqx = dqy = None
    cp = closed_orbit(rl4, delta=delta0 + d_delta_chrom, guess=co_p)
    cm = closed_orbit(rl4, delta=delta0 - d_delta_chrom, guess=co_m)
    Tp = track_once(rl4, cp[:, None])[:, 0]
    Tm = track_once(rl4, cm[:, None])[:, 0]
    L = rl.length
    eta = -((Tp[4] - cp[4]) - (Tm[4] - cm[4])) / (2.0 * d_delta_chrom * L)
    alpha_c = eta + 1.0 / ref.gamma0 ** 2
    gamma_t = 1.0 / math.sqrt(alpha_c) if alpha_c > 0.0 else None
    if chromaticity:
        mup = normal_form(one_turn_matrix(rl4, cp)[:4, :4])[1]
        mum = normal_form(one_turn_matrix(rl4, cm)[:4, :4])[1]
        d = (mup - mum) / (2.0 * math.pi)
        d -= np.round(d)
        dqx, dqy = (float(v) for v in d / (2.0 * d_delta_chrom))

    qs = None
    R_out, W_out, co_out, mode = R, W4, co, "4d"
    if six_d:
        if not rl.rf_on:
            raise ValueError("6-D optics need RF: the ring has no cavity with a voltage")
        _check_stationary(rl)
        co6 = closed_orbit(rl, six_d=True, guess=co)
        R6 = one_turn_matrix(rl, co6)
        W6, mu6 = normal_form(R6)
        # the longitudinal rotation reverses above transition (eta > 0),
        # where the eigenvalue's angle is 1 - Qs: the synchrotron tune is the
        # magnitude of the signed fraction (Xsuite's |muzeta|)
        frac_s = float(_tune_fraction(mu6[2]))
        qs = min(frac_s, 1.0 - frac_s)
        R_out, W_out, co_out, mode = R6, W6, co6, "6d"

    table = {
        "name": list(rl.element_names),
        "type": list(rl.element_types),
        "s": np.asarray(rl.element_s) + np.asarray(rl.element_length),
        "betx": tw["beta_x"], "alfx": tw["alpha_x"], "bety": tw["beta_y"], "alfy": tw["alpha_y"],
        "mux": phx / (2.0 * math.pi), "muy": phy / (2.0 * math.pi),
        "dx": disp[:, 0], "dpx": disp[:, 1], "dy": disp[:, 2], "dpy": disp[:, 3],
        "x": orbit[:, 0], "px": orbit[:, 1], "y": orbit[:, 2], "py": orbit[:, 3],
    }
    start = {"betx": tw0["beta_x"], "alfx": tw0["alpha_x"], "bety": tw0["beta_y"],
             "alfy": tw0["alpha_y"], "dx": D0[0], "dpx": D0[1], "dy": D0[2], "dpy": D0[3]}
    return RingOptics(
        mode=mode, length=L, gamma0=ref.gamma0, beta0=ref.beta0, qx=float(qx), qy=float(qy),
        dqx=dqx, dqy=dqy, alpha_c=float(alpha_c), eta=float(eta), gamma_t=gamma_t, qs=qs,
        closed_orbit=co_out, R=R_out, W=W_out, start={k: float(v) for k, v in start.items()},
        table=table, options=dict(rl.options), warnings=list(rl.warnings),
        wall_s=time.perf_counter() - t0)


def _check_stationary(rl):
    gain = sum(rl.cavity_voltage(c) * math.sin(math.radians(c["lag_deg"])) for c in rl.cavities)
    vtot = sum(abs(rl.cavity_voltage(c)) for c in rl.cavities)
    if abs(gain) > 1e-12 * max(vtot, 1e-300):
        raise ValueError(
            f"the cavities give the reference particle {gain:.6g} MeV per turn but the "
            "reference energy is fixed (no ramp in this milestone): there is no 6-D closed "
            "orbit — use lag 0 or 180 deg (a stationary bucket)")


def rf_bucket(rl, optics: RingOptics) -> dict:
    """The stationary RF bucket of ``rl`` from its cavities (summed as one
    harmonic voltage) and the 4-D slip factor: harmonic, voltage [MV], the
    stable phase, the small-amplitude synchrotron tune, the bucket's
    half-height in delta and its length in zeta, and ``separatrix(zeta)`` →
    the half-height at zeta (NaN outside the bucket)."""
    if not rl.rf_on:
        raise ValueError("the ring has no cavity with a voltage")
    _check_stationary(rl)
    ref = rl.reference
    T0 = rl.revolution_time
    hs = {round(c["frequency_hz"] * T0, 9) for c in rl.cavities if rl.cavity_voltage(c)}
    if len(hs) != 1:
        raise ValueError(f"the cavities are not at one harmonic of the revolution "
                         f"frequency ({sorted(hs)})")
    h = hs.pop()
    phasor = sum(rl.cavity_voltage(c) * complex(math.cos(math.radians(c["lag_deg"])),
                                            math.sin(math.radians(c["lag_deg"])))
                 for c in rl.cavities)
    V = abs(phasor)
    lag = math.degrees(math.atan2(phasor.imag, phasor.real))
    eta = optics.eta
    E0 = ref.energy0
    b2E = ref.beta0 ** 2 * E0
    cos_l = math.cos(math.radians(lag))
    stable = eta * cos_l < 0.0
    qs = math.sqrt(h * abs(eta) * V * abs(cos_l) / (2.0 * math.pi * b2E)) if stable else float("nan")
    k = 2.0 * math.pi * h / rl.length                       # RF wave number in zeta
    dmax = math.sqrt(2.0 * V / (math.pi * h * abs(eta) * b2E)) if stable else float("nan")
    zeta_s = 0.0                     # the stable point: lag 0 (eta < 0) or 180 (eta > 0)

    def separatrix(zeta):
        z = np.asarray(zeta, dtype=float)
        u = np.cos(k * (z - zeta_s) / 2.0)
        out = dmax * np.abs(u)
        return np.where(np.abs(k * (z - zeta_s)) <= math.pi, out, np.nan)

    return {"harmonic": h, "voltage_mv": V, "lag_deg": lag, "stable": stable, "qs": qs,
            "delta_max": dmax, "bucket_length_m": rl.length / h, "separatrix": separatrix}


def accelerating_bucket(voltage_mv: float, phi_s_deg: float, harmonic: float, eta: float,
                        beta0: float, energy_mev: float, length_m: float) -> dict:
    """The RF bucket of a turn of a ramp (one harmonic voltage ``voltage_mv``
    at the synchronous phase ``phi_s_deg``, sine convention: the reference
    gains V sin(phi_s); slip factor ``eta``; reference velocity ``beta0`` and
    total energy ``energy_mev``; circumference ``length_m``).

    From the synchrotron motion's Hamiltonian in (phi, delta), turn by turn,
    H = pi h eta delta^2 + (V/(beta0^2 E)) (cos phi + phi sin phi_s), with
    phi = phi_s - (2 pi h / C) zeta: the separatrix through the unstable fixed
    point pi - phi_s, delta^2 = (V/(pi h eta beta0^2 E)) [cos phi_u + phi_u
    sin phi_s - cos phi - phi sin phi_s] (phi_s = 0: the stationary bucket).
    Returns the phase, whether it is stable (eta cos phi_s < 0), the
    small-amplitude synchrotron tune, the half-height at the synchronous
    particle, the bucket's ends and length in zeta, its area in zeta-delta [m]
    and ``separatrix(zeta)`` → the half-height (NaN outside)."""
    phs = math.radians(float(phi_s_deg))
    h = float(harmonic)
    b2E = float(beta0) ** 2 * float(energy_mev)
    k = 2.0 * math.pi * h / float(length_m)
    cos_s, sin_s = math.cos(phs), math.sin(phs)
    stable = (eta * cos_s < 0.0) and float(voltage_mv) > 0.0
    out = {"voltage_mv": float(voltage_mv), "phi_s_deg": float(phi_s_deg), "harmonic": h,
           "eta": float(eta), "stable": bool(stable)}
    if not stable or abs(cos_s) < 1e-12:
        out.update(qs=float("nan"), delta_max=0.0, zeta_min=0.0, zeta_max=0.0,
                   bucket_length_m=0.0, area_m=0.0,
                   separatrix=lambda zeta: np.full(np.shape(zeta), np.nan))
        return out
    scale = float(voltage_mv) / (math.pi * h * eta * b2E)

    def f(ph):
        return np.cos(ph) + ph * sin_s

    # the unstable fixed points next to phi_s (pi - phi_s + 2 pi n, one on
    # each side): the separatrix runs through the one whose level closes the
    # bucket — the higher of f below transition (phi_s a maximum of f), the
    # lower above it (a minimum)
    base = math.pi - phs
    ph_r = base + 2.0 * math.pi * (math.floor((phs - base) / (2.0 * math.pi)) + 1)
    ph_l = ph_r - 2.0 * math.pi
    f_l, f_r = float(f(ph_l)), float(f(ph_r))
    if (eta < 0.0) == (f_l > f_r):
        phu, phm = ph_l, ph_r
    else:
        phu, phm = ph_r, ph_l
    cu = float(f(phu))

    # the bucket's other end: from phi_s towards the other fixed point, where
    # delta^2 first turns negative — or that point itself, where the
    # separatrix only touches zero (phi_s = 0: the stationary bucket's ends)
    grid = np.linspace(phs, phm, 4097)[1:]
    neg = np.flatnonzero(scale * (cu - f(grid)) < 0.0)
    if neg.size == 0:
        ph2 = phm
    else:
        a = grid[neg[0] - 1] if neg[0] > 0 else phs
        b = grid[neg[0]]
        for _ in range(200):
            mid = 0.5 * (a + b)
            if scale * (cu - f(mid)) < 0.0:
                b = mid
            else:
                a = mid
            if abs(b - a) < 1e-15:
                break
        ph2 = 0.5 * (a + b)
    z_u, z_2 = (phs - phu) / k, (phs - ph2) / k
    z_min, z_max = min(z_u, z_2), max(z_u, z_2)
    dmax = math.sqrt(max(scale * (cu - f(phs)), 0.0))
    qs = math.sqrt(h * abs(eta) * float(voltage_mv) * abs(cos_s) / (2.0 * math.pi * b2E))

    def separatrix(zeta):
        z = np.asarray(zeta, dtype=float)
        d2 = scale * (cu - f(phs - k * z))
        ok = (z >= z_min) & (z <= z_max) & (d2 >= 0.0)
        return np.where(ok, np.sqrt(np.where(ok, d2, 0.0)), np.nan)

    zz = np.linspace(z_min, z_max, 4001)
    dd = np.nan_to_num(separatrix(zz))
    area = float(np.sum(0.5 * (dd[1:] + dd[:-1]) * np.diff(zz))) * 2.0
    out.update(qs=qs, delta_max=dmax, zeta_min=z_min, zeta_max=z_max,
               bucket_length_m=z_max - z_min, area_m=area, separatrix=separatrix)
    return out


def ring_optics(lattice, config=None, *, beam=None, six_d: bool = False,
                should_stop=None) -> RingOptics:
    """Compile ``lattice`` (see :func:`linac_gen.ring.compile.compile_from_config`)
    and compute its periodic optics."""
    from linac_gen.ring.compile import compile_from_config
    from linac_gen.ring.config import RingConfig
    rl = compile_from_config(lattice, config or RingConfig(), beam=beam)
    return twiss(rl, six_d=six_d, should_stop=should_stop)
