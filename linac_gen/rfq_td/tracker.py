"""Time loop of the time-domain RFQ solver.

Particles carry position (mm) and γβ; time (ns) is the independent
variable.  One step of length dt is a kick–drift–kick leapfrog
(velocity Verlet, second order, symplectic — Toutatis's scheme):

    γβ += (q/|q|)·k_e·E(r, t)·dt/2;  r += c·γβ/γ·dt;  γβ += …(r, t+dt)·dt/2

with E(r, t) = E_static(r)·sin(ωt + φ₀) from the electrode field map and
k_e = |q|·c/(m c²) (V/mm → 1/ns of γβ).  Space charge enters as a
separate leapfrog kick every ``sc_every`` steps.

A particle is lost when it touches metal (vane tip arcs and bodies, end
plates), turns back, or is still inside the RFQ after ``t_max`` (a
straggler).  Its position, time and energy at that moment are kept.

Each particle's crossing of every requested z plane is recorded by
linear interpolation inside the step: (x, x', y, y', t, W).
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

import numpy as np

C_MM_NS = 299.792458            # speed of light, mm/ns

#: loss reasons
REASON_METAL = 1
REASON_BACKWARD = 2
REASON_STALLED = 3


class Aborted(RuntimeError):
    """Raised when the caller's ``should_abort`` asks to stop."""


@dataclass
class TdResult:
    planes: np.ndarray            # (P,) mm
    # (P, N, 6): x, x', y, y', t - t_ref[k], W - w_ref[k]; NaN = no
    # crossing.  float32 when references are given (the chain driver),
    # else float64 with t_ref = w_ref = 0
    rec: np.ndarray
    lost: np.ndarray              # (N,) bool
    loss_z: np.ndarray            # (N,) mm
    loss_t: np.ndarray            # (N,) ns
    loss_xy: np.ndarray           # (N, 2) mm
    loss_W: np.ndarray            # (N,) MeV
    loss_reason: np.ndarray       # (N,) int8
    steps: int = 0
    wall_s: float = 0.0
    sc_solves: int = 0
    outside_map_steps: int = 0
    snapshots: list = field(default_factory=list)


def electrode_hit(electrodes, X, Y, Z):
    """Vectorised metal test (vane tips/bodies with parallel flanks, end
    plates) for points in any quadrant."""
    return electrodes.inside(X, Y, Z)


class TimeDomainTracker:
    """Track one RFQ in time."""

    def __init__(self, fieldmap, electrodes, *, freq_mhz: float,
                 mass_mev: float, charge: float, steps_per_period: int = 48,
                 phi0_deg: float = 0.0, space_charge=None,
                 sc_every: int = 4, t_max_ns: float | None = None):
        self.map = fieldmap
        self.el = electrodes
        self.f_ghz = freq_mhz * 1e-3
        self.omega = 2.0 * math.pi * self.f_ghz           # rad/ns
        self.T = 1.0 / self.f_ghz                          # ns
        self.mc2_ev = mass_mev * 1e6
        self.qsign = 1.0 if charge > 0 else -1.0
        self.k_e = abs(charge) * (1e3 * 299792458.0 / self.mc2_ev) * 1e-9
        self.nstep = int(steps_per_period)
        self.dt = self.T / self.nstep
        self.phi0 = math.radians(phi0_deg)
        self.sc = space_charge
        self.sc_every = max(int(sc_every), 1)
        self.t_max = t_max_ns

    def _efield(self, X, Y, Z, act):
        Ex, Ey, Ez, n_out = self.map.field(X, Y, Z, act)
        return Ex, Ey, Ez, n_out

    def run(self, X, Y, Z, ux, uy, uz, t0, planes, *, t_ref=None,
            w_ref=None, progress=None, should_abort=None,
            snapshot=None) -> TdResult:
        """Track from time ``t0`` until every particle has crossed the last
        plane or been lost.  Arrays are modified in place.

        With ``t_ref``/``w_ref`` (per plane, ns / MeV) the crossing times
        and energies are stored relative to them in float32 (0.01 deg and
        0.01 eV resolution), which halves the memory of a large run; a
        record above 1 GB lives in a temporary memory-mapped file."""
        n = X.size
        planes = np.asarray(planes, float)
        npl = planes.size
        relative = t_ref is not None
        t_ref = np.zeros(npl) if t_ref is None else np.asarray(t_ref, float)
        w_ref = np.zeros(npl) if w_ref is None else np.asarray(w_ref, float)
        dtype = np.float32 if relative else np.float64
        nbytes = npl * n * 6 * np.dtype(dtype).itemsize
        if nbytes > 1 << 30:
            import tempfile
            tmp = tempfile.TemporaryFile()
            rec = np.memmap(tmp, dtype=dtype, mode="w+", shape=(npl, n, 6))
            rec[:] = np.nan
        else:
            rec = np.full((npl, n, 6), np.nan, dtype=dtype)
        nxt = np.searchsorted(planes, Z, side="right")    # next plane index
        act = nxt < npl
        lost = np.zeros(n, bool)
        loss_z = np.full(n, np.nan)
        loss_t = np.full(n, np.nan)
        loss_xy = np.full((n, 2), np.nan)
        loss_W = np.full(n, np.nan)
        reason = np.zeros(n, np.int8)
        res = TdResult(planes, rec, lost, loss_z, loss_t, loss_xy, loss_W,
                       reason)
        dt = self.dt
        t = float(t0)
        tw = time.time()
        step = 0
        z_end = planes[-1]
        t_cap = (t + self.t_max) if self.t_max else math.inf
        Ex, Ey, Ez, n_out = self._efield(X, Y, Z, act)
        res.outside_map_steps += n_out
        s = self.qsign * self.k_e * math.sin(self.omega * t + self.phi0)
        if self.sc is not None:
            self.sc.kick(X, Y, Z, ux, uy, uz, act, 0.5 * dt * self.sc_every,
                         self.k_e)
        while act.any():
            Xo, Yo, Zo = X.copy(), Y.copy(), Z.copy()
            uxo, uyo, uzo = ux.copy(), uy.copy(), uz.copy()
            s_old = s
            t += dt
            step += 1
            s = self.qsign * self.k_e * math.sin(self.omega * t + self.phi0)
            # kick dt/2 - drift dt - field - kick dt/2: compiled in one pass
            # when available (the same operations as the numpy step below)
            n_out = self.map.push(X, Y, Z, ux, uy, uz, act, Ex, Ey, Ez,
                                  0.5 * dt * s_old, 0.5 * dt * s,
                                  C_MM_NS * dt)
            if n_out is None:
                a = act
                ux[a] += 0.5 * dt * s_old * Ex[a]
                uy[a] += 0.5 * dt * s_old * Ey[a]
                uz[a] += 0.5 * dt * s_old * Ez[a]
                g = np.sqrt(1.0 + ux[a] ** 2 + uy[a] ** 2 + uz[a] ** 2)
                f = C_MM_NS * dt / g
                X[a] += ux[a] * f
                Y[a] += uy[a] * f
                Z[a] += uz[a] * f
                Ex, Ey, Ez, n_out = self._efield(X, Y, Z, act)
                ux[a] += 0.5 * dt * s * Ex[a]
                uy[a] += 0.5 * dt * s * Ey[a]
                uz[a] += 0.5 * dt * s * Ez[a]
            res.outside_map_steps += n_out
            # plane crossings (several per step are possible)
            while True:
                cand = act & (nxt < npl)
                zp = planes[np.minimum(nxt, npl - 1)]
                cr = cand & (Z >= zp) & (Zo < zp)
                if not cr.any():
                    break
                i = np.flatnonzero(cr)
                fr = (zp[i] - Zo[i]) / (Z[i] - Zo[i])
                gx = uxo[i] + fr * (ux[i] - uxo[i])
                gy = uyo[i] + fr * (uy[i] - uyo[i])
                gz = uzo[i] + fr * (uz[i] - uzo[i])
                u2 = gx * gx + gy * gy + gz * gz
                gam = np.sqrt(1.0 + u2)
                k = nxt[i]
                rec[k, i, 0] = Xo[i] + fr * (X[i] - Xo[i])
                rec[k, i, 1] = gx / gz
                rec[k, i, 2] = Yo[i] + fr * (Y[i] - Yo[i])
                rec[k, i, 3] = gy / gz
                rec[k, i, 4] = (t - dt + fr * dt) - t_ref[k]
                # gamma - 1 = u^2/(gamma + 1): no cancellation at low energy
                rec[k, i, 5] = u2 / (gam + 1.0) * self.mc2_ev * 1e-6 - w_ref[k]
                nxt[i] += 1
                act[i[nxt[i] >= npl]] = False
            # losses
            a_idx = np.flatnonzero(act)
            if a_idx.size:
                hit = electrode_hit(self.el, X[a_idx], Y[a_idx], Z[a_idx])
                back = uz[a_idx] <= 0.0
                late = np.zeros(a_idx.size, bool) if t < t_cap \
                    else np.ones(a_idx.size, bool)
                why = np.where(hit, REASON_METAL,
                               np.where(back, REASON_BACKWARD,
                                        np.where(late, REASON_STALLED, 0)))
                dead = why > 0
                if dead.any():
                    j = a_idx[dead]
                    act[j] = False
                    lost[j] = True
                    loss_z[j] = Z[j]
                    loss_t[j] = t
                    loss_xy[j, 0] = X[j]
                    loss_xy[j, 1] = Y[j]
                    u2 = ux[j] ** 2 + uy[j] ** 2 + uz[j] ** 2
                    loss_W[j] = u2 / (np.sqrt(1.0 + u2) + 1.0) \
                        * self.mc2_ev * 1e-6
                    reason[j] = why[dead]
            if (self.sc is not None and step % self.sc_every == 0
                    and act.sum() >= max(2, int(0.01 * n))):
                self.sc.kick(X, Y, Z, ux, uy, uz, act, dt * self.sc_every,
                             self.k_e)
            if snapshot is not None:
                snapshot(step, t, X, Y, Z, ux, uy, uz, act)
            if step % self.nstep == 0:
                if should_abort is not None and should_abort():
                    raise Aborted("time-domain RFQ run aborted")
                if progress is not None and act.any():
                    progress(float(np.median(Z[act])), z_end)
        res.steps = step
        res.wall_s = time.time() - tw
        res.sc_solves = 0 if self.sc is None else self.sc.n_solve
        return res
