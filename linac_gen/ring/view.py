"""A ring run seen through a linac run's result names, turn by turn.

The Results tab's tiles read a linac run's series by name — ``s`` [mm] and, at
each entry, ``sigma_x`` [mm], ``emit_nx`` [mm mrad], ``sigma_matrix`` in
(x mm, x' mrad, y mm, y' mrad, dphi deg, dW MeV), the reference — as
:class:`~linac_gen.diagnostics.recorder.DiagnosticRecorder` records them along a
line.  :class:`RingResultsView` gives them a ring run's statistics rows in the
same names and units, one entry per row: ``x_domain = "turn"`` and the turn of
each row in ``s``, so the tiles show the beam turn after turn at the start of
the ring (s = 0).

Each row's centroid is :func:`linac_gen.ring.coords.to_helix` of the ring
centroid and its matrix is J C J^T, J the Jacobian of ``to_helix`` at that
centroid (x' = px/pz, dphi = -360 f zeta/(beta0 c), dW = W(delta) - W0):
exact to first order about the centroid, an orbit bump or an off-momentum
centroid included.  The phase is in degrees at one frequency f0 for the whole
run (the RF's at the first turn; without RF the revolution frequency), so the
bunch length in mm, the longitudinal emittance in mm mrad, the dispersion and
the canonical eigenemittances do not depend on it.  (The tiles' longitudinal
conversions — the dispersion D, the emittance in mm mrad, sigma(dp/p) — take
the reference's beta^2 gamma m: for a centroid off momentum by delta_c they
differ from the ring's <x delta>/<delta^2> and sqrt(det) of (zeta, delta) by a
relative delta_c/gamma0^2, as the linac recorder's would on the same
particles.)

The derived quantities are the linac recorder's (central moments; beta =
<u^2>/eps, alpha = -<u u'>/eps; the longitudinal emittance in mm mrad as
:func:`~linac_gen.diagnostics.recorder._convert_emit_z_to_mmmrad`; the normal-
mode and eigen-emittances as ``_normal_mode_emittances`` and
:func:`~linac_gen.diagnostics.eigenemittance.eigenemittances`), computed for
all rows at once — a run keeps up to some 10^5 rows — with the rows those two
helpers treat apart (a matrix with no positive determinant, invariants that
overflow) handed to the helpers themselves.  Series are lists (2-D ones lists
of arrays), as a linac results file is read back; a series a ring run does
not record is absent (the peak excursion, the halo parameter, per-particle
loss tables).
"""
from __future__ import annotations

import math

import numpy as np

C_LIGHT = 299792458.0


def received_by_turn(res, turns) -> np.ndarray | None:
    """With injection, the particles the ring received by each turn in
    ``turns`` — those it started with and the stripped ones injected so far;
    None without injection."""
    inj = getattr(res, "injection", None) or {}
    if not inj.get("settings") or "stripped" not in inj:
        return None
    cum = np.concatenate([[0.0], np.cumsum(np.asarray(inj["stripped"], dtype=float))])
    t = np.clip(np.asarray(turns, dtype=np.int64), 0, len(cum) - 1)
    return float(np.asarray(res.n_alive)[0]) + cum[t]


def rf_frequency_hz(res) -> float:
    """f0: the frequency the phase is measured at — the RF's at the run's first
    turn (a ramp's; else beta0 c over the period the run folds zeta into: C/h
    for the cavities' common harmonic h, C without RF; else h/T0), the
    revolution frequency without RF.  (The harmonic of a lattice is often not
    declared — a cavity given by its frequency — while the fold always follows
    the cavities.)"""
    ramp = getattr(res, "ramp", None) or {}
    f = ramp.get("frequency_hz")
    if f is not None and len(f):
        return float(np.asarray(f, dtype=float)[0])
    meta = res.meta or {}
    wrap = float(meta.get("zeta_wrap_m") or 0.0)
    beta0 = float(meta.get("beta0") or 0.0)
    if wrap > 0.0 and beta0 > 0.0:
        return beta0 * C_LIGHT / wrap
    t0 = float(meta.get("revolution_time_s") or 0.0)
    if t0 <= 0.0:
        return float("nan")
    h = meta.get("harmonic") if meta.get("rf_on") else None
    return float(h or 1) / t0


def _jacobians(mean, p0c, beta0, mass: float, f0: float):
    """Centroids (n, 6) and Jacobians (n, 6, 6) of coords.to_helix at each
    row's ring centroid ``mean`` (n, 6) (each row's reference p0c [MeV] and
    beta0; mass [MeV]; phase at f0 [Hz]); NaN where pz is not real."""
    mean = np.asarray(mean, dtype=float).reshape(-1, 6)
    p0c = np.asarray(p0c, dtype=float).reshape(-1)
    beta0 = np.asarray(beta0, dtype=float).reshape(-1)
    x, px, y, py, zeta, d = mean.T
    n = mean.shape[0]
    c = np.empty((n, 6))
    J = np.zeros((n, 6, 6))
    with np.errstate(invalid="ignore", divide="ignore"):
        opd = 1.0 + d
        pz2 = opd * opd - px * px - py * py
        pz = np.where(pz2 > 0.0, np.sqrt(pz2), np.nan)
        pc = p0c * opd
        e = np.sqrt(pc * pc + mass * mass)
        w = pc * pc / (e + mass)
        w0 = p0c * p0c / (np.sqrt(p0c * p0c + mass * mass) + mass)
        kphi = 360.0 * f0 / (beta0 * C_LIGHT)
        c[:, 0] = 1e3 * x
        c[:, 1] = 1e3 * px / pz
        c[:, 2] = 1e3 * y
        c[:, 3] = 1e3 * py / pz
        c[:, 4] = -kphi * zeta
        c[:, 5] = w - w0
        pz3 = pz2 * pz
        J[:, 0, 0] = 1e3
        J[:, 2, 2] = 1e3
        J[:, 1, 1] = 1e3 * (pz2 + px * px) / pz3
        J[:, 1, 3] = 1e3 * px * py / pz3
        J[:, 1, 5] = -1e3 * px * opd / pz3
        J[:, 3, 3] = 1e3 * (pz2 + py * py) / pz3
        J[:, 3, 1] = 1e3 * px * py / pz3
        J[:, 3, 5] = -1e3 * py * opd / pz3
        J[:, 4, 4] = -kphi
        J[:, 5, 5] = p0c * p0c * opd / e
    return c, J


def _jacobian(mean, p0c: float, beta0: float, mass: float, f0: float):
    """:func:`_jacobians` of one centroid: (c (6,), J (6, 6))."""
    c, J = _jacobians(np.asarray(mean, dtype=float).reshape(1, 6), [p0c], [beta0], mass, f0)
    return c[0], J[0]


def _sqrt_det2(S, i: int) -> np.ndarray:
    """sqrt(det) of each matrix's 2 x 2 block at (i, i); 0 where det <= 0."""
    d = S[:, i, i] * S[:, i + 1, i + 1] - S[:, i, i + 1] * S[:, i + 1, i]
    return np.sqrt(np.where(d > 0.0, d, 0.0))


_J4 = np.array([[0.0, 1.0, 0.0, 0.0], [-1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0], [0.0, 0.0, -1.0, 0.0]])


def _normal_modes(S4, det4):
    """(e1, e2) of each 4 x 4 block, e1 >= e2: ``recorder.
    _normal_mode_emittances`` for all rows at once (|Im| of the eigenvalues
    of J4 S4; (0, 0) where det <= 0)."""
    n = S4.shape[0]
    e1, e2 = np.zeros(n), np.zeros(n)
    ok = det4 > 0.0
    if ok.any():
        ims = np.sort(np.abs(np.linalg.eigvals(_J4 @ S4[ok]).imag), axis=1)
        e2[ok] = ims[:, 1]
        e1[ok] = ims[:, 3]
    return e1, e2


def _eigen(S, det6):
    """(n, 3) eigenemittances: :func:`~linac_gen.diagnostics.eigenemittance.
    eigenemittances` for all rows at once — the roots of its cubic in eps^2
    from the invariants I2, I4, I6 — with the rows it treats apart (det <= 0,
    invariants not finite, a zero constant term) through it."""
    from linac_gen.diagnostics.eigenemittance import _J6, eigenemittances
    n = S.shape[0]
    out = np.zeros((n, 3))
    regular = np.flatnonzero(det6 > 0.0)
    apart = [np.flatnonzero(~(det6 > 0.0))]
    if regular.size:
        with np.errstate(over="ignore", invalid="ignore"):
            s1 = S[regular] @ _J6
            s2 = s1 @ s1
            s4 = s2 @ s2
            s6 = s4 @ s2
            i2 = -0.5 * np.trace(s2, axis1=1, axis2=2)
            i4 = +0.5 * np.trace(s4, axis1=1, axis2=2)
            i6 = -0.5 * np.trace(s6, axis1=1, axis2=2)
            b = -i2
            c = 0.5 * (i2 * i2 - i4)
            d = -(i2 ** 3) / 6.0 + 0.5 * i2 * i4 - i6 / 3.0
        good = np.isfinite(b) & np.isfinite(c) & np.isfinite(d) & (d != 0.0)
        if good.any():
            # (the companion matrix of x^3 + b x^2 + c x + d, as numpy.roots)
            A = np.zeros((int(good.sum()), 3, 3))
            A[:, 1, 0] = A[:, 2, 1] = 1.0
            A[:, 0, 0], A[:, 0, 1], A[:, 0, 2] = -b[good], -c[good], -d[good]
            real = np.sort(np.real(np.linalg.eigvals(A)), axis=1)
            out[regular[good]] = np.sqrt(np.abs(real))[:, [1, 2, 0]]
        apart.append(regular[~good])
    for k in np.concatenate(apart):
        out[k] = eigenemittances(S[k])
    return out


class RingResultsView:
    """A ring run (:class:`~linac_gen.ring.engine.RingResults`) through the
    names of a linac run's results, one entry per statistics row (see the
    module docstring); ``ring`` is the run itself, ``kpi`` the six values of
    the Results tab's strip."""

    x_domain = "turn"
    is_ring_view = True

    def __init__(self, res, source_path=None):
        self.ring = res
        meta = res.meta or {}
        self.source_path = source_path or meta.get("dump_path")
        turns = np.asarray(res.turns, dtype=np.int64)
        n = int(turns.size)
        cov = np.asarray(res.cov, dtype=float).reshape(-1, 6, 6)
        mean = np.asarray(res.mean, dtype=float).reshape(-1, 6)
        mass = float(meta.get("mass_mev", float("nan")))
        bg = (np.asarray(res.row_beta0gamma0, dtype=float) if res.row_beta0gamma0 is not None
              else np.full(n, float(meta.get("beta0gamma0", float("nan")))))
        p0c = (np.asarray(res.row_p0c_mev, dtype=float) if res.row_p0c_mev is not None
               else np.full(n, float(meta.get("p0c_mev", float("nan")))))
        gamma = np.sqrt(1.0 + bg * bg)
        beta = bg / gamma
        w0 = p0c * p0c / (np.sqrt(p0c * p0c + mass * mass) + mass)
        f0 = rf_frequency_hz(res)
        rf_on = bool(meta.get("rf_on"))
        self.f0_hz = f0
        self.turn_recorded = int(turns[-1]) if n else None
        self.mass_mev = mass

        # rows with particles (a row without has NaN moments)
        fin = (np.all(np.isfinite(mean), axis=1) & np.all(np.isfinite(cov), axis=(1, 2))
               if n else np.zeros(0, dtype=bool))
        cen, J = _jacobians(mean, p0c, beta, mass, f0)
        S = J @ cov @ np.swapaxes(J, 1, 2)
        with np.errstate(invalid="ignore", divide="ignore"):
            sx, sy, sphi, sw = (np.sqrt(np.maximum(S[:, i, i], 0.0)) for i in (0, 2, 4, 5))
            ex, ey, ez = _sqrt_det2(S, 0), _sqrt_det2(S, 2), _sqrt_det2(S, 4)
            # (recorder._convert_emit_z_to_mmmrad, row by row: deg MeV -> mm mrad)
            wavelength = C_LIGHT / f0 * 1e3
            k_phi = 360.0 / (beta * wavelength)
            k_w = beta * beta * gamma * mass * 1e-3
            denom = k_phi * k_w
            ezm = np.where((beta <= 0.0) | (wavelength <= 0.0) | (denom <= 0.0), 0.0,
                           ez / denom)
            twiss = []
            for i, j, e in ((0, 1, ex), (2, 3, ey), (4, 5, ez)):
                big = e > 1e-30
                twiss.append((np.where(big, -S[:, i, j] / e, 0.0),
                              np.where(big, S[:, i, i] / e, 0.0)))
        (ax, bx), (ay, by), (az, bz) = twiss
        e4 = np.full(n, np.nan)
        en1, en2 = np.full(n, np.nan), np.full(n, np.nan)
        ee = np.full((n, 3), np.nan)
        if fin.any():
            S4 = S[fin, :4, :4]
            det4 = np.linalg.det(S4)
            e4[fin] = np.sqrt(np.maximum(det4, 0.0))
            en1[fin], en2[fin] = _normal_modes(S4, det4)
            ee[fin] = _eigen(S[fin], np.linalg.det(S[fin]))
        cols = [sx, sy, sphi, sw, ex, ey, ez, ezm, ax, bx, ay, by, az, bz]
        for a in cols:
            a[~fin] = np.nan
        cen[~fin] = np.nan
        S[~fin] = np.nan

        # along "s" — here the turn of each row (x_domain = "turn")
        self.s = turns.astype(float).tolist()
        self.element_names: list = []
        self.element_exit_idx: list = []
        self.sigma_x, self.sigma_y, self.sigma_phi, self.sigma_w = (
            sx.tolist(), sy.tolist(), sphi.tolist(), sw.tolist())
        self.emit_x, self.emit_y, self.emit_z = ex.tolist(), ey.tolist(), ez.tolist()
        self.emit_z_mmmrad = ezm.tolist()
        self.emit_nx = (ex * bg).tolist()
        self.emit_ny = (ey * bg).tolist()
        self.emit_nz = (ezm * bg).tolist()
        self.alpha_x, self.beta_x, self.alpha_y, self.beta_y = (
            ax.tolist(), bx.tolist(), ay.tolist(), by.tolist())
        self.alpha_z, self.beta_z = az.tolist(), bz.tolist()
        self.emit_4d, self.emit_n1, self.emit_n2 = e4.tolist(), en1.tolist(), en2.tolist()
        self.emit_e1, self.emit_e2, self.emit_e3 = (
            ee[:, 0].tolist(), ee[:, 1].tolist(), ee[:, 2].tolist())
        self.sigma_matrix = list(S)
        self.centroid = list(cen)
        received = received_by_turn(res, turns)
        n_alive = np.asarray(res.n_alive, dtype=float)
        if received is None:
            received = np.full(n, n_alive[0] if n else float("nan"))
        with np.errstate(divide="ignore", invalid="ignore"):
            trans = np.where(received > 0.0, 100.0 * n_alive / received, np.nan)
        self.transmission = trans.tolist()
        self.ref_w_kin = w0.tolist()
        self.ref_beta = beta.tolist()
        self.ref_gamma = gamma.tolist()
        self.ref_bg = bg.tolist()
        # (a list, as a live linac run's: tiles test it for truth)
        self.ref_frequency = [f0 * 1e-6] * n
        self.continuous_at = [not rf_on] * n
        # (zeta folded into the RF period — or the ring, without RF — while tracking)
        self.periodic_phase = bool(meta.get("zeta_wrap_m"))

        # turn by turn, in the ring's own units (the RING RUN tiles)
        self.turns = turns.tolist()
        self.n_alive = np.asarray(res.n_alive).astype(np.int64).tolist()
        self.received = None if received_by_turn(res, turns) is None else received.tolist()
        en = np.asarray(res.emit_normalized, dtype=float).reshape(-1, 2) * 1e6
        self.emit_nx_projected = en[:, 0].tolist()
        self.emit_ny_projected = en[:, 1].tolist()
        self.emit_long_evs = np.asarray(res.emit_long_evs, dtype=float).tolist()
        eb = _betatron_x(cov) * bg
        self.emit_nx_betatron = (eb * 1e6).tolist()
        with np.errstate(invalid="ignore"):
            self.sigma_ring = list(np.sqrt(np.maximum(np.diagonal(cov, axis1=1, axis2=2),
                                                      0.0)))
        self.mean_ring = list(mean.copy())

        self.kpi = self._kpi(cov, eb, trans)

    @staticmethod
    def _kpi(cov, eb, trans) -> dict:
        """The Results tab's six values: the beam at the end of the run (s = 0),
        the normalized betatron x emittance's growth over the run (``eb``: per
        row [m rad]), the survival."""
        out = {"sigma_x_mm": None, "sigma_y_mm": None, "sigma_z_mm": None, "ex_growth": None,
               "transmission": None, "loss": None}
        if not len(cov):
            return out
        last = cov[-1]
        if np.all(np.isfinite(last)):
            out["sigma_x_mm"] = math.sqrt(max(last[0, 0], 0.0)) * 1e3
            out["sigma_y_mm"] = math.sqrt(max(last[2, 2], 0.0)) * 1e3
            out["sigma_z_mm"] = math.sqrt(max(last[4, 4], 0.0)) * 1e3
        fin = np.flatnonzero(np.isfinite(eb) & (eb > 0.0))
        if fin.size >= 2:
            out["ex_growth"] = float(eb[fin[-1]] / eb[fin[0]])
        t = np.asarray(trans, dtype=float)
        tf = t[np.isfinite(t)]
        if tf.size:
            out["transmission"] = float(tf[-1])
            out["loss"] = 100.0 - float(tf[-1])
        return out


def _betatron_x(cov) -> np.ndarray:
    """sqrt(det) of each row's x-px block conditioned on delta (the dispersion
    removed) [m rad]; NaN where it is not defined."""
    cov = np.asarray(cov, dtype=float).reshape(-1, 6, 6)
    out = np.full(len(cov), np.nan)
    ok = np.all(np.isfinite(cov), axis=(1, 2))
    c = cov[ok]
    a00, a01, a10, a11 = c[:, 0, 0], c[:, 0, 1], c[:, 1, 0], c[:, 1, 1]
    b0, b1, d = c[:, 0, 5], c[:, 1, 5], c[:, 5, 5]
    pos = d > 0.0
    with np.errstate(invalid="ignore", divide="ignore"):
        a00 = np.where(pos, a00 - b0 * b0 / d, a00)
        a01 = np.where(pos, a01 - b0 * b1 / d, a01)
        a10 = np.where(pos, a10 - b1 * b0 / d, a10)
        a11 = np.where(pos, a11 - b1 * b1 / d, a11)
    det = a00 * a11 - a01 * a10
    out[ok] = np.sqrt(np.where(det > 0.0, det, 0.0))
    return out
