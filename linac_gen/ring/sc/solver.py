"""One space-charge kick after another: :class:`SpaceCharge`.

At each kick (:mod:`.placement`) of a turn:

1. the circulating particles' moments (:mod:`.kernels`; the centroid and rms
   sizes are the charge's, each particle weighted by its weight) size the
   grid: centred on the beam, it covers the beam out to its extent or
   ``box_sigma`` rms sizes, whichever is smaller — its cells on a ladder of
   ratio 2^(1/8) and its origin on whole cells, so the Green's functions
   (:mod:`.greens`) are reused from kick to kick and turn to turn;
2. the deposit gives the grid's charge and the line density (integers);
3. the field and potential of the normalized transverse distribution at the
   nodes, open boundary, and the pipe's harmonic (:mod:`.boundary`), added at
   each particle inside the wall; the multipole series of the grid's charge for the
   particles off the grid; the line density filtered (a Gaussian of
   ``z_smoothing_bins`` bins) and its slope, spectrally (periodic over the RF
   period);
4. the kick of every circulating particle — and of the tune probes, which
   feel the field without adding to it.

A coasting beam (no RF) has a uniform line density, N_b over the
circumference, and no longitudinal kick.  The frozen model replaces 1-3 by
the field of a Gaussian with the beam's centroid and rms sizes at that kick,
measured (with the line density) every ``refresh_turns`` turns.
"""
from __future__ import annotations

import bisect
import math
from collections import OrderedDict

import numpy as np

from linac_gen.ring.sc import greens
from linac_gen.ring.sc import kernels as K
from linac_gen.ring.sc.boundary import make_boundary
from linac_gen.ring.sc.config import SpaceChargeConfig
from linac_gen.ring.sc.placement import kick_positions

#: e^2 / (4 pi eps0) [eV m]: the classical radius of a particle of mass m is this over m c^2
E2_4PIEPS0_EVM = 1.43996454784e-9
#: order of the grid charge's multipole series (particles off the grid)
FAR_ORDER = 16
#: off the grid, the series is used beyond this many times the farthest charge
#: from its centre (its terms then fall at least as (1/1.5)^n: 1e-3 at order
#: 16); nearer, the direct sum over the charged cells
FAR_SERIES_MARGIN = 1.5
#: the frozen model keeps the fields of at most this many bytes of kicks (the
#: first kicks that fit; the others are solved again when needed: the same bits)
FROZEN_CACHE_BYTES = 512 * 2 ** 20
#: z_bins "auto": bins per rms of the bunch's core, the range (powers of two),
#: the start before a beam, and how far the bins asked for may stray from those
#: in use before they change (a quarter of a factor two past the rounding point,
#: either way)
AUTO_BINS_PER_SIGMA = 16.0
AUTO_NZ_MIN = 16
AUTO_NZ_MAX = 65536
AUTO_NZ_START = 256
AUTO_HYSTERESIS = 2.0 ** 0.75
#: z_bins "auto": with fewer particles than this the line is no bunch to
#: judge (one particle's is a spike): the bins and the width stay as they are
AUTO_MIN_PARTICLES = 100
#: z_bins "auto": the width (rms of a Gaussian filter) over which the line
#: density — and with it its slope — is smoothed, for N particles in a bunch
#: core of rms sigma: AUTO_WIDTH sigma N^(-1/5), the normal-reference
#: bandwidth of kernel density estimation, which balances a density's noise
#: against its smoothing (B. W. Silverman, Density Estimation for Statistics
#: and Data Analysis, 1986); never narrower than z_smoothing_bins bins
AUTO_WIDTH = (4.0 / 3.0) ** 0.2
#: the cell ladder: dx = LADDER_BASE * 2^(k / LADDER_STEPS)
LADDER_BASE = 1e-12
LADDER_STEPS = 8


def classical_radius(mass_mev: float) -> float:
    """e^2/(4 pi eps0 m c^2) [m] for a particle of rest energy ``mass_mev``."""
    return E2_4PIEPS0_EVM / (float(mass_mev) * 1e6)


def ladder(d: float) -> float:
    """The smallest cell size of the ladder at least d."""
    k = math.ceil(LADDER_STEPS * math.log2(max(d, LADDER_BASE) / LADDER_BASE) - 1e-9)
    return LADDER_BASE * 2.0 ** (k / LADDER_STEPS)


def _power_of_two_scale(total: float) -> float:
    """2^k with total x 2^k at most 2^60 (the deposit's integer scale)."""
    return 2.0 ** math.floor(60.0 - math.log2(total))


def combine_moments(rows: np.ndarray, xp: float, yp: float) -> dict:
    """The block rows of kernels.moments combined in order: the circulating
    particles (``n``), their weights' sum (``w``) and extents, and the
    charge's centroid and rms sizes — each block's weighted sums (relative to
    its largest weight) scaled to the largest weight of all, by exactly 1
    when the weights are equal: then the sums of the offsets and the
    particles' count, bit for bit."""
    rows = np.asarray(rows, dtype=float)
    used = rows[:, 0] > 0.0
    if not np.any(used):
        return {"n": 0, "w": 0.0}
    w_max = float(np.max(rows[used, 11]))
    f = rows[:, 11] / w_max if w_max > 0.0 else np.ones(rows.shape[0])
    # (summed block after block in order — an accumulation from 0, not a
    # pairwise sum: the same sums for any number of threads)
    terms = np.zeros((rows.shape[0] + 1, 7))
    terms[1:, :2] = rows[:, :2]
    terms[1:, 2:6] = f[:, None] * rows[:, 2:6]
    terms[1:, 6] = f * rows[:, 10]
    tot = np.add.accumulate(terms, axis=0)[-1]
    n = tot[0]
    s = tot[6]                                       # (the relative weights: n if all are equal)
    mxv, myv = tot[2] / s, tot[3] / s
    return {"n": int(n), "w": float(tot[1]), "x": xp + mxv, "y": yp + myv,
            "sx": math.sqrt(max(tot[4] / s - mxv * mxv, 0.0)),
            "sy": math.sqrt(max(tot[5] / s - myv * myv, 0.0)),
            "xmin": rows[:, 6].min(), "xmax": rows[:, 7].max(),
            "ymin": rows[:, 8].min(), "ymax": rows[:, 9].max()}


class Box:
    """The grid of one kick: node 0 at (x0, y0), cells dx x dy."""

    def __init__(self, nx, ny, x0, y0, dx, dy):
        self.nx, self.ny = int(nx), int(ny)
        self.x0, self.y0, self.dx, self.dy = float(x0), float(y0), float(dx), float(dy)

    def nodes(self):
        xs = self.x0 + np.arange(self.nx) * self.dx
        ys = self.y0 + np.arange(self.ny) * self.dy
        return np.meshgrid(xs, ys, indexing="ij")


def make_box(nx, ny, xc, yc, ax, ay) -> Box:
    """The grid holding [xc - ax, xc + ax] x [yc - ay, yc + ay] where the shape
    fits (grid coordinates 1 .. n - 2): cells from the ladder, origin on whole
    cells."""
    dx = ladder(2.0 * ax / (nx - 6))
    dy = ladder(2.0 * ay / (ny - 6))
    x0 = (math.floor(xc / dx) - nx // 2) * dx
    y0 = (math.floor(yc / dy) - ny // 2) * dy
    return Box(nx, ny, x0, y0, dx, dy)


#: the smallest half width of the grid [m]: a beam of (nearly) no size — one
#: particle — gets cells no finer than this, where the FFT's rounding of a
#: particle's own field stays far below any physical kick
MIN_HALF_WIDTH = 1e-6


def half_widths(m: dict, box_sigma: float) -> tuple[float, float]:
    """The half widths the grid must hold: the beam's extent or box_sigma rms
    sizes, the smaller (at least MIN_HALF_WIDTH, and 1e-3 of the other plane's
    rms size)."""
    out = []
    for c, s, lo, hi in (("x", m["sx"], m["xmin"], m["xmax"]),
                         ("y", m["sy"], m["ymin"], m["ymax"])):
        ext = max(hi - m[c], m[c] - lo) * (1.0 + 1e-9)
        a = min(ext, box_sigma * s) if s > 0.0 else ext
        out.append(max(a, MIN_HALF_WIDTH, 1e-3 * max(m["sx"], m["sy"])))
    return out[0], out[1]


def far_moments(q, X, Y, xc, yc, order=FAR_ORDER) -> np.ndarray:
    """(2, order + 1): the grid charge's moments M_n = sum q w^n about (xc, yc)
    (real arithmetic: the same bits wherever the arrays sit in memory)."""
    sel = q != 0.0
    wr = np.ascontiguousarray(X[sel] - xc)
    wi = np.ascontiguousarray(Y[sel] - yc)
    pr = np.ascontiguousarray(q[sel], dtype=float)
    pi = np.zeros_like(pr)
    out = np.zeros((2, order + 1))
    out[0, 0] = float(np.sum(pr))
    for n in range(1, order + 1):
        pr, pi = pr * wr - pi * wi, pr * wi + pi * wr
        out[0, n], out[1, n] = float(np.sum(pr)), float(np.sum(pi))
    return out


def charged_cells(q, X, Y, xc, yc, dx, dy):
    """(cells (3, m): the charged nodes and their charges, the radius beyond
    which the multipole series about (xc, yc) is used off the grid).  Nodes
    holding less than 1e-12 of the largest charge are left out (a frozen
    Gaussian's far tails): their field is below rounding."""
    qmax = float(np.max(q)) if q.size else 0.0
    sel = q > 1e-12 * qmax if qmax > 0.0 else q != 0.0
    cells = np.ascontiguousarray(np.vstack([X[sel], Y[sel], q[sel]]), dtype=float)
    if cells.shape[1] == 0:
        return np.zeros((3, 1)), 0.0
    r_q = float(np.max(np.hypot(cells[0] - xc, cells[1] - yc))) + 0.5 * math.hypot(dx, dy)
    return cells, FAR_SERIES_MARGIN * r_q


def gaussian_cells(box: Box, xc, yc, sx, sy) -> np.ndarray:
    """The charge of a unit Gaussian (centroid xc, yc; rms sx, sy) in each
    node's cell (exact integrals; a size of 0 puts it on its nearest nodes)."""
    from scipy.special import erf

    def one(n0, x0, d, c, s):
        edges = x0 + (np.arange(n0 + 1) - 0.5) * d
        if s <= 0.0:
            out = np.zeros(n0)
            out[int(np.clip(np.round((c - x0) / d), 0, n0 - 1))] = 1.0
            return out
        return 0.5 * np.diff(erf((edges - c) / (math.sqrt(2.0) * s)))
    return np.outer(one(box.nx, box.x0, box.dx, xc, sx), one(box.ny, box.y0, box.dy, yc, sy))


class SpaceCharge:
    """The kicks of a run (``cfg`` a SpaceChargeConfig or its dict) on the
    compiled ring ``rl``: ``zwrap`` the RF period in zeta (None: zeta not
    wrapped), ``rf_on`` whether the beam is bunched."""

    def __init__(self, cfg, rl, *, zwrap, rf_on):
        self.cfg = cfg if isinstance(cfg, SpaceChargeConfig) else SpaceChargeConfig.from_dict(cfg)
        c = self.cfg
        self.rl = rl
        self.nodes, self.s, self.ds = kick_positions(rl, c.kicks_per_turn)
        self.boundary = make_boundary(c)
        self.nx, self.ny = c.grid
        self.order = 2 if c.deposit == "tsc" else 1
        self.bunched = bool(rf_on)
        if self.bunched and not zwrap:
            raise ValueError("space charge: a bunched beam needs zeta wrapped into the RF "
                             "period (zeta_wrap 'auto', the cavities at a harmonic of the "
                             "revolution frequency)")
        self.period = float(zwrap) if self.bunched else float(rl.length)
        # (z_bins "auto": the bins and the smoothing follow the bunch's core —
        # set from the starting beam, then each turn from the line deposited at
        # its first kick, the same integers on a CPU and a GPU; see auto_bins
        # and auto_width)
        self.auto_z = c.z_bins == "auto"
        self.nz = AUTO_NZ_START if self.auto_z else int(c.z_bins)
        self.dz = self.period / self.nz
        self.nz_history = [(0, self.nz)]
        self._first_line = None
        self._nz_pending = None
        #: "auto": the smoothing width in use [m] (None: z_smoothing_bins bins)
        #: and the one the last turn asked for
        self.width = None
        self._width_pending = None
        self._filters = None
        self.long_mode = 0
        if self.bunched and c.longitudinal != "off":
            self.long_mode = 1 if c.longitudinal == "potential" else 2
        self.r_c = classical_radius(rl.reference.mass)
        self.frozen = c.model == "frozen"
        self._frozen_state = None           # per kick: (x, y, sx, sy), line (f, f')
        # (the run's diagnostic counters: see diagnostics_state)
        self.n_outside_max = 0
        self.n_kicks_done = 0
        self.n_beyond_wall = 0              # kicks with charge at or beyond a round pipe's wall
        self.last_box = None                # the grid of the last kick (diagnostics)
        self._zero_grid = np.zeros((self.nx, self.ny))

    def needs_refresh(self, turn: int) -> bool:
        """Whether the frozen model measures the beam again this turn."""
        if not self.frozen:
            return False
        if self._frozen_state is None:
            return True
        r = int(self.cfg.refresh_turns)
        return r > 0 and int(turn) % r == 0

    # ---- what a kick needs ----------------------------------------------------
    def _reference(self, k):
        j = int(self.nodes[k])
        beta0 = float(self.rl.ref_beta0[j])
        bg2 = float(self.rl.ref_bg2[j])
        gamma0 = math.sqrt(1.0 + bg2)
        return beta0, gamma0

    def _params(self, k, box: Box, centre, n_b: float):
        beta0, gamma0 = self._reference(k)
        kp = np.zeros(K.KP_N)
        ki = np.zeros(K.KI_N, dtype=np.int64)
        kp[K.KP_X0], kp[K.KP_Y0] = box.x0, box.y0
        kp[K.KP_DX], kp[K.KP_DY] = box.dx, box.dy
        kp[K.KP_DXC], kp[K.KP_DYC] = box.dx, box.dy
        kp[K.KP_XC], kp[K.KP_YC] = centre
        kp[K.KP_XP], kp[K.KP_YP] = self.boundary.centre
        kp[K.KP_IR] = 1.0 / self.boundary.scale
        kp[K.KP_PA] = float(self.cfg.pipe_a_m or 0.0)
        kp[K.KP_PB] = float(self.cfg.pipe_b_m or self.cfg.pipe_a_m or 0.0)
        kp[K.KP_Z0] = -0.5 * self.period + 0.5 * self.dz
        kp[K.KP_DZ] = self.dz
        f = 2.0 * self.r_c * n_b * float(self.ds[k])
        kp[K.KP_KT] = f / (beta0 * beta0 * gamma0 ** 3)
        kp[K.KP_KL] = f / (beta0 * gamma0 ** 3)
        kp[K.KP_PHI_G] = 0.5 * (self.cfg.g_factor or 0.0)
        kp[K.KP_F0] = 1.0 / float(self.rl.length)
        kp[K.KP_BETA0] = beta0
        ki[K.KI_NX], ki[K.KI_NY], ki[K.KI_NZ] = self.nx, self.ny, self.nz
        ki[K.KI_ORDER] = self.order
        ki[K.KI_NFAR] = FAR_ORDER
        ki[K.KI_LONG] = self.long_mode
        ki[K.KI_LINE] = 1 if self.bunched else 0
        ki[K.KI_PIPE] = {"open": 0, "round": 1, "rectangle": 2, "ellipse": 3}[self.cfg.pipe]
        return kp, ki

    def _line_filter(self, size: int):
        """The Gaussian filter on the rfft modes (None: none), kept until the
        bins or the smoothing change."""
        key = (self.nz, size, self.width)
        if self._filters is not None and self._filters[0] == key:
            return self._filters[1]
        sb = float(self.cfg.z_smoothing_bins)
        if not self.auto_z or self.width is None:
            # (scalar exponentials: the same bits in every process)
            g = (np.array([math.exp(-0.5 * (2.0 * math.pi * kk * sb / self.nz) ** 2)
                           for kk in range(size)]) if sb > 0.0 else None)
        else:
            w = self.filter_width()
            g = (np.array([math.exp(-0.5 * (2.0 * math.pi * kk * w / self.period) ** 2)
                           for kk in range(size)]) if w > 0.0 else None)
        self._filters = (key, g)
        return g

    def line_density(self, counts, scale, total):
        """(f, f') [1/m, 1/m^2] at the bin centres from the deposit's counts."""
        f = np.asarray(counts, dtype=float) / (scale * total * self.dz)
        F = np.fft.rfft(f)
        fr, fi = np.ascontiguousarray(F.real), np.ascontiguousarray(F.imag)
        k = np.arange(F.size)
        g = self._line_filter(F.size)
        if g is not None:
            fr, fi = fr * g, fi * g
        # (real arithmetic throughout: numpy's complex multiply may round
        # otherwise depending on where the arrays sit in memory)
        Fs = np.empty(F.size, dtype=complex)
        Fs.real, Fs.imag = fr, fi
        fl = np.fft.irfft(Fs, n=self.nz)
        om = 2.0 * np.pi * k / self.period
        D = np.empty(F.size, dtype=complex)
        D.real, D.imag = -fi * om, fr * om
        if self.nz % 2 == 0:
            D[-1] = 0.0                          # (the Nyquist mode has no slope)
        dfl = np.fft.irfft(D, n=self.nz)
        return np.ascontiguousarray(fl), np.ascontiguousarray(dfl)

    def fields(self, q, box: Box, *, need_phi: bool):
        """(Phi, F_x, F_y) of the normalized cell charges ``q`` in free space at
        the nodes of ``box``, and the pipe's coefficients: its harmonic is added
        at each particle by the kick (inside the wall; its constant term
        everywhere) — not on the nodes, where a particle near the wall would
        gather nodes on both sides of it."""
        phi, fx, fy = greens.solve(q, box.dx, box.dy, potential=need_phi)
        X, Y = box.nodes()

        def wall(px, py):
            return _wall_potential(px, py, X, Y, q)
        c = self.boundary.coefficients(q, X, Y, potential_at=wall)
        bnd = np.zeros((2, len(c)))
        bnd[0], bnd[1] = c.real, c.imag
        phi = phi if need_phi else self._zero_grid
        return np.ascontiguousarray(phi), np.ascontiguousarray(fx), \
            np.ascontiguousarray(fy), bnd

    # ---- one kick ---------------------------------------------------------------
    def kick(self, k: int, beam, probes=None, *, n_b: float | None = None,
             refresh: bool = True) -> None:
        """Kick ``beam`` (a CPU or CUDA beam: moments, deposit, kick) at kick
        k, and the tune ``probes`` (host arrays: coords, ptau, rpp, rvv,
        state) with the same field."""
        if self.frozen:
            self._kick_frozen(k, beam, probes, n_b=n_b, refresh=refresh)
            return
        rows = beam.moments(0.0, 0.0)
        m = combine_moments(rows, 0.0, 0.0)
        if m["n"] == 0 or not m["w"] > 0.0:
            return
        ax, ay = half_widths(m, self.cfg.box_sigma)
        box = make_box(self.nx, self.ny, m["x"], m["y"], ax, ay)
        self.last_box = box
        kp, ki = self._params(k, box, (m["x"], m["y"]), m["w"])
        scale = _power_of_two_scale(m["w"])
        grid, line, n_out = beam.deposit(kp, ki, scale, True)
        q = grid / (scale * m["w"])
        need_phi = self.long_mode == 1
        phi, fx, fy, bnd = self.fields(q, box, need_phi=need_phi)
        beyond = self._charge_beyond()
        ki[K.KI_NBND] = bnd.shape[1] - 1
        X, Y = box.nodes()
        far = far_moments(q, X, Y, m["x"], m["y"])
        cells, kp[K.KP_RCONV] = charged_cells(q, X, Y, m["x"], m["y"], box.dx, box.dy)
        ki[K.KI_NCELL] = cells.shape[1] if kp[K.KP_RCONV] > 0.0 else 0
        if self.bunched:
            fl, dfl = self.line_density(line, scale, m["w"])
            if self.auto_z and k == 0:
                self._first_line = (np.array(line, dtype=float), m["n"])
        else:
            fl = dfl = np.zeros(self.nz)
        beam.kick(fx, fy, phi, fl, dfl, far, cells, bnd, kp, ki)
        if probes is not None:
            kick_host(probes, fx, fy, phi, fl, dfl, far, cells, bnd, kp, ki)
        self.n_outside_max = max(self.n_outside_max, int(n_out))
        self.n_beyond_wall += int(beyond)
        self.n_kicks_done += 1

    # ---- the frozen model -----------------------------------------------------
    def _kick_frozen(self, k, beam, probes, *, n_b, refresh):
        st = self._frozen_state
        if st is None:
            st = self._frozen_state = {"gauss": np.full((len(self.nodes), 4), np.nan),
                                       "line": np.zeros((len(self.nodes), 2, self.nz)),
                                       "cache": OrderedDict(), "cache_bytes": 0}
        if refresh:
            rows = beam.moments(0.0, 0.0)
            m = combine_moments(rows, 0.0, 0.0)
            if m["n"] == 0:
                return
            st["gauss"][k] = (m["x"], m["y"], m["sx"], m["sy"])
            old = st["cache"].pop(k, None)
            if old is not None:
                st["cache_bytes"] -= self._cached_bytes(old)
            if self.bunched and m["w"] > 0.0:
                # (the line density alone: no grid deposit)
                kp0, ki0 = self._params(k, Box(self.nx, self.ny, 0.0, 0.0, 1.0, 1.0),
                                        (m["x"], m["y"]), m["w"])
                scale = _power_of_two_scale(m["w"])
                _g, line, _n = beam.deposit(kp0, ki0, scale, False)
                st["line"][k, 0], st["line"][k, 1] = self.line_density(line, scale, m["w"])
                if self.auto_z and k == 0:
                    self._first_line = (np.array(line, dtype=float), m["n"])
            n_b = m["w"]
        if n_b is None or not n_b > 0.0 or not np.isfinite(st["gauss"][k, 0]):
            return
        built = st["cache"].get(k)
        if built is None:
            built = self._frozen_fields(k)
            size = self._cached_bytes(built)
            # (kept while they fit; beyond, solved each time: a turn visits the
            # kicks in order, so evicting the oldest would keep none it needs)
            if st["cache_bytes"] + size <= FROZEN_CACHE_BYTES or not st["cache"]:
                st["cache"][k] = built
                st["cache_bytes"] += size
        box, centre, phi, fx, fy, bnd, far, cells, r_conv, beyond = built
        kp, ki = self._params(k, box, centre, n_b)
        ki[K.KI_NBND] = bnd.shape[1] - 1
        kp[K.KP_RCONV] = r_conv
        ki[K.KI_NCELL] = cells.shape[1] if r_conv > 0.0 else 0
        fl, dfl = st["line"][k, 0], st["line"][k, 1]
        beam.kick(fx, fy, phi, fl, dfl, far, cells, bnd, kp, ki)
        if probes is not None:
            kick_host(probes, fx, fy, phi, fl, dfl, far, cells, bnd, kp, ki)
        # (counted per kick, with the flag kept with the fields: not per solve,
        # whose number depends on the cache)
        self.n_beyond_wall += int(beyond)
        self.n_kicks_done += 1

    def _cached_bytes(self, built) -> int:
        """The bytes a frozen-model cache entry holds (the potential shared by
        every kick without a longitudinal kick not counted)."""
        return sum(a.nbytes for a in built[2:5] if a is not self._zero_grid) + built[7].nbytes

    def _frozen_fields(self, k):
        x, y, sx, sy = self._frozen_state["gauss"][k]
        bs = self.cfg.box_sigma
        ax = max(bs * sx, MIN_HALF_WIDTH, 1e-3 * max(sx, sy))
        ay = max(bs * sy, MIN_HALF_WIDTH, 1e-3 * max(sx, sy))
        box = make_box(self.nx, self.ny, x, y, ax, ay)
        q = gaussian_cells(box, x, y, sx, sy)
        phi, fx, fy, bnd = self.fields(q, box, need_phi=self.long_mode == 1)
        beyond = self._charge_beyond()
        X, Y = box.nodes()
        far = far_moments(q, X, Y, x, y)
        cells, r_conv = charged_cells(q, X, Y, x, y, box.dx, box.dy)
        return box, (x, y), phi, fx, fy, bnd, far, cells, r_conv, beyond

    def _charge_beyond(self) -> bool:
        """Whether the pipe's last solve had charge at or beyond its wall (a
        round pipe's: left out of its images)."""
        return bool(getattr(self.boundary, "charge_beyond", False))

    # ---- the line density's bins, "auto" -------------------------------------
    def auto_target(self, sigma_core: float, n: float) -> float | None:
        """The bins over the RF period a bunch core of rms ``sigma_core`` asks
        for, not rounded: AUTO_BINS_PER_SIGMA bins per rms.  None without a
        bunch to judge (no core, no particles)."""
        if not (sigma_core > 0.0 and n > 0):
            return None
        return max(1.0, self.period * AUTO_BINS_PER_SIGMA / sigma_core)

    def auto_width(self, sigma_core: float, n: float) -> float | None:
        """The width [m] over which "auto" smooths the line density (and its
        slope) for a core of rms ``sigma_core`` and ``n`` particles: AUTO_WIDTH
        sigma n^(-1/5), before the floor of z_smoothing_bins bins.  None
        without a bunch to judge."""
        if not (sigma_core > 0.0 and n > 0):
            return None
        return AUTO_WIDTH * sigma_core * float(n) ** -0.2

    def filter_width(self, nz: int | None = None, width: float | None = None) -> float:
        """The rms width [m] of the filter on the line density (and its slope)
        with bins ``nz`` (None: the current ones) and "auto" width ``width``
        (None: the current one) — at least z_smoothing_bins bins."""
        dz = self.dz if nz is None else self.period / int(nz)
        floor = float(self.cfg.z_smoothing_bins) * dz
        w = self.width if width is None else width
        if not self.auto_z or w is None:
            return floor
        return max(floor, float(w))

    def auto_limit(self, sigma_core: float, n: float, nz: int, line_width=None) -> str:
        """What kept "auto"'s line density coarse for a core of rms
        ``sigma_core``, bins ``nz`` and line filter ``line_width`` [m]:
        "particles" (the filter widened for the particles' noise), "largest"
        (AUTO_NZ_MAX bins), else "follow" (the bins follow the core a turn
        late and in factors of two)."""
        floor = float(self.cfg.z_smoothing_bins) * self.period / int(nz)
        if line_width is not None and float(line_width) > floor * (1.0 + 1e-12):
            return "particles"
        return "largest" if int(nz) >= AUTO_NZ_MAX else "follow"

    def auto_bins(self, sigma_core: float, n: float) -> int | None:
        """:meth:`auto_target` as a power of two from AUTO_NZ_MIN to
        AUTO_NZ_MAX (None without a bunch to judge)."""
        x = self.auto_target(sigma_core, n)
        return None if x is None else _power_of_two_bins(x)

    def start_bins(self, sigma_core: float, n: float, turn: int = 0) -> None:
        """"auto": the bins and the smoothing for the starting beam (its core's
        rms and particles)."""
        if not self.auto_z or not n >= AUTO_MIN_PARTICLES:
            return
        nz = self.auto_bins(sigma_core, n)
        if nz is not None and nz != self.nz:
            self._set_nz(nz, turn)
        w = self.auto_width(sigma_core, n)
        if w is not None:
            self.width = w
            self._filters = None

    def begin_turn(self, turn: int, refresh: bool) -> None:
        """"auto": the bins and the smoothing the last turn's line asked for,
        from this turn on (for the frozen model, on a turn that measures the
        beam again: its stored line densities all have the bins and the
        smoothing they were made with)."""
        if self.frozen and not refresh:
            return
        if self._nz_pending is not None:
            self._set_nz(self._nz_pending, turn)
            self._nz_pending = None
        if self._width_pending is not None:
            self.width, self._width_pending = self._width_pending, None
            self._filters = None

    def end_turn(self, turn: int) -> None:
        """"auto": the bins for the next turns, from the line deposited at
        this turn's first kick (its core's rms: 1.4826 times the median
        absolute deviation); they change once the bunch asks for more than
        AUTO_HYSTERESIS times the bins in use, or less than 1/AUTO_HYSTERESIS
        (a bunch near a rounding point does not switch them to and fro)."""
        got, self._first_line = self._first_line, None
        if not self.auto_z or got is None:
            return
        counts, n = got
        if not n >= AUTO_MIN_PARTICLES:
            return
        sigma = _core_rms(counts, self.dz)
        x = self.auto_target(sigma, n)
        if x is None:
            return
        nz = _power_of_two_bins(x)
        inside = self.nz / AUTO_HYSTERESIS <= x <= self.nz * AUTO_HYSTERESIS
        self._nz_pending = None if (nz == self.nz or inside) else nz
        self._width_pending = self.auto_width(sigma, n)

    def _set_nz(self, nz: int, turn: int) -> None:
        self.nz = int(nz)
        self.dz = self.period / self.nz
        self.nz_history.append((int(turn), self.nz))
        self._filters = None
        if self._frozen_state is not None:
            self._frozen_state["line"] = np.zeros((len(self.nodes), 2, self.nz))

    def bins_at(self, turn: int) -> int:
        """The bins of turn ``turn``."""
        i = bisect.bisect_right([t for t, _b in self.nz_history], int(turn)) - 1
        return self.nz_history[max(i, 0)][1]

    def bins_state(self) -> dict:
        """The bins' and the smoothing's state (for a checkpoint; floats as
        their hex text, for the same bits)."""
        def hx(w):
            return None if w is None else float(w).hex()
        return {"nz": self.nz, "pending": self._nz_pending,
                "history": [list(h) for h in self.nz_history],
                "width": hx(self.width), "width_pending": hx(self._width_pending)}

    def restore_bins(self, state) -> None:
        if not state:
            return
        self.nz = int(state["nz"])
        self.dz = self.period / self.nz
        self._nz_pending = None if state.get("pending") is None else int(state["pending"])
        self.nz_history = [(int(t), int(b)) for t, b in state["history"]]

        def unhex(w):
            return None if w is None else float.fromhex(w)
        self.width = unhex(state.get("width"))
        self._width_pending = unhex(state.get("width_pending"))
        self._filters = None

    def diagnostics_state(self) -> dict:
        """The run's diagnostic counters so far (for a checkpoint: a run
        continued from it reports what the uninterrupted run does): the most
        particles off the grid at a kick, the kicks with charge at or beyond a
        round pipe's wall, the kicks made."""
        return {"outside_grid_max": int(self.n_outside_max),
                "kicks_beyond_wall": int(self.n_beyond_wall),
                "kicks_done": int(self.n_kicks_done)}

    def restore_diagnostics(self, state) -> None:
        """The counters of :meth:`diagnostics_state` (None or empty — a
        checkpoint without them: left as they are)."""
        if not state:
            return
        self.n_outside_max = int(state.get("outside_grid_max", 0))
        self.n_beyond_wall = int(state.get("kicks_beyond_wall", 0))
        self.n_kicks_done = int(state.get("kicks_done", 0))

    def frozen_arrays(self) -> dict | None:
        """The frozen model's state (for a checkpoint)."""
        if self._frozen_state is None:
            return None
        return {"gauss": self._frozen_state["gauss"].copy(),
                "line": self._frozen_state["line"].copy()}

    def restore_frozen(self, arrays) -> None:
        if arrays is None:
            return
        self._frozen_state = {"gauss": np.array(arrays["gauss"], dtype=float),
                              "line": np.array(arrays["line"], dtype=float),
                              "cache": OrderedDict(), "cache_bytes": 0}

    def peak_ratio(self, sigma_zeta: float, *, nz: int | None = None,
                   line_width: float | None = None) -> float | None:
        """How much of a Gaussian bunch's peak line density (rms length
        ``sigma_zeta``) the bins (``nz``, None: the current ones) and the
        filter (rms ``line_width`` [m], None: z_smoothing_bins bins) keep:
        they show it as one of rms sqrt(sigma^2 + filter^2 + dz^2/3) (None for
        a coasting beam, which has no line density)."""
        if not self.bunched or not sigma_zeta > 0.0:
            return None
        dz = self.dz if nz is None else self.period / int(nz)
        fw = float(self.cfg.z_smoothing_bins) * dz if line_width is None else float(line_width)
        s2 = float(sigma_zeta) ** 2
        return math.sqrt(s2 / (s2 + fw * fw + dz * dz / 3.0))

    def summary(self) -> dict:
        gaps = np.mod(np.roll(self.s, -1) - self.s, float(self.rl.length))
        if len(self.s) == 1:
            gaps = np.array([float(self.rl.length)])
        hist = self.nz_history
        if len(hist) > 200:
            # (a long run's changes: the first and the last hundred; how many
            # in all beside them)
            hist = hist[:100] + hist[-100:]
        return {"z_bins_used": [list(h) for h in hist], "z_bins_auto": self.auto_z,
                "z_bins_changes": len(self.nz_history),
                "kicks": int(len(self.nodes)), "kick_s_m": [float(v) for v in self.s],
                "kick_length_m": [float(v) for v in self.ds],
                "largest_spacing_m": float(np.max(gaps)), "bunched": self.bunched,
                "line_period_m": self.period, "longitudinal_kick": self.long_mode != 0,
                "classical_radius_m": self.r_c,
                "kicks_charge_beyond_wall": int(self.n_beyond_wall),
                "pipe_fit_residual": (float(self.boundary.fit_residual)
                                      if hasattr(self.boundary, "fit_residual") else None)}


def _power_of_two_bins(x: float) -> int:
    """The power of two nearest to ``x`` (in its logarithm), from AUTO_NZ_MIN
    to AUTO_NZ_MAX."""
    nz = 2 ** int(round(math.log2(max(float(x), 1.0))))
    return int(min(max(nz, AUTO_NZ_MIN), AUTO_NZ_MAX))


def _core_rms(counts, dz: float) -> float:
    """The rms of a bunch's core from its line deposit (counts at the bin
    centres, spread evenly over each bin): 1.4826 times the median absolute
    deviation, from the cumulative counts — 0 for an empty line.  The line is
    periodic: the bunch is first moved to the middle of the period (by its
    circular mean), so that one lying across the period's ends is measured
    whole.  The deposit (cloud in cell: variance dz^2/6) and the counts read
    as spread over their bins (dz^2/12) widen a core by dz^2/4 in quadrature:
    taken off (a core within a bin or two keeps at least half its width)."""
    c = np.asarray(counts, dtype=float)
    tot = float(c.sum())
    if not tot > 0.0:
        return 0.0
    ang = 2.0 * math.pi * np.arange(c.size) / c.size
    mean = math.atan2(float(np.dot(c, np.sin(ang))), float(np.dot(c, np.cos(ang))))
    c = np.roll(c, c.size // 2 - int(round(mean / (2.0 * math.pi) * c.size)))
    edges = (np.arange(c.size + 1) - 0.5 * c.size) * dz
    cdf = np.concatenate(([0.0], np.cumsum(c) / tot))
    med = float(np.interp(0.5, cdf, edges))
    lo, hi = 0.0, 0.5 * c.size * dz
    for _ in range(60):
        d = 0.5 * (lo + hi)
        inside = np.interp(med + d, edges, cdf) - np.interp(med - d, edges, cdf)
        lo, hi = (d, hi) if inside < 0.5 else (lo, d)
    s = 1.4826 * 0.5 * (lo + hi)
    return math.sqrt(max(s * s - 0.25 * dz * dz, 0.25 * s * s))


def _wall_potential(px, py, X, Y, q):
    """The free-space Phi at wall points from the grid's node charges (point
    charges; numba when present)."""
    fn = _wall_fn()
    out = np.empty(len(px))
    sel = q != 0.0
    fn(np.ascontiguousarray(px, dtype=float), np.ascontiguousarray(py, dtype=float),
       np.ascontiguousarray(X[sel]), np.ascontiguousarray(Y[sel]),
       np.ascontiguousarray(q[sel]), out)
    return out


def _wall_sum(px, py, xs, ys, qs, out):
    for k in range(px.shape[0]):
        acc = 0.0
        for c in range(xs.shape[0]):
            dx = px[k] - xs[c]
            dy = py[k] - ys[c]
            acc -= qs[c] * 0.5 * math.log(dx * dx + dy * dy)
        out[k] = acc


_WALL = None


def _wall_fn():
    global _WALL
    if _WALL is None:
        try:
            import numba
            _WALL = numba.njit(cache=False)(_wall_sum)
        except ImportError:
            _WALL = _wall_sum
    return _WALL


def kick_host(arrays, fx, fy, phi, fl, dfl, far, cells, bnd, kp, ki) -> None:
    """The kick of a few particles on the host (the tune probes): ``arrays``
    = (coords, ptau, rpp, rvv, state)."""
    coords, ptau, rpp, rvv, state = arrays
    _m, _d, _kpar, kser = K.sc_backend()
    kser(coords, ptau, rpp, rvv, state, coords.shape[1], fx, fy, phi, fl, dfl, far, cells, bnd,
         kp, ki)
