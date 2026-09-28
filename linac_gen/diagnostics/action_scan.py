"""Halo action scan: the part of the beam outside nested rms ellipses.

At one position s and in one phase-space plane (u, u'), the
Courant–Snyder invariant of particle i with respect to the beam's OWN
rms Twiss at that position,

    W_i = gamma*u_i^2 + 2*alpha*u_i*u'_i + beta*u'_i^2,

is the emittance of the ellipse (of the beam's rms shape) through the
particle; ``<W> = 2*eps_rms`` for any distribution
(:func:`linac_gen.diagnostics.tail.cs_actions` returns the same W for the
raw planes).  The scan measures ellipse sizes in units of the LOCAL rms
emittance,

    n_i = W_i / eps_rms,

so n = 1 is the rms ellipse (it reaches 1 sigma in u and in u') and a
particle at n reaches sqrt(n) rms sizes.  At every recorded step it
counts, on a fixed grid of n, the particles OUTSIDE each ellipse:

    N_out(n_j) = #{i : n_i > n_j}.

Analytic anchors (pinned in tests):

* Gaussian: N_out(n)/N = exp(-n/2) — 1 % beyond n = 9.21, 0.1 % beyond
  13.82, 0.01 % beyond 18.42 (so the 95 % ellipse is 6.0 eps_rms);
* uniformly filled ellipse: N_out(n)/N = 1 - n/4 for n <= 4.

Relation to the other halo diagnostics: the fractional emittances of
:mod:`linac_gen.diagnostics.tail` (raw x and y) are
eps_q = quantile_q(W)/2 (e.g. eps_99 = 4.6 eps_rms), so in the raw planes
a fractional emittance eps_q corresponds to n = 2*eps_q/eps_rms here.

Planes
------
``x``, ``y``          betatron (dispersion-corrected): u -> u - (S_u5/S_55)*dW
                      and u' -> u' - (S_u'5/S_55)*dW, whose 2x2 block is
                      the Schur complement used by the Results tab's
                      "Dispersion-corrected (betatron only)" view.  Where
                      S_55 <= 1e-30 (no energy spread) nothing is removed.
``x_raw``, ``y_raw``  as tracked.
``z``                 (phi [deg], W [MeV]); never corrected.

Units: eps of x/y/x_raw/y_raw in mm*mrad (geometric), of z in deg*MeV.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

PLANES = ("x", "y", "z", "x_raw", "y_raw")
_COLS = {"x": (0, 1), "y": (2, 3), "z": (4, 5),
         "x_raw": (0, 1), "y_raw": (2, 3)}
_BETATRON = frozenset(("x", "y"))
# Same gate as the Results tab's _betatron_sigma_blocks.
_S55_EPS = 1e-30
# A plane whose 2x2 block has det <= this * S11*S22 has no ellipse
# (zero plane, perfectly correlated / rank-1 distribution).
_DET_REL_TOL = 1e-12
# Fewer alive particles than this: no rms ellipse to speak of.
_MIN_PARTICLES = 3

SCHEMA_VERSION = 1
DEFINITION = (
    "N_out(n) = number of alive particles with W/eps_rms > n, "
    "W = gamma*u^2 + 2*alpha*u*u' + beta*u'^2 with the beam's own rms "
    "Twiss at that step (n = 1: rms ellipse; Gaussian N_out/N = exp(-n/2)); "
    "x, y dispersion-corrected (betatron), x_raw, y_raw as tracked, "
    "z = (phi, W)")
EPS_UNITS = ("eps_x, eps_y, eps_x_raw, eps_y_raw: mm*mrad (geometric); "
             "eps_z: deg*MeV; eps_n_*: mm*mrad (normalized)")

# Levels drawn by the GUI, in % of the particles present at that step.
LEVELS_PCT = (1.0, 0.1, 0.01)


def default_action_grid() -> np.ndarray:
    """The ellipse-size grid, in units of eps_rms: 0-50 in steps of 0.25
    (resolves the core and reaches 7 sigma), then 51-400 in steps of 1
    (20 sigma) — 551 points."""
    return np.unique(np.concatenate([np.arange(0.0, 50.0 + 1e-9, 0.25),
                                     np.arange(51.0, 400.0 + 1e-9, 1.0)]))


def validate_grid(grid) -> np.ndarray:
    """A float copy of ``grid``; raises ValueError unless it is 1-D,
    finite, non-negative and strictly increasing."""
    g = np.array(grid, dtype=float, copy=True)
    if g.ndim != 1 or g.size == 0:
        raise ValueError("action-scan grid must be a non-empty 1-D array")
    if not np.all(np.isfinite(g)) or np.any(g < 0.0):
        raise ValueError("action-scan grid must be finite and >= 0")
    if np.any(np.diff(g) <= 0.0):
        raise ValueError("action-scan grid must be strictly increasing")
    return g


def gaussian_percent_outside(n):
    """% of a Gaussian beam outside the ellipse of size n*eps_rms."""
    return 100.0 * np.exp(-0.5 * np.asarray(n, dtype=float))


def gaussian_level_n(pct: float) -> float:
    """Ellipse size n beyond which ``pct`` % of a Gaussian beam lies."""
    return 2.0 * math.log(100.0 / float(pct))


def counts_above(k, grid) -> np.ndarray:
    """``int32`` array: for every grid value n_j the number of ``k``
    values STRICTLY greater than n_j.  O(N log G) (searchsorted +
    bincount), no sort of ``k``."""
    g = np.asarray(grid, dtype=float)
    k = np.asarray(k, dtype=float).ravel()
    if k.size == 0:
        return np.zeros(g.size, dtype=np.int32)
    # idx_i = number of grid values < k_i, i.e. particle i is outside
    # the ellipses j = 0 .. idx_i - 1.
    idx = np.searchsorted(g, k, side="left")
    at_most = np.cumsum(np.bincount(idx, minlength=g.size + 1))[:g.size]
    return (k.size - at_most).astype(np.int32)


def _moments(p, mean, sigma):
    if mean is None or sigma is None:
        from linac_gen.diagnostics.moments import compute_moments
        m = compute_moments(p)
        mean, sigma = m["mean"], m["sigma_matrix"]
    return np.asarray(mean, dtype=float), np.asarray(sigma, dtype=float)


def _plane_sizes(p, plane, mean, sigma, centred):
    """(k, eps) for one plane; k is None when the plane has no rms
    ellipse (eps 0, or NaN when its moments are not finite).  ``centred``
    caches the mean-subtracted columns across planes."""
    def _col(c):
        if c not in centred:
            centred[c] = p[:, c] - mean[c]
        return centred[c]

    i, j = _COLS[plane]
    s11, s12, s22 = float(sigma[i, i]), float(sigma[i, j]), float(sigma[j, j])
    u, up = _col(i), _col(j)
    s55 = float(sigma[5, 5])
    if plane in _BETATRON and not (math.isfinite(s55)
                                   and math.isfinite(float(sigma[i, 5]))
                                   and math.isfinite(float(sigma[j, 5]))):
        # A non-finite energy moment must not silently turn the betatron
        # plane into the raw one (the S_55 gate below is False for NaN).
        return None, float("nan")
    if plane in _BETATRON and s55 > _S55_EPS:
        s15, s25 = float(sigma[i, 5]), float(sigma[j, 5])
        a, b = s15 / s55, s25 / s55
        s11 -= s15 * a
        s12 -= s15 * b
        s22 -= s25 * b
        d = _col(5)
        u = u - a * d
        up = up - b * d
    det = s11 * s22 - s12 * s12
    if not math.isfinite(det):
        return None, float("nan")
    if not (det > 0.0 and det > _DET_REL_TOL * s11 * s22):
        return None, 0.0
    k = (s22 * u * u - 2.0 * s12 * u * up + s11 * up * up) / det
    return k, math.sqrt(det)


def ellipse_sizes(particles, plane: str = "x", *, mean=None, sigma=None):
    """Per-particle ellipse size n_i = W_i/eps_rms in ``plane`` (one of
    :data:`PLANES`) and the plane's eps_rms.  ``<n> = 2`` identically.

    Returns ``(n, eps)``; ``n`` is ``None`` when the plane has no rms
    ellipse (fewer than 3 particles, a zero or rank-1 plane: eps = 0;
    non-finite moments: eps = NaN).  ``particles`` is never modified."""
    if plane not in _COLS:
        raise ValueError(f"plane must be one of {PLANES}, got {plane!r}")
    p = np.asarray(particles, dtype=float)
    if p.ndim != 2 or p.shape[0] < _MIN_PARTICLES:
        return None, 0.0
    mean, sigma = _moments(p, mean, sigma)
    return _plane_sizes(p, plane, mean, sigma, {})


def action_scan_row(particles, grid, *, mean=None, sigma=None,
                    planes=PLANES) -> dict:
    """Counts outside the n*eps_rms ellipses for one step.

    ``particles`` is the (N, 6) array of ALIVE particles (never
    modified).  ``mean``/``sigma`` may pass the step's first and second
    moments (1/N convention, e.g. ``compute_moments``) to avoid
    recomputing them.

    Returns ``{"n_alive": N, "count_<p>": int32 (G,), "eps_<p>": float,
    "n_max_<p>": float}`` for every plane p.  ``n_max`` is the outermost
    particle's n.  A plane without an rms ellipse (fewer than 3
    particles, zero or rank-1 plane) gives zero counts, eps = 0 and
    n_max = 0 (eps = NaN when its moments are not finite).
    """
    g = np.asarray(grid, dtype=float)
    p = np.asarray(particles, dtype=float)
    n_alive = int(p.shape[0]) if p.ndim == 2 else 0
    row: dict = {"n_alive": n_alive}
    if n_alive < _MIN_PARTICLES:
        for plane in planes:
            row[f"count_{plane}"] = np.zeros(g.size, dtype=np.int32)
            row[f"eps_{plane}"] = 0.0
            row[f"n_max_{plane}"] = 0.0
        return row
    mean, sigma = _moments(p, mean, sigma)
    centred: dict = {}
    for plane in planes:
        k, eps = _plane_sizes(p, plane, mean, sigma, centred)
        if k is None:
            row[f"count_{plane}"] = np.zeros(g.size, dtype=np.int32)
            row[f"n_max_{plane}"] = 0.0
        else:
            row[f"count_{plane}"] = counts_above(k, g)
            row[f"n_max_{plane}"] = float(np.max(k))
        row[f"eps_{plane}"] = eps
    return row


# ---------------------------------------------------------------------------
# Reading a recorded scan back (live recorder or loaded HDF5 results)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ActionScan:
    """A recorded scan, as arrays.  ``s`` in mm (recorder convention)."""
    n: np.ndarray                         # (G,) ellipse sizes, eps_rms units
    s: np.ndarray                         # (S,) mm
    n_alive: np.ndarray                   # (S,)
    counts: dict                          # plane -> (S, G) int32
    eps: dict                             # plane -> (S,) geometric
    eps_n: dict                           # plane -> (S,) mm*mrad normalized
    n_max: dict                           # plane -> (S,)
    continuous: np.ndarray                # (S,) bool — DC records
    element: tuple = field(default=())    # (S,) element names, or ()
    # The run tracked with periodic phase coordinates (a DC-bunched beam
    # folded to one bunch period); None when unknown.
    periodic_phase: object = None

    @property
    def planes(self) -> tuple:
        return tuple(p for p in PLANES if p in self.counts)

    def percent_outside(self, plane: str) -> np.ndarray:
        """(S, G) % of the particles present at each step lying outside
        each ellipse; NaN rows where the step has no rms ellipse (see
        ``_undefined``: includes DC records in z)."""
        c = self.counts[plane].astype(float)
        na = self.n_alive.astype(float)
        with np.errstate(invalid="ignore", divide="ignore"):
            pct = 100.0 * c / na[:, None]
        pct[self._undefined(plane), :] = np.nan
        return pct

    def level_crossing(self, plane: str, pct: float) -> np.ndarray:
        """(S,) smallest grid n with FEWER than ``pct`` % of the present
        particles outside it.  NaN where the step has no ellipse, where
        the level is below one particle (not resolved by the sample), or
        where more than ``pct`` % lie beyond the whole grid."""
        c = self.counts[plane]
        level = float(pct) / 100.0 * self.n_alive.astype(float)
        below = c < level[:, None]
        has = below.any(axis=1)
        out = np.where(has, self.n[below.argmax(axis=1)], np.nan)
        out[self._undefined(plane) | (level < 1.0)] = np.nan
        return out

    def _undefined(self, plane: str) -> np.ndarray:
        """Steps without an rms ellipse in ``plane``: no particles, a
        degenerate or non-finite plane, and — for z — DC (unbunched)
        records, whose φ–W has no bunch to measure against."""
        eps = np.asarray(self.eps[plane], dtype=float)
        bad = (self.n_alive <= 0) | ~(eps > 0.0)
        if plane == "z":
            bad = bad | self.continuous
        return bad


def _as_list(v):
    if v is None:
        return None
    if isinstance(v, np.ndarray):
        return v
    return list(v)


def action_scan_from_results(results) -> "ActionScan | None":
    """The scan carried by ``results`` — a live ``DiagnosticRecorder``
    (``results.action_scan`` holds per-step lists) or loaded HDF5 results
    (``load_results_hdf5`` dict, or the GUI's attribute adapter; the scan
    is a dict of arrays).  ``None`` when the run did not record one.
    Raises ``ValueError`` when the scan is not aligned with ``results.s``.
    """
    if isinstance(results, dict):
        scan = results.get("action_scan")
        get = results.get
    else:
        scan = getattr(results, "action_scan", None)

        def get(name, default=None):
            return getattr(results, name, default)
    if not scan or "n" not in scan:
        return None
    n = np.asarray(scan["n"], dtype=float)
    s_raw = get("s")
    s = np.asarray([] if s_raw is None else s_raw, dtype=float)
    n_steps = s.size
    counts, eps, eps_n, n_max = {}, {}, {}, {}
    for p in PLANES:
        key = f"count_{p}"
        if key not in scan:
            continue
        c = np.asarray(scan[key], dtype=np.int32)
        c = c.reshape(-1, n.size) if c.size else np.zeros((0, n.size), np.int32)
        counts[p] = c
        eps[p] = np.asarray(scan.get(f"eps_{p}", []), dtype=float)
        eps_n[p] = np.asarray(scan.get(f"eps_n_{p}", []), dtype=float)
        n_max[p] = np.asarray(scan.get(f"n_max_{p}", []), dtype=float)
    if not counts:
        return None
    n_alive = np.asarray(scan.get("n_alive", []), dtype=np.int64)
    per_step = [n_alive] + [a for d in (counts, eps, eps_n, n_max)
                            for a in d.values()]
    bad = [len(a) for a in per_step if len(a) != n_steps]
    if bad:
        raise ValueError(
            f"action scan has {sorted(set(bad))} rows but the results have "
            f"{n_steps} recorded steps")
    cont = _as_list(scan.get("continuous"))
    if cont is None:
        cont = _as_list(get("continuous_at"))
    continuous = (np.asarray(cont, dtype=bool) if cont is not None
                  and len(cont) == n_steps else np.zeros(n_steps, dtype=bool))
    names = _as_list(scan.get("element"))
    if names is None:
        names = _as_list(get("element_names"))
    element = (tuple(str(x) for x in names)
               if names is not None and len(names) == n_steps else ())
    periodic = get("periodic_phase")
    if periodic is None:
        periodic = (scan.get("attrs") or {}).get("periodic_phase")
    return ActionScan(n=n, s=s, n_alive=n_alive, counts=counts, eps=eps,
                      eps_n=eps_n, n_max=n_max, continuous=continuous,
                      element=element,
                      periodic_phase=(None if periodic is None
                                      else bool(periodic)))
