"""The independent terms of the beam's 6×6 second-moment matrix along a run
(the Results tab's "Σ matrix along s" panel).  Pure functions, no Qt.

Every row of a run records Σ in ``results.sigma_matrix``: for a
multiparticle run the centred covariance (1/N) of the particles the
statistics describe, for an envelope run the propagated matrix.  Order and
units: (x mm, x′ mrad, y mm, y′ mrad, Δφ deg of the row's RF, ΔW MeV).  Σ is
symmetric, so 21 of its 36 entries are independent: six rms values and
fifteen correlations r_ij = Σ_ij / √(Σ_ii Σ_jj) (or covariances).

* :func:`sigma_terms` -- the terms along the run, in that basis or in
  TraceWin's (x, x′, y, y′, z, δ), converted row by row with each row's
  reference particle and RF frequency (Δφ changes units at a frequency
  jump; z and δ do not);
* :func:`largest_coupling` -- the largest |r| between two planes, row by row;
* :func:`plane_summary` -- rms values, emittance, α and β of each plane at
  one row.

Undefined values are NaN, never 0: correlations with a zero variance (no
particles left, a beam without energy spread), every term of a row with no
particles, and the phase terms of rows where the beam is continuous.
z = −k·Δφ, so in TraceWin's basis the correlations with z change sign.
"""
from __future__ import annotations

import math
import warnings
from dataclasses import dataclass

import numpy as np

from linac_gen.core.constants import C_LIGHT

BASES = ("ours", "tracewin")
COORDS = {"ours": ("x", "x′", "y", "y′", "Δφ", "ΔW"),
          "tracewin": ("x", "x′", "y", "y′", "z", "δ")}
#: display units of each coordinate; δ in ‰ (= mrad, as TraceWin's z′)
UNITS = {"ours": ("mm", "mrad", "mm", "mrad", "deg", "keV"),
         "tracewin": ("mm", "mrad", "mm", "mrad", "mm", "‰")}
#: recorded (ours) or converted (TraceWin: z in m, δ) units → display units
_SCALE = {"ours": np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1e3]),
          "tracewin": np.array([1.0, 1.0, 1.0, 1.0, 1e3, 1e3])}
PLANE = (0, 0, 1, 1, 2, 2)
PLANE_NAMES = ("x", "y", "z")
#: the 21 independent entries, lower triangle row by row: (i, j), j <= i
TERMS = tuple((i, j) for i in range(6) for j in range(i + 1))
#: the 12 entries that couple two planes
COUPLING = tuple((i, j) for i, j in TERMS if PLANE[i] != PLANE[j])


@dataclass
class SigmaTerms:
    """Σ along a run in one basis; arrays have one entry per recorded row."""

    basis: str
    s: np.ndarray            # (n,) lattice position, m
    sigma: np.ndarray        # (n, 6, 6) display units (``units``)
    rms: np.ndarray          # (n, 6) √Σ_ii, display units
    corr: np.ndarray         # (n, 6, 6) correlations r_ij
    continuous: np.ndarray   # (n,) bool: the beam is not bunched on this row
    element: list            # (n,) element name of each row, '' if unknown
    converted: bool          # False: TraceWin basis asked, no reference data
    mass_mev: float
    # "s": ``s`` is the position in m; "turn": a ring run turn by turn, ``s``
    # the turn of each row
    domain: str = "s"

    @property
    def coords(self) -> tuple:
        return COORDS[self.basis]

    @property
    def units(self) -> tuple:
        return UNITS[self.basis]

    def label(self, i: int, j: int) -> str:
        """``σ(x)`` on the diagonal, ``r(x, x′)`` off it."""
        c = self.coords
        return f"σ({c[i]})" if i == j else f"r({c[j]}, {c[i]})"

    def covariance_label(self, i: int, j: int) -> str:
        c = self.coords
        return f"⟨{c[j]} {c[i]}⟩"

    def covariance_units(self, i: int, j: int) -> str:
        u = self.units
        return f"{u[i]}²" if i == j else f"{u[j]}·{u[i]}"

    def __len__(self) -> int:
        return int(self.s.size)


# ---------------------------------------------------------------------------
def sigma_stack(results) -> np.ndarray | None:
    """The recorded matrices as an (n, 6, 6) float array, or None (no
    results, no Σ recorded, or not one per row of ``results.s``).  Accepts
    the lists of a live run and of a results file read back."""
    if results is None:
        return None
    mats = getattr(results, "sigma_matrix", None)
    if mats is None or len(mats) == 0:
        return None
    try:
        S = np.asarray(mats, dtype=float)
    except (TypeError, ValueError):
        return None
    if S.ndim != 3 or S.shape[1:] != (6, 6):
        return None
    s = getattr(results, "s", None)
    if s is None or len(s) != S.shape[0]:
        return None
    return S


def continuous_rows(results, S: np.ndarray) -> np.ndarray:
    """Rows where the beam is continuous (no bunch, so no phase moments).

    Multiparticle runs flag each row (``continuous_at``).  An envelope run
    has one flag for the whole run; its longitudinal block is zero until
    the first RF element bunches the beam, so those rows are the ones with
    Σ_ΔφΔφ = 0."""
    n = S.shape[0]
    flags = getattr(results, "continuous_at", None)
    if flags is not None and len(flags) == n:
        return np.asarray(flags, dtype=bool)
    if bool(getattr(results, "continuous", False)):
        return ~(S[:, 4, 4] > 0.0)
    return np.zeros(n, dtype=bool)


def correlations(S: np.ndarray) -> np.ndarray:
    """r_ij = Σ_ij / √(Σ_ii Σ_jj) for (n, 6, 6) matrices; NaN where a
    variance is not positive."""
    d = np.diagonal(S, axis1=-2, axis2=-1)
    with np.errstate(invalid="ignore", divide="ignore"):
        sd = np.where(d > 0.0, np.sqrt(np.where(d > 0.0, d, 1.0)), np.nan)
        r = S / (sd[..., :, None] * sd[..., None, :])
    return np.clip(r, -1.0, 1.0)


def rms(S: np.ndarray) -> np.ndarray:
    """√Σ_ii for (n, 6, 6) matrices; NaN for a negative (unphysical) one."""
    d = np.diagonal(S, axis1=-2, axis2=-1)
    with np.errstate(invalid="ignore"):
        return np.where(d >= 0.0, np.sqrt(np.where(d >= 0.0, d, 0.0)), np.nan)


def to_tracewin(S: np.ndarray, beta, gamma, freq_mhz, mass_mev: float) -> np.ndarray:
    """Σ in TraceWin's (x, x′, y, y′, z [m], δ) basis, row by row.

    Same transformation as :func:`~linac_gen.tracking.longitudinal_coords.
    sigma_to_tracewin_custom`: Δφ = k_φ·z with k_φ = −360/(βλ), ΔW = k_W·δ
    with k_W = β²γ·mc².  Rows without a usable reference (β, f ≤ 0 or not
    finite) get NaN in the z and δ rows and columns."""
    n = S.shape[0]
    beta = np.broadcast_to(np.asarray(beta, dtype=float), (n,))
    gamma = np.broadcast_to(np.asarray(gamma, dtype=float), (n,))
    freq = np.broadcast_to(np.asarray(freq_mhz, dtype=float), (n,))
    with np.errstate(invalid="ignore", divide="ignore"):
        wavelength_m = C_LIGHT / (freq * 1e6)
        k_phi = -360.0 / (beta * wavelength_m)
        k_w = beta * beta * gamma * float(mass_mev)
    ok = (beta > 0.0) & (freq > 0.0) & np.isfinite(k_phi) & np.isfinite(k_w) \
        & (k_w != 0.0)
    t = np.ones((n, 6))
    t[:, 4] = np.where(ok, k_phi, np.nan)
    t[:, 5] = np.where(ok, k_w, np.nan)
    return S / (t[:, :, None] * t[:, None, :])


def _per_row(results, name: str, n: int) -> np.ndarray | None:
    v = getattr(results, name, None)
    if v is None:
        return None
    a = np.asarray(v, dtype=float).ravel()
    if a.size == n:
        return a
    if a.size == 1:                       # legacy files: one value per run
        return np.full(n, float(a[0]))
    return None


def _mass(results) -> float:
    """Rest energy in MeV: the run's ``mass_mev``, else from the first
    row's reference (W = (γ − 1)·mc²)."""
    m = float(getattr(results, "mass_mev", 0.0) or 0.0)
    if m > 0.0:
        return m
    g = getattr(results, "ref_gamma", None)
    w = getattr(results, "ref_w_kin", None)
    if g is not None and w is not None and len(g) and len(w):
        g0, w0 = float(g[0]), float(w[0])
        if g0 > 1.0 + 1e-12:
            return w0 / (g0 - 1.0)
    return 0.0


def sigma_terms(results, basis: str = "ours") -> SigmaTerms | None:
    """The 21 independent terms of Σ at every recorded row, or None when
    the results carry no Σ (see :func:`sigma_stack`)."""
    if basis not in BASES:
        raise ValueError(f"basis must be one of {BASES}, not {basis!r}")
    S = sigma_stack(results)
    if S is None:
        return None
    n = S.shape[0]
    # both read the matrices as recorded, before any conversion adds NaN
    empty = ~np.any(S != 0.0, axis=(1, 2))         # a row with no particles
    dc = continuous_rows(results, S)
    turn = getattr(results, "x_domain", "s") == "turn"
    # (results record s in mm; a ring run's rows are turns)
    s_m = np.asarray(getattr(results, "s"), dtype=float) * (1.0 if turn else 1e-3)
    mass = _mass(results)
    converted = True
    if basis == "tracewin":
        beta = _per_row(results, "ref_beta", n)
        gamma = _per_row(results, "ref_gamma", n)
        freq = _per_row(results, "ref_frequency", n)
        if beta is None or gamma is None or freq is None or mass <= 0.0:
            S = S.copy()
            S[:, 4:, :] = np.nan
            S[:, :, 4:] = np.nan
            converted = False
        else:
            S = to_tracewin(S, beta, gamma, freq, mass)
    scale = _SCALE[basis]
    S = S * scale[None, :, None] * scale[None, None, :]
    S[empty] = np.nan
    S[dc, 4, :] = np.nan
    S[dc, :, 4] = np.nan

    names = getattr(results, "element_names", None)
    element = list(names) if names is not None and len(names) == n else [""] * n
    return SigmaTerms(basis=basis, s=s_m, sigma=S, rms=rms(S),
                      corr=correlations(S), continuous=dc, element=element,
                      converted=converted, mass_mev=mass,
                      domain="turn" if turn else "s")


def largest_coupling(results) -> np.ndarray | None:
    """Row by row, the largest |r| among the 12 entries that couple two
    planes (NaN where none is defined); None without Σ."""
    terms = sigma_terms(results, "ours")
    if terms is None:
        return None
    vals = np.stack([np.abs(terms.corr[:, i, j]) for i, j in COUPLING], axis=1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)     # all-NaN rows
        return np.nanmax(vals, axis=1)


def plane_twiss(s11: float, s12: float, s22: float) -> tuple[float, float, float]:
    """(ε, α, β) of a 2×2 block; NaN when it has no area."""
    det = s11 * s22 - s12 * s12
    if not (np.isfinite(det) and det > 0.0):
        return (math.nan, math.nan, math.nan)
    eps = math.sqrt(det)
    return (eps, -s12 / eps, s11 / eps)


def plane_summary(terms: SigmaTerms, row: int) -> list[dict]:
    """Per plane at one row: the two rms values (display units) and the
    rms emittance, α and β of the plane's 2×2 block.  Transverse planes and
    TraceWin's z plane use the display units (mm·mrad, mm/mrad; z in mm, δ
    in ‰ = mrad); the (Δφ, ΔW) plane uses deg·MeV and deg/MeV, as HELIX
    reports it elsewhere."""
    S = terms.sigma[row]
    out = []
    for p in range(3):
        a, b = 2 * p, 2 * p + 1
        s11, s12, s22 = float(S[a, a]), float(S[a, b]), float(S[b, b])
        if p == 2 and terms.basis == "ours":            # ΔW keV → MeV
            s12, s22 = s12 * 1e-3, s22 * 1e-6
            e_units, b_units = "deg·MeV", "deg/MeV"
        else:
            e_units, b_units = "mm·mrad", "mm/mrad"
        eps, alpha, beta = plane_twiss(s11, s12, s22)
        out.append(dict(plane=PLANE_NAMES[p],
                        rms=(float(terms.rms[row, a]), float(terms.rms[row, b])),
                        coords=(terms.coords[a], terms.coords[b]),
                        units=(terms.units[a], terms.units[b]),
                        eps=eps, alpha=alpha, beta=beta,
                        eps_units=e_units, beta_units=b_units))
    return out
