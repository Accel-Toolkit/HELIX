"""PIC space-charge solver: orchestrates the full kick cycle.

Each call to :meth:`kick` performs:
1. Extract spatial coordinates from beam
2. Lorentz boost to rest frame
3. Set up / reuse the PIC grid
4. Compute macro-particle charges from beam current
5. Deposit charge with CIC
6. Solve Poisson equation (FFT) in the rest frame
7. Interpolate E-fields back to particles
8. Apply transverse and longitudinal momentum kicks
"""
import logging

import numpy as np

from linac_gen.pic.coordinates import beam_to_spatial
from linac_gen.pic.lorentz_boost import boost_to_rest_frame
from linac_gen.pic.charge_deposition import (
    deposit_cic as _py_deposit_cic,
    deposit_tsc as _py_deposit_tsc,
)
from linac_gen.pic.poisson_solver import PoissonSolverFFT
from linac_gen.pic.field_interpolation import (
    interpolate_cic as _py_interpolate_cic,
    interpolate_tsc as _py_interpolate_tsc,
)
from linac_gen.core.config import SpaceChargeConfig
from linac_gen.core.constants import E_CHARGE, C_LIGHT

_log = logging.getLogger(__name__)

# Preload torch (if installed) BEFORE loading our C++ kernel.  Both link
# against libomp.dylib; on macOS, torch bundles its own copy and our kernel
# links the same one by install_name.  If our kernel loads first, it pulls
# in the libomp image; when torch later imports for the MPS backend it
# brings in a *second* libomp image and OpenMP errors with "libomp.dylib
# already initialized".  Importing torch first ensures dyld de-duplicates
# to a single image.  Silently no-op when torch is unavailable.
try:  # noqa: SIM105
    import torch  # type: ignore  # noqa: F401
except Exception:
    pass

# Reproducibility note: OMP_DYNAMIC is pinned FALSE in
# linac_gen/__init__.py — it must be set BEFORE the torch import above
# (torch initializes the shared libomp, which reads the env var once),
# so setting it here would be too late.  Results still differ ~1e-10
# ACROSS different OMP_NUM_THREADS settings — documented in the manual.

try:
    from linac_gen._pic_kernels import (
        deposit_cic as _cpp_deposit,
        interpolate_cic as _cpp_interpolate,
    )
    _USE_CPP = True
    _log.info("PIC: using C++ kernels (linac_gen._pic_kernels)")
except ImportError:
    _USE_CPP = False
    _log.warning(
        "PIC: C++ kernels unavailable (linac_gen._pic_kernels not importable); "
        "falling back to slower Python implementation. Rebuild the extension "
        "with `pip install -e .` to regain C++ speed."
    )

# CIC: use C++ when available; TSC has no C++ kernel yet so always uses Python.
deposit_cic = _cpp_deposit if _USE_CPP else _py_deposit_cic
interpolate_cic = _cpp_interpolate if _USE_CPP else _py_interpolate_cic
deposit_tsc = _py_deposit_tsc
interpolate_tsc = _py_interpolate_tsc


def _select_kernel(kind: str):
    """Return the (deposit, interpolate) pair for the given kernel kind."""
    if kind == "tsc":
        return deposit_tsc, interpolate_tsc
    return deposit_cic, interpolate_cic


# ---------------------------------------------------------------------------
# Analytic 2-D continuous-beam (DC) space-charge kick
#
# For a continuous (unbunched) beam the longitudinal SC force is zero and the
# transverse SC field of a uniformly-charged elliptical beam-pipe cross
# section is (lab frame, dropping common factors):
#
#     E_x(x, y) = K · x / (a · (a + b))
#     E_y(x, y) = K · y / (b · (a + b))
#
# with generalised perveance ``K = 2 · q · I / (4πε₀ · m₀c³ · β³γ³)`` and
# semi-axes a = 2σ_x, b = 2σ_y of the uniform-density equivalent ellipse.
# This is TraceWin's continuous-beam formula (manual §"Continuous beam").
# Apply as momentum kicks:
#
#     Δx'[rad]  = (q · E_x / (γ · m₀c²)) · Δs        (in consistent units)
#     Δy'[rad]  = analogous
#
# No ΔW kick (no longitudinal SC force in a DC beam).
# ---------------------------------------------------------------------------
def kick_continuous_2d(beam, ds_mm: float) -> None:
    """Apply one 2-D analytic DC space-charge kick over ``ds_mm``.

    Parameters
    ----------
    beam : Beam
        Must have ``beam.continuous == True``; modified in place.
    ds_mm : float
        Step length in millimetres.
    """
    if beam.current == 0 or beam.n_alive < 2 or ds_mm <= 0:
        return

    alive = beam.alive_mask
    xs = beam.particles[alive, 0]   # mm
    ys = beam.particles[alive, 2]   # mm
    # The beam's SELF-field is centred on the beam's own centroid — the
    # analytic formulas below take positions RELATIVE to the charge
    # distribution.  Evaluating them at raw (axis-relative) coordinates
    # gave a displaced beam a spurious coherent self-deflection
    # (momentum non-conserving; 2026-07-25 review).  The 2-D PIC DC
    # kernel was always centred (beam-centred grid); these two now match.
    cx = float(np.mean(xs))
    cy = float(np.mean(ys))
    # RMS sizes in metres.  Uniform-density equivalent semi-axes a, b = 2σ.
    sigma_x_m = float(np.std(xs)) * 1e-3
    sigma_y_m = float(np.std(ys)) * 1e-3
    if sigma_x_m <= 0 or sigma_y_m <= 0:
        return
    a_m = 2.0 * sigma_x_m
    b_m = 2.0 * sigma_y_m
    ab_sum = a_m + b_m

    ref = beam.ref
    beta  = float(ref.beta)
    gamma = float(ref.gamma)
    mass_MeV = float(ref.species.mass)
    q_abs = abs(ref.species.charge) * E_CHARGE
    # Loss-scaled current — the macrocharge convention (pic/macrocharge.py):
    # every LAUNCHED macroparticle carries a fixed share of the configured
    # current, so the transported current decays with transmission.  The
    # bunched 3-D PIC always did this; the DC kernels used the configured
    # current outright, overdriving the field by 1/transmission after
    # scraping (PXIE LEBT at 77 % transmission: σ_x +16 % at the SOL2 exit
    # vs TW partran; loss-scaled it agrees to ~1 %, 2026-08-01).  For a
    # lossless beam the fraction is exactly 1.0 → bit-identical.
    I_A = (abs(beam.current) * 1e-3) * (beam.n_alive / beam.n_particles)
    # Generalised perveance K = 2·q·I / (4πε₀·m₀c³·β³γ³) — SI (N/m per mm⁻¹).
    # Using 4πε₀ implicitly: E_x = (2/4πε₀) · I / (v·a·(a+b)) · x
    # where v = β·c.  Simpler: use constant K in terms of (E/r) per unit r.
    # We'll compute directly:
    #   E_x[V/m] = I / (2πε₀·v·a·(a+b)) · x[m]
    EPS0 = 8.8541878128e-12                             # F/m
    v_m_s = beta * C_LIGHT
    if v_m_s <= 0:
        return
    # Uniform elliptic-cylinder SC field (e.g. Wiedemann §3.3):
    #   E_x = λ·x / (π·ε₀·a·(a+b))
    #   E_y = λ·y / (π·ε₀·b·(a+b))
    # with line charge density λ = I/v.  In the round-beam limit (a=b=r)
    # this correctly reduces to E_r = λ·r/(2πε₀·r²).
    factor = I_A / (np.pi * EPS0 * v_m_s * ab_sum)      # [V/m per m]
    k_x = factor / a_m                                  # [V/m per m of x]
    k_y = factor / b_m

    # Convert E-field → x' kick:
    #   Δx'[rad] = (q·E·Δs) / (β²γ · m₀c²[J])
    # In our mm/mrad convention, x[mm]×1e-3 → m, then Δx'[rad] = Δx' = ...;
    # since 1 rad == 1 mrad numerically after unit cancellation (both sides
    # scaled by 1e-3), we write the formula with positions in metres and
    # obtain Δx' directly in rad == mrad.
    mc2_J = mass_MeV * 1e6 * E_CHARGE                   # rest energy in J
    ds_m = ds_mm * 1e-3
    pre  = q_abs * ds_m / (beta * beta * gamma * mc2_J)  # rad per V/m per m-of-r
    # Kick per particle:
    #   Δx'[rad] = pre · E_x[V/m] = pre · k_x · x[m]
    alive_idx = np.where(alive)[0]
    xs_m = (xs - cx) * 1e-3          # centroid-relative (self-field)
    ys_m = (ys - cy) * 1e-3
    beam.particles[alive_idx, 1] += (pre * k_x * xs_m) * 1e3   # rad → mrad
    beam.particles[alive_idx, 3] += (pre * k_y * ys_m) * 1e3


# ---------------------------------------------------------------------------
# Bassetti-Erskine 2-D Gaussian DC SC kick.  Bassetti & Erskine, CERN-ISR-
# TH/80-06 (1980).  For a Gaussian charge density of σ_x, σ_y the transverse
# E-field has a closed form involving the Faddeeva function w(z) = exp(-z²)
# erfc(-iz).  For σ_x ≈ σ_y the formula degenerates; we fall back to the
# round-beam limit (1 − exp(−r²/2σ²))/r² in that branch.
# ---------------------------------------------------------------------------
def _gauss_field_2d(xs_m, ys_m, sigma_x_m: float, sigma_y_m: float,
                    lam: float):
    """(E_x, E_y) [V/m] of a 2-D Gaussian line-charge distribution.

    Round-beam analytic limit below 5 % asymmetry, Bassetti-Erskine
    otherwise — factored out of :func:`kick_continuous_2d_gauss` so the
    frozen-field tune-footprint kicker replays EXACTLY the same field
    formula with taped σ values.
    """
    from scipy.special import wofz
    EPS0 = 8.8541878128e-12

    # Asymmetry threshold: relative |σ_x − σ_y| / max(σ) < 0.05 → round beam.
    # Tighter than 5 % triggers Bassetti-Erskine numerical instability where
    # the (w(z1) − decay·w(z2)) cancellation loses precision.  5 % is well
    # within accuracy of the round-beam approximation for our purposes.
    if abs(sigma_x_m - sigma_y_m) / max(sigma_x_m, sigma_y_m) < 0.05:
        sigma2 = 0.5 * (sigma_x_m**2 + sigma_y_m**2)
        r2 = xs_m * xs_m + ys_m * ys_m
        # (1 - exp(-r²/2σ²))/r² with limit 1/(2σ²) at r→0.
        with np.errstate(invalid="ignore", divide="ignore"):
            ratio = np.where(
                r2 > 1e-30,
                (1.0 - np.exp(-r2 / (2.0 * sigma2))) / r2,
                1.0 / (2.0 * sigma2),
            )
        Ex = (lam / (2.0 * np.pi * EPS0)) * ratio * xs_m
        Ey = (lam / (2.0 * np.pi * EPS0)) * ratio * ys_m
    else:
        # Bassetti-Erskine.  Convention: assume σ_x > σ_y; if not, swap.
        if sigma_y_m > sigma_x_m:
            sx, sy = sigma_y_m, sigma_x_m
            x_eff, y_eff = ys_m, xs_m
            swap = True
        else:
            sx, sy = sigma_x_m, sigma_y_m
            x_eff, y_eff = xs_m, ys_m
            swap = False
        K = 2.0 * (sx * sx - sy * sy)
        sqrtK = np.sqrt(K)
        z1 = (x_eff + 1j * y_eff) / sqrtK
        z2 = ((sy / sx) * x_eff + 1j * (sx / sy) * y_eff) / sqrtK
        decay = np.exp(-x_eff * x_eff / (2.0 * sx * sx)
                       - y_eff * y_eff / (2.0 * sy * sy))
        # Bassetti & Erskine: E_y + i·E_x = (λ/2ε₀√(πK)) [w(z1) − decay·w(z2)]
        psi = (lam / (2.0 * EPS0 * np.sqrt(np.pi * K))) * (
            wofz(z1) - decay * wofz(z2)
        )
        Ey_eff = psi.real
        Ex_eff = psi.imag
        if swap:
            Ex, Ey = Ey_eff, Ex_eff
        else:
            Ex, Ey = Ex_eff, Ey_eff
    return Ex, Ey


def kick_continuous_2d_gauss(beam, ds_mm: float) -> None:
    """Apply one Bassetti-Erskine DC space-charge kick over ``ds_mm``.

    Uses the per-particle non-linear field of a 2-D Gaussian charge
    distribution with σ_x, σ_y measured from the alive particles.  This is
    the closest analytic equivalent to TraceWin's PICNIC_2D solver and
    captures the core-vs-tail field non-linearity that the linear uniform-
    elliptical-cylinder kick (``kick_continuous_2d``) cannot.

    Parameters
    ----------
    beam : Beam
        Must have ``beam.continuous == True``; modified in place.
    ds_mm : float
        Step length in millimetres.
    """
    if beam.current == 0 or beam.n_alive < 2 or ds_mm <= 0:
        return
    alive = beam.alive_mask
    alive_idx = np.where(alive)[0]
    xs = beam.particles[alive, 0]   # mm
    ys = beam.particles[alive, 2]
    # Self-field is centroid-centred — see kick_continuous_2d (2026-07-25).
    cx = float(np.mean(xs))
    cy = float(np.mean(ys))
    sigma_x_m = float(np.std(xs)) * 1e-3
    sigma_y_m = float(np.std(ys)) * 1e-3
    if sigma_x_m <= 0 or sigma_y_m <= 0:
        return

    ref = beam.ref
    beta  = float(ref.beta)
    gamma = float(ref.gamma)
    mass_MeV = float(ref.species.mass)
    q_abs = abs(ref.species.charge) * E_CHARGE
    # Loss-scaled current — macrocharge convention; see kick_continuous_2d.
    I_A = (abs(beam.current) * 1e-3) * (beam.n_alive / beam.n_particles)
    v_m_s = beta * C_LIGHT
    if v_m_s <= 0:
        return
    lam = I_A / v_m_s                     # line charge density [C/m]

    Ex, Ey = _gauss_field_2d((xs - cx) * 1e-3, (ys - cy) * 1e-3,
                             sigma_x_m, sigma_y_m, lam)

    mc2_J = mass_MeV * 1e6 * E_CHARGE
    ds_m = ds_mm * 1e-3
    pre = q_abs * ds_m / (beta * beta * gamma * mc2_J)
    beam.particles[alive_idx, 1] += (pre * Ex) * 1e3
    beam.particles[alive_idx, 3] += (pre * Ey) * 1e3


# ---------------------------------------------------------------------------
# 2-D Hockney FFT PIC for continuous (DC) beams.  Mirrors what TraceWin's
# PICNIC_2D does: deposit particles into an (x, y) grid, solve 2-D open-BC
# Poisson via a doubled-grid FFT with the 2-D Coulomb Green's function
# G(x, y) = -ln(r)/(2πε₀), gather E-field per particle via CIC, apply per-
# particle kicks.  Captures the per-particle non-linearity of the actual
# distribution (not just rigid σ), which is what neither the linear
# uniform-cylinder nor the rigid-Gaussian Bassetti-Erskine kick can do.
# ---------------------------------------------------------------------------
def kick_continuous_2d_pic(beam, ds_mm: float, *,
                           nx: int = 128, ny: int = 128,
                           grid_extent: float = 6.0) -> None:
    """Apply one 2-D Hockney FFT PIC space-charge kick over ``ds_mm``.

    Parameters
    ----------
    beam : Beam
        Must have ``beam.continuous == True``; modified in place.
    ds_mm : float
        Step length in millimetres.
    nx, ny : int
        2-D grid resolution.  Defaults are tuned for LEBT-class beams
        (σ ≈ 1–10 mm) where 128² gives ~0.1–0.5 mm resolution.
    grid_extent : float
        Grid half-width in σ.  6σ keeps the Gaussian tails inside the
        domain so the doubled-grid open-BC trick is accurate.
    """
    if beam.current == 0 or beam.n_alive < 2 or ds_mm <= 0:
        return

    alive = beam.alive_mask
    alive_idx = np.where(alive)[0]
    xs_mm = beam.particles[alive, 0]
    ys_mm = beam.particles[alive, 2]
    xs = xs_mm * 1e-3
    ys = ys_mm * 1e-3
    n = xs.size

    sx = max(float(np.std(xs)), 1e-12)
    sy = max(float(np.std(ys)), 1e-12)
    cx = float(np.mean(xs))
    cy = float(np.mean(ys))
    half_x = grid_extent * sx
    half_y = grid_extent * sy
    x_lo, x_hi = cx - half_x, cx + half_x
    y_lo, y_hi = cy - half_y, cy + half_y
    dx = (x_hi - x_lo) / nx
    dy = (y_hi - y_lo) / ny

    ref = beam.ref
    beta  = float(ref.beta)
    gamma = float(ref.gamma)
    mass_MeV = float(ref.species.mass)
    q_abs = abs(ref.species.charge) * E_CHARGE
    I_A = abs(beam.current) * 1e-3
    EPS0 = 8.8541878128e-12
    v_m_s = beta * C_LIGHT
    if v_m_s <= 0:
        return
    # Charge-per-macroparticle, expressed as charge per unit beam-length.
    # A continuous beam carries λ = I/v [C/m] spread over the LAUNCHED
    # macro count (macrocharge convention — see kick_continuous_2d):
    # dividing by the alive count instead pinned the deposited total at
    # the configured current no matter how many particles were scraped.
    # Only alive particles are deposited, so the transported charge now
    # decays with transmission.  Lossless: n_particles == n → identical.
    q_macro = (I_A / v_m_s) / beam.n_particles

    # CIC deposit into rho [C / (m of beam · m² of (x,y))]
    fx = (xs - x_lo) / dx
    fy = (ys - y_lo) / dy
    ix = np.floor(fx).astype(np.int64)
    iy = np.floor(fy).astype(np.int64)
    wx = fx - ix
    wy = fy - iy
    inside = (ix >= 0) & (ix < nx - 1) & (iy >= 0) & (iy < ny - 1)
    ix_in = ix[inside]; iy_in = iy[inside]
    wx_in = wx[inside]; wy_in = wy[inside]

    rho = np.zeros((nx, ny))
    np.add.at(rho, (ix_in,     iy_in),     (1.0 - wx_in) * (1.0 - wy_in))
    np.add.at(rho, (ix_in + 1, iy_in),     wx_in         * (1.0 - wy_in))
    np.add.at(rho, (ix_in,     iy_in + 1), (1.0 - wx_in) * wy_in)
    np.add.at(rho, (ix_in + 1, iy_in + 1), wx_in         * wy_in)
    rho *= q_macro / (dx * dy)

    # 2-D Hockney: doubled grid + open-BC Coulomb Green's function.
    Nx2, Ny2 = 2 * nx, 2 * ny
    rho_pad = np.zeros((Nx2, Ny2))
    rho_pad[:nx, :ny] = rho

    # Wrap-around offsets so the doubled-grid FFT acts as open-BC convolution
    # for the inner Nx×Ny domain.  Index i → min(i, Nx2-i) gives the absolute
    # offset for the periodic-image trick.
    ig = np.arange(Nx2)
    jg = np.arange(Ny2)
    ix_off = np.minimum(ig, Nx2 - ig)
    iy_off = np.minimum(jg, Ny2 - jg)
    XOFF, YOFF = np.meshgrid(ix_off, iy_off, indexing="ij")
    rsq = (XOFF * dx) ** 2 + (YOFF * dy) ** 2
    # Self-cell regularisation: replace r=0 with the Gauss-equivalent radius
    # of a square cell, r_eff = √(dx·dy / π), giving the right asymptotic
    # field at distances >> cell-size.
    r_eff_sq = (dx * dy) / np.pi
    rsq[0, 0] = r_eff_sq
    G = -0.5 * np.log(rsq) / (2.0 * np.pi * EPS0)  # G(x,y) = −ln(r)/(2πε₀)

    rho_hat = np.fft.rfft2(rho_pad)
    G_hat   = np.fft.rfft2(G)
    # Discrete convolution → continuous integral: multiply by cell area.
    phi_pad = np.fft.irfft2(rho_hat * G_hat, s=(Nx2, Ny2)) * (dx * dy)
    phi = phi_pad[:nx, :ny]

    # E = -∇φ on the grid (central differences for interior; first-order at
    # the edges — those particles are far in the tail and their kick error
    # is dominated by the deposition itself).
    Ex = np.zeros_like(phi)
    Ey = np.zeros_like(phi)
    Ex[1:-1, :] = -(phi[2:,    :] - phi[:-2, :]) / (2.0 * dx)
    Ex[0,    :] = -(phi[1,     :] - phi[0,    :]) / dx
    Ex[-1,   :] = -(phi[-1,    :] - phi[-2,   :]) / dx
    Ey[:, 1:-1] = -(phi[:,    2:] - phi[:, :-2]) / (2.0 * dy)
    Ey[:,    0] = -(phi[:,     1] - phi[:,    0]) / dy
    Ey[:,   -1] = -(phi[:,    -1] - phi[:,   -2]) / dy

    # CIC gather to particles
    Ex_p = np.zeros(n)
    Ey_p = np.zeros(n)
    Ex_p[inside] = (
        (1.0 - wx_in) * (1.0 - wy_in) * Ex[ix_in,     iy_in]
        + wx_in         * (1.0 - wy_in) * Ex[ix_in + 1, iy_in]
        + (1.0 - wx_in) * wy_in         * Ex[ix_in,     iy_in + 1]
        + wx_in         * wy_in         * Ex[ix_in + 1, iy_in + 1]
    )
    Ey_p[inside] = (
        (1.0 - wx_in) * (1.0 - wy_in) * Ey[ix_in,     iy_in]
        + wx_in         * (1.0 - wy_in) * Ey[ix_in + 1, iy_in]
        + (1.0 - wx_in) * wy_in         * Ey[ix_in,     iy_in + 1]
        + wx_in         * wy_in         * Ey[ix_in + 1, iy_in + 1]
    )

    mc2_J = mass_MeV * 1e6 * E_CHARGE
    ds_m = ds_mm * 1e-3
    pre = q_abs * ds_m / (beta * beta * gamma * mc2_J)
    beam.particles[alive_idx, 1] += (pre * Ex_p) * 1e3
    beam.particles[alive_idx, 3] += (pre * Ey_p) * 1e3


class PicSolver:
    """Orchestrates one PIC space-charge kick per call to :meth:`kick`."""

    def __init__(self, config: SpaceChargeConfig):
        self.config = config
        # Lazily-built companion solver for bunch-train neighbour images
        # (3× nz grid, train_images forced off — see _kick_bunch_train).
        self._train_solver: "PicSolver | None" = None
        self._train_engaged: bool = False
        self._solver: PoissonSolverFFT | None = None
        self._grid_min: np.ndarray | None = None
        self._grid_max: np.ndarray | None = None
        self._n_grid: np.ndarray | None = None
        # Mid-run grid-extent override (HELIX_SC_GRID card).  Kept on the
        # SOLVER instance, never written into ``config`` — the config
        # object is caller-owned and reused across runs, so mutating it
        # would leak this run's override into the next one.
        self._extent_override: float | None = None
        self._extent_dirty: bool = False

    # ------------------------------------------------------------------
    def set_grid_extent(self, extent_sigma: float) -> None:
        """Change the grid extent (in beam sigmas) from now on.

        Called by the multi-particle tracker when it reaches a
        ``; HELIX_SC_GRID`` card.  Takes effect on the NEXT kick: in
        adaptive grid mode the extent is re-read every kick anyway; in
        fixed mode the ``_extent_dirty`` flag forces one re-derivation
        of the frozen grid from the CURRENT beam, after which fixed
        mode freezes again at the new extent.
        """
        self._extent_override = float(extent_sigma)
        self._extent_dirty = True

    # ------------------------------------------------------------------
    def _kick_bunch_train(self, beam, ds: float) -> None:
        """One SC kick with ±1 bunch-train neighbour images.

        Deposits the alive beam three times (centre and copies shifted
        ±360° in phase), solves on a 3×-long grid (nz tripled so the
        per-bunch resolution is unchanged), and applies the gathered
        kicks to the centre copy only.  Charge bookkeeping is exact:
        current ×3 with launched-count ×3 leaves the macro-charge and
        the loss-scaled current untouched.  ±1 image suffices — the ±2
        shell scales as (σ_z/2βλ)³ (measured inert on the PXIE RFQ).
        Validated against the TW/Toutatis 5 mA identical-beam benchmark
        (2026-08-02): isolated 78.5 % → train 80.7 % vs Toutatis 80.3.
        """
        import dataclasses as _dc

        from linac_gen.core.beam import Beam as _Beam
        from linac_gen.core.reference import ReferenceParticle as _Ref

        if self._train_solver is None:
            self._train_solver = PicSolver(_dc.replace(
                self.config, nz=3 * self.config.nz, train_images=False))
        # deck HELIX_SC_GRID extent overrides must reach the companion
        self._train_solver._extent_override = self._extent_override
        al = beam.alive_mask
        idx = np.where(al)[0]
        q = beam.particles[al]
        ncen = len(q)
        # neighbour spacing in LOCAL RF degrees: one bunch period =
        # 360 * (f_local / f_train) -- for a sub-harmonic train the
        # images sit h periods away, not one (adversarial review
        # 2026-08-02, finding 6)
        f_train = getattr(beam, "bunch_train_frequency", 0.0) or 0.0
        h = max(1, int(round(beam.ref.frequency / f_train))) \
            if f_train > 0 else 1
        shift = 360.0 * h
        qp = q.copy()
        qp[:, 4] += shift
        qm = q.copy()
        qm[:, 4] -= shift
        aug = _Beam(ref=_Ref(species=beam.ref.species,
                             w_kin=beam.ref.w_kin,
                             frequency=beam.ref.frequency),
                    n_particles=3 * beam.n_particles,
                    current=3.0 * beam.current)
        aug.particles[:3 * ncen] = np.vstack([q, qp, qm])
        aug.lost[3 * ncen:] = True
        aug.bunch_frequency = beam.bunch_frequency
        aug.continuous = False
        aug.bunch_train = False              # recursion guard
        before = aug.particles[:ncen, [1, 3, 5]].copy()
        self._train_solver.kick(aug, ds)
        d = aug.particles[:ncen, [1, 3, 5]] - before
        for col_j, col_b in ((0, 1), (1, 3), (2, 5)):
            beam.particles[idx, col_b] += d[:, col_j]

    def kick(self, beam, ds: float) -> None:
        """Apply one space-charge kick over step *ds* (mm).

        Parameters
        ----------
        beam : Beam
            Beam object (modified in-place).
        ds : float
            Step length in mm.
        """
        if beam.current <= 0 or beam.n_alive < 2:
            return

        # Bunch-train neighbour images (config.train_images; None = on
        # exactly when the beam was injected DC and bunched in flight).
        # Only while the bunch is LONG (σφ ≥ 30°): there the charge
        # still overlaps the neighbouring periods and the images are
        # real physics.  Once bunched, the neighbours (a full βλ away)
        # are negligible while the 3×-span train grid would degrade the
        # self-field resolution — so short bunches use the isolated
        # solve, which is also TraceWin's downstream semantics.
        use_train = self.config.train_images
        if use_train is None:
            use_train = bool(getattr(beam, "bunch_train", False))
        if use_train:
            # CORE phase spread, not np.std: out-of-bucket slippers can
            # drag a plain std to hundreds of degrees forever, so a
            # bunched core downstream would never leave the train path
            # (adversarial review 2026-08-02).  Half the 16-84 percentile
            # span equals sigma for a Gaussian and ignores the tails.
            phis = beam.particles[beam.alive_mask, 4]
            p16, p84 = np.percentile(phis, (16.0, 84.0))
            core = 0.5 * float(p84 - p16)
            # Hysteresis (engage >=35 deg, release <=25 deg): the train
            # and isolated solves differ at the few-percent level, so a
            # loss-driven drift across one fixed threshold must not
            # toggle the solver back and forth.
            if self._train_engaged:
                self._train_engaged = core > 25.0
            else:
                self._train_engaged = core >= 35.0
            if self._train_engaged:
                return self._kick_bunch_train(beam, ds)

        alive_mask = beam.alive_mask

        # 1. Spatial coordinates (mm)
        coords_lab = beam_to_spatial(beam)  # (N_alive, 3)

        # 2. Lorentz boost to rest frame.  Longitudinal positions stretch
        #    by γ; total charge is conserved.  Since the grid is built on
        #    the boosted coords (below), the rest-frame cell_vol grows by
        #    γ automatically, so ρ_rest = Σq/cell_vol is correctly reduced
        #    by 1/γ vs the lab density (no explicit rescale needed).
        gamma = beam.ref.gamma
        coords_rest = boost_to_rest_frame(coords_lab, gamma)

        # 3. Grid setup
        self._setup_grid(coords_rest)

        # 4. Macro-particle charges.  Q = I / f_bunch per the TW manual
        # ("FREQ command changes the R.F. frequency of the following
        # structure, the beam frequency is not affected") and the partran
        # SC DLL signature ("freq : beam frequency (Hz)").  After the
        # FREQ-jump dphi rescale fix, this convention beats the prior
        # cavity-freq choice on the MEBT+HWR+SSR1+SSR2 lattice: combined
        # |σ LG/TW − 1|_avg drops from 0.053 → 0.037, σ_y oscillation std
        # collapses 4× (0.057 → 0.014).  Residual ~3-4 % LG bias at SSR2
        # is the kernel calibration offset (separate concern, see
        # project_lg_pic_kernel_calibration_offset.md).
        n_alive = beam.n_alive
        from linac_gen.pic.macrocharge import macro_charge_for
        macro_charge = macro_charge_for(beam)   # shared convention
        charges = np.full(n_alive, macro_charge)

        # 5. Charge deposition (CIC or TSC, selected by config)
        deposit_fn, interpolate_fn = _select_kernel(
            getattr(self.config, "kernel", "cic")
        )
        rho = deposit_fn(
            coords_rest, charges,
            self._grid_min, self._grid_max, self._n_grid,
        )

        # 6. Poisson solve in rest frame
        Ex_rest, Ey_rest, Ez_rest = self._solver.solve(rho)

        # 7. Interpolate E-field to particles (matching shape function)
        E_at_particles = interpolate_fn(
            Ex_rest, Ey_rest, Ez_rest,
            coords_rest, self._grid_min, self._grid_max, self._n_grid,
        )  # (N_alive, 3) in solver units

        # 8. Apply kicks
        ds_m = ds * 1e-3  # mm -> m
        beta = beam.ref.beta
        mass = beam.ref.species.mass   # MeV/c^2

        # ---------------------------------------------------------------
        # Sign convention: we always deposit UNSIGNED |q|/N per macroparticle
        # (step 4 above), so the Poisson solve returns the field a *positive*
        # test charge would see from a positive charge cloud.  Inside such a
        # cloud E points outward.  The physical force on a particle of charge
        # q_real (signed) is F = q_real·E_real.  Because we modelled the cloud
        # as positive, E_real = sign(q_real) · E_code.  Substituting:
        #   F = q_real · sign(q_real) · E_code = |q_real| · E_code
        # → like charges always repel, which is the correct physics — no
        # species-sign multiplier belongs in the kick formula.  (The older
        # `charge_sign` prefix here used to FLIP the force for negative
        # species, turning the SC kick attractive; that is why H- beams
        # appeared to pinch instead of expand.)
        # ---------------------------------------------------------------
        # Units: the Poisson solver uses r_mm (millimetre grid) in the
        # Green's function, so
        #   phi_solver = Q / (4 pi eps0 * r_mm) = phi_SI / 1000
        # and the finite-difference gradient dphi/dx_mm shrinks by another
        # factor of 1000, giving
        #   E_solver = E_SI / 1e6.
        # We multiply by 1e6 to recover SI V/m before the kick formula.
        #
        # Field frame: E_* returned here is the REST-frame field (Poisson
        # was solved on rest-frame positions).  E_z is Lorentz-invariant
        # under a z-boost, so E_z_lab = E_z_rest.  E_perp transforms by γ,
        # but the transverse Lorentz force in the lab (E_perp + v×B) picks
        # up a (1−β²)=1/γ² cancellation from the induced B-field, so the
        # net lab-frame transverse force on the particle is |q|·E_rest_perp/γ.
        # Hence the β²γ² in factor_t: one γ from the force cancellation and
        # one βγ from the p_z used to convert Δp_perp → Δx'.
        # ---------------------------------------------------------------
        E_si = E_at_particles * 1e6  # V/m

        # Test-particle charge-state magnitude |Z| (units of e).  The bunch
        # charge deposited above (I/f) fixes the SOURCE field; the kick on
        # each particle still scales with its own |q| = |Z|·e.  For protons
        # and H- (|Z| = 1) this is a no-op; omitting it under-kicked
        # multiply charged ions by |Z| (fixed 2026-07-10).
        z_state = abs(beam.ref.species.charge)

        # Transverse angular kick (derivation in comment above):
        #   Δx'[rad] = |q| · E_x_rest[V/m] · Δs[m] / (β²γ²·m·c²)
        #   with |q| = |Z|·e and m·c²[J] = mass_MeV · 1e6 · e, the e cancels:
        #   Δx'[rad] = |Z| · E_x · Δs / (β²γ² · mass_MeV · 1e6); × 1e3 → mrad.
        factor_t = z_state * ds_m * 1e3 / (mass * 1e6 * beta**2 * gamma**2)

        # Longitudinal energy kick:
        #   ΔW[J] = |q| · E_z_rest[V/m] · Δs[m]   (no γ; like-charge repulsion)
        #   ΔW[MeV] = |Z| · E_z · Δs · 1e-6.
        factor_z = z_state * ds_m * 1e-6

        # Apply to alive particles
        alive_indices = np.where(alive_mask)[0]
        beam.particles[alive_indices, 1] += factor_t * E_si[:, 0]  # xp
        beam.particles[alive_indices, 3] += factor_t * E_si[:, 1]  # yp
        beam.particles[alive_indices, 5] += factor_z * E_si[:, 2]  # dW

    # ------------------------------------------------------------------
    def _setup_grid(self, coords_rest: np.ndarray) -> None:
        """Set up or reuse the PIC grid.

        Fixed mode short-circuits after the first call.  Adaptive mode
        recomputes extent every kick and either:
          * builds a fresh ``PoissonSolverFFT`` on the very first kick, or
          * calls ``solver.update_grid(...)`` to rebuild the Green's function
            in place on subsequent kicks (preserves backend, FFT plan cache,
            and device buffers — order-of-magnitude wall-time saving).
        """
        if (self._solver is not None and self.config.grid_mode == "fixed"
                and not self._extent_dirty):
            return

        extent = (self._extent_override if self._extent_override is not None
                  else self.config.grid_extent)
        self._extent_dirty = False
        std = np.std(coords_rest, axis=0)
        mean = np.mean(coords_rest, axis=0)
        half_size = extent * std
        half_size = np.maximum(half_size, 1e-6)  # prevent zero-size grid

        self._grid_min = mean - half_size
        self._grid_max = mean + half_size
        self._n_grid = np.array(
            [self.config.nx, self.config.ny, self.config.nz], dtype=np.int64,
        )
        if self._solver is None:
            self._solver = PoissonSolverFFT(
                self._grid_min, self._grid_max, self._n_grid,
                use_gpu=getattr(self.config, "use_gpu", "auto"),
                green_kind=getattr(self.config, "green_kind", "igf"),
            )
        else:
            # Adaptive: reuse existing solver instance, just retarget the grid.
            self._solver.update_grid(self._grid_min, self._grid_max)
