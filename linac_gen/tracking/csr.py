"""1-D steady-state Coherent Synchrotron Radiation (CSR) kick.

When a short bunch travels around a bend, each particle radiates; the
radiation travels along the straight chord while the bunch follows the
arc, so radiation emitted by a particle "cuts the corner" and reaches
particles *ahead* of it on the orbit.  A given test particle therefore
feels the CSR field of every particle *behind* it (toward the tail).
The dominant, well-established model is the **1-D steady-state CSR
wake** (Derbenev 1995; Saldin, Schneidmiller, Yurkov 1997): for a bunch
on a circular orbit of radius ``R`` the energy change per unit path
length of a particle at internal longitudinal coordinate ``s`` is

    dW/ds(s) = -A · ∫_{-∞}^{s} λ'(s') · (s - s')^(-1/3) ds'

with

    A = ( e / (4 π ε₀) ) · ( 2 / ( 3^(1/3) · R^(2/3) ) )

``λ(s)`` is the longitudinal **number** line density (particles per
metre, ∫λ ds = N_phys), ``s`` increases toward the bunch head, and the
integral runs over particles *behind* the test particle (s' < s) — the
retarded-field geometry.  The result is the classic CSR signature: the
bunch head gains energy, the tail loses more, and the bunch as a whole
loses energy.  The kernel singularity ``(s-s')^(-1/3)`` is integrable;
we integrate it analytically over each density bin.

The CSR kick only changes a particle's energy (``dw`` coordinate).
Transverse emittance growth then arises naturally because the bend has
dispersion: an energy-correlated kick along the bunch becomes an
(x, x') correlation downstream.

Scope (HELIX Month 2): steady-state only.  Transient entrance/exit
fields (drift→bend and bend→drift) are not modelled — valid when the
bend is long compared with the CSR formation/slippage length.  CSR is a
multi-particle effect: it needs the actual bunch profile, so it acts
only in multi-particle tracking, never in the envelope solver.
"""
from __future__ import annotations

import math

import numpy as np

from linac_gen.core.constants import C_LIGHT, E_CHARGE, EPSILON_0

# Beam particle-array column indices (mirror linac_gen/core/beam.py).
_DPHI = 4
_DW = 5

# 1/3 and 2/3 as floats — used in the kernel.
_THIRD = 1.0 / 3.0
_TWO_THIRD = 2.0 / 3.0


class CsrKicker:
    """1-D steady-state CSR kicker, applied inside :class:`Dipole` elements.

    Parameters
    ----------
    n_bins : int
        Number of longitudinal bins for the line-density histogram.
        200 is a good default — fine enough to resolve the profile,
        coarse enough that ``1/√N`` shot noise stays manageable after
        smoothing.
    model : str
        Forward-compatibility selector.  Only ``"1d_steady"`` is
        implemented; other values raise ``ValueError``.
    smooth_sigma_bins : float
        Gaussian-smoothing width (in bins) applied to the line density
        before differentiation.  CSR is sensitive to ``dλ/ds`` so the
        raw histogram's shot noise must be suppressed.
    """

    def __init__(self, n_bins: int = 200, model: str = "1d_steady",
                 smooth_sigma_bins: float = 1.5) -> None:
        if n_bins < 8:
            raise ValueError(f"csr n_bins must be >= 8, got {n_bins}")
        if model != "1d_steady":
            raise ValueError(
                f"CsrKicker model must be '1d_steady', got {model!r}"
            )
        self.n_bins = int(n_bins)
        self.model = model
        self.smooth_sigma_bins = float(smooth_sigma_bins)

    # ------------------------------------------------------------------
    def apply(self, beam, element, ds_mm: float) -> None:
        """Apply one CSR energy kick over a sub-step of length *ds_mm* (mm).

        ``element`` is the :class:`Dipole` being tracked; its ``rho``
        (bend radius, mm) sets the CSR strength.  Modifies
        ``beam.particles[:, DW]`` in place.  A no-op when the beam has no
        current, fewer than a handful of live particles, or a degenerate
        (zero-length) profile.
        """
        if ds_mm <= 0.0:
            return
        current_mA = abs(getattr(beam, "current", 0.0) or 0.0)
        if current_mA <= 0.0:
            return
        alive = beam.alive_mask
        n_alive = int(np.count_nonzero(alive))
        if n_alive < 16:
            return

        rho_mm = getattr(element, "rho", 0.0) or 0.0
        R_m = abs(rho_mm) * 1e-3
        if R_m <= 0.0:
            return

        # --- longitudinal coordinate of each live particle, in metres ---
        # Δφ [deg] = k_phi · z [m]  with  k_phi = -360 / (β · λ_m)
        # ⇒  z [m] = -Δφ · β · λ_m / 360.  Larger z ⇒ bunch head.
        ref = beam.ref
        beta = float(ref.beta)
        wavelength_m = float(ref.wavelength) * 1e-3   # ref.wavelength is mm
        if beta <= 0.0 or wavelength_m <= 0.0:
            return
        dphi = beam.particles[alive, _DPHI]
        z = -dphi * beta * wavelength_m / 360.0       # m, head = +z

        z_min, z_max = float(np.min(z)), float(np.max(z))
        span = z_max - z_min
        if span <= 0.0:
            return     # degenerate (all particles at same phase)

        # --- line number-density λ(z) [particles / m] ------------------
        # Total physical particles per bunch:  N_phys = (I / f_bunch) / e.
        freq_Hz = float(getattr(beam, "bunch_frequency", ref.frequency)) * 1e6
        if freq_Hz <= 0.0:
            return
        charge_per_bunch_C = (current_mA * 1e-3) / freq_Hz
        n_phys = charge_per_bunch_C / E_CHARGE
        weight_per_macro = n_phys / n_alive           # phys particles / macro

        n_bins = self.n_bins
        counts, edges = np.histogram(z, bins=n_bins, range=(z_min, z_max))
        dz = (z_max - z_min) / n_bins                 # bin width, m
        if dz <= 0.0:
            return
        lam = counts.astype(np.float64) * weight_per_macro / dz   # 1/m

        # --- smooth to suppress 1/√N shot noise ------------------------
        lam = self._gaussian_smooth(lam, self.smooth_sigma_bins)

        # --- dλ/ds by central differences ------------------------------
        dlam = np.gradient(lam, dz)                   # (1/m) / m

        # --- steady-state CSR wake -------------------------------------
        # dW/ds(s_k) = -A · Σ_{j>=k} dlam_j · ∫_bin_j (s'-s_k)^(-1/3) ds'
        # The per-bin integral of u^(-1/3) is (3/2)[u^(2/3)] between the
        # bin edges, measured from the test-particle position s_k.
        A = (E_CHARGE / (4.0 * math.pi * EPSILON_0)) * (
            2.0 / (3.0 ** _THIRD * R_m ** _TWO_THIRD)
        )
        centers = 0.5 * (edges[:-1] + edges[1:])      # m
        dW_ds = self._convolve_wake(centers, dlam, dz, A)   # eV / m

        # --- map per-bin dW/ds onto each particle, apply over ds -------
        ds_m = ds_mm * 1e-3
        # Particle's bin index (clamp to valid range).
        idx = np.clip(
            ((z - z_min) / dz).astype(np.int64), 0, n_bins - 1
        )
        dW_eV = dW_ds[idx] * ds_m                     # eV
        dW_MeV = dW_eV * 1e-6
        beam.particles[alive, _DW] += dW_MeV

    # ------------------------------------------------------------------
    @staticmethod
    def _gaussian_smooth(arr: np.ndarray, sigma_bins: float) -> np.ndarray:
        """Gaussian-smooth a 1-D array; falls back to a no-op for σ≈0."""
        if sigma_bins <= 0.0:
            return arr
        try:
            from scipy.ndimage import gaussian_filter1d
            return gaussian_filter1d(arr, sigma_bins, mode="nearest")
        except ImportError:
            # Minimal box-smooth fallback if scipy is unavailable.
            k = max(1, int(round(sigma_bins)))
            kernel = np.ones(2 * k + 1) / (2 * k + 1)
            return np.convolve(arr, kernel, mode="same")

    @staticmethod
    def _convolve_wake(centers: np.ndarray, dlam: np.ndarray,
                       dz: float, A: float) -> np.ndarray:
        """Discrete steady-state CSR convolution (vectorised).

        Returns ``dW/ds`` (eV / m) at each bin centre.  A test particle
        in bin ``k`` is affected by source bins *behind* it (toward the
        tail, ``j <= k``).  Because the bins are uniform, the source
        contribution depends only on the offset ``m = k - j``, so the
        double sum collapses to a 1-D correlation:

            out[k] = Σ_{m>=0} dlam[k-m] · kernel[m]

        ``kernel[m]`` is the analytic integral of ``u^(-1/3)`` across a
        source bin sitting ``m`` bins behind the test particle:

            kernel[0]   = (3/2)·(dz/2)^(2/3)
            kernel[m>0] = (3/2)·[ (m·dz + dz/2)^(2/3)
                                  − (m·dz − dz/2)^(2/3) ]
        """
        n = centers.size
        m = np.arange(n, dtype=np.float64)
        u_hi = m * dz + 0.5 * dz
        u_lo = np.maximum(m * dz - 0.5 * dz, 0.0)
        kernel = 1.5 * (u_hi ** _TWO_THIRD - u_lo ** _TWO_THIRD)

        out = np.zeros(n, dtype=np.float64)
        # out[k] += dlam[k-offset] * kernel[offset]  for every offset.
        for offset in range(n):
            kval = kernel[offset]
            if kval == 0.0:
                continue
            out[offset:] += dlam[: n - offset] * kval
        # Sign convention: with sources integrated toward the tail, the
        # leading -A gives the physical CSR signature — head gains, tail
        # loses, bunch loses net energy.  Verified by tests/tracking/test_csr.
        return -A * out


__all__ = ["CsrKicker"]
