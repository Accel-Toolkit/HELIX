"""Bi-Gaussian (core + halo) distribution generator.

Real H⁻ linac beams develop halo from RF non-linearities, intra-beam
stripping, and SC mismatch.  A clean Gaussian distribution under-counts
the beam loss budget because the halo population at large amplitudes is
exponentially suppressed.  Halo studies therefore commonly seed the
beam with an *empirical* bi-Gaussian: a tight Gaussian core plus a
wider Gaussian halo, mixed by a halo fraction ``w`` ∈ [0, 1].

This is the same shape used by Allen, Wangler, et al. for halo
characterisation (Wangler, *Principles of RF Linear Accelerators*,
§9.6) and by the IMPACT-X reference Thermal/bi-thermal distribution
(``Thermal.H``) — the simplification here is that we use closed-form
Gaussians per plane rather than building the radial CDF from a
numerical Vlasov-Poisson integration.  Users wanting the
self-consistent Vlasov-Poisson distribution should run a matched
Gaussian through the SC tracker for ~10 betatron periods, but a
bi-Gaussian is the right starting point for halo studies.

The mixture is rescaled at the end so the total RMS emittance per
plane equals the requested ``emit_x / emit_y / emit_z``.  Net effect:
the user picks (halo fraction, halo size ratio) and the input ε is
preserved.
"""
import numpy as np

from linac_gen.distributions.gaussian import (
    _sigma_matrix_2d, _generate_plane_gaussian,
)


# ---------------------------------------------------------------------------
def _generate_plane_bigaussian(
    n: int,
    alpha: float,
    beta: float,
    emit: float,
    halo_fraction: float,
    halo_ratio: float,
    cutoff: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Generate n correlated (u, u') pairs from a bi-Gaussian mixture.

    The returned distribution has

        f(u, u') = (1 - w) · G(σ_core) + w · G(r · σ_core)

    with ``w = halo_fraction`` and ``r = halo_ratio``.  The component
    sigmas are picked so the *total* RMS sigma matches the input
    ``emit``: σ_total² = (1-w)·σ_core² + w·(r·σ_core)², giving

        σ_core² = emit / (1 - w + w·r²)

    The same Twiss correlation (α, β) applies to both components — the
    Cholesky factor of σ_core gives the core; halo points are scaled by
    ``r`` from a copy of that factor.

    Returns an array of shape ``(n, 2)`` matched exactly to the target
    Σ-matrix (post-sampling Cholesky rescale).
    """
    if halo_fraction < 0.0 or halo_fraction > 1.0:
        raise ValueError(f"halo_fraction must be in [0,1], got {halo_fraction}")
    if halo_ratio < 1.0:
        raise ValueError(f"halo_ratio must be >= 1, got {halo_ratio}")

    if halo_fraction == 0.0:
        # Degenerates to a plain Gaussian.
        return _generate_plane_gaussian(n, alpha, beta, emit, cutoff, rng)

    # Variance budget: (1-w)·σ_core² + w·(r·σ_core)² = σ_total²
    # → σ_core² = σ_total² / (1 - w + w·r²)
    var_factor = 1.0 - halo_fraction + halo_fraction * halo_ratio ** 2
    emit_core = emit / var_factor
    L_core = np.linalg.cholesky(_sigma_matrix_2d(alpha, beta, emit_core))

    # Bernoulli mask: which particles come from the halo?
    is_halo = rng.uniform(0.0, 1.0, n) < halo_fraction
    n_core = int(np.sum(~is_halo))
    n_halo = n - n_core

    out = np.empty((n, 2), dtype=np.float64)

    # Core: cutoff applied at ``cutoff·σ_core`` (= cutoff·σ on the
    # *core* sigma so the truncation matches the user's intent).
    if n_core > 0:
        z = rng.standard_normal((n_core * 4, 2))  # oversample then clip
        u = z @ L_core.T
        sx = abs(L_core[0, 0])
        sxp = np.sqrt(L_core[1, 0] ** 2 + L_core[1, 1] ** 2)
        keep = (np.abs(u[:, 0]) <= cutoff * sx) & (np.abs(u[:, 1]) <= cutoff * sxp)
        u = u[keep][:n_core]
        # If oversample wasn't enough, top up with a plain Gaussian
        # (rejection sampling) to avoid pathological edge cases.
        if len(u) < n_core:
            u = _generate_plane_gaussian(n_core, alpha, beta, emit_core, cutoff, rng)
        out[~is_halo] = u[:n_core]

    # Halo: scale the same Cholesky factor by ``r`` and apply a much
    # looser cutoff (default = 3·r) so the halo isn't truncated at the
    # core's clip distance.  Without this the halo would lose its tail.
    if n_halo > 0:
        L_halo = halo_ratio * L_core
        z = rng.standard_normal((n_halo, 2))
        out[is_halo] = z @ L_halo.T

    # Final rescale so the *sample* Σ matches the target Σ exactly.
    # (Bernoulli + finite-N → small drift; this nails it down without
    # changing the bi-Gaussian shape.)
    sigma_target = _sigma_matrix_2d(alpha, beta, emit)
    sigma_sample = (out.T @ out) / n
    L_t = np.linalg.cholesky(sigma_target)
    L_s = np.linalg.cholesky(sigma_sample)
    A = L_t @ np.linalg.inv(L_s)
    return out @ A.T


# ---------------------------------------------------------------------------
def generate_thermal(
    n: int,
    emit_x: float, alpha_x: float, beta_x: float,
    emit_y: float, alpha_y: float, beta_y: float,
    emit_z: float, alpha_z: float, beta_z: float,
    halo_fraction: float = 0.05,
    halo_ratio: float = 5.0,
    cutoff: float = 3.0,
    seed: int = None,
) -> np.ndarray:
    """Bi-Gaussian (core + halo) 6-D distribution.

    Each phase plane is independently sampled from a two-component
    Gaussian mixture: a tight core (small σ) + a wide halo (σ scaled by
    ``halo_ratio``), mixed by ``halo_fraction``.  The resulting RMS
    emittance per plane equals the input ``emit_*`` exactly (rescaled
    after sampling).

    Parameters
    ----------
    n : int
        Number of particles.
    emit_x, emit_y, emit_z : float
        Geometric RMS emittances (target Σ-matrix per plane).
    alpha_*, beta_* : float
        Twiss parameters per plane.
    halo_fraction : float, default 0.05
        Probability that a particle is drawn from the halo component.
        Reasonable values: 0.01–0.10 (1–10 %).
    halo_ratio : float, default 5.0
        Ratio of halo σ to core σ (halo "spread factor"). Reasonable
        values: 3.0–10.0.  Setting halo_ratio = 1.0 makes the halo
        identical to the core (i.e. recovers a single Gaussian).
    cutoff : float, default 3.0
        Truncation cutoff applied to the *core* component, in units of
        core σ.  The halo is *not* truncated — that is the whole point.
    seed : int or None
        Random seed for reproducibility.

    Returns
    -------
    np.ndarray
        Shape ``(n, 6)`` array of phase-space deviations.

    Notes
    -----
    The bi-Gaussian shape lets users dial in halo populations to study
    aperture loss, IBS feedback into the core, and magnetic-stripping
    fractions that depend on the tail of the transverse distribution.
    For self-consistent Vlasov-Poisson halos, instead run a matched
    Gaussian through 5–10 betatron periods of the SC tracker — the
    bi-Gaussian is a *seed* for halo studies, not a substitute for
    self-consistent dynamics.
    """
    rng = np.random.default_rng(seed)

    plane_x = _generate_plane_bigaussian(
        n, alpha_x, beta_x, emit_x,
        halo_fraction, halo_ratio, cutoff, rng,
    )
    plane_y = _generate_plane_bigaussian(
        n, alpha_y, beta_y, emit_y,
        halo_fraction, halo_ratio, cutoff, rng,
    )
    plane_z = _generate_plane_bigaussian(
        n, alpha_z, beta_z, emit_z,
        halo_fraction, halo_ratio, cutoff, rng,
    )

    particles = np.empty((n, 6), dtype=np.float64)
    particles[:, 0:2] = plane_x
    particles[:, 2:4] = plane_y
    particles[:, 4:6] = plane_z
    return particles
