"""Parabolic distribution generator.

Density proportional to (1 - r^2/R^2) inside a 6D ellipsoid.

Algorithm: rejection sampling on top of waterbag (uniform 6D ellipsoid).
For each candidate point sampled uniformly from the 6D ellipsoid, accept it
with probability (1 - r^2) where r is the normalised radius in [0, 1].
This yields the desired parabolic radial density.

The resulting RMS emittance equals the input geometric emittance because the
Cholesky transform that encodes emittance is applied after the shape of the
distribution has been determined.
"""
import numpy as np

from linac_gen.units import REJECTION_SAMPLING_MAX_ITERS


def _sigma_matrix_2d(alpha: float, beta: float, emit: float) -> np.ndarray:
    """Return the 2x2 phase-space sigma matrix from Twiss parameters."""
    gamma_t = (1.0 + alpha * alpha) / beta
    return np.array([
        [beta * emit,   -alpha * emit],
        [-alpha * emit,  gamma_t * emit],
    ], dtype=np.float64)


def generate_parabolic(
    n: int,
    emit_x: float, alpha_x: float, beta_x: float,
    emit_y: float, alpha_y: float, beta_y: float,
    emit_z: float, alpha_z: float, beta_z: float,
    seed: int = None,
) -> np.ndarray:
    """Generate an n-particle 6D parabolic distribution.

    Particles are distributed with density proportional to (1 - r^2) inside
    a 6D hyperellipsoid, where r is the normalised radius (r=0 at centre,
    r=1 at boundary).

    Algorithm:
    1. Sample candidate points uniformly inside the 6D unit ball (waterbag method).
    2. Compute the normalised radius r for each candidate.
    3. Accept each candidate with probability (1 - r^2).
    4. Repeat until n accepted particles are collected.
    5. Transform each 2D plane with the Cholesky factor of its sigma matrix.

    The parabolic marginal halo parameter is lower than the waterbag value (1.4)
    because the distribution is more centrally peaked.

    Parameters
    ----------
    n : int
        Number of particles.
    emit_x, emit_y, emit_z : float
        Geometric RMS emittances.
    alpha_x, alpha_y, alpha_z : float
        Twiss alpha parameters.
    beta_x, beta_y, beta_z : float
        Twiss beta parameters.
    seed : int or None
        Random seed for reproducibility.

    Returns
    -------
    np.ndarray
        Shape (n, 6) array of phase-space deviations.
    """
    rng = np.random.default_rng(seed)

    def cholesky_2d(alpha, beta, emit):
        sigma = _sigma_matrix_2d(alpha, beta, emit)
        return np.linalg.cholesky(sigma)

    L_x = cholesky_2d(alpha_x, beta_x, emit_x)
    L_y = cholesky_2d(alpha_y, beta_y, emit_y)
    L_z = cholesky_2d(alpha_z, beta_z, emit_z)

    # Acceptance rate for parabolic is E[1 - r^2] over the 6D ball.
    # For a uniform 6D ball the pdf of r^2 is (d/2) * (r^2)^(d/2-1) / R^d,
    # i.e. p(r^2) = 3*(r^2)^2 for r in [0,1] with d=6.
    # E[1 - r^2] = integral_0^1 (1-u) * 3*u^2 du = 3*(1/3 - 1/4) = 0.25.
    # Oversample by ~5x to be safe under finite batch sizes.
    batch = max(int(n * 5), 2000)

    raw_accepted = np.empty((n, 6), dtype=np.float64)
    filled = 0

    for _ in range(REJECTION_SAMPLING_MAX_ITERS):
        if filled >= n:
            break
        # Sample uniform points on 6D unit sphere surface then scale radially
        z = rng.standard_normal((batch, 6))
        norms = np.linalg.norm(z, axis=1, keepdims=True)
        v = z / norms

        r_uni = rng.uniform(0.0, 1.0, size=(batch, 1))
        r = r_uni ** (1.0 / 6.0)  # uniform inside 6D ball
        v_scaled = v * r           # shape (batch, 6), radius = r_uni^(1/6)

        # Normalised radius squared for each candidate, in [0, 1]
        r2_norm = r_uni[:, 0] ** (1.0 / 3.0)

        # Accept with probability (1 - r^2)
        u = rng.uniform(0.0, 1.0, size=batch)
        accept_mask = u < (1.0 - r2_norm)

        accepted = v_scaled[accept_mask]
        take = min(len(accepted), n - filled)
        raw_accepted[filled:filled + take] = accepted[:take]
        filled += take
    else:
        raise RuntimeError(
            f"Parabolic rejection sampling failed to produce {n} particles "
            f"within {REJECTION_SAMPLING_MAX_ITERS} iterations (got {filled}). "
            f"This should never happen for healthy inputs."
        )

    # The raw v columns have variance = 1/(d+2) = 1/8 per component
    # (6D uniform ball: <vi^2> = 1/(d+2) with d=6) before rejection.
    # After parabolic rejection, the variance changes. We need to rescale
    # so that after the Cholesky transform the RMS emittance matches.
    #
    # For parabolic in 6D, the variance of each raw coordinate:
    # <vi^2> = integral_0^1 r^2*(1/6) * (1 - r^2) * S6 dr / Z
    # where S6 is the 6D sphere surface element and Z is the normalization.
    # By isotropy, <vi^2> = (1/6) * <r^2>.
    # <r^2> for parabolic = integral_0^1 r^2 * (1-r^2) * 6*r^5 dr / integral_0^1 (1-r^2)*6*r^5 dr
    #                     = 6*(1/8 - 1/10) / (6*(1/6 - 1/8))
    #                     = 6*(1/40) / (6*(1/24))
    #                     = (1/40)/(1/24) = 24/40 = 3/5
    # So <vi^2> = (1/6) * (3/5) = 1/10
    # We need variance = 1 before Cholesky (which encodes emittance).
    # Scale factor = sqrt(1 / (1/10)) = sqrt(10)
    scale = np.sqrt(10.0)

    particles = np.empty((n, 6), dtype=np.float64)
    particles[:, 0:2] = (scale * raw_accepted[:, 0:2]) @ L_x.T
    particles[:, 2:4] = (scale * raw_accepted[:, 2:4]) @ L_y.T
    particles[:, 4:6] = (scale * raw_accepted[:, 4:6]) @ L_z.T

    return particles
