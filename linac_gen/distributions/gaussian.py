"""Gaussian distribution generator with Cholesky decomposition and cutoff truncation."""
import numpy as np

from linac_gen.units import REJECTION_SAMPLING_MAX_ITERS


def _sigma_matrix_2d(alpha: float, beta: float, emit: float) -> np.ndarray:
    """Return the 2x2 phase-space sigma matrix from Twiss parameters.

    sigma_11 = beta * emit
    sigma_12 = -alpha * emit
    sigma_22 = gamma_t * emit  where gamma_t = (1 + alpha^2) / beta
    """
    gamma_t = (1.0 + alpha * alpha) / beta
    sigma = np.array([
        [beta * emit,   -alpha * emit],
        [-alpha * emit,  gamma_t * emit],
    ], dtype=np.float64)
    return sigma


def _cholesky_2d(alpha: float, beta: float, emit: float) -> np.ndarray:
    """Return the lower Cholesky factor L such that sigma = L @ L.T."""
    sigma = _sigma_matrix_2d(alpha, beta, emit)
    return np.linalg.cholesky(sigma)


def _generate_plane_gaussian(
    n: int,
    alpha: float,
    beta: float,
    emit: float,
    cutoff: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Generate n correlated (u, u') pairs for one plane with cutoff rejection.

    Returns array of shape (n, 2).
    """
    L = _cholesky_2d(alpha, beta, emit)
    result = np.empty((n, 2), dtype=np.float64)
    filled = 0

    # Oversample to reduce loop iterations for typical cutoff values
    batch = max(int(n * 1.5), 1000)

    # Precompute per-plane sigmas (constant across iterations)
    sigma_u = abs(L[0, 0])                              # sigma_x
    sigma_up = np.sqrt(L[1, 0] ** 2 + L[1, 1] ** 2)     # sigma_xp

    for _ in range(REJECTION_SAMPLING_MAX_ITERS):
        if filled >= n:
            # Rescale sample to exactly match the target sigma matrix so the
            # recorded ε at s=0 equals the requested ε (no sampling noise).
            sigma_target = _sigma_matrix_2d(alpha, beta, emit)
            sigma_sample = (result.T @ result) / n
            L_t = np.linalg.cholesky(sigma_target)
            L_s = np.linalg.cholesky(sigma_sample)
            A = L_t @ np.linalg.inv(L_s)
            return result @ A.T
        z = rng.standard_normal((batch, 2))
        u = z @ L.T  # shape (batch, 2), correlated pairs
        keep = (np.abs(u[:, 0]) <= cutoff * sigma_u) & \
               (np.abs(u[:, 1]) <= cutoff * sigma_up)
        accepted = u[keep]
        take = min(len(accepted), n - filled)
        result[filled:filled + take] = accepted[:take]
        filled += take

    raise RuntimeError(
        f"Gaussian rejection sampling failed to produce {n} particles within "
        f"{REJECTION_SAMPLING_MAX_ITERS} iterations (got {filled}). "
        f"Check cutoff ({cutoff}), Twiss parameters (alpha={alpha}, beta={beta}, "
        f"emit={emit}) for ill-conditioning."
    )


def generate_gaussian(
    n: int,
    emit_x: float, alpha_x: float, beta_x: float,
    emit_y: float, alpha_y: float, beta_y: float,
    emit_z: float, alpha_z: float, beta_z: float,
    cutoff: float = 3.0,
    seed: int = None,
) -> np.ndarray:
    """Generate an n-particle 6D Gaussian distribution.

    Each plane is generated independently using Cholesky decomposition of the
    2x2 phase-space sigma matrix.  Particles with any coordinate beyond
    ``cutoff`` RMS sigma in that plane are rejected and regenerated.

    Parameters
    ----------
    n : int
        Number of particles.
    emit_x, emit_y, emit_z : float
        Geometric RMS emittances (mm.mrad for transverse, deg.MeV for longitudinal).
    alpha_x, alpha_y, alpha_z : float
        Twiss alpha parameters (dimensionless).
    beta_x, beta_y, beta_z : float
        Twiss beta parameters (mm/mrad for transverse, deg/MeV for longitudinal).
    cutoff : float
        Truncation in units of RMS sigma per coordinate (default 3.0).
    seed : int or None
        Random seed for reproducibility.

    Returns
    -------
    np.ndarray
        Shape (n, 6) array of phase-space deviations
        [x(mm), x'(mrad), y(mm), y'(mrad), dphi(deg), dW(MeV)].
    """
    rng = np.random.default_rng(seed)

    plane_x = _generate_plane_gaussian(n, alpha_x, beta_x, emit_x, cutoff, rng)
    plane_y = _generate_plane_gaussian(n, alpha_y, beta_y, emit_y, cutoff, rng)
    plane_z = _generate_plane_gaussian(n, alpha_z, beta_z, emit_z, cutoff, rng)

    particles = np.empty((n, 6), dtype=np.float64)
    particles[:, 0:2] = plane_x
    particles[:, 2:4] = plane_y
    particles[:, 4:6] = plane_z

    return particles
