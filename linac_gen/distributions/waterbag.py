"""Waterbag distribution generator: uniform fill of a 6D hyperellipsoid."""
import numpy as np


def _sigma_matrix_2d(alpha: float, beta: float, emit: float) -> np.ndarray:
    """Return the 2x2 phase-space sigma matrix from Twiss parameters."""
    gamma_t = (1.0 + alpha * alpha) / beta
    return np.array([
        [beta * emit,   -alpha * emit],
        [-alpha * emit,  gamma_t * emit],
    ], dtype=np.float64)


def generate_waterbag(
    n: int,
    emit_x: float, alpha_x: float, beta_x: float,
    emit_y: float, alpha_y: float, beta_y: float,
    emit_z: float, alpha_z: float, beta_z: float,
    seed: int = None,
) -> np.ndarray:
    """Generate an n-particle 6D waterbag distribution.

    Particles are distributed uniformly inside a 6D hyperellipsoid defined
    by the Twiss parameters.  The algorithm:

    1. Generate 6 standard normal deviates z1..z6.
    2. Normalise to the unit 6-sphere surface: v = z / |z|.
    3. Scale radially: v *= r^(1/6) where r ~ U(0, 1) — fills interior uniformly.
    4. Transform each 2D plane with the Cholesky factor of its sigma matrix to
       introduce the correct phase-space correlations and emittance scaling.

    For a 6D waterbag the single-plane kurtosis parameter H = <u^4>/<u^2>^2 - 1
    equals 1/3.

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

    # Step 1: sample uniform direction on 6D unit sphere surface
    z = rng.standard_normal((n, 6))
    norms = np.linalg.norm(z, axis=1, keepdims=True)
    v = z / norms  # uniform on 6-sphere surface

    # Step 2: scale radially to fill interior uniformly
    r = rng.uniform(0.0, 1.0, size=(n, 1))
    v *= r ** (1.0 / 6.0)

    # Step 3: apply per-plane Twiss transformation
    # The sigma matrix for a waterbag with geometric emittance `emit` is the
    # same form as for a Gaussian.  We use a scaling factor so that the RMS
    # emittance of the waterbag matches the input geometric emittance.
    #
    # For a uniform 6D ball the variance in each pair of coordinates
    # (before Twiss transformation) is 1/(6+2) = 1/8.
    # After the normalisation below the variance of each z_i before
    # transformation is 1, so we need to scale by sqrt(emit * twiss) as usual.
    # The Cholesky factor already encodes the emittance scaling.

    def cholesky_2d(alpha, beta, emit):
        sigma = _sigma_matrix_2d(alpha, beta, emit)
        return np.linalg.cholesky(sigma)

    L_x = cholesky_2d(alpha_x, beta_x, emit_x)
    L_y = cholesky_2d(alpha_y, beta_y, emit_y)
    L_z = cholesky_2d(alpha_z, beta_z, emit_z)

    # The raw v columns have variance = 1/(6+2) = 1/8 per component
    # (6D uniform ball: <vi^2> = 1/(d+2) with d=6).
    # We need variance = 1 before applying Cholesky (which encodes emittance).
    scale = np.sqrt(8.0)  # = sqrt(d+2) with d=6

    particles = np.empty((n, 6), dtype=np.float64)
    particles[:, 0:2] = (scale * v[:, 0:2]) @ L_x.T
    particles[:, 2:4] = (scale * v[:, 2:4]) @ L_y.T
    particles[:, 4:6] = (scale * v[:, 4:6]) @ L_z.T

    return particles
