"""KV (Kapchinsky-Vladimirsky) distribution generator.

Transverse particles lie uniformly on the surface of a 4D hyperellipsoid.
Longitudinal particles follow a 2D Gaussian distribution.
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


def generate_kv(
    n: int,
    emit_x: float, alpha_x: float, beta_x: float,
    emit_y: float, alpha_y: float, beta_y: float,
    emit_z: float, alpha_z: float, beta_z: float,
    seed: int = None,
) -> np.ndarray:
    """Generate an n-particle KV distribution.

    Transverse phase space: particles lie uniformly on the surface of a
    4D hyperellipsoid (KV distribution).  Algorithm:

    1. Generate 4 standard normal deviates z1..z4.
    2. Normalise to unit 4-sphere surface: v = z / |z|.
    3. Apply per-plane Twiss transformation (Cholesky) to give the correct
       emittance and correlations.

    For a KV distribution the single-plane halo parameter H = <u^4>/<u^2>^2 - 1
    equals 0.  The RMS emittance equals the geometric emittance.

    Longitudinal: 2D Gaussian (truncated at 3 sigma) with z-plane Twiss.

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
        Shape (n, 6) array of phase-space deviations
        [x(mm), x'(mrad), y(mm), y'(mrad), dphi(deg), dW(MeV)].
    """
    rng = np.random.default_rng(seed)

    def cholesky_2d(alpha, beta, emit):
        sigma = _sigma_matrix_2d(alpha, beta, emit)
        return np.linalg.cholesky(sigma)

    # --- Transverse: uniform on 4D sphere surface ---
    z4 = rng.standard_normal((n, 4))
    norms = np.linalg.norm(z4, axis=1, keepdims=True)
    v4 = z4 / norms  # uniform on 4-sphere surface

    # For a 4D sphere surface: <vi^2> = 1/4 per component.
    # Scale so variance = 1 before applying Cholesky.
    scale_4d = 2.0  # = sqrt(d) with d=4

    L_x = cholesky_2d(alpha_x, beta_x, emit_x)
    L_y = cholesky_2d(alpha_y, beta_y, emit_y)

    # --- Longitudinal: Gaussian (with 3-sigma cutoff) ---
    L_z = cholesky_2d(alpha_z, beta_z, emit_z)

    # Truncated Gaussian for longitudinal
    gamma_z = (1.0 + alpha_z ** 2) / beta_z
    sigma_phi = np.sqrt(beta_z * emit_z)
    sigma_w = np.sqrt(gamma_z * emit_z)
    cutoff = 3.0

    z_long = np.empty((n, 2), dtype=np.float64)
    filled = 0
    batch = max(int(n * 1.5), 1000)
    for _ in range(REJECTION_SAMPLING_MAX_ITERS):
        if filled >= n:
            break
        raw = rng.standard_normal((batch, 2))
        corr = raw @ L_z.T
        keep = (np.abs(corr[:, 0]) <= cutoff * sigma_phi) & \
               (np.abs(corr[:, 1]) <= cutoff * sigma_w)
        accepted = corr[keep]
        take = min(len(accepted), n - filled)
        z_long[filled:filled + take] = accepted[:take]
        filled += take
    else:
        raise RuntimeError(
            f"KV longitudinal rejection sampling failed to produce {n} particles "
            f"within {REJECTION_SAMPLING_MAX_ITERS} iterations (got {filled}). "
            f"Check Twiss parameters for ill-conditioning."
        )

    # Assemble
    particles = np.empty((n, 6), dtype=np.float64)
    particles[:, 0:2] = (scale_4d * v4[:, 0:2]) @ L_x.T
    particles[:, 2:4] = (scale_4d * v4[:, 2:4]) @ L_y.T
    particles[:, 4:6] = z_long

    return particles
