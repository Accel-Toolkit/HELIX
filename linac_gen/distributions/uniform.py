"""Uniform distribution generator.

Flat density inside a 6D ellipsoid — identical to the waterbag distribution.
This module provides a thin wrapper for API consistency and clarity when a
caller explicitly requests a "uniform" distribution type.
"""
import numpy as np
from linac_gen.distributions.waterbag import generate_waterbag


def generate_uniform(
    n: int,
    emit_x: float, alpha_x: float, beta_x: float,
    emit_y: float, alpha_y: float, beta_y: float,
    emit_z: float, alpha_z: float, beta_z: float,
    seed: int = None,
) -> np.ndarray:
    """Generate an n-particle 6D uniform (waterbag) distribution.

    Particles are distributed uniformly inside a 6D hyperellipsoid defined
    by the Twiss parameters.  This is equivalent to and delegates to the
    waterbag generator.

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
    return generate_waterbag(
        n,
        emit_x, alpha_x, beta_x,
        emit_y, alpha_y, beta_y,
        emit_z, alpha_z, beta_z,
        seed=seed,
    )
