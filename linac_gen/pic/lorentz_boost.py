"""Lorentz boost between lab frame and beam rest frame.

The PIC solver performs an electrostatic Poisson solve in the beam rest
frame, where the bunch is elongated by gamma.  These functions convert
spatial coordinates between the two frames.

Only the longitudinal (z) coordinate is affected:
    z_rest = gamma * z_lab
    z_lab  = z_rest / gamma
Transverse coordinates (x, y) are unchanged.
"""
import numpy as np


def boost_to_rest_frame(coords: np.ndarray, gamma: float) -> np.ndarray:
    """Boost (N, 3) spatial coords [x, y, z] from lab to rest frame.

    Parameters
    ----------
    coords : np.ndarray
        (N, 3) array of [x, y, z] positions in lab frame (mm).
    gamma : float
        Lorentz factor of the beam.

    Returns
    -------
    np.ndarray
        (N, 3) array of [x, y, z] positions in rest frame (mm).
    """
    result = coords.copy()
    result[:, 2] *= gamma
    return result


def boost_to_lab_frame(coords: np.ndarray, gamma: float) -> np.ndarray:
    """Boost (N, 3) spatial coords [x, y, z] from rest frame to lab frame.

    Parameters
    ----------
    coords : np.ndarray
        (N, 3) array of [x, y, z] positions in rest frame (mm).
    gamma : float
        Lorentz factor of the beam.

    Returns
    -------
    np.ndarray
        (N, 3) array of [x, y, z] positions in lab frame (mm).
    """
    result = coords.copy()
    result[:, 2] /= gamma
    return result
