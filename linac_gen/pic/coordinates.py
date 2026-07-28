"""Coordinate conversion between beam phase-space and spatial coordinates.

The beam stores particles as deviations [x(mm), xp(mrad), y(mm), yp(mrad),
dphi(deg), dW(MeV)].  The PIC solver needs spatial [x, y, z] in mm.

Transverse x, y are already in mm.  Longitudinal conversion:
    z_lab = -dphi * beta_ref * wavelength / 360.0
Sign convention: negative dphi (particle ahead in phase) = positive z
(ahead in space).
"""
import numpy as np


def beam_to_spatial(beam) -> np.ndarray:
    """Extract (N_alive, 3) spatial array [x, y, z] in mm from beam deviations.

    Parameters
    ----------
    beam : Beam
        Beam object with .particles, .alive_mask, and .ref attributes.

    Returns
    -------
    np.ndarray
        (N_alive, 3) array of [x, y, z] positions in mm.
    """
    alive = beam.alive_mask
    x = beam.particles[alive, 0]      # mm
    y = beam.particles[alive, 2]      # mm
    dphi = beam.particles[alive, 4]   # deg
    z = -dphi * beam.ref.beta * beam.ref.wavelength / 360.0  # mm
    return np.column_stack([x, y, z])


def spatial_z_to_dphi(z, ref) -> np.ndarray:
    """Convert z position (mm) back to dphi (deg).

    Parameters
    ----------
    z : float or np.ndarray
        Longitudinal position(s) in mm.
    ref : ReferenceParticle
        Reference particle providing beta and wavelength.

    Returns
    -------
    float or np.ndarray
        Phase deviation(s) in degrees.
    """
    return -z * 360.0 / (ref.beta * ref.wavelength)
