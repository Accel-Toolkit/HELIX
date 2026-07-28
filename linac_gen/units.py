"""Unit conversions and numerical constants used across the codebase.

Centralises magic numbers so that:
1. The meaning of each constant is named and documented in one place.
2. Tolerances and floors can be tuned without hunting through source files.

Physical constants (c, e, eps0, ...) live in :mod:`linac_gen.core.constants`.
This module is strictly for unit conversion factors and numerical tolerances.
"""

# -- Unit conversions -------------------------------------------------------

MM_TO_M = 1.0e-3        #: millimetre -> metre
M_TO_MM = 1.0e3         #: metre -> millimetre
MRAD_TO_RAD = 1.0e-3    #: milliradian -> radian
RAD_TO_MRAD = 1.0e3     #: radian -> milliradian
MEV_TO_EV = 1.0e6       #: megaelectronvolt -> electronvolt
MHZ_TO_HZ = 1.0e6       #: megahertz -> hertz
MA_TO_A = 1.0e-3        #: milliampere -> ampere
DEG_TO_RAD = 0.017453292519943295  #: pi / 180

# -- Numerical tolerances ---------------------------------------------------

#: Default floor for "is this value effectively zero?" comparisons on
#: field strengths, emittances, and beam sizes.  Anything below this is
#: treated as zero (drift behaviour, no kick, skip SC, etc.).
FIELD_ZERO_TOL = 1.0e-12

#: Emittance floor for sigma -> Twiss inversions.  Below this emittance,
#: Twiss parameters are considered undefined and the module returns zeros
#: rather than dividing.
EMIT_FLOOR = 1.0e-30

#: Default sigma cutoff for truncated beam distributions (Gaussian tails).
SIGMA_CUTOFF = 3.0

#: Hard cap on iterations in distribution rejection-sampling loops.
#: At ~1 000 standard-normal draws per iteration this is plenty for any
#: well-conditioned input; hitting the cap signals pathological inputs
#: (e.g. near-singular Twiss matrix, absurdly tight cutoff).
REJECTION_SAMPLING_MAX_ITERS = 1000

#: Default rcond for SVD pseudo-inverse in orbit-correction response
#: matrices.  Singular values below ``rcond * s_max`` are treated as zero.
SVD_RCOND_DEFAULT = 1.0e-10
