"""Beam-input error specs (TraceWin's ``ERROR_BEAM_STAT`` family).

Distinct from ``ErrorDef`` (which perturbs lattice elements), a
``BeamErrorDef`` perturbs the **input beam** before it enters the
lattice — centroid offsets, emittance scaling, Twiss mismatch, and
beam-current jitter.  Each error study seed draws fresh values per
spec and applies them to a copy of the user's BeamConfig before
``create_beam`` is called.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class BeamErrorDef:
    """One per-seed perturbation on the input beam.

    Parameters
    ----------
    parameter
        Which BeamConfig field to perturb.  Supported:

        * Centroid: ``centroid_x, centroid_xp, centroid_y, centroid_yp,
          centroid_dphi, centroid_dw``
        * Emittance scale (multiplicative): ``emit_nx_rel, emit_ny_rel,
          emit_z_rel``  — each multiplies the design ε by ``(1+δ)``
        * Twiss mismatch (additive % adjustment, like the existing
          ``mismatch_x``): ``mismatch_x, mismatch_y, mismatch_z``
        * Beam current: ``current_rel`` — multiplies design current
          by ``(1+δ)``

    distribution
        ``"gaussian"`` (use ``sigma`` + ``cutoff``) or ``"uniform"``
        (use ``half_width``).
    sigma, half_width, cutoff
        Distribution params.  ``cutoff`` only applies to gaussian.
    """
    parameter: str
    distribution: str = "gaussian"
    sigma: float = 0.0
    half_width: float = 0.0
    cutoff: float = 3.0
