"""Arbitrary linear transfer-map element.

Stores a raw 6x6 transfer matrix (optionally with an affine centroid
offset) and applies it verbatim — the HELIX analogue of Cheetah's
``CustomTransferMap``.  It exists so importers for external formats can
faithfully carry element types that encode an explicit map rather than
physical parameters: Elegant ``EMATRIX`` / ``ILMATRIX``, Bmad
``taylor`` / ``match`` (linear part), Ocelot ``Matrix``.

Coordinate basis and unit note
------------------------------
The matrix acts on HELIX phase-space rows
``[x_mm, x'_mrad, y_mm, y'_mrad, dphi_deg, dW_MeV]``.  The transverse
4x4 block is **unit-invariant** between the (m, rad) basis most codes
export and HELIX's (mm, mrad) — mm/mrad == m/rad — so a transverse map
imports numerically unchanged.  The longitudinal block, however, is in a
code-specific basis (Elegant ``(t[s], p=βγ)``, Bmad ``(z, pz)``): a map
with *non-trivial* longitudinal coupling would need a basis change that
is NOT applied here.  Importers should warn when the longitudinal block
is non-identity; a pure transverse (or identity-longitudinal) matrix —
the common case — is exact.

The map is energy-agnostic (the same at any ``ref`` energy), matching
Cheetah; it therefore cannot be re-scaled to a different reference energy
and cannot be sub-sliced (``n_steps == 1``).  Only its phase units follow
the run's clock (below).

Clock: ``dphi_deg`` is in degrees of an RF clock.  A map converted into
HELIX's basis at a known clock (a MAD-X ``MATRIX``, whose ``t`` holds none)
records it as ``clock_mhz``; run at another frequency it is
``D M D⁻¹`` (and its offset ``D o``), ``D[4, 4]`` = run frequency /
``clock_mhz``, so it gives the same map in the run's degrees.  With
``clock_mhz = 0`` (the default) the map is taken in the run's degrees.

Reference gain: ``ref_gain`` (MeV) moves HELIX's reference energy through
the element (a linearised cavity, a MAD-X ``MATRIX`` whose ``kick6`` gains
energy); the map then takes the particles from the entrance reference to
the exit one, as a cavity's does.  ``offset[5]`` instead shifts every
particle's energy and leaves the reference where it was.
"""
from __future__ import annotations

import numpy as np

from linac_gen.elements.base import TransferMapElement
from linac_gen.core.reference import ReferenceParticle
from linac_gen.core.beam import Beam


class MatrixElement(TransferMapElement):
    """Element defined by an explicit 6x6 transfer matrix (+ optional
    affine centroid offset)."""

    # The matrix is fixed at construction; disambiguate cache entries by
    # element identity rather than by a (non-scalar) matrix fingerprint
    # (and by the clock it is counted on).
    _cache_keys: tuple[str, ...] = ("clock_mhz",)

    def __init__(self, name: str, matrix, length: float = 0.0,
                 offset=None, aperture: float = 0.0, clock_mhz: float = 0.0,
                 ref_gain: float = 0.0):
        super().__init__(name=name, length=float(length),
                         aperture=aperture, n_steps=1)
        M = np.asarray(matrix, dtype=float)
        if M.shape != (6, 6):
            raise ValueError(
                f"MatrixElement '{name}': matrix must be 6x6, got {M.shape}")
        self.matrix = M
        self.offset = (None if offset is None
                       else np.asarray(offset, dtype=float).reshape(6))
        self.clock_mhz = float(clock_mhz)    # the clock dphi is counted on (0: the run's)
        self.ref_gain = float(ref_gain)      # MeV the reference gains through it

    def map_at(self, ref: ReferenceParticle):
        """The map and offset in degrees of ``ref``'s frequency:
        ``(D M D⁻¹, D o)`` with ``D[4, 4]`` = ``ref.frequency / clock_mhz``
        (the stored ones, unchanged, at that clock or with no clock)."""
        f = float(getattr(ref, "frequency", 0.0) or 0.0)
        if not (self.clock_mhz > 0.0 and f > 0.0 and f != self.clock_mhz):
            return self.matrix, self.offset
        r = f / self.clock_mhz
        M = self.matrix.copy()
        M[4, :4] *= r                       # D M D⁻¹: row 4 times r, column 4
        M[4, 5] *= r                        # over r, M44 as it is
        M[:4, 4] /= r
        M[5, 4] /= r
        o = None
        if self.offset is not None:
            o = self.offset.copy()
            o[4] *= r
        return M, o

    def transfer_matrix(self, ref: ReferenceParticle,
                        ds: float = None) -> np.ndarray:
        # Energy-agnostic: the stored map, in the run's degrees.  A general
        # matrix cannot be sub-sliced, so ``ds`` is accepted (tracker API)
        # but does not scale the map.
        return self.map_at(ref)[0]

    def track(self, beam: Beam, ds: float = None) -> None:
        L = ds if ds is not None else self.length
        # Advance the reference along the element as a drift of length L
        # (the map acts on deviations about an unchanged reference).
        if L:
            beam.ref.s += L
            beam.ref.phi_s += 360.0 * L / (beam.ref.beta
                                           * beam.ref.wavelength)
        alive = beam.alive_mask
        M, o = self.map_at(beam.ref)
        p = (M @ beam.particles[alive].T).T
        if o is not None:
            p = p + o
        beam.particles[alive] = p
        if self.ref_gain:
            beam.ref.w_kin = beam.ref.w_kin + self.ref_gain
