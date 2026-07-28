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

The map is energy-agnostic (returned as-is regardless of ``ref``),
matching Cheetah; it therefore cannot be re-scaled to a different
reference energy and cannot be sub-sliced (``n_steps == 1``).
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
    # element identity rather than by a (non-scalar) matrix fingerprint.
    _cache_keys: tuple[str, ...] = ()

    def __init__(self, name: str, matrix, length: float = 0.0,
                 offset=None, aperture: float = 0.0):
        super().__init__(name=name, length=float(length),
                         aperture=aperture, n_steps=1)
        M = np.asarray(matrix, dtype=float)
        if M.shape != (6, 6):
            raise ValueError(
                f"MatrixElement '{name}': matrix must be 6x6, got {M.shape}")
        self.matrix = M
        self.offset = (None if offset is None
                       else np.asarray(offset, dtype=float).reshape(6))

    def transfer_matrix(self, ref: ReferenceParticle,
                        ds: float = None) -> np.ndarray:
        # Energy-agnostic: the stored map is returned verbatim.  A general
        # matrix cannot be sub-sliced, so ``ds`` is accepted (tracker API)
        # but does not scale the map.
        return self.matrix

    def track(self, beam: Beam, ds: float = None) -> None:
        L = ds if ds is not None else self.length
        # Advance the reference along the element as a drift of length L
        # (the map acts on deviations about an unchanged reference).
        if L:
            beam.ref.s += L
            beam.ref.phi_s += 360.0 * L / (beam.ref.beta
                                           * beam.ref.wavelength)
        alive = beam.alive_mask
        p = (self.matrix @ beam.particles[alive].T).T
        if self.offset is not None:
            p = p + self.offset
        beam.particles[alive] = p
