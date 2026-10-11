"""Mid-lattice PIC grid-extent directive (``; HELIX_SC_GRID`` card).

A zero-length passive element that changes the 3-D bunched PIC solver's
grid extent (in beam sigmas) from its position onward.  Motivating case:
a combined linac+transfer-line deck where the linac converges at the
default extent but the long no-RF transfer line (large dispersive
orbits, halo) needs a much wider grid — e.g. the PIP-II BTL at 20 sigma
(``examples/MEBT_To_Foil``).  A single per-run ``grid_extent`` forced
one compromise value on both sections.

Deck syntax (HELIX-specific, TraceWin-invisible — same comment-card
pattern as ``; HELIX_FOIL``)::

    ; HELIX_SC_GRID 20

Scope: the 3-D **bunched** PIC solver only (``PicSolver.kick``).  The
DC/continuous 2-D kick models and the envelope solver's linear SC have
no sigma-scaled grid and ignore the card (the multi-particle tracker
warns once if no bunched PIC solver is active).  Several cards may
appear; each takes effect from its own position (last one wins).

``SpaceChargeConfig.grid_card`` decides whether cards apply: "apply"
(classic), "ignore", or "auto" (the named adaptive methods: ignored while
the grid follows the beam).  Every card a run reaches is recorded in the
results provenance.  A backward walk does not undo a card's extent change.
"""
from __future__ import annotations

from linac_gen.core.config import MAX_GRID_EXTENT_SIGMA, valid_grid_extent
from linac_gen.elements.base import PassiveElement


class ScGridDirective(PassiveElement):
    """Zero-length marker carrying the new PIC grid extent [sigma].

    The multi-particle tracker reads ``extent_sigma`` when it reaches
    this element and forwards it to the live ``PicSolver`` (which
    re-derives the grid on the next kick, in both fixed and adaptive
    grid modes).  The element itself never touches the beam.

    NOTE: must NOT grow an attribute named ``factor`` — the tracker's
    passive-element hook duck-types ``SpaceChargeComp`` on that name.
    """

    def __init__(self, name: str, extent_sigma: float):
        super().__init__(name=name)
        if not valid_grid_extent(extent_sigma):
            raise ValueError(
                "HELIX_SC_GRID extent must be a finite value in (0, "
                f"{MAX_GRID_EXTENT_SIGMA:g}] sigma, got {extent_sigma!r}")
        self.extent_sigma = float(extent_sigma)

    def apply(self, beam) -> None:
        """No-op: beam is unchanged.  The tracker uses ``extent_sigma``."""
