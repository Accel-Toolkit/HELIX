"""SPACE_CHARGE_COMP card placement — shared by the GUI Apply button and
the self-consistency iterator (:mod:`linac_gen.analysis.scc.iterate`).

Placement semantics are load-bearing and pinned by the popup Apply test:
cards go to the NEAREST element boundary (a searchsorted-left variant
placed cards up to a full element upstream), and cards that quantise to
the same boundary are merged by averaging their factors — otherwise one
silently shadows the other with the wrong winner.
"""
from __future__ import annotations

import numpy as np

from linac_gen.elements.space_charge_comp import SpaceChargeComp


def apply_scc_cards(lattice, cards) -> dict[int, float]:
    """Insert the analysis' suggested cards into ``lattice`` (in place).

    Existing ``SpaceChargeComp`` cards are stripped first, so repeated
    application is idempotent.  ``cards`` is the ``scc_cards`` list from
    :func:`~linac_gen.analysis.scc.driver.scc_analysis` (only ``z_m`` and
    ``factor`` are read).  Returns the ``{boundary_index: factor}`` map
    that was actually inserted (post-merge).
    """
    elements = [e for e in lattice.elements
                if not isinstance(e, SpaceChargeComp)]
    # Element BOUNDARIES incl. the line start, in metres.
    bounds = np.concatenate(([0.0], np.cumsum(
        [float(getattr(e, "length", 0.0) or 0.0)
         for e in elements]) * 1e-3))
    merged: dict[int, list] = {}
    for c in cards:
        idx = int(np.argmin(np.abs(bounds - c["z_m"])))
        merged.setdefault(idx, []).append(float(c["factor"]))
    inserted: dict[int, float] = {}
    for i, idx in enumerate(sorted(merged, reverse=True)):
        factor = float(np.clip(np.mean(merged[idx]), 0.0, 1.0))
        inserted[idx] = factor
        elements.insert(idx, SpaceChargeComp(
            name=f"SCC_{len(merged) - i}", factor=factor))
    lattice.elements[:] = elements
    return inserted
