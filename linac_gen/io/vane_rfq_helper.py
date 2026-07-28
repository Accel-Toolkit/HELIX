"""Helpers for opting into the :class:`VaneRFQ` Toutatis-equivalent path.

Workflow:

1. Parse a TraceWin ``.dat`` lattice with the standard
   :func:`linac_gen.io.tracewin_parser.parse_tracewin` — RFQ_CELL lines
   become :class:`RfqCell` elements as usual, leaving every existing
   test/lattice path unchanged.
2. Call :func:`replace_rfq_cells_with_vane` with the parsed
   :class:`Lattice` and the path to a sibling ``.vane`` file.  Any
   contiguous chain of :class:`RfqCell` elements is replaced in-place by
   a single :class:`VaneRFQ` whose per-z geometry comes from the
   ``.vane`` file.  Other elements (drifts, solenoids, RF gaps, etc.)
   are left untouched.

This keeps the alternative tracker isolated to lattices where the user
explicitly opts in.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Union

from linac_gen.core.lattice import Lattice
from linac_gen.elements.rfq_cell import RfqCell
from linac_gen.elements.vane_rfq import CellSpan, VaneRFQ
from linac_gen.io.tracewin_vane import VaneGeometry, parse_vane_file


def cell_spans_from_rfq_chain(rfq_cells: List[RfqCell],
                              z_offset_mm: float = 0.0) -> List[CellSpan]:
    """Convert an ordered list of :class:`RfqCell` into :class:`CellSpan`.

    The first cell starts at ``z_offset_mm`` (default ``0.0``), each
    subsequent cell starts at the previous cell's exit, so the resulting
    span list is contiguous and covers ``sum(c.length for c in rfq_cells)``.
    """
    spans: List[CellSpan] = []
    z = float(z_offset_mm)
    for c in rfq_cells:
        spans.append(CellSpan(
            z_start_mm=z,
            z_end_mm=z + c.length,
            voltage_V=c.voltage_V,
            A10=c.A10,
            modulation=c.modulation,
            length_mm=c.length,
            phi_s_deg=c.phi_s_deg,
            cell_type=c.cell_type,
            type_prev=c.type_prev,
            type_next=c.type_next,
            r0_dat_mm=c.r0_mm,
            Tc_mm=c.Tc_mm,
            dP_deg=c.dP_deg,
        ))
        z += c.length
    return spans


def replace_rfq_cells_with_vane(lattice: Lattice,
                                vane: Union[str, Path, VaneGeometry],
                                name: str = "VANE_RFQ",
                                n_steps: int | None = None,
                                field_model: str = "2term",
                                laplace_cache=None,
                                laplace_kwargs: dict | None = None,
                                ) -> Lattice:
    """Replace contiguous RfqCell chains in ``lattice`` with one VaneRFQ.

    Parameters
    ----------
    lattice : Lattice
        Lattice produced by the standard parser.  Modified in place; the
        same instance is returned for chaining.
    vane : str | Path | VaneGeometry
        ``.vane`` file path, or an already-parsed :class:`VaneGeometry`.
    name : str
        Name to assign to the resulting :class:`VaneRFQ` element.  When
        the lattice has multiple separate RfqCell chains (rare), the
        name is suffixed with ``_2``, ``_3``, ….
    n_steps : int, optional
        Total substeps for the resulting :class:`VaneRFQ` element.
        ``None`` lets :class:`VaneRFQ` pick its default (~2× vane
        resolution).
    field_model : {"2term", "8term", "laplace2d"}, optional
        Forwarded to :class:`VaneRFQ`.  Default ``"2term"`` is the M1
        bit-identical RfqCell-equivalent path.  ``"8term"`` engages the
        matcher-aware per-z r₀ refinement.  ``"laplace2d"`` engages the
        M3 numerical 2-D Laplace per z slice (true Toutatis equivalent).
    laplace_cache : Laplace2DCache, optional
        Pre-built cache shared across multiple VaneRFQ elements (or
        repeated runs on the same .vane).  Skips the per-element
        Laplace solve at construction.  Ignored unless
        ``field_model="laplace2d"``.
    laplace_kwargs : dict, optional
        Forwarded to :class:`Laplace2DCache`'s constructor (``nx``,
        ``ny``, ``box_factor``, ``z_subsample``, ``verbose``).  Ignored
        unless ``field_model="laplace2d"`` and no ``laplace_cache`` is
        supplied.

    Returns
    -------
    lattice : Lattice
        The same lattice instance, with RfqCell chains replaced.

    Raises
    ------
    ValueError
        If the lattice contains zero RfqCell elements (nothing to replace).
    """
    if isinstance(vane, (str, Path)):
        vane_geom = parse_vane_file(vane)
    else:
        vane_geom = vane

    new_elements: list = []
    cell_chain: List[RfqCell] = []
    # Zero-length passive elements (markers, DIAG_*, lattice/lattice_end
    # directives) that fall *between* RfqCells don't break the chain —
    # they have no physical effect, but we keep them in the new lattice
    # by emitting them right after the consolidated VaneRFQ so their
    # s-position lands at the RFQ exit (close enough for diagnostics).
    pending_intra_chain_markers: list = []
    chain_count = 0

    def flush():
        nonlocal chain_count
        if not cell_chain:
            # No chain to flush — but if any intra-chain markers were
            # queued before any RfqCell appeared, those are *before* the
            # chain, not inside it.  Emit them in place.
            new_elements.extend(pending_intra_chain_markers)
            pending_intra_chain_markers.clear()
            return
        chain_count += 1
        spans = cell_spans_from_rfq_chain(cell_chain)
        elem_name = name if chain_count == 1 else f"{name}_{chain_count}"
        elem = VaneRFQ(name=elem_name, vane=vane_geom,
                       cells=spans, n_steps=n_steps,
                       field_model=field_model,
                       laplace_cache=laplace_cache,
                       laplace_kwargs=laplace_kwargs)
        new_elements.append(elem)
        cell_chain.clear()
        # Emit intra-chain markers right after the VaneRFQ.
        new_elements.extend(pending_intra_chain_markers)
        pending_intra_chain_markers.clear()

    for el in lattice.elements:
        if isinstance(el, RfqCell):
            cell_chain.append(el)
        elif getattr(el, "length", 0.0) == 0.0 and cell_chain:
            # Zero-length element inside an active chain — defer.
            pending_intra_chain_markers.append(el)
        else:
            flush()
            new_elements.append(el)
    flush()

    if chain_count == 0:
        raise ValueError(
            "replace_rfq_cells_with_vane: lattice has no RfqCell elements"
        )

    lattice.elements = new_elements
    return lattice
