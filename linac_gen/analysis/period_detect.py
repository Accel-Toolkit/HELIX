"""Auto-detect the periodic structure of a lattice.

Two complementary passes are run, and their results are concatenated
in order of confidence:

1. **LATTICE-card pass.** ``parse_tracewin`` keeps the bracket markers
   from a ``LATTICE n_cells 0 … LATTICE_END`` block as ``Marker``
   elements named ``LATTICE_<NNN>`` / ``LATTICE_END_<NNN>``, and stores
   the original numeric args on the opening marker as
   ``lattice_card_args``.  We pair them up and emit one
   :class:`PeriodicStructure` per bracket, with the inner-cell length
   computed from ``n_cells``.

2. **Type-hash autocorrelation pass.** For lattices without LATTICE
   cards (or for the *content* of a LATTICE bracket whose ``n_cells``
   was not preserved), we hash each non-trivial element to a tuple of
   ``(class_name, length, key_param)`` so that QF and QD don't collide,
   and search for the smallest period ``n`` such that the bracketed
   slice is an exact integer multiple of ``n`` element-blocks.

If neither pass finds anything, a single fallback entry covering the
whole lattice is appended so callers always have at least one option.

The returned list is ordered most-confident first.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from linac_gen.elements.aperture import Aperture
from linac_gen.elements.drift import Drift
from linac_gen.elements.lattice_commands import LatticeCommand
from linac_gen.elements.marker import Marker


# ---------------------------------------------------------------------------
# Public dataclass
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class PeriodicStructure:
    """A candidate period inside a :class:`Lattice`.

    Attributes
    ----------
    start, end : int
        Element indices ``[start, end)`` covered by the OUTER bracket.
        ``end`` is exclusive (Python-slice convention).
    inner_period_length : int
        Number of *significant* elements in one repeat unit (Markers
        and zero-length drifts excluded; matches the auto-correlation
        unit, not the raw element count).  Equal to the size of the
        slice ``lattice.elements[start:start + inner_period_length]``
        WHEN no Markers are interleaved — see ``inner_slice_end`` for
        the exact end index of the first repeat including any
        interleaved trivial elements.
    inner_slice_end : int
        Element index immediately after the first inner-period repeat,
        including any interleaved Markers.  Use this when slicing
        ``lattice.elements`` for the period.
    n_repeats : int
        How many times the inner pattern repeats inside the bracket.
    label : str
        Short human label for the GUI dropdown.
    source : str
        ``"lattice_card"`` | ``"type_sequence"`` | ``"fallback"`` —
        which detector emitted this candidate.
    repeat_spans : tuple[tuple[int, int], ...]
        Explicit ``(start, end)`` raw element-index pairs — one per
        repeat, contiguous (``spans[k][1] == spans[k+1][0]``) — built by
        walking *significant* elements.  Empty for manually-constructed
        instances; use :meth:`spans`, which falls back to the legacy
        constant-raw-stride tiling.  The explicit form is required when
        skipped elements (Markers / LatticeCommands / zero-length
        drifts) are unevenly distributed between repeats: a constant
        raw stride then stops corresponding to physical cells.
    """

    start: int
    end: int
    inner_period_length: int
    inner_slice_end: int
    n_repeats: int
    label: str
    source: str
    repeat_spans: tuple = ()

    def slice(self, elements: list) -> list:
        """Return the elements covered by the FIRST inner repeat."""
        return elements[self.start:self.inner_slice_end]

    def spans(self) -> tuple:
        """Per-repeat ``(start, end)`` element-index spans (end exclusive).

        Prefers the detector-built :attr:`repeat_spans`; falls back to
        the legacy constant-raw-stride tiling for instances constructed
        without them (manual test fixtures, old pickles).
        """
        if self.repeat_spans:
            return tuple(tuple(sp) for sp in self.repeat_spans)
        inner = self.inner_slice_end - self.start
        return tuple(
            (self.start + k * inner, self.start + (k + 1) * inner)
            for k in range(max(1, self.n_repeats))
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
_LATTICE_OPEN_RE = re.compile(r"^LATTICE_\d+$")
_LATTICE_CLOSE_RE = re.compile(r"^LATTICE_END_\d+$")


def _is_marker(elem) -> bool:
    return isinstance(elem, Marker)


def _is_lattice_open(elem) -> bool:
    return _is_marker(elem) and bool(_LATTICE_OPEN_RE.match(getattr(elem, "name", "")))


def _is_lattice_close(elem) -> bool:
    return _is_marker(elem) and bool(_LATTICE_CLOSE_RE.match(getattr(elem, "name", "")))


def _is_significant(elem) -> bool:
    """Skip elements TraceWin doesn't count toward a LATTICE cell.

    Per the TraceWin manual (LATTICE command), the following are NOT
    counted: ``DIAG_XXX``, ``APERTURE``, ``THIN_STEERING``.  In HELIX:

      * ``DIAG_XXX`` → ``Marker``           (skipped via _is_marker)
      * ``APERTURE`` → ``Aperture``         (skipped here)
      * sub-LATTICE directives → ``LatticeCommand`` (skipped here)

    Zero-length Drifts are also skipped — they contribute no transport.
    """
    if _is_marker(elem):
        return False
    # isinstance, not exact type-name matching: LatticeCommand has many
    # subclasses (SetSyncPhase, SetTwiss, …) that TraceWin does not count
    # toward a LATTICE cell.  The old ``type(elem).__name__`` comparison
    # missed every subclass, so e.g. the PIP-II HWR ``LATTICE 4 0`` block
    # (solenoid–drift–cavity–drift + one SET_SYNC_PHASE per cell) was
    # detected as 10 five-element cells instead of 8 four-element cells.
    if isinstance(elem, Drift) and not getattr(elem, "length", 0):
        return False
    if isinstance(elem, (Aperture, LatticeCommand)):
        return False
    return True


def _key_param(elem) -> float:
    """Hashing tie-breaker — the dominant parameter of an element type.

    We round to 6 dp so floating-point fuzz doesn't break repeat
    detection on lattices that came through TraceWin write/read cycles.
    """
    type_name = type(elem).__name__
    for attr in (
        "gradient",        # Quadrupole
        "field",           # Solenoid
        "Bz",              # Solenoid alternate
        "voltage",         # RFGap
        "voltage_V",       # RfqCell
        "angle",           # Dipole
        "amplitude",       # RFGap (legacy)
        "factor",          # SpaceChargeComp
        "ke",              # FieldMap*
    ):
        v = getattr(elem, attr, None)
        if isinstance(v, (int, float)):
            return round(float(v), 6)
    return 0.0


def _hash(elem) -> tuple:
    return (
        type(elem).__name__,
        round(float(getattr(elem, "length", 0.0) or 0.0), 6),
        _key_param(elem),
    )


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
def detect_periods(lattice) -> list[PeriodicStructure]:
    """Return ranked period candidates for ``lattice``.

    The list is never empty — at minimum, the whole lattice is
    returned as a single ``"fallback"`` entry.
    """
    if lattice is None or not lattice.elements:
        return []

    out: list[PeriodicStructure] = []

    # ---- Pass 1: LATTICE / LATTICE_END markers ----------------------------
    open_idx: list[int] = []
    pairs: list[tuple[int, int]] = []
    for i, el in enumerate(lattice.elements):
        if _is_lattice_open(el):
            open_idx.append(i)
        elif _is_lattice_close(el) and open_idx:
            pairs.append((open_idx.pop(), i))

    for o, c in pairs:
        # The opening marker carries the original numeric args from the
        # ``LATTICE n1 n2`` card.  Per the TraceWin manual, n1 is the
        # number of *significant* elements per basic-lattice cell — NOT
        # the number of cells.  The cell count is derived inside
        # ``_build_from_cell_count`` from sig_count // n1.
        args = getattr(lattice.elements[o], "lattice_card_args", []) or []
        n_per_cell = int(args[0]) if args else 0

        # Slice the *content* between the brackets (exclusive of the
        # markers themselves).
        inner_start = o + 1
        inner_end = c
        if inner_end <= inner_start:
            continue

        if n_per_cell > 0:
            structure = _build_from_cell_count(
                lattice.elements, inner_start, inner_end, n_per_cell, source="lattice_card"
            )
            if structure is not None:
                out.append(structure)
                continue

        # n_per_cell unknown / inconsistent → fall through to autocorrelation.
        structure = _build_from_autocorrelation(
            lattice.elements, inner_start, inner_end, source="lattice_card_recovered"
        )
        if structure is not None:
            out.append(structure)

    # ---- Pass 2: autocorrelation over the whole lattice (skipped if
    # we already found a LATTICE bracket that covers everything). ---------
    have_full_bracket = any(
        s.start == 0 and s.end == len(lattice.elements) for s in out
    )
    if not have_full_bracket:
        full = _build_from_autocorrelation(
            lattice.elements, 0, len(lattice.elements), source="type_sequence"
        )
        if full is not None:
            out.append(full)

    # ---- Pass 3: always-available fallback --------------------------------
    out.append(PeriodicStructure(
        start=0,
        end=len(lattice.elements),
        inner_period_length=len(lattice.elements),
        inner_slice_end=len(lattice.elements),
        n_repeats=1,
        label="(whole lattice)",
        source="fallback",
    ))
    return out


# ---------------------------------------------------------------------------
# Detector internals
# ---------------------------------------------------------------------------
def _build_from_cell_count(
    elements: list, start: int, end: int, n_per_cell: int, *, source: str
) -> PeriodicStructure | None:
    """Trust the explicit ``n1`` from a LATTICE card.

    Per the TraceWin manual: ``LATTICE n1 n2`` declares ``n1``
    significant elements per *basic lattice* cell.  ``n2`` is
    cells-per-macro-lattice and is currently ignored by HELIX.  The
    number of cells inside the bracket is *implicit*:
    ``n_cells = sig_count // n1``.

    Counts only *significant* elements when sizing the cell —
    Markers / zero-length drifts / Apertures / LatticeCommands inside
    the bracket are passed through transparently (matches TraceWin's
    own exclusion list: ``DIAG_XXX``, ``APERTURE``, ``THIN_STEERING``).

    **Transition-tolerant** since 2026-05-11: when
    ``sig_count % n_per_cell != 0`` we treat the leading
    ``residue = sig_count − n_cells * n_per_cell`` significant elements
    as a *transition* region (entry-matching quads/drifts that don't
    repeat as part of the canonical cell).  The reported ``start``
    shifts past the transition to the first true cell's element.  This
    keeps SSR1/SSR2/LB650/HB650 brackets in real PIP-II / XFEL-style
    .dats analyzable — those cryomodules carry matching elements at
    the bracket entrance whose count never divides ``n_per_cell``.
    """
    sig_count = sum(1 for e in elements[start:end] if _is_significant(e))
    if sig_count == 0 or n_per_cell <= 0:
        return None
    n_cells = sig_count // n_per_cell
    if n_cells <= 0:
        # Bracket holds fewer significant elements than a single cell —
        # the LATTICE card's declared cell size is bigger than the
        # whole bracket.
        return None
    residue = sig_count - n_cells * n_per_cell
    # Shift ``start`` past the leading transition residue so the first
    # cell aligns at the new ``start``.  ``_advance_significant`` walks
    # the element array forward by ``residue`` significant elements
    # (skipping markers / zero-length drifts).  It expects k>=1 and
    # returns ``len(elements)`` for k=0 — handle that explicitly.
    cell_start = start if residue == 0 else _advance_significant(
        elements, start, residue)
    # Explicit contiguous per-repeat spans: each cell ends immediately
    # after its n_per_cell-th significant element, and the next cell
    # starts right there.  This stays correct when Markers / commands
    # are unevenly interleaved (constant raw stride does not).
    repeat_spans: list[tuple[int, int]] = []
    cur = cell_start
    for _ in range(n_cells):
        nxt = _advance_significant(elements, cur, n_per_cell)
        repeat_spans.append((cur, nxt))
        cur = nxt
    inner_slice_end = repeat_spans[0][1]
    label = f"LATTICE bracket · {n_cells} × {n_per_cell}-element cell"
    if residue > 0:
        label += f"  (+{residue} transition)"
    return PeriodicStructure(
        start=cell_start,
        end=end,
        inner_period_length=n_per_cell,
        inner_slice_end=inner_slice_end,
        n_repeats=n_cells,
        label=label,
        source=source,
        repeat_spans=tuple(repeat_spans),
    )


def _build_from_autocorrelation(
    elements: list, start: int, end: int, *, source: str
) -> PeriodicStructure | None:
    """Find the smallest period that exactly tiles the slice."""
    sig: list[tuple] = []
    sig_idx: list[int] = []   # original element-index for each sig entry
    for i in range(start, end):
        if _is_significant(elements[i]):
            sig.append(_hash(elements[i]))
            sig_idx.append(i)
    n = len(sig)
    if n < 2:
        return None

    best: PeriodicStructure | None = None
    best_score = -1
    # Try every period that exactly divides the slice.
    for p in range(1, n // 2 + 1):
        if n % p != 0:
            continue
        if all(sig[i] == sig[i % p] for i in range(p, n)):
            n_reps = n // p
            score = n_reps * p   # = n; tie-break by smaller p (loop order favours that)
            if best is None or score > best_score:
                # Map back from sig-index to element-index.  Spans are
                # contiguous: cell k ends right after its last
                # significant element, and cell k+1 starts there.
                inner_slice_end = sig_idx[p - 1] + 1
                repeat_spans: list[tuple[int, int]] = []
                cur = start
                for k in range(n_reps):
                    nxt = sig_idx[(k + 1) * p - 1] + 1
                    repeat_spans.append((cur, nxt))
                    cur = nxt
                best = PeriodicStructure(
                    start=start,
                    end=end,
                    inner_period_length=p,
                    inner_slice_end=inner_slice_end,
                    n_repeats=n_reps,
                    label=f"{n_reps} × {p}-element cell",
                    source=source,
                    repeat_spans=tuple(repeat_spans),
                )
                best_score = score
                # Smallest p that works wins; stop as soon as we have one.
                break
    return best


def _advance_significant(elements: list, start: int, k: int) -> int:
    """Return the element index immediately after the ``k``-th
    significant element starting at ``start``."""
    seen = 0
    for i in range(start, len(elements)):
        if _is_significant(elements[i]):
            seen += 1
            if seen == k:
                return i + 1
    return len(elements)
