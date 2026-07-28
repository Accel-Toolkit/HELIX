"""Period detection: LatticeCommand subclasses are not cell elements, and
per-repeat spans are explicit.

Two verified bugs pinned here:

1. ``_is_significant`` compared ``type(elem).__name__ == "LatticeCommand"``
   — every SUBCLASS (SetSyncPhase, …) was counted as a significant cell
   element, so the PIP-II HWR ``LATTICE 4 0`` block (8 cells of
   solenoid–drift–cavity–drift, each with one SET_SYNC_PHASE and one BPM)
   was detected as 10 five-element cells with wrong boundaries.

2. Consumers tiled cells by a CONSTANT raw stride
   (``inner_slice_end − start``), which stops corresponding to physical
   cells when skipped elements (markers / commands) are unevenly
   distributed between repeats.  ``PeriodicStructure.repeat_spans`` /
   ``spans()`` provide the explicit walk.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from linac_gen.analysis.period_detect import (
    PeriodicStructure, detect_periods, _is_significant,
)
from linac_gen.core.lattice import Lattice
from linac_gen.elements.drift import Drift
from linac_gen.elements.lattice_commands import SetSyncPhase
from linac_gen.elements.marker import Marker
from linac_gen.elements.quadrupole import Quadrupole

_REPO = Path(__file__).resolve().parents[2]
_HWR_DAT = _REPO / "examples" / "mebt_plus_hwr.dat"
_HWR_SOL_FIELD = _REPO / "Fields" / "HWR-SOL-ANLMAP.bsz"


def test_lattice_command_subclasses_are_not_significant():
    """isinstance, not exact-type-name: SetSyncPhase must be skipped."""
    assert not _is_significant(SetSyncPhase(name="SSP"))
    assert not _is_significant(Marker(name="BPM_1"))
    assert _is_significant(Drift(name="D", length=10.0, aperture=10.0))
    assert not _is_significant(Drift(name="D0", length=0.0, aperture=10.0))


def _uneven_lattice() -> Lattice:
    """3 identical 4-element cells, but markers/commands distributed
    UNEVENLY between the repeats — a constant raw stride cannot tile
    this correctly."""
    lat = Lattice()
    for n in range(3):
        lat.add(Drift(name="D", length=100.0, aperture=10.0))
        lat.add(Quadrupole(name="QF", length=50.0, gradient=+10.0,
                           aperture=10.0))
        if n == 0:
            # Two skipped elements in cell 1 only.
            lat.add(Marker(name="BPM_0", is_bpm=True))
            lat.add(SetSyncPhase(name="SSP_0"))
        lat.add(Drift(name="D", length=100.0, aperture=10.0))
        lat.add(Quadrupole(name="QD", length=50.0, gradient=-10.0,
                           aperture=10.0))
        if n == 2:
            lat.add(SetSyncPhase(name="SSP_2"))
    return lat


def test_uneven_commands_produce_correct_spans():
    lat = _uneven_lattice()
    periods = detect_periods(lat)
    cell = next(p for p in periods if p.n_repeats >= 3)
    assert cell.n_repeats == 3
    assert cell.inner_period_length == 4

    spans = cell.spans()
    assert len(spans) == 3
    # Contiguous tiling.
    for k in range(len(spans) - 1):
        assert spans[k][1] == spans[k + 1][0]
    # Each span holds exactly 4 significant elements.
    for a, b in spans:
        sig = [e for e in lat.elements[a:b] if _is_significant(e)]
        assert len(sig) == 4, (a, b, [type(e).__name__ for e in lat.elements[a:b]])
        assert [type(e).__name__ for e in sig] == [
            "Drift", "Quadrupole", "Drift", "Quadrupole"]
    # A constant raw stride (inner = 6, set by cell 1's two extras)
    # would tile (0,6),(6,12),(12,18) — wrong from cell 2's END on.
    inner = cell.inner_slice_end - cell.start
    assert spans[1][1] != cell.start + 2 * inner
    assert spans[2][0] != cell.start + 2 * inner
    # First span must agree with the legacy first-cell fields.
    assert spans[0] == (cell.start, cell.inner_slice_end)


def test_manual_period_spans_fallback_is_constant_stride():
    p = PeriodicStructure(start=2, end=14, inner_period_length=3,
                          inner_slice_end=5, n_repeats=4,
                          label="manual", source="manual")
    assert p.spans() == ((2, 5), (5, 8), (8, 11), (11, 14))


@pytest.mark.skipif(
    not (_HWR_DAT.exists() and _HWR_SOL_FIELD.parent.exists()),
    reason="PIP-II example lattice / field maps not present",
)
def test_hwr_lattice_card_detects_8x4_cells():
    """The real PIP-II HWR bracket: 8 cells × 4 significant elements
    (solenoid FieldMap, Drift, cavity FieldMap3D, Drift) — previously
    detected as 10 × 5 because SET_SYNC_PHASE was counted."""
    import warnings
    from linac_gen.io.tracewin_parser import parse_tracewin
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        lat, _ = parse_tracewin(str(_HWR_DAT))
    cards = [p for p in detect_periods(lat) if p.source == "lattice_card"]
    assert cards, "no LATTICE-card period found"
    hwr = cards[0]
    assert hwr.n_repeats == 8
    assert hwr.inner_period_length == 4
    spans = hwr.spans()
    assert len(spans) == 8
    for k in range(7):
        assert spans[k][1] == spans[k + 1][0]
    for a, b in spans:
        sig = [type(e).__name__ for e in lat.elements[a:b]
               if _is_significant(e)]
        assert sig == ["FieldMap", "Drift", "FieldMap3D", "Drift"]
