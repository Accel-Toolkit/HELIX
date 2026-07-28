# Marker

A zero-length, no-dynamics named position.  Used as a label for
diagnostics, snapshot triggers, or as a target for matching
constraints.

## TL;DR

```python
from linac_gen.elements.marker import Marker

mk = Marker(name="MEBT_END", snapshot=True)
```

`snapshot=True` causes the tracker to record full per-particle phase
space at this s-position (usually disabled for memory reasons; only
enabled at named locations).

## Tutorial

Markers are diagnostic: they don't change the beam, just attach a
named position to the lattice.  Common uses:

* **Reference points** for matching constraints (`SET_POSITION
  RFQ_EXIT` in TraceWin matching → constraint on the `RFQ_EXIT`
  marker).
* **Snapshot triggers** — `Beam.snapshot()` saves the full 6-D
  phase-space ndarray; combined with `Marker(snapshot=True)` you
  get a full beam dump at every named location.
* **Section boundaries** — sectioning a long lattice into named
  pieces makes diagnostics and post-processing easier.
* **BPMs** — `is_bpm=True` flags the marker as a beam-position
  monitor.  The orbit-correction driver depends on it: it resolves
  TraceWin `DIAG_POSITION N` references and filters "BPM-only"
  markers when building a response matrix.  The parser sets the
  flag automatically for `BPM` and `DIAG_POSITION` cards;
  user-constructed markers are not BPMs by default.
* **Diagnostic matching** — a `DIAG_POSITION N X Y [dm]` card also
  stores its operands on the marker: `diag_family` (the number `N`
  linking the monitor to `ADJUST N v` variables), `x_target_mm` /
  `y_target_mm` (wanted centroid; `None` = plane unconstrained,
  TraceWin's `1e50` sentinel) and `accuracy_mm` (`dm`, default 1 mm;
  residuals are weighted `1/dm`).  Targets feed both the matcher's
  auto-generated position constraints (either cost solver — the
  envelope tracks a first moment and fits them noiselessly in a
  fraction of the MP time) and target-aware orbit correction.
  An external target file loaded at runtime (`BPM targets…` in the
  Lattice tab, or `--diag-targets` on the matching CLI) overrides
  the card operands without ever being written back into the deck.
  BPM *readings* are always computed from the tracking itself —
  targets are set-points, never measurements.

### Example

```python
from linac_gen.core.lattice import Lattice
from linac_gen.elements.drift import Drift
from linac_gen.elements.marker import Marker

lat = Lattice()
lat.add(Marker(name="MEBT_START"))
# ... MEBT elements ...
lat.add(Marker(name="MEBT_END", snapshot=True))
```

After running, `results.element_names` includes `"MEBT_START"` and
`"MEBT_END"` at the corresponding s indices, and `Beam` has a saved
snapshot at MEBT_END.

### Transfer matrix

A `Marker` has no effect on the beam — its transfer matrix is
the 6×6 identity:

$$
M_{\text{marker}} = I_{6\times 6}
$$

Markers are pure metadata.

## API reference

```{.python .skip}
linac_gen.elements.marker.Marker(
    name: str,
    snapshot: bool = False,
    is_bpm: bool = False,
)
```

| Parameter | Default | Notes |
|---|---|---|
| `name` | (required) | identifier |
| `snapshot` | False | if True, recorder saves full per-particle phase space at this s |
| `is_bpm` | False | flags the marker as a BPM for the orbit-correction driver (set automatically by the parser for `BPM` / `DIAG_POSITION` cards) |
| `diag_family` | None | diagnostic number `N` from `DIAG_POSITION N …`, linking the monitor to `ADJUST N v` variables |
| `x_target_mm`, `y_target_mm` | None | wanted centroid at this monitor (mm); `None` = plane unconstrained (TraceWin `1e50`) |
| `accuracy_mm` | 1.0 | diagnostic accuracy `dm` (mm); consumers weight residuals by `1/dm` |
| `origin_keyword` | None | `"BPM"` when the marker came from a native `BPM :` card so the writer round-trips it as `BPM` |

### Source

`linac_gen/elements/marker.py:17`

← [Aperture](12_aperture.md) ·
[Continue to Steerer →](14_steerer.md)
