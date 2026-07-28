# SpaceChargeComp

A zero-length, passive marker that sets the space-charge
compensation (neutralisation) factor **from that point on** in the
lattice.  The canonical use is LEBT neutralisation: residual-gas
ionisation partially cancels the beam's self-field, and this
element tells the tracker how much.

## TL;DR

```
SPACE_CHARGE_COMP 0.95
```

```python
from linac_gen.elements.space_charge_comp import SpaceChargeComp

scc = SpaceChargeComp(name="SCC1", factor=0.95)   # 95 % neutralised
```

`factor` is the neutralisation fraction: 0 = no compensation (full
space charge), 1 = full compensation (no space-charge kick).

## Tutorial

The element itself does nothing to the beam — `apply(beam)` is a
no-op.  When the tracker encounters it, it updates its internal
space-charge scale to `1 − factor`; every subsequent SC kick
(2-D DC or 3-D PIC) is multiplied by that scale **until the next
`SpaceChargeComp` is encountered**.  Place one at the start of the
LEBT and another (`factor=0`) where neutralisation ends (e.g. at
the RFQ entrance chicane).

Two interactions to know about:

* A `FieldMap` with `Ki > 0` and a `.scc` profile **overrides** the
  global factor locally inside that map (see
  [FieldMap](07_fieldmap.md)).
* Without an active SC path (zero current or no solver configured),
  the factor is stored but has nothing to scale.

### Transfer matrix

Like any passive element, the transfer matrix is the 6×6 identity —
its entire effect is on the tracker's space-charge state.

## API reference

```{.python .skip}
linac_gen.elements.space_charge_comp.SpaceChargeComp(
    name: str,
    factor: float = 0.0,
)
```

| Parameter | Default | Notes |
|---|---|---|
| `name` | (required) | identifier |
| `factor` | 0.0 | neutralisation fraction, 0 (no compensation) to 1 (full compensation) |

### Source

`linac_gen/elements/space_charge_comp.py:18` (element),
`linac_gen/tracking/tracker.py:233-235` (the tracker applying
`_sc_factor = 1 - factor`).

## See also

* [DC mode](../05_space_charge/04_dc_mode.md) — the LEBT
  continuous-beam workflow where compensation matters most.
* [FieldMap](07_fieldmap.md) — per-element `Ki` / `.scc` override.

← [ThinLens](16_thinlens.md) ·
[Continue to Beam → Distributions →](../04_beam/01_distributions.md)
