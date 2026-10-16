# Synthetic RFQ demonstration

A complete, runnable RFQ that turns a DC beam into a bunched,
accelerated one — **75 keV → ~1.9 MeV protons at 352.21 MHz over
2.01 m**. Lives in `examples/rfq_demo/`.

```bash
cd examples/rfq_demo
python3 make_rfq_demo.py      # regenerate the deck from the design
```

Open `rfq_demo.lgproj` in the GUI, or drive it from Python exactly as
in [Basic FODO](01_basic_fodo.md) with `rfq_demo.dat`.

## The design

Not a copy of any existing machine: every per-cell number is generated
from textbook two-term RFQ relations, and nothing is fitted to any
TraceWin output. It is a conventional four-section RFQ — radial
matcher, shaper, gentle buncher, accelerator, exit matcher:

| | |
|---|---|
| cells | 199 |
| r₀ | 3.40 mm |
| vane voltage | 85 kV |
| modulation `m` | 1 → 1.95 |
| synchronous phase | −90° → −28° |
| focusing parameter `B = qVλ²/(mc²r₀²)` | 5.7 (usual range 4–8) |

Cell length is `L = βλ/2`, with β advanced from the energy the model
itself produces, so the deck is synchronous with the code that runs
it. `A₁₀` is solved from `(r₀, m, L)` through the Crandall/Wangler
two-term relation and iterated to its fixed point, so each card's
triplet is internally consistent — the same relation
[`modulation_consistency`](../03_elements/09_rfqcell.md) uses to
cross-check cards.

## Results

2000 particles (seed 42), matched input (α = 0, β = 0.01 mm/mrad,
ε_n = 0.20 π·mm·mrad), with each RFQ solver:

| | RFQ solver | transmission | exit energy |
|---|---|---|---|
| no space charge | time domain (the default) | 91.1 % | 1.913 MeV ± 27 keV |
| 15 mA, 32³ adaptive PIC | time domain (the default) | 84.8 % | 1.917 MeV ± 22 keV |
| no space charge | per cell | 80.8 % | 1.899 MeV ± 28 keV |
| 15 mA, 32³ adaptive PIC | per cell | 76.0 % | 1.898 MeV ± 26 keV |

Exit energy: mean ± rms of the accelerated particles (above 0.9 of the
reference energy).  With the per-cell model every surviving particle
is accelerated; the time-domain solver also brings a few unaccelerated
particles to the exit (0.3 % of the beam without space charge, 1.1 % at
15 mA).  They count in the transmission, but by default not in the
recorded rms sizes, emittances and energy spread (see
[statistics of the accelerated beam](../03_elements/20_rfq_time_domain.md#physics)).
The deck has no `RFQ_GEOM` vane file, so the
[time-domain solver](../03_elements/20_rfq_time_domain.md) builds the
vanes from the `RFQ_CELL` cards; its first run builds their field map
(about 2.5 minutes) and caches it.

!!! note "A teaching deck, not an optimised design"
    A production RFQ iterates the gentle-buncher ramp for >95 %
    capture. Here the remaining loss is longitudinal, concentrated in
    the buncher where the bucket shrinks faster than the beam can
    follow adiabatically. Raising `N_BUNCHER` and the phase-ramp
    exponent in `make_rfq_demo.py` improves it — which is why the
    generator ships alongside the deck rather than the deck alone.

## What it exercises

* **The time-domain RFQ solver**, the default for multiparticle runs
  (`rfq_solver = "auto"`): the whole RFQ tracked in time through the
  3-D field of vanes built from the cards.  See
  [Time-domain RFQ solver](../03_elements/20_rfq_time_domain.md).
* **`field_model="tw2term"`**, the per-cell model's default field model
  (`rfq_solver = "cells"`) — per-particle longitudinal phase slip and
  vane-tip aperture losses, without which a DC beam cannot bunch at all.
  See [RFQ cell](../03_elements/09_rfqcell.md).
* **The DC → bunched transition**, taken automatically at the first
  cell.
* **[`periodic_phase`](../04_beam/03_beam_config.md)**, enabled in the
  project file. An RFQ makes one bunch per RF period while the
  simulation seeds one period, so without it the particles space
  charge pushes across a bucket boundary sit a full spacing away and
  inflate every reported σ_φ and ε_z. Set it `false` in
  `rfq_demo.lgproj` to see the difference.
* **Space charge through an RFQ** — the 15 mA row above, the hardest
  place for the PIC path (tight bore, strong bunching).

The per-cell model on this deck is pinned by
`tests/rfq/test_rfq_demo_example.py`, which — unlike the rest of
`tests/rfq/` — needs no external reference data and therefore runs on
any checkout; the time-domain solver has its own tests on a small
synthetic RFQ (`tests/rfq_td/`).

← [Eigenemittance](09_eigenemittance.md)
