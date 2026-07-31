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

2000 particles, matched input (α = 0, β = 0.01 mm/mrad,
ε_n = 0.20 π·mm·mrad):

| | transmission | exit energy |
|---|---|---|
| no space charge | 71.8 % | 1.900 ± 28 keV |
| 15 mA, 32³ adaptive PIC | 64.3 % | 1.898 ± 62 keV |

Transmission and capture are the **same number** — every surviving
particle is accelerated, so nothing exits as un-bunched low-energy
junk.

!!! note "A teaching deck, not an optimised design"
    A production RFQ iterates the gentle-buncher ramp for >95 %
    capture. Here the remaining loss is longitudinal, concentrated in
    the buncher where the bucket shrinks faster than the beam can
    follow adiabatically. Raising `N_BUNCHER` and the phase-ramp
    exponent in `make_rfq_demo.py` improves it — which is why the
    generator ships alongside the deck rather than the deck alone.

## What it exercises

* **`field_model="tw2term"`**, the default — per-particle longitudinal
  phase slip and vane-tip aperture losses, without which a DC beam
  cannot bunch at all. See [RFQ cell](../03_elements/09_rfqcell.md).
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

Pinned by `tests/rfq/test_rfq_demo_example.py`, which — unlike the rest
of `tests/rfq/` — needs no external reference data and therefore runs
on any checkout.

← [Eigenemittance](09_eigenemittance.md)
