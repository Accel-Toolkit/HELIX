# Synthetic RFQ — demonstration deck

A complete, runnable RFQ that turns a **DC beam into a bunched,
accelerated one**: 75 keV → ~1.9 MeV protons at 352.21 MHz over 2.01 m.

| file | what it is |
|---|---|
| `rfq_demo.dat` | the lattice — **generated, do not hand-edit** |
| `rfq_demo.lgproj` | project file; open this in the GUI |
| `make_rfq_demo.py` | the generator — the design lives here |

```bash
cd examples/rfq_demo && python3 make_rfq_demo.py     # regenerate
```

## This is not any real machine

Every per-cell number is generated from textbook two-term RFQ design
relations. Nothing is copied from, or fitted to, any TraceWin output or
any existing machine's design.

The design is a conventional four-section RFQ — radial matcher, shaper,
gentle buncher, accelerator, exit matcher:

* modulation `m` ramps 1 → 1.95, synchronous phase −90° → −28°;
* cell length `L = βλ/2`, with β advanced from the energy the model
  itself produces, so the deck is synchronous with the code that runs it;
* `A₁₀` solved from `(r₀, m, L)` through the Crandall/Wangler two-term
  relation (`rfq_coefficients.modulation_consistency`), iterated to its
  fixed point, so each card's triplet is internally consistent;
* focusing parameter `B = qVλ²/(mc²r₀²)` = **5.7**, inside the usual
  4–8 range.

199 cells, r₀ = 3.40 mm, 85 kV vane voltage.

## What it does

Measured with 2000 particles, matched input (α = 0, β = 0.01 mm/mrad,
ε_n = 0.20 π·mm·mrad):

| | transmission | exit energy |
|---|---|---|
| no space charge | 71.8 % | 1.900 ± 28 keV |
| 15 mA, 32³ adaptive PIC | 64.3 % | 1.898 ± 62 keV |

Every surviving particle is accelerated — transmission and capture are
the same number, so nothing exits as un-bunched low-energy junk.

**It is a teaching deck, not an optimised design.** A production RFQ
iterates the gentle-buncher ramp for >95 % capture; here the remaining
loss is longitudinal, concentrated in the buncher where the bucket
shrinks faster than the beam can follow adiabatically. Lengthening
`N_BUNCHER` and raising the phase-ramp exponent in `make_rfq_demo.py`
improves it — a good exercise, and the reason the generator is shipped
rather than just the deck.

## What it demonstrates

* **`field_model="tw2term"`**, the default RFQ model — per-particle
  longitudinal phase slip and vane-tip aperture losses, without which a
  DC beam cannot bunch at all.
* **DC → bunched transition**, handled automatically at the first cell.
* **`periodic_phase`**, enabled in the project. An RFQ makes one bunch
  per RF period but the simulation seeds one period, so without it the
  particles that space charge pushes across a bucket boundary are
  stored a full spacing away and inflate every reported σ_φ and ε_z.
  Compare by setting it false in `rfq_demo.lgproj`.
* **Space charge through an RFQ** — the 15 mA column above.

## Cross-references

* [RFQ cell](../../docs/manual/03_elements/09_rfqcell.md) — the physics
  model, its validation status and known gaps
* [BeamConfig](../../docs/manual/04_beam/03_beam_config.md) —
  `continuous`, `periodic_phase`
