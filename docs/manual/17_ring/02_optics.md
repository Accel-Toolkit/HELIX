# Ring optics

The periodic optics of a ring — what MAD-X's `TWISS` gives for a closed
machine — computed by tracking particles through the ring kernels, the same
element maps a ring run uses.  Every number is therefore that of the tracking
itself: the chromaticity includes each element's full map, its pole faces,
its sextupole component.

## What is computed

| Quantity | Definition |
|---|---|
| Qx, Qy | Betatron tunes, integer part included: the phase advance per turn of the two transverse normal modes, divided by 2π |
| Q′x, Q′y | Chromaticities dQ/dδ, δ = (P − P0)/P0 the relative momentum deviation (a MAD-X value per PT times β0 is the same quantity) |
| α_c | Momentum compaction: the relative change of the closed orbit's length per unit δ |
| η | Slip factor: the relative change of the revolution time per unit δ, η = α_c − 1/γ0² |
| γt | Transition gamma, 1/√α_c (none when α_c ≤ 0) |
| β, α, μ | Twiss functions and phase advance (in tune units, from the start) at every element's exit |
| D, D′ | Dispersion per unit δ, at every element's exit |
| x, px, y, py | The closed orbit at every element's exit |
| Qs | With RF: the synchrotron tune, from the 6-D one-turn map (its magnitude: the longitudinal motion turns the other way above transition) |
| bucket | With RF: harmonic, voltage, stable phase, small-amplitude synchrotron tune, half-height in δ, length |

Units are SI: metres, radians, momenta divided by the reference momentum
(the ring coordinates of the chapter on element models).

## How

1. **Closed orbit** — Newton's method on the one-turn map.  4-D (the
   default): the momentum deviation is held fixed and the cavities are
   switched off, as in MAD-X's `TWISS`; 6-D (*with RF*): all six coordinates
   are periodic.
2. **One-turn matrix** — central differences of the tracked map about the
   closed orbit.
3. **Normal form** — the eigenvectors of the one-turn matrix give the
   fractional tunes and the normalizing matrix W (Courant–Snyder form for
   uncoupled motion; for coupled motion, the amplitude functions of the two
   normal modes).
4. **Along the ring** — particles started along W (± a small amplitude) and
   at the closed orbits of δ = ±10⁻⁶ are tracked once around, recording each
   element's exit: their differences give W, hence β, α and the phase, and
   the dispersion.  The phase, followed from element exit to element exit,
   gives the integer part of the tunes: a mode that turns forward in its own
   plane (always, without coupling) never steps back, so each element's
   advance is taken between 0 and 2π — a long bend or a matrix element may
   advance it by more than π (more than 2π within one element is not seen).
   Where strong coupling turns a mode backward in a plane, the step there is
   taken between −π and π.
5. **Chromaticity and momentum compaction** — the tunes and the path length
   of the closed orbits at δ = ±10⁻⁵.

A lattice that is not stable at the reference energy (an eigenvalue of the
one-turn map off the unit circle) is reported as such.

## In the GUI

**Ring tab ▶ Ring & Optics ▶ Compute optics**.  The first computation in a
session also compiles the ring kernels (a few seconds).  The results appear
as text, as plots of β and dispersion along the ring, and as the line of
tunes, chromaticities and transition gamma at the top of the tab.  **Use the
matched beam in the Beam tab** gives the Beam tab the beam matched to the ring
at the start of the run ([Ring runs](04_run.md)): the periodic α, β and
dispersion at the start of the ring (dispersion converted to HELIX's mm/MeV and
mrad/MeV), the longitudinal Twiss functions with the RF on, the closed orbit
as the centroid and, for a deck that declares its beam, its species and the
run's starting energy.  It computes them itself, from the optics of the run's
first turn (the ramp's turn 0, the injection magnets of turn 0), not from the
optics shown here.  *with RF* adds the 6-D closed orbit, the synchrotron
tune and the bucket; it needs a stationary bucket (the cavities giving the
reference particle no net energy — the RF of a ramp is set by its programs,
see [The ramp](05_programs.md)).
Below transition the bucket is at a sine lag of 0° (HELIX RF phase −90°),
above it at 180° (phase +90°); the other phase is reported as unstable.

## From the command line

```
python -m linac_gen ring optics booster.madx
python -m linac_gen ring optics booster.madx --six-d --twiss-out twiss.csv
python -m linac_gen ring optics ring.dat --energy 800 --species proton
python -m linac_gen ring optics booster.madx --model curvilinear --json
```

A deck that declares its beam — a MAD-X or MAD8 `BEAM` that gives the
energy — sets the ring's species and energy.  A deck that states none (a
TraceWin or elegant deck, a MAD-X `BEAM` that names only the particle) runs
at the project's beam, the Beam tab's species and energy, or for a bare deck
at `--energy` and `--species`; its magnets keep the strengths the deck gives
them (the normalized strengths of an elegant or MAD-X deck come back
exactly).
`--twiss-out` writes the table per element (`.csv` or `.json`), `--json`
prints the summary as JSON.  Other options: `--model`, `--edge-model`,
`--constants`, `--harmonic`, `--bend-steps`.  For a project, the element
model, pole faces and constants saved by the Ring tab are used unless the
command line gives them.

## Accuracy

On the simple Fermilab Booster (24 cells, 474.2 m; the public deck in
`tests/ring/fixtures/`) the optics agree with MAD-X 5.09's `TWISS` to 2×10⁻⁹
in the tunes, 6.3×10⁻⁹ relative in β at the 72 quadrupoles and cell ends
compared, 10⁻⁹ relative in D, 1.6×10⁻⁹ in α_c, and within 0.003 in the
chromaticity (3×10⁻⁵ from MAD-X's `PTC_TWISS`); a
thin-lens FODO ring agrees with the analytic optics to 10⁻¹².  The details
are on the [ring benchmarks](../12_validation/06_ring_benchmarks.md) page.
