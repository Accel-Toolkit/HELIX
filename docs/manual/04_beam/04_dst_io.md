# TraceWin `.dst` loading

The `.dst` is TraceWin's binary distribution file — used to checkpoint
beams at any point in a simulation, share initial distributions across
machines, and import measured beam data from upstream codes (Toutatis,
PIC simulations, etc.).  HELIX reads and writes `.dst` files
faithfully.

## TL;DR

```python
from linac_gen.core.config import BeamConfig
from linac_gen.distributions.factory import create_beam
from linac_gen.io.tracewin_dst import write_dst

# Create a .dst to load (normally it comes from TraceWin or a prior
# HELIX run — see "Writing" below):
_src = create_beam(BeamConfig(n_particles=1000, current=5.0), seed=3)
write_dst("initial.dst", _src.particles, current_mA=5.0,
          frequency_MHz=352.21, mass_MeV=_src.ref.species.mass,
          w_kin_ref=3.0)

beam_cfg = BeamConfig(
    species="proton",
    source="file",
    distribution_file="initial.dst",
)
beam = create_beam(beam_cfg)
```

When `source="file"` and the file is a `.dst`, the file is
authoritative for the beam state: the reference **energy** and
**frequency**, plus the sample-derived **emittances** and **Twiss**
parameters, override the corresponding BeamConfig values, and the
particle count is the file's count.  This follows TraceWin's
convention:

> "If the input dst file is specified, the input beam parameters
> (number of particles, emittances, Twiss parameters, beam centroid,
> beam current and energy) are automatically extracted from the
> specified file and used for the calculation."
> — TraceWin user manual

with one deliberate exception: the header's **current is *not*
applied** — `BeamConfig.current` stays in force, so you control the
space-charge current explicitly.  The file's rest mass is checked
against the configured species (a warning is logged if they differ by
more than 1 MeV) but the species comes from `BeamConfig`.

`n_particles` and `seed` have **no effect** on file loads — every
particle in the file is used, in file order.  There is no
subsampling; if you want fewer particles, thin the file yourself
before loading.

## Tutorial

### Reading

```python
from linac_gen.io.tracewin_dst import load_dst

particles, header = load_dst("initial.dst")
# particles: ndarray (N, 6) in HELIX conventions:
#   [x_mm, xp_mrad, y_mm, yp_mrad, dphi_deg, dW_MeV]
# with dphi / dW as deviations from the bunch centroid.
print(f"Loaded {len(particles)} particles")
print(f"  energy: {header['w_kin_ref']:.3f} MeV")
print(f"  freq:   {header['frequency_MHz']:.1f} MHz")
print(f"  mass:   {header['mass_MeV']:.3f} MeV")
```

`header` also carries `n_particles`, `current_mA`, `phi_ref_rad`, and
the sample-derived emittances / Twiss (`emit_x`, `emit_nx`,
`alpha_x`, `beta_x`, … for all three planes).

### Writing

```python
from linac_gen.io.tracewin_dst import write_dst

write_dst(
    "final.dst",
    beam.alive_particles,            # (N, 6) — filtering losses is your job
    current_mA=beam.current,
    frequency_MHz=beam.ref.frequency,
    mass_MeV=beam.ref.species.mass,
    w_kin_ref=beam.ref.w_kin,
)

# tidy up the demo files
from pathlib import Path
Path("initial.dst").unlink(missing_ok=True)
Path("final.dst").unlink(missing_ok=True)
```

The writer takes a raw `(N, 6)` particle array in HELIX units and
converts back to TraceWin's on-disk conventions.  It does **not**
filter dead particles — pass `beam.alive_particles`, not
`beam.particles`.

## Layout

The `.dst` is a packed little-endian binary file (no alignment
padding):

```
Header (23 bytes):
    2 × char   magic bytes (TraceWin writes 0x7d 0x64)
    uint32     Np             — number of particles
    float64    Ib             — beam current [mA]
    float64    freq           — bunch frequency [MHz]
    1 × char   separator byte (TraceWin writes 0x7d)

Particle block (Np × 48 bytes):
    For each particle, 6 × float64:
        x    [cm]
        x'   [rad]
        y    [cm]
        y'   [rad]
        phi  [rad]   # absolute, relative to the RF clock
        W    [MeV]   # absolute kinetic energy, not a deviation

Trailer (8 bytes):
    float64    mc²  — particle rest-mass energy [MeV]
```

!!! warning "`freq`: HELIX writes the LOCAL RF clock, TraceWin writes the BUNCH RATE"
    The two codes do not agree on this field, and downstream of a
    frequency jump they differ.

    **What TraceWin does** (measured 2026-07-31 across six genuine
    PIP-II files — `part_rfq.dst`, `part_dtl1.dst`, `lbin.dst`,
    `input.dst`, `ssr2out.dst`, `output.dst`): every one carries
    **162.5 MHz**, whether its particles sit at 30 keV, at 166 MeV in a
    325 MHz section, or at 752 MeV in a 650 MHz section.  TraceWin
    writes the bunch repetition rate, always.

    **What HELIX does**: writes `beam.ref.frequency`, the live RF clock.
    That is the clock its Δφ degrees are measured in — `write_dst` does
    a pure deg→rad conversion and `load_dst` the inverse, the frequency
    never entering either — so HELIX↔HELIX round-trips are exact.

    **Consequence**: a HELIX `.dst` written downstream of a frequency
    jump is **not interchangeable with TraceWin**, and re-importing one
    gives a macrocharge `Q = I/f` computed at the local clock rather
    than the bunch rate.  HELIX warns
    (`DstHeaderWarning`) whenever it writes such a file.  Reading a
    *TraceWin* file is unaffected: its 162.5 MHz header sets
    `bunch_frequency` correctly.

    **Why this is not simply "fixed"**: writing the bunch rate while
    leaving the phase column in local degrees would produce a file
    whose label and data disagree — a worse failure, because it looks
    standard-compliant.  Whether TraceWin converts its phases on export
    is unresolved and needs one round-trip through real TraceWin to
    settle; the σ_z test cannot separate the two hypotheses because
    both give physically plausible millimetre-scale answers.  Until
    then HELIX stays self-consistent and says so loudly.

!!! info "Deferred fix — what it will take, and what is safe meanwhile"
    **Two halves, together**: `frequency_MHz = beam.bunch_frequency`
    **and** the phase column rescaled by `f_bunch / f_local`.  Changing
    only the header would fix the macrocharge and simultaneously break
    the clock, because `write_dst` performs no frequency conversion.
    Blocked on an external anchor: a round-trip test cannot validate
    this, since scaling on write and unscaling on read agrees with
    itself either way.

    **Also planned**: `BeamConfig.bunch_frequency_MHz`.  A loaded
    `.dst` is authoritative for the RF clock, but it cannot be
    authoritative for the bunch repetition rate — the format has no
    such field.  Today nothing can state it: the file's frequency
    overrides the project's, and `bunch_frequency` derives from that,
    so a Beam-tab value is silently ignored.  Needed before any
    segmented workflow (export mid-linac, restart downstream).

    **Safe today**: this needs a frequency jump *and* a `.dst`
    round-trip.  A through-and-through run is unaffected —
    `bunch_frequency` stays pinned at the true rate from launch to the
    end of the line, so space charge is correct through every jump, and
    the openPMD and partran exports take the bunch rate from the beam
    configuration rather than the live clock.  If you do export across
    a jump and re-import, set `beam.bunch_frequency` back by hand.

Total file size = 31 + 48·Np bytes.  There is no charge-state field;
the species is not stored in the file.

On load, HELIX converts to its internal `(mm, mrad, deg, MeV)`
deviation convention:

* x, y: cm → mm (×10); x', y': rad → mrad (×1000).
* phi, W: the file stores absolute values and no synchronous
  particle, so the loader subtracts the **bunch centroid**
  (`mean(phi)`, `mean(W)`) and reports the residuals as
  `dphi` [deg] / `dW` [MeV].  The raw centroid is returned in the
  header as `phi_ref_rad` / `w_kin_ref` so you can restore absolute
  values if needed.

## API reference

```{.python .skip}
linac_gen.io.tracewin_dst.load_dst(path: str) -> (particles, header)
linac_gen.io.tracewin_dst.write_dst(
    path: str, particles: np.ndarray,
    current_mA: float, frequency_MHz: float,
    mass_MeV: float, w_kin_ref: float,
    phi_ref_rad: float = 0.0,
) -> None
```

### Source

`linac_gen/io/tracewin_dst.py:1`

## As backtracking input

A `.dst` at a downstream plane (a diagnostic export, a TraceWin dump,
or a prior HELIX `--write-dst`) is the natural input for **backward
tracking** — reconstructing the distribution that entered the line:

```bash
python -m linac_gen backtrack lattice.dat --dst measured_exit.dst \
    --species H- --energy 2.1 --freq 162.5 --write-dst entrance.dst
```

Energy policy: the file's centroid energy is authoritative for the
*beam*, but the machine's maps are always evaluated on the **design**
reference (the `--energy` you pass).  If the file energy deviates from
the design exit energy the walk re-centres `dW` on the design
reference — warning above 0.1 % relative deviation, refusing above
5 % (both thresholds adjustable).  A refusal almost always means the
`.dst` and the lattice/energy pairing don't belong together.

See [CLI: backtrack](../06_running/11_cli_backtrack.md) for the full
workflow including the forward-closure `--validate` check.

## See also

* [Distributions](01_distributions.md) — `from_file` distribution.
* Worked example: LEBT + RFQ —
  uses a `.dst` to chain LEBT output into RFQ input.
* [CLI: backtrack](../06_running/11_cli_backtrack.md) — reconstruct
  the upstream distribution from a downstream `.dst`.

← [BeamConfig reference](03_beam_config.md) ·
[Continue to Space charge → Models →](../05_space_charge/01_models.md)
