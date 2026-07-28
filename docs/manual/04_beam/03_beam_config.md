# BeamConfig field reference

`BeamConfig` is the dataclass passed to `create_beam()`.  It carries
every parameter needed to specify a beam.  This page lists every
field, default, units, and meaning.

## Reference

```{.python .skip}
@dataclass
class BeamConfig:
    # Species & energy
    species: str = "proton"             # "proton" | "deuteron" | "H-"
    energy: float = 3.0                 # MeV (kinetic)
    frequency: float = 352.21           # MHz

    # Current & duty cycle
    current: float = 0.0                # mA (peak)
    duty_cycle: float = 100.0           # % (<100 → pulsed operation)

    # Particles & distribution
    n_particles: int = 10000
    distribution: str = "waterbag"
    cutoff: float = 3.0                 # σ truncation (gaussian + thermal)

    # Transverse Twiss (normalised emittance)
    emit_nx: float = 0.25               # mm·mrad (NORMALISED, βγ·ε)
    alpha_x: float = 0.0
    beta_x: float = 0.1                 # mm/mrad = m

    emit_ny: float = 0.25
    alpha_y: float = 0.0
    beta_y: float = 0.1

    # Longitudinal Twiss
    emit_z: float = 0.3                 # deg·MeV (NATIVE)
    alpha_z: float = 0.0
    beta_z: float = 1.0                 # deg/MeV

    # Centroid offsets (deterministic; per-seed jitter via BeamErrorDef)
    centroid_x: float = 0.0             # mm
    centroid_xp: float = 0.0            # mrad
    centroid_y: float = 0.0
    centroid_yp: float = 0.0
    centroid_dphi: float = 0.0          # deg
    centroid_dw: float = 0.0            # MeV

    # Input dispersion (generate path only; x += disp_x·ΔW before the
    # centroid offsets).  The matched beam of a bending line (arc FODO /
    # BTL) carries these — the matching dialog's Apply fills them.
    disp_x: float = 0.0                 # mm/MeV
    disp_xp: float = 0.0                # mrad/MeV
    disp_y: float = 0.0                 # mm/MeV
    disp_yp: float = 0.0                # mrad/MeV

    # Mismatch (% emittance scaling, generate path only)
    mismatch_x: float = 0.0
    mismatch_y: float = 0.0
    mismatch_z: float = 0.0

    # File source
    source: str = "generate"            # "generate" | "file"
    distribution_file: Optional[str] = None

    # DC (continuous, pre-RFQ) mode
    continuous: bool = False
    dc_energy_spread_keV: float = 0.0

    # Thermal halo (only used when distribution="thermal")
    halo_fraction: float = 0.05
    halo_ratio: float = 5.0
```

## Field-by-field

### Species & energy

| Field | Default | Notes |
|---|---|---|
| `species` | `"proton"` | string lookup; see `SPECIES_MAP` in `linac_gen/distributions/factory.py:31` for valid keys (`proton`, `deuteron`, `H-`) |
| `energy` | `3.0` | kinetic energy in MeV |
| `frequency` | `352.21` | RF frequency at lattice entrance, in MHz; updates per-FREQ-card |

### Current

| Field | Default | Notes |
|---|---|---|
| `current` | `0.0` | peak beam current in mA — **0.0 disables space charge** |
| `duty_cycle` | `100.0` | percentage; <100 ⇒ pulsed (effective current is reduced) |

### Particles

| Field | Default | Notes |
|---|---|---|
| `n_particles` | `10000` | number of macroparticles to generate (ignored when `source="file"` — the file's full count is used) |
| `distribution` | `"waterbag"` | one of: `gaussian`, `waterbag`, `kv`, `parabolic`, `uniform`, `thermal` — file loading is `source="file"`, not a distribution |
| `cutoff` | `3.0` | σ truncation, applied by `gaussian` **and** `thermal` |

### Transverse Twiss

| Field | Default | Units | Notes |
|---|---|---|---|
| `emit_nx` | `0.25` | mm·mrad | **normalised** ε_n = βγ·ε |
| `alpha_x` | `0.0` | — | Courant-Snyder α |
| `beta_x` | `0.1` | mm/mrad ≡ m | Courant-Snyder β |
| `emit_ny` | `0.25` | mm·mrad | normalised |
| `alpha_y` | `0.0` | — | |
| `beta_y` | `0.1` | mm/mrad | |

### Longitudinal Twiss

| Field | Default | Units | Notes |
|---|---|---|---|
| `emit_z` | `0.3` | deg·MeV | native (TraceWin convention) |
| `alpha_z` | `0.0` | — | |
| `beta_z` | `1.0` | deg/MeV | |

### Centroid offsets (deterministic)

| Field | Units |
|---|---|
| `centroid_x`, `centroid_y` | mm |
| `centroid_xp`, `centroid_yp` | mrad |
| `centroid_dphi` | deg |
| `centroid_dw` | MeV |

These offsets are applied to the generated beam centroid.  For
**random** centroid jitter (one draw per seed in an error study),
use `BeamErrorDef` instead — see
[Errors → Beam errors](../08_errors/04_beam_errors.md).

### Mismatch

| Field | Units | Effect |
|---|---|---|
| `mismatch_x`, `mismatch_y`, `mismatch_z` | % | scales ε in that plane: `ε_eff = ε · (1 + mismatch/100)` |

Used to inject deliberate Twiss mismatch for stability studies.

### File source

| Field | Notes |
|---|---|
| `source` | `"generate"` (default) or `"file"` |
| `distribution_file` | path to `.dst` (TraceWin binary) or ASCII text file — required when `source="file"` |

When `source="file"`, the file's metadata (energy, frequency, ε,
Twiss, current) **overrides** the corresponding BeamConfig fields.
This matches TraceWin's behaviour with input `.dst` files.

### DC mode

| Field | Default | Notes |
|---|---|---|
| `continuous` | `False` | True = continuous (DC) beam, no longitudinal bunching |
| `dc_energy_spread_keV` | `0.0` | for continuous beams: σ_W in keV |

For pre-RFQ LEBT runs always set `continuous=True` — see
[DC mode](../05_space_charge/04_dc_mode.md).

### Thermal halo

| Field | Default | Used for |
|---|---|---|
| `halo_fraction` | `0.05` | `distribution="thermal"` only |
| `halo_ratio` | `5.0` | halo σ multiplier |

See [Distributions → thermal](01_distributions.md#thermal-bi-gaussian-halo).

## Source

`linac_gen/core/config.py:7` (BeamConfig dataclass).

## See also

* [Distributions](01_distributions.md)
* [Twiss & emittance](02_twiss.md)
* [.dst loading](04_dst_io.md)

← [Twiss & emittance](02_twiss.md) ·
[Continue to .dst loading →](04_dst_io.md)
