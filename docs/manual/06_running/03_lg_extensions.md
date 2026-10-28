# HELIX-specific extensions

HELIX inherits TraceWin's `.dat` format faithfully but needs a few
controls TraceWin doesn't have (e.g. PIC kernel choice, GPU
backend).  These live as **comment-prefixed directives** so the
file remains valid TraceWin input.

## TL;DR — `;@LG` directives

```
; Standard TraceWin syntax — these are comments to TraceWin
;@LG kernel=tsc           # PIC particle-mesh shape
;@LG green_kind=igf       # PIC Green's function
;@LG dc_kernel=gaussian   # DC mode kernel
;@LG nx=96 ny=96 nz=96    # PIC grid
;@LG grid_extent=5.0      # PIC ±5σ box
;@LG use_gpu=auto         # CPU/GPU selection
```

The parser recognises the `;@LG` prefix.  The remainder is parsed as
`key=value` pairs (several may share a line; text after `#` or `;` is a
comment) and stored in `metadata["lg_options"]` (the second return value
of `parse_tracewin`) and mirrored onto `lattice.lg_options`.  Numbers
become numbers, `true` and `false` become booleans, anything else stays
text.

A value runs to the next `key=` on its line, so it may contain spaces:
`;@LG note=booster ring study` gives `note` the value
`"booster ring study"`.  A key on its own is a flag (`true`).  A flag
therefore goes before the pairs of its line, or on a line of its own:
`;@LG ring harmonic=84` declares a ring with harmonic 84, while
`;@LG harmonic=84 ring` gives `harmonic` the text `"84 ring"`.

!!! note "Parsed and stored — mostly informational"
    The space-charge and tracking directives below are **informational**:
    HELIX parses them and keeps them — the deck writer (`write_tracewin`,
    the GUI's Save Lattice) writes them back — and external tooling can
    read them, but nothing in the simulation core or the GUI reads them
    when building a run's configuration.  To actually set SC / tracking
    options, use the GUI (Numerics tab), the CLI flags (`--nx`,
    `--kernel`, `--sc NAME=VALUE`, …), or the Python API
    (`SpaceChargeConfig`, `Simulation` arguments).  Two keys are read:
    `ring` declares the deck a ring (the Lattice tab then offers ring mode,
    as it does for a lattice whose bends close on themselves), and
    `harmonic` gives ring mode the RF harmonic number h — unless the ring
    settings name one, every cavity runs at h times the revolution
    frequency.

## Recognised keys

The keys below follow the names (and defaults) of the runtime config
fields they mirror — set the real thing via GUI / CLI / Python as
noted above.

### PIC parameters

| Key | Type | Default | Notes |
|---|---|---|---|
| `kernel` | str | `cic` | particle-mesh shape: `cic` or `tsc` |
| `green_kind` | str | `igf` | Green's function: `igf` or `point` |
| `dc_kernel` | str | `uniform` | DC kernel: `uniform`, `gaussian`, `pic2d` |
| `nx`, `ny`, `nz` | int | 96 | grid size |
| `grid_extent` | float | 5.0 | σ multiplier |
| `boundary` | str | `open` | `open` only — the Poisson solve is open-boundary Hockney; periodic is refused at validation |
| `use_gpu` | str | `auto` | `auto` / `cpu` / `gpu` / `cuda` / `mps` — `auto` never selects the FP32 `mps` backend; request it explicitly.  The torch engine runs on an NVIDIA GPU only for `cuda` / `gpu` |
| `gpu_particles` | bool | `false` | with the CUDA backend, the charge deposit and field gather on the GPU too (needs torch with CUDA; not with `reuse_every` > 1); `LINAC_GEN_GPU_PARTICLES=1` / `=0` overrides |

### Tracker parameters

| Key | Type | Default | Notes |
|---|---|---|---|
| `integrator_kind` | str | `kd` | RK4 family: `kd` or `dkd` |
| `interp_kind` | str | `linear` | field-map interpolation |
| `record_substeps` | bool | False | full diagnostic per substep |

### Envelope parameters

| Key | Type | Default | Notes |
|---|---|---|---|
| `env_solver` | str | `matrix` | envelope flavour: `matrix` or `sacherer` |

### Ring mode (read by HELIX)

| Key | Type | Default | Notes |
|---|---|---|---|
| `ring` | flag | — | declares the deck a ring: the Lattice tab offers ring mode ([Ring mode](../17_ring/01_overview.md)) |
| `harmonic` | int | — | the RF harmonic number h: in ring mode every cavity runs at h times the revolution frequency, unless the ring settings name a harmonic |

## Why comment-prefix?

Two reasons:

1. **Cross-tool portability** — a `.dat` with `;@LG` lines opens in
   TraceWin without modification; TraceWin sees them as comments.
2. **Explicit opt-in** — the HELIX-specific knobs are visually
   distinct from the standard TraceWin element cards, so a user
   reading the file understands which lines are HELIX-specific.

## Where they end up

The directives are read at parse time and stored:

| Where | Contents |
|---|---|
| `metadata["lg_options"]` | dict of every `;@LG key=value` pair (a flag as `True`), in file order |
| `lattice.lg_options` | the same dict, attached to the `Lattice` for downstream consumers |
| a deck saved by `write_tracewin` | the same options as `;@LG` lines at the top, one per line, so they read back as they were; an option that could not (a key with a blank, a value holding `#` or `;` …) is left out with a warning |

If you write a driver script, you can consume them yourself, e.g.:

```{.python data-needs="examples/pipii/mebt/mebt.dat"}
from linac_gen.core.config import SpaceChargeConfig
from linac_gen.io.tracewin_parser import parse_tracewin

lattice, metadata = parse_tracewin("examples/pipii/mebt/mebt.dat")
opts = metadata["lg_options"]          # or lattice.lg_options
sc = SpaceChargeConfig(
    kernel=opts.get("kernel", "cic"),
    nx=int(opts.get("nx", 96)),
)
```

The corresponding runtime settings the tables above mirror live in
`SpaceChargeConfig` (PIC keys), the `Tracker` / `Simulation`
arguments (tracker keys) and the envelope-solver selection
(`env_solver`).

## Cross-references

* [.dat file reference](02_tracewin_dat.md) — full TraceWin keyword list.
* [SpaceChargeConfig](../05_space_charge/02_pic_solver.md) — the
  fields these knobs map to.

← [.dat reference](02_tracewin_dat.md) ·
[Continue to Reading results →](04_results.md)
