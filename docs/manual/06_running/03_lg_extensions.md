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
`key=value` pairs and stored in `metadata["lg_options"]` (the second
return value of `parse_tracewin`) and mirrored onto
`lattice.lg_options`.

!!! note "Parsed and stored — not yet wired into the runtime"
    `;@LG` directives are currently **informational**: HELIX parses
    them, keeps them for round-tripping and for external tooling, but
    nothing in the simulation core or the GUI reads
    `lattice.lg_options` when building a run's configuration.  To
    actually set SC / tracking options, use the GUI (Numerics tab),
    the CLI flags (`--nx`, `--kernel`, `--sc NAME=VALUE`, …), or the
    Python API (`SpaceChargeConfig`, `Simulation` arguments).

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
| `boundary` | str | `open` | `open` or `periodic` |
| `use_gpu` | str | `auto` | `auto` / `cpu` / `gpu` |

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

## Why comment-prefix?

Two reasons:

1. **Cross-tool portability** — a `.dat` with `;@LG` lines opens in
   TraceWin without modification; TraceWin sees them as comments.
2. **Explicit opt-in** — the HELIX-specific knobs are visually
   distinct from the standard TraceWin element cards, so a user
   reading the file understands which lines are HELIX-specific.

## Where they end up

The directives are read at parse time and stored — nothing more:

| Where | Contents |
|---|---|
| `metadata["lg_options"]` | dict of every `;@LG key=value` pair, in file order |
| `lattice.lg_options` | the same dict, attached to the `Lattice` for downstream consumers |

If you write a driver script, you can consume them yourself, e.g.:

```python
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
