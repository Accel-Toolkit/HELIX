# CLI: `orm`

Orbit-response matrices from the shell: compare a measured matrix with the
HELIX model, fit quadrupole gradient scale factors, trim calibrations and BPM
gains to it (the linac version of LOCO), and write a recalibrated deck.  The
physics, file formats and conventions are described in
[Orbit-response calibration](../08_errors/08_orm_calibration.md).

```
python -m linac_gen orm compare  <input> --measured DIR [--measured DIR2] [options]
python -m linac_gen orm fit      <input> --measured … [--stage trims+quads] [--calibration cal.json] [--export-deck out.dat]
python -m linac_gen orm export   <input> --calibration cal.json --out-deck out.dat [--lgproj]
python -m linac_gen orm validate <input> --measured … [--seed 7 --g-sigma 0.03]
python -m linac_gen orm model    <input> --out DIR [--method maps|tracking] [--unit mm/Tm|mm/mrad]
```

`<input>` is a `.lgproj` project or a lattice file (`--energy`, `--freq`,
`--species` for a bare `.dat`, or `--tracewin-ini` to take the beam from
the deck's TraceWin `.ini` options file).  `--measured` takes a FORMA scan folder (one
driven plane; repeat it for the other plane) or a HELIX ORM CSV.

## Modes

| Mode | What it does | Writes (with `--out DIR`) |
|---|---|---|
| `compare` | maps the measured devices onto the lattice, builds the model response, prints the per-plane summary (global calibration, median correlation, residual, noise floor, polarity flips) | `orm_compare.json`, `orm_model_xx.csv` / `_yy.csv`, `orm_measured_xx.csv` / `_yy.csv` (+ `_error`), `orm_trim_calibration.csv`; with `--plots` the heatmaps, per-trim trends and calibration figures |
| `fit` | the LOCO-style fit; prints the residual before → after, the largest quad corrections and the quads held fixed | the compare outputs, `orm_calibration.json` (or `--calibration PATH`), `orm_fit_summary.txt`; with `--plots` the quad corrections, residuals before/after, residual maps and trends after the fit; `--export-deck` writes the recalibrated deck (`--lgproj` adds a project file) |
| `export` | rewrites a deck from a calibration JSON (text surgery on the original file, verified by re-parsing) | the deck (and project) named by `--out-deck` |
| `validate` | synthetic closed loop: random quad errors (`--g-sigma`), trim calibrations and BPM gains (`--G-sigma`) plus measured-level noise, refit, recovery report | `orm_validation.json` |
| `model` | the model response of every BPM to every steerer, no measurement needed | `orm_model_xx/xy/yx/yy.csv` |

## Options shared by `compare`, `fit` and `validate`

| Option | Argument | Default | Meaning |
|---|---|---|---|
| `--map` | JSON | built-in rules | device map (`L:DnnBPH` → `DnnBPM`, `L:DnnTMH` → `DnnT`, `L:BPH5OT` → the first BPM …); `--write-map` saves the resolved map for editing |
| `--exclude-bpm` / `--exclude-trim` | device or label | — | leave a BPM row / trim column out (repeatable) |
| `--sys-floor` | fraction | `0.05` | systematic floor added in quadrature to the measured errors, as a fraction of each column's maximum |
| `--stage` | `trims` \| `trims+quads` \| `trims+quads+bpms` \| `all` | `trims+quads` | which parameters are fitted (`all` chains the three stages with warm starts) |
| `--prior-g` / `--prior-G` | fraction | `0.10` / `0.05` | Gaussian prior widths on the quad scale factors and the BPM gains |
| `--fix-quad` | label | — | hold a quad at scale 1 (repeatable); quads with fewer than two fitted downstream BPMs are held automatically |
| `--max-iter` | int | `12` | Levenberg–Marquardt iterations |
| `--check-tracking` | flag | off | `compare` only: also build the kick-and-read model and print the largest deviation from the map model |
| `-q` | flag | off | print only the per-plane summary lines |

## Exit codes

`0` success · `1` refused (no measured device matched, a plane without fitted
entries, the fit did not converge, the exported deck failed its re-parse check) ·
`2` bad arguments or a missing input · `130` interrupted.

## Example (synthetic measurement, no machine data needed)

`examples/orm_demo/make_synthetic_forma.py` writes a FORMA-layout folder pair
from the HELIX model of the six-cell demo FODO `examples/orm_demo/fodo_orm.lgproj`
with random quad scale errors, trim calibrations and noise, and records the
injected values in `truth.json`:

```{.bash .skip}
python examples/orm_demo/make_synthetic_forma.py
python -m linac_gen orm compare examples/orm_demo/fodo_orm.lgproj \
    --measured examples/orm_demo/synthetic_forma/20260101_000000_H \
    --measured examples/orm_demo/synthetic_forma/20260101_000000_V --out orm_out --plots
python -m linac_gen orm fit examples/orm_demo/fodo_orm.lgproj --measured … \
    --out orm_out --plots --export-deck orm_out/fodo_orm_fit.dat --lgproj
```

On a machine measurement the same commands take the scan folders; for the
Fermilab SCL, for instance, `--exclude-bpm L:D74BPH --exclude-bpm L:D74BPV`
(a BPM with inverted polarity) and `--fix-quad Q73` (the last quad, seen by
one BPM only) are the typical extra flags.

← [CLI: backtrack](11_cli_backtrack.md) · [Orbit-response calibration](../08_errors/08_orm_calibration.md) →
