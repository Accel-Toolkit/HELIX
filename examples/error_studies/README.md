# Error & misalignment study examples

Five self-contained example projects, each demonstrating one category
of TraceWin-style tolerance study.  Open any `.lgproj` in HELIX, click
**Run study** in the new Errors tab, and the ensemble plots populate
in the Results tab.

| Folder | Study type | TraceWin directive used | Highlights |
|---|---|---|---|
| [`quad_alignment_tolerance/`](quad_alignment_tolerance/) | Random quad dx, dy | `ERROR_QUAD_NCPL_STAT` | classic alignment-tolerance fan |
| [`quad_field_errors/`](quad_field_errors/) | Quad gradient + g3/g4 jitter | `ERROR_QUAD_NCPL_STAT` (dG, dG3, dG4) | proves the new wired-up multipole content |
| [`cavity_rf_jitter/`](cavity_rf_jitter/) | Cavity voltage + phase errors | `ERROR_CAV_NCPL_STAT` | LLRF stability budget |
| [`beam_input_jitter/`](beam_input_jitter/) | Input-beam centroid / ε / I jitter | `ERROR_BEAM_STAT` | source / chopper stability |
| [`combined_realistic/`](combined_realistic/) | All of the above stacked | every `ERROR_*` directive | template for a real PIP-II tolerance run |

## How to use any of them

1. **Open the project** in HELIX: `File → Open Project…` → pick the
   `.lgproj`.  Beam / lattice / convergence / errors all populate.
2. **Open the Errors tab** (between Convergence and Results).
3. **Click Run study** — `n_seeds = 50` is plenty for these small
   lattices.  Errors registered in the `.dat` are auto-absorbed by
   the engine; you don't need to add anything in the form.
4. **Open the Results tab** → ENERGY · KINEMATICS section → click the
   **Error study ensemble** tile.  The popup shows σ_x and σ_y mean ±
   1σ envelopes and the final-transmission histogram.

## How to extend

* The five examples are deliberately tiny (~10 elements, ~5000
  particles).  They each finish 100 seeds in a few seconds on a
  modern CPU.  Scale up to PIP-II-size by replacing the lattice
  path in the `.lgproj` and bumping `n_seeds`.
* Every TraceWin `ERROR_*` directive supported by HELIX is listed in
  `linac_gen/io/tracewin_error_parsing.py` — you can write your own
  `.dat` cards directly.
* For Python-driven studies (no `.dat`, programmatic spec), see
  `tests/errors/test_error_study.py` and
  `tests/errors/test_error_beam_path.py` for working idioms.

## Related

* The PIP-II reference lattices in [`../pipii/`](../pipii/) are
  starting points for real production-grade tolerance studies.
* The [`../impactx_features/`](../impactx_features/) example
  demonstrates the eigenemittance + thermal-halo + multipole feature
  trio (no errors — different feature group).
