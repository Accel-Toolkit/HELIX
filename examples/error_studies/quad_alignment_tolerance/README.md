# Quad alignment tolerance

**Study type:** random transverse offsets (`dx`, `dy`) on every
quadrupole.

## Errors applied

The `.dat` file contains a single directive at the top:

```
ERROR_QUAD_NCPL_STAT 8 2 0.2 0.2 0 0 0 0 0 0 0 0
```

Translates to: every quadrupole (8 of them) gets a fresh draw per
seed of

* `dx ~ Normal(0, 0.2 mm)` truncated at 3σ
* `dy ~ Normal(0, 0.2 mm)` truncated at 3σ

No tilt, no field error, no higher-order multipole — pure alignment.

## What you'll see

* **Centroid drift** vs s — every seed gets a different orbit; the
  ensemble mean stays near zero, the spread grows through each
  defocusing-quadrupole stretch.
* **σ_x ensemble envelope** — the mean-curve is similar to a clean
  run, but the ±1σ band widens at every quad.
* **Transmission histogram** — a long tail toward the low end if the
  4 σ orbit excursion exceeds the 20 mm aperture in worst-case seeds.

## How to run (GUI)

1. `File → Open Project…` → `quad_alignment.lgproj`
2. Beam, lattice, convergence load automatically.  The TraceWin
   `ERROR_*` directive is parsed and stored on the lattice.
3. Open the **Errors** tab — no manual entry needed; click
   **Run study** with `n_seeds = 100`.
4. **Results** tab → ENERGY · KINEMATICS section → **Error study
   ensemble** tile.

## How to run (Python)

```python
from linac_gen.io.tracewin_parser import parse_tracewin
from linac_gen.core.config import BeamConfig
from linac_gen.errors.error_model import ErrorStudy

cfg = BeamConfig(species="proton", energy=10.0, frequency=325.0,
                  n_particles=5000,
                  emit_nx=0.25, beta_x=2.0,
                  emit_ny=0.25, beta_y=2.0,
                  emit_z=0.30, beta_z=3.0)
lat, _ = parse_tracewin(
    "examples/error_studies/quad_alignment_tolerance/quad_alignment.dat"
)
results = ErrorStudy(lat, cfg, n_seeds=100).run()
print(results.transmission_stats())
```

## Tweaks worth trying

* **Tighten/loosen σ**: change `0.2` in the `ERROR_QUAD_NCPL_STAT`
  line to `0.05`, `0.5`, `1.0` and re-run — see at what σ_dx
  transmission falls below 99 %.
* **Add tilt**: replace the seventh slot (φz) with `0.1` to add
  σ = 0.1° rolls on every quad.
* **Per-family correlation**: split the directive in two and target
  only the F-family or D-family quads (requires renaming quads via
  TraceWin labels — see `tests/io/test_tracewin_error_parsing.py`).
