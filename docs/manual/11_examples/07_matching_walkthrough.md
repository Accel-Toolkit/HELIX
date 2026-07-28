# Worked example: Matching walkthrough

End-to-end matching: a 4-quad section needs a specific Twiss output.
Demonstrates the full SET/ADJUST workflow.

## Source

This is a standalone 4-quad exercise.  The shipped
`examples/matching_demo.dat` / `examples/matching_demo.py` pair is a
*simpler* demo of the same workflow — two quads with two generic
`ADJUST` cards and one `SET_SIZE` target:

```bash
python examples/matching_demo.py
```

## Lattice

HELIX uses the generic `ADJUST target param_idx [link_group vmin vmax
start_step kn]` card (param_idx 2 = quad gradient) and the 12-argument
`SET_TWISS` card whose k-flags select which Twiss parameters enter
the residual (see [SET / ADJUST](../07_matching/02_set_adjust.md)):

```
DRIFT 100 20
ADJUST QUAD 2 0   5.0  15.0 0.5 0   ; G1 — tunes the NEXT quad
QUAD 100 10.0 20
DRIFT 200 20
ADJUST QUAD 2 0 -15.0  -5.0 0.5 0   ; G2
QUAD 100 -10.0 20
DRIFT 200 20
ADJUST QUAD 2 0   5.0  15.0 0.5 0   ; G3
QUAD 100 10.0 20
DRIFT 200 20
ADJUST QUAD 2 0 -15.0  -5.0 0.5 0   ; G4
QUAD 100 -10.0 20
DRIFT 100 20

SET_TWISS "" -1.0 2.0 1.0 2.0 0 0 1 1 1 1 0 0
```

Reads as: "4 variables (the four quad gradients — each `ADJUST` card
sits *before* its quad and is unlinked, `link_group=0`), 4 constraints
(α, β at exit in both planes, selected by the four k-flags set to 1).
Find values such that the exit Twiss matches α_x = −1, β_x = 2,
α_y = +1, β_y = 2."

## Run

### Via CLI

```bash
python -m linac_gen.matching my_4quad_match.dat \
    --out matched.dat \
    --report
```

(`--report` is a boolean flag that prints the summary report; there
is no `--verbose` option.)

### Via Python

```python
import copy
from pathlib import Path
from linac_gen.io.tracewin_parser import parse_tracewin
from linac_gen.matching.engine import match
from linac_gen.core.config import BeamConfig

# Write the lattice shown above so the walkthrough is self-contained.
Path("my_4quad_match.dat").write_text("""\
DRIFT 100 20
ADJUST QUAD 2 0   5.0  15.0 0.5 0
QUAD 100 10.0 20
DRIFT 200 20
ADJUST QUAD 2 0 -15.0  -5.0 0.5 0
QUAD 100 -10.0 20
DRIFT 200 20
ADJUST QUAD 2 0   5.0  15.0 0.5 0
QUAD 100 10.0 20
DRIFT 200 20
ADJUST QUAD 2 0 -15.0  -5.0 0.5 0
QUAD 100 -10.0 20
DRIFT 100 20
SET_TWISS "" -1.0 2.0 1.0 2.0 0 0 1 1 1 1 0 0
END
""")
lat, _ = parse_tracewin("my_4quad_match.dat")
Path("my_4quad_match.dat").unlink()
lat_before = copy.deepcopy(lat)       # keep the unmatched optics for the plot
beam_cfg = BeamConfig(species="proton", energy=3.0, frequency=352.21,
                      emit_nx=0.25, emit_ny=0.25, emit_z=0.30)

result = match(lat, beam_cfg)   # variables/constraints from the .dat
print(f"Converged: {result.success} ({result.n_iter} iter)")
print(f"Residuals: {result.residuals}")
```

### Via GUI

Open the `.dat` → Matching tab → variables and constraints already
populated → click **Match**.

## Plot before/after

```python
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from linac_gen.core.particle import PROTON
from linac_gen.core.reference import ReferenceParticle
from linac_gen.tracking.envelope import EnvelopeSolver

# Geometric initial Twiss from the BeamConfig
ref = ReferenceParticle(species=PROTON, w_kin=3.0, frequency=352.21)
bg = ref.bg
initial_twiss = dict(
    emit_x=beam_cfg.emit_nx / bg, alpha_x=beam_cfg.alpha_x,
    beta_x=beam_cfg.beta_x,
    emit_y=beam_cfg.emit_ny / bg, alpha_y=beam_cfg.alpha_y,
    beta_y=beam_cfg.beta_y,
    emit_z=beam_cfg.emit_z, alpha_z=beam_cfg.alpha_z,
    beta_z=beam_cfg.beta_z,
)

# Before matching (the deep copy taken above) vs after — match()
# already applied the matched gradients to `lat` in place.
res_before = EnvelopeSolver(lat_before, ref, initial_twiss).run()
res_after = EnvelopeSolver(lat, ref, initial_twiss).run()

# Plot σ_x for both
plt.plot(res_before.s, res_before.sigma_x, label="before")
plt.plot(res_after.s, res_after.sigma_x, label="after")
plt.legend()
plt.close("all")
```

## Convergence

A typical run:

```
[matching] LM iter 1: cost = 1.234e-01
[matching] LM iter 2: cost = 2.157e-03
[matching] LM iter 3: cost = 4.591e-06
[matching] LM iter 4: cost = 1.298e-08
[matching] Converged in 4 iterations.
```

For well-posed problems, expect 3-10 LM iterations.  Many more
suggests bad initial guess or ill-conditioned Jacobian.

## Cross-references

* [Matching → Overview](../07_matching/01_overview.md)
* [Matching → Recipes](../07_matching/05_recipes.md)
* [GUI → Matching tab](../10_gui/05_matching_tab.md)

[Continue to Tolerance study →](08_tolerance_study.md)
