# Matching recipes

Worked examples that walk through each matcher feature in **all three
modes**: Python API, CLI (`python -m linac_gen.matching`), and GUI.
Every recipe uses HELIX's lattice-card-driven workflow — variables come
from `ADJUST_*` cards in the `.dat`; constraints come from `SET_*` cards.
`match()` does NOT accept `variables=` / `constraints=` arguments.

For card syntax, see [SET / ADJUST cards](02_set_adjust.md).

---

## Recipe 1 — Single-quad matching (the smallest possible)

**Problem**: Tune one quadrupole so the beam exits at a target σ_x.

### The lattice (`one_quad.dat`)

```
DRIFT 200 10 0 0 0
ADJUST QUAD_001 2 1 0.5 30 0.5 0       ; vary QUAD_001.gradient ∈ [0.5, 30]
QUAD 100 5.0 10 0 0 0 0 0 0
DRIFT 200 10 0 0 0
SET_SIZE 1 3.0 0 0                     ; target σ_x = 3.0 mm at exit
END
```

### Python API

```python
from pathlib import Path
from linac_gen.io.tracewin_parser import parse_tracewin
from linac_gen.core.config import BeamConfig
from linac_gen.matching import match

# Write the lattice shown above so this recipe is fully self-contained.
Path("one_quad.dat").write_text("""\
DRIFT 200 10 0 0 0
ADJUST QUAD_001 2 1 0.5 30 0.5 0
QUAD 100 5.0 10 0 0 0 0 0 0
DRIFT 200 10 0 0 0
SET_SIZE 1 3.0 0 0
END
""")
lat, _ = parse_tracewin("one_quad.dat")
Path("one_quad.dat").unlink()          # lattice is in memory now
cfg = BeamConfig(species="proton", energy=3.0, frequency=352.21,
                 n_particles=1000, distribution="waterbag",
                 emit_nx=0.25, alpha_x=0.0, beta_x=2.0,
                 emit_ny=0.25, alpha_y=0.0, beta_y=2.0,
                 emit_z=0.30, alpha_z=0.0, beta_z=10.0)

result = match(lat, cfg, algorithm="least_squares", max_iter=80)
print(f"baseline cost: {result.baseline_cost:.3e}")
print(f"final cost   : {result.cost:.3e}")
print(f"matched grad : {result.x_final[0]:.4f} T/m")
print(result.report())
```

`match()` mutated `lat` in place — the QUAD_001 element now carries
the matched gradient.

### CLI

```bash
python -m linac_gen.matching one_quad.dat \
    --algorithm least_squares --max-iter 80 --report
```

Writes `one_quad.matched.dat` next to the input.

### GUI

1. **File → Open Lattice** → pick `one_quad.dat`.
2. **Beam tab** — set energy, Twiss, emittances → **Apply**.
3. **Matching tab** → **Algorithm: least_squares**, **Max iter: 80**.
4. Click **Match**.  The Live Convergence dialog plots cost + εnx/εny/εnz.
5. Click **Apply** to commit the matched gradient, or **Save matched .dat**
   to export.

---

## Recipe 2 — Emittance minimisation with CMA-ES + LS polish

**Problem**: A 4-cryomodule HWR section grows εn during transport.
Tune 4 solenoid `kb`, 4 cavity `ke`, 4 cavity `phase` (12 variables)
to minimise the 4-D emittance growth.

This is the recommended workflow for HWR-class matching problems.

### The lattice cards (excerpt)

```
ADJUST FMAP 5 1 -1.98 -1.62 0.02 0     ; sol1 kb
ADJUST FMAP 6 2  0.248 0.372 0.005 0   ; cav1 ke
ADJUST FMAP 2 3 -60.0 -40.0  1.0 0     ; cav1 phase
…  ; sol2..4 + cav2..4 the same way
MIN_EMIT_4D_GROWTH 1.0 1.05 1.20       ; cost
SET_KE_OUT_MIN 5.0 10.0                ; energy floor
```

### Python API

```python
result = match(
    lat, cfg,
    space_charge=True,                # use SC during matching
    algorithm="cmaes",
    max_iter=80,                      # CMA-ES generations
    cmaes_sigma=0.15,                 # 15% of bound width per step
    cmaes_popsize=0,                  # 0 = library default (4 + 3·ln 12 ≈ 11)
    cmaes_parallel=4,                 # evaluate population on 4 cores
    refine=True,                      # chain LS polish from best CMA-ES x
)
print(f"baseline cost {result.baseline_cost:.3e} → final {result.cost:.3e}")
print(result.message)                 # "CMA-ES: tolfun + tolx  (parallel x4) + LS refine"
```

### CLI

```bash
python -m linac_gen.matching pipii_hwr_cm_match_4d_relaxed.dat \
    --algorithm cmaes \
    --max-iter 80 \
    --cmaes-sigma 0.15 \
    --cmaes-parallel 4 \
    --space-charge --current 10 \
    --report
```

### GUI

1. **File → Open Lattice** → pick the HWR `.dat`.
2. **Matching tab** → **Algorithm: cmaes**.
3. Set **σ** = 0.15, **popsize** = 0 (auto), **Max iter** = 80,
   **Space charge** ticked.
4. Click **Match**.  Watch εnx/εny/εnz drop in the Live Convergence
   dialog's emittance plot.  CMA-ES population evaluations are clearly
   visible as "noisy" cost trace; the dashed best-so-far line is monotone.
5. After convergence, the status line shows
   `OK — N iters, cost baseline → final, T s`.
6. **Apply** writes the matched values into the in-memory lattice and
   marks the project dirty.
7. **File → Save Lattice As…** + **File → Save Project** to persist.

---

## Recipe 3 — Sequential_scan with seed-exit threshold

**Problem**: Same HWR lattice as Recipe 2, but you want **physics-aware
coordinate descent** instead of CMA-ES — walk each element in lattice
order, watch the emittance impact directly, reverse when the trial
overshoots.

This is the natural choice for "validating a hand-tuned baseline" or
when you want transparency into which knob did what.

### Python API

```python
result = match(
    lat, cfg,
    space_charge=True,
    algorithm="sequential_scan",
    seqscan_passes=2,                 # 2 passes through the element list
    seqscan_steps=7,                  # 7 bracket steps per knob
    seqscan_step_frac=0.10,           # 10% of (vmax-vmin) per step
    seqscan_reversal="both_grew",     # reverse when BOTH εx,εz above ref
    seqscan_threshold="seed_exit",    # ref = NOMINAL lattice exit ε
                                      # (natural for intrinsic growth)
    seqscan_element_names=None,       # None = scan all ADJUST'd elements
)
```

When to pick each `seqscan_threshold`:

* **`"input"`** — for *coupling-resonance* lattices.  ε_x ↓ pulls ε_z ↑
  but reversal fires only if both planes exceed their **input** values
  simultaneously.  Plane-exchange is permitted; absolute growth on both
  is rejected.
* **`"seed_exit"`** — for lattices with *intrinsic growth* (lattice +
  SC make ε grow at x0).  Reversal fires only when the trial is
  **worse than the nominal/unmatched exit ε** — gives the optimiser
  freedom to explore while still improving on baseline.

### CLI

```bash
python -m linac_gen.matching pipii_hwr_cm_match_4d_relaxed.dat \
    --algorithm sequential_scan \
    --seqscan-passes 2 \
    --seqscan-steps 7 \
    --seqscan-step-frac 0.10 \
    --seqscan-reversal both_grew \
    --seqscan-threshold seed_exit \
    --space-charge --current 10 \
    --report
```

To restrict the walk to specific elements (e.g. cavities 1–4 only):

```bash
    --seqscan-element FMAP_002 --seqscan-element FMAP_005 \
    --seqscan-element FMAP_008 --seqscan-element FMAP_011
```

### GUI

1. **Matching tab** → **Algorithm: sequential_scan**.
2. Click **Match**.  The **Sequential Scan Setup** dialog opens.
3. **Element table** — tick which elements to include (defaults to
   all ADJUST'd).
4. **Passes** = 2, **Steps per parameter** = 7, **Step size** = 0.10,
   **Direction reversal** = `Reverse when BOTH …`,
   **Reversal reference** = `vs SEED EXIT ε`.
5. **Start Scan**.  Live Convergence dialog shows the per-element
   scan progress in the "Live values" panel:

   ```
   pass 2/2  step 4/7 (+)  ·  FMAP_002.[ke]  ·  ke=0.3187
   εnx 0.2143  εny 0.2089  εnz 1.2340  W_out 10.2630 MeV
   ```

6. **Stop** cancels mid-step; lattice is restored to the best-cost x
   seen so far, Variables table populates from that.

### Runnable example

The `examples/sequential_scan/` directory (in the repo, with its own
`README.md`) ships `seqscan_demo.dat` + `run_sequential_scan.py`, which
contrasts the two reversal references (`input` vs `seed_exit`) and
exercises the hard loss-rejection rail (`seqscan_reject_loss`).

```bash
python examples/sequential_scan/run_sequential_scan.py
```

---

## Recipe 4 — MP-mode cost solver for physics-accurate matching

**Problem**: The envelope match in Recipe 2 converged, but the matched
lattice's MP simulation shows ε_4D growth that envelope didn't predict
(nonlinear SC + halo).  Re-match using full multi-particle tracking
as the cost-driver.

**Important**: Wall time is **50–100× longer per evaluation** than
envelope.  Budget hours, not minutes.

### Python API

```{.python .skip}
# Long-running (~440 multi-particle PIC runs) — shown for the parameter
# pattern; budget an hour-class job for a real lattice.
from linac_gen.core.config import SpaceChargeConfig
from linac_gen.core.step_config import StepConfig

result = match(
    lat, cfg,
    space_charge=True,
    algorithm="cmaes",
    max_iter=40,                      # smaller budget; ~440 MP runs
    cmaes_popsize=11,
    cost_solver="mp",                 # ← key change
    mp_n_particles=1000,              # macroparticle count
    mp_seed=42,                       # fixed seed = reproducible cost
    mp_sc_config=SpaceChargeConfig(
        nx=48, ny=48, nz=48,
        grid_extent=5.0,
        green_kind="igf",
        kernel="cic",
        use_gpu="auto",
    ),
    mp_step_config=StepConfig(
        integration_steps_per_metre=100,
        sc_steps_per_metre=50,
    ),
)
```

The MP cost-solver runs each evaluation through `Simulation` →
`DiagnosticRecorder`, so the cost reflects the same physics MP
validation runs use.  Recommended workflow: envelope-match first
(Recipe 2, minutes) → MP-match for polish (Recipe 4, overnight).

### CLI

```bash
python -m linac_gen.matching pipii_hwr_cm_match_4d_relaxed.dat \
    --algorithm cmaes --max-iter 40 --cmaes-popsize 11 \
    --cost-solver mp --mp-n-particles 1000 --mp-seed 42 \
    --space-charge --current 10 \
    --report
```

(The CLI uses library-default `SpaceChargeConfig` / `StepConfig` for
the MP path; pass the `--nx --grid-extent --step1 --step2 --kernel
--backend` flags through `linac_gen.cli.run` if you need to override
them.  The GUI reads these from the Numerics tab.)

### GUI

1. **Numerics tab** — set PIC grid, SC engine, Green's function,
   kernel, step density to the values you want for the cost-solver MP
   pass.  (These are the same settings used by Run Envelope / Run MP
   buttons.)
2. **Matching tab** → **Cost solver** dropdown → select **Multi-particle**.
3. **n particles** spinbox appears — set to 1000 (or fewer for a quicker
   first pass).
4. **Algorithm: cmaes**, **Max iter: 40**, **Space charge** ticked.
5. Click **Match**.  Expect each evaluation to take ~10–60 s (vs ~0.1–1 s
   in env mode).  Total wall time for 40 generations × 11 popsize ≈
   440 evals × 30 s = ~3.7 hours.
6. The Live Convergence dialog ticks slowly but shows the same cost +
   emittance traces.

---

## Recipe 5 — Cancellation: clicking Stop preserves best-x

**Problem**: A long CMA-ES run has clearly converged; you want to stop
early and keep the best result so far.

### Python API (`callback` raises `StopIteration`)

```python
def cb(it, x, cost):
    if cost < 1e-6 or it > 200:
        raise StopIteration("good enough or budget spent")

result = match(lat, cfg, algorithm="cmaes", max_iter=1000, callback=cb)
print(result.success)        # False
print(result.message)        # "...cancelled by user"
print(result.x_final)        # best-cost x seen, not the in-flight one
```

The cancellation contract: when the callback raises `StopIteration`
during *any* of the seven algorithms (LS, DE, DA, CMA-ES sequential or
parallel, gradient, sequential_scan, bayesopt), `match()` applies `state["best_x"]` to the
lattice and returns a `MatchResult` with `success=False`.  `result.x_final`
holds the best-cost x, not the trial point that triggered cancellation.
The lattice is in sync with `result.x_final` on return.

### CLI

There's no `--max-time` flag yet; the CLI runs to convergence or
`--max-iter`.  Pressing `Ctrl-C` kills the process — no graceful
cancellation; the matched `.dat` is not written.  Use the Python API
(callback raising `StopIteration`) or the GUI Stop button if you need
the "stop and keep best-x" behaviour from a long-running CLI session.

### GUI

1. While a match is running, click **Stop**.
2. The matcher cancels at the next callback firing.
3. Lattice is restored to best-cost x; **Variables table** populates
   from that x (you can still click **Apply** to commit).
4. Status line shows `Cancelled — N iters, cost baseline → best, T s`.

---

## Recipe 6 — Parallel CMA-ES on a heavy lattice

**Problem**: Each forward pass on a FieldMap3D-heavy lattice takes ~5 s;
CMA-ES popsize=15 × 50 generations = 3750 s ≈ 1 h sequential.  You have
8 cores idle.

### Python API

```{.python .skip}
# Long-running (spawns 8 worker processes; startup cost dominates on a
# demo lattice) — the pattern matters, not the wall-time here.
result = match(
    lat, cfg,
    algorithm="cmaes",
    max_iter=50,
    cmaes_popsize=15,
    cmaes_parallel=8,                 # 8 worker processes
)
```

`cmaes_parallel=0` auto-detects `min(popsize, cpu_count-1)`.
`cmaes_parallel=1` is sequential (matches non-parallel exactly).
The result is **bit-identical** to the sequential one for the same RNG
seed and popsize (just faster).

**Caveat**: when surrogates are active, the fork overhead can eat the
speedup.  Profile both before committing.

### CLI

```bash
python -m linac_gen.matching heavy_fieldmap.dat \
    --algorithm cmaes --max-iter 50 --cmaes-popsize 15 \
    --cmaes-parallel 8 \
    --report
```

### GUI

Currently the GUI does not expose `cmaes_parallel` directly — it
defaults to `1`.  Workaround: drive parallel matches from the CLI or
Python API.  The GUI worker single-threads anyway because it runs in
a Qt `QThread`; the speedup comes from worker processes spawned
*inside* the matcher.

---

## Recipe 7 — Guarding against beam loss with `MIN_TRANSMISSION`

**Problem**: Recipe 2's HWR match minimises ε_4D using
`MIN_EMIT_4D_GROWTH`.  In MP mode this constraint is *gameable* — the
matcher can drop ε by killing halo particles (ε is computed from
*alive* particles only).  Without an explicit transmission penalty,
the matcher may find a solution that loses particles to clean up the
distribution.  Add a transmission floor.

### Add the card

Edit the lattice's constraints block:

```text
; ---------- Constraints ----------
MIN_EMIT_4D_GROWTH 1.0 1.05 1.20         ; existing ε cost
SET_KE_OUT_MIN 5.0 10.0                  ; existing energy floor
MIN_TRANSMISSION 99.9 50.0               ; NEW: loss penalty
END
```

### What it does

The new constraint contributes one residual per evaluation:

```
r = 50.0 × max(0, 99.9 − T_final) / 100
```

So a trial that drops transmission to 99% (1% loss below the 99.9%
threshold) pays a residual of **0.5**.  Compare to the ε residuals
(typically O(0.01–0.1)): the matcher will strongly avoid losing
particles to "win" on ε.

### Python API / CLI / GUI

The constraint is opt-in via the lattice card; no Python kwarg, CLI
flag, or GUI control is needed.  Once the card is in the `.dat`,
every match() / `python -m linac_gen.matching` / GUI Match run picks
it up automatically via `collect_constraints()`.

### Critical: MP mode required

`MIN_TRANSMISSION` is **inert in envelope mode** because envelope's
matrix tracker doesn't model aperture clipping.  You'll see a one-time
stderr warning the first time the constraint is evaluated against env
results:

```text
[match] MIN_TRANSMISSION found but results have no transmission field
(envelope mode?).  Constraint is INERT -- switch to cost_solver='mp'
to enforce the transmission floor.
```

For the constraint to actually do anything, run the match with the MP
cost-solver (Recipe 4): `--cost-solver mp` on the CLI or the GUI's
**Cost solver → Multi-particle** dropdown.

### Weight calibration

| Threshold | Weight | Effect |
|---|---|---|
| 99.0 | 1.0 | 0.5% loss costs residual 0.005 — soft hint |
| 99.0 | 10.0 | 0.5% loss costs 0.05 — comparable to ε residual |
| 99.9 | 50.0 | 0.1% loss costs 0.05 — strict floor |
| 99.99 | 100.0 | any detectable loss is catastrophic |

For a 10-mA CW linac where 0.1 mA hitting structures means quenching
or activation, pick threshold 99.9–99.99% and weight 50–100×.  For
loose proof-of-concept matching, 99.0 / 1.0 is fine.

---

### Soft penalty vs hard rejection

`MIN_TRANSMISSION` is a **soft residual** added to the cost — the
optimiser pays for loss but a big ε drop could still outweigh it.
For sequential_scan there's also a **hard rejection** rule
(`seqscan_reject_loss`) that doesn't give the loss-inducing step
*any* credit, regardless of its ε:

| Mechanism | Algorithms | Behaviour |
|---|---|---|
| `MIN_TRANSMISSION` card | all | soft residual `weight × max(0, threshold − T_final) / 100` added to cost.  Loss is *expensive* but not forbidden. |
| `seqscan_reject_loss=True` | sequential_scan only | **hard**: a loss-inducing step is rolled back (best_x not updated to it) and direction flips.  Walk only enters loss-free regions. |

You can combine both for belt-and-braces (soft penalty across all
trial points + hard rejection during the coordinate-descent walk).

### Python API — combined recipe

```{.python .skip}
# Long-running (multi-particle runs with space charge per scan step).
result = match(
    lat, cfg,
    algorithm="sequential_scan",
    cost_solver="mp",                          # required for loss tracking
    mp_n_particles=1000,
    space_charge=True,
    seqscan_passes=2, seqscan_steps=7,
    seqscan_step_frac=0.10,
    seqscan_reversal="both_grew",
    seqscan_threshold="seed_exit",
    seqscan_reject_loss=True,                  # HARD: roll back loss steps
    seqscan_loss_threshold_pct=99.9,           # 0.1% loss tolerated
)
```

### CLI

```bash
python -m linac_gen.matching pipii_hwr_cm_match.dat \
    --algorithm sequential_scan \
    --cost-solver mp --mp-n-particles 1000 \
    --space-charge --current 10 \
    --seqscan-passes 2 --seqscan-steps 7 \
    --seqscan-step-frac 0.10 \
    --seqscan-reversal both_grew \
    --seqscan-threshold seed_exit \
    --seqscan-reject-loss \
    --seqscan-loss-threshold-pct 99.9 \
    --report
```

### GUI

1. Matching tab → **Algorithm: sequential_scan**, **Cost solver: Multi-particle**.
2. Click **Match** → setup dialog.
3. Tick **"Reject any step that causes beam loss (HARD rule)"**.
4. **Loss-rejection threshold: 99.9 %**.

### Runnable example

The `examples/min_transmission/` directory (in the repo, with its own
`README.md`) ships `min_transmission_demo.dat` (a 6 mm aperture that clips
~2% of a moderate beam at the seed) + `run_min_transmission.py`, which
matches with a `MIN_TRANSMISSION 99.0` card and shows transmission recover
from the seed to ≥99% while the constraint residual falls to ~0:

```bash
python examples/min_transmission/run_min_transmission.py
```

---

## Recipe 8 — Algorithm selection by problem shape

A practical cheat-sheet for picking `algorithm`:

| Symptom | Algorithm | Why |
|---|---|---|
| Smooth cost, well-conditioned, you trust x0 | `least_squares` | LM is fastest when applicable |
| Correlated knobs (cavity ke+phase pairs, quad triplets) | `cmaes` | Learns the covariance; walks diagonally |
| Multimodal cost, suspected local minima | `differential_evolution` | Population-based global; slower |
| Strong nonlinearity in cavity phase / synchrotron motion | `cmaes` + `refine=True` | LS-style polish struggles with sin curves; CMA-ES doesn't |
| Lattice has FieldMaps with (kb, ke, phase) groupings, physics-intuitive ordering desired | `sequential_scan` | Coordinate descent with FieldMap-aware grouping |
| Linear lattice (no RF/FieldMap), exact gradients desired | `gradient` | Autograd Jacobian; fastest when applicable |
| Need to inspect WHICH knob did WHAT | `sequential_scan` | Per-element transparency |
| 1-D matching (one variable) | `least_squares` | Overkill to use anything else |

For HWR matching specifically: **sequential_scan first** (gets close
quickly with physics intuition), **then CMA-ES** with `refine=True`
(finds the optimum precisely), **optionally MP-mode CMA-ES** for the
physics-validated final answer.

---

## Tips

* **Always check `baseline_cost`** in the result.  If `cost > 0.1 *
  baseline_cost` after a "successful" match, you might have ill-posed
  constraints (too many vs variables).
* **Bounds matter**: tight bounds help convergence but can cause failure
  if the true solution is outside.  Start loose (`±50%` of best-guess
  value), tighten only after success.
* **`refine=True`** is almost always worth it after CMA-ES — sub-second
  polish that tightens active constraints to LM precision.
* **MP-mode determinism**: with a fixed `mp_seed`, calling `fn(x)` twice
  gives the *same* cost.  Without it, RNG noise would defeat the
  optimiser.
* **Cancellation is safe**: clicking Stop never leaves the lattice in a
  half-mutated state.  See Recipe 5.

## Cross-references

* [Overview](01_overview.md) — what the matcher solves.
* [SET / ADJUST](02_set_adjust.md) — `.dat` card syntax.
* [Python API](03_python_api.md) — `match()` and `MatchResult`.
* [CLI](04_cli.md) — every flag.
* [Sequential-scan](07_emittance_min.md#sequential-scan) — the
  full algorithm description.
* [Worked example: matching walkthrough](../11_examples/07_matching_walkthrough.md).

← [CLI](04_cli.md) ·
[Continue to Gradient algorithm →](06_gradient_algorithm.md)
