# Python API

This is the developer reference for HELIX's top-level entry points.
Every workflow comes back to one of these classes/functions.

## At a glance

```python
from linac_gen.core.particle import PROTON
from linac_gen.core.reference import ReferenceParticle
from linac_gen.core.lattice import Lattice
from linac_gen.core.simulation import Simulation
from linac_gen.core.config import BeamConfig, SpaceChargeConfig
from linac_gen.distributions.factory import create_beam
from linac_gen.io.tracewin_parser import parse_tracewin
from linac_gen.tracking.envelope import EnvelopeSolver
```

## `Simulation` — top-level facade

Wraps lattice + beam + SC config and gives you `run()` (multi-particle)
and `run_envelope()` (RMS).

```{.python .skip}
sim = Simulation(
    lattice: Lattice,
    beam: Beam,
    space_charge: SpaceChargeConfig | None = None,
    snapshot_locations: list[str] | None = None,
    snapshot_every_n: int | None = None,
    record_substeps: bool = False,
    progress_callback: Callable | None = None,
    should_abort: Callable[[], bool] | None = None,
)

results = sim.run()                # multi-particle tracking
env_results = sim.run_envelope()   # envelope (RMS) tracking
bt_results = sim.run_backtrack()   # backward tracking (see below)
```

| Parameter | Meaning |
|---|---|
| `lattice` | `Lattice` to track through |
| `beam` | `Beam` (carries `ReferenceParticle`) |
| `space_charge` | `SpaceChargeConfig` for PIC; `None` ⇒ no space charge |
| `snapshot_locations` | list of element names to dump full phase space at |
| `snapshot_every_n` | integer — every N steps |
| `record_substeps` | True ⇒ record diagnostics at every substep (more memory) |
| `progress_callback` | called as `cb(s_done, s_total)` for GUI progress bars |
| `should_abort` | called every step; if returns True, simulation stops |

### `Simulation.run() → DiagnosticRecorder`

Multi-particle tracking via `Tracker`.  Returns the
`DiagnosticRecorder` with all per-step arrays.

### `Simulation.run_envelope() → EnvelopeResults`

RMS envelope tracking via `EnvelopeSolver`.  Returns a results
object with the same `s, sigma_x, sigma_y, sigma_phi, ...` arrays.

### Source

`linac_gen/core/simulation.py:1`

## `EnvelopeSolver` — direct envelope path

For matching loops or when you want envelope-only without the
`Simulation` wrapper:

```{.python .skip}
from linac_gen.tracking.envelope import EnvelopeSolver

solver = EnvelopeSolver(
    lattice: Lattice,
    ref: ReferenceParticle,
    initial: dict,               # {emit_x, alpha_x, beta_x, ...}
    current: float = 0.0,        # mA
    progress_callback: Callable | None = None,
    should_abort: Callable | None = None,
    record_substeps: bool = False,
)
results = solver.run()
```

`initial` keys (all required):

```python
initial = dict(
    emit_x=..., alpha_x=..., beta_x=...,   # geometric ε in mm·mrad
    emit_y=..., alpha_y=..., beta_y=...,
    emit_z=..., alpha_z=..., beta_z=...,   # ε_z in deg·MeV
    continuous=False,                       # True for DC pre-RFQ
)
```

!!! warning "ε in `initial` is GEOMETRIC, not normalised"
    Convert from `BeamConfig.emit_nx` (which is normalised) by
    dividing by `ref.bg`:
    ```python
    bg = ref.bg
    initial = dict(
        emit_x=cfg.emit_nx / bg,
        ...
    )
    ```

### Source

`linac_gen/tracking/envelope.py:1`

## `Tracker` — direct multi-particle path

For when you need to drive the tracker manually (e.g. injecting
errors mid-flight, custom diagnostics):

```python
from linac_gen.tracking.tracker import Tracker
from linac_gen.core.config import BeamConfig
from linac_gen.core.lattice import Lattice
from linac_gen.distributions.factory import create_beam
from linac_gen.elements.drift import Drift
from linac_gen.elements.quadrupole import Quadrupole

lattice = Lattice()
lattice.add(Quadrupole(name="QF", length=100.0, gradient=+10.0))
lattice.add(Drift(name="D1", length=200.0))
lattice.add(Quadrupole(name="QD", length=100.0, gradient=-10.0))
lattice.add(Drift(name="D2", length=200.0))
beam = create_beam(BeamConfig(n_particles=1000), seed=42)
pic = None                         # or PicSolver(SpaceChargeConfig(...))

tracker = Tracker(
    lattice=lattice,
    beam=beam,
    pic_solver=pic,                # PicSolver instance or None
    snapshot_locations=None,
    snapshot_every_n=None,
    record_substeps=False,
    progress_callback=None,
    should_abort=None,
)
results = tracker.run()
```

`pic_solver` is created from `SpaceChargeConfig` via:

```python
from linac_gen.pic.pic_solver import PicSolver
pic = PicSolver(SpaceChargeConfig(nx=64, ny=64, nz=64))
```

### Source

`linac_gen/tracking/tracker.py:1`

## `create_beam()` — distribution factory

```{.python .skip}
from linac_gen.distributions.factory import create_beam
beam = create_beam(config: BeamConfig, seed: int | None = None) -> Beam
```

* Looks up species, creates `ReferenceParticle`.
* Converts normalised `emit_nx, emit_ny` to geometric (÷ βγ).
* Applies mismatch factors.
* Dispatches to the right distribution generator.
* Constructs the `Beam` with centroid offsets applied.

### Source

`linac_gen/distributions/factory.py:1`

## `parse_tracewin()` — `.dat` parser

```{.python .skip}
from linac_gen.io.tracewin_parser import parse_tracewin

lattice, metadata = parse_tracewin(
    filepath: str,
    strict: bool = False,
    base_dir: str | None = None,
)
```

* `strict=False` (default) — unknown or malformed cards are **skipped**
  and a warning string is appended to `metadata["warnings"]`.
* `strict=True` — unknown or malformed cards raise `ValueError`.
* `base_dir` — directory for resolving `FIELD_MAP` file references;
  default is the `.dat` file's directory.

Returns:

* `lattice` — fully constructed `Lattice`, including `errors`,
  `beam_errors`, `error_ratios`, `error_cutoff` from `ERROR_*`
  directives.  A `PARTRAN_STEP` card lands on `lattice.step_config`;
  `;@LG` directives are mirrored onto `lattice.lg_options`.
* `metadata` — dict with exactly three keys: `title` (from the
  `TITLE` card), `warnings` (list of non-fatal parse issues) and
  `lg_options` (parsed `;@LG key=value` pairs).

### Source

`linac_gen/io/tracewin_parser.py:1`

## Backward tracking

`backtrack_distribution` walks a distribution from the exit of an
element range to its entrance by undoing every forward operation in
reverse order — see
[Concepts → Backward tracking](../02_concepts/03_tracking_modes.md#backward-tracking)
for the physics and the invertibility limits.  Field maps are undone
exactly by default (``field_map_mode="rk4"`` — each forward
integration step inverts in closed form; full-MEBT round trips close
at ~1e-13); pass ``field_map_mode="linear"`` for the v1 fitted-matrix
inverse.

A complete round trip — forward-track a beam, backtrack the exit
distribution, and recover the input exactly (drifts, quads, steerers
and RF gaps invert in closed form):

```python
import numpy as np
from linac_gen.core.beam import Beam
from linac_gen.core.lattice import Lattice
from linac_gen.core.particle import PROTON
from linac_gen.core.reference import ReferenceParticle
from linac_gen.elements.drift import Drift
from linac_gen.elements.quadrupole import Quadrupole
from linac_gen.elements.rf_gap import RFGap
from linac_gen.tracking.tracker import Tracker
from linac_gen.tracking.backtrack import backtrack_distribution

lat = Lattice()
lat.add(Quadrupole("QF", 50.0, gradient=5.0, n_steps=5))
lat.add(Drift("D1", 200.0))
lat.add(Quadrupole("QD", 50.0, gradient=-5.0, n_steps=5))
lat.add(Drift("D2", 200.0))
lat.add(RFGap("GAP", voltage=0.4, phase=-30.0, frequency=352.21, ttf=0.85))
lat.add(Drift("D3", 150.0))

def make_ref():
    return ReferenceParticle(species=PROTON, w_kin=3.0, frequency=352.21)

rng = np.random.default_rng(42)
beam = Beam(ref=make_ref(), n_particles=500, current=0.0)
for col, sig in enumerate([1.0, 0.3, 1.0, 0.3, 4.0, 0.003]):
    beam.particles[:, col] = rng.normal(0.0, sig, 500)
p_in = beam.particles.copy()          # snapshot the true input

Tracker(lat, beam).run()              # forward: beam is now at the exit
rec = backtrack_distribution(lat, beam, make_ref())   # walk it back

assert rec.direction == "backward"
assert np.allclose(beam.particles, p_in, rtol=1e-8, atol=1e-11)
print(f"round-trip max error: {np.abs(beam.particles - p_in).max():.2e}")
```

Signature and the key options:

```{.python .skip}
rec = backtrack_distribution(
    lattice, beam, entrance_ref,      # beam = the EXIT distribution
    start=0, end=None,                # element range [start, end]
    sc_config=None, pic_solver=None,  # space charge (see note below)
    allow_dc_crossing=False,          # assert DC beam upstream of buncher
    energy_tolerance=1e-3,            # warn if |ΔW|/W_design_exit above
    energy_hard_limit=5e-2,           # refuse above
)
```

* `beam` — the distribution **at the exit plane** (e.g. loaded from a
  measured `.dst`; its `beam.ref` then carries the file's energy).  On
  return it holds the reconstructed upstream distribution.
* `entrance_ref` — the **design entrance reference** (element 0); the
  machine's fields and phases are evaluated on its forward replay.
* The returned recorder is reversed to increasing `s`: index 0 is the
  reconstructed entrance, the last row is the supplied exit
  distribution (tagged `"INPUT"`), and `rec.direction == "backward"`.

!!! tip "Undoing a run with space charge"
    Pass the **forward run's own solver** for an exact SC undo — with
    the default fixed-grid PIC a fresh backward solver cannot rebuild
    the forward grid.  Via the facade this is automatic:
    `sim.run()` followed by `sim.run_backtrack()` reuses the solver.
    From a `.dst` alone (no forward run in the session), either track
    forward with `grid_mode="adaptive"` or accept the warned
    adaptive-grid fallback.

### `Simulation.run_backtrack()`

The facade wires the same call into the `Simulation` object.  Undoing
a forward run made in the same session is two lines:

```python
from linac_gen.core.simulation import Simulation

beam2 = Beam(ref=make_ref(), n_particles=500, current=0.0)
beam2.particles[:] = p_in
sim = Simulation(lat, beam2)
sim.run()                             # forward
rec2 = sim.run_backtrack()            # exact undo (reuses SC solver too)
assert np.allclose(beam2.particles, p_in, rtol=1e-8, atol=1e-11)
```

For the measured-`.dst` workflow, build the `Simulation` with the
loaded exit distribution and pass the design entrance state
explicitly: `sim.run_backtrack(entrance_ref=design_entrance_ref)`.

### `backtrack_envelope()` — RMS backward transport

The envelope counterpart propagates `Σ_entry = M⁻¹ Σ M⁻ᵀ` per element
off the same replay table.  Give it the **desired exit Twiss** and it
returns the entrance Twiss that produces it:

```python
from linac_gen.tracking.backtrack import backtrack_envelope

exit_twiss = dict(alpha_x=0.5, beta_x=2.0, emit_x=0.25,
                  alpha_y=-0.3, beta_y=1.5, emit_y=0.22,
                  alpha_z=0.1, beta_z=3.0, emit_z=0.4)
env = backtrack_envelope(lat, make_ref(), exit_twiss)
print(f"required entrance Twiss: alpha_x={env.alpha_x[0]:+.3f} "
      f"beta_x={env.beta_x[0]:.3f} mm/pi.mrad")
```

Space charge raises `ValueError` here (the backward SC kick would
depend on the unknown upstream σ) — use
`matching.periodic.find_sc_matched_input_twiss` (forward shooting) for
the SC-consistent input match.

### Source

`linac_gen/tracking/backtrack.py:1`

## `match()` — matching engine

See [Matching → Python API](../07_matching/03_python_api.md).

## `ErrorStudy` — Monte-Carlo ensemble

See [Errors → Running studies](../08_errors/05_running_studies.md).

## Cross-references

* [Data model](../02_concepts/02_data_model.md) — what the objects are.
* [Tracking modes](../02_concepts/03_tracking_modes.md) — when to use each.
* [Reading results](04_results.md) — what you get back.

← [Differentiable PIC](../05_space_charge/06_differentiable.md) ·
[Continue to .dat reference →](02_tracewin_dat.md)
