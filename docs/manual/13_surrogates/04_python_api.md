# Surrogates Python API

The full surrogate library lives in `linac_gen.surrogates`.  Five
small modules cover training, persistence, registration,
comparison, and the per-element surrogate class itself.

```{.python .skip}
from linac_gen.surrogates import (
    SurrogateFieldMap,        # the surrogate element class
    MlpHead,                  # the underlying MLP module
    Scope, SurrogateMetadata, # data classes
    OutOfScopeError,          # raised when input is OOD
    registry,                 # global runtime registry
)
from linac_gen.surrogates.training import (
    train_surrogate_for_element,
    generate_training_data, train_surrogate,
    save_surrogate, load_surrogate,
    find_cached_surrogate, discover_cached_surrogates,
)
from linac_gen.surrogates.compare import (
    compare_envelope, plot_compare_report,
    CompareReport,
)
```

## Training

The one-shot orchestrator:

```{.python .skip}
from linac_gen.core.particle import H_MINUS
from linac_gen.core.reference import ReferenceParticle
from linac_gen.io.tracewin_parser import parse_tracewin
from linac_gen.surrogates.training import train_surrogate_for_element

lat, _ = parse_tracewin("examples/pipii/mebt/mebt.dat")
elem = next(e for e in lat.elements if e.name == "FMAP_001")

ref_template = ReferenceParticle(species=H_MINUS, w_kin=2.12, frequency=162.5)

mlp, meta = train_surrogate_for_element(
    element=elem,
    ref_template=ref_template,
    n_samples=50_000,
    ref_w_kin_range=(2.0, 2.5),
    param_ranges={
        "ke":    (elem.ke * 0.80, elem.ke * 1.20),
        "phase": (elem.phase - 20.0, elem.phase + 20.0),
    },
    hidden_dims=(128, 128, 128),
    epochs=300,
    out_dir="weights/FMAP_001",
    lattice_hash="bb088243c9fb33ce",
    element_key="FMAP_001",
    n_workers=12,             # CPU parallel data gen
)
print(f"val MAPE = {meta.val_mape:.3e}")
```

### `train_surrogate_for_element(...)` signature

| Arg | Type | Required | Notes |
|---|---|---|---|
| `element` | `FieldMapElement` | ✅ | source whose `fitted_matrix(ref)` provides ground truth |
| `ref_template` | reference particle | ✅ | cloned per sample; only `w_kin` is varied |
| `n_samples` | int | ✅ | LHS sample count |
| `ref_w_kin_range` | `(lo, hi)` MeV | ✅ | scope on incoming kinetic energy |
| `param_ranges` | `dict[str, (lo, hi)]` | ✅ | element attributes to sweep |
| `out_dir` | path | ✅ | where to save `weights.pt` + `metadata.json` |
| `hidden_dims` | tuple of int | `(128,128,128)` | MLP hidden layer sizes |
| `activation` | `"silu"`/`"tanh"`/`"relu"` | `"silu"` | smooth = better Jacobians (M7-friendly) |
| `epochs` | int | `200` | training epochs |
| `lr` | float | `1e-3` | Adam learning rate |
| `batch_size` | int | `256` | minibatch size |
| `val_frac` | float | `0.2` | hold-out split for val MAPE |
| `seed` | int | `42` | LHS + train RNG seed |
| `lattice_hash` | str | `"unknown"` | recorded in metadata |
| `element_key` | str / None | `None` | defaults to `element.name` |
| `verbose` | bool | `False` | progress prints |
| `n_workers` | int / None | `None` | ≥2 enables `multiprocessing.spawn`; output bit-identical to serial for fixed seed |

### Building blocks (use if you need finer control)

```{.python .skip}
from linac_gen.surrogates.training import (
    generate_training_data, train_surrogate, save_surrogate,
)

X, Y, info = generate_training_data(
    element=elem, ref_template=ref_template,
    n_samples=200, ref_w_kin_range=(2.0, 2.5),
    param_ranges={"ke": (0.054, 0.082), "phase": (-110, -70)},
    seed=42, n_workers=8,
)
mlp, norm, val_mape = train_surrogate(
    X, Y, hidden_dims=(64, 64), epochs=40, lr=3e-3, seed=42,
)
save_surrogate(mlp, metadata, "weights/FMAP_001")
```

## Loading and discovery

```{.python .skip}
from linac_gen.surrogates.training import (
    load_surrogate, find_cached_surrogate, discover_cached_surrogates,
)

# Always-load (raises if missing).
mlp, meta = load_surrogate("weights/FMAP_001")

# Try-load: returns None if dir missing / files incomplete / metadata bad.
result = find_cached_surrogate("weights/FMAP_001")
if result:
    mlp, meta = result

# Discover everything in a per-lattice weights root:
cached = discover_cached_surrogates(
    "linac_gen/surrogates/weights/bb088243c9fb33ce",
    element_names=["FMAP_001", "FMAP_002"],  # restrict; None = everything
)
# -> {"FMAP_001": (mlp, meta, Path), ...}
```

## The runtime registry

The envelope hook in `tracking/envelope.py:_fitted_matrix_slice_at`
looks up surrogates by element name on every matrix call.  Register
to engage, unregister to disengage.

```{.python .skip}
from linac_gen.surrogates import SurrogateFieldMap, registry

# Bind the trained MLP to a concrete element instance in the lattice.
surr = SurrogateFieldMap(elem, mlp, meta)

registry.register(surr)                       # engage
surr_back = registry.get_by_element_name("FMAP_001")
assert surr_back is surr

registry.unregister(meta.lattice_hash, "FMAP_001")  # disengage
registry.clear()                              # wipe everything
```

The registry is **process-global state**.  In a long-running process
that runs many comparisons, clear between runs to avoid stale
surrogates from a previous lattice contaminating the current one.

## Comparing envelopes

```{.python .skip}
from linac_gen.surrogates.compare import (
    compare_envelope, plot_compare_report,
)

ref = ReferenceParticle(species=H_MINUS, w_kin=2.12, frequency=162.5)
init_twiss = dict(
    alpha_x=1.228, beta_x=0.316, emit_x=0.21 / max(ref.bg, 1e-9),
    alpha_y=-0.0954, beta_y=0.113, emit_y=0.21 / max(ref.bg, 1e-9),
    alpha_z=0.0, beta_z=819.05492, emit_z=0.06231832,
)
report = compare_envelope(
    lat, ref, init_twiss,
    current=5.0,                       # mA; > 0 to engage SC slice path
    surrogates=[surr1, surr2, surr3, surr4],
)
print(report.summary_text())
plot_compare_report(report, "/tmp/cmp.png",
                     title="MEBT @ 5 mA: all-surrogate vs RK4")
```

### `CompareReport` fields

| Attribute | Type | Meaning |
|---|---|---|
| `s` | `ndarray` | arc length (mm) |
| `sigma_baseline` / `sigma_surrogate` | `dict[str, ndarray]` | keys `"x"`, `"y"`, `"phi"`, `"w"` |
| `end_of_line` | `dict[str, float]` | `sigma_{k}_base`, `sigma_{k}_surr`, `rel_diff_{k}` |
| `wall_baseline_s` / `wall_surrogate_s` | float | seconds |
| `scope_ok` | bool | `True` if every query was in scope (OOD silently falls back) |
| `surrogate_names` | `list[str]` | names engaged this run |
| `notes` | `list[str]` | warnings (e.g. "no surrogate engaged") |
| `speedup()` | method | `wall_baseline_s / wall_surrogate_s` |
| `worst_rel_diff()` | method | max of the four end-of-line rel.diffs |
| `summary_text()` | method | multi-line CLI-friendly summary |

`compare_envelope` **preserves the global registry state** —
whatever was registered before the call is restored after, so it's
safe to call in a long-running session.

### Comparing multi-particle runs

The MP analogue of `compare_envelope` runs the multi-particle
tracker twice — a pure-RK4 baseline (registry temporarily cleared,
MP engagement off), then the hybrid linear-anchor + RK4-residual
path with the surrogates registered — and diffs the two:

```{.python .skip}
from linac_gen.surrogates.compare import (
    compare_mp, plot_compare_mp_report,
)

report = compare_mp(lat, beam, surrogates=[surr1, surr2],
                    residual_n_steps=15)
print(report.summary_text())
plot_compare_mp_report(report, "/tmp/cmp_mp.png")
```

`CompareMpReport` mirrors `CompareReport` but for MP quantities:
`s`, `sigma_baseline` / `sigma_surrogate` (keys `"x"`, `"y"`,
`"z"`), `emit_baseline` / `emit_surrogate` (keys `"nx"`, `"ny"`),
`transmission_baseline` / `transmission_surrogate`, `end_of_line`
rel-diffs (σ_x/σ_y/σ_z/ε_nx/ε_ny/transmission), wall times,
`residual_n_steps`, plus the same `speedup()` / `worst_rel_diff()`
/ `summary_text()` helpers.  Like the envelope version, `compare_mp`
restores the whole registry + MP-engagement state afterwards, and
each surrogate's own `residual_n_steps` setting is restored too.

## Drop-in inference

When you've already registered surrogates, normal HELIX code paths
(envelope solver, matcher, parameter scans) pick them up
automatically — no changes to your scripts needed.

```{.python .skip}
from linac_gen.tracking.envelope import EnvelopeSolver
res = EnvelopeSolver(lat, ref, init_twiss, current=5.0).run()
# -> uses every surrogate currently registered when admissible.
```

For ad-hoc inference outside the envelope solver, call
`surr.fitted_matrix(ref)` directly:

```{.python .skip}
M = surr.fitted_matrix(ref)         # 6x6 numpy array
# raises OutOfScopeError if ref kinematics or surr's bound
# element params fall outside the trained scope.
```

## Out-of-scope handling

A surrogate is queried with an input vector
`[w_kin, beta, gamma, *param_values]`.  Each component must lie
within the metadata's scope bounds; otherwise:

```{.python .skip}
from linac_gen.surrogates import OutOfScopeError

try:
    M = surr.fitted_matrix(ref)
except OutOfScopeError as e:
    print(f"OOD: {e}")
    # The envelope hook catches this exception and silently falls
    # back to the wrapped RK4 element -- runs are correct but slow.
```

Programmatic check before calling:

```{.python .skip}
import numpy as np
x = np.array([ref.w_kin, ref.beta, ref.gamma, surr._wrapped.ke,
              surr._wrapped.phase])
if not surr.metadata.scope.is_in_scope(x):
    print("would fall back to RK4")
```

## Cross-references

* [Overview](01_overview.md) — what surrogates are.
* [GUI walkthrough](02_gui.md) / [CLI](03_cli.md) — equivalent
  end-user flows.
* [Training guide](05_training_guide.md) — sample sizing, MP,
  scope, troubleshooting.

← [CLI](03_cli.md) ·
[Continue to Training guide →](05_training_guide.md)
