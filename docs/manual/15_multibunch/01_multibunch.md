# Pulse & bunch-train studies (multibunch)

HELIX normally tracks **one** bunch (or a DC beam).  The multibunch
study (`linac_gen.train`) simulates a **pulse**: a train of bunches at
the bunch frequency with a definable chopped fill pattern, coupled
bunch-to-bunch through up to three physics channels —

1. **Cavity beam loading** — each bunch drains the fundamental mode;
   chopped patterns produce sawtooth/droop voltage along the pulse and
   bunch-by-bunch synchronous-phase shifts;
2. **Dipole-HOM long-range wakes** — cumulative beam breakup (BBU)
   along a linac, anchored against the Delayen/Gluckstern closed forms;
3. **Direct bunch-to-bunch space charge** — the PIC's ±1 bunch-train
   images made pattern-aware (and optionally fed the *previous* bunch's
   real distribution).

!!! warning "Strictly opt-in — off means bit-identical"
    This is a rarely-run study, not a default behaviour.  Nothing in
    `linac_gen.train` is imported by normal single-bunch workflows, and
    with every physics flag at its default (off) an N-bunch train is
    **bit-identical to N independent single-bunch runs** — the
    zero-coupling contract, enforced as a regression test
    (`tests/train/test_zero_coupling.py`) and treated as a product
    requirement.  The other half of the contract: required physics
    inputs are validated **loudly** — a missing cavity sidecar is a
    refusal that names exactly what is missing, never a silent default.

## Quick start

```python
from linac_gen.train import PulsePattern, TrainConfig, TrainPhysics, run_train

tc = TrainConfig(
    bunch_frequency_MHz=162.5,
    pattern=PulsePattern.from_rle("1*10 0*54 1*26"),   # 1=bunch, 0=chopped
    mode="mp",                                          # tracked
    physics=TrainPhysics(beam_loading=True),
    cavity_params="cavity_params.yaml",                 # sidecar, below
)
results = run_train(lattice, beam_config, tc, sc_config=sc)
results.save_hdf5("train.h5")
```

`run_train` dispatches on `mode`; `results` is a
[`TrainResults`](#results-hdf5-schema) with a per-bunch summary table.

## The four modes

| mode | what runs | scale | use for |
|---|---|---|---|
| `mp` | one full multiparticle pass per filled slot over the **shared** lattice; cavity state carried between passes | 10s of bunches | full-fidelity short trains, direct SC |
| `envelope` | one envelope solve per bunch | 100s | cheap transverse/longitudinal envelope trains (no loading/HOM hooks) |
| `fast` | per-slot phasor + centroid recursion, no tracking (one tracked design pass first) | full ~10⁵-slot pulses in seconds | whole-pulse droop/recovery, BBU trends |
| `hybrid` | fast pass over the whole pulse **records** per-cavity state per slot → selected bunches replayed as independent full-MP runs with that state applied | pulse + probes | "what does bunch 61,440 look like?" |

**Hybrid replay semantics**: pass 1 is the fast recursion; pass 2
replays the selected bunches (`select_bunches`, default `"auto"` —
pattern edges + probes) through `element_overrides`, in-process or in
parallel worker processes (`replay_parallel=True`, which fingerprints
the deck against the live lattice before running).  Hybrid **requires**
`physics.beam_loading` — with all coupling off every bunch is identical
and a replay would be a mislabelled single-bunch run (refused).

## Physics channels

### Cavity beam loading (fundamental mode)

Single-resonator phasor model in the rotating frame at each cavity's RF
frequency: the generator holds `V_design` rigidly (v1 generator model);
a bunch of charge `q` induces `|dV| = ω(R/Q)q/2` per passage
(**linac convention** `R/Q = V²/(ωU)`), the phasor decays and rotates
between arrivals with `τ = 2Q_L/ω` and the detuning; the kick a bunch
sees maps onto the existing per-element `voltage_rel`/`phase_offset`
slots — the same slots the error model uses, composed on any prior,
restored on teardown.

- **Half self-kick**: a bunch sees HALF its own induced voltage (the
  fundamental theorem of beam loading), predicted from the entry
  charge; the full induced phasor of the charge that actually
  traversed is added at exit.
- **ψ-pinning**: `SET_SYNC_PHASE` cavities calibrate lazily per pass
  and would re-fit against the *loaded* voltage — wrong, because the
  LLRF pins the design operating point.  The train's one **design
  pass** calibrates every such cavity once at nominal voltage and pins
  ψ for all bunch passes; pins are runner-scoped and cleared on
  teardown (even on abort), so a train can never contaminate later
  runs on the shared lattice.
- **Design quantities**: `dW_design` is measured by the design pass;
  `V_design = |dW/cos φ_s|` where derivable.  At a zero crossing
  (buncher, `cos φ_s ≈ 0`) it is **not** derivable — `v_design_MV`
  becomes a required sidecar input (loud refusal otherwise).

### Dipole-HOM wakes / cumulative BBU

Per-cavity dipole-mode tables (sidecar `hom_modes`) excite transverse
wake phasors; each bunch receives the accumulated kick and adds its own
excitation (transverse `R/Q_t` convention and kick law anchored against
Delayen, PRST-AB **6** 084402 / **7** 074402, and Gluckstern PRA **45**
5964 — `tests/train/test_hom.py`).

### Direct bunch-to-bunch space charge

`physics.direct_sc` makes the numpy 3-D PIC's ±1 bunch-train images
**pattern-aware** (a bunch after a chopped gap has no leading image),
with two neighbour models — `direct_sc_neighbors="images"` (scaled
copies of the live bunch; the Toutatis-validated machinery) or
`"distinct"` (the leading image is the previously tracked bunch's
subsampled snapshot).  Requires `mode="mp"` (or hybrid, images only)
and a numpy `SpaceChargeConfig`.  The PIC's σφ ≥ 35° engagement gate
still applies: a train whose images never engage **warns loudly**;
`direct_sc_force_engage=True` overrides the gate.

## Approximation ledger

What the v1 model does and does not couple — the honest boundary:

| approximation | statement |
|---|---|
| weak-coupling sequential pass | bunch k+1 sees the cavity state left by bunches 1..k (exact causality for loading/HOMs); there is **no back-action within a bunch's own pass** beyond the half self-kick |
| ψ-pin | the design operating point is LLRF-pinned; bunches never re-fit synchronous phases against loaded voltages |
| half self-kick | entry applies half the bunch's own predicted induced voltage (fundamental theorem); exit adds the full induced phasor of the traversed charge |
| generator model | the generator holds `V_design` rigidly — no LLRF feedback/adaptive compensation in v1 |
| hybrid: no pass-2→pass-1 feedback | replayed bunches do not update the fast pass's phasor histories (single forward sweep) |
| fast-mode ledger | per-bunch exit energy assumes a separable gain `V_eff·cos φ_eff` (exact for thin `RFGap`, first order for field maps/NCells), **no time-of-flight feedback**, loss-free charge `I/f_bunch`; the HOM centroid perturbation drifts between cavities (exact for the Delayen reduced lattice, first order elsewhere) |
| fast-mode ledger at zero crossings | at a **pure buncher** (\|cos φ_s\| ≈ 0) the ledger amplitude is the sidecar `v_design_MV` with a sign that follows how the phase was established — ψ-**calibrated** cavities (`SET_SYNC_PHASE`) absorb the species charge (effective `+V`, verified against tracked H⁻ on the PIP-II MEBT bunchers), **prescribed**-phase elements keep the raw `q·V` law; the *sign* matches tracking in both cases, but the *magnitude* remains per-cavity first order and overestimates chained/multi-gap responses by factors of ~2–4 (MEBT 4-cavity chain ×1.9, 4-cell NCells ×4) — the phase/phasor channel is unaffected and stays the validated output |
| distinct neighbours: shared Lorentz boost | the neighbour snapshot is deposited in the live bunch's boosted frame — first order in the inter-bunch energy difference of adjacent bunches |
| envelope frequency asymmetry | the envelope's exit-frequency carry-over ignores `frequency_offset` (it carries `element.frequency`; MP carries the effective frequency) — the solver **warns loudly** when one is set on a `mode="envelope"` train |
| unsupported backends | the torch PIC and the HALO-PIC learned corrector have **no train path** (`NotImplementedError` / refusal — numpy 3-D PIC only); `periodic_phase=True` beams are refused (the fold collapses the train structure) |
| inert knobs are refused | `charge_scale`, `jitter`, and explicit `select_bunches` outside hybrid are **refused at construction** — no v1 runner consumes them, and the study never accepts an input it would ignore |

Arrival-time observables across RF-frequency jumps additionally require
`FREQ` cards (the per-element fallback leaves the phase clock
unrescaled; the runner warns).

## Parameters

### `TrainConfig`

| field | meaning |
|---|---|
| `bunch_frequency_MHz` | bunch-slot rate; must equal `BeamConfig.bunch_frequency_MHz` when that is set (reconciled, mismatch refused) |
| `pattern` | a `PulsePattern` — see below |
| `mode` | `"mp"` / `"envelope"` / `"fast"` / `"hybrid"` |
| `physics` | `TrainPhysics(direct_sc=…, beam_loading=…, hom=…)`, all `False` by default |
| `cavity_params` | sidecar path (required when loading/HOM on) |
| `select_bunches` | hybrid replay selection: `"auto"` or absolute filled-slot indices (refused outside hybrid) |
| `keep_full_results` | `False` = summary-only per bunch (big tracked trains) |
| `seed` | bunch generation seed (all bunches identical in v1) |
| `direct_sc_neighbors` / `direct_sc_force_engage` / `direct_sc_subsample` | direct-SC knobs (above) |

### `PulsePattern`

`from_rle("1*10 0*54 1*26")` (value*count tokens), `uniform(n)`,
`from_duty(n_slots, keep, period)` (keep the first `keep` of every
`period`), `from_array(bool_array)`; `to_rle()` round-trips.  PIP-II
scale — 0.55 ms at 162.5 MHz ≈ 89,375 slots — is a trivially small
array.

### Cavity sidecar (YAML or JSON)

Maps element-**name patterns** (`fnmatch`) to cavity parameters.
Physics inputs are never defaulted: with beam loading on, `r_over_q` +
`q_loaded` are required per entry; with HOMs on, a non-empty
`hom_modes` table is required.

```yaml
# cavity_params.yaml — matched to lattice cavities by name pattern
"CAV_A*":
  r_over_q: 455.0        # Ohm, LINAC convention R/Q = V^2/(omega U)
  q_loaded: 5280.0
  detuning_Hz: 0.0       # optional
  v_design_MV: 0.085     # optional; REQUIRED for zero-crossing bunchers
  # phi_s_deg: -90.0     # optional; else derived from the element
  hom_modes:             # required only when physics.hom is on
    - {f_MHz: 585.0, r_over_q_t: 50.0, q_loaded: 1.0e4, polarization_deg: 0.0}
```

A pattern that matches **no** lattice cavity by name is refused loudly.

## Results & HDF5 schema

`TrainResults.save_hdf5(path)` writes `provenance/` (with
`run_type = "train"`), `train/` (pattern, per-bunch summary table,
per-cavity `cavity_state/`/`hom/` ledgers, `fast/`, hybrid `replay/`)
and `bunches/b_%04d/` groups; **single-bunch files are unchanged** and
old readers keep working.  The authoritative schema tree lives in the
`linac_gen/train/results.py` module docstring.  Load with:

```python
from linac_gen.train import load_train_results
ld = load_train_results("train.h5")
ld.summary["ref_w_kin"]       # per-bunch exit energies
ld.truncated                  # True when the run was aborted mid-train
```

A mid-train abort (GUI Stop, `should_abort` callback) stops
cooperatively **between** bunches: the partial result is fully
saveable/loadable and carries `truncated = True`.

## From the assistant / MCP

The `run_train` tool (long-running, background job) mirrors
`TrainConfig`: `bunch_frequency_MHz`, `pattern` (RLE) or `n_bunches`
(+ `duty_keep`/`duty_period`), `beam_loading`/`hom`/`direct_sc`,
`cavity_params`, `select_bunches`/`replay_parallel`/`history_stride`,
`space_charge`/`grid`, `out_path`.  TrainConfig's refusals surface
verbatim.  It saves the train HDF5 and returns bunch counts plus
W_exit droop numbers; `load_results` auto-routes train files to the
train loader, and `result_summary` then reports the per-bunch summary.

## From the GUI

**Simulate → Multibunch / Pulse Study…** opens the opt-in dialog: the
study switch (off by default — the form is inert until enabled), mode,
bunch frequency, the RLE pattern with a live slot-count/duty preview
**and a live pulse strip** (the fill pattern drawn against time in µs,
redrawn as you type), the physics checkboxes (enabling loading/HOM
reveals the sidecar picker), and the Numerics-tab space-charge switch.
OK validates the
whole configuration **in-dialog** (verbatim refusals).  On OK the
config dialog closes and the modeless **Multibunch summary** window
opens immediately in live-progress mode: it announces the design pass
(one full single-bunch calibration pass, so the bar starts moving only
after it), then tracks per-bunch progress; **Stop** aborts between
bunches and still auto-dumps the loadable partial result.  On
completion the run is auto-saved as `<timestamp>_train.h5` in the calc
directory and the same window becomes the per-bunch summary plot.  The
quantity dropdown includes the **fill pattern** (the pulse structure
itself), and the **time axis [µs]** toggle replots any quantity against
time in the pulse instead of slot index.  A saved train file can be
reopened later through the Results tab's **Import results…** — a train
file is detected by its schema and opens in its own summary window
(titled with the file name) instead of the single-run results cards.

