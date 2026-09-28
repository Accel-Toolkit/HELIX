# Reading results

`Simulation.run()` and `EnvelopeSolver.run()` both return a results
object containing per-step diagnostic arrays.  This page is the
reader's reference: how to access fields, plot them, and save them.

## Result objects

| `.run()` call | Returns |
|---|---|
| `Simulation.run()` (multi-particle) | `DiagnosticRecorder` |
| `Simulation.run_envelope()` | `EnvelopeResults` |
| `EnvelopeSolver.run()` | `EnvelopeResults` |
| `Tracker.run()` | `DiagnosticRecorder` |

Both share the same field names for the common arrays
(`s, sigma_x, sigma_y, sigma_phi, sigma_w, emit_x, ...`), so plot
code is portable between them.

## Per-step arrays

Every field below is a `list[float]` with one entry per recording
point.  Common fields:

| Field | Units | Notes |
|---|---|---|
| `s` | mm | path length |
| `sigma_x`, `sigma_y` | mm | RMS transverse beam size |
| `sigma_phi` | deg | RMS phase spread (at local frequency) |
| `sigma_w` | MeV | RMS energy spread |
| `emit_x`, `emit_y` | mm·mrad | geometric ε |
| `emit_z` | deg·MeV | longitudinal ε (native) |
| `emit_z_mmmrad` | mm·mrad | ε_z converted to mm·mrad |
| `emit_nx`, `emit_ny`, `emit_nz` | mm·mrad | normalised |
| `emit_4d` | mm²·mrad² | 4-D coupling-invariant |
| `emit_n1`, `emit_n2` | mm·mrad | normal-mode 4-D eigenemittances |
| `emit_e1`, `emit_e2`, `emit_e3` | mm·mrad | 6-D Balandin eigenemittances |
| `alpha_x`, `beta_x`, `alpha_y`, `beta_y` | — / m | Twiss |
| `alpha_z`, `beta_z` | — / deg/MeV | longitudinal Twiss (2026-07) — HELIX-internal (Δφ, ΔW) convention (α_z = −TraceWin's), β_z at the local machine clock |
| `halo_x`, `halo_y` | — | Wangler halo parameter |
| `transmission` | % | live-particle fraction |
| `centroid` | (6,) | beam centroid in 6-D |
| `sigma_matrix` | (6,6) | full Σ |
| `x_max`, `y_max` | mm | peak excursion |
| `ref_w_kin`, `ref_phi_s` | MeV / deg | reference particle state |
| `ref_beta`, `ref_gamma`, `ref_bg` | — | reference kinematic factors |
| `ref_frequency` | MHz | per-step RF frequency (changes at FREQ jumps) |
| `element_names` | str | name of element this entry corresponds to |
| `continuous_at` | bool | True if pre-RFQ (no longitudinal physical meaning) |

For full coverage see [Recorder fields](../09_diagnostics/01_recorder.md).

## Quick plotting

```python
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from linac_gen.core.config import BeamConfig
from linac_gen.core.lattice import Lattice
from linac_gen.core.simulation import Simulation
from linac_gen.distributions.factory import create_beam
from linac_gen.elements.drift import Drift
from linac_gen.elements.quadrupole import Quadrupole

lattice = Lattice()
for c in range(2):
    lattice.add(Quadrupole(name=f"QF_{c}", length=100.0, gradient=+10.0))
    lattice.add(Drift(name=f"D1_{c}", length=200.0))
    lattice.add(Quadrupole(name=f"QD_{c}", length=100.0, gradient=-10.0))
    lattice.add(Drift(name=f"D2_{c}", length=200.0))
beam_cfg = BeamConfig(n_particles=1000)
beam = create_beam(beam_cfg, seed=42)
sim = Simulation(lattice, beam)

results = sim.run()
s_m = np.asarray(results.s) / 1e3   # mm → m for axes

fig, ax = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
ax[0].plot(s_m, results.sigma_x, label="σ_x")
ax[0].plot(s_m, results.sigma_y, label="σ_y")
ax[0].set_ylabel("σ [mm]")
ax[0].legend()
ax[1].plot(s_m, results.transmission)
ax[1].set_ylabel("transmission [%]")
ax[1].set_xlabel("s [m]")
fig.tight_layout()
plt.savefig("sigma.png", dpi=120)
```

## Saving / loading

### HDF5

```python
from linac_gen.io.hdf5_output import save_results_hdf5, load_results_hdf5

save_results_hdf5(results, "run.h5", beam_config=beam_cfg)
results2 = load_results_hdf5("run.h5")   # dict of arrays
```

`save_results_hdf5(recorder, filepath, beam_config=None, lattice=None,
lattice_path=None, seed=None, sc_config=None)` writes:

* **`envelope/`** — the per-step arrays `s`, `sigma_x`, `sigma_y`,
  `sigma_phi`, `sigma_w`, `emit_x`, `emit_y`, `emit_z`, `emit_nx`,
  `emit_ny`, `alpha_x`, `beta_x`, `alpha_y`, `beta_y`, `alpha_z`,
  `beta_z` (longitudinal Twiss, 2026-07 — internal convention,
  deg/MeV), `halo_x`, `halo_y`, `transmission`.  Run-level attributes
  ride on the group: `continuous`, plus the run-current pair
  `current_mA` — the beam current the run was configured with (the
  envelope solver, multi-particle tracker and backtracker all record
  it; `0.0` means the run was genuinely at 0 mA) — and a boolean
  `run_current_known` marker.  `current_mA` is present only when the
  current is known.  Files written before this pair existed stored
  `0.0` for every multi-particle run; `load_results_hdf5` recognises
  those legacy files (no marker) and substitutes `beam_config/current`,
  or omits `current_mA` entirely when the file has no `beam_config`
  group (unknown).
  The group also holds everything else a run records per step, so an
  imported file fills the same Results tiles as the live run (2026-09):
  `emit_z_mmmrad`, `emit_nz`, `emit_4d`, `emit_n1`, `emit_n2`, the
  eigenemittances `emit_e1`–`emit_e3`, `x_max`, `y_max`,
  `continuous_at`, `centroid` (S, 6), the full beam matrix
  `sigma_matrix` (S, 6, 6) and `element_names` (UTF-8 strings), plus
  the attributes `mass_mev` and `periodic_phase`.
  `load_results_hdf5` returns `centroid`, `sigma_matrix` and
  `element_names` as per-step lists, like a live recorder.
* **`reference/`** — 6 reference-particle arrays: `w_kin`, `phi_s`,
  `beta`, `gamma`, `bg` and the RF clock `frequency` (loaded back as
  `ref_w_kin`, `ref_phi_s`, …, `ref_frequency`).
* **`density/<axis>/`** — the recorded density vs s (`counts`,
  (S, bins) int32, and the bin `edges`), when the run recorded it;
  loaded as `density` / `density_edges` dicts.
* **`tail/`** — the fractional-emittance / radial-quantile series
  (`emit_x_q99`, …, attribute `fractions`), when recorded.
* **`particles/`** — full phase-space snapshots, when the recorder
  holds any.
* **`action_scan/`** — the halo action scan, when the run recorded one
  (`Simulation(record_action_scan=True)`, the Numerics-tab checkbox or
  `python -m linac_gen run <input> --mode mp --action-scan`; see
  [Halo analysis → Action scan](../09_diagnostics/03_halo.md#action-scan)):
  the grid `n` (G,), per step `n_alive`, `continuous` and `element`,
  and per plane `x`, `y`, `z`, `x_raw`, `y_raw` the counts
  `count_<p>` (S, G, int32, gzip-compressed — about 0.1 kB per step and
  plane, about 1 kB per step for the whole group), `eps_<p>`,
  `eps_n_<p>` and `n_max_<p>` (S,).  Attributes `schema_version`,
  `definition`, `eps_units`, `planes` and, when known,
  `periodic_phase`.  Rows align with `envelope/s`;
  `load_results_hdf5` returns the group as the nested dict
  `results["action_scan"]` (datasets by name, attributes under
  `"attrs"`; members a later version may add are ignored; a group that
  is incomplete, from a newer schema or not aligned with `s` is dropped
  with a warning, its reason kept in `results["action_scan_error"]` and
  shown by the Results-tab window, and the rest of the file loads).
  Saving loaded results writes the group back unchanged.  Files without
  the scan are unchanged.
* **`beam_config/`** — scalar `BeamConfig` values as HDF5 attributes,
  when `beam_config` is provided.
* **`provenance/`** — always present (honesty round, 2026-07): the
  `linac_gen` version, best-effort git commit, numpy/h5py versions and
  the write timestamp; plus — when the caller supplies the optional
  kwargs — the source deck path with its SHA-256 (`lattice_path=`),
  the beam RNG seed (`seed=`), and the space-charge configuration with
  the **effective** backend resolution, including any
  `LINAC_GEN_USE_GPU` environment override (`sc_config=`).  The
  `linac_gen run` / `backtrack` CLIs and the GUI auto-save pass what
  they have in scope automatically.  A results file now pins which
  code, which machine description and which numerics produced it.

Files written before 2026-09-27 lack the per-step series above (they
stored only the first 19): importing one leaves the tiles that need
them — normalised ε_nz, 6-D emittance, dispersion, longitudinal Twiss,
divergence, 4-D and eigen-emittances, IBS, magnetic stripping — empty.
Re-run to get a complete file.  The openPMD companion file
(`*.opmd.h5`) carries the same per-step `envelope/` series, the loss
record and the halo action scan in its HELIX extension groups (the
density and tail series are written to the native file only).  Large
datasets (the beam matrices, density histograms) are gzip-compressed.

Since the 2026-07 completeness round the `provenance/` group also
records what `lattice_sha256` does **not** cover: SHA-256 hashes of the
resolved **field-map data files** and (via `input_beam_path=`) an
imported beam file, the **parser downgrade ledger** (`parse_downgrades`
— attached to the lattice by the CLI loader), the diagnostic cadence
(`integration_steps_per_metre` / `sc_steps_per_metre` /
`drift_single_push`), the effective
FP precision (`fp_dtype`), the contractual OpenMP `schedule(static)`
clause (`omp_schedule`), and a manifest of every **registered
surrogate** (element key, class, training seed, validation MAPE) that
name-scoped tracking lookup could have consulted.  The `lattice`
argument feeds these entries.

### TraceWin `.dst` (final beam)

```python
from linac_gen.io.tracewin_dst import write_dst

write_dst(
    "final.dst",
    beam.alive_particles,            # (N, 6) — filtering losses is your job
    current_mA=beam.current,
    frequency_MHz=beam.ref.frequency,
    mass_MeV=beam.ref.species.mass,
    w_kin_ref=beam.ref.w_kin,
)
```

`write_dst(path, particles, current_mA, frequency_MHz, mass_MeV,
w_kin_ref, phi_ref_rad=0.0)` takes a raw `(N, 6)` particle array in
HELIX units — it does **not** filter dead particles for you (pass
`beam.alive_particles`, not `beam.particles`).  See
[.dst I/O](../04_beam/04_dst_io.md) for the binary layout.

### openPMD (interchange format)

```python
from linac_gen.io.openpmd_output import save_results_openpmd
save_results_openpmd(results, "run.opmd.h5", beam_config=beam_cfg)

# tidy up the demo output files
from pathlib import Path
for f in ("run.h5", "run.opmd.h5", "final.dst"):
    Path(f).unlink(missing_ok=True)
```

Writes the run in the **openPMD-beamphysics 1.1** standard (HDF5):
per-particle phase-space data plus the envelope arrays.  Unlike the
HELIX-native HDF5 above, an `.opmd.h5` file can be read by any
openPMD-aware tool (openPMD-viewer, the openPMD-beamphysics package).
Use it to hand HELIX results to external analysis or cross-code
comparison.  From the GUI, use the toolbar's **Export openPMD
output…** action.

## Automatic dumps from the GUI

Every GUI run auto-saves its results — no export step needed.  After
each run, two files land in the **calculation directory** (set it via
**File → Set Calculation Directory…**):

* `<timestamp>_<runtype>.h5` — the HELIX-native HDF5 schema above.
* `<timestamp>_<runtype>.opmd.h5` — the openPMD companion (non-fatal
  if it fails; the native file is the source of truth).

## Cross-references

* [Recorder fields](../09_diagnostics/01_recorder.md) — every field
  with type and units.
* [Plotting examples](../11_examples/01_basic_fodo.md).

← [HELIX extensions](03_lg_extensions.md) ·
[Continue to From the GUI →](05_gui.md)
