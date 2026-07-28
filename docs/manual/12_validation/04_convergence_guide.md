# Convergence checklist

A printable one-pager: run through this list before reporting any
HELIX result as "converged".

## Pre-flight checks

- [ ] **`continuous=True`** for any pre-RFQ (DC) beam.
- [ ] **`Ki=1`** on any LEBT FIELD_MAP that has SC compensation.
- [ ] **`emit_nx`, `emit_ny`** are *normalised* (β·γ·ε), not geometric.
- [ ] **`emit_z`** is in deg·MeV (TraceWin native).
- [ ] **Lattice path** in `.lgproj` resolves on the current system
  (or use the relative-path fallback in `parse_tracewin`).

## Particle count

- [ ] Run with `n_particles ∈ {2k, 5k, 10k, 20k}`; verify σ_x at
  end of lattice plateaus by 10k.

## PIC grid

- [ ] Grid `nx=ny=nz` ≥ 64 for tests, ≥ 96 for production.
- [ ] `grid_extent ≥ 5σ`.  Reduce to 3σ only if the beam is
  pinched and you've verified tail truncation is safe.
- [ ] `kernel="cic"` for RF-frequent lattices, `"tsc"` for long
  no-RF transports (BTL).
- [ ] `green_kind="igf"` always.

## Step density

- [ ] `step1_per_m` ≥ 50 for drift / quad sections, ≥ 100 for
  cavity sections.
- [ ] For RFQ sections, `n_steps` per cell auto-picked or ≥ 20.

## SC convergence

- [ ] Run with `nx ∈ {32, 48, 64, 96}` and verify σ_x at end
  plateaus by 96.

## Validation

- [ ] **TW parity** — for production lattices, compare σ_x, σ_y end
  values to TraceWin partran.  Expect ~3 % residual (PIC kernel
  calibration).
* > 5 % residual ⇒ check convergence again, or check for parameter
    mismatch (Ki, continuous, ε normalisation).

## Cross-references

* [Convergence guide → models](../05_space_charge/05_convergence.md)
* [Reading results](../06_running/04_results.md)
* [TraceWin parity](01_tracewin_parity.md)

← [Known limitations](03_known_limitations.md) ·
[Continue to Surrogates → Overview →](../13_surrogates/01_overview.md)
