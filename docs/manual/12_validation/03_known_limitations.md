# Known limitations & deferred features

A complete list of what HELIX deliberately *doesn't* do, what's
parsed-but-not-honoured, and what's on the roadmap.  Useful for
deciding whether HELIX is the right tool for a given project.

## Out of scope (deferred)

### Time-varying / dynamic errors

TraceWin's `ERROR_*_DYN` directives apply per-step random jitter
inside the tracker hot loop.  The dynamic semantics are not
implemented — note the parser does **not** skip these cards: it
absorbs them as ordinary *static* NCPL errors (one draw per seed).

* **Why deferred**: adds RNG calls at every integration step
  (significant performance hit) and physics complication.
* **Workaround**: static per-seed errors cover ~95 % of real
  workflows.  Multi-seed ensembles approximate the dynamic case.

### Coupled (CPL) error groups

`ERROR_*_CPL_*` directives apply the *same* random draw to a group
of elements (e.g. a triplet sharing a power supply).  The coupled
semantics are not implemented — as with `_DYN`, the parser absorbs
these cards as *uncoupled* static errors (independent draw per
element) rather than skipping them.

* **Why deferred**: needs a "group pattern" extension to ErrorDef
  that wasn't in the initial scope.
* **Workaround**: implement the group draw manually in Python.

### Element-error slots without tracking effect: dz, pitch, yaw

`dz` (longitudinal alignment) and φx/φy (pitch/yaw) draws are stored
on the element, but the tracker only honours `dx`, `dy` and
`tilt_deg`.  The parser now emits a warning when an `ERROR_*` card
declares a non-zero value for any of them, so a tolerance that has no
effect is at least visible.

(*Fixed 2026-07:* `voltage_rel` on `FieldMap`/`FieldMap3D` cavities
and `field_rel` on `Dipole` — formerly silent no-ops — are now
applied via the elements' own `_rel` slots.  Re-run pre-fix tolerance
studies that used cavity amplitude or bend-field errors.)

See [Element-level errors](../08_errors/03_element_errors.md).

### `SUPERPOSE_MAP` v1 scope

Straight-axis field-map superposition is implemented
([SuperposedFieldMap](../03_elements/19_superposedfieldmap.md),
2026-07).  Three deliberate v1 restrictions remain:

* **`SUPERPOSE_MAP_OUT`** (dipole maps that curve the reference
  trajectory, with a user-specified exit frame) is refused at parse —
  the deviated-reference bookkeeping is separate work.  Without it,
  TraceWin itself ignores the X/Y/θ operands of `SUPERPOSE_MAP`, so
  HELIX ignoring them is faithful.
* **Mixed RF frequencies** in one cluster are refused at construction
  (a single `ref.phi_s` clock exists); permissive parses fall back to
  the legacy end-to-end layout with a warning.
* **Misalignment and `ERROR_CAV` apply cluster-wide**, not per child
  card as TraceWin does (documented divergence; the parser warns when
  alignment errors land on a container).  `ADJUST` cards targeting a
  container are warn-skipped.

### Written `.dat` field-map paths are absolute

`write_tracewin` records field-map file locations as **absolute
paths**, so a saved `.dat` that references `FIELD_MAP` files is not
portable to another machine or directory tree — re-point the field
map folder (or edit the paths) after moving a project.

### Periodic-Twiss helpers omit dispersion

`find_periodic_twiss` and `find_matched_input_twiss` solve the
4-D/6-D betatron problem only — they do **not** carry dispersion.
For lattices with bends (BTL arcs, HEBT) use
`find_sc_matched_input_twiss`, which matches dispersion through an
8-state formulation.  See
[Matching → Python API](../07_matching/03_python_api.md).

### Envelope mode: apertures and MIN_TRANSMISSION are inert

Envelope (RMS) tracking propagates Σ only — no particles means no
aperture cuts, so transmission is implicitly 100 % and the
`MIN_TRANSMISSION` matching constraint is silently inert (a one-time
stderr warning is printed).  Use `cost_solver="mp"` for
loss-sensitive matching, and multi-particle runs for any loss
prediction.  See
[SET / ADJUST → MIN_TRANSMISSION](../07_matching/02_set_adjust.md).

### `ADJUST_STEERER` auto-correction (now shipped)

Implemented as of 2026-05-09.  TraceWin's closed-orbit auto-
correction loop is now driven by
:func:`linac_gen.errors.correction.run_correction_from_lattice`,
which scans the lattice for ``ADJUST_STEERER`` /
``ADJUST_STEERER_BX`` / ``ADJUST_STEERER_BY`` cards and pairs each
with a ``DIAG_POSITION`` (BPM) marker via the ``diag_n`` index.

Picks one-to-one when the cardinality is 1:1 and pairing is
clean, otherwise SVD with truncated pseudoinverse.  Honours
``vmax`` per card as a per-steerer T·m clip.  See
[Errors → Orbit correction](../08_errors/07_correction.md).

**vmax-units convention**: the ``max`` field on
``ADJUST_STEERER`` is interpreted as an integrated kick in T·m
on the partner ``Steerer.bx_l`` / ``Steerer.by_l``, since HELIX
steerers are zero-length thin kicks.  Magnetic-steerer ``Bmax``
in the ``STEERER`` card itself uses the same convention.

### `ERROR_RFQ_CEL_NCPL_STAT`

Per-cell RFQ electrode displacements.  Niche; deferred.

### Wakefields

* **Longitudinal wake** — bunch-induced longitudinal field.
* **Transverse wake** — bunch-induced kicks in subsequent bunches.
* **Resistive-wall wake** — image-charge / pipe-current effects.

None modelled.  For a proton/H⁻ linac they're typically a small
correction; for high-power accumulators they're critical and
HELIX is not the right tool.

### Image-charge / pipe-current effects

Open-boundary PIC; the beam pipe is just an aperture cut, not a
boundary condition for the Poisson solve.

### Synchrotron radiation damping

Irrelevant for the proton-linac use case (proton SR is negligible
below ~100 GeV).  Relevant for high-energy electron storage rings —
HELIX is not the right tool.

### Coupling beyond linear

Transverse-longitudinal coupling beyond first-order
dispersion / phase-slip mixing.  Higher-order coupling (e.g.
chromatic-dispersion in a strong dipole) is not modelled.

### Acceleration in Sacherer ODE

The Sacherer continuous-beam envelope solver assumes constant β,
γ.  RF cavities aren't supported in that path — for accelerated
beams use the matrix tracker plus bunched PIC SC.

### Pitch / yaw misalignment

`Misalignment` mixin reserves `pitch_deg` and `yaw_deg` slots, but
the tracker only honours `tilt_deg` (rotation about z).  Pitch /
yaw are non-symplectic in 4-D tracking and need the longitudinal
coordinate to absorb the path-length change — separate work.

## Parsed-but-not-honoured

| Field | Status |
|---|---|
| `Quadrupole.gfr` | parsed, no effect (TraceWin: gradient-fall-region) |
| `pitch_deg`, `yaw_deg` | reserved on every element; only `tilt_deg` is honoured |
| `geom = 8` (3-D cylindrical field map) | parsed, but TraceWin manual says "not implemented" |

### PIC results are bit-reproducible only at a fixed thread count

The C++ deposit kernel accumulates per-thread buffers and reduces
them in a fixed order, so runs are bit-identical **at the same
`OMP_NUM_THREADS`** — but changing the thread count regroups the
floating-point sums and shifts results at the ~1e-10 level.  HELIX
pins `OMP_DYNAMIC=FALSE` at import (so the count cannot drift within
a session), makes the particle→thread partition contractual with an
explicit `schedule(static)` clause on the kernel loops (2026-07 — it
previously rested on the implementation-default schedule), and records
`omp_num_threads` and `omp_schedule` in the HDF5 `provenance/` group.
For bit-reproducible studies across machines, set `OMP_NUM_THREADS`
explicitly.

### Backtracking through CSR is approximate (opt-in)

CSR kicks have no backward model.  `run_backtrack()` on a forward run
with `csr_enabled=True` **refuses** unless you pass
`approximate_backtracking=True`, which skips the CSR kicks on the
backward walk and warns that the reconstruction is approximate.  The
same contract applies to `sc_backend="halo"` (2026-07): the learned
defect corrector has no backward model, so the backward walk refuses
unless `approximate_backtracking=True` accepts the plain-PIC undo.

### Release-hardening tasks (accepted as such in review, 2026-07)

Two deliberate deferrals, neither a correctness defect in current
behavior:

* **Surrogate fingerprint uses the resolved field-file *path***, not
  a content hash — an in-place edit of a field-map file (same path,
  same length, same drive parameters) would not be caught by the
  cross-lattice engagement guard.  The planned hardening replaces the
  path with the SHA-256 already computed for HDF5 provenance.
* **Backtracking returns no machine-readable status enum** — the
  exact/linearized/approximate/non-invertible distinctions are all
  enforced by refusals and warnings (and the CLI prints a
  human-readable forward-closure check), but callers cannot yet
  branch on which reversal mode actually ran.  The planned ledger:
  `exact_discrete_replay` / `linearized_inverse` /
  `approximate_collective_replay` / `noninvertible` plus a
  per-element fallback list.

## Calibration biases (not bugs, known offsets)

* **PIC kernel** — ~3 % σ overshoot vs TraceWin partran on PIP-II.
  See PIP-II validation.
* **Longitudinal σ_z** — ~22 % overshoot in HELIX vs TW.  Cause not
  localised.

## Roadmap

Items being actively considered (no commitments):

* Dynamic errors (per-step RNG).
* Wakefield infrastructure (longitudinal first).
* Per-step density / time-domain recording for sub-element diagnostics.
* Pitch / yaw misalignment (needs 6-D rather than 4-D tracker
  tweaks).
* Higher-order (3-D) field-map coupling.
* PDF rendering of this manual.

For requests / status updates see GitHub issues.

## Cross-references

* [TraceWin parity](01_tracewin_parity.md)
* PIP-II validation
* [Errors directives](../08_errors/02_error_directives.md) — what's parsed.

← PIP-II validation ·
[Continue to Convergence checklist →](04_convergence_guide.md)
