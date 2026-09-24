# Reliability studies — overview

A reliability study asks how a linac behaves when something breaks, how far
its optics tolerate the errors that are always present, what happens to the
beam at the stripper foil under those conditions, and how often the machine
delivers beam at all.  The Reliability Study mode runs those four legs as
**one resumable campaign** on the loaded lattice and beam, on the engines the
rest of HELIX already uses (the failure engine, the error engine, the scan
pool, the loss-power analysis, the foil element), and ends in a results folder
with per-leg CSV files, `summary.json` with the verdicts against the study
rule, PNG figures and one self-contained HTML report.

| Leg | Question | Engine | Page |
|---|---|---|---|
| A — fault tolerance | which single, paired and grouped failures push the beam past the criticality rule, and which of them a re-tune of the neighbours recovers | `linac_gen.failures` + the matcher | [Fault tolerance and compensation](02_fault_tolerance.md) |
| B — availability | how often the machine is down and for how long, from a reliability block diagram fed by leg A's recovered-by fractions | `linac_gen.reliability.availability` | [Availability and the beam-trip budget](03_availability.md) |
| C — imperfections | how the error budget (alignment, field, phase, input beam) moves the beam with orbit correction on, and whether the top faults stay recoverable on error seeds | `linac_gen.errors` | [Imperfections and faults on error seeds](04_imperfections.md) |
| D — foil | spot size, power density and stripping efficiency at the foil for its nominal, thinned, thick, missing and offset states | the `Foil` element | [Foil scenarios](05_foil.md) |

Every leg is a list of *items* (scan-pool runs, compensation jobs, seed draws,
one availability Monte Carlo) with a `status.json` of facts; the deltas,
rankings and verdicts are derived views rebuilt by `summarize`, so a campaign
can be stopped and resumed at any item and its report rebuilt at any time.  The
[results folder](06_results_and_report.md) page lists every file.

## Two RF brackets

HELIX re-solves `SET_SYNC_PHASE` cavities on every run and applies a thin
`GAP`'s deck phase as its synchronous phase, so a fault upstream is normally
tracked with the downstream RF **ideally re-phased** — the optimistic bracket.
A campaign also runs the pessimistic one, the machine **frozen as it was set
before the fault**: every field-map cavity keeps the ψ its design pass
calibrated and every thin gap the reference RF-clock phase it saw at its
entrance (`sync_phase_pin`, harvested once per engine), so the slower reference
of a faulted line shows up as a phase error at every cavity downstream.  The
clock pin is in degrees at the reference frequency: a thin gap whose frequency
differs from the reference clock's at its entrance (a per-element frequency
jump with no `FREQ` card) is refused by the harvest, because the two engines
switch the reference frequency at different points of such a jump.  The
two nominal runs are bit-identical (the pin *is* the design pass); the fault
cases differ — the frozen-phase deficit of an upstream cavity is usually
*smaller* than the re-phased one on a negative-phase linac, because a late
reference climbs the cosine towards the crest, until the slip overshoots it.

## Presets

`quick` (envelope model, single faults, steerers, the frozen bracket, five
error seeds, the analytic stripping curve — minutes on a linac deck) and
`full` (every scenario class, multiparticle verification of every critical
case and every recovered compensation, 200 seeds with 20 × 50 faults on
seeds, 2 000 availability trials, tracked foil scenarios — hundreds of
core-hours, meant for [job export](../06_running/14_cli_reliability.md#job-export-and-import)
to a compute machine).  `custom` keeps the quick shape and takes the classes
you list.

## Running one

```
PYTHONPATH=.:gui python -m linac_gen reliability plan examples/reliability_demo/reliability_demo.lgproj
PYTHONPATH=.:gui python -m linac_gen reliability run  examples/reliability_demo/reliability_demo.lgproj --parallel 4
```

The CLI page ([CLI: reliability](../06_running/14_cli_reliability.md)) has
every verb and option; the assistant tool `run_reliability` drives the same
engine.  The shipped demo deck (`examples/reliability_demo/`) is a 2.8 m proton
channel carrying every element type the legs touch, with a `truth.json` of
answers computed independently of HELIX — the [self-test](07_selftest.md)
compares against it.

## What a campaign needs from you

* a deck or `.lgproj` saved on disk (the campaign pins its SHA-256 and refuses
  to run a changed one);
* a **circuit map** (`circuits.json` next to the deck, or `--circuits`): which
  cavities share a cryomodule or RF station, which magnets share a circuit,
  which field maps are unpowered spares, where the sections end, and the
  treaty-point and foil landmarks — a minimal map is derived from the deck
  when you give none;
* an **error budget** for leg C (the same rows `ErrorStudy.add_error` takes),
  inline in the spec or in a JSON file the spec names (`"error_budget":
  "error_budget.json"`, relative to the spec — inlined when the spec is
  loaded, so `campaign.json` stays self-contained);
* a **block table** for leg B (MTBF / MTTR per unit — copied into the campaign
  folder when it is created, so a job or a moved campaign never needs the
  original path; without one the campaign starts from a surrogate template it
  labels as such);
* the physics conventions of the study rule: 5 % normalised-emittance growth,
  1e-4 loss fraction, 0.5 % exit-energy deviation and the W/m limit per section
  (defaults from the PIP-II study; change them in the spec — a threshold set to
  `null` switches its term off, see [the two rule sets](02_fault_tolerance.md#two-rule-sets-and-the-pip-ii-reconciliation)).
