# CLI: reliability

`python -m linac_gen reliability` runs the Reliability Study mode headless: the
four legs of a reliability study — fault tolerance with compensation,
imperfections with faults on error seeds, foil scenarios and availability — as
one resumable, folder-backed campaign that ends in per-leg CSV files, a
`summary.json` with the verdicts against the study rule, PNG figures and one
self-contained HTML report.  The same engine drives the GUI window
(*Tools → Reliability Study…*) and the assistant tool `run_reliability`; the
chapter [Reliability studies](../16_reliability/01_overview.md) explains what
each leg computes.

```
python -m linac_gen reliability VERB [TARGET] [options]
```

| Verb | Target | What it does |
|---|---|---|
| `plan` | deck / `.lgproj` (or a campaign dir) | print the static plan — items per leg — for a preset; `--write-spec FILE` saves the filled spec; nothing runs |
| `run` | deck / `.lgproj`, campaign dir or job dir | create-if-needed, then execute every pending item; **resume-by-default** |
| `resume` | campaign dir or job dir | alias of `run` for an existing campaign |
| `summarize` / `report` | campaign dir | rebuild the CSVs, `summary.json`, figures and `report.html` from the per-item facts |
| `export` | campaign dir, `--to DIR` | write a job folder for another machine (`--legs`, `--bundle-fields`) |
| `import` | job dir, `--into CAMPAIGN` | bring a job folder's results back (`--allow-partial`) |
| `selftest` | — | the built-in self-test on the shipped demo deck (`--quick`, `--regression --before DIR`, `--json`) |

Options: `--preset quick|full|custom` (a new campaign), `--spec campaign.json`
(start from a saved spec), `--name`, `--circuits circuits.json` (default: the
one next to the deck), `--dir` (default `<input dir>/reliability/<name>/`),
`--legs faults,imperfections,foil,availability`, `--parallel N`, `--serial`,
`--force` (wipe `legs/`), `--retry-failed`, `-q`, and `--tracewin-ini` (beam 1 of
the deck's TraceWin options file for a bare deck; a project refuses it).

Exit codes: 0 success / self-test PASS, 1 execution failure / self-test FAIL,
2 bad input.

## A first campaign

```
PYTHONPATH=.:gui python -m linac_gen reliability plan examples/reliability_demo/reliability_demo.lgproj
PYTHONPATH=.:gui python -m linac_gen reliability run  examples/reliability_demo/reliability_demo.lgproj --parallel 4
```

`plan` prints the items per leg (the quick preset on the demo deck: 2
baselines + 14 fault cases, 3 seed draws + 3 seed runs, 7 foil scenarios, 1
availability Monte Carlo) and notes that the compensation, MP-verification and
faults-on-seeds waves are planned from the fault ranking at run time.  `run`
creates `examples/reliability_demo/reliability/reliability_quick/` and
executes; run it again and completed items are skipped.  Open `report.html`
when it finishes.

## The campaign folder

```
<dir>/campaign.json   circuits.json   pins.json   lattice/<deck copy>
  plan/manifest.json  plan/manifest_comp.json  plan/manifest_mp.json  plan/manifest_fos.json
  legs/faults/runs/<id>/{results.h5,status.json}   legs/faults/compensation/<case>_<strategy>.json
  legs/faults/{faults.csv,compensation.csv,criticality_map.csv,unrecoverable.json}
  legs/imperfections/{draws/seed_NNNN.json, runs/…, seeds.csv, faults_on_seeds.csv, robustness.csv}
  legs/foil/{runs/…, foil.csv}
  legs/availability/{blocks.csv, fault_classes.json, availability.json, availability.csv, trip_histogram.csv, sensitivity.csv}
  figures/*.png   summary.json   report.html
```

`campaign.json` is the spec: its `lattice_sha256` pins the deck (for a
`.lgproj`, the deck it points to as well), `spec_sha256` the spec itself and
`circuits_sha256` the circuit map; `run` refuses a campaign whose deck, spec
or circuit map changed.  `--force` wipes only the legs named by `--legs` (all
of them without it) and keeps `legs/availability/blocks.csv`; `--retry-failed`
re-plans the compensation, verification and faults-on-seeds waves when a fault
scenario is re-queued, since they were planned from the ranking.  Every scan-pool item's
`status.json` holds the worker row (`metrics`) and the per-run observables
(`extras`: exit clock, treaty-point energy and phase, loss power per section,
foil-plane spot and stripping populations); deltas, criticality and rankings
are derived when the summary is rebuilt, so `summarize` can be re-run at any
time.

## Presets

| | quick | full |
|---|---|---|
| forward model | envelope | envelope, then multiparticle verification of every critical case (+ the top 30 non-critical) and of every recovered compensation |
| scenario classes | S1 single cavity off, S2 single magnet off, S5 steerer off, S10 = S1 with the RF phases frozen | S1–S10 (adds detuned cavities, partial magnets, adjacent pairs, cryomodule / RF-station / magnet-circuit trips) |
| compensation | top 5 critical cases, k-out-of-n neighbours | top 20, plus spares and section re-match |
| error seeds | 5, faults on seeds 3 cases × 2 seeds | 200, faults on seeds 20 × 50 |
| availability | 200 trials | 2 000 trials |
| foil | envelope spot + the analytic stripping curve | tracked scenarios with snapshots at the foil |

`custom` keeps the quick shape and takes the classes you list.  `plan
--write-spec` writes the filled spec; edit it and start from it with `run --spec`.
Three spec details worth knowing: a threshold of `criticality.rule` set to
`null` switches that term off (the [two rule sets](../16_reliability/02_fault_tolerance.md#two-rule-sets-and-the-pip-ii-reconciliation));
`error_budget` may name a JSON file next to the spec instead of holding the
rows (inlined on load); and `correction.pairing` = `cards` (the deck's
`ADJUST_STEERER` cards) or `auto` (every steerer against every BPM — see
[Leg C](../16_reliability/04_imperfections.md#draws-as-overrides)).

## Job export and import

The full preset on a real linac is hundreds of core-hours: `export` writes a
self-contained job folder (spec, manifests, pins, circuits, error draws, the
items of the exported legs already completed so the remote resumes rather than
re-runs, a copy of the deck at the same relative depth, optionally the field
maps) with a `README.txt` holding the exact command.  On the compute machine
the job folder IS a campaign — `reliability run <job> --parallel N` runs the
job's own legs unless `--legs` says otherwise — and `import <job> --into
<campaign>` checks the deck and spec hashes and every item's `status.json`,
`results.h5` and provenance before copying anything (a refused import copies
nothing): everything the job's own plan lists for its legs is brought back,
including the compensation, verification and faults-on-seeds waves it planned
while running; an item the campaign already completed is never overwritten,
and a job with missing items is refused unless `--allow-partial`.  The
[Reliability Study window](../10_gui/10_reliability_dialog.md) has the same
two steps as **Export job…** and **Import results…**.

## Self-test

`python -m linac_gen reliability selftest` runs every leg on the shipped demo
deck (`examples/reliability_demo/`) and compares against `truth.json`, whose
answers come from the generator's own kinematics; `--regression --before DIR`
additionally re-runs the public control cases of `scripts/physics_fix_report.py`
and compares them bit for bit with a snapshot of the reference tree.  See
[Verifying an installation](../16_reliability/07_selftest.md).
