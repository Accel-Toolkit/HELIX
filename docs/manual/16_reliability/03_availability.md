# Leg B — availability and the beam-trip budget

Leg B is a Monte Carlo over a **reliability block diagram**: the machine is a
series of repairable units, each with an MTBF, an MTTR distribution
(exponential, lognormal or fixed), optionally `n_parallel` identical units of
which `k_required` must be up, and a *fault class* that says how an event
ends.  Every unit is an alternating renewal process (a unit under repair cannot
fail again until it is back — the model behind the closed form
`MTBF / (MTBF + MTTR)`, which the Monte Carlo is pinned against).

## Inputs

`blocks.csv` (`availability.blocks` in the spec, relative to the spec file;
the campaign copies it into `legs/availability/blocks.csv` when it is created
and reads that copy from then on — a job exported to another machine carries
it):

```
name,parent,mtbf_h,mttr_h,mttr_dist,n_parallel,k_required,fault_class,scenario_class
RF_STATION,RF,2000,4,exponential,1,1,downtime,S8
SRF_CAVITY_TRIP,RF,60,0.003,fixed,1,1,auto_rephase,S1
ION_SOURCE,FRONT_END,500,1,exponential,2,1,downtime,
```

`scenario_class` names the fault-leg class (S1 … S10) whose recovered-by
fractions route the block's events (see below); blank = every event ends the
way the block's own `fault_class` says.

Without a table the campaign writes a **surrogate template** whose values sit
in the range of published linac component figures and are labelled as
placeholders — replace them with the project's MTBF/MTTR data before reading
anything into the numbers.

Fault classes (`availability.fault_classes`, on top of the defaults):
`downtime` (the block's own repair time), `auto_rephase` (a fixed 10 s — the
RF re-phases itself), `operator_retune` (lognormal, 10 min mean),
`degraded` (the beam continues at reduced performance: the event is counted,
never unavailable).  A class may carry its own recovery distribution:
`{"dist": "fixed" | "exponential" | "lognormal" | "empirical", "value_s" |
"mean_s", "sigma_ln", "samples_s"}` — `empirical` draws from a list of
measured recovery times.

The **class split** routes a block's events over classes: the campaign builds
it from leg A's recovered-by fractions per scenario class (non-critical →
`auto_rephase`, recovered → `operator_retune`, every strategy failed →
`downtime`; a critical case that was never attempted counts as unknown and
enters no fraction) and applies it to every block whose `scenario_class`
names that class, so the trip budget reflects what the beam dynamics said
about each fault.  `availability.json` records the split by class, the split
as applied per block, and the blocks that received none; a block table
without `scenario_class` values runs with its own `fault_class` only.

`availability.variants` names the SRF-trip rate variants run side by side —
`srf_fdr` (the design MTBF of an SRF trip) and `srf_sns` (an operating-machine
rate) by default — each as an extra `auto_rephase` block.

## Outputs

Every unit starts in its steady state (up with probability `MTBF/(MTBF+MTTR)`,
otherwise inside a residual outage), so a finite horizon carries no
everything-starts-up bias.

`availability.json` (per variant: availability mean / p05 / p95 and the
closed-form value, trips per year by class, downtime hours and event counts
per block, the trip-duration histogram in ESS-style bins — ≤ 10 s, 1 min,
5 min, 20 min, 1 h, 4 h, longer — and the analytic sensitivities `dA` per
e-fold of every block's MTBF and MTTR), the CSVs `availability.csv`,
`trip_histogram.csv`, `sensitivity.csv`, and the figures
`availability_hist.png` and `trip_budget.png`.

The closed forms `linac_gen.reliability.availability.analytic_availability`
(series product with the k-of-n binomial) are what the self-test and the unit
tests pin the Monte Carlo against.
