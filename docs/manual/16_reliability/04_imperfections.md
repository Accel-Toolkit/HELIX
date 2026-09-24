# Leg C — imperfections and faults on error seeds

## The error budget

`error_budget.element` rows are the vocabulary of `ErrorStudy.add_error`:

```json
{"element": [{"pattern": "QUAD_*", "parameter": "dx", "sigma": 0.10, "kind": "Quadrupole"},
             {"pattern": "FMAP_*", "parameter": "voltage_rel", "sigma": 0.01, "kind": "cavity"},
             {"pattern": "FMAP_*", "parameter": "phase_offset", "sigma": 1.0, "kind": "cavity"}],
 "beam":    [{"parameter": "centroid_x", "sigma": 0.1}, {"parameter": "emit_nx_rel", "sigma": 0.05}]}
```

(`sigma` for Gaussian at `cutoff` 3σ, `half_width` for uniform; `kind` gates
a name glob by element type — mandatory on decks where solenoids and cavities
share the `FMAP_*` prefix).  The budget may also live in its own file next to
the spec (`"error_budget": "error_budget.json"`); it is inlined when the spec
is loaded, so the campaign folder is self-contained.  At creation the campaign runs the engine's audit
and **refuses** a budget with a selector that matches nothing, a warn-skipped
or inert parameter, an unknown distribution, or overlapping selectors — the
misalignment-study driver's policy, so a malformed budget never silently
draws nothing.

## Draws as overrides

Each seed is drawn once by the engine's own `_apply_errors` /
`_apply_beam_errors` (`ErrorStudy.draw_overrides`) and transported as
`@index.attr` overrides to a scan-pool run, so the seeds run in the process
pool instead of the serial run loop; the draws are saved in
`legs/imperfections/draws/seed_NNNN.json` and travel with an exported job.
With `correction.enabled` the orbit is corrected on the errored copy and the
steerer kicks join the overrides (the envelope reading backend by default, a
per-seed BPM-noise stream).  `correction.pairing` says how steerers and BPMs
are paired: `cards` (the default) goes through the deck's `ADJUST_STEERER`
cards (`run_correction_from_lattice`) — a deck without cards then corrects
nothing and the seed row says `none`; `auto` takes every magnetic `Steerer`
against every BPM marker (`auto_paired_correction`: one-to-one with the
nearest downstream BPM when the counts match, otherwise a global SVD over all
BPMs), the same `apply_correction` the card driver calls, so a deck whose
cards pair each steerer with its next BPM gives the same kicks either way;
without cards there is no per-steerer `vmax` clip.  A correction pass costs
tracking runs of the whole deck: the SVD measures one response column per
steerer and plane (two runs per steerer plus the reading), the one-to-one
method probes every pair on its own (five runs per pair), so on a long linac
`n_iter` is the knob to watch.  A draw that lands on an attribute the
design element does not carry (an alignment offset on a marker) is reported
as *untransportable* rather than dropped.

## Faults on seeds

The top `seeds.faults_on_seeds.top_n` fault cases are replayed on the first
`n_seeds` error seeds, *before* and *after* compensation (the compensator
settings found on the design lattice), and judged by the criticality rule
against **that seed's** own run; the robust fraction per case is the share of
seeds on which the fault stays non-critical — the answer to "does the
compensation found on paper survive the machine's imperfections".

## Outputs

`seeds.csv` (per seed: transmission, exit energy, emittances, sizes, loss
power, the correction's status, the orbit rms over every BPM before the
correction and after its last pass with the number of passes, number of
draws; the control row is the nominal lattice with the same beam), `faults_on_seeds.csv`, `robustness.csv`, and the
figures `seeds_transmission.png`, `seeds_emittance.png`,
`faults_on_seeds_robustness.png`.
