# The results folder and the report

```
<dir>/campaign.json          the spec, with lattice_sha256 (deck; for a .lgproj the deck it points to) and spec_sha256
      circuits.json          the circuit map the campaign used
      pins.json              the frozen-phase pins per engine (envelope, mp)
      lattice/<deck copy>    provenance only — runs use the pinned original
      plan/manifest.json     the static plan; manifest_comp / manifest_mp / manifest_fos the waves planned at run time
      legs/faults/runs/<id>/{results.h5,status.json}
      legs/faults/compensation/<case>_<strategy>.json
      legs/faults/{faults.csv, compensation.csv, criticality_map.csv, unrecoverable.json}
      legs/imperfections/{draws/seed_NNNN.json, runs/…, seeds.csv, faults_on_seeds.csv, robustness.csv}
      legs/foil/{runs/…, foil.csv}
      legs/availability/{blocks.csv, fault_classes.json, availability.json, availability.csv, trip_histogram.csv, sensitivity.csv}
      figures/*.png
      summary.json
      report.html
      import.json            written by a job import
```

Item ids read `A_<case>_<bracket>_<env|mp>` (fault leg), `A_comp_<case>_<strategy>`,
`C_draw_NNNN`, `C_seed_NNNN_<model>`, `C_fos_<case>_seed_NNNN_<before|after>_<model>`,
`D_<scenario>_<model>`, `B_availability`.

`status.json` of a scan-pool item: `status` (`ok` / `failed`), `finished`,
`elapsed`, `error`, the `item` (id, leg, case, class, bracket, label,
elements, model, seed), the `overrides` applied, `metrics` (the worker row:
sizes, emittances, transmission, exit energy, loss power) and `extras` (the
exit RF clock, the treaty-point energy, phase, sizes and Twiss, loss power
per section, the foil spot, power density and stripping populations).  A
`.part` results file is an orphan of an interrupted run and is swept on
resume.

`summary.json` carries the provenance (deck path and digest, spec digest,
preset, when), the item counts per leg, the rule, the fault leg's verdicts
(cases, critical, by class, recovered-by counts, the unrecoverable list, the
top ten), the compensation tally, the imperfection statistics and robustness
rows, the foil rows, and the availability per variant.

`report.html` is one file with the figures embedded as data URIs and only the
HTML subset a Qt text view renders (no scripts, no CSS grid): provenance, the
rule, every leg's verdicts and first rows with the CSV named for the rest.
`summarize` rebuilds all of it from the per-item facts at any time.
