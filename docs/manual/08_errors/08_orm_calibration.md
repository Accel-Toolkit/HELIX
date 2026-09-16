# Orbit-response calibration (LOCO for a linac)

An orbit-response matrix (ORM) is the machine's answer to the question
"how much does BPM *i* move when trim *j* is driven by one ampere?".  Measured
against the model's answer it is the most direct test of the transverse
optics a linac offers: every quadrupole between a trim and a BPM leaves its
signature in the response, so a set of measured columns constrains the
gradients, the trim calibrations and the BPM gains at once.  HELIX reads a
measured matrix, maps its devices onto the deck, builds the model response,
compares the two, fits the calibration factors (the linac version of the
LOCO method of storage rings) and writes a recalibrated deck.  The same
engine is available from the batch CLI ([CLI: `orm`](../06_running/13_cli_orm.md)),
the GUI (**Tools → Orbit-Response Calibration (LOCO)…**) and the assistant
tools `orm_calibrate` / `orm_apply`.

## TL;DR

| | |
|---|---|
| Measured input | FORMA scan folders (one driven plane each) or HELIX ORM CSV, mm per ampere |
| Device names | Fermilab rules built in (`L:DnnBPH` → `DnnBPM`, `L:DnnTMH` → `DnnT`, tank BPMs listed as unmatched); mapping editable |
| Model | envelope phase-probe maps, unit kick at the trim exit chained to every downstream BPM, mm per T·m |
| Compare | per-trim calibration *k* (T·m/A, signed), correlation, residual, noise floor, polarity flips |
| Fit | quad scale factors *g*, trim calibrations *k*, optional BPM gains *G*; Levenberg–Marquardt with Gaussian priors |
| Apply | `Quadrupole.gradient_rel` (undoable in the GUI) or baked gradients; calibration JSON; recalibrated deck by text surgery |

## Measured data

**FORMA folders.**  A FORMA scan drives the trims of one plane (H or V) one
after the other and records both BPM planes.  HELIX reads the folder as
written by FORMA v8:

```
20260826_133454/
  response_matrix_20260826_133454_horizontal.csv          rows: BPM devices, columns: trim devices, mm/A
  response_matrix_20260826_133454_horizontal_error.csv    1σ of the same shape (optional)
  response_matrix_20260826_133454_vertical.csv
  response_matrix_20260826_133454_vertical_error.csv
  response_matrix_20260826_133454_corrector_gains.csv     stored and displayed, not applied
  scan_setup.json  scan_info.json  baseline_rms.json      metadata; dead BPMs are honoured
```

The driven plane is inferred from the trim names (`…TMH` / `…TMV`); an export
without that convention declares it with `"kick_plane": "x"|"y"` in
`scan_info.json`.  Value and error files must list the same devices in the
same order (a mismatch is refused, never re-aligned).  Load the H and the V
folder together to get both in-plane blocks.

**HELIX ORM CSV.**  One block per file, `# helix_orm_csv 1` header with
`# kick_plane`, `# read_plane`, `# units`, then `bpm,<trim>,…` rows; an
`_error` companion carries the σ.  `write_orm_csv` writes it, the CLI's
`--out` folder contains the aligned measured and model blocks in this form.

## Device mapping

Measured devices are matched to the deck by **label first, then name**.
Labels are the `NAME:` prefixes of the deck cards (kept by the parser on every
element, see [TraceWin decks](../06_running/02_tracewin_dat.md)),
names are HELIX's generated ids (`QUAD_012`, `STEER_003`).  The built-in
rules cover the Fermilab SCL (`L:D21BPH` → `D21BPM`, `L:D21TMV` → `D21T`,
`L:BPH5OT` → `D01BPM`); devices no rule covers, and devices whose element
does not exist in the deck, are listed as **unmatched** — never dropped
silently — and the deck's BPMs/trims without data are listed too.  A label
carried by two elements at the same *s* (the SCL's split `D01T` steerer)
resolves to the first with a note; at different *s* it stays unresolved.

The mapping is a JSON document (`__kind__: helix_orm_device_map`) with
`bpms`, `trims`, `bpm_sign` (−1 for an inverted BPM), `exclude_bpms`,
`exclude_trims`; `--write-map` / the GUI's *Export map JSON…* save the
resolved map for editing, `--map` / *Import map JSON…* load it back.  The
GUI's Mapping tab edits the same fields in place (element, sign, include).

## The model response

One envelope phase probe at zero current gives the bare per-element
transfer maps; the response of BPM *i* to trim *j* is the unit kick at the
trim exit chained through those maps, converted from mm per mrad to mm per
T·m with the steerer's own convention (`Δx' = sign(q)·B_y·L/Bρ`, `Δy' =
sign(q)·B_x·L/Bρ`; H⁻ flips the sign) at the trim's energy.  The centroid
does not feel space charge, so the map model is exact for the centroid;
`--check-tracking` (CLI) or *cross-check by tracking* (GUI) recomputes the
same matrix by actually kicking the envelope run and reading the BPMs
(`tracked_response`) and reports the largest deviation — 1e-11 on the SCL.
BPMs upstream of a trim carry no signal; their rms is the **noise floor** of
the measurement.

## Comparison

Per trim column (downstream BPMs only, both planes handled separately):

* **k** — weighted least-squares calibration *measured = k · model* in T·m
  per ampere, signed, with its error; the kick per ampere is |k| / Bρ.
* **r** — correlation of the measured and model patterns (needs ≥ 3 BPMs).
* **residual** — rms of *measured − k·model* over the signal rms.
* **polarity flipped** — trims whose *k* has the opposite sign of the
  plane's global *k*.

The weights are σ_eff² = σ² + (floor · column max)², `--sys-floor` 0.05 by
default: measured error bars are usually far smaller than the systematic
model error, and without the floor a few precise points dominate.  The
global *k* of a plane and the pooled correlation summarise the plane; the
per-trim table is where individual hardware faults show up (an inverted
BPM, a wrong cable, a trim whose calibration is off).

## The fit

Parameters: *g* (one scale factor per quadrupole between the first trim
and the last BPM — `effective_gradient = gradient · (1 + gradient_rel)`,
so the fit sets `gradient_rel = (1 + rel₀)·g − 1`), *k* (one signed
calibration per trim and plane), and optionally *G* (one gain per BPM and
plane).  Residuals are `(measured − G·k·M(g)) / σ_eff` over the included
downstream entries of both in-plane blocks, plus prior rows `(g − 1)/σ_g`
and `(G − 1)/σ_G` (`--prior-g` 0.10, `--prior-G` 0.05).  The solver is
Levenberg–Marquardt with λ damping; the Jacobian is analytic in *k* and
*G* and a forward difference in *g* (one probe per free quad), *g* is
bounded to [0.5, 1.5], the covariance is `pinv(JᵀJ)·χ²/dof`.

Stages: `trims` (closed form, the comparison's *k*), `trims+quads`
(default), `trims+quads+bpms`; `--stage all` chains them with warm starts.
**Degeneracies** are handled before the fit, not after: a quad with fewer
than two distinct fitted downstream BPMs is held at *g* = 1 with a reason
(the SCL's last quad Q73 sees one BPM), `--fix-quad` holds any other, an
excluded BPM keeps *G* = 1, a trim without downstream data gets *k* = NaN.
Quads seen by a single trim column (the first cell of a linac) are only
weakly determined — their errors say so; the prior keeps them near 1.

`validate` (CLI), *Synthetic validation…* (GUI) or `mode="validate"`
(assistant) injects random quad errors, trim calibrations and BPM gains
into the model, adds measured-level noise, refits and reports the
recovery — the honest precision statement for a given lattice and BPM
layout.

## Apply, save, export

* **Apply** sets `gradient_rel` on the live lattice (GUI: one undoable
  command, plain Save then reroutes to Save-As; assistant: through
  `ctx.apply_param_changes`; Python: `apply_changes(apply_calibration(...))`
  and `revert_changes`).  `mode="bake"` writes the fitted gradients into
  `gradient` instead.  The design gradients are checked first (rtol 1e-9)
  and the apply is refused atomically if the deck is not the fitted one.
  Note that an `ErrorStudy` overwrites `gradient_rel` per seed and the deck
  writer never serialises it — bake or export before either.
* **Calibration JSON** (`__kind__: helix_orm_calibration`, version 1):
  created, HELIX version and commit, lattice path + sha256, beam, measured
  provenance, device map, selection notes, fit options and χ² history,
  `quads` (label, name, index, design gradient, scale ± err, gradient_rel,
  fixed + reason), `trims` (k per plane ± err, kick per ampere, Bρ, energy),
  `bpms` (gains), metrics before/after, unit and sign conventions.
* **Recalibrated deck**: the original `.dat` text with the fitted quads'
  gradient tokens rescaled (`Q13: QUAD 100 -12.3 …  ; ORM fit 2026-09-14:
  x0.87700 ± 0.0020 (was -14.03)`), a provenance header, byte-exact
  everything else (latin-1, line endings, comments).  Quads are found by
  label; an unlabelled deck falls back to the k-th `QUAD` card.  The token
  must equal the calibration's design gradient, and the written deck is
  re-parsed and checked against the fitted lattice before the call returns.
  `--lgproj` writes a sibling project pointing at the new deck.

## Walk-through on the synthetic demo

`examples/orm_demo/make_synthetic_forma.py` writes a FORMA-layout folder
pair from the HELIX model of a six-cell labelled FODO with random quad scale
errors (3 % rms), random trim calibrations and 1 % noise, and records the
truth:

```{.python data-needs="examples/orm_demo/fodo_orm.lgproj examples/orm_demo/make_synthetic_forma.py"}
import importlib.util, json, tempfile
from pathlib import Path

spec = importlib.util.spec_from_file_location("synth", "examples/orm_demo/make_synthetic_forma.py")
synth = importlib.util.module_from_spec(spec); spec.loader.exec_module(synth)
work = Path(tempfile.mkdtemp()); out = work / "synthetic"
synth.main(["--out", str(out), "--seed", "1"])
truth = json.loads((out / "truth.json").read_text())
folders = [str(out / "20260101_000000_H"), str(out / "20260101_000000_V")]
```

The Python API mirrors the CLI modes:

```{.python data-needs="examples/orm_demo/fodo_orm.lgproj"}
from linac_gen.cli.common import load_input
from linac_gen.orm import (load_measured, default_device_map, resolve_devices, OrmModel,
                           compare_orm, summary_lines, fit_orm, OrmFitOptions,
                           calibration_from_fit, apply_calibration, apply_changes, revert_changes)

lat, cfg, _ = load_input("examples/orm_demo/fodo_orm.lgproj")
meas = load_measured(folders)                       # {"x": MeasuredOrm, "y": MeasuredOrm}
dm = default_device_map(lat, meas)                  # built-in rules; labels equal the device names here
sel = resolve_devices(lat, meas, dm)
print(sel.summary())
model = OrmModel(lat, cfg, sel)
R = model.response()                                # mm per T·m, four (kick, read) blocks
cmp = compare_orm(meas, R, sel, model.brho_trim, model.w_trim_MeV)
print("\n".join(summary_lines(cmp)))

fit = fit_orm(model, meas, OrmFitOptions(stage="trims+quads"))
print("\n".join(fit.summary_lines()))
cal = calibration_from_fit(fit, model, meas, lattice_path="examples/orm_demo/fodo_orm.dat",
                           beam_cfg=cfg, device_map=dm)
err = [q["scale"] - truth["quads"][q["label"]] for q in cal["quads"] if not q["fixed"]]
print(f"recovered {len(err)} quads, rms error {100 * (sum(e * e for e in err) / len(err)) ** 0.5:.2f} %")

changes = apply_calibration(lat, cal, mode="rel")   # nothing mutated yet
apply_changes(changes)                              # the quads now carry gradient_rel
revert_changes(changes)                             # …and are back to the design
```

The same from the shell, with figures and the recalibrated deck:

```{.bash .skip}
python -m linac_gen orm fit examples/orm_demo/fodo_orm.lgproj \
    --measured synthetic/20260101_000000_H --measured synthetic/20260101_000000_V \
    --out orm_out --plots --export-deck orm_out/fodo_orm_fit.dat --lgproj
python -m linac_gen orm validate examples/orm_demo/fodo_orm.lgproj --measured … --g-sigma 0.03
```

## A machine measurement (Fermilab SCL, FORMA scans of 26 August 2026)

The PIP-II SCL deck `examples/piplattice/fnalscl.lgproj` against the H and V
FORMA scans: 29 of 34 BPMs and 18 of 19 trims per plane matched (the five
tank BPMs `L:BP?2OT…5IN` have no deck element, `L:D73TM?` has no steerer
card), the split `D01T` steerer resolved with a note.  The comparison shows
median correlations of 0.86 (H) and 0.98 (V) with residuals of 44 % and
24 % of the signal after the per-trim calibration alone, an inverted BPM
(`L:D74BPH`) and an inverted trim winding (`L:D01TMV`).  The
`trims+quads` fit with the inverted BPM excluded and the single-BPM last
quad held (`--exclude-bpm L:D74BPH --exclude-bpm L:D74BPV --fix-quad Q73`)
brings the residuals to 4.2 % and 5.3 %; the largest correction is
Q13 −12.3 % ± 0.2 %, the rms correction 4.6 %.  A synthetic validation on the
same layout recovers 2.5 % injected errors to ≈ 0.6 % rms.  Reproduce with

```{.bash .skip}
python -m linac_gen orm fit examples/piplattice/fnalscl.lgproj \
    --measured 1_H_ORM/20260826_133454 --measured 2_V_ORM/20260826_135009 \
    --exclude-bpm L:D74BPH --exclude-bpm L:D74BPV --fix-quad Q73 \
    --out orm_out --plots --export-deck fnalscl_orm_fitted.dat --lgproj
```

(the scan folders are not distributed with HELIX).

## Adding a device the deck lacks

A measured trim without a deck element (the SCL's `L:D73TM?`) stays
unmatched until the deck has a steerer at its position: add
`D73T: THIN_STEERING 0 0 20 0` after the corresponding quad (the SCL's
D73 trim sits 74.3 mm downstream of Q73), reload, and the built-in rule
picks it up.  The same holds for BPMs (`D73BPM: DIAG_POSITION n`).

## GUI

**Tools → Orbit-Response Calibration (LOCO)…** opens a non-modal dialog
with four tabs: *Mapping* (the device table — element, sign and include are
editable; *Auto-map*, *Import/Export map JSON…*), *Compare* (*Compute model
and compare*, optional tracking cross-check; measured / model / difference
heatmaps, one panel per trim with error bars, the per-trim metrics table),
*Fit* (stage, priors, floor, iterations, quads to hold; progress and
*Cancel*; the quad table and bar chart, the residual per trim before and
after; *Synthetic validation…*) and *Apply / Export* (*Apply to lattice
(undoable)*, *bake* option, *Export recalibrated deck…*, *Save / Load
calibration JSON…*).  Workers run on snapshots; a result is discarded when
the lattice changed while it was computing; the loaded measurement,
mapping and last calibration stay on the session when the dialog is
closed.

## Assistant

`orm_calibrate` (compute, background job) takes the measured folders and
the options of the CLI (`mode` compare / fit / validate, `mapping_file`,
`stage`, priors, `fix_quads`, exclusions, `check_tracking`,
`calibration_out`, `export_deck`) and answers with the selection summary,
the per-plane metrics, the sorted quad corrections and the files it wrote;
the fit runs on a copy and is kept on the session.  `orm_apply` (mutate)
applies the session's last fit or a calibration file through the same
undoable path as the GUI.  Both are exposed through the MCP server.

## Source

* `linac_gen/orm/measured.py` — FORMA / CSV readers and writer
* `linac_gen/orm/devices.py` — naming rules, `DeviceMap`, `resolve_devices`, `align_measured`
* `linac_gen/orm/model.py` — `OrmModel.response` / `tracked_response`
* `linac_gen/orm/compare.py`, `fit.py`, `calibration.py`, `plots.py`
* `linac_gen/cli/orm.py`, `linac_gen/assist/tools_analysis.py` (`orm_calibrate`, `orm_apply`),
  `gui/linac_gen_gui/interphase/dialogs/orm_dialog.py`
* tests: `tests/orm/`, `tests/cli/test_orm_cli.py`, `tests/assist/test_orm_tool.py`, `tests/gui/test_orm_dialog.py`

← [Orbit correction](07_correction.md)
