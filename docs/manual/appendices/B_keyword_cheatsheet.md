# Appendix B — TraceWin keyword cheatsheet

One-page printable: every TraceWin `.dat` keyword HELIX recognises,
mapped to its HELIX class.

## Header / control

| TraceWin | Effect |
|---|---|
| `TITLE` | informational title |
| `FREQ f_MHz` | reference RF frequency — materialized as an active `Freq` command element: the machine clock (`ref.phi_s`, with the exact Δφ/σ rescale) switches **at the card**, TraceWin-style, not at the first downstream cavity |
| `PARTRAN_STEP step1 step2` | integration step density (per-metre) → `lattice.step_config` |
| `LATTICE` / `LATTICE_END` | subsection markers |
| `END` | end of file |
| `FIELD_MAP_PATH path` | directory for FIELD_MAP files |

## Element cards

| TraceWin (with slots) | HELIX class |
|---|---|
| `DRIFT L aper [aper_y x_shift y_shift]` | Drift |
| `QUAD L G aper [skew g3 g4 g5 g6 gfr]` | Quadrupole |
| `SOLENOID L B aper` | Solenoid |
| `BEND angle ρ [field_index aper hv]` | Dipole |
| `EDGE pole_rotation ρ gap k1 k2 [aper hv]` | Edge |
| `GAP V φ_s aper [p_flag]` | RFGap |
| `FIELD_MAP geom L θᵢ R kb ke Ki Ka FileName [p_flag]` | FieldMap / FieldMap3D |
| `SUPERPOSE_MAP Z0 [X0 Y0 θz θx θy]` before each `FIELD_MAP` of a cluster | [SuperposedFieldMap](../03_elements/19_superposedfieldmap.md) — overlapping maps summed per slice; X/Y/θ ignored (as TraceWin does without `SUPERPOSE_MAP_OUT`) |
| `SHIFT_IN_FIELD_MAP dz` before diagnostics | interior diagnostic dz mm inside the next map/cluster (recorder row at s_entry + dz) |
| `RFQ_CELL` (Toutatis format) | RfqCell / VaneRFQ |
| `THIN_STEERING bx_l by_l aper [elec]` | Steerer |
| `STEERER` | Steerer |
| `APERTURE dx dy ap_type` | Aperture |
| `MARKER name` | Marker |

## Diagnostics

| TraceWin | Effect |
|---|---|
| `DIAG_SIZE` / `DIAG_EMIT` | Marker (no snapshot) |
| `DIAG_PHASE` | Marker with a full phase-space snapshot |
| `DIAG_POSITION` | BPM-flagged Marker carrying diagnostic-matching data: `DIAG_POSITION N X Y [dm]` — family `N` links to `ADJUST N v` variables, `X`/`Y` are wanted centroid targets in mm (`|value|≥1e50` leaves the plane free), `dm` the accuracy in mm (default 1). Targets drive the matcher's position constraints (envelope or MP cost solver) and the target-aware orbit correction; also the anchor for `ADJUST_STEERER N`. |
| `SPACE_CHARGE_COMP factor` | SpaceChargeComp (SC neutralisation factor) |

## Error directives

| TraceWin | Effect |
|---|---|
| `ERROR_GAUSSIAN_CUT_OFF σ_max` | global Gaussian cutoff |
| `ERROR_QUAD_NCPL_STAT N r dx dy φx φy φz dG dG3 dG4 dG5 dG6 [Nb]` | quad alignment + field |
| `ERROR_CAV_NCPL_STAT N r dx dy φx φy E φ dz [Nb]` | cavity alignment + voltage + phase |
| `ERROR_BEND_NCPL_STAT N r dx dy φx φy φz dg dz` | dipole alignment + field |
| `ERROR_BEAM_STAT r dx dy dφ dxp dyp de dEx dEy dEz mx my mz dIb` | beam-input jitter |
| `ERROR_SET_RATIO r1 r2 …` | multi-scale sweep ratios |

Distribution code `r`: 0 = gaussian (no constant special-case; the
default cutoff applies), 1 = uniform, 2 = gaussian (4, 5 = binary,
treated as gaussian).

The `_DYN` and `_CPL_` variants of `ERROR_QUAD*` / `ERROR_CAV*` /
`ERROR_BEND*` / `ERROR_BEAM*` are **absorbed as static NCPL errors**
(the prefix match catches them) — not skipped, but the dynamic /
coupled semantics are not simulated.

## Matching directives

| TraceWin (with slots) | Effect |
|---|---|
| `SET_SYNC_PHASE` | (no args) next `FIELD_MAP`'s θᵢ read as synchronous phase |
| `SET_TWISS family αx βx αy βy αz βz kax kbx kay kby kaz kbz` | Twiss target (k-flags select constrained values; no emittance slots) |
| `SET_POSITION k x xp y yp` | centroid-position constraint |
| `SET_SIZE k x y φ/z k2` | beam-size constraint (4th slot: >0 = σ_φ deg, <0 = σ_z \|mm\|; k2 = centroid-inclusive transverse sizes, warn-skipped) |
| `SET_SIZE_MAX k N x y φ/z k2` | σ upper bound over N elements (same 4th-slot sign rule) |
| `SET_SIZE_MIN k N x y φ/z k2` | σ lower bound over N elements (same 4th-slot sign rule) |
| `SET_ACHROMAT k f1 f2 plane` | dispersion = 0 |
| `SET_BEAM_PHASE_ERROR dφ random_flag` | beam phase-error target |
| `SET_BEAM_E0_P0 k dE dφ ke kp` | energy / phase offset target |
| `SET_BEAM_ENERGY k E_MeV` | beam-energy target |
| `SET_GAUSSIAN_CUT_OFF σ` | Gaussian truncation for matching |
| `SET_BEAM_PHASE_ADV k N μx μy μz` | phase-advance target |
| `SET_SEPARATION k sx sy` | beam-separation target |
| `SET_ADV kxot kyot` | phase-advance conditioning |
| `SET_KE_OUT_MIN E_MeV weight` | output kinetic-energy floor |
| `MIN_EMIT_GROWTH plane weight` | minimise per-plane ε growth |
| `MIN_EMIT_4D_GROWTH weight tol_4d tol_z` | minimise 4-D ε growth |
| `MIN_TRANSMISSION threshold_pct weight` | transmission floor |
| `ADJUST target param_idx [link_group vmin vmax start_step kn]` | generic matching variable (`start_step` is round-tripped but ignored — HELIX optimisers pick their own initial steps; `kn` legacy, unused) |
| `ADJUST_STEERER N vmax first_step` | steerer variable, both planes (orbit correction) |
| `ADJUST_STEERER_BX` / `ADJUST_STEERER_BY` | single-knob variants: `_BX` = `bx_l` knob → **vertical** plane; `_BY` = `by_l` → **horizontal** |
| `ADJUST_BEAM_TWISS` / `_CENTROID` / `_EMIT` / `_CURRENT` | beam-input variables (`diag_n` + flag tail) |

There are no per-element `ADJUST_QUAD` / `ADJUST_SOLENOID` /
`ADJUST_DIPOLE` / `ADJUST_DRIFT` / `ADJUST_GAP` or `SET_TWISS_X`
keywords — use the generic `ADJUST` / `SET_TWISS` cards.

## Recognised no-op markers

Three classes, all kept as plain `Marker` elements — but since the
honesty round (2026-07) they are **reported differently** because they
mean different things physically:

**Diagnostic hardware** (no transport effect by nature; silent; `BPM`
is additionally BPM-flagged for orbit correction):

```
BPM  XCOR  YCOR  ACCT  COL  RPU  FFC  DPI  DCCT  RWCM  ASCN  FASTGV
LASERPROFILE  MEBTABSORBER  CHOPPER
```

**Physics downgrades** (TraceWin would change the fields or the beam;
HELIX cannot honour them — each card WARNS in permissive mode and is a
hard **error under `strict=True`**):

```
SUPERPOSE_MAP_OUT   (curved reference trajectory through dipole maps —
    straight-axis SUPERPOSE_MAP clusters ARE supported)
READ_DST  BEAM_ROT                                     (the beam is not
    re-loaded or rotated mid-lattice)
```

The same warn-or-refuse rule applies to `ERROR_*_DYN` (treated as
static), coupled `ERROR_*_CPL_*` groups (sampled independently),
unimplemented `ERROR_*` forms (ignored, with a visible warning),
negative `FIELD_MAP` geometry codes (second-order flag ignored), and
missing/rejected field-map files (dropped with a loud warning in
permissive mode; fatal in strict — a strict parse never returns a
lattice missing a cavity), `SUPERPOSE_MAP` clusters that mix RF
frequencies (fall back to the legacy end-to-end layout), and
`SHIFT_IN_FIELD_MAP` in front of a `DIAG_POSITION` that carries
orbit-correction targets (BPMs must stay lattice elements).

**Fit hints / RFQ setup** (no transport meaning in HELIX; reported as
one grouped warning line per deck, never fatal):

```
MATCH_FAM_FIELD  MATCH_FAM_GRAD  MIN_FIELD_VARIATION
RFQ_GEOM  RFQ_GAP_RMS_FFS
```

## HELIX extensions (comment-prefixed)

| HELIX directive | Effect |
|---|---|
| `;@LG kernel=tsc` | PIC kernel (cic / tsc) |
| `;@LG green_kind=igf` | Green's function (igf / point) |
| `;@LG dc_kernel=gaussian` | DC kernel (uniform / gaussian / pic2d) |
| `;@LG nx=64 ny=64 nz=64` | PIC grid |
| `;@LG grid_extent=5.0` | PIC ±σ box |
| `;@LG use_gpu=auto` | CPU/GPU backend |
| `;@LG integrator_kind=dkd` | RK4 family |
| `; HELIX_FOIL name material thickness_ug_cm2` | scattering / stripper Foil element |
| `; HELIX_SC_GRID extent_sigma` | 3-D PIC grid extent (σ) from here on (MP bunched only) |

`;@LG` values are parsed and stored on `lattice.lg_options` for
tooling and round-trip, but are not currently wired into the runtime
configs — see [HELIX extensions](../06_running/03_lg_extensions.md).

## Deferred / not honoured

| Not implemented | Status |
|---|---|
| `REPEAT_ELE` | no parser branch — reported as an unsupported card |
| `ERROR_STAT_FILE` | file-driven errors — kept as a Marker |
| `ERROR_RFQ_CEL_NCPL_STAT` | per-cell RFQ errors — kept as a Marker |
| `ERROR_*_DYN` / `ERROR_*_CPL_*` semantics | absorbed as static NCPL (see above); dynamic / coupled behaviour not simulated |

`ADJUST_STEERER`, formerly deferred, now drives steerer-based orbit
correction.

← [Appendix A](A_physics_references.md) ·
[Continue to Appendix C →](C_glossary.md)
