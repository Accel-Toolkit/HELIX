# Leg A — fault tolerance and compensation

## Scenario classes

| Class | Scenario | Failure knob |
|---|---|---|
| S1 | one cavity off | `voltage_rel = −1` |
| S2 | one magnet (quadrupole, solenoid, dipole) off | `gradient_rel` / `field_rel = −1` |
| S3 | one cavity detuned (default 90 % amplitude, +5° phase) | `voltage_rel`, `phase_offset` |
| S4 | one magnet at partial field (default 90 %) | `*_rel = −0.10` |
| S5 | one steerer off | `bx_l = by_l = 0` (absolute) |
| S6 | two adjacent cavities off | both `voltage_rel = −1` |
| S7 | a cryomodule off | every member of the circuit-map group |
| S8 | an RF station off | every cavity of the station |
| S9 | a magnet circuit off (doublet or arc bus) | every magnet of the group |
| S10 | S1 with the RF phases **frozen** | the pins of the design pass + `voltage_rel = −1` |

Every case is a `FailureScenario` of the failure engine (per-member modes),
tracked as one scan-pool point with the class's overrides composed on the
pins of its bracket.  The quick preset runs S1, S2, S5 and S10; the full
preset every class.  Names come from the circuit map (`groups.cryomodule`,
`groups.rf_station`, `groups.doublet`, `groups.arc_bus`); a `NCells` multi-gap
cavity fails through its error slots but has no matcher knob, so it never
compensates.

## The criticality rule and the score

Each case is compared with the baseline of the **same bracket and forward
model**.  It is *critical* when any of: normalised-emittance growth above
`emit_growth_pct` (5 %) in any plane, loss fraction above `loss_frac` (1e-4,
multiparticle only), exit-energy deviation above `energy_pct` (0.5 %), or a
section's peak loss density above its `w_per_m` limit (0.1 W/m in a LINAC
section, 1 W/m elsewhere by default; sections and their names come from the
circuit map).  A lost beam is always critical.  The failure engine's
criticality *score* (weighted transmission loss, emittance growths and energy
deviation) ranks the cases; the rule decides the verdict.  Both are recorded
per case in `legs/faults/faults.csv`, together with the exit-clock slip
`d_phi_end_deg`, the treaty-point energy, phase and sizes, and the loss power
per section.  A threshold set to `null` in `criticality.rule` switches its
term off (the growth or deviation is still reported); a run whose exit energy
is unknown stays critical whatever the rule.

### Two rule sets and the PIP-II reconciliation

The defaults are the FDR-style rule.  The earlier PIP-II physics paper
classified a single-element failure as critical on **> 10 % rms-emittance
growth in any plane or beam loss above 1 %**, with no energy term — as a spec
it is

```json
"criticality": {"rule": {"emit_growth_pct": 10.0, "loss_frac": 0.01, "energy_pct": null}}
```

— and found 43 of the 119 SRF cavities critical with multiparticle TraceWin
runs (HWR 8/8, SSR1 16/16, SSR2 19/35, LB650 0/36, HB650 0/24).  The quick
campaign on the MEBT-to-foil lattice (`examples/MEBT_To_Foil/reliability/`,
envelope model, 123 powered cavities = the 119 SRF cavities plus the four MEBT
bunchers) gives, per family and rule, for a single cavity off with the RF
downstream re-phased (S1) and frozen (S10):

| family | n | FDR rule, re-phased | same, energy term off | paper rule, re-phased | paper rule, frozen |
|---|---|---|---|---|---|
| MEBT bunchers | 4 | 4 | 4 | 3 | 3 |
| HWR | 8 | 8 | 8 | 7 | 7 |
| SSR1 | 16 | 16 | 13 | 12 | 7 |
| SSR2 | 35 | 35 | 9 | 5 | 1 |
| LB650 | 36 | 36 | 2 | 0 | 2 |
| HB650 | 24 | 24 | 1 | 0 | 0 |
| **all** | 123 | 123 | 37 | 27 | 20 |

Read across a row: the 0.5 % energy term alone makes every cavity critical
(the smallest cavity of the line takes more than 0.5 % of 800 MeV — the term
restates the voltage list, and is what a compensation has to recover), the
emittance terms carry the physics, and under the paper's own rule the pattern
of that study reappears — the front end critical, the elliptical sections
not — with two model differences left: the envelope model has no losses and no
separatrix, so the SSR2 cases the paper found critical through longitudinal
blow-up (its ε_z growth reaches 10⁵ %) stay below 10 % here, and its HWR
losses of up to 19.5 % are absent.  Those are exactly what the full preset's
multiparticle verification of the critical and top non-critical cases is for.
(The paper's introduction quotes the 5 % / 10⁻⁴ / 0.5 % thresholds while its
classification section applies 10 % / 1 %; the table above is against the
latter, the numbers it reports.)

## Compensation

The top `compensation.top_n` cases (critical first, then by score) are
re-tuned by the matcher with one job per (case × strategy) in a process pool:

* `k_out_of_n` — the `k` nearest same-category elements on each side;
* `spares` — the unpowered spare cavities of the circuit map, brought up from
  zero (an unpowered field map is a `"spare"` in the failure engine's extended
  vocabulary; `_CATEGORY["spare"] = "cavity"`);
* `section_rematch` — every same-category element downstream of the fault to
  the end of its section;
* `l_neighboring_lattices` — `l` FODO periods around the fault.

The amplitude window of every knob follows the **family headroom rule**:
`(0, headroom[family]) × the family's nominal amplitude` (the signed median of
its powered members), so a spare can be raised to the family's headroom and a
neighbour cannot exceed it; the family of an element is the leading letters
of its name (`compensation.family_pattern`).  The objectives are the exit
energy (`SET_KE_OUT_MIN`), the transmission in multiparticle mode
(`MIN_TRANSMISSION`), the nominal **arrival phase at the exit**
(`SET_PHASE_OUT`, the reference RF clock of the nominal run — so the RF
downstream of the treaty point stays in phase), the end-of-line emittance
growth in every plane (`MIN_EMIT_GROWTH` x/y/z — a magnet fault leaves the
energy untouched, so without it a magnet compensation would be a no-op) and
an optional size ceiling over the line (`SET_SIZE_MAX`).  The matcher's own
verdict, `recovered`, is the energy / transmission criterion; the campaign's
verdict, `recovered_by_rule`, additionally requires the compensated beam to be
**not critical by the study rule** against the case's baseline
(`after_critical` in `compensation.csv`).  A case recovered by the rule is
`operator_retune`, a non-critical one `auto_rephase`, a critical one whose
every attempted strategy failed `unrecoverable`, one never attempted
`not_compensated`; those fractions feed the availability leg.  On the
MEBT-to-foil lattice the most critical single fault, a MEBT quadrupole off,
is not recovered by its two neighbours within a short matcher budget: it
needs the full-preset section re-match.

!!! note "A physical floor"
    With the **last** cavity of a line down, every compensator sits upstream:
    the beam is then faster over the middle section, arrives earlier, and no
    knob can slow it without decelerating — the arrival-phase objective has a
    floor there (about 15° on the demo deck).  A downstream neighbour absorbs
    the timing.

In the full preset every critical case, the top non-critical ones and every
recovered compensation are re-run with the multiparticle model
(`forward.mp_verify`, `compensation.verify_mp`), with the pins harvested by a
multiparticle design pass (the two engines round the RF clock differently, so
each bracket is pinned by the engine that runs it).

## Outputs

`legs/faults/faults.csv` (one row per case, bracket and model),
`compensation.csv` (per case × strategy: compensators, settings, energy and
transmission after, residual cost), `criticality_map.csv`,
`unrecoverable.json`, and the figures `criticality_by_case.png`,
`bracket_frozen_vs_rephased.png`, `compensation_recovery.png`.
