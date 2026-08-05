# LEBT compensation (SCC)

`linac_gen.analysis.scc` computes **residual-gas space-charge
compensation** for DC/continuous (LEBT) transport — the physics behind
the neutralisation factor that the
[`SpaceChargeComp`](../03_elements/17_spacechargecomp.md) card consumes.
Until now that factor had to be guessed; this analysis derives it from
the gas physics and your actual run.

Ported from the author's standalone SCC simulator (analysis layer only —
HELIX's own [DC tracking](04_dc_mode.md) supersedes its trackers), and
pinned against that tool's cross-language test-vector contract.

## Physics

* **Gas library** — H₂, He, N₂, Ar, Kr, Xe with NIST/ICRU-sourced
  parameters; ionisation cross sections use an analytic Rudd
  reciprocal-sum model, stripping/capture are tabulated
  (5–200 keV H-equivalent, per-channel extrapolation).
* **τ_scc** = 1 / (Σᵢ nᵢ σᵢ v_b) over the mixture; f_c build-up is the
  exact relaxation `df_c/dt = (η_ss − f_c)/τ_scc`.
* **η_ss, "Computed" mode** — a self-consistent 1-D radial nonlinear
  Poisson–Boltzmann balance: trapped secondaries (ions for negative
  beams, electrons for positive) are Boltzmann-distributed in the
  self-consistent potential, and their production
  (Σ n σ_ion I/|q|) balances escape over the well; over-neutralisation
  is flagged, not hidden. "Assumed" mode uses your η directly.
* **Beam potential** φ(z) = ±I(z)/(4πε₀βc)·[1 + 2 ln(r_pipe/a)]·(1−f_c),
  with a = 2σ from the run's actual envelope and r_pipe from the
  lattice apertures.
* **Residual-gas beam loss** — stripping (H⁻/D⁻) and capture (H⁺/D⁺/He⁺)
  optical depth along z; a loss channel independent of the
  [magnetic/intra-beam stripping](../09_diagnostics/05_stripping.md)
  diagnostics.
* **End taper** — a phenomenological reduction of η over the first/last
  15 % of the line (floor 0.25), standing in for axial escape at the
  extractor and RFQ repeller.  Optional; the three magic numbers are
  disclosed, not derived.

!!! warning "Calibration disclosure"
    Cross-section *shapes* and peak positions are sourced; the twelve
    absolute magnitudes are an **inherited calibration** to
    Valerio-Lizarraga CERN-THESIS-2015-121 (roughly 10× published bare
    σ_ion — defensible as effective values folding in secondary-electron
    cascade ionisation).  Do not quote them as literature values.  The
    45 keV anchors are pinned by `tests/analysis/test_scc.py`.

!!! note "External anchor: published PIP2IT/PXIE measurements"
    On a three-solenoid 30 keV H⁻ LEBT at the published operating
    conditions, the analysis lands inside every measured band from
    Prost, Carneiro & Shemyakin (PRAB **21**, 020101 (2018)) and
    Carneiro *et al.* (NAPAC2016, TUPOB64): build-up completes well
    inside the measured ≲250 µs bound at 5×10⁻⁶ Torr, τ_scc is "tens of
    microseconds" at source-chamber pressures, the upstream
    steady-state neutralisation degree (≈0.97) sits in the 80–100 %
    fitted band, and a cleared section downstream of the second
    solenoid reproduces the published un-neutralised-section pattern
    self-consistently.  One honest tension: at the chopper-region
    pressure (~10⁻⁷ Torr) the inherited calibration gives τ_scc ≈
    0.2 ms where the papers' bare-σ estimate is ~ms — the measurements
    only *bound* τ, so both pass, but low-pressure build-up times
    should be treated as order-of-magnitude.

## Python API

```python
from linac_gen.analysis.scc.driver import scc_analysis

a = scc_analysis(results, lattice, species="H-", gas="N2",
                 pressure_mbar=2e-5)          # DC results required
a["fc"], a["phi_V"], a["tau_scc_us"]          # per-z profiles
a["scc_cards"]                                # suggested card factors
```

`mode="assumed"` with `eta_assumed=…` bypasses the balance;
`build_up_us=…` evaluates the transient instead of the steady state;
`gas={"Ar": 1e-5, "H2": 4e-6}` describes mixtures (an H₂ baseline of
10⁻⁶ mbar is always added, as in the source tool).  Bunched results are
refused with a `reason` — the exact mirror of the
[Hofmann diagnostic](../09_diagnostics/07_hofmann_footprint.md)'s
DC refusal.

### Cleared regions

Real LEBTs are not uniformly neutralised: the PIP2IT/PXIE line is
deliberately operated with the last ~1 m before the RFQ **un-neutralised**
(the chopper kicker at −300 V DC sweeps the compensating ions out, while
+50 V electrically-isolated diaphragms confine them upstream — Prost,
Carneiro & Shemyakin, PRAB **21**, 020101 (2018)).  Pass
`cleared_regions=[[start, end], …]` (element index ranges, inclusive,
against the lattice the results were tracked with) and an optional
`cleared_residual` (default 0): inside each range f_c is forced toward
the residual through a smooth flat-top window (same tanh edge shape as
the end taper — a clearing field's fringe is soft, not a step), applied
after the end-taper and before the build-up transient.  Suggested cards
over a cleared stretch inherit the low factors automatically.  The
window needs a resolved record grid: on a coarse one-record-per-element
run the analysis flags the region as UNRESOLVED (enable
`record_substeps` in the Numerics tab, or use **Iterate**, which
resolves substeps itself).

### Self-consistency

The plain analysis is **one-shot**: f_c comes from the σ(z) profile of a
run that was tracked at the run's *own* space-charge state, so the
suggested cards are a first iteration, not the fixed point (the result
carries a note saying so).  `scc_self_consistent` closes the loop — it
repeatedly re-runs the **envelope** on a working copy of the lattice
with the suggested `SPACE_CHARGE_COMP` cards until the factors stop
moving (the physical fixed point `f_card = f_c(σ(f_card))`):

```python
from linac_gen.analysis.scc.iterate import scc_self_consistent

a = scc_self_consistent(lattice, beam_config, gas="N2",
                        pressure_mbar=2e-5)   # DC beam config required
a["iterate"]                # {converged, n_iter, history, omega, tol}
a["scc_cards"]              # the set the final run was tracked with
```

The σ-coupling enters the balance only through the well-depth logarithm,
so the loop typically converges in 2–3 envelope passes; `omega < 1`
under-relaxation is available as insurance if a configuration ever
oscillates, and non-convergence is reported honestly
(`iterate["converged"] = False` plus a note) rather than raised.  The
caller's lattice is never modified — pre-existing cards are set aside
(with a note) so the iteration starts from the uncompensated line.
Limitation: the loop iterates the envelope model (no scraping losses);
verify the converged state with one MP run.

## GUI

Results tab → **LEBT compensation (SCC)** tile (TWISS · DIVERGENCE ·
HALO section; needs DC results).  Controls mirror the source tool: gas
species, log-pressure slider (10⁻⁷–10⁻⁴ mbar), temperature, solver mode
(Computed / Assumed with the η spin greyed accordingly), trapped-ion
temperature, end-taper toggle, and build-up time.  **Compute** runs off
the GUI thread and plots f_c(z) with η_ss(z), φ(z), τ_scc(z), and the
gas-survival profile; **Pressure scan** sweeps 10⁻⁷–10⁻⁴ mbar and plots
the scalars against log₁₀ pressure.  The **cleared region** row forces
f_c toward a residual value over an element range (tick the box, pick
the inclusive element span and the residual — see *Cleared regions*
above).  **Iterate** runs the
self-consistency loop (needs the Beam tab set to a continuous beam; it
works even when no results are loaded, since it runs its own envelopes)
and reports the convergence status in the info line.  **Export cards**
copies the suggested `SPACE_CHARGE_COMP` deck lines; **Apply to
lattice…** inserts them into the loaded lattice after confirmation
(existing SCC cards are replaced, current results clear, and the deck
must be saved to persist — the dialog says all of this).

## Assistant

The `lebt_scc` tool (tier *compute*, background job) runs the same
analysis headlessly with the same parameters and returns the profiles,
scalars, suggested cards, and deck lines as JSON.
`self_consistent=true` switches it to the iteration loop and adds the
`iterate` convergence block to the reply; `cleared_regions` /
`cleared_residual` mirror the driver parameters (validated in-band).
