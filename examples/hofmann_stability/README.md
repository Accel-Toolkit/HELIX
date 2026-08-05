# Hofmann stability — 46-period demo linac

A machine-generic worked example of the corrected anisotropic Hofmann
stability analysis (`linac_gen.analysis.hofmann_dispersion` /
`hofmann_stability` / `hofmann_probabilistic`; manual page:
*Diagnostics → Hofmann stability & tune footprint*).

## Files

| file | what |
|---|---|
| `hofmann_demo.lgproj` | GUI project — open this to load lattice + matched 5 mA beam in one step |
| `make_lattice.py` | generates the deck (run it to regenerate) |
| `hofmann_demo_linac.dat` | 46 LATTICE-bracketed periods, four sections |
| `run_stability_demo.py` | the scenario-matrix driver |
| `hofmann_demo_results.json` | per-scenario / per-section / per-cell output (generated) |
| `hofmann_demo_chart.png` | growth chart + section trajectories (generated) |
| `make_exchange_lattices.py` | generates the emittance-exchange validation pair below |
| `exchange_resonant.(dat\|lgproj)` | 150 cells ON the l=2 band (γ/ν₀ₓ = 0.099, 150/150 flagged) |
| `exchange_control.(dat\|lgproj)` | same beam/current, quads +10% — chart quiet (0/150) |
| `make_validation_figures.py` | fast validation figures: port fidelity + printed-equation anchors, flag anatomy + margin scan |
| `run_exchange_validation.py` | the PIC tracking cross-validation (~10 min; `--plot-only` re-renders from the saved npz) |

## The lattice

2.5 MeV proton, 162.5 MHz, `phi_s = −90°` buncher gaps (no acceleration,
so every cell in a section sits at the same working point).  The four
sections put the per-cell working points in *different* regions of the
chart (zero-current `R = ν_z/ν_x` shown):

| section | cells | focusing | μ_x / μ_z (deg) | R₀ |
|---|---|---|---|---|
| A | 16 | quads ±50 T/m, gap 0.10 MV | 40.2 / 24.9 | 0.62 |
| B | 12 | quads ±35 T/m, gap 0.10 MV | 24.8 / 24.9 | 1.00 |
| C | 12 | quads ±29 T/m, gap 0.15 MV | 13.1 / 30.6 | 2.34 |
| D | 6 | solenoids 0.40 T | x–y coupled | — |

Deck-dialect gotcha baked into `make_lattice.py`: the 4th `QUAD`
positional is the **skew angle in degrees** (not a step count), and `GAP`
voltage is in **volts**.

## Run it

```bash
cd <repo root>
PYTHONPATH=. python examples/hofmann_stability/run_stability_demo.py
```

One envelope run with `phase_probe=True` per scenario feeds
`hofmann_stability` (per-cell flags + S² gate), `anisotropy_margin`
(distance-to-onset in ε_z/ε_x), and `instability_probability`
(Monte-Carlo jitter budget) for each of the four sections.

## What the scenarios demonstrate

* **I = 0** — every quad-section cell trivially stable (η ≈ 1), zero
  flags.
* **I = 5 mA** (ε_z/ε_x ≈ 1.3) — section A produces a genuine
  **higher-order l = 3-odd flag inside the S² ≤ 10 domain**
  (γ/ν₀ₓ ≈ 0.033, P(unstable) ≈ 0.96) that a classical l = 2-only chart
  would miss, and section B a weak **l = 2 envelope flag exactly on the
  R = 1 band** (R = 1.006, γ ≈ 0.010, P ≈ 0.3 — the jitter budget shows
  it is marginal).  Hot cells with S² of 30–80 appear only as
  `flagged_extrap`, never as validity-gated flags.
* **I = 15 / 30 mA** — deeper depression pushes more cells outside the
  perturbative gate (honest `valid` counts drop) while the higher-order
  anisotropy margins of the surviving cells shrink toward onset
  (× 5.4 → × 1.4 in section C).
* Section C also demonstrates an extreme-coordinate regime: its weakly
  focused transverse plane depresses much harder than z under space
  charge, pushing the per-cell R to 3–6 — beyond the plotted chart (the
  figure labels it off-scale) yet still handled by the solver and gate.
* **DC beam** — refused per section with the no-longitudinal-tune
  reason.
* **Section D** — refused as x–y coupled in *every* scenario (solenoid
  channel tunes are normal modes, not Hofmann's x/z planes).

The two validity-gated flags were cross-checked **bitwise** against the
source manuscript package's solver (`PRAB_Hofmann/analysis/dispersion.py`).

The same analysis is available interactively: open `hofmann_demo.lgproj`
in the GUI (loads the deck + the matched 5 mA beam), run the envelope,
then Results tab → *Hofmann stability chart* → **Compute chart**.  It is
also exposed through the assistant tool `hofmann_stability`.

## Tracking cross-validation: emittance exchange (GUI-runnable)

`exchange_resonant.lgproj` / `exchange_control.lgproj` are a matched
pair for validating the chart against self-consistent multiparticle
tracking.  Both carry the identical SC-matched 20 mA proton beam with
ε_z/ε_x = 3; they differ only by a 10 % quad change, which moves the
depressed working point onto / off the l=2 coupling band.

In the GUI: open either project → run the **multi-particle** simulation
(20 000 particles are preset; a PIC grid of 32³ in the Numerics tab is
plenty) → watch the emittance popups.  The resonant channel transfers
longitudinal into transverse emittance over the first ~20 cells
(ε_z −9 %, ε_x/ε_y +38 %, anisotropy 3.0 → 2.0, then self-detunes and
saturates); the control shows only the initial redistribution transient
and stays flat for 140 cells (ε_z −1 %).  The envelope run + Hofmann
popup on the same projects shows the chart predicting exactly this
on/off contrast (150/150 cells flagged vs 0/150).
