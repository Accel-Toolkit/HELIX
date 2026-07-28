# Hofmann chart & tune footprint

Two space-charge diagnostics that build on the [channel
tunes](06_phase_advance.md).  Both are honest about what is exact and what
is indicative.

## Hofmann stability chart

`linac_gen.analysis.hofmann` overlays the machine's per-cell channel-tune
trajectory on the Hofmann anisotropy-resonance chart (Hofmann, Franchetti,
Boine-Frankenheim, Qiang & Ryne, *PRST-AB* **6**, 024202, 2003).

* **abscissa** — depressed tune ratio `k_z/k_x` (σ_l/σ_t);
* **ordinate** — transverse tune depression `k_x/k_0x` (1 at zero current);
* the resonance structure is a **family** of charts parameterized by the
  emittance ratio ε_z/ε_x; the anisotropy resonances sit at rational tune
  ratios `k_z/k_x = m/n` (1/3, 1/2, 2/3, 1, 3/2, 2).

```python
from linac_gen.analysis.hofmann import hofmann_trajectory, resonance_bands
from linac_gen.analysis.phase_advance import channel_phase_advance

ch = channel_phase_advance(results, period)       # probe-bearing results
traj = hofmann_trajectory(ch, results)
print(traj["ratio"], traj["depression"])          # per-cell chart coords
print(traj["transverse_label"], traj["emit_ratio"])
```

`hofmann_trajectory` uses the x plane as the transverse reference when
uncoupled; when coupled it picks the normal mode with the larger bare tune
(reported in `transverse_label`).  `emit_ratio` (ε_z/ε_x at the period
entry) tells you which published family to compare against.

**What is exact vs indicative.** The trajectory points are exact — they are
the det-normalized channel tunes of the solver's own tangent map.  The
resonance *lines* at `k_z/k_x = m/n` are exact conditions.  The shaded band
*widths* from `resonance_bands()` are **indicative only**: they follow a
`w ∝ (1 − depression)` heuristic that reproduces the chart topology (bands
grow toward strong space charge, vanish at η → 1) but are **not** the
published growth-rate maps.  For quantitative stop-band edges, read the
published chart for your ε_z/ε_x family.

## Frozen-SC tune footprint

`linac_gen.analysis.footprint.tune_footprint` measures the **incoherent**
per-particle tune spread of one period cell.

A *linear* monodromy gives every particle the same eigenfrequency — no
footprint.  So HELIX tracks a sparse amplitude ladder repeatedly through
the cell with the **real nonlinear element transport**, while the
space-charge field is **frozen** from a reference pass of the **matched
Gaussian field beam**:

* the frozen field is the 2-D Gaussian-equivalent transverse SC field
  (the same `_gauss_field_2d` formula the PIC solver uses), with σ_x(s),
  σ_y(s) taped from a beam drawn from the matched Σ — **not** from the
  sparse probe ladder, whose std would set an arbitrary field strength;
* because the Gaussian field is nonlinear, core particles see the full
  central gradient and tail particles a weaker one → a **real
  amplitude-dependent spread**;
* per-particle tunes come from a windowed-FFT NAFF-lite of the
  Courant–Snyder-normalized coordinate.

```python
from linac_gen.analysis.footprint import tune_footprint
fp = tune_footprint(lattice, ref, period, base_initial=initial,
                    current=5.0, n_turns=256, n_particles=120)
print(fp["mu_x_core_deg"], fp["mu_x_spread_pp_deg"])   # core tune, spread
```

Returned keys include per-particle `qx` / `qy` (fraction of a cell) and
launch amplitudes `ax_sigma` / `ay_sigma`; the small-amplitude
`mu_{x,y}_core_deg`; and the spread `mu_{x,y}_spread_pp_deg` (peak-to-peak)
/ `mu_{x,y}_spread_rms_deg`.

**Physics & honesty.** For a defocusing SC field the core is the *most*
depressed (tune rises with amplitude toward the bare tune), and the
Gaussian core sees ~2× the rms-equivalent gradient — so the ordering is
**core < rms channel tune < bare**, *not* a core-equals-centroid identity.
At zero current the footprint collapses to a point (bare tune, zero width).
This is a **frozen-field** footprint, not a self-consistent PIC footprint
(future work); the longitudinal plane is not modelled, and accelerating
cells use a frozen-energy-per-turn approximation.  Absolute tunes near
strong depression approach the FFT resolution floor (~360°/`n_turns` per
cell) and are approximate.

## GUI

Both live on the **Results** tab: the **Hofmann stability chart** popup
(period selector, resonance lines with m/n labels, indicative bands at the
median depression, trajectory scatter) and the **Tune footprint (frozen
SC)** popup (a **Compute** button runs the footprint **off-thread** — it
re-tracks the cell — then scatters the per-particle tunes coloured by
launch amplitude).

## Cross-references

* [Phase advance & tune depression](06_phase_advance.md)
* `linac_gen/analysis/hofmann.py:1`
* `linac_gen/analysis/footprint.py:1`

← [Phase advance](06_phase_advance.md) ·
[Continue to GUI → Overview →](../10_gui/01_overview.md)
