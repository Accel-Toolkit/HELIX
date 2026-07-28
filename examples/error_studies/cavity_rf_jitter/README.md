# Cavity RF-jitter tolerance

**Study type:** random voltage amplitude + phase errors on every RF gap
— a typical Low-Level-RF (LLRF) stability budget study.

## Errors applied

```
ERROR_CAV_NCPL_STAT 3 2 0  0  0  0   1.0  1.0  0
```

Per-seed Gaussian draws on every RF gap (3 of them):

* `voltage_rel ~ N(0, 1 %)` — peak field amplitude jitter
* `phase_offset ~ N(0, 1°)` — synchronous-phase error
* No alignment, no `dz`, no frequency error in this minimal example

To add frequency jitter, extend the directive: TraceWin doesn't have
a direct slot, but the Python API exposes `frequency_offset` directly:

```python
study.add_error("GAP_*", "frequency_offset",
                distribution="gaussian", sigma=0.05)  # 50 kHz σ
```

## What you'll see

* **Final-energy histogram** — every seed accelerates the beam by a
  slightly different amount; the histogram quantifies the energy-spread
  contribution from RF jitter.
* **σ_φ ensemble** — phase errors create longitudinal mismatch, growing
  σ_φ across the cavity chain.
* **Transmission stats** — should remain near 100 % at this
  conservative jitter level; bump σ_E to 5 % or σ_φ to 10° to break
  things on purpose.

## Realistic LLRF tolerance numbers

PIP-II-class superconducting cavities target ≤0.01 % amplitude / ≤0.01°
phase stability over a pulse, but per-pulse-to-pulse drift is
~0.1–1 %.  The 1 % / 1° defaults here are deliberately loose to make
the effect visible at 100 seeds — tighten to 0.1 / 0.1 for a real
PIP-II-style budget.

## How to run

`File → Open Project…` → `cavity_rf_jitter.lgproj` → Errors tab →
**Run study** with `n_seeds = 100`.
