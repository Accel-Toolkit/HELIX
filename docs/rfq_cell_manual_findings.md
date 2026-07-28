# TraceWin `RFQ_CELL` — manual deep-dive

Source: `/mnt/c/Users/abhishek/TraceWin/doc/tracewin.htm`, sections
"RFQ cell" (parameter table) and "RFQ cell (V, ro, A10, m, L, θs, Type, Tc, dP)"
(physics).  Equations are PNG images; `tracewin_fichiers/imageNNN.png` indices
shown.

## Card syntax & parameter list

```
RFQ_CELL  V  ro  A10  m  L  θs  Type  Tc  -  [a]
```

| pos | symbol | unit  | meaning                                                                      |
|----:|:-------|:------|:-----------------------------------------------------------------------------|
| 1   | V      | V     | effective gap voltage (inter-vane voltage)                                   |
| 2   | r₀     | mm    | average vane radius                                                          |
| 3   | A₁₀    | -     | longitudinal acceleration parameter (Crandall A₁₀)                           |
| 4   | m      | -     | vane modulation                                                              |
| 5   | L      | mm    | cell length                                                                  |
| 6   | θs     | deg   | synchronous-particle RF phase                                                |
| 7   | Type   | int   | cell type / sign flag (see below)                                            |
| 8   | Tc     | mm    | transverse curvature                                                         |
| 9   | -      | -     | *legacy slot — manual: "Not used any more"*                                  |
| 10  | a      | mm    | minimum bore radius — **optional**, required only for `TWOTERM` runs (red ⓘ) |

*Note — the user's PXIE `.dat` puts a `dP` value in slot 9 (Toutatis-style
phase shift).  TraceWin manuals from different vintages show this slot
either as `dP` (older, Toutatis-only) or "Not used any more" (newer).
Our parser already accepts it as `dP_deg`.*

`Type` flag values (manual, "RFQ cell (V, ...)" section):

| sign × abs | name              | C-coefficients depend on…           |
|:-----------|:------------------|:------------------------------------|
| ±2         | accelerating cell | own type sign                       |
| ±3         | front-end cell    | next cell type (+) or prev (−)      |
| ±4         | transcell         | next cell type (+) or prev (−)      |

## Per-particle physics (manual "Transport through a RFQ cell")

The cell is sliced into N substeps of length `dz = L/N`.  Each substep is
a Strang **drift–kick–drift** in both transverse and longitudinal planes.
Per-substep state update (image 297 list):

```
W_{i+1}   = W_i + |q|·dz·E_z                                              (img 265)
Φ_{i+1}   = Φ_i + dz · 2π / (β · λ)                                       (img 267)
E_z(z,t)  = (π·A₁₀·V) / (2L) · sin(πz/L) · sin(ω·t_s + φ_s)               (img 268)
```

The transverse kick coefficients (image 277, k_y form; k_x flips the S sign):

```
k_y = − (|q|·dz / (γ·β²·m_c²))
       · [ −S · (V/2) · A · C₁ · cos(ω·t_s + φ_s)
           − (π/L)² · (A₁₀·V/2) · C₂ ]                                    (img 277)
```

Longitudinal kick (image 269):

```
M_z = [ 1  dz/(2γ_o²) ]   [ 1   0  ]   [ 1  dz/(2γ_i²) ]
      [ 0       1     ] · [ K₁  K₂ ] · [ 0       1     ]
```

with K₁, K₂ depending on C₃ (image 270/271 — partial OCR; key point:
K-block adds a longitudinal kick proportional to `(A₁₀V/2)·sin(ωt+φ)·C₃`).

`A` (the DC quadrupole coefficient) is **not** a card argument — TraceWin
derives it from `m`, `r₀`, the 8-term Crandall expansion (or the 2-term
`A ≈ (1−A₁₀)/r₀²` short-form when `RFQ_GEOM` is `TWOTERM`).

## Cell-type C-coefficients (manual, transverse section)

Read from images 281–291 at 4× / 8× upscale.  Conventions:

* `arg = π · z_local / L`
* `sign(x) = +1 if x>0, −1 if x<0`

### ±2 — accelerating

* **C₁ = 1**                       (img 281)
* **C₂ = sin(πz/L)**               (img 282) — z-shape mirrors the longitudinal
* **S  = −sign(Type)**             (img 283)

### +3 — front-end (forward)

* C₁ = some long expression in `[1 − cos] / [1 − cos(0)]`-style normalisation
                                   (img 284 — partial OCR; multi-term)
* **C₂ = 0**                       (img 285)
* **S  = −sign(Type[n+1])**        (img 286)

### −3 — front-end (reverse)

* C₁ = different long expression, again `[1 − cos]` style
                                   (img 287 — partial OCR)
* **C₂ = 0**                       (img 285, same)
* **S  = −sign(Type[n−1])**        (img 288)

### +4 — transcell (forward)

* **C₁ = 1**                       (img 281, same as ±2)
* **C₂ = ½ · [cos(πz/L) + 1]**     (img 289)
* **S  = −sign(Type[n+1])**        (img 286)

### −4 — transcell (reverse)

* **C₁ = 1**                       (img 290)
* **C₂ = ½ · [1 − cos(πz/L)]**     (img 291)
* **S  = −sign(Type[n−1])**        (img 288)

## Discrepancies vs current `linac_gen/elements/rfq_cell.py::_type_coeffs`

Current implementation (lines 162–195):

```python
if abs_type == 2:
    C1 = 1.0
    C2 = float(np.cos(arg))             # ← manual says sin(πz/L)
    S = -sign_type
elif abs_type == 3:
    C1 = 1.0
    C2 = 0.5 * (np.cos(arg) + 1.0)      # ← manual says 0
    S = -1 if self.type_next > 0 else +1
                                         # ← manual: type_next for +3, type_prev for −3
elif abs_type == 4:
    C1 = 1.0
    C2 = 0.5 * (1.0 - np.cos(arg))      # ← manual: matches −4 only; +4 is ½(cos+1)
    S = -1 if self.type_prev > 0 else +1
                                         # ← manual: type_next for +4, type_prev for −4
```

**Three confirmed bugs:**

1. **±2 cells: `C₂ = sin(πz/L)` per manual, code has `cos(πz/L)`.**
   Affects every accelerating cell — ~190 cells out of 203 in the PXIE
   lattice.  Spatial profile of RF defocusing is wrong (zero at
   entrance/exit, peak at mid-cell vs the current cos which has
   the opposite envelope), so the substep-by-substep transverse
   focusing pattern is inverted.

2. **±3 cells: `C₂ = 0` per manual, code has `½(cos+1)`.**
   Affects the few front-end cells (cells #1, ~#2–3) — minor
   absolute impact but the front-end is where σ matching to the
   downstream RFQ is set, so sensitive.

3. **+4 vs −4 not distinguished — manual differentiates `½(cos+1)` vs
   `½(1−cos)`; code lumps both into `½(1−cos)`.**  Affects transcells.

Plus two **S-coefficient sign-source bugs**:

4. **+3 should consult `Type[n+1]` (next cell), code uses `type_next` ✓.
   −3 should consult `Type[n−1]` (prev cell), code uses `type_next` ✗.**

5. **+4 should consult `Type[n+1]`, code uses `type_prev` ✗.
   −4 should consult `Type[n−1]`, code uses `type_prev` ✓.**

## Empirical confirmation

`diag_rfq_mp_nosc.py` (LG MP, no SC, 10k Gaussian particles) shows:

* `W` at RFQ exit: LG 1.082 MeV vs TW 1.078 — within 0.4 % (longitudinal physics OK).
* `σ_x` at RFQ exit: LG 20.0 mm vs TW 0.31 mm — **60×** over-prediction
  (transverse focusing OK in sign but wrong in magnitude / phase).
* Transmission counter wraps to 8842 % (separate accounting bug; particles
  don't get lost the way TW reports it, but σ is computed on whatever's
  alive so the σ comparison is still meaningful).

This confirms that the bugs above are the dominant cause of the σ blow-up
in env mode — the env-mode matrix wouldn't help if the underlying
per-particle physics is itself off.

## Plan

1. **Fix `_type_coeffs` per manual** (5 changes above).  Re-run MP no-SC
   through LEBT+RFQ; expect σ_x to settle within a factor of 2 of TW.
2. **Investigate transmission accounting** (88× wrap) — separate issue
   in `_track_field_map` cell loop or alive-mask handling.
3. **Build `fitted_matrix_slice` per-substep DKD matrix** that mirrors
   the corrected `track_rk4` so envelope mode reproduces MP statistics.
4. **Validate** against `diag_rfq_env_nosc.py`; expect σ_x within ~10 %
   of TW after fix (envelope assumes perfect Twiss matching at RFQ
   entry — the LEBT-RFQ matching is the dominant remaining gap).
