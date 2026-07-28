# Element failure analysis: impact ranking + fault recovery

Which element failures hurt the beam most, and can the machine recover? This
demo puts elements into a failure state, ranks them by beam impact, and — the
MYRRHA / LightWin local-compensation scheme — re-tunes neighbouring cavities to
recover the design energy after the worst failure.

## Files

- `demo.dat` — a compact PIP-II-like SC section: four RF cavities focused by
  three solenoids (enough cavities that one failure can be compensated).
- `run_failure_analysis.py` — single-element OFF sweep → criticality ranking,
  then compensation of the worst failure.

## Run it

```bash
python examples/failure_analysis/run_failure_analysis.py
```

Representative output:

```
7 failable elements: ['SOL_001','GAP_001','SOL_002','GAP_002','SOL_003','GAP_003','GAP_004']
Baseline exit energy = 6.2196 MeV

Criticality ranking (single-element OFF):
  element      criticality    ΔE [MeV]
  GAP_004           1.1176      1.1276
  GAP_003           0.9267      0.9397
  GAP_002           0.9140      0.9272
  GAP_001           0.7112      0.7250
  SOL_001           0.0000      0.0000   ← solenoids only focus: no energy impact
  ...

Worst single failure: GAP_004:off
  compensators : ['GAP_003','GAP_002','GAP_001']
  recovered    : True
  exit energy  : 6.2196 MeV  (baseline 6.2196)
```

The cavities rank by their energy contribution; the solenoids score 0 on this
(loss-free envelope) run because they only focus. The worst cavity failure is
**fully recovered** by ramping its three neighbours' voltage/phase.

## Failure modes & combinations

| Mode | Meaning | Applies to |
|---|---|---|
| `off` | element transfers nothing (`*_rel = -1`) | all |
| `detune` | cavity amplitude scale + phase offset | cavities |
| `partial` | magnet field/gradient scaled (e.g. 90 %) | magnets |

Combinations: **single** (one at a time), **pairs** (N×N criticality
heatmap), **custom** (named sets failing together).

## Surfaces

- **GUI** — the **Failures** tab (Targets → Mode → Combination → optional
  Compensation → Run). See [Failures tab](../../docs/manual/10_gui/06c_failures_tab.md).
- **CLI** —
  ```bash
  python -m linac_gen failures examples/failure_analysis/demo.dat \
      --mode off --combination single --forward envelope --workers 1 \
      --energy 2.5 --freq 162.5 \
      --compensate --strategy k_out_of_n --k 2 \
      --comp-cost-solver envelope --comp-algorithm least_squares \
      --out failures.csv
  ```
- **API** — `from linac_gen.failures import enumerate_scenarios, FailureStudy,
  compensate, CompensationConfig, FailureKind`.

## Notes

- Envelope mode is fast but tracks no particle loss; use `--forward mp`
  (cavity-OFF / magnet-PARTIAL with apertures) to see transmission impact.
- Pairs are O(N²); the heatmap diagonal reuses the single-failure results.
- Compensation recovers the energy where the surviving elements have headroom;
  a failure with no feasible compensator reports `recovered = False` honestly.
