# Sequential-scan matcher: reversal reference + loss rejection

`sequential_scan` is HELIX's physics-aware coordinate-descent matcher: it
walks the ADJUST'd elements in lattice order and bracket-scans each
parameter, reversing direction when the beam emittance grows past a
reference. This demo exercises its two robustness controls.

## Files

- `seqscan_demo.dat` — the 6-knob emittance-min lattice (same layout as
  `examples/ml_bayesopt/bo_demo.dat`) with one deliberately tight **6 mm**
  aperture downstream of solenoid 2.
- `run_sequential_scan.py` — runs both demos below.

## Run it

```bash
python examples/sequential_scan/run_sequential_scan.py
```

## 1. Reversal-threshold reference (`seqscan_threshold`)

When the bracket scan decides whether to reverse direction, *what* does it
compare the trial exit emittance against?

| `seqscan_threshold` | Reverse when trial exit ε exceeds… | Use when |
|---|---|---|
| `"input"` | the **input beam** ε (tight) | coupling-resonance lattices where plane-exchange is fine as long as both planes stay below input |
| `"seed_exit"` | the **unmatched lattice's own exit** ε | the lattice has intrinsic growth — reverse only when *worse than doing nothing* |

Representative output (envelope cost, mismatched seed beam):

```
threshold     evals      baseline         final
input            85     2.330e-02     2.317e-02
seed_exit        85     2.330e-02     9.023e-03
```

On this intrinsic-growth lattice `seed_exit` reaches a much lower cost:
`input` keeps reversing as soon as the trial exceeds the (small) input ε
and barely moves, while `seed_exit` lets the scan work down toward the
achievable minimum.

## 2. Hard loss rejection (`seqscan_reject_loss`)

A multi-particle-only safety rail (envelope mode tracks no losses). With
`seqscan_reject_loss=True` + `cost_solver="mp"`, any trial step whose
transmission falls below `seqscan_loss_threshold_pct` is **rolled back** —
`best_x` is not moved to it even if the cost dropped — so the matcher can
never lower emittance by scraping beam at the aperture. It closes the
ε-gaming gap. On this gentle lattice the scan never needs a clipping step,
so ON and OFF coincide and the rail simply stays inactive; it bites on
lattices where lowering ε *does* require clipping (see
[`examples/min_transmission/`](../min_transmission/README.md)).

## CLI equivalents

```bash
# reversal reference
python -m linac_gen.matching examples/sequential_scan/seqscan_demo.dat \
    --algorithm sequential_scan --seqscan-threshold seed_exit \
    --energy 2.5 --current 5 --frequency 162.5 --report

# hard loss rejection (multi-particle)
python -m linac_gen.matching examples/sequential_scan/seqscan_demo.dat \
    --algorithm sequential_scan --cost-solver mp --mp-n-particles 400 \
    --space-charge --seqscan-reject-loss --seqscan-loss-threshold-pct 95.0 \
    --energy 2.5 --current 5 --frequency 162.5 --report
```

GUI: Matching tab → Algorithm = `sequential_scan` → **Match** opens the
setup dialog (element checklist, passes/steps, reversal criterion +
reference, and the **Beam loss** hard-rejection checkbox + threshold).

See [Robust emittance minimisation](../../docs/manual/07_matching/07_emittance_min.md)
for the physics.
