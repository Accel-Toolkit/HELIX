# Fine-PIC reference baselines (not committed — 89 MB, reproducible)

Ten-seed fine reference ensemble for the mismatched-FODO halo testbed:
`ref_m1.4_seed{0..9}.npz` — N = 200 000, 96³ grid, 100 SC kicks/m,
mismatch 1.4, seeds 0–9 (~2 min/seed on the M-series Mac CPU).

Regenerate with:

```bash
PYTHONPATH=. python3 scripts/halo_testbed.py --reference
```

Consumed by `scripts/halo_m3_eval.py` and `scripts/halo_m2b_sweep.py`
(along-s integrated errors + envelope-tune gate are computed against the
ensemble mean; tolerances against the ensemble self-scatter).
