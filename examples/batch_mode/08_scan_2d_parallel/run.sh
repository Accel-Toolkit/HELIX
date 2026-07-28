#!/bin/sh
# Batch-mode example 08 — 2-D scan, run in parallel
# ---------------------------------------------------------------------------
# Options shown : multiple --vary  (Cartesian product) · --parallel N
#
# Sweeps beam current x quadrupole gradient — 3 x 3 = 9 points — spread
# across 4 worker processes.  The CSV carries both varied columns plus the
# metrics.  Repeating --vary forms the Cartesian product of the value sets.
#
# Run from the repository root.
# ---------------------------------------------------------------------------
cd "$(dirname "$0")/../../.." || exit 1

python -m linac_gen scan examples/batch_mode/chicane.lgproj \
    --vary current=0:4:2 \
    --vary @2.gradient=4:8:2 \
    --mode envelope \
    --parallel 4 \
    --out examples/batch_mode/08_scan_2d_parallel/out/scan_2d.csv
