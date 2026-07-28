#!/bin/sh
# Batch-mode example 06 — scan a beam parameter
# ---------------------------------------------------------------------------
# Options shown : scan subcommand · --vary VAR=start:stop:step
#
# Sweeps the beam current from 0 to 8 mA in 2 mA steps (0,2,4,6,8) and
# writes a CSV summary — one row per point, with the swept variable plus
# the end-of-lattice metrics (sigma, emittance, transmission, ...).
#
# Run from the repository root.
# ---------------------------------------------------------------------------
cd "$(dirname "$0")/../../.." || exit 1

python -m linac_gen scan examples/batch_mode/chicane.lgproj \
    --vary current=0:8:2 \
    --mode envelope \
    --out examples/batch_mode/06_scan_beam/out/scan_current.csv
