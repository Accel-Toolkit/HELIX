#!/bin/sh
# Batch-mode example 03 — run, matrix mode
# ---------------------------------------------------------------------------
# Options shown : --mode matrix
#
# Linear transfer-matrix tracking.  Writes the 6x6 transfer matrix and the
# per-plane Twiss parameters to a text file (and prints them to stdout).
#
# Run from the repository root.
# ---------------------------------------------------------------------------
cd "$(dirname "$0")/../../.." || exit 1

python -m linac_gen run examples/batch_mode/chicane.lgproj \
    --mode matrix \
    --out examples/batch_mode/03_run_matrix/out
