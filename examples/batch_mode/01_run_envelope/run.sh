#!/bin/sh
# Batch-mode example 01 — run, envelope mode
# ---------------------------------------------------------------------------
# Options shown : run subcommand · --mode envelope · --out · a .lgproj input
#
# Envelope (rms Sigma-matrix) tracking — fast, no macroparticles. Writes a
# HELIX-native HDF5 results file into ./out/.
#
# Run from the repository root:  sh examples/batch_mode/01_run_envelope/run.sh
# ---------------------------------------------------------------------------
cd "$(dirname "$0")/../../.." || exit 1

python -m linac_gen run examples/batch_mode/chicane.lgproj \
    --mode envelope \
    --out examples/batch_mode/01_run_envelope/out
