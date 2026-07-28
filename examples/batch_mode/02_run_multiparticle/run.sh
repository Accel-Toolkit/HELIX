#!/bin/sh
# Batch-mode example 02 — run, multi-particle mode
# ---------------------------------------------------------------------------
# Options shown : --mode mp · --n-particles · --nx · --write-dst
#                 --fail-under-transmission
#
# Multi-particle PIC tracking.  --write-dst additionally dumps the final
# beam as a TraceWin .dst.  --fail-under-transmission makes the process
# exit non-zero when the beam loses more than the allowed fraction — handy
# for CI / shell scripts that must branch on the result.
#
# Run from the repository root.
# ---------------------------------------------------------------------------
cd "$(dirname "$0")/../../.." || exit 1

python -m linac_gen run examples/batch_mode/chicane.lgproj \
    --mode mp \
    --n-particles 5000 \
    --nx 32 \
    --write-dst \
    --fail-under-transmission 90 \
    --out examples/batch_mode/02_run_multiparticle/out

echo "exit code: $?   (0 = final transmission stayed above 90%)"
