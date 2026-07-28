#!/bin/sh
# Batch-mode example 09 — multi-run campaign from a job file
# ---------------------------------------------------------------------------
# Options shown : batch subcommand · a JSON job file
#
# Runs every job described in jobs.json — each with its own beam / element
# settings / mode — and collects one row of end-of-lattice metrics per job
# into batch_summary.csv.  See jobs.json in this folder for the schema.
#
# Run from the repository root.
# ---------------------------------------------------------------------------
cd "$(dirname "$0")/../../.." || exit 1

python -m linac_gen batch examples/batch_mode/09_batch/jobs.json \
    --out examples/batch_mode/09_batch/out
