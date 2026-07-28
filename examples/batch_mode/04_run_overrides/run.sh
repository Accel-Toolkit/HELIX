#!/bin/sh
# Batch-mode example 04 — run with scalar & element overrides
# ---------------------------------------------------------------------------
# Options shown : --energy / --current / --species   (common beam flags)
#                 --beam NAME=VALUE                  (any BeamConfig field)
#                 --set ELEM.attr=VALUE              (element-parameter override)
#                 --nx / --step1 / --step2 / --kernel  (convergence / PIC)
#
# A bare .dat lattice with EVERY parameter supplied on the command line —
# nothing is read from a project file.  '--set @2.gradient=5.5' retunes the
# 2nd element (a quadrupole); element selectors are NAME.attr or @INDEX.attr.
#
# Run from the repository root.
# ---------------------------------------------------------------------------
cd "$(dirname "$0")/../../.." || exit 1

python -m linac_gen run examples/batch_mode/chicane.dat \
    --mode mp \
    --energy 800 --species H- --current 4 \
    --beam emit_nx=0.25 --beam beta_x=4.0 \
    --set @2.gradient=5.5 \
    --nx 32 --step1 100 --step2 50 --kernel cic \
    --n-particles 5000 \
    --out examples/batch_mode/04_run_overrides/out
