#!/bin/sh
# Batch-mode example 07 — scan an element parameter
# ---------------------------------------------------------------------------
# Options shown : --vary ELEM.attr=v1,v2,v3   (an explicit value list)
#
# Sweeps the gradient of element #2 (a quadrupole) over an explicit list
# of values.  An element selector is NAME.attr or @INDEX.attr; the value
# spec is either start:stop:step or a comma-separated list.
#
# Run from the repository root.
# ---------------------------------------------------------------------------
cd "$(dirname "$0")/../../.." || exit 1

python -m linac_gen scan examples/batch_mode/chicane.lgproj \
    --vary @2.gradient=4,5,6,7,8 \
    --mode envelope \
    --out examples/batch_mode/07_scan_element/out/scan_gradient.csv
