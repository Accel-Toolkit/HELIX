#!/bin/sh
# Batch-mode example 05 — output formats
# ---------------------------------------------------------------------------
# Options shown : --format hdf5 | openpmd | partran
#
#   hdf5     HELIX-native HDF5 (the default)
#   openpmd  openPMD-beamphysics 1.1 HDF5 — a portable interchange format
#   partran  TraceWin-style tab-separated ASCII, one row per element
#
# The same envelope run is written in all three formats, each into its own
# sub-directory of ./out/.
#
# Run from the repository root.
# ---------------------------------------------------------------------------
cd "$(dirname "$0")/../../.." || exit 1

for FMT in hdf5 openpmd partran; do
    python -m linac_gen run examples/batch_mode/chicane.lgproj \
        --mode envelope \
        --format "$FMT" \
        --out "examples/batch_mode/05_run_output_formats/out/$FMT"
done
