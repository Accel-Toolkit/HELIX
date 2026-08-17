#!/bin/bash
# HELIX one-click setup for macOS - double-click me in Finder.
# Thin wrapper around setup.sh that keeps the Terminal window open so
# the outcome (or any error) stays readable.
cd "$(dirname "$0")"
./setup.sh
status=$?
echo ""
if [ $status -eq 0 ]; then
    read -n 1 -s -r -p "Done - press any key to close..."
else
    read -n 1 -s -r -p "Setup FAILED (see above) - press any key to close..."
fi
echo ""
exit $status
