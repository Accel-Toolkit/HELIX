#!/bin/bash
# HELIX MCP-server launcher — exposes HELIX's tools over stdio so an
# MCP client (Claude Code / Claude Desktop, logged in with YOUR OWN
# Claude subscription) can drive the simulator.  HELIX makes no network
# call and needs no API key here; auth lives entirely in the client.
#
# Register with Claude Code:
#   claude mcp add helix -s user -- <abs-path>/run_mcp.sh
# then just run `claude` and ask it to use the "helix" tools.
set -e

DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$DIR"

# Same interpreter-resolution as run_gui.sh (venv > uv > system python3).
if [ -x "$DIR/.venv/bin/python" ]; then
    PYBIN="$DIR/.venv/bin/python"
elif command -v uv >/dev/null 2>&1; then
    export PYTHONPATH="$DIR:$DIR/gui:$PYTHONPATH"
    exec uv run python -m linac_gen assist --mcp "$@"
else
    PYBIN="${HELIX_PYTHON:-python3}"
fi

export PYTHONPATH="$DIR:$DIR/gui:$PYTHONPATH"
exec "$PYBIN" -m linac_gen assist --mcp "$@"
