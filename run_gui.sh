#!/bin/bash
# HELIX GUI launcher for Linux / macOS / WSL.
# Run from anywhere -- resolves to the script's own directory.
set -e

DIR="$(cd "$(dirname "$0")" && pwd)"

# Pick the right Python interpreter:
#   1. If the repo has a .venv/ (uv pip install -e ... creates one), use it.
#   2. Else if uv is on PATH, hand off to "uv run" so it picks the env.
#   3. Else fall back to system python3 (must have PyQt6 installed there).
if [ -x "$DIR/.venv/bin/python" ]; then
    PYBIN="$DIR/.venv/bin/python"
elif command -v uv >/dev/null 2>&1; then
    cd "$DIR"
    export PYTHONPATH="$DIR:$DIR/gui:$PYTHONPATH"
    exec uv run python -m linac_gen_gui.interphase.app
else
    PYBIN="${HELIX_PYTHON:-python3}"
fi

# Locate Qt plugin directory so Qt finds its platform plugins (esp. macOS).
QT_PLUGINS="$("$PYBIN" -c 'import PyQt6, os; print(os.path.join(os.path.dirname(PyQt6.__file__), "Qt6", "plugins"))' 2>/dev/null)"
if [ -z "$QT_PLUGINS" ]; then
    echo "ERROR: PyQt6 not found in $PYBIN" >&2
    echo "       Install with: uv pip install -e \".[gui]\"" >&2
    echo "                 or: $PYBIN -m pip install PyQt6" >&2
    exit 1
fi

export PYTHONPATH="$DIR:$DIR/gui:$PYTHONPATH"
export QT_PLUGIN_PATH="$QT_PLUGINS"

cd "$DIR"
echo "Launching HELIX GUI from $DIR (python: $PYBIN) ..."
exec "$PYBIN" -m linac_gen_gui.interphase.app
