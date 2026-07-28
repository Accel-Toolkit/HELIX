#!/bin/bash
# HELIX GUI launcher — double-click in Finder to start.
# Equivalent of a Windows .bat file.

# Resolve the directory this script lives in (so it works no matter
# where it's placed or symlinked from).
HELIX_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Python + Qt plugin path
PYBIN="${HELIX_PYTHON:-python}"
QT_PLUGINS="$("$PYBIN" -c 'import PyQt6, os; print(os.path.join(os.path.dirname(PyQt6.__file__), "Qt6", "plugins"))' 2>/dev/null)"

if [ -z "$QT_PLUGINS" ]; then
    echo "ERROR: Could not locate PyQt6 plugins. Is PyQt6 installed in '$PYBIN'?"
    echo "       Try: $PYBIN -m pip install PyQt6"
    read -n 1 -s -r -p "Press any key to close..."
    exit 1
fi

export PYTHONPATH="$HELIX_ROOT:$HELIX_ROOT/gui:$PYTHONPATH"
export QT_PLUGIN_PATH="$QT_PLUGINS"

cd "$HELIX_ROOT" || exit 1

echo "Launching HELIX GUI from $HELIX_ROOT ..."
"$PYBIN" -m linac_gen_gui.interphase.app
