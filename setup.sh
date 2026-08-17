#!/bin/bash
# HELIX one-click setup for Linux / macOS (terminal).
#
# Creates an isolated .venv next to this script, installs HELIX with
# the GUI extra into it, smoke-tests the install, and offers to launch.
# Idempotent: re-running reuses the .venv and updates the install (use
# it after `git pull`).  Never installs into the invoking interpreter.
#
# C++ PIC/field-map kernels are OPTIONAL by default: without a working
# compiler the install still succeeds on a pure-Python fallback (the
# GUI notes this at startup; PIC runs are ~20x slower).  To make a
# missing compiler a hard error instead:  LINAC_GEN_REQUIRE_CPP=1 ./setup.sh
set -e

DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$DIR"

if [ ! -f "$DIR/pyproject.toml" ]; then
    echo "ERROR: run this from a full HELIX checkout (pyproject.toml not found)." >&2
    exit 1
fi

# ---- 1. find a Python >= 3.10 --------------------------------------
PYBIN=""
for c in python3.13 python3.12 python3.11 python3.10 python3; do
    if command -v "$c" >/dev/null 2>&1 && \
       "$c" -c 'import sys; raise SystemExit(0 if sys.version_info >= (3,10) else 1)' 2>/dev/null; then
        PYBIN="$(command -v "$c")"
        break
    fi
done
if [ -z "$PYBIN" ]; then
    echo "ERROR: HELIX needs Python 3.10 or newer and none was found on PATH." >&2
    echo "       Install Python 3.10+ (https://www.python.org) and re-run." >&2
    exit 1
fi
echo "Using $PYBIN ($("$PYBIN" -c 'import platform; print(platform.python_version())'))"

# ---- 2. create / reuse the .venv and install -----------------------
VENV_PY="$DIR/.venv/bin/python"
if command -v uv >/dev/null 2>&1; then
    echo "uv detected - using it for a faster install."
    [ -x "$VENV_PY" ] || uv venv --python "$PYBIN" "$DIR/.venv"
    if ! uv pip install --python "$VENV_PY" -e ".[gui]"; then
        echo "" >&2
        echo "ERROR: install failed." >&2
        [ "$(uname)" = "Darwin" ] && \
            echo "       On macOS a missing compiler is fixed by: xcode-select --install" >&2
        exit 1
    fi
else
    if [ ! -x "$VENV_PY" ]; then
        echo "Creating virtual environment in .venv/ ..."
        "$PYBIN" -m venv "$DIR/.venv"
    else
        echo "Reusing existing .venv/"
    fi
    "$VENV_PY" -m pip install --upgrade pip --quiet
    if ! "$VENV_PY" -m pip install -e ".[gui]"; then
        echo "" >&2
        echo "ERROR: install failed." >&2
        [ "$(uname)" = "Darwin" ] && \
            echo "       On macOS a missing compiler is fixed by: xcode-select --install" >&2
        exit 1
    fi
fi

# ---- 3. smoke test -------------------------------------------------
echo ""
if ! "$VENV_PY" - <<'EOF'
import importlib.util
import linac_gen                                    # noqa: F401
import PyQt6                                        # noqa: F401
built = importlib.util.find_spec("linac_gen._pic_kernels") is not None
if built:
    print("C++ kernels: built (fast PIC path active).")
else:
    print("C++ kernels: not built - HELIX runs on the pure-Python "
          "fallback (~20x slower PIC).")
    print("  To build them, install a C++ compiler and re-run "
          "./setup.sh  (see the manual: Installation > Build "
          "prerequisites by platform).")
print("Smoke test OK: linac_gen %s with PyQt6." % linac_gen.__version__)
EOF
then
    echo "ERROR: the installed environment failed its smoke test." >&2
    exit 1
fi

# ---- 4. done -------------------------------------------------------
echo ""
echo "Setup complete.  Launch the GUI any time with:  ./run_gui.sh"
if [ -t 0 ]; then
    printf "Launch it now? [y/N] "
    read -r ans
    case "$ans" in
        y|Y) exec "$DIR/run_gui.sh" ;;
    esac
fi
