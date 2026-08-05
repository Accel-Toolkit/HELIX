@echo off
REM HELIX GUI Launcher for Windows.
REM Run from the HELIX project root directory.
REM Interpreter probe order: active conda env -> .venv next to this
REM script (both venv and conda-created layouts) -> `python` from PATH.
REM (A dedicated env avoids the Anaconda-base ICU/PyQt6 DLL conflict
REM that breaks Qt6Core.)

setlocal
set PYTHONPATH=%~dp0;%~dp0gui
set "_PY="
if defined CONDA_PREFIX if exist "%CONDA_PREFIX%\python.exe" set "_PY=%CONDA_PREFIX%\python.exe"
if not defined _PY if exist "%~dp0.venv\Scripts\python.exe" set "_PY=%~dp0.venv\Scripts\python.exe"
if not defined _PY if exist "%~dp0.venv\python.exe" set "_PY=%~dp0.venv\python.exe"
if not defined _PY set "_PY=python"
"%_PY%" -m linac_gen_gui.interphase
REM NEQ, not `if errorlevel 1`: hard crashes (DLL conflicts) exit with
REM NEGATIVE codes on Windows, which `if errorlevel 1` (a >=1 test) misses.
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo HELIX GUI failed to start. Ensure Python 3.11+ with PyQt6 is
    echo available - either activate your conda env first, or create a
    echo virtual environment next to this script:
    echo     python -m venv .venv
    echo     .venv\Scripts\pip install .[gui]
    echo and run this script again.
)
