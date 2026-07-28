@echo off
REM HELIX GUI Launcher for Windows.
REM Run from the HELIX project root directory.
REM Uses %CONDA_PREFIX% when a conda env is active (a dedicated PyQt6 env
REM avoids the Anaconda-base ICU/PyQt6 DLL conflict that breaks Qt6Core),
REM otherwise falls back to `python` from PATH.

set PYTHONPATH=%~dp0;%~dp0gui
if defined CONDA_PREFIX (
    "%CONDA_PREFIX%\python.exe" -m linac_gen_gui.interphase
) else (
    python -m linac_gen_gui.interphase
)
if errorlevel 1 (
    echo.
    echo HELIX GUI failed to start. Ensure Python 3.11+ with PyQt6 is
    echo available - e.g. activate your conda env first.
)
