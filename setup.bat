@echo off
REM HELIX one-click setup for Windows - double-click me.
REM Creates an isolated .venv next to this script, installs HELIX with
REM the GUI extra into it, smoke-tests the install, and offers to
REM launch.  Idempotent: re-running reuses the .venv and updates the
REM install (use it after "git pull").
REM
REM C++ PIC/field-map kernels are OPTIONAL: without MSVC Build Tools
REM the install still succeeds on a pure-Python fallback (the GUI notes
REM this at startup; PIC runs are ~20x slower).
REM
REM Style notes for maintainers: no %ERRORLEVEL% inside parenthesized
REM blocks (it expands at parse time there - use && / || or goto), and
REM status tests use "NEQ 0", never "if errorlevel 1" (hard crashes
REM exit with NEGATIVE codes that a >=1 test misses).
setlocal
cd /d "%~dp0"

if not exist "%~dp0pyproject.toml" (
    echo ERROR: run this from a full HELIX checkout - pyproject.toml not found.
    goto :fail
)

REM ---- long-path pre-flight (torch's deep dist-info vs MAX_PATH) ----
set "_P=%~dp0"
if not "%_P:~90,1%"=="" (
    echo WARNING: this folder path is quite long. Windows' 260-character
    echo MAX_PATH limit can break the install of torch. If the install
    echo fails with WinError 206, either move HELIX to a shorter path
    echo ^(e.g. C:\HELIX^) or enable Win32 long paths in Group Policy /
    echo the registry ^(see the manual's Windows notes^).
    echo.
)

REM ---- 1. find a Python 3.10+ --------------------------------------
set "_PY="
py -3 -c "import sys; raise SystemExit(0 if sys.version_info >= (3,10) else 1)" >nul 2>nul && set "_PY=py -3"
if not defined _PY python -c "import sys; raise SystemExit(0 if sys.version_info >= (3,10) else 1)" >nul 2>nul && set "_PY=python"
if not defined _PY (
    echo ERROR: HELIX needs Python 3.10 or newer and none was found.
    echo        Install it from https://www.python.org ^(tick "Add to PATH"^)
    echo        and double-click this file again.
    goto :fail
)
echo Using interpreter: %_PY%

REM ---- 2. create / reuse the .venv and install ---------------------
if exist "%~dp0.venv\Scripts\python.exe" goto :have_venv
echo Creating virtual environment in .venv\ ...
%_PY% -m venv "%~dp0.venv"
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: could not create the virtual environment.
    goto :fail
)
goto :venv_ready
:have_venv
echo Reusing existing .venv\
:venv_ready
set "_VPY=%~dp0.venv\Scripts\python.exe"

"%_VPY%" -m pip install --upgrade pip --quiet
"%_VPY%" -m pip install -e ".[gui]"
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo ERROR: install failed. Common Windows causes:
    echo   - WinError 206: path too long, see the warning above
    echo   - no internet access / proxy blocking pypi.org
    goto :fail
)

REM ---- 3. smoke test ----------------------------------------------
echo.
"%_VPY%" -c "import importlib.util, linac_gen, PyQt6; b = importlib.util.find_spec('linac_gen._pic_kernels') is not None; print('C++ kernels: built (fast PIC path active).' if b else 'C++ kernels: not built - pure-Python fallback (~20x slower PIC). To build them, install MSVC Build Tools and re-run setup.bat (see manual: Installation, Build prerequisites).'); print(f'Smoke test OK: linac_gen {linac_gen.__version__} with PyQt6.')"
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: the installed environment failed its smoke test.
    goto :fail
)

REM ---- 4. done -----------------------------------------------------
echo.
echo Setup complete. Launch the GUI any time by double-clicking run_gui.bat
echo (it auto-detects the .venv created here).
set /p _ANS="Launch it now? [y/N] "
if /i "%_ANS%"=="y" start "" "%_VPY%" -m linac_gen_gui.interphase
echo.
pause
exit /b 0

:fail
echo.
pause
exit /b 1
