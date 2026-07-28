# Installation

HELIX runs on Linux, macOS, Windows (WSL2 or native), and works
identically as a Python library or as a bundled `.exe` GUI.

## Prerequisites

| Component | Required version | Notes |
|---|---|---|
| Python | 3.10 or newer | 3.12 is the build target for the Windows `.exe` |
| NumPy | 1.24+ | |
| SciPy | 1.10+ | |
| h5py | 3.8+ | for HDF5 result archives |
| matplotlib | 3.6+ | plotting |
| PyTorch | 2.0+ | **required** — used by surrogates, MPS PIC backend, torch-native matcher objective |
| C++ toolchain | optional | for the C++ PIC kernel; falls back to pure-Python silently |
| libomp | optional | OpenMP runtime for the threaded C++ kernel (auto-detected; see [Build prerequisites](#3-build-prerequisites-by-platform)) |
| CUDA 12.x + cupy | optional | for GPU PIC FFT on NVIDIA hardware (Linux/Windows only) |
| PyQt6 | optional | for the GUI |

## Install paths

### 1. Python library (recommended)

From a clone of the repository:

```bash
git clone https://github.com/Abhishek-Pathak-90/HELIX.git
cd HELIX
pip install -e .
```

This installs `linac_gen` in editable mode and (if a C++ compiler is
present) builds the `_pic_kernels.so` extension.  If the build fails
the package still installs — the simulator silently falls back to the
pure-Python PIC kernels (numerically equivalent, ~3× slower).

For the GUI and developer extras:

```bash
pip install -e ".[gui,dev]"
```

For the manual build extras (this document):

```bash
pip install -e ".[docs]"
```

### 1b. Quick install on macOS with uv (recommended)

For macOS users (Intel or Apple Silicon), `uv` is the fastest path to
a working install — it manages Python versions and the virtual env
automatically, with no system-Python interference.

**One-time prerequisites**:

```bash
# Xcode Command Line Tools (provides git + compilers).
xcode-select --install     # opens a GUI installer; click Install.

# uv package manager.
curl -LsSf https://astral.sh/uv/install.sh | sh
exec $SHELL                # reload your shell so 'uv' is on PATH.

# Python 3.11 (uv-managed; doesn't touch /usr/bin/python3).
uv python install 3.11
```

**Install HELIX**:

```bash
git clone https://github.com/Abhishek-Pathak-90/HELIX.git
cd HELIX
uv venv --python 3.11
uv pip install -e ".[gui,dev]"
```

The full install pulls ~400 MB (PyQt6, matplotlib, torch, numpy,
scipy, h5py, pytest, etc.) and takes 30–90 s depending on network.

**Launch the GUI**:

```bash
./run_gui.sh
```

The launcher auto-detects the `.venv` directory created by `uv venv`
and uses its Python interpreter.  For subsequent runs just
`cd HELIX && ./run_gui.sh` — no shell activation needed.

You can also double-click `HELIX.command` from Finder for a no-terminal
launch — but note it runs `${HELIX_PYTHON:-python}` directly (no
`.venv` detection), so `python` on your `PATH` must have PyQt6, or set
`HELIX_PYTHON` to the venv's interpreter.

!!! note "Skip `[gpu]` on macOS"
    The `[gpu]` extra pulls `cupy-cuda12x`, which is NVIDIA CUDA only
    and has no macOS wheels.  Since HELIX `v3`, the marker on that
    dependency makes `pip install .[gpu]` a silent no-op on macOS, so
    you can ignore the extra entirely.  Apple Silicon users get GPU
    acceleration via the MPS backend that ships automatically with
    the main `torch` dependency.

### 2. GPU acceleration

The PIC Poisson solver can offload its 3-D FFTs to a GPU.  Two backends
ship: **cupy** for NVIDIA CUDA, **torch.mps** for Apple Silicon Metal.
Both implement the same six-method `FftBackend` interface and are
selected by the same `use_gpu` knob.

| Backend | Hardware | Precision | Install |
|---|---|---|---|
| `cpu`  | any (scipy.fft via OpenBLAS / pocketfft, multi-threaded) | FP64 | bundled |
| `cuda` | NVIDIA GPU (compute capability ≥ 7.0) | FP64 | `pip install cupy-cuda12x` |
| `mps`  | Apple Silicon (M1/M2/M3/M4) | **FP32** (Metal hardware limit) | `pip install torch` |

Or as a single extras group:

```bash
pip install -e ".[gpu]"        # NVIDIA cupy (Linux/Windows; silent no-op on macOS)
# Apple Silicon MPS is automatic -- torch is in the main install since v3.
```

Enable per-run via:

* `SpaceChargeConfig(use_gpu="auto" | "cpu" | "gpu" | "cuda" | "mps")`, or
* environment variable `LINAC_GEN_USE_GPU=auto|cpu|gpu|cuda|mps|0|1`, or
* the **PIC backend** dropdown in the GUI's Numerics tab.

Resolution order:

* `"auto"` — best available: cupy → MPS → CPU.
* `"gpu"`  — any GPU; raises if neither cupy nor MPS is available.
* `"cuda"` — force cupy; raises if no CUDA device.
* `"mps"`  — force torch.mps; raises if MPS is unavailable.
* `"cpu"`  — force scipy.fft on the host.

Numerical parity:

* `cuda` vs `cpu`: `~1e-10` relative on σ (FP64 FFTs on both paths).
* `mps`  vs `cpu`: `~6e-7` relative on σ (FP32 FFT round-off floor).
  Well below the PIC shot-noise floor (~`1/√N`), so MP+SC residuals
  vs TraceWin are unchanged.

!!! tip "When the GPU is *not* a win"
    For grids ≤ 96³ on WSL2-class PCIe, host↔device transfer cost can
    dominate.  Apple unified memory makes this transfer effectively
    free, but the inverse real-FFT path on MPS is currently slower
    than scipy.fft in PyTorch 2.10 (known upstream issue), so the
    end-to-end Poisson round-trip is near-neutral on small grids.
    See [Convergence guide](../05_space_charge/05_convergence.md) for
    measured crossover points.

### 3. Build prerequisites by platform

`setup.py` auto-detects the right OpenMP flag for the host compiler.
You only need to install the runtime; everything else is automatic.

| Platform | Compiler | OpenMP runtime needed |
|---|---|---|
| Linux | gcc / clang | shipped with the compiler (`-fopenmp`) |
| macOS (Apple Silicon) | Apple clang | libomp.  Already present if you have torch *or* a conda env with numpy/scipy.  Otherwise `brew install libomp`. |
| macOS (Intel) | Apple clang | libomp.  Same auto-detection; `brew install libomp` as fallback. |
| Windows | MSVC 2019 v16.10+ | bundled with Visual Studio (`/openmp:llvm`) |
| Windows | MSVC older | set `LINAC_GEN_OPENMP_FALLBACK=1` for `/openmp` (OpenMP 2.0, no `collapse(3)` parallelism) |
| Windows | MinGW-w64 / clang-cl | shipped with the compiler (`-fopenmp`) |

The build probes libomp in this order on macOS:

1. PyTorch's bundled copy at `site-packages/torch/lib/libomp.dylib`
   (preferred so the C++ kernel shares one OMP image with torch's MPS
   path — avoids the "two libomp instances" segfault).
2. The active env's `sys.prefix/lib/libomp.dylib` (conda numpy/scipy
   ship this).
3. Homebrew at `/opt/homebrew/opt/libomp` (Apple Silicon) or
   `/usr/local/opt/libomp` (Intel).

If none of the three are present the build prints a warning and emits
a single-threaded C++ kernel — still works, just slower.

### 4. Windows `.exe` (no Python install)

For end users who want to run HELIX without managing a Python
environment, a self-contained Windows distribution is available:

* `Linac_Gen_build/dist/HELIX/HELIX.exe` — ~13 MB launcher.
* `Linac_Gen_build/dist/HELIX/_internal/` — bundled Python, PyQt6,
  numpy, scipy, cupy, CUDA runtime DLLs (~5 GB).

To verify a fresh `.exe`:

```powershell
cd C:\path\to\HELIX
.\HELIX.exe --smoke
```

Expected output:

```
OK  C++ kernel: <module 'linac_gen._pic_kernels' ...>
OK  cupy 14.0.1, CUDA runtime 12090, devices=1
OK  cupy compute: sum=6.0
```

To rebuild the `.exe` from source see
[Contributing → Building the Windows distribution](../appendices/F_contributing.md#building-the-windows-distribution).

## Verifying the install

A 30-second smoke test:

```bash
python -c "
from linac_gen.core.particle import PROTON
from linac_gen.core.reference import ReferenceParticle
ref = ReferenceParticle(species=PROTON, w_kin=3.0, frequency=352.21)
print(f'{PROTON.name} at {ref.w_kin} MeV: β={ref.beta:.4f}, γ={ref.gamma:.4f}')
"
```

Expected: `proton at 3.0 MeV: β=0.0798, γ=1.0032`.

To run the full test suite:

```bash
pytest tests/
```

A clean install on a modern laptop completes the suite in ~3 minutes.

## Launching the GUI

### Quick launch

```bash
./run_gui.sh                 # Linux / WSL / macOS
run_gui.bat                  # Windows
./HELIX.command              # macOS — also double-clickable from Finder
```

Or run the Python module directly:

```bash
python -m linac_gen_gui.interphase.app
# Or with uv (no venv activation needed):
uv run python -m linac_gen_gui.interphase.app
```

### What to expect

A successful launch shows the HELIX splash screen for ~5 seconds (HELIX
logo, version, last-modified date, developer credit), then opens the
main window with eight tabs: Beam, Lattice, Matching, Numerics,
Surrogates, Error Study, Failure Study, Results.

First launch may take a few extra seconds because Python imports torch
and matplotlib lazily.  Subsequent launches in the same shell are
faster (Python compiles `.pyc` files once).

### What the launchers do under the hood

The three launchers are **not** identical — they pick the interpreter
differently:

| Launcher | Interpreter selection | Notes |
|---|---|---|
| `run_gui.sh` | `./.venv/bin/python` if present → else `uv run python` if `uv` is on `PATH` → else `${HELIX_PYTHON:-python3}` | The `uv run` branch execs immediately and does **not** set `QT_PLUGIN_PATH`.  `HELIX_PYTHON` is **ignored** when a `.venv` exists or the `uv` branch is taken. |
| `HELIX.command` | `${HELIX_PYTHON:-python}` only | No `.venv` / `uv` detection — the interpreter (default: `python` on `PATH`) must already have PyQt6 installed. |
| `run_gui.bat` | a hard-coded conda-env interpreter (`...\anaconda3\envs\linac_gui\python.exe`) | Edit the path in the script to match your machine; runs `-m linac_gen_gui.interphase`. |

Beyond interpreter choice, `run_gui.sh` and `HELIX.command` share the
same logic:

1. Resolve the repo root from the script's own location, so they work
   regardless of where the script is invoked from.
2. Locate Qt's plugin directory by importing PyQt6 in the chosen
   interpreter and reading its on-disk path.  Sets `QT_PLUGIN_PATH`
   so Qt finds its platform plugins (`libqcocoa.dylib` on macOS,
   `libqxcb.so` on Linux/WSL).  (Exception: `run_gui.sh`'s `uv run`
   branch skips this step.)
3. Set `PYTHONPATH=<repo>:<repo>/gui` so the `linac_gen_gui` package
   is importable when the package isn't installed into the venv.
4. `cd` into the repo root and run `python -m linac_gen_gui.interphase.app`.

`run_gui.sh` prints one line at startup naming the interpreter it
picked, e.g.:

```
Launching HELIX GUI from /Users/.../HELIX (python: /Users/.../HELIX/.venv/bin/python) ...
```

This is the first thing to check when debugging launch failures — the
interpreter shown there must be the one that has PyQt6 installed.

### Environment variable overrides

The launchers honour these knobs (set them before invoking the script):

| Variable | What it does | Example |
|---|---|---|
| `HELIX_PYTHON` | Interpreter override.  Always honoured by `HELIX.command`; honoured by `run_gui.sh` only as the last fallback (no `.venv`, no `uv`) | `HELIX_PYTHON=/opt/homebrew/bin/python3.12 ./HELIX.command` |
| `QT_QPA_PLATFORM` | Force a Qt platform plugin | `QT_QPA_PLATFORM=cocoa ./run_gui.sh` (macOS), `xcb` (Linux) |
| `LINAC_GEN_USE_GPU` | PIC backend selection | `LINAC_GEN_USE_GPU=cpu ./run_gui.sh` to skip GPU init |
| `QT_LOGGING_RULES` | Verbose Qt diagnostics | `QT_LOGGING_RULES='*=true' ./run_gui.sh` for full trace |

### Headless / offscreen mode

For running over SSH, in CI, or when generating screenshots without a
visible window:

```bash
QT_QPA_PLATFORM=offscreen ./run_gui.sh
```

The GUI constructs normally, runs all Qt event-loop logic, but produces
no on-screen windows.  Useful for the GUI test suite.

### Cross-platform behaviour

| Platform | Launcher | Interpreter detection |
|---|---|---|
| macOS | `./run_gui.sh` | `.venv/bin/python` → `uv run` → `${HELIX_PYTHON:-python3}` |
| macOS | `./HELIX.command` or double-click | `${HELIX_PYTHON:-python}` only (no `.venv` / `uv` detection) |
| Linux / WSL | `./run_gui.sh` | `.venv/bin/python` → `uv run` → `${HELIX_PYTHON:-python3}` |
| Windows | `run_gui.bat` | hard-coded conda-env `python.exe` (edit to match your machine) |
| Windows `.exe` | `HELIX.exe` (no Python install required) | embedded Python |

For a tour of what each tab does once the GUI is open, see
[Part X — GUI](../10_gui/01_overview.md).

## Common installation issues

!!! warning "`pip install -e .` fails on `_pic_kernels`"
    The C++ extension build needs a working compiler:

    * Linux/WSL: `apt install build-essential python3-dev`
    * macOS: `xcode-select --install`
    * Windows native: install Visual Studio Build Tools 2022 (Desktop
      C++ workload).

    If you can't install a compiler, the package still installs and
    HELIX falls back to the pure-Python kernels automatically (the
    fallback triggers on import, with a one-line log warning) — all
    features work, just slower.

!!! warning "`ImportError: DLL load failed` for PyQt6 on Windows"
    Conda-forge PyQt6 needs its own conda interpreter; pip-installed
    PyQt6 inheriting Anaconda's `python312.dll` has an ABI mismatch.
    Either use the bundled `.exe`, or install in a fresh conda
    environment built entirely from conda-forge.

!!! warning "`cupy: CUDA driver not found`"
    Install the matching CUDA runtime alongside cupy.  For CUDA 12.x:

    ```bash
    pip install cupy-cuda12x
    ```

    Then verify with `python -c "import cupy; print(cupy.cuda.runtime.runtimeGetVersion())"`.

!!! warning "`OMP: Error #15: Initializing libomp.dylib, but found libomp.dylib already initialized` (macOS)"
    Two different libomp images got loaded into the same Python process.
    Fixed in HELIX `setup.py` by linking the C++ kernel against torch's
    bundled libomp (when torch is installed) so dyld de-duplicates to
    one image, plus an `import torch` preload in
    `linac_gen.pic.pic_solver` so the load order is deterministic.
    If you still see this after a `pip install -e .` rebuild, check
    `otool -L linac_gen/_pic_kernels.cpython-*.so` — the `LC_LOAD_DYLIB`
    entry for libomp must match the install_name embedded inside
    torch's libomp (`otool -L .../torch/lib/libomp.dylib`).

!!! warning "`use_gpu='mps'` requested but torch unavailable"
    Since HELIX `v3`, `torch` is a main dependency, so this should not
    happen on a clean install.  If you see it, your venv is incomplete
    — re-run `pip install -e .` (or `uv pip install -e .`) and verify
    with `python -c "import torch; print(torch.backends.mps.is_available())"` —
    must print `True`.  Requires macOS 12.3+ on Apple Silicon.

!!! warning "macOS: `./run_gui.sh` reports `Permission denied`"
    Some clones don't preserve the executable bit.  Fix once:

    ```bash
    chmod +x run_gui.sh HELIX.command
    ```

    Subsequent pulls keep the bit set.

!!! warning "macOS: `ModuleNotFoundError: No module named 'PyQt6'` after install"
    `./run_gui.sh` looks for `.venv/bin/python` in the repo directory.
    If `uv pip install` was run without first creating a venv, the
    packages are installed elsewhere and the launcher can't find them.
    Create the venv explicitly:

    ```bash
    uv venv --python 3.11
    uv pip install -e ".[gui,dev]"
    ./run_gui.sh
    ```

!!! warning "macOS: splash screen shows but the main window never appears"
    Force the Cocoa platform plugin:

    ```bash
    QT_QPA_PLATFORM=cocoa ./run_gui.sh
    ```

!!! warning "macOS: `cupy-cuda12x has no wheels with a matching platform tag`"
    You ran `pip install -e ".[gpu]"` on macOS.  CuPy is NVIDIA CUDA
    only and has no macOS wheels.  Since HELIX `v3` the marker on the
    `[gpu]` extra makes it a silent no-op on Darwin; if you still see
    this error you're on an older revision — `git pull` and retry.

For more troubleshooting see
[Appendix D — Troubleshooting](../appendices/D_troubleshooting.md).

Continue to [Quick start](03_quick_start.md) →
