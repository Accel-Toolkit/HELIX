# Appendix D — Troubleshooting

Common HELIX problems and their fixes.  Roughly grouped by where
the symptom appears.

## Installation

### `pip install -e .` fails on `_pic_kernels`

The C++ extension build needs a working compiler:

* Linux/WSL: `apt install build-essential python3-dev`
* macOS: `xcode-select --install`
* Windows native: install Visual Studio Build Tools 2022 (Desktop
  C++ workload).

If you can't install a compiler, no action is needed: the package
still installs, and HELIX falls back to the pure-Python kernels
automatically when `_pic_kernels` fails to import (a one-line
warning is logged).  All features work, just slower.

### `ImportError: DLL load failed` for PyQt6 on Windows

Conda-forge PyQt6 needs its own conda interpreter; pip-installed
PyQt6 inheriting Anaconda's `python312.dll` has an ABI mismatch.
Either use the bundled `.exe`, or install in a fresh conda
environment built entirely from conda-forge.

### `cupy: CUDA driver not found`

Install the matching CUDA runtime alongside cupy.  For CUDA 12.x:

```bash
pip install cupy-cuda12x
```

Verify with `python -c "import cupy; print(cupy.cuda.runtime.runtimeGetVersion())"`.

### `OMP: Error #15: libomp.dylib already initialized` (macOS)

Two different libomp images got loaded into the same Python process —
classically because the C++ kernel was built against one libomp (e.g.
Homebrew at `/opt/homebrew/...`) while torch ships its own copy at
`site-packages/torch/lib/libomp.dylib`.  `setup.py` is now configured
to share torch's libomp when torch is installed, and
`linac_gen.pic.pic_solver` does `import torch` before loading
`_pic_kernels` to fix the dyld load order.  If you still see the
error after rebuilding:

```bash
otool -L linac_gen/_pic_kernels.cpython-*.so | grep libomp
otool -L "$(python -c 'import torch, os; print(os.path.dirname(torch.__file__))')/lib/libomp.dylib" | head -3
```

The `LC_LOAD_DYLIB` entry in the kernel must match the install_name
embedded inside torch's libomp (typically
`/opt/homebrew/opt/libomp/lib/libomp.dylib`).  If they differ, force
a clean rebuild:

```bash
rm linac_gen/_pic_kernels.cpython-*.so
pip install -e . --no-build-isolation
```

### `_pic_kernels` segfaults on first PIC kick (macOS, conda + torch)

Same root cause as the OMP #15 error above — two libomp images in one
process.  The segfault path triggers when both libomps initialize
threads simultaneously; the error-15 path triggers when one detects
the other first.  Same fix: `pip install -e .` with the current
`setup.py`, which is libomp-aware.

### `_pic_kernels` builds but `OMP_NUM_THREADS` has no effect on Windows

Probably built with classic MSVC `/openmp` (OpenMP 2.0) which silently
ignores `collapse(3)` and only parallelises the outermost loop — most
of the speedup is lost.  Rebuild with Visual Studio 2019 v16.10+
(or newer) which `setup.py` will detect and use `/openmp:llvm`
automatically.  To verify, look for `/openmp:llvm` in the `cl.exe`
invocation during build.

If you must use an older MSVC, opt into the legacy switch with
`LINAC_GEN_OPENMP_FALLBACK=1 pip install -e .` — single-loop
parallelism only, but the build will succeed.

### `use_gpu='mps'` requested but torch unavailable

Install PyTorch with MPS support and verify:

```bash
pip install torch
python -c "import torch; print(torch.backends.mps.is_available())"
```

Must print `True`.  Requires macOS 12.3+ on Apple Silicon (M1/M2/M3/M4).
Intel Macs cannot use MPS — fall back to `use_gpu="cpu"`.

### How do I confirm the GPU is actually being used?

Run a quick sim with the appropriate backend forced and check:

```bash
# Banner in INFO log:
LINAC_GEN_USE_GPU=mps python -c "
import logging; logging.basicConfig(level=logging.INFO)
from linac_gen.pic.gpu_backend import get_backend
print(get_backend('mps').name)
"
# Expected: PicSolver: using MPS backend (...)  +  prints 'mps'
```

For MPS specifically, `torch.mps.driver_allocated_memory()` should be
non-zero after a PIC kick fires.  For CUDA, `nvidia-smi` shows the
process holding GPU memory.

## Beam / lattice setup

### σ_x is huge at LEBT entrance (~10× expected)

Almost always: you forgot that `BeamConfig.emit_nx` is **normalised**.
The factory divides by ref.bg before the matched ellipse — at LEBT
entrance β·γ ≈ 0.0079, so geometric ε is 1/0.0079 ≈ 127× larger
than normalised.  Verify:

```python
from linac_gen.core.config import BeamConfig
from linac_gen.core.particle import H_MINUS
from linac_gen.core.reference import ReferenceParticle

beam_cfg = BeamConfig(species="H-", energy=0.033)   # 33 keV LEBT beam
ref = ReferenceParticle(species=H_MINUS, w_kin=beam_cfg.energy,
                        frequency=162.5)

print(f"emit_nx = {beam_cfg.emit_nx} mm·mrad (normalised)")
print(f"emit_x  = {beam_cfg.emit_nx / ref.bg} mm·mrad (geometric)")
```

### σ_φ blows up in LEBT

Forgot `continuous=True` in BeamConfig.  Pre-RFQ beams have no
longitudinal structure; HELIX's bunched-beam tracking imposes a
non-physical phase coordinate, which then blows up under SC.

### Empty / zero results

Check `Beam.n_alive > 0` — an aperture cut may have killed every
particle.  The recorder records zeros for dead beams.

## Tracking

### Build is slow (>10 minutes for a small lattice)

Likely on multi-particle + 3-D PIC.  Try envelope first:

```python
from linac_gen.core.lattice import Lattice
from linac_gen.core.simulation import Simulation
from linac_gen.distributions.factory import create_beam
from linac_gen.elements.drift import Drift

lat = Lattice(); lat.add(Drift(name="D1", length=500.0))
sim = Simulation(lat, create_beam(BeamConfig(n_particles=500), seed=1))
results = sim.run_envelope()    # ~100× faster
```

Or reduce `n_particles` to 2000 for a quick first look.

### σ disagrees with TraceWin by ~3 %

Expected — see [TraceWin parity](../12_validation/01_tracewin_parity.md).
HELIX has a documented PIC-kernel calibration bias of ~3 %.

### σ_φ disagrees with TraceWin by 4×

This is the **convention difference**: HELIX reports σ_φ at the
local cavity frequency, TraceWin at the bunch frequency
(162.5 MHz).  Multiply HELIX's σ_φ by `162.5 / local_f_MHz` to
match TW.

### ε_z grows ×14 in BTL section

Switch CIC → TSC kernel.  In long RF-free transport the CIC
deposition's grid noise accumulates as artificial longitudinal
emittance growth; the smoother TSC kernel suppresses it.  See
[SC kernels](../05_space_charge/03_kernels.md).

### MP run hangs / appears stuck

* Check `progress_callback` output — long lattices take minutes.
* For PIP-II 256m + MP+SC + 5000 particles, expect ~9 minutes.
* If truly stuck, run via `python -u` (unbuffered) so prints
  appear; check `htop` to confirm CPU is busy.

## GUI

### Window is blank / black

PyQt6 + WSL2 can have OpenGL issues.  Try:

* Set `QT_QPA_PLATFORM=xcb` env variable before launching.
* Use the bundled Windows `.exe` instead.

### Tab is empty / "no lattice loaded"

Open a `.dat` first (Lattice tab → Open).

### Run button does nothing

Verify a lattice and beam are configured (status bar shows green
checkmarks for both).  If button is gray, focus is on a different
tab.

## Build / contributing

### Manual snippet test failures

`docs/manual/_build/verify_snippets.py` runs every `python` block
in the manual.  Failures usually mean:

* Missing import (add to top of code block).
* Outdated API (chapter is out of date with current source).
* External file missing (e.g. `examples/...` was deleted).

Run with a path filter to debug one chapter at a time:

```bash
python docs/manual/_build/verify_snippets.py 03_elements
```

## Cross-references

* [Installation](../01_getting_started/02_installation.md)
* [Convergence checklist](../12_validation/04_convergence_guide.md)
* [Known limitations](../12_validation/03_known_limitations.md)

← [Appendix C](C_glossary.md) ·
[Continue to Appendix E →](E_migrating_from_tw.md)
