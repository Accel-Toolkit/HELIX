# Appendix F — Contributing

How to extend HELIX, fix bugs, build the Windows distribution, and
add to this manual.

## Repository layout

```
linac_gen/
├── core/            # ReferenceParticle, Beam, Lattice, Simulation
├── elements/        # Drift, Quadrupole, Solenoid, Dipole, ...
├── tracking/        # Tracker, EnvelopeSolver, Sacherer ODE
├── pic/             # CIC/TSC deposition, FFT Poisson, C++ kernels
├── distributions/   # Gaussian, KV, Waterbag, Parabolic, Uniform, Thermal
├── matching/        # Levenberg-Marquardt matcher, variables, constraints
├── errors/          # ErrorStudy, ErrorDef, BeamErrorDef
├── diagnostics/     # DiagnosticRecorder, moments, eigenemittance
├── analysis/        # Aperture profile, magnetic stripping, IBS, phase advance
├── io/              # TraceWin parser/writer, FieldMap reader, .dst, .h5
└── csrc/            # C++ pybind11 kernels (compiled to _pic_kernels.so)
gui/linac_gen_gui/   # PyQt6 GUI
tests/               # pytest suite
examples/            # Runnable Python scripts + sample .dat lattices
docs/                # This manual + physics.md + fieldmap_guide.md
```

## Adding a new element

1. Create `linac_gen/elements/my_element.py`.
2. Inherit from one of the four base classes:
   `TransferMapElement`, `ThinKickElement`, `FieldMapElement`,
   `PassiveElement`.
3. Add Misalignment + FieldError mixins as appropriate.
4. Implement `transfer_matrix()` (linear) or `track()` (custom).
5. Add a unit test in `tests/elements/test_my_element.py`.
6. Wire into the parser if it has a TraceWin counterpart
   (`linac_gen/io/tracewin_parser.py`).
7. Add a manual chapter (copy a stub from `03_elements/`).

## Adding a new distribution

1. Create `linac_gen/distributions/my_dist.py`.
2. Implement `generate(config: BeamConfig, seed) -> ndarray (N, 6)`.
3. Register in `linac_gen/distributions/factory.py:_DIST_MAP`.
4. Add to `BeamConfig.distribution` valid-values list.
5. Add a unit test.

## Adding a new diagnostic

1. Add the field to `DiagnosticRecorder.__init__()`.
2. Append to it inside `DiagnosticRecorder.record()`.
3. Add a manual section in `09_diagnostics/`.
4. Add a Results-tab tile if relevant
   (`gui/linac_gen_gui/interphase/tabs/results_tab.py`).

## Test conventions

* `pytest tests/` runs the full suite (~3 minutes).
* `pytest tests/ -m "not slow"` skips long-running integration tests.
* `pytest tests/ -m physics` runs *only* physics benchmarks.
* New tests should reuse fixtures from `tests/conftest.py` where
  possible.
* For new elements / physics: include both a unit test (single
  element / single physics scenario) AND an integration test
  (full lattice, end-to-end behaviour).

## Building the Windows distribution

The `Linac_Gen_build/` tree (including `BUILD.md`, the PyInstaller
spec and the `dist\HELIX\HELIX.exe` output) lives on the dedicated
Windows build machine — it is **not** part of this repository.  Per
`Linac_Gen_build/BUILD.md` on that machine:

```powershell
cd C:\Project\Linac_Gen_build
condaenv\Scripts\pyinstaller.exe Linac_Gen.spec --clean --noconfirm
```

Always pass `--clean` for a full rebuild.  Output goes to
`dist\HELIX\HELIX.exe`.

To smoke-test the result:

```powershell
.\dist\HELIX\HELIX.exe --smoke
```

Expected output:

```
OK  C++ kernel: ...
OK  cupy 14.0.1, CUDA runtime 12090, devices=1
OK  cupy compute: sum=6.0
```

## Adding to this manual

1. The chapter you want to add either exists as a stub or is
   covered by an existing chapter.
2. To add real content, just rewrite the stub.  The build script
   (`make_stubs.py`) only writes if the existing file is < 1500
   bytes — your real content is preserved on re-runs.
3. Add code blocks as ```python; they're auto-tested by
   `verify_snippets.py`.
4. Add figures via `regen_plots.py`:
   ```python
   @figure("fig_03_05_my_diagram")
   def plot_my_diagram():
       """Caption goes here."""
       fig, ax = plt.subplots(...)
       ...
       return fig
   ```
5. Run `mkdocs build` to verify no broken links.

## Coding conventions

* **Line length**: 100 chars (configured in `pyproject.toml`).
* **Style**: ruff + black.
* **Type hints**: use them on public APIs; `mypy` checks.
* **Docstrings**: NumPy-style for public functions; module-level
  docstring with physics references where applicable.
* **No unnecessary comments** — well-named identifiers explain
  what; comments explain why.

## Git workflow

* Branch off `main`.
* Conventional-commit prefixes:
  `feat:` / `fix:` / `docs:` / `test:` / `refactor:`.
* CI runs one workflow (`.github/workflows/docs.yml`): a **strict
  `mkdocs build`** of this manual (plus a non-blocking snippet
  check).  A PR must pass it.
* ruff, mypy and pytest are **local conventions** — run them before
  pushing; they are not (yet) enforced by CI.

## Cross-references

* [Python API](../06_running/01_python_api.md)
* [Physics references](A_physics_references.md)
* [Glossary](C_glossary.md)

← [Appendix E](E_migrating_from_tw.md)
