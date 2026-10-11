# C++ kernels (`_pic_kernels`, `_fieldmap_kernels`, `_tsc_kernels`)

Compiled artifacts (`linac_gen/_*.so`) are NOT tracked in git — build them
locally. Python falls back to the pure-Python paths automatically when a
module is missing, so a build is an optimization, never a requirement.

## Standard build (CMake)

```bash
cd linac_gen/csrc && mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -Dpybind11_DIR=$(python3 -m pybind11 --cmakedir)
make
cp _*.so ../../   # place next to the linac_gen package modules
```

## macOS + anaconda workaround (direct compile)

Anaconda's CMake Python detection resolves the `python.app` framework stub
whose advertised include path does not exist
(`anaconda3/python.app/Contents/include/python3.11`), which breaks
`pybind11_add_module` configure. (This is very likely also what broke
`pip install -e .` rebuilds historically.) Bypass CMake entirely:

```bash
cd linac_gen/csrc
c++ -O3 -march=native -ffp-contract=off -shared -std=c++17 -fPIC \
  -undefined dynamic_lookup \
  $(python3 -m pybind11 --includes) \
  -I"$(python3 -c 'import sysconfig; print(sysconfig.get_paths()["include"])')" \
  fieldmap_kernels.cpp -o ../_fieldmap_kernels$(python3-config --extension-suffix)
```

`-ffp-contract=off` is REQUIRED for `_fieldmap_kernels`: its bit-identity
contract with scipy RGI (pinned by tests/elements/test_fieldmap_kernels.py)
forbids FMA re-rounding of `f*w + acc`.

Runtime kill-switch: `LINAC_GEN_FIELDMAP_KERNEL=0` forces the scipy path.

## TSC deposit/gather twin (`_tsc_kernels`)

Same direct compile, with the source and module name swapped:

```bash
cd linac_gen/csrc
c++ -O3 -march=native -ffp-contract=off -shared -std=c++17 -fPIC \
  -undefined dynamic_lookup \
  $(python3 -m pybind11 --includes) \
  -I"$(python3 -c 'import sysconfig; print(sysconfig.get_paths()["include"])')" \
  tsc_kernels.cpp -o ../_tsc_kernels$(python3-config --extension-suffix)
```

To thread the field gather as `setup.py` does on macOS, add
`-Xpreprocessor -fopenmp -I"$(brew --prefix libomp)/include"` and link no
libomp: the OpenMP symbols bind at import to the libomp that torch has
already loaded (`linac_gen.pic.pic_solver` imports torch first).
`linac_gen._tsc_kernels._openmp_enabled` reports whether the build is
threaded.

The kernel is bit-identical to the Python TSC kernels
(`linac_gen/pic/charge_deposition.py`, `field_interpolation.py`): same
expressions, the deposit sums every cell in `np.add.at` order (27 offset
passes, single-threaded), the gather is threaded over particles only
(disjoint writes), and `-ffp-contract=off` forbids FMA re-rounding.
`tests/pic/test_tsc_kernels_cpp.py` pins this with `np.array_equal`,
including a thread-count comparison.

The identity is proven inside an envelope.  `linac_gen.pic.pic_solver`
sends the kernel only float64 `(N, 3)` positions with `N > 0`, float64
bounds, at least 3 nodes per axis and (for the gather) field arrays of
shape `n_grid`.  The kernel returns `None` when any particle maps to a
grid coordinate that is not finite or lies 2**62 or more cells out
(non-finite positions or bounds, zero spacing), where numpy's
float-to-int cast is platform-dependent; the dispatcher then runs the
Python kernels.  Every PicSolver call with finite grid bounds is inside
the envelope.  A direct call with malformed shapes raises `ValueError`.
Runtime kill-switch: `LINAC_GEN_TSC_KERNEL=0` forces the Python kernels.
