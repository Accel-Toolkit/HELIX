# C++ kernels (`_pic_kernels`, `_fieldmap_kernels`)

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
