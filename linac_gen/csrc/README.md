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

The same module holds `kd_step`, the particle part of one FieldMap3D
kick-then-drift substep (sampling of every channel, energy kick, damping,
phase slip, Lorentz kick, drift) in one pass over the particles.  It
reproduces the numpy step in `FieldMap3D._track_kd` bit for bit: the same
expressions in the same order, each particle on its own (threaded over
particles, so any thread count gives the same bits), no FP contraction,
and the cos/sin phasors computed by numpy and passed in, so no
transcendental function is evaluated in C++.  The sampling shares one
inline routine with `interp3_multi` (the z cell, common to every particle
of a substep, is located once).  `tests/elements/test_fieldmap_step.py`
pins it with `np.array_equal` (every channel kind, absent components,
lost, out-of-box and NaN particles, 1 against 3 threads).  For macOS
threading add `-Xpreprocessor -fopenmp -I"$(brew --prefix libomp)/include"`
as for `_tsc_kernels` below; `_openmp_enabled` reports the build.  Kill
switch: `LINAC_GEN_FIELDMAP_STEP=0` (the numpy step; the sampler stays).

`fm1d_step` is the particle part of one `FieldMap.track_rk4` slice for maps
whose every channel is 1-D on axis (geometry 1).  It follows that numpy
code, not `kd_step`, which differs in three places: the field is scaled as
`(scale * phasor) * F`, the damping denominator is `ref.beta ** 2 *
ref.gamma * mass` (computed in Python and passed in), and the Lorentz kick
reads x' and y' after the damping.  numpy still computes the phasors,
`r = hypot(x, y)` and the on-axis samples `Fz` and `dFz/dz` (`np.interp`
of the map and of `np.gradient` of it); the kernel adds the channels in
the map's order from +0.0, forms `Fr = (-0.5 * r) * dFz/dz` and
`Fr * (x / safe_r)` with numpy's two comparisons (`r > 1e-12` for
`safe_r`, `r < 1e-12` for the zeroed field), then applies the energy
kick, damping, slip, Lorentz kick and drift.  Particles are independent,
so it is threaded over them, on about sqrt(n / 1500) threads
(`helix::team_for`, `team_particles`: a thread costs a fixed time to start
in every region, so the best team grows as the square root of the work;
the envelope's twelve probes stay on the calling thread), every thread in
the caller's floating-point environment (`helix::FpEnv`).
`tests/elements/test_fieldmap1d_step.py` pins it with raw-bit comparisons
(electric, static and RF magnetic channels, several at once, r = 0 and
r = 1e-12, lost and non-finite particles, a frequency jump, `p_flag = 1`,
the envelope's matrices, 1/2/7/14 threads).  Kill switch:
`LINAC_GEN_FIELDMAP1D_KERNEL=0`.

`curved_rk4_steps` pushes particles through a curved field-map cluster
(`SUPERPOSE_MAP_OUT`, `linac_gen/elements/curved_field_map.py`): RK4 in the
reference's path length, in the planes normal to the reference, with the
exact equations of motion in a static magnetic field.  Its inputs are the
particles (updated in place), the stage frames of the steps (position,
tangent, the two transverse axes and the curvatures, computed in Python),
the children's maps with their placements, and q/p and 1/β per particle.
It reproduces `curved_rk4_numpy` bit for bit: the same expressions in the
same order, the trilinear sampling of `interp3_multi`, only square roots
and divisions (no transcendental function), no FP contraction, each
particle on its own (threaded over particles with a static schedule, so
any thread count gives the same bits).
`tests/elements/test_curved_field_map.py` pins it with `np.array_equal`
against the numpy code and across 1, 2 and 4 threads.  Kill switch:
`LINAC_GEN_CURVED_KERNEL=0`.

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
passes, particles in index order inside a pass), the gather is threaded
over particles only (disjoint writes), and `-ffp-contract=off` forbids FMA
re-rounding.  `deposit_tsc` does the deposit in one loop on one thread;
`deposit_tsc_planes` (what `linac_gen.pic.pic_solver` uses) runs it on the
thread setting, each thread owning whole x-planes: a stable counting sort
buckets the particles by x-cell, and plane X adds, for pass a, the bucket
X+1−a, so every cell gets its terms in the same order and the bits do not
depend on the thread count.  It takes the memory `deposit_tsc` takes (96
bytes per particle), uses about sqrt(N / 700) threads (`team_particles`,
`helix::team_for`: the best teams measured on a 14-core Mac were 2 at 2k
particles, 4 at 5k-20k, 8 at 50k, 14 at 200k), on one thread runs
`deposit_tsc` itself, and hands the caller's floating-point
environment (rounding mode, flush-to-zero) to its threads (`helix::FpEnv`
in `helix_threads.h`; libomp copies it on x86 only).
`LINAC_GEN_TSC_DEPOSIT_PLANES=0`, or a module built before it, keeps
`deposit_tsc`.  `tests/pic/test_tsc_kernels_cpp.py` pins all of it with
raw-bit comparisons at 1–32 threads, including golden values the
one-thread kernel wrote (`tests/pic/fixtures/tsc_deposit_golden.npz`).

The same module holds `field_from_potential`, the Poisson solve's field
E = -grad(phi) (`linac_gen.pic.poisson_solver._field_from_potential`) bit
for bit: `-(phi[i+1] - phi[i-1]) / (2.0 * d)` inside and the one-sided
`/ d` differences on the boundary planes, per axis, on a possibly strided
phi (the padded potential's corner), into fresh C-contiguous arrays,
threaded over x-planes on about sqrt(points / 7000) threads (`team_points`;
every value is computed on its own), in the caller's floating-point
environment.  It returns None -- numpy then computes the same values and
warns or raises as before -- for an axis of fewer than 2 nodes or when a
thread raised a floating-point exception numpy reports (invalid,
divide-by-zero, overflow, underflow).  `LINAC_GEN_LEAN_SOLVE=0` keeps numpy;
`tests/pic/test_lean_solve.py` pins it.

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

### The torch engine's deposit and gather (`deposit_torch`, `gather_products_torch`)

The same module carries the torch space-charge engine's CIC and TSC
deposit and gather (`linac_gen/pic/torch/deposition.py`,
`interpolation.py`) in torch's arithmetic, used by
`linac_gen.pic.torch.cpu_kernels` whenever no gradient is needed: the same
expressions in the same order (TSC `w-` is `0.5*((1-f)*(1-f))` there, not
the numpy twin's `(0.5*(1-f))*(1-f)`), corners in torch's order, no
contraction.  The deposit sums every cell in particle order, as torch's
float64 `index_put_(accumulate=True)` does on the CPU (one thread), on any
number of threads: each thread owns whole x-planes and merges the
particles that reach its plane in index order; it returns the sums and
torch divides by the cell volume.  The gather returns the products
weight × field per particle and corner, shape `(3, N, corners)`, and
torch sums them over the corners in its own order, which depends on the
CPU's vector width.  Both return `None` outside the envelope above (the
caller runs torch's code).  `tests/pic/test_torch_engine_kernels.py` pins
torch's numbers at 1–14 threads.  Kill switch:
`LINAC_GEN_TORCH_SC_KERNELS=0`.

## CIC deposit/gather (`_pic_kernels`)

Same direct compile, but WITHOUT `-ffp-contract=off` -- the flags
`setup.py` uses (pybind11 adds `-fvisibility=hidden`) -- plus the macOS
OpenMP flags:

```bash
cd linac_gen/csrc
c++ -O3 -march=native -shared -std=c++14 -fPIC -fvisibility=hidden \
  -undefined dynamic_lookup \
  -Xpreprocessor -fopenmp -I"$(brew --prefix libomp)/include" \
  $(python3 -m pybind11 --includes) \
  -I"$(python3 -c 'import sysconfig; print(sysconfig.get_paths()["include"])')" \
  pic_kernels.cpp -o ../_pic_kernels$(python3-config --extension-suffix)
```

The CIC kernel has no Python twin to match bit for bit: clang and gcc
contract its `acc += a * b` into an FMA by default, and every CIC result of
a clang or gcc build of HELIX was computed that way (MSVC's `/fp:precise`
does not contract).  Keep the contraction, or the last bits of every CIC
run change.

The deposit adds every cell's contributions in particle order, the order
of a single-threaded loop, whatever the thread count.  On one thread it is
that loop, each particle's eight corners in turn.  On several, the grid is
cut into x-planes; plane `i` takes the particles of x-cell `i` (their
low-x corners) and of x-cell `i-1` (their high-x corners), merged by
particle index after a stable counting sort of the particles by x-cell
(each particle's cell and weights are computed once, in the sorting
pass).  The planes share no cells, so they run in parallel (dealt
round-robin, a static schedule: `schedule(dynamic)` would need
`__kmpc_dispatch_deinit`, which older libomp builds lack).  Both paths end
as the old kernel did on one thread: each cell's sum is added to 0.0 (its
reduce over threads), which turns the -0.0 of an FMA that underflows into
+0.0, then divided by the cell volume.  The arithmetic is the old
per-thread-grid kernel's statement for statement, so the two give the same
bits on one thread; `tests/pic/test_deposit_threads.py` pins 1, 2, 3, 4,
7 and 14 threads against each other and, on macOS arm64, against golden
values the old kernel wrote (`tests/pic/fixtures/cic_deposit_golden.npz`).
The module reports this as `_deposit_order = "particle"`; a binary built
before 2026-10-11 lacks the attribute.

## Thread count (all three modules)

`helix_threads.h` gives every module `set_num_threads(n)` and
`get_num_threads()`, and every parallel region names its team with
`num_threads(helix::num_threads())`.  `linac_gen.parallel.threads` sets
it from the one HELIX thread setting (GUI "CPU threads", CLI
`--threads`, `linac_gen.set_threads`, env `HELIX_THREADS`, else a user's
`OMP_NUM_THREADS`), so the team does not depend on the calling thread's
OpenMP default (a GUI run is a worker thread, and the GUI caps
`OMP_NUM_THREADS` at 1).  The setting has internal linkage, so each module
keeps its own whatever the visibility flags.  0, the value before Python
sets one, keeps the OpenMP default.  `_team_size()` returns the team a
parallel region gets from the calling thread (the tests call it from a
thread whose OpenMP default is 1), and
`tests/parallelization/test_threads.py` checks that every
`#pragma omp parallel` in these sources names its team.  No result depends
on the count -- in the default floating-point mode: a caller that turns on
flush-to-zero (e.g. `torch.set_flush_denormal(True)`) changes subnormal
values on its own thread only (on arm64 libomp does not copy the mode to
its workers).  HELIX never does.  A module built before 2026-10-11 has no `set_num_threads`;
HELIX logs a warning and the module keeps its OpenMP default.
