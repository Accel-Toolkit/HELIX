# CPU threads

HELIX runs the costly parts of a simulation on several CPU threads at once:

* the space-charge field solve (the FFTs of the Poisson solver);
* the field maps (the field sampler and the kick-and-drift step of
  `FieldMap3D`);
* the CIC charge deposition, and the CIC and TSC field gathers (the TSC
  deposition runs on one thread).

By default they use every CPU of the computer, in the GUI, the CLI and
Python alike, and whatever the PIC backend: with `cuda` or `mps` the FFTs
run on the GPU, and the threads serve everything that stays on the CPU.

## The results do not depend on the thread count

Every result is the same, bit for bit, for any number of threads:

* the FFT gives scipy's one-thread result however many threads share it
  (see [How the FFT is threaded](#how-the-fft-is-threaded));
* the gathers and the field-map step compute each particle on its own;
* the CIC deposit adds the contributions to every grid cell in particle
  order, the order of a single-threaded loop; the TSC deposit runs on one
  thread, in a fixed order.

This holds since 2026-10-11.  Before, the CIC deposit added up per-thread
partial sums, so its last digits depended on the thread count.  GUI runs
did it on one thread and CLI runs on ten on a 14-core Mac: torch's default
thread count is the number of performance cores, 10 of the 14, and HELIX's
kernels share torch's OpenMP runtime.  Both now give the one-thread
numbers: GUI results are unchanged, and CLI runs that use the CIC deposit
changed once, in the last digits.

## Setting the number

| Where | How |
|---|---|
| GUI | **Numerics** tab → *Space charge & PIC* → **CPU threads**: *All (N)* or 1 … N.  Saved for this computer, not in the project. |
| CLI | `--threads N` or `--threads all`, on every subcommand, the matcher and the surrogate CLI |
| Python | `linac_gen.set_threads(4)` or `linac_gen.set_threads("all")`; `linac_gen.get_threads()` returns the count in use |
| Environment | `HELIX_THREADS=4` (or `all`) before HELIX starts; failing that, `OMP_NUM_THREADS` |

A change takes effect at once: the compiled kernels at their next call and
the FFT at its next transform.  The count must be between 1 and 1024.

The first match wins:

1. an explicit choice (the GUI control, `--threads`, `set_threads`);
2. `HELIX_THREADS`;
3. `OMP_NUM_THREADS`, when you set it (the GUI sets it to 1 for other
   libraries in its threads and marks that, so HELIX ignores the GUI's
   own value);
4. every CPU.

In the GUI, a value from the environment at launch also takes precedence
over the choice saved on the computer; the tooltip says which variable set
it.  Processes that HELIX starts inherit the setting, except the GUI's own
restart after an update.

## Where it applies

The compiled kernels and the FFT follow the setting wherever they are
called.  The runs of `Simulation` (multi-particle, backtracking, envelope),
`EnvelopeSolver`, the matcher, the multi-objective optimiser, the error
study, the multibunch train and the `backtrack --validate` replay also keep
torch and BLAS on one thread while they last (see the two sections below).
Calling the `Tracker` directly skips that.

## Parallel scans, studies and matches

Worker processes share the setting: each of *W* workers gets
⌊threads / *W*⌋ threads, at least one.  A pool of one worker per CPU runs
one thread per worker; a 3-point scan on 14 CPUs runs 4 threads per point.
This applies to scans, studies, reliability campaigns, parallel CMA-ES
matching, error ensembles and surrogate training.  `LINAC_GEN_FFT_WORKERS`,
when set, fixes the FFT's thread count in every worker regardless of its
share.

## BLAS stays on one thread

numpy's matrix products (BLAS) run on one thread during a run.  HELIX's
matrices are 6×6 or N×6, too small to gain from splitting, and idle BLAS
worker threads keep spinning, taking CPU time from the threads above.  The
GUI fixes BLAS at one thread at start-up.  Elsewhere a run limits it with
threadpoolctl and restores the previous count when the last of any
overlapping runs ends.

## torch stays on one thread

torch's own CPU work — the torch space-charge engine
(`sc_backend="torch"`), ML surrogates, and the gradient, Bayesian and
qNEHVI optimisers — runs on one thread while a run lasts, whatever torch's
own thread count.  With more threads torch's last digits change with the
thread count: it splits long sums between its threads, and on Linux its
FFT is Intel MKL's, which is threaded too (the torch engine gave different
last digits at 1 and 4 threads on a Linux test machine).  A run sets torch
to one thread in its own thread when it starts, or when HELIX loads torch
during the run, and restores torch's previous count when it ends.

Before version 1.16.0, GUI runs ran torch on one thread, and CLI and Python
runs on torch's default count (the 10 performance cores of a 14-core Mac);
1.16.0 gave torch the HELIX thread count everywhere.  From 1.16.1 on it is
one thread everywhere: GUI results are those from before 1.16.0, and
torch-engine results of CLI and Python runs can differ from earlier ones in
the last digits.

## How the FFT is threaded

HELIX threads each 3-D transform itself, on every thread: it cuts each of
the transform's three passes into slabs across its own threads, every slab
a single-threaded scipy call.  scipy's own threading (`workers` > 1) is
used nowhere, for two reasons:

* scipy shuts its thread pool down and restarts it whenever the process
  forks (on macOS, every subprocess it starts).  A transform running in
  another thread at that moment fails with "Work item submitted after
  shutdown", and in rarer timings it hangs; the GUI runs simulations in a
  worker thread and may start a subprocess at any time.
* Its result can depend on the number of workers.  scipy's FFT transforms
  the lines of a call two at a time with vector instructions and an odd
  last line on its own, and some builds round the two ways differently
  (scipy 1.18.1 on macOS arm64 does; 1.17 does not), so a worker's share
  of the lines decides which lines go alone.

HELIX's slabs start on multiples of 8 lines, so every line is batched
exactly as in scipy's one-thread transform: the result is scipy's
one-thread result, bit for bit, for any number of threads and in every
build.  A transform of fewer than 2¹⁷ grid points (for example 32³, the
doubled grid of a 16³ beam grid) runs as one single-threaded call:
splitting it would cost more than it saves.

## In the results file

The HDF5 `provenance/` group records what the run used, read in the run's
own thread:

| Attribute | Contents |
|---|---|
| `threads` | the thread count |
| `threads_source` | where it came from: `set`, `env` (`HELIX_THREADS`), `omp` (`OMP_NUM_THREADS`) or `all` |
| `threads_fft` | the FFT's thread count |
| `threads_kernels` | each compiled kernel module's thread count |
| `threads_torch` | torch's thread count in the run's thread: 1 (absent when torch was not loaded) |
| `threads_blas` | 1 when BLAS was held on one thread |
| `cic_deposit_order` | `particle` (the summation order) |

## Kernels built before 2026-10-11

A compiled kernel module built before thread control cannot take the
setting, and its CIC deposit still adds per-thread partial sums, so its
results depend on the thread count again.  HELIX logs a warning; the module
keeps its OpenMP default, which is one thread in the GUI.  Rebuild the
kernels as `linac_gen/csrc/README.md` describes.

## Cross-references

* [PIC solver](../05_space_charge/02_pic_solver.md) — the CPU and GPU
  backends.
* [Numerics tab](../10_gui/04_convergence_tab.md).
* [CLI: run](07_cli_run.md).
