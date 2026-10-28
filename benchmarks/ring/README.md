# Ring-mode benchmarks

Every ring-mode result is pinned to an independent reference: analytic
formulas, MAD-X (the language of the ring decks), or another code run
locally — reference code A (element maps with the bend multipoles expanded
in Cartesian coordinates) and reference code B (exact elements with the bend
multipoles in the curved frame).  The reference generators are dev-only
(`dev/refgen/`, left out of public releases); they run with `python -I` and
never import `linac_gen`, so a HELIX bug cannot leak into a reference.  Each
stored reference has a manifest next to it: the SHA-256 of its inputs and of
the data, the seeds and the options.

`cases.yaml` lists the benchmark cases by level (L0 analytic and
bookkeeping, L1 maps and ring optics, L2 single particle over many turns,
L3 space charge, L4 injection, L5 many turns: losses and space charge
through transition, painting).  They use public lattices only: analytic
rings, the simple Booster deck in
`tests/ring/fixtures/` (BSD licence; its licence and notice files are next
to it) and, for a later case, the PSB injection chicane from reference code
A's public test data.

Public references live in `tests/ring/fixtures/refs/` (small `.npz`/`.json`
files); large particle dumps are kept outside git.

The long campaigns are pytest tests under `tests/ring/` with the `bench`
marker; they run only with `HELIX_RUN_BENCH=1`.

## CUDA sessions

On the machine with the GPU (numba with CUDA, the NVIDIA driver), from the
repository root with `PYTHONPATH=.`:

    python scripts/verify_cuda_ring.py --quick     # about 5 minutes
    python scripts/verify_cuda_ring.py --full
    python benchmarks/ring/gpu_roofline.py         # achieved vs peak FP64

`verify_cuda_ring.py` checks the GPU against the CPU (every node on its own
and one turn, particle by particle, the same losses; 200 turns — 1000 with
`--full` — of small-amplitude particles), the same run twice and a restarted
run on the GPU bit for bit, the symplecticity of the one-turn map on both,
a bunch through a generic ramp on both (the reference's record bit for bit,
the particles within 1e-7, the same losses, a restarted ramp on the GPU bit
for bit), the 2.5-D space charge on both (the particles within 1e-6, the
deposit's integers the same for the same particles, run to run and a restart
on the GPU bit for bit, the time per turn of 5 000 particles), and times
58 000 particles.  The thresholds (1e-12 per node, 1e-10
per turn, 1e-7 after the turns, each of a coordinate's largest value) come
from the CPU run with fused multiply-adds, which the GPU compiler makes.
Its JSON report goes to `dev/cuda_sessions/` (in a public tree,
`./cuda_sessions/`), with the GPU's kernel timeout and driver model.
`--tiny` runs it at a few particles — in numba's CUDA simulator when
`NUMBA_ENABLE_CUDASIM=1` is set, where `tests/ring/test_cuda_session.py`
checks the script itself.  `gpu_roofline.py` counts the floating-point work
of a particle-turn by running the kernel source with numbers that count
their operations, and compares the achieved rate with the GPU's FP64 peak.

Both keep each GPU launch to about a quarter of a second (the ring kernels'
`Launcher`): under Windows a GPU that also drives a display stops a kernel
running past its timeout (registry `TdrDelay`, 2 s by default) with
`CUDA_ERROR_LAUNCH_TIMEOUT`.  A GPU in TCC mode (no display) has no
timeout.
