# GPU runs

Ring mode runs on an NVIDIA GPU through numba's CUDA compiler, in double
precision: the same kernels as on the CPU, compiled for the GPU.

!!! warning "Not yet run on an NVIDIA GPU"
    Ring mode's CUDA kernels have so far been checked in numba's CUDA
    simulator only, which runs their code on the CPU thread by thread
    (below); they have not yet run on an NVIDIA GPU.  Run
    `python scripts/verify_cuda_ring.py --quick` on the GPU machine before
    relying on a GPU run.

## What it needs

* An NVIDIA GPU and its CUDA driver, and numba with CUDA support (ring
  mode's `[ring]` extra installs numba; see numba's documentation for the
  CUDA toolkit libraries it needs).  macOS has no CUDA: there ring mode runs
  on the CPU.
* `python scripts/verify_cuda_ring.py --quick` (about five minutes) checks
  the GPU against the CPU (below).

## Choosing the device

* **GUI**: Ring → Run → **Device** (on macOS the CUDA item is greyed out:
  there is no CUDA there).  A run asked for on a machine where numba cannot
  reach a GPU stops at once and says so.
* **Command line**: `python -m linac_gen ring run … --device cuda`.
* **Python**: `RingConfig(device="cuda")`.

## What runs where

| On the GPU | On the host |
|---|---|
| the particles' turns: every element map, the apertures, the RF | the statistics rows, snapshots and checkpoints (the particles copied back for them — with injection, the slots used so far) |
| the space charge's deposit of the charge and the line density, and its kicks | the space charge's field solve (the FFTs) and the line density's filter |
| with a foil, the search for the particles inside it | the foil's crossings and the injection: only the particles inside the foil, and the slots the injection fills, cross between the two |
| | a ramp's or a waveform's tables for the turn (written to the GPU before it) |
| | the tune probes (a few particles, tracked on one CPU thread) |

The particles stay on the GPU between those events.  The launches are
split over particles and turns so that each takes about a quarter of a
second: under Windows a GPU that also drives a display stops a kernel that
runs longer than its timeout (2 s by default).

## The same results?

* On one device, a run gives the same bits every time, and a run restarted
  from a checkpoint ends bit for bit as the one not interrupted.
* Between the GPU and the CPU the results agree to rounding, not bit for
  bit: the GPU compiler fuses multiply-adds and the GPU's sine and cosine
  are not the CPU's.  On the simple Booster the difference is about
  7 × 10⁻¹³ of each coordinate's largest value after one turn and 5 × 10⁻¹⁰
  after 1000 (estimated by fusing the multiply-adds on the CPU).  For the
  same particles, the integer parts are the same — the space charge's
  deposit (and the line density's bins chosen from it), which particles are
  inside the foil, the particle ledger.  As the coordinates part by rounding,
  a particle at the edge of a cell, of an aperture or of the foil can fall
  the other way on one device: in the simulator checks below the particles
  lost and the foil's record came out the same.

## Speed

A GPU pays off with many particles: a turn of a few thousand particles is
mostly launch and copy overhead.  Double precision matters: many cards run
it at 1/32 or 1/64 of their single-precision rate.
`python benchmarks/ring/gpu_roofline.py --particles 50000` times the ring
kernels on every CPU core and on the GPU and reports the rate of
double-precision operations it reaches against the GPU's peak — the
operations counted from a model of the kernels, not measured on the GPU.

## Checking a GPU

`python scripts/verify_cuda_ring.py --quick` (`--full`: longer runs) checks,
on the simple Fermilab Booster:

| Check | What |
|---|---|
| G1 | every node of the ring alone, then a whole turn: CPU and GPU within 10⁻¹² (a node) and 10⁻¹⁰ (a turn) of each coordinate's largest value; with an aperture, the same particles lost at the same node and turn |
| G2 | 200 turns (`--full`: 1000) of small-amplitude particles in the RF bucket: within 10⁻⁷ |
| G3 | on the GPU, the same run twice and a run restarted from a checkpoint give the same bits; the one-turn Jacobian is symplectic on both devices |
| ramp | a bunch through a ramp on each device: the reference's record identical, the particles within 10⁻⁷, the same losses; a restarted ramp bit for bit |
| space charge | the 2.5-D model from the same matched beam: within 10⁻⁶ after the turns, the deposit's integers and the line density's bins the same, repeated and restarted runs bit for bit |
| injection | an H⁻ beam injected through a foil while sextupoles follow a waveform: within 10⁻⁷, the injection's record the same, repeated and restarted runs bit for bit |
| timing | turns of a large bunch on the GPU and on every CPU core |

Without a GPU, HELIX's own tests run the CUDA kernels in numba's CUDA
simulator (`NUMBA_ENABLE_CUDASIM=1`): the same Python code, executed on the
CPU thread by thread — a check of the logic, not of the GPU's arithmetic.
