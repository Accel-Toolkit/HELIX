# Kernels & Green's function

The PIC solver has two orthogonal choices: the **particle-mesh
shape function** (how charge is spread to the grid and field
gathered back) and the **Green's function** (how Poisson is solved).

## Particle-mesh shape: CIC vs TSC

| | CIC | TSC |
|---|---|---|
| Order | 1st (linear B-spline) | 2nd (quadratic B-spline) |
| Cells per particle | 8 (one corner each octant) | 27 (3×3×3 around centre) |
| Cost per kick | 1× | about 1.4× (compiled kernel) |
| Default | yes (legacy) | recommended for noise-sensitive cases |

### When CIC is fine

For lattices with frequent RF cavities (HWR, SSR1, SSR2, LB650, HB650
in PIP-II), CIC is sufficient — the RF restoring force damps grid
noise.  The 3% PIC-kernel calibration offset documented in HELIX's
internal memory notes is **separate** from grid-noise issues —
it's a known calibration bias (see
PIP-II validation).

### When TSC is needed

For long no-RF transports (BTL — beam transport lines), grid noise
+ self-force aliasing in CIC accumulate without RF damping and ε_z
grows unphysically.  Switching to TSC drops the artificial growth
rate substantially:

| BTL stage | CIC ε_z growth | TSC ε_z growth | TraceWin reference |
|---|---|---|---|
| MEBT exit → BTL end | ×14 | ×1.6 | ×1.7 |

TSC is the recommended kernel for BTL-style sections.

!!! note "TSC grid convention"
    The TSC weights anchor the deposited charge's centroid exactly
    half a cell low relative to the particle position (verified
    2026-07-25).  Deposit and gather share the convention, so it
    cancels in every space-charge kick — no beam-dynamics number is
    affected.  Only raw density maps read directly off the ρ grid are
    shifted by −dx/2.  Any future change must move deposit and gather
    together.

### Performance

TSC deposit and gather run in a compiled kernel, `linac_gen._tsc_kernels`.
It returns the same charge density and fields as the Python
implementation, bit for bit, at any thread count.  Measured on the
PIP-II MEBT with 100 000 particles on a 48³ grid (Apple M3 Max):

| | CIC | TSC, compiled | TSC, Python |
|---|---|---|---|
| Space-charge kick | 19.0 ms | 26.7 ms | 117.7 ms |
| Whole MEBT run | 60 s | 70 s | 153 s |

TSC therefore costs about 1.4× CIC per kick, instead of about six
times.  CIC remains sufficient for lattices with frequent RF (see above).

The kernel is built with HELIX's other C++ extensions.  If it is
missing, HELIX logs one line and runs the Python kernels, with identical
results.  `LINAC_GEN_TSC_KERNEL=0` forces the Python kernels, and the
HDF5 provenance attribute `sc_tsc_cpp_kernel` records whether the
compiled kernel was enabled.  *(Changed 2026-10: TSC used to run in
Python only.)*

## Green's function: IGF vs point

| | IGF | point |
|---|---|---|
| Source | Qiang et al. PRSTAB 9, 044204 (2006) | sampled 1/(4πε₀ r) |
| Symmetry | symmetric, no near-source bias | sampled, has near-source artefacts |
| Default | yes | legacy reproduction |

The Integrated Green Function (IGF) is the **closed-form 8-point
cell-difference** of the analytic Coulomb antiderivative — it's the
exact Green's function for a uniform-density unit cube.  The
"point" Green's function samples 1/(4πε₀ r) at cell centres with a
self-potential regulariser at r=0; it has been shown to introduce
grid-resolution bias near the source.

**Always use IGF.**  The "point" option is kept only for legacy
regression reproduction.

### Build cost

The IGF is built by evaluating the antiderivative once at each vertex
of the (2n+1)³ cell-corner lattice and differencing neighbouring
vertices.  This gives the same bits as evaluating it at the eight
corners of every cell, at about one ninth of the cost (48³: 365 ms
before, 41 ms now).  With
`grid_mode="fixed"` it is built once per run.  With
`grid_mode="adaptive"` it is rebuilt at every kick whose cell spacing
changes, so its cost sets the adaptive kick time:

| Adaptive ±6σ kick, PIP-II MEBT, 100 000 particles, 48³ | IGF from cell corners | IGF from shared vertices |
|---|---|---|
| CIC | 466 ms | 82 ms |
| TSC (compiled kernel) | 471 ms | 89 ms |

A grid that only moves, with its spacing unchanged, keeps its Green's
function.  *(Changed 2026-10: the IGF used to be evaluated at every
cell corner.)*

`green_build="symmetric"` builds one octant and mirrors it: the
Green's function is even in each coordinate, so the result equals the
full build to rounding (about 5·10⁻¹³ of its peak at 48³) and is
exactly even.  The build takes 14 ms instead of 76 ms at 48³ and 37 ms
instead of 142 ms at 64³.  The named adaptive
[methods](08_methods.md) use it; everything else keeps the full build.

## Configuration

```python
from linac_gen.core.config import SpaceChargeConfig

# PIP-II-style linac with RF cavities (nx=ny=nz default to 96):
sc_default = SpaceChargeConfig(
    kernel="cic",           # fine with frequent RF
    green_kind="igf",
)

# For BTL or other long no-RF transport:
sc_btl = SpaceChargeConfig(
    kernel="tsc",           # required to avoid ε_z runaway
    green_kind="igf",
)
```

## Cross-references

* [PIC solver](02_pic_solver.md) — the 8-step cycle.
* [Convergence guide](05_convergence.md) — grid sizing.
* `linac_gen/pic/pic_solver.py:1`

← [PIC solver](02_pic_solver.md) ·
[Continue to DC mode →](04_dc_mode.md)
