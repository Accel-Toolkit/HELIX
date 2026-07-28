# Kernels & Green's function

The PIC solver has two orthogonal choices: the **particle-mesh
shape function** (how charge is spread to the grid and field
gathered back) and the **Green's function** (how Poisson is solved).

## Particle-mesh shape: CIC vs TSC

| | CIC | TSC |
|---|---|---|
| Order | 1st (linear B-spline) | 2nd (quadratic B-spline) |
| Cells per particle | 8 (one corner each octant) | 27 (3×3×3 around centre) |
| Cost | ~1× | ~3.4× |
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

TSC is ~3.4× more expensive per step than CIC.  For PIP-II's full
linac with frequent RF, this is unnecessary cost.  For BTL-only or
DC-only runs, TSC is essential.

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
