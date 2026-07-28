# IMPACT-X feature demo

Three-feature showcase for the IMPACT-X-inspired additions landed
2026-05-08:

| Feature | Used in this folder | Tested by |
|---|---|---|
| **Eigenemittance** ε₁/ε₂/ε₃ (Balandin invariants) | Always recorded; visible in any project | Open the *Eigenemittances ε₁·ε₂·ε₃* tile in the Results tab |
| **Bi-Gaussian thermal halo** distribution | `.lgproj` sets `distribution=thermal`, `halo_fraction=0.05`, `halo_ratio=6.0` | Beam tab → halo controls light up |
| **Multipole offsets + tilt** (`Multipole(dx, dy, tilt_deg)`) | *Python-API only* — TraceWin `.dat` doesn't carry these | See `examples/impactx_features_demo.py` |

## Layout

| File | Purpose |
|---|---|
| `coupled_fodo.dat` | TraceWin-format lattice: 3 FODO periods + two 25° skew quads (drives x-y coupling) |
| `coupled_fodo.lgproj` | HELIX project: 50 000-particle 5 % halo @ 6×σ, 20 mA proton beam at 10 MeV / 325 MHz |
| `README.md` | this file |

## To run from the GUI

1. `File → Open Project…` → pick `coupled_fodo.lgproj`
2. Beam tab: confirm `Distribution = thermal`, `Halo fraction = 0.05`,
   `Halo σ ratio = 6.0` populated automatically
3. Hit **Run** (multiparticle or envelope, both populate eigenemittances)
4. Results tab → ENERGY · KINEMATICS section → click
   **Eigenemittances ε₁·ε₂·ε₃**

## What you should see

* **ε_x and ε_y oscillate** — the 25° skew quads couple horizontal and
  vertical phase space, so the projected emittances wobble even though
  no real growth has occurred.
* **ε₁ and ε₂ stay much flatter** — they are constants of motion under
  any linear symplectic transport, including the skew-quad coupling.
  Whatever residual variation you see *is* genuine non-symplectic drift
  (SC, numerical errors).
* **Halo signature** — open the beam-density popup at any cross-section.
  The 5 % halo at 6×σ shows up as a faint "ring" around the dense core,
  characteristic of a bi-Gaussian distribution.

## To run from Python

```python
from linac_gen.io.tracewin_parser import parse_tracewin
from linac_gen.core.config import BeamConfig
from linac_gen.distributions.factory import create_beam
from linac_gen.core.simulation import Simulation

cfg = BeamConfig(
    species="proton", energy=10.0, frequency=325.0, current=20.0,
    n_particles=50_000, distribution="thermal",
    halo_fraction=0.05, halo_ratio=6.0,
    emit_nx=0.25, beta_x=2.0,
    emit_ny=0.25, beta_y=2.0,
    emit_z=0.30,  beta_z=3.0,
)
beam = create_beam(cfg, seed=42)
lattice, _ = parse_tracewin("examples/impactx_features/coupled_fodo.dat")
results = Simulation(lattice, beam).run()

print("ε₁ range:", min(results.emit_e1), "..", max(results.emit_e1))
print("ε_x range:", min(results.emit_x), "..", max(results.emit_x))
```

## See also

* `examples/impactx_features_demo.py` — fully Python-built version that
  also exercises the standalone Multipole element with `dx`, `dy`, and
  `tilt_deg` parameters.  TraceWin `.dat` files cannot express
  multipole offsets/tilts, so that feature is API-only.
