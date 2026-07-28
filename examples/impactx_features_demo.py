"""Demo of the three IMPACT-X-inspired features added 2026-05-08.

This example builds a short FODO cell decorated with:

  * a tilted skew quadrupole that x-y couples the beam (so the projected
    ε_x / ε_y oscillate while the eigenemittances stay flat);
  * a misaligned thin sextupole that introduces nonlinearity;
  * a thin octupole;

and tracks a 5 % bi-Gaussian thermal-halo distribution through it.  The
script then reports projected vs eigenemittance evolution to show the
key payoff: **eigenemittances expose real growth that the projected ε
masks under coupling**.

Run from the repository root::

    python examples/impactx_features_demo.py

To explore interactively in HELIX:

  1. Save a project that points to this script-built lattice (or just
     paste the lattice-build snippet into your project) — alternatively,
     set ``distribution = "thermal"`` in the GUI Beam tab, set ``halo_fraction = 0.05``,
     and load any FODO project for a quick visual check.
  2. Run.  In the Results tab, open the "Eigenemittances ε₁·ε₂·ε₃" tile
     in the ENERGY · KINEMATICS section.
"""
import numpy as np

from linac_gen.core.beam import Beam
from linac_gen.core.config import BeamConfig, SpaceChargeConfig
from linac_gen.core.lattice import Lattice
from linac_gen.core.particle import PROTON
from linac_gen.core.reference import ReferenceParticle
from linac_gen.core.simulation import Simulation
from linac_gen.distributions.factory import create_beam
from linac_gen.elements.drift import Drift
from linac_gen.elements.multipole import Multipole, Sextupole, Octupole
from linac_gen.elements.quadrupole import Quadrupole


def build_lattice() -> Lattice:
    """FODO cell with one tilted (skew) quad + sextupole + octupole."""
    lat = Lattice()

    lat.add(Drift("D0", length=100.0))

    # Normal focusing quad
    lat.add(Quadrupole("QF1", length=100.0, gradient=+15.0))
    lat.add(Drift("D1", length=150.0))

    # Skew quad (tilted by 25°) — drives x-y coupling so the projected
    # ε_x / ε_y will visibly oscillate downstream while ε_1 / ε_2 stay flat.
    lat.add(Quadrupole("QSKEW", length=80.0, gradient=-8.0, skew_angle=25.0))
    lat.add(Drift("D2", length=150.0))

    # Defocusing quad
    lat.add(Quadrupole("QD1", length=100.0, gradient=-15.0))
    lat.add(Drift("D3", length=200.0))

    # Misaligned thin sextupole.  k2L = 4 [1/m^2], offset 0.3 mm in x
    # — the offset converts a fraction of the sextupole into a feed-down
    # quadrupole + dipole kick, broadening the halo.
    lat.add(Sextupole("SX1", k2L=4.0, dx=0.3, dy=0.0))
    lat.add(Drift("D4", length=150.0))

    # Thin octupole — pure r^3 nonlinearity, exaggerates halo dynamics.
    lat.add(Octupole("OC1", k3L=80.0))
    lat.add(Drift("D5", length=200.0))

    # Free-form multipole: combined normal sextupole + skew octupole at 10° tilt
    lat.add(Multipole("MIX", knl=[0.0, 0.0, 2.0],
                      ksl=[0.0, 0.0, 0.0, 30.0],
                      tilt_deg=10.0))
    lat.add(Drift("D6", length=200.0))

    return lat


def build_thermal_beam(seed: int = 42) -> tuple[Beam, BeamConfig]:
    """Generate a 20 000-particle bi-Gaussian (thermal) proton beam.

    halo_fraction = 0.05 → 5 % of particles drawn from a wider Gaussian.
    halo_ratio    = 6.0  → halo σ is 6× the core σ.
    """
    cfg = BeamConfig(
        species="proton",
        energy=10.0,           # MeV
        frequency=325.0,       # MHz
        current=20.0,          # mA peak
        n_particles=20_000,
        distribution="thermal",
        halo_fraction=0.05,
        halo_ratio=6.0,
        cutoff=4.0,
        emit_nx=0.25, alpha_x=0.0, beta_x=2.0,
        emit_ny=0.25, alpha_y=0.0, beta_y=2.0,
        emit_z=0.30,  alpha_z=0.0, beta_z=3.0,
    )
    beam = create_beam(cfg, seed=seed)
    return beam, cfg


def report(results, label: str) -> None:
    """Print projected vs eigenemittance evolution + halo metric."""
    s = np.asarray(results.s)
    ex = np.asarray(results.emit_x)
    ey = np.asarray(results.emit_y)
    ez = np.asarray(results.emit_z)
    e1 = np.asarray(results.emit_e1)
    e2 = np.asarray(results.emit_e2)
    e3 = np.asarray(results.emit_e3)

    def cv(a):  # coefficient of variation
        m = abs(a.mean())
        return a.std() / m * 100.0 if m > 0 else 0.0

    print(f"\n=== {label} ===")
    print(f"  s = 0 .. {s[-1]:.0f} mm, {len(s)} steps")
    print(f"  projected emittance variation:")
    print(f"    ε_x : {cv(ex):6.2f}%   range {ex.min():.4f} .. {ex.max():.4f} mm·mrad")
    print(f"    ε_y : {cv(ey):6.2f}%   range {ey.min():.4f} .. {ey.max():.4f} mm·mrad")
    print(f"    ε_z : {cv(ez):6.2f}%   range {ez.min():.4f} .. {ez.max():.4f} deg·MeV")
    print(f"  eigenemittance variation:")
    print(f"    ε_1 : {cv(e1):6.2f}%   range {e1.min():.4f} .. {e1.max():.4f}")
    print(f"    ε_2 : {cv(e2):6.2f}%   range {e2.min():.4f} .. {e2.max():.4f}")
    print(f"    ε_3 : {cv(e3):6.2f}%   range {e3.min():.4f} .. {e3.max():.4f}")
    print(f"  halo signature: ε_x/ε_y ratio swing  =  "
          f"{(ex / ey).max() / (ex / ey).min():.2f}×")


def main() -> None:
    lat = build_lattice()
    beam, cfg = build_thermal_beam(seed=42)

    print(f"Lattice: {len(lat.elements)} elements, length {lat.total_length:.1f} mm")
    print(f"Beam:    {cfg.distribution} ({cfg.n_particles} particles), "
          f"halo_fraction={cfg.halo_fraction}, halo_ratio={cfg.halo_ratio}")

    sc = SpaceChargeConfig(nx=32, ny=32, nz=32, grid_extent=4.0)
    sim = Simulation(lat, beam, space_charge=sc)
    results = sim.run()

    report(results, "Multi-particle tracking with thermal halo + coupling")

    print("\nKey takeaway:")
    print("  • ε_x and ε_y oscillate visibly because the 25°-tilted skew quad")
    print("    couples horizontal and vertical phase space.")
    print("  • ε_1 and ε_2 are *invariants* of the linear coupling and stay")
    print("    nearly flat — what variation you see is the genuine growth")
    print("    from the offset sextupole, octupole, and SC nonlinearity.")
    print("  • Open the 'Eigenemittances ε₁·ε₂·ε₃' tile in the GUI to see")
    print("    the same data graphically.")


if __name__ == "__main__":
    main()
