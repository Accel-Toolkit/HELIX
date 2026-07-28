"""Minimal multi-particle tracking example.

Build a short FODO + RF gap lattice, generate a Gaussian proton beam, and
track it with space charge enabled.  Prints RMS sizes and emittances along
the lattice.

Run from the repository root::

    python examples/basic_tracking.py
"""
import numpy as np

from linac_gen.core.beam import Beam
from linac_gen.core.config import SpaceChargeConfig
from linac_gen.core.lattice import Lattice
from linac_gen.core.particle import PROTON
from linac_gen.core.reference import ReferenceParticle
from linac_gen.core.simulation import Simulation
from linac_gen.elements.drift import Drift
from linac_gen.elements.quadrupole import Quadrupole
from linac_gen.elements.rf_gap import RFGap


def make_gaussian_beam(n: int, ref: ReferenceParticle, current_mA: float,
                        sigma_x=1.0, sigma_xp=0.3, sigma_phi=3.0,
                        sigma_dw=0.005, seed=1) -> Beam:
    """Build an uncorrelated Gaussian proton beam."""
    rng = np.random.default_rng(seed)
    beam = Beam(ref=ref, n_particles=n, current=current_mA)
    beam.particles[:, 0] = rng.normal(0.0, sigma_x, n)
    beam.particles[:, 1] = rng.normal(0.0, sigma_xp, n)
    beam.particles[:, 2] = rng.normal(0.0, sigma_x, n)
    beam.particles[:, 3] = rng.normal(0.0, sigma_xp, n)
    beam.particles[:, 4] = rng.normal(0.0, sigma_phi, n)
    beam.particles[:, 5] = rng.normal(0.0, sigma_dw, n)
    return beam


def main() -> None:
    ref = ReferenceParticle(species=PROTON, w_kin=3.0, frequency=352.21)
    beam = make_gaussian_beam(n=2000, ref=ref, current_mA=20.0)

    lat = Lattice()
    lat.add(Drift("D_IN", length=50.0))
    lat.add(Quadrupole("QF1", length=100.0, gradient=+10.0))
    lat.add(Drift("D1", length=200.0))
    lat.add(Quadrupole("QD1", length=100.0, gradient=-10.0))
    lat.add(Drift("D2", length=200.0))
    lat.add(RFGap("RF1", voltage=1.0, phase=-30.0, frequency=352.21, ttf=0.8))
    lat.add(Drift("D3", length=200.0))

    sc = SpaceChargeConfig(nx=32, ny=32, nz=32, grid_extent=4.0)
    sim = Simulation(lat, beam, space_charge=sc)
    results = sim.run()

    print(f"{'s [mm]':>10}  {'sigma_x [mm]':>12}  {'emit_x [mm.mrad]':>18}  "
          f"{'W [MeV]':>10}  {'trans [%]':>10}")
    for s, sx, ex, w, t in zip(
        results.s, results.sigma_x, results.emit_x,
        results.ref_w_kin, results.transmission,
    ):
        print(f"{s:>10.1f}  {sx:>12.4f}  {ex:>18.4e}  {w:>10.4f}  {t:>10.2f}")

    print(f"\nFinal energy: {results.ref_w_kin[-1]:.4f} MeV "
          f"(gained {results.ref_w_kin[-1] - results.ref_w_kin[0]:+.4f} MeV)")
    print(f"Final transmission: {results.transmission[-1]:.2f}%")


if __name__ == "__main__":
    main()
