"""Envelope (RMS) tracking with analytical space charge.

Build the same FODO cell as ``basic_tracking.py`` but solve the RMS
envelope equation instead of tracking individual particles.  Compares
zero-current and finite-current cases so the space-charge-driven beta
shift is visible.

Run from the repository root::

    python examples/envelope_demo.py
"""
from linac_gen.core.beam import Beam
from linac_gen.core.lattice import Lattice
from linac_gen.core.particle import PROTON
from linac_gen.core.reference import ReferenceParticle
from linac_gen.core.simulation import Simulation
from linac_gen.elements.drift import Drift
from linac_gen.elements.quadrupole import Quadrupole


INITIAL_TWISS = {
    "alpha_x": 0.0, "beta_x": 2.0,
    "alpha_y": 0.0, "beta_y": 2.0,
    "alpha_z": 0.0, "beta_z": 1.0,
    "emit_x": 0.25, "emit_y": 0.25, "emit_z": 0.3,
}


def fodo() -> Lattice:
    lat = Lattice()
    lat.add(Drift("D_IN", length=50.0))
    lat.add(Quadrupole("QF", length=100.0, gradient=+10.0))
    lat.add(Drift("D1", length=200.0))
    lat.add(Quadrupole("QD", length=100.0, gradient=-10.0))
    lat.add(Drift("D2", length=200.0))
    return lat


def run(current_mA: float):
    ref = ReferenceParticle(species=PROTON, w_kin=3.0, frequency=352.21)
    beam = Beam(ref=ref, n_particles=1, current=current_mA)  # RMS mode: n unused
    sim = Simulation(fodo(), beam)
    sim.beam_envelope_params = INITIAL_TWISS
    return sim.run_envelope()


def main() -> None:
    no_sc = run(current_mA=0.0)
    with_sc = run(current_mA=60.0)

    print(f"{'s [mm]':>8}  {'sigma_x (I=0)':>14}  {'sigma_x (I=60mA)':>18}")
    for s, a, b in zip(no_sc.s, no_sc.sigma_x, with_sc.sigma_x):
        print(f"{s:>8.1f}  {a:>14.4f}  {b:>18.4f}")

    print(f"\nSpace-charge beta-shift at end: "
          f"{(with_sc.sigma_x[-1] - no_sc.sigma_x[-1]) / no_sc.sigma_x[-1] * 100:+.2f}%")


if __name__ == "__main__":
    main()
