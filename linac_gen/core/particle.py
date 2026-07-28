# linac_gen/core/particle.py
"""Particle species definitions."""
from dataclasses import dataclass
from linac_gen.core.constants import M_PROTON, M_ELECTRON, AMU

@dataclass(frozen=True)
class Particle:
    """A particle species defined by rest mass and charge state."""
    mass: float    # rest mass (MeV/c^2)
    charge: int    # charge state (units of e, can be negative)
    name: str = ""

    def __repr__(self) -> str:
        return f"Particle(name='{self.name}', mass={self.mass:.3f} MeV/c², charge={self.charge:+d})"

# Built-in species
PROTON = Particle(mass=M_PROTON, charge=1, name="proton")
DEUTERON = Particle(mass=1875.61294, charge=1, name="deuteron")  # CODATA 2018 value
# H⁻ ion = proton + 2 electrons (binding energy 0.75 eV is negligible).
# 939.294 MeV — the value TraceWin uses (verified against its γ−1 output at
# two energies on the PIP-II HB650 reference run).  Using M_PROTON here was
# a bug: it made every H⁻ beam fly 0.05% fast (wrong TOF/RF phasing) and
# shifted Bρ by −0.05% (wrong quad focusing).
H_MINUS = Particle(mass=M_PROTON + 2.0 * M_ELECTRON, charge=-1, name="H-")
