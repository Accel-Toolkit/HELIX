"""The ring's reference particle.

Unlike the linac :class:`~linac_gen.core.reference.ReferenceParticle` (which
keeps an RF clock phase), the ring reference carries absolute time, the turn
counter and the position within the turn, because RF programs are functions of
time and the reference momentum changes during a ramp.  Momentum is the primary
quantity (``p0c`` in MeV); everything else is derived from it.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

from linac_gen.ring.constants import ConstantsProfile, profile


@dataclass
class RingReference:
    mass: float                 # rest mass, MeV/c²
    charge: int                 # charge state (units of e)
    p0c: float                  # reference momentum × c, MeV
    t: float = 0.0              # absolute time, s
    s: float = 0.0              # position within the turn, m
    turn: int = 0
    constants: ConstantsProfile = field(default_factory=lambda: profile("helix"))

    @classmethod
    def from_kinetic(cls, species: str, w_kin: float, *, constants: str | ConstantsProfile = "helix",
                     **kw) -> RingReference:
        prof = profile(constants) if isinstance(constants, str) else constants
        m = prof.mass_of(species)
        charge = -1 if species.lower() in ("h-", "hminus", "h_minus") else 1
        # p c = sqrt(W (W + 2 m)) — no cancellation at low energy
        p0c = math.sqrt(w_kin * (w_kin + 2.0 * m))
        return cls(mass=m, charge=charge, p0c=p0c, constants=prof, **kw)

    # ---- derived (always from p0c; never cached, so a ramp cannot leave them stale)
    @property
    def beta0gamma0(self) -> float:
        return self.p0c / self.mass

    @property
    def gamma0(self) -> float:
        return math.hypot(1.0, self.beta0gamma0)

    @property
    def beta0(self) -> float:
        bg = self.beta0gamma0
        return bg / math.hypot(1.0, bg)

    @property
    def energy0(self) -> float:
        """Total energy, MeV."""
        return math.hypot(self.p0c, self.mass)

    @property
    def kinetic0(self) -> float:
        """Kinetic energy, MeV, without cancellation: p²/(E + m)."""
        return self.p0c * self.p0c / (self.energy0 + self.mass)

    @property
    def brho(self) -> float:
        """Magnetic rigidity |p/q|, T·m."""
        return self.p0c * 1e6 / (self.constants.c_light * abs(self.charge))

    def copy(self) -> RingReference:
        return RingReference(self.mass, self.charge, self.p0c, self.t, self.s, self.turn, self.constants)
