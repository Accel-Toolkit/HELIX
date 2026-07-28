"""ReferenceParticle: synchronous particle state tracking."""
import math
from linac_gen.core.particle import Particle
from linac_gen.core.constants import C_LIGHT

class ReferenceParticle:
    """Tracks the synchronous particle's absolute state through the lattice."""

    def __init__(self, species: Particle, w_kin: float, frequency: float,
                 phi_s: float = 0.0, s: float = 0.0):
        self.species = species
        self._frequency = frequency
        self.phi_s = phi_s
        self.s = s
        self._w_kin = w_kin
        self._update_derived()

    @property
    def w_kin(self) -> float:
        return self._w_kin

    @w_kin.setter
    def w_kin(self, value: float) -> None:
        self._w_kin = value
        self._update_derived()

    @property
    def frequency(self) -> float:
        """RF frequency (MHz)."""
        return self._frequency

    @frequency.setter
    def frequency(self, value: float) -> None:
        self._frequency = value
        self._update_derived()

    def _update_derived(self) -> None:
        mass = self.species.mass
        # Compute gamma-1 directly to avoid catastrophic cancellation when
        # gamma is very close to 1 (sub-MeV ion injection). The identity
        # beta*gamma = sqrt((gamma-1)*(gamma+1)) then gives beta = bg/gamma
        # without ever subtracting 1 - 1/gamma^2.
        gamma_minus_one = self._w_kin / mass
        self.gamma = 1.0 + gamma_minus_one
        self.bg = math.sqrt(gamma_minus_one * (self.gamma + 1.0))
        self.beta = self.bg / self.gamma
        p_ev = mass * self.bg * 1e6
        # brho uses abs(charge) — always positive magnetic rigidity.
        # Charge sign effects are handled by elements (e.g., quadrupole
        # uses species.charge to determine focusing/defocusing direction).
        self.brho = p_ev / (C_LIGHT * abs(self.species.charge))
        self.wavelength = C_LIGHT / (self._frequency * 1e6) * 1000.0

    def copy(self) -> "ReferenceParticle":
        return ReferenceParticle(
            species=self.species, w_kin=self._w_kin,
            frequency=self._frequency, phi_s=self.phi_s, s=self.s,
        )
