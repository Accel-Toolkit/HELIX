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
        self._clock = frequency       # the machine clock phi_s is counted on
        self.phi_s = phi_s
        self.s = s
        # SET_BEAM_PHASE_ERROR's RF shift (deg of the clock): added to the phase every
        # following cavity's beam sees; and its correction at a Dp = 0 card (an error
        # study's: the error-free arrival minus the seed's), added to the clock the
        # cavities that see the arrival read (relative-phase cavities read clock
        # differences, in which it cancels).  The beam's own phase, phi_s, is not touched.
        self.rf_shift = 0.0
        self.rf_rephase = 0.0
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
        self._clock = value
        self._update_derived()

    def switch_clock(self, f_new: float, f_effective: float | None = None) -> float:
        """Change the machine clock to ``f_new`` at this instant, as a FREQ card does.

        ``phi_s`` is ω·t in degrees of the machine clock, so the same instant reads
        ``phi_s · f_new / f_old`` on the new one.  An element whose oscillator runs
        at ``f_effective`` instead (an LLRF frequency offset) beats against that
        clock from its own entrance: the reference then runs at ``f_effective``
        while ``phi_s`` stays counted on the clock.  Returns the ratio of the
        reference's frequencies, by which the particles' Δφ scale; 1 when nothing
        moves.  A clock that was not set is simply set, and a frequency that is not
        positive changes nothing."""
        f_new = float(f_new)
        f_eff = f_new if f_effective is None else float(f_effective)
        old = float(self._frequency or 0.0)
        if f_eff <= 0.0 or (f_eff == old and f_new == self._clock):
            return 1.0
        if old <= 0.0:
            self.frequency = f_eff
            self._clock = f_new if f_new > 0.0 else f_eff
            return 1.0
        clock = float(self._clock or 0.0)
        if f_new > 0.0 and clock > 0.0 and f_new != clock:
            self.phi_s = self.phi_s * (f_new / clock)
            # the RF shifts are times too, counted on the clock
            self.rf_shift = self.rf_shift * (f_new / clock)
            self.rf_rephase = self.rf_rephase * (f_new / clock)
        ratio = f_eff / old
        self._frequency = f_eff
        self._clock = f_new if f_new > 0.0 else f_eff
        self._update_derived()
        return ratio

    def clock_scale(self, f_new: float, f_effective: float | None = None) -> float:
        """The factor :meth:`switch_clock` ``(f_new, f_effective)`` would scale
        ``phi_s`` and the RF shifts by, changing nothing: for an element that
        reads a phase before it makes its own clock jump."""
        f_new = float(f_new)
        f_eff = f_new if f_effective is None else float(f_effective)
        old = float(self._frequency or 0.0)
        clock = float(self._clock or 0.0)
        if f_eff <= 0.0 or old <= 0.0 or (f_eff == old and f_new == self._clock):
            return 1.0
        if f_new > 0.0 and clock > 0.0 and f_new != clock:
            return f_new / clock
        return 1.0

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
        new = ReferenceParticle(
            species=self.species, w_kin=self._w_kin,
            frequency=self._frequency, phi_s=self.phi_s, s=self.s,
        )
        new._clock = self._clock
        new.rf_shift = self.rf_shift
        new.rf_rephase = self.rf_rephase
        return new
