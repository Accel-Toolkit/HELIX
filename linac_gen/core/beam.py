"""Beam class: particle ensemble with reference state."""
import numpy as np
from linac_gen.core.reference import ReferenceParticle
from linac_gen.core.particle import Particle

X, XP, Y, YP, DPHI, DW = 0, 1, 2, 3, 4, 5

LOSS_DTYPE = np.dtype([
    ("particle_id", np.int32),
    ("s", np.float64),
    ("x", np.float64),
    ("y", np.float64),
    ("energy", np.float64),
    ("element_name", "U32"),
])

class Beam:
    def __init__(self, ref: ReferenceParticle, n_particles: int, current: float,
                 duty_cycle: float = 100.0):
        """Particle ensemble with a reference state.

        Parameters
        ----------
        ref : ReferenceParticle
            Synchronous reference.
        n_particles : int
            Number of macroparticles to allocate.
        current : float
            **Peak** bunch current (mA).  This is the quantity the space-charge
            solver uses.  For CW operation (``duty_cycle == 100``) it is also
            the time-averaged current.
        duty_cycle : float, optional
            Pulse duty cycle in percent (default 100, i.e. CW).  Drives
            :attr:`average_current` but does **not** affect tracking.
        """
        self.ref = ref
        self.current = current                # mA, peak
        self.duty_cycle = duty_cycle          # %
        self.particles = np.zeros((n_particles, 6), dtype=np.float64)
        self.lost = np.zeros(n_particles, dtype=bool)
        self._loss_list: list = []
        # DC / continuous-beam flag.  ``False`` means a normal bunched
        # beam — all existing code paths assume this.  ``True`` means
        # pre-RFQ ion-source / LEBT: uniform phase, 4-D tracking, 2-D
        # analytic SC kick.  Flipped automatically by the tracker when
        # the beam encounters its first RF bunching element.
        self.continuous: bool = False
        # Bunch repetition frequency [MHz] — fixed by the upstream RFQ /
        # buncher and constant downstream of it.  Captured at beam
        # creation from ref.frequency so SC kicks at FREQ-jump boundaries
        # (e.g. MEBT 162.5 → SSR1 325 MHz) keep using the bunch's actual
        # repetition rate for Q = I / f_bunch, independent of cavity freq.
        self.bunch_frequency: float = float(ref.frequency)

    @property
    def average_current(self) -> float:
        """Time-averaged current (mA) = peak * duty_cycle / 100."""
        return self.current * self.duty_cycle / 100.0

    @property
    def n_particles(self) -> int:
        return self.particles.shape[0]

    @property
    def n_alive(self) -> int:
        return int(np.count_nonzero(~self.lost))

    @property
    def alive_mask(self) -> np.ndarray:
        return ~self.lost

    @property
    def alive_particles(self) -> np.ndarray:
        return self.particles[self.alive_mask]

    @property
    def species(self) -> Particle:
        return self.ref.species

    @property
    def frequency(self) -> float:
        return self.ref.frequency

    def record_loss(self, particle_id: int, s: float, element_name: str) -> None:
        if self.lost[particle_id]:
            return  # already lost, don't duplicate
        self.lost[particle_id] = True
        p = self.particles[particle_id]
        self._loss_list.append((
            particle_id, s, p[X], p[Y],
            self.ref.w_kin + p[DW], element_name,
        ))

    @property
    def loss_table(self) -> np.ndarray:
        if not self._loss_list:
            return np.array([], dtype=LOSS_DTYPE)
        return np.array(self._loss_list, dtype=LOSS_DTYPE)
