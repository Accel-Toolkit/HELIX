# linac_gen/elements/steerer.py
"""Steerer (corrector) element: thin horizontal and vertical dipole kicks."""
import numpy as np
from linac_gen.elements.base import ThinKickElement

_C_LIGHT = 299_792_458.0  # m/s


class Steerer(ThinKickElement):
    """Thin steering corrector. Applies horizontal and vertical kicks.

    Two flavours (TraceWin ``THIN_STEERING ... Elec``):

    * ``elec=False`` (default) — magnetic: ``bx_l``/``by_l`` are the
      integrated fields ∫Bx·dl / ∫By·dl in T·m; the crossed Lorentz
      kick is Δx' = q·BLy/Bρ, Δy' = q·BLx/Bρ.
    * ``elec=True`` — electric: the same two slots hold the integrated
      fields ∫Ex·dl / ∫Ey·dl in VOLTS, and the kick is same-plane via
      the electric rigidity Eρ = βc·Bρ: Δx' = q·ELx/(βc·Bρ),
      Δy' = q·ELy/(βc·Bρ)  (TraceWin manual, "Thin steering" matrix).

    Sign convention: HELIX applies ``+ sign(q)·field/rigidity`` in both
    flavours (the TraceWin manual writes the magnetic x'-kick with the
    opposite sign; a corrector's polarity is a knob-sign convention and
    is absorbed by orbit correction / matching).
    """

    # Steerer's kick_matrix is constant identity — no params affect it.
    _cache_keys: tuple[str, ...] = ()

    def __init__(self, name: str, bx_l: float = 0.0, by_l: float = 0.0,
                 elec: bool = False):
        super().__init__(name=name)
        self.bx_l = bx_l   # ∫Bx dl (T.m) -> vertical kick | elec: ∫Ex dl (V) -> horizontal
        self.by_l = by_l   # ∫By dl (T.m) -> horizontal kick | elec: ∫Ey dl (V) -> vertical
        self.elec = bool(elec)

    def _kick_mrad(self, ref) -> tuple:
        """(Δx', Δy') in mrad for the reference rigidity, signed by q."""
        charge_sign = 1 if ref.species.charge > 0 else -1
        if self.elec:
            # Electric rigidity Eρ = βc·Bρ (volts); same-plane kick.
            erho = ref.beta * _C_LIGHT * ref.brho
            return (charge_sign * self.bx_l / erho * 1e3,
                    charge_sign * self.by_l / erho * 1e3)
        # Magnetic: crossed kick, 1/Bρ.
        return (charge_sign * self.by_l / ref.brho * 1e3,
                charge_sign * self.bx_l / ref.brho * 1e3)

    def apply_kick(self, beam) -> None:
        dxp, dyp = self._kick_mrad(beam.ref)
        alive = beam.alive_mask
        beam.particles[alive, 1] += dxp  # mrad
        beam.particles[alive, 3] += dyp  # mrad

    def kick_matrix(self, ref) -> np.ndarray:
        # Steerer is a constant kick (not position-dependent), so the linear
        # transfer matrix is identity. Centroid offsets are handled separately.
        return np.eye(6)

    def inverse_kick(self, beam, ref_entry) -> None:
        """Undo the constant kick.  brho (and β for the electric flavour)
        is identical on both sides of a zero-length static element, so
        this is an exact negation."""
        dxp, dyp = self._kick_mrad(beam.ref)
        alive = beam.alive_mask
        beam.particles[alive, 1] -= dxp
        beam.particles[alive, 3] -= dyp
