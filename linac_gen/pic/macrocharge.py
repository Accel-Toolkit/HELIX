"""THE macroparticle-charge convention — single source of truth.

TraceWin-validated (production PIC vs TraceWin partran benchmarks):

    Q_bunch = I_peak / f_bunch          charge per bunch
    q_macro = Q_bunch / N_launched      per macroparticle

- ``f_bunch`` is the **bunch repetition frequency** — frozen at beam
  creation (``Beam.bunch_frequency``).  It is NOT the local RF clock:
  a FREQ card doubles ``ref.frequency`` mid-machine while the physical
  bunch rate is unchanged (every other bucket is empty).
- Dividing by the **launched** count (not the alive count) keeps each
  macroparticle's charge fixed when particles are lost, so the total
  transported charge decays with transmission — the physical picture.

Every consumer (production PIC, torch/differentiable PIC, ML-corrected
PIC, the gradient-matching objective) computes this number through
:func:`macro_charge_coulombs` so the paths can never drift apart.  The
float operation order below is the historical one — results are
bit-identical to the previous inline expressions.
"""
from __future__ import annotations


def macro_charge_coulombs(current_ma: float, bunch_frequency_mhz: float,
                          n_launched: int) -> float:
    """Charge per macroparticle in Coulombs.

    Parameters use HELIX-native units: peak current in mA, bunch
    repetition frequency in MHz, and the LAUNCHED macroparticle count.
    """
    return ((float(current_ma) * 1e-3)
            / (float(bunch_frequency_mhz) * 1e6)
            / float(n_launched))


def macro_charge_for(beam) -> float:
    """Convenience: the convention applied to a :class:`Beam`."""
    return macro_charge_coulombs(beam.current, beam.bunch_frequency,
                                 beam.n_particles)
