"""THE macrocharge convention — single source of truth for all four
space-charge consumers (production PIC, torch PIC, ML PIC, gradient
objective).  Pins the TraceWin-validated formula and the semantic the
whole convention rests on: the bunch frequency is FROZEN at beam
creation and does not follow the RF clock across FREQ cards."""
from __future__ import annotations

from linac_gen.core.beam import Beam
from linac_gen.core.particle import PROTON
from linac_gen.core.reference import ReferenceParticle
from linac_gen.pic.macrocharge import (
    macro_charge_coulombs, macro_charge_for,
)


def test_formula_pinned_independent_arithmetic():
    """Q_macro = (I/f_bunch)/N_launched, HELIX units mA/MHz — pinned
    against an independently-ordered evaluation."""
    got = macro_charge_coulombs(20.0, 352.2, 2000)
    q_bunch = (20.0 / 1000.0) / (352.2 * 1.0e6)     # different op order
    assert abs(got - q_bunch / 2000.0) < 1e-30
    assert got > 0.0


def test_beam_convenience_uses_launched_count_not_alive():
    ref = ReferenceParticle(species=PROTON, w_kin=5.0, frequency=162.5)
    beam = Beam(ref=ref, n_particles=100, current=5.0)
    q0 = macro_charge_for(beam)
    # kill half the beam — per-macro charge must NOT change (total
    # transported charge decays with transmission instead)
    beam.alive_mask[: 50] = False
    assert macro_charge_for(beam) == q0


def test_bunch_frequency_frozen_across_freq_jump():
    """A FREQ card advances the RF clock; the bunch repetition rate —
    and therefore the macrocharge — must not follow it."""
    ref = ReferenceParticle(species=PROTON, w_kin=5.0, frequency=162.5)
    beam = Beam(ref=ref, n_particles=1000, current=2.0)
    q_before = macro_charge_for(beam)
    assert beam.bunch_frequency == 162.5
    ref.frequency = 325.0                    # the 162.5 -> 325 jump
    assert beam.bunch_frequency == 162.5     # frozen snapshot
    assert macro_charge_for(beam) == q_before
