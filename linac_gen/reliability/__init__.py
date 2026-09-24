"""Reliability Study mode: fault tolerance with compensation, availability,
imperfection Monte Carlo with faults on error seeds, and foil scenarios,
run as one resumable campaign on the existing engines
(``linac_gen.failures``, ``linac_gen.errors``, ``linac_gen.parallel``).

Modules arrive by work package: ``pins`` (frozen-phase pins), ``errors``
(error budget → study, audit policy, per-seed draw worker),
``availability`` (block-diagram Monte Carlo and the beam-trip budget).
"""
from linac_gen.reliability.availability import (Block, FaultClass,
                                                AvailabilityResult,
                                                analytic_availability,
                                                read_blocks_csv, simulate,
                                                write_blocks_csv,
                                                write_blocks_template)
from linac_gen.reliability.errors import (ErrorBudget, audit_or_raise,
                                          build_error_study, draw_seed,
                                          steerer_kick_overrides)
from linac_gen.reliability.pins import PinSet, collect_pins, harvest_pins

__all__ = ["PinSet", "collect_pins", "harvest_pins",
           "ErrorBudget", "build_error_study", "audit_or_raise", "draw_seed",
           "steerer_kick_overrides",
           "Block", "FaultClass", "AvailabilityResult", "analytic_availability",
           "read_blocks_csv", "simulate", "write_blocks_csv", "write_blocks_template"]
