"""Opt-in multibunch / pulse study (plan 2026-08-10).

Nothing in this subpackage is imported by default single-bunch workflows;
with all physics flags off, an N-bunch train is bit-identical to N
independent single-bunch runs.
"""
from linac_gen.train.config import (PulsePattern, TrainConfig, TrainJitter,
                                    TrainPhysics)
from linac_gen.train.driver import TrainRunner, run_train
from linac_gen.train.fast import FastPulseRunner
from linac_gen.train.hom import HOMMode
from linac_gen.train.replay import HybridReplayRunner, auto_select_bunches
from linac_gen.train.results import (TrainResults, load_train_results)

__all__ = ["FastPulseRunner", "HOMMode", "HybridReplayRunner",
           "PulsePattern", "TrainConfig", "TrainJitter", "TrainPhysics",
           "TrainRunner", "TrainResults", "auto_select_bunches",
           "load_train_results", "run_train"]
