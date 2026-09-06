"""StepConfig: global integration + space-charge sub-step sizes."""
import pytest
from linac_gen.core.step_config import StepConfig


def test_defaults_are_sensible():
    cfg = StepConfig()
    assert cfg.integration_steps_per_metre > 0
    assert cfg.sc_steps_per_metre > 0
    assert cfg.sc_steps_per_metre <= cfg.integration_steps_per_metre, \
        "SC cadence should not be finer than integration"


def test_integration_steps_for_length():
    cfg = StepConfig(integration_steps_per_metre=100.0)
    # 50 mm drift -> 0.050 m -> 5 sub-steps, clamped to minimum of 2.
    assert cfg.integration_steps_for_length_mm(50.0) == 5
    assert cfg.integration_steps_for_length_mm(5.0) == 2  # minimum
    assert cfg.integration_steps_for_length_mm(0.0) == 2


def test_sc_steps_for_length():
    cfg = StepConfig(sc_steps_per_metre=50.0)
    assert cfg.sc_steps_for_length_mm(100.0) == 5
    assert cfg.sc_steps_for_length_mm(5.0) == 1  # minimum


def test_rejects_non_positive_step_density():
    with pytest.raises(ValueError):
        StepConfig(integration_steps_per_metre=0.0)
    with pytest.raises(ValueError):
        StepConfig(sc_steps_per_metre=-1.0)


def test_min_steps_are_class_constants_not_fields():
    from dataclasses import fields, FrozenInstanceError
    field_names = {f.name for f in fields(StepConfig)}
    assert field_names == {"integration_steps_per_metre", "sc_steps_per_metre",
                           "drift_single_push"}
    cfg = StepConfig()
    with pytest.raises(FrozenInstanceError):
        cfg.integration_steps_per_metre = 200.0


def test_drift_single_push_field():
    """Defaults on; positional construction of the two densities still works
    (the flag is the last field); numpy/JSON truthy values are stored as bool;
    dataclasses.replace keeps it."""
    import dataclasses
    import numpy as np
    assert StepConfig().drift_single_push is True
    cfg = StepConfig(100.0, 50.0)
    assert (cfg.integration_steps_per_metre, cfg.sc_steps_per_metre, cfg.drift_single_push) == (100.0, 50.0, True)
    off = StepConfig(drift_single_push=False)
    assert off.drift_single_push is False
    assert StepConfig(drift_single_push=np.bool_(False)).drift_single_push is False
    assert StepConfig(drift_single_push=0).drift_single_push is False
    assert StepConfig(drift_single_push=1).drift_single_push is True
    rep = dataclasses.replace(off, integration_steps_per_metre=200.0)
    assert rep.drift_single_push is False and rep.integration_steps_per_metre == 200.0
    with pytest.raises(dataclasses.FrozenInstanceError):
        off.drift_single_push = True
