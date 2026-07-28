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
    assert field_names == {"integration_steps_per_metre", "sc_steps_per_metre"}
    cfg = StepConfig()
    with pytest.raises(FrozenInstanceError):
        cfg.integration_steps_per_metre = 200.0
