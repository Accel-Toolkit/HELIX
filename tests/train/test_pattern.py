"""PulsePattern: constructors, round-trips, and validation."""
from __future__ import annotations

import numpy as np
import pytest

from linac_gen.train import PulsePattern, TrainConfig, TrainPhysics


def test_rle_array_roundtrip():
    p = PulsePattern.from_rle("1*10 0*54 1*26 0*10")
    assert p.n_slots == 100 and p.n_bunches == 36
    assert p.to_rle() == "1*10 0*54 1*26 0*10"
    q = PulsePattern.from_array(p.filled)
    assert np.array_equal(q.filled, p.filled)


def test_from_duty_pip2_like():
    # PIP-II-like: MEBT chopper keeps ~1/3 of slots
    p = PulsePattern.from_duty(n_slots=900, keep=2, period=6)
    assert p.n_bunches == 300
    assert p.filled[:8].tolist() == [True, True, False, False,
                                     False, False, True, True]


def test_pulse_length():
    p = PulsePattern.uniform(89375)
    assert p.pulse_length_us(162.5) == pytest.approx(550.0)


def test_validation():
    with pytest.raises(ValueError):
        PulsePattern.from_array(np.zeros(10, bool))       # nothing filled
    with pytest.raises(ValueError):
        PulsePattern.from_rle("banana")
    with pytest.raises(ValueError):
        TrainConfig(bunch_frequency_MHz=0.0,
                    pattern=PulsePattern.uniform(3))
    # mode="fast" is legal since M3; "hybrid" since M6 — but only with
    # its minimum physics on (beam_loading; tests/train/test_replay.py)
    TrainConfig(bunch_frequency_MHz=162.5,
                pattern=PulsePattern.uniform(3), mode="fast")
    with pytest.raises(ValueError, match="beam_loading"):
        TrainConfig(bunch_frequency_MHz=162.5,
                    pattern=PulsePattern.uniform(3), mode="hybrid")
    TrainConfig(bunch_frequency_MHz=162.5,
                pattern=PulsePattern.uniform(3), mode="hybrid",
                physics=TrainPhysics(beam_loading=True),
                cavity_params="cav.json")


def test_physics_requires_inputs_loudly():
    p = PulsePattern.uniform(3)
    with pytest.raises(ValueError, match="cavity_params"):
        TrainConfig(bunch_frequency_MHz=162.5, pattern=p,
                    physics=TrainPhysics(beam_loading=True))
