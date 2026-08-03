"""study.json round-trip, drift tolerance, and shape validation."""
import json

import pytest

from linac_gen.study.spec import (ObservableSpec, ParamSpec, StudySpec,
                                  load_spec, save_spec)


def _spec():
    return StudySpec(
        name="rt", input="deck.dat", mode="mp", strategy="oat",
        parameters=[ParamSpec(selector="QP1.gradient", start=1, stop=2,
                              n=3, baseline=1.5)],
        observables=[ObservableSpec(name="sx_end", quantity="sigma_x")],
        beam={"current": 5.0}, numerics={"nx": 32}, repeats=2)


def test_roundtrip(tmp_path):
    p = tmp_path / "study.json"
    save_spec(_spec(), p)
    back = load_spec(p)
    assert back.name == "rt" and back.mode == "mp"
    assert back.parameters[0].selector == "QP1.gradient"
    assert back.parameters[0].baseline == 1.5
    assert back.observables[0].quantity == "sigma_x"
    assert back.repeats == 2 and back.beam == {"current": 5.0}


def test_unknown_keys_tolerated(tmp_path):
    p = tmp_path / "study.json"
    save_spec(_spec(), p)
    doc = json.loads(p.read_text())
    doc["future_field"] = 123
    doc["parameters"][0]["future_param_field"] = "x"
    p.write_text(json.dumps(doc))
    back = load_spec(p)                    # must not raise
    assert back.parameters[0].n == 3


def test_wrong_kind_refused(tmp_path):
    p = tmp_path / "study.json"
    p.write_text(json.dumps({"__kind__": "something_else"}))
    with pytest.raises(ValueError, match="__kind__"):
        load_spec(p)


@pytest.mark.parametrize("mutate,msg", [
    (lambda s: setattr(s, "mode", "magic"), "mode"),
    (lambda s: setattr(s, "strategy", "sweepy"), "strategy"),
    (lambda s: setattr(s, "parameters", []), "at least one"),
    (lambda s: setattr(s, "repeats", 0), "repeats"),
    (lambda s: setattr(s.parameters[0], "spacing", "quadratic"),
     "spacing"),
])
def test_shape_validation(tmp_path, mutate, msg):
    s = _spec()
    mutate(s)
    with pytest.raises(ValueError, match=msg):
        s.validate_shape()


def test_random_needs_n_samples():
    s = _spec()
    s.strategy = "random"
    with pytest.raises(ValueError, match="n_samples"):
        s.validate_shape()
