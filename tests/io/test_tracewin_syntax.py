"""Test the positional-argument schema for each TraceWin card."""
import pytest
from linac_gen.io.tracewin_syntax import SCHEMA, parse_positionals


def test_drift_schema():
    schema = SCHEMA["DRIFT"]
    out = parse_positionals(schema, ["50", "30", "25", "0.5", "-0.3"])
    assert out == dict(length=50.0, aperture=30.0, aperture_y=25.0,
                       x_shift=0.5, y_shift=-0.3)


def test_drift_defaults_for_missing_optionals():
    schema = SCHEMA["DRIFT"]
    out = parse_positionals(schema, ["50", "30"])
    assert out["aperture_y"] is None  # circular by default
    assert out["x_shift"] == 0.0


def test_quad_with_skew_only():
    schema = SCHEMA["QUAD"]
    out = parse_positionals(schema, ["50", "5", "20", "10"])
    assert out == dict(length=50.0, gradient=5.0, aperture=20.0,
                       skew_angle=10.0,
                       g3=0.0, g4=0.0, g5=0.0, g6=0.0, gfr=0.0)


def test_required_missing_raises():
    schema = SCHEMA["QUAD"]
    with pytest.raises(ValueError, match="requires at least"):
        parse_positionals(schema, ["50"])  # missing gradient
