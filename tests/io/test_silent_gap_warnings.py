"""Parsed-then-ignored features must WARN, not stay silent (2026-07-11).

Four gaps disclosed by warnings:
* GAP ``p_flag != 0`` — absolute/sync-phase mode unimplemented on thin gaps;
* ``SET_SYNC_PHASE`` pending when a GAP is parsed — binds to FIELD_MAP only;
* ``ERROR_SET_RATIO`` — stored on ``lattice.error_ratios``, never consumed
  (error-study amplitudes would silently NOT be scaled);
* programmatic ``add_error(..., "dz"/"pitch_deg"/"yaw_deg")`` — tracker
  applies only dx/dy/tilt, so the draw is inert (directive path already
  warned; this covers the API path).

One gap deliberately NOT warned (docs-only disclosure):
* ``ADJUST`` ``start_step`` — routine card furniture in real decks, an
  optimizer hint with no effect on results; warning on it would spam
  every load.

Plus a no-spam guard: a clean deck and the real PIP-II MEBT example must
not pick up any of the new messages.
"""
import warnings
from pathlib import Path

import pytest

from linac_gen.io.tracewin_parser import parse_tracewin

REPO = Path(__file__).resolve().parents[2]


def _parse_text(tmp_path, text):
    p = tmp_path / "deck.dat"
    p.write_text(text)
    return parse_tracewin(str(p))


def test_gap_p_flag_nonzero_warns(tmp_path):
    _, meta = _parse_text(tmp_path, """\
FREQ 162.5
DRIFT 100 15 0
GAP 80000 -85.0 15.0 1
END
""")
    assert any("p_flag=1" in w and "not implemented" in w
               for w in meta["warnings"]), meta["warnings"]


def test_gap_p_flag_zero_is_silent(tmp_path):
    _, meta = _parse_text(tmp_path, """\
FREQ 162.5
DRIFT 100 15 0
GAP 80000 -85.0 15.0 0
GAP 80000 -85.0 15.0
END
""")
    assert not any("p_flag" in w for w in meta["warnings"]), meta["warnings"]


def test_sync_phase_pending_before_gap_warns(tmp_path):
    _, meta = _parse_text(tmp_path, """\
FREQ 162.5
SET_SYNC_PHASE
GAP 80000 -85.0 15.0
END
""")
    assert any("SET_SYNC_PHASE" in w and "GAP" in w
               for w in meta["warnings"]), meta["warnings"]


def test_adjust_start_step_parses_without_warning(tmp_path):
    """DECISION (2026-07-11): ADJUST start_step does NOT warn.  Real decks
    (incl. the project's own fixtures/examples) carry it as routine card
    furniture, and it is only an optimizer hint — HELIX optimizers pick
    their own initial steps and results are unaffected.  Disclosed in the
    keyword cheatsheet + Adjust docstring instead of spamming every load."""
    _, meta = _parse_text(tmp_path, """\
FREQ 162.5
ADJUST QUAD 2 1 -20 20 0.5 0
QUAD 100 10 15
END
""")
    assert not any("start_step" in w for w in meta["warnings"]), meta["warnings"]


def test_error_set_ratio_warns(tmp_path):
    with pytest.warns(UserWarning, match="ERROR_SET_RATIO.*not implemented"):
        _parse_text(tmp_path, """\
FREQ 162.5
ERROR_SET_RATIO 1.0 0.5 0.5
ERROR_QUAD_NCPL_STAT 1 0 0.1 0.1 0 0 0 0 0 0
QUAD 100 10 15
END
""")


def test_programmatic_dz_pitch_yaw_warn():
    from linac_gen.core.config import BeamConfig
    from linac_gen.core.lattice import Lattice
    from linac_gen.elements.drift import Drift
    from linac_gen.errors.error_model import ErrorStudy

    lat = Lattice()
    lat.add(Drift("D1", 100.0))
    cfg = BeamConfig(species="proton", energy=3.0, frequency=162.5,
                     current=0.0, n_particles=100)
    study = ErrorStudy(lat, cfg, n_seeds=2)
    for param in ("dz", "pitch_deg", "yaw_deg"):
        with pytest.warns(UserWarning, match=f"{param}.*NO effect"):
            study.add_error("D*", param, sigma=0.1)
    # dx stays silent — it IS applied by the tracker.
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        study.add_error("D*", "dx", sigma=0.1)


def test_real_mebt_deck_gains_no_new_warnings():
    """The PIP-II MEBT example (FieldMaps + SET_SYNC_PHASE) must not pick
    up any of the new disclosures — guard against warning spam."""
    dat = REPO / "examples" / "mebt_pipii.dat"
    if not dat.exists():
        pytest.skip("examples/mebt_pipii.dat not present")
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)   # ERROR_SET_RATIO path
        _, meta = parse_tracewin(str(dat))
    for needle in ("p_flag", "start_step", "SET_SYNC_PHASE is pending"):
        assert not any(needle in w for w in meta["warnings"]), (
            needle, meta["warnings"])
