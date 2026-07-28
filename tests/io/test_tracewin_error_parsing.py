"""TraceWin ERROR_* directive parser tests.

Verifies that each ERROR_* command flavour we support produces the
expected ErrorDef / BeamErrorDef entries on the resulting lattice.
"""
from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pytest

from linac_gen.io.tracewin_parser import parse_tracewin


def _write_dat(text: str) -> str:
    """Write a TraceWin .dat file to a temp path and return the path."""
    fd, path = tempfile.mkstemp(suffix=".dat")
    os.close(fd)
    Path(path).write_text(text)
    return path


# ---------------------------------------------------------------------------
class TestErrorQuadParsing:
    def test_quad_alignment_only(self):
        """ERROR_QUAD_NCPL_STAT N=2 r=2 dx=0.1 dy=0.05 → two quads carry dx + dy errors."""
        dat = """\
TITLE Test
FREQ 325
ERROR_QUAD_NCPL_STAT 2 2 0.1 0.05 0 0 0 0 0 0 0 0 0
QUAD 100 5 20 0 0 0 0 0 0
DRIFT 50 20
QUAD 100 -5 20 0 0 0 0 0 0
END
"""
        lat, _ = parse_tracewin(_write_dat(dat))
        # Two QUAD elements, each should have 2 ErrorDefs (dx, dy).
        names = sorted({e.pattern for e in lat.errors})
        assert len(names) == 2  # 2 distinct quads targeted
        # Every error should have non-zero spread.
        for ed in lat.errors:
            assert ed.parameter in ("dx", "dy")
            assert ed.distribution == "gaussian"
            assert ed.sigma in (0.1, 0.05)

    def test_quad_count_limit_consumes_only_n_quads(self):
        """N=1 should only attach to the first quad, even when 3 follow."""
        dat = """\
FREQ 325
ERROR_QUAD_NCPL_STAT 1 2 0.1 0 0 0 0 0 0 0 0 0 0
QUAD 100 5 20 0 0 0 0 0 0
QUAD 100 -5 20 0 0 0 0 0 0
QUAD 100 5 20 0 0 0 0 0 0
END
"""
        lat, _ = parse_tracewin(_write_dat(dat))
        names = {e.pattern for e in lat.errors}
        assert len(names) == 1  # only the first QUAD got the error

    def test_quad_uniform_distribution(self):
        """r=1 should produce uniform-distribution ErrorDefs."""
        dat = """\
FREQ 325
ERROR_QUAD_NCPL_STAT 1 1 0.2 0 0 0 0 0 0 0 0 0 0
QUAD 100 5 20 0 0 0 0 0 0
END
"""
        lat, _ = parse_tracewin(_write_dat(dat))
        assert all(e.distribution == "uniform" for e in lat.errors)
        assert any(e.parameter == "dx" and e.half_width == 0.2 for e in lat.errors)

    def test_quad_field_errors_dG_and_higher_orders(self):
        """dG, dG3, dG4 in % become gradient_rel, g3_rel, g4_rel fractional."""
        dat = """\
FREQ 325
ERROR_QUAD_NCPL_STAT 1 2 0 0 0 0 0 5 2 1 0 0 0
QUAD 100 5 20 0 0 0 0 0 0
END
"""
        lat, _ = parse_tracewin(_write_dat(dat))
        params = {e.parameter: e for e in lat.errors}
        assert "gradient_rel" in params
        assert params["gradient_rel"].sigma == pytest.approx(0.05)  # 5 % → 0.05
        assert "g3_rel" in params
        assert params["g3_rel"].sigma == pytest.approx(0.02)
        assert "g4_rel" in params
        assert params["g4_rel"].sigma == pytest.approx(0.01)


# ---------------------------------------------------------------------------
class TestErrorCavParsing:
    def test_cav_voltage_phase(self):
        """E and φ slots become voltage_rel + phase_offset on the next gap."""
        dat = """\
FREQ 325
ERROR_CAV_NCPL_STAT 1 2 0 0 0 0 5 2 0
GAP 1.0e6 -30 20 0
END
"""
        lat, _ = parse_tracewin(_write_dat(dat))
        params = {e.parameter: e for e in lat.errors}
        assert "voltage_rel" in params
        assert params["voltage_rel"].sigma == pytest.approx(0.05)
        assert "phase_offset" in params
        assert params["phase_offset"].sigma == pytest.approx(2.0)


# ---------------------------------------------------------------------------
class TestErrorBendParsing:
    def test_bend_alignment_and_field(self):
        dat = """\
FREQ 325
ERROR_BEND_NCPL_STAT 1 2 0.1 0.05 0 0 0 1.5 0
BEND 30 1000 0
END
"""
        lat, _ = parse_tracewin(_write_dat(dat))
        params = {e.parameter for e in lat.errors}
        assert "dx" in params
        assert "dy" in params
        assert "field_rel" in params

    def test_bend_nb_subset_warns(self):
        """The optional 10th operand (Nb, random-subset count) is
        parsed-and-ignored like quad/cav — but it must WARN, not vanish
        silently (2026-07-25 review: bend was the silent one)."""
        dat = """\
FREQ 325
ERROR_BEND_NCPL_STAT 1 2 0.1 0.05 0 0 0 1.5 0 3
BEND 30 1000 0
END
"""
        with pytest.warns(UserWarning, match="ERROR_BEND Nb"):
            lat, _ = parse_tracewin(_write_dat(dat))
        # Errors still applied to ALL N matched elements.
        params = {e.parameter for e in lat.errors}
        assert {"dx", "dy", "field_rel"} <= params


# ---------------------------------------------------------------------------
class TestErrorBeamParsing:
    def test_beam_centroid_and_emit_growth(self):
        dat = """\
FREQ 325
ERROR_BEAM_STAT 2 0.1 0.0 1.0 0.05 0 0 5 0 0 0 0 0 0.5
DRIFT 50 20
END
"""
        lat, _ = parse_tracewin(_write_dat(dat))
        params = {e.parameter for e in lat.beam_errors}
        assert "centroid_x" in params
        assert "centroid_dphi" in params
        assert "centroid_xp" in params
        assert "emit_nx_rel" in params
        assert "current_rel" in params


# ---------------------------------------------------------------------------
class TestErrorGlobalDirectives:
    def test_gaussian_cutoff(self):
        dat = """\
FREQ 325
ERROR_GAUSSIAN_CUT_OFF 2.5
ERROR_QUAD_NCPL_STAT 1 2 0.1 0 0 0 0 0 0 0 0 0 0
QUAD 100 5 20 0 0 0 0 0 0
END
"""
        lat, _ = parse_tracewin(_write_dat(dat))
        assert lat.error_cutoff == pytest.approx(2.5)
        assert all(e.cutoff == pytest.approx(2.5)
                   for e in lat.errors if e.distribution == "gaussian")

    def test_set_ratio(self):
        dat = """\
FREQ 325
ERROR_SET_RATIO 0 0.25 0.5 1.0
END
"""
        lat, _ = parse_tracewin(_write_dat(dat))
        assert lat.error_ratios == [0.0, 0.25, 0.5, 1.0]


# ---------------------------------------------------------------------------
class TestNoErrorDirectivesNoEffect:
    """A .dat with no ERROR_* keywords leaves lattice.errors empty."""

    def test_clean_lattice_has_no_errors(self):
        dat = """\
FREQ 325
QUAD 100 5 20 0 0 0 0 0 0
DRIFT 50 20
END
"""
        lat, _ = parse_tracewin(_write_dat(dat))
        assert lat.errors == []
        assert lat.beam_errors == []
        assert lat.error_ratios == []
        assert lat.error_cutoff == 0.0
