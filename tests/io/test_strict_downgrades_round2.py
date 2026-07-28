"""Strict-mode coverage for the 2026-07-25 review round (claims 5 & 6).

* GAP p_flag is a physics downgrade (absolute phase ignored) — it must
  warn permissively AND raise under ``strict=True``.  It used to bypass
  ``_downgrade`` via a bare warnings-append.
* MAD-X ``CALL`` silently truncated imports — it must warn that the
  included file's contents are missing.
"""
from __future__ import annotations

import os
import tempfile

import pytest

from linac_gen.io.tracewin_parser import parse_tracewin


def _write(text: str, suffix=".dat") -> str:
    fd, path = tempfile.mkstemp(suffix=suffix)
    with os.fdopen(fd, "w") as fh:
        fh.write(text)
    return path


_GAP_DECK = """\
FREQ 162.5
DRIFT 100 20 0
GAP 100 -30 0 1 0 0 0 0 1
DRIFT 100 20 0
END
"""


def test_gap_p_flag_warns_permissive():
    p = _write(_GAP_DECK)
    lat, meta = parse_tracewin(p)
    assert any("p_flag" in w for w in meta["warnings"])
    os.unlink(p)


def test_gap_p_flag_raises_strict():
    p = _write(_GAP_DECK)
    with pytest.raises(ValueError, match="p_flag"):
        parse_tracewin(p, strict=True)
    os.unlink(p)


def test_madx_call_warns_incomplete_import():
    from linac_gen.io.madx_parser import parse_madx
    p = _write(
        "beam, particle=proton, energy=1.0;\n"
        "call, file=\"strengths.madx\";\n"
        "d1: drift, l=0.1;\n"
        "seq: sequence, l=0.1;\n"
        "d1, at=0.05;\n"
        "endsequence;\n",
        suffix=".madx")
    lat, meta = parse_madx(p)
    assert any("CALL" in w and "MISSING" in w for w in meta["warnings"]), \
        meta["warnings"]
    os.unlink(p)


def test_write_final_dst_excludes_dead_particles(tmp_path):
    """2026-07-26: --write-dst carried lost particles frozen at their
    aperture-strike coordinates — corrupting file moments and
    resurrecting the dead on re-import.  Alive only."""
    import numpy as np
    from linac_gen.cli.common import write_final_dst
    from linac_gen.core.config import BeamConfig
    from linac_gen.distributions.factory import create_beam
    from linac_gen.io.tracewin_dst import load_dst

    cfg = BeamConfig(species="H-", energy=2.0, frequency=162.5,
                     current=1.0, n_particles=200,
                     emit_nx=0.2, alpha_x=0.0, beta_x=0.5,
                     emit_ny=0.2, alpha_y=0.0, beta_y=0.5,
                     emit_z=0.06, alpha_z=0.0, beta_z=500.0)
    beam = create_beam(cfg, seed=5)
    beam.lost[:17] = True                       # 17 casualties
    beam.particles[:17, 1] = 25.0               # frozen strike coords

    p = tmp_path / "final.dst"
    write_final_dst(beam, p)
    parts, hdr = load_dst(str(p))
    assert parts.shape[0] == 200 - 17
    assert np.abs(parts[:, 1]).max() < 25.0     # no strike coordinates
