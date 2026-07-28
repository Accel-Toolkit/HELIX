"""The `python -m linac_gen backtrack` subcommand."""

from tests.dataguard import needs, require  # noqa: E402
from pathlib import Path

import numpy as np
import pytest

from linac_gen.__main__ import main

_REPO = Path(__file__).resolve().parents[2]
_DAT = _REPO / "examples" / "pipii" / "mebt" / "mebt.dat"

# Real PIP-II MEBT entrance Twiss (examples/pipii/mebt/mebt.lgproj) —
# unmatched defaults debunch the beam and push the bunchers far outside
# their linear range, so a matched beam is load-bearing here.
_TW = ["--species", "H-", "--energy", "2.1", "--freq", "162.5",
       "--beam", "emit_nx=0.21", "--beam", "alpha_x=1.228",
       "--beam", "beta_x=0.316", "--beam", "emit_ny=0.21",
       "--beam", "alpha_y=-0.095394", "--beam", "beta_y=0.113",
       "--beam", "emit_z=0.06231832", "--beam", "alpha_z=0.0",
       "--beam", "beta_z=819.05492"]


def test_backtrack_registered():
    with pytest.raises(SystemExit) as exc:
        main(["backtrack", "--help"])
    assert exc.value.code == 0


@needs("examples/pipii/mebt/mebt.dat")
def test_backtrack_design_mode(tmp_path):
    rc = main(["backtrack", str(_DAT), *_TW,
               "--n-particles", "300", "--seed", "5",
               "--out", str(tmp_path), "-q"])
    assert rc == 0
    assert list(tmp_path.glob("*_backtrack.h5"))


@needs("examples/pipii/mebt/mebt.dat")
def test_backtrack_envelope_mode(tmp_path):
    rc = main(["backtrack", str(_DAT), "--mode", "envelope", *_TW,
               "--out", str(tmp_path), "-q"])
    assert rc == 0
    assert list(tmp_path.glob("*_backtrack.h5"))


@needs("examples/pipii/mebt/mebt.dat")
def test_backtrack_dst_roundtrip_with_validate(tmp_path):
    """Forward run → .dst → backtrack → --validate closure must pass."""
    rc = main(["run", str(_DAT), "--mode", "mp", *_TW,
               "--n-particles", "500", "--seed", "11",
               "--write-dst", "--out", str(tmp_path), "-q"])
    assert rc == 0
    dst = tmp_path / "mebt_final.dst"
    assert dst.is_file()
    out_dst = tmp_path / "entrance.dst"
    rc = main(["backtrack", str(_DAT), "--dst", str(dst), *_TW,
               "--write-dst", str(out_dst), "--validate",
               "--out", str(tmp_path), "-q"])
    assert rc == 0                      # validate exits 1 on closure FAIL
    assert out_dst.is_file()

    # The reconstructed entrance must resemble the design entrance beam
    # (same Twiss family): sigma_x within the buncher fidelity floor.
    from linac_gen.io.tracewin_dst import load_dst
    particles, header = load_dst(str(out_dst))
    assert header["w_kin_ref"] == pytest.approx(2.1, rel=1e-3)
    sig_x = float(np.std(particles[:, 0]))
    assert 0.2 < sig_x < 2.0            # sane physical scale, not blown up


@needs("examples/pipii/mebt/mebt.dat")
def test_backtrack_subrange(tmp_path):
    rc = main(["backtrack", str(_DAT), *_TW,
               "--n-particles", "200", "--from-element", "2",
               "--to-element", "30", "--out", str(tmp_path), "-q"])
    assert rc == 0


def test_backtrack_bad_range_exits_2(tmp_path):
    rc = main(["backtrack", str(_DAT), *_TW,
               "--from-element", "50", "--to-element", "10",
               "--out", str(tmp_path), "-q"])
    assert rc == 2


def test_backtrack_missing_dst_exits_2(tmp_path):
    rc = main(["backtrack", str(_DAT), *_TW,
               "--dst", str(tmp_path / "nope.dst"),
               "--out", str(tmp_path), "-q"])
    assert rc == 2
